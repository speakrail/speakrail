"""MicroSession: the live harness (the turn-taking LLM decides every turn-taking step, one token per input event).

Same transport and plumbing as cascade.Session (ASR websocket + turn head + fork peek, Breeze TTS, tape, events.jsonl,
the browser protocol); only the brain differs. The context is built with protocol.py, the ONE
definition shared with the tape builder, so the model sees exactly what it was trained on:

  inputs    live Voxtral words; <complete> by the turn head's endpoint rule (a new word arms it; 3 frames at >= 0.9 of
            P(complete)+P(backchannel) while we are silent, of P(complete) while we speak; never on a backchannel over our
            speech), preceded by the fork-peek's pending words; <user_bc> at a head P(backchannel) >= 0.95 onset;
            <sil:Ns> ladder (1, 2, 4 ... 256 s) after the last speech (the user's or our own playback) while nobody speaks.
  calls     serial: inputs that arrive while a call is in flight wait for it, then go in as one batch = one call.
            idle {speak, interrupt, listen, listen_muted}; overlap {continue, yield, listen}. vLLM completions API with
            allowed_token_ids; an optional per-decision logit bias (none by default).
  reply     speak / interrupt: streamed from the context (control tokens banned), clauses to TTS as they arrive;
            a web_search call is answered by SearXNG, the local tools (tools.py: notes, lists, todos, counter,
            stopwatch, Open-Meteo weather, unit_convert, calculator, dice, time) in process; the reply continues after.
  overlap   the user is heard while we speak: the reply is pinned at the word playing at the onset, the rest is not saved,
            an overlap turn opens; continue -> the reply is resumed (rest saved at its end), yield -> audio stops.
            Never before the reply has a word (input waits for the first one: no empty model turn).
  interrupt what the user said before they could hear us (our audio start + react_ms) is the tail of the turn we cut
            into and is dropped, its <complete> / <user_bc> too (the tape builder never shows it); what they say after
            that waits until our first sentence has played (hold_max_s at most) and only then opens the overlap.
  rules     barge_in "duck": duck the audio while the head hears the user over a reply (not an interrupt), the model
            still decides; + the safety net (MicroCfg.safety_yield_s). "word": the first non-backchannel word yields at once
            (no call); "head": also duck the audio while the
            head hears speech; "off": the model decides. Early start: a non-backchannel word while <= undo_words of our
            reply have played undoes the reply (context rolled back to before the speak call; the words join the turn).
  spec      on the first silent frame with P(fire) >= spec_tau (0.5) while we are silent: peek, fork the context with the
            peek's words + <complete>, decide on the fork and, on speak, generate the reply with its audio and text held
            (spec_parallel, 10-04 late: the reply is requested together with the decision and killed unless it is speak).
            The endpoint rule's commit adopts the fork if nothing arrived in between (held audio goes out at once);
            speech, a word the peek didn't have, or any other input discards it and the normal path runs.

Per session: sessions/<id>/{events.jsonl, turns.json, summary.json, words.jsonl, tape.wav} as cascade, plus
timeline.txt (inputs, calls, replies), calls.jsonl and context.txt (the rendered context).
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import datetime
import time
from dataclasses import dataclass, field

from ..config import CFG                                    # noqa: E402
from . import protocol as P                                        # noqa: E402
from .tokens_v1 import SIL_LADDER_S                          # noqa: E402

from .backchannel import is_backchannel, normalize           # noqa: E402
from .cascade import Session, Turn                           # noqa: E402
from .tts import Segment, strip_tags           # noqa: E402
from .tools import CAT, DEFAULT_CITY, LOCAL_TOOLS, RESET_TOOL, RESET_TOOL_SPEC, STATEFUL_TOOLS, ToolBox   # noqa: E402
from .llm_engine import Engine                                # noqa: E402
from .claude_tool import CLAUDE_TOOL, CLAUDE_TOOL_SPEC, ClaudeTasks   # noqa: E402
from .frontier_tool import FRONTIER_TOOL, FrontierTasks   # noqa: E402

TTS_RATE = 24000
TOOL_RESPONSE_OPEN = 50
CHANNEL = P.encode("<|channel>")[0]
TURN_OPEN_ID = P.encode("<|turn>")[0]
BANNED_IN_REPLY = {str(i): -100 for i in [CHANNEL, TURN_OPEN_ID] + list(range(6, 28))}   # thought channel, turns, controls
# one match per call: non-greedy up to the first "}<tool_call|>" (a greedy .* swallowed a second call in the same message
# into the first call's arguments, which then parsed as {}), dotted names allowed
CALL_RE = re.compile(r"<\|tool_call>call:([\w.\-]+)(\{.*?\})<tool_call\|>", re.S)
INTERJECT = P.INTERJECT                                     # live tasks: a few words without taking the turn (ids 26 / 27)
INTERJECT_END = P.INTERJECT_END
INTERJECT_MAX_TOKENS = 64        # per round: interjections are 1-4 words (+ an act task's calls)


def _think_sys():
    """the listening-notes system prompt, verbatim from training, so live notes match the training notes"""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "notes_prompt.txt")) as f:
        return f.read()
THINK_SYS = _think_sys()
# the catalog's ask_frontier description + one sentence: without it the model sent questions about the user's own files and
# projects to the bigger model, which can't see them. It names no other tool, so it stays true whatever else is declared.
FRONTIER_SPEC_LIVE = {**CAT[FRONTIER_TOOL], "description": CAT[FRONTIER_TOOL]["description"].replace(
    " Async:", " It can't see the user's computer or files, so it can't check their own projects, runs or results. Async:", 1)}
assert FRONTIER_SPEC_LIVE["description"] != CAT[FRONTIER_TOOL]["description"]
NOTES_MAX_CHARS = 3000           # as the tape builder (CFG notes_max_chars): a long monologue keeps its most recent notes
SEARCH_TOOL_SPEC = {"domain": "search", "mode": "sync", "args": {"query": "string"},
                    "description": "Search the web; returns the top results as title and snippet pairs. Results can be stale "
                                   "or wrong: report them as what the source says."}
_BEHAVIOUR = ("You may cut in only when something is time critical; otherwise wait for the user to finish, and correct a "
              "wrong fact briefly at your next turn. Speak when addressed or when a result arrives. After a long silence you "
              "may check in once. If the user asks for quiet, say nothing until they release you. Facts: 1. You can search "
              "the web. 2. The user lives in {city}.")
PERSONAS = {                                    # the turn-taking sentences and the facts are the same in every persona
    "nova": ("You are Nova, a voice assistant in a live spoken conversation with one user. Your manner is warm, gentle and "
             "calm, kind without gushing. Keep answers short and plain, in spoken words; be longer only when the user asks "
             "for it. " + _BEHAVIOUR),
}
# the date is not in the prose: the prompt ends with protocol.date_line (date and time of day), as in training
DEFAULT_TZ = CFG.home_timezone
# reset_chat confirmation: the user's turn must say yes (and nothing like no / wait / cancel) after we asked
RESET_YES_RE = re.compile(r"\b(yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|confirm|confirmed|absolutely|please do|of course|correct)\b", re.I)
RESET_NO_RE = re.compile(r"\b(no|nope|don't|do not|not|cancel|stop|wait|never mind)\b", re.I)
RESET_ASKED_RE = re.compile(r"\b(reset|start over|start fresh|fresh start|wipe|clear|new chat|from scratch)\b", re.I)
RESET_ASK_S = 120.0                  # a request to confirm is valid this long


@dataclass
class MicroCfg:
    url: str = "http://127.0.0.1:8001/v1/completions"
    lora: str = "speakrail"          # the adapter's name in vLLM (--lora-modules speakrail=...)
    bias: dict = field(default_factory=dict)   # optional per-decision logit bias (token id -> bias); none by default
    reply_temp: float = 0.0          # replies and interjections; 0 = greedy
    reply_top_p: float = 1.0
    reply_top_k: int = 0
    undo_words: int = 2              # early start: a non-backchannel word while <= this many reply words played -> undo
    result_nudge_s: float = 5.0      # an async result is still unreported and the user has been quiet this long:
                                     # the harness takes the turn so the model reports it (0 = off)
    tool_hold_ms: int = 0            # run a reply's tool calls only once the user has been quiet this long (0 = off);
                                     # if they go on meanwhile, the early-start undo / overlap yield cancels the reply and the call
                                     # never runs (Full-Duplex-Bench v3: fewer premature tool calls, higher pass@1)
    spec_parallel: bool = True       # the speculative reply starts WITH the fork's decision, not after it (the decision,
                                     # ~60 ms, would otherwise sit on the critical path); not speak -> killed
    duck_gain: float = 0.15          # barge_in "head" / "duck": output gain while the head hears the user over us (never an
                                     # interrupt or an interjection: those keep full volume)
    unduck_ms: float = 400           # ... back to full volume after this much silence from the user
    safety_yield_s: float = 1.5      # overlap: the model keeps <listen>ing while the user has said this much non-backchannel
                                     # speech over us -> the harness yields (0 = off; never on an interrupt)
    reply_max_tokens: int = 450      # about 2 min of speech (shorter caps cut stories mid-sentence)
    tool_rounds: int = 4
    first_clause_chars: int = 3
    next_clause_chars: int = 40
    tau: float = 0.9                 # endpoint rule
    k: int = 3
    early: str = "on"                # (10-06) a held speculative reply goes out before the rule commits when the fork
                                     # decided speak with p >= early_tau and the head still hears no speech with P(fire) >=
                                     # spec_tau: on short greetings the head hovers at 0.7-0.9 and "Hey, you there?" waited
                                     # 500 ms+ with its reply ready. "shadow" only logs when it would have gone; "off"
    early_tau: float = 0.99
    bc_close_s: float = 1.5
    user_bc_tau: float = 0.95
    search: bool = True              # declare web_search (SearXNG answers it)
    tools: bool = True               # declare the local tools (tools.LOCAL_TOOLS)
    client_tools: list = field(default_factory=list)   # {name, description, parameters}, run by the API client. A name here replaces ours
    client_tool_runner: object = None   # async (name, args) -> result
    instructions: str = ""           # the API client's own, after the turn-taking sentences, which stay in charge
    persona: str = "nova"
    claude: bool = False             # declare claude_code (+ task_status / cancel_task): tasks for Claude Code on this
                                     # machine, reported as async results (server.py turns it on only with the access key)
    claude_model: str | None = None  # claude -p --model (None: the CLI default)
    claude_mode: str = "auto"        # claude -p --permission-mode
    claude_cwd: str = os.path.expanduser("~")
    frontier: bool = False           # declare ask_frontier (catalog text), backed by GLM-5.3 Flash on Fireworks (paid)
    reset_tool: bool = True          # declare reset_chat (wipe the context, optional standing instructions); harness-confirmed
    allow_interject: bool = True     # offer <interject> at idle calls (live tasks: translate / count / act while the user talks)
    silence_fallback_ms: int = 0     # >0: this much silence after user words with no head fire -> <complete> (model decides)
    notes: bool = False              # listening notes. Stock Gemma (adapter off, same vLLM) writes a note at each
                                     # protocol.note_points point while the user speaks (the training prompt and message
                                     # shape); at a speak decision the finished notes go in the thought channel (an unfinished one
                                     # is dropped: training kept only notes done before the user stopped)
    think_model: str = "speakrail-base"   # the base model without the adapter (vLLM --served-model-name)
    phrase_cache: bool = True        # a reply's first clause from phrase_cache.py's pre-rendered clips (Breeze, same voice)
    react_ms: float = 300            # interrupt: user words starting before our audio start + this are the tail of their turn
    interrupt_hold: bool = True      # interrupt: overlap waits until the first sentence has played ...
    hold_max_s: float = 4.0          # ... or this much of it


class Reply:
    """One assistant reply: generation state, TTS segments, playback, and what of it goes into the context."""
    def __init__(self, kind: str, utt: int, turn: Turn, held: bool = False):
        self.kind, self.utt, self.turn = kind, utt, turn
        self.units: list[tuple] = []        # ("w", word) / ("tool", name, args, result), in spoken order
        self.emitted_w = 0; self.emitted_u = 0
        self.segs: list[Segment] = []       # TTS segments (text, samples) in order
        self.out_buf = ""; self.n_seg = 0; self.raw = ""
        self.gen_done = False; self.cut = False
        self.task: asyncio.Task | None = None
        self.t_play: float | None = None
        self.end_handle = None              # the reply_end timer
        self.held = held                    # speculative: text deltas and audio buffered until promoted
        self.held_deltas: list[str] = []
        self.held_pcm: list[bytes] = []
        self.ctx_snapshot = None            # context before the speak call (early-start undo)
        self.start_ms = 0.0                 # audio ms when the reply became live
        self.text = ""
        self.t0 = time.monotonic()
        self.play_ms: float | None = None   # mic ms at which our audio started playing
        self.fresh_words = 0                # interrupt: user words heard after the user could hear us
        self.min_pin = 0                    # an overlap released after a hold pins no earlier than this word

    def first_sentence_words(self) -> int | None:
        """words up to the end of the first sentence (all of them once generation is done); None: not generated yet"""
        n = 0
        for u in self.units:
            if u[0] != "w": continue
            n += 1
            if re.search(r"[.!?][\"')\]]*$", u[1]): return n
        return n if self.gen_done else None

    def n_words(self) -> int:
        return sum(1 for u in self.units if u[0] == "w")

    def audio_s(self) -> float:
        return sum(s.samples for s in self.segs) / TTS_RATE

    def played_words(self, t: float) -> int:
        """words fully played at wall time t, from each segment's audio length (characters spread evenly over it)"""
        if self.t_play is None or t <= self.t_play:
            return 0
        el = t - self.t_play; n = 0
        for s in self.segs:
            dur = s.samples / TTS_RATE
            ws = s.text.split()
            if dur <= 0 or not s.done:
                return n
            if el >= dur:
                n += len(ws); el -= dur; continue
            frac = el / dur; chars = sum(len(w) + 1 for w in ws); acc = 0
            for w in ws:
                acc += len(w) + 1
                if acc / chars <= frac: n += 1
                else: break
            return n
        return n


# v4 package (10-13/14): early speculation from the turn head's anticipation outputs. ANT4="<p index>:<tau>[,...]":
# a speculation starts as soon as one output is over its threshold (6:0.7 = P(the user's turn ends within 160 ms) >= 0.7,
# all_ant80 head) or today's trigger fires, whichever comes first. One slot. When today's trigger (silent,
# P(fire) >= spec_tau) comes while an early speculation runs, it peeks again: same words -> the early one is kept
# (confirmed), else it is replaced by a fresh speculation from that peek. The endpoint rule's release waits for the
# confirmation. An early speculation survives ANT4_AGE_S of continued speech. Empty ANT4 = off (old heads have no
# anticipation outputs; p shorter than the index never fires).
ANT4 = CFG.ant4_triggers
ANT4_AGE_S = CFG.ant4_age_s
# ANT4_CONFIRM_SKIP (default on): today's trigger confirms an early speculation without a second peek when nothing changed
# since the early one: its fork is built and no head frame past the early peek's coverage heard speech (a new word
# discards the speculation anyway). Each peek stalls the live decoder, so the second one delayed the endpoint.
ANT4_CONFIRM_SKIP = CFG.ant4_confirm_skip


class Spec:
    """A speculative decision (+ reply) on a forked context, waiting for the endpoint rule to commit the turn."""
    def __init__(self, t: float, frame_ms: float, ctx_len: int, n_live: int):
        self.t, self.frame_ms, self.ctx_len, self.n_live = t, frame_ms, ctx_len, n_live
        self.peek_id = f"mspec:{frame_ms}"
        self.peek_ev: dict | None = None
        self.new_words: list[dict] = []     # the peek's words beyond the live stream
        self.norm: list[str] = []           # normalized words of the whole turn as the fork has them
        self.fork: P.Context | None = None
        self.dec: int | None = None
        self.rec: dict | None = None
        self.reply: Reply | None = None
        self.task: asyncio.Task | None = None
        self.promote_at: tuple | None = None   # (t, frame_ms): the rule committed while the spec was still running
        self.held_live: list[tuple] = []    # live words the peek already had: pushed only if the spec is discarded
        self.dead = False
        self.early_seen = False             # the early rule's condition held once (logged once per spec)
        self.early = False                  # (v4) fired by an anticipation output, possibly while the user still speaks
        self.confirmed = False              # (v4) today's trigger re-peeked and found the same words
        self.conf_sent = False              # (v4) that confirmation peek was sent
        self.conf_id = None
        self.wait_task = None               # (v4) building the fork waits for in-flight word calls
        self.cover_ms = None                # (v4) the last head frame when the early peek came back: what it covered
        self.speech_ms = float("-inf")      # (v4) the last head frame that heard speech since the early fire


class MicroSession(Session):
    def __init__(self, cfg, hub, send, log_dir=None, mcfg: MicroCfg | None = None):
        super().__init__(cfg, hub, send, log_dir)
        self.m = mcfg or MicroCfg()
        self.llm = Engine(self.m.url.removesuffix("/v1/completions"), self.m.lora, served=self.m.think_model,
                          reply_temp=self.m.reply_temp, reply_top_p=self.m.reply_top_p, reply_top_k=self.m.reply_top_k)
        tools = [P.tool_declaration("web_search", SEARCH_TOOL_SPEC)] if (self.m.search and cfg.search != "off") else []
        if self.m.tools: tools += [P.tool_declaration(n, CAT[n]) for n in LOCAL_TOOLS]
        if self.m.tools and self.m.reset_tool: tools.append(P.tool_declaration(RESET_TOOL, RESET_TOOL_SPEC))
        self.claude = None; self.frontier = None; self.async_pending: list[tuple] = []   # results waiting for our reply
        if self.m.claude:
            tools.append(P.tool_declaration(CLAUDE_TOOL, CLAUDE_TOOL_SPEC))
            self.claude = ClaudeTasks(lambda tid, r: self._task_done(CLAUDE_TOOL, tid, r), log_dir=log_dir,
                                      model=self.m.claude_model, mode=self.m.claude_mode, cwd=self.m.claude_cwd)
        if self.m.frontier:
            tools.append(P.tool_declaration(FRONTIER_TOOL, FRONTIER_SPEC_LIVE))
            self.frontier = FrontierTasks(lambda tid, r: self._task_done(FRONTIER_TOOL, tid, r), log_dir=log_dir)
        if self.m.claude or self.m.frontier:
            tools += [P.tool_declaration("task_status", CAT["task_status"]), P.tool_declaration("cancel_task", CAT["cancel_task"])]
        self.client_tool_names = {t["name"] for t in self.m.client_tools}
        if self.client_tool_names:
            tools = [d for d in tools if d["function"]["name"] not in self.client_tool_names]
            tools += [P.tool_declaration(t["name"], {"description": t.get("description", ""), "parameters": t.get("parameters") or {}}) for t in self.m.client_tools]
        self.toolbox = ToolBox(None)
        self.user_rules = self.m.instructions   # standing instructions (end of the system prompt): from the API client, or given with reset_chat
        self.reset_ask: dict | None = None  # reset_chat asked for confirmation: {t, utt, instructions}
        self.reset_due: str | None = None   # confirmed: the reset happens when this reply ends (value = the new instructions)
        self.n_resets = 0
        self.system = self._system_text()
        self.ctx = P.Context(); self.ctx.start(self.system, tools)
        self.inq: list[tuple] = []; self.inq_ev = asyncio.Event(); self.busy = False
        self.decider: asyncio.Task | None = None
        self.R: Reply | None = None; self.overlap: dict | None = None
        self.ducked = False                 # the output is ducked (barge_in "head" / "duck")
        self.IJ: Reply | None = None        # the interjections since the last reply (one utt: their audio queues, never cuts)
        self.decls = tools                  # declared tools (the thinker's tool list; subclasses that redeclare set it too)
        self.hist: list[str] = []           # the conversation as plain text, synth_notes.history() format (thinker input)
        self.think: dict[int, dict] = {}    # turn id -> thinker state (messages, notes, queued points, task)
        self.mspec: Spec | None = None
        self.head_frame_ms = None           # the newest audio frame the turn head has seen (ANT4_CONFIRM_SKIP)
        self.bc_idle = False                # an idle <user_bc> was sent since the last word (one per gap)
        self.result_unreported = False
        self.unreported_results: list[dict] = []   # async results not yet mentioned: {name, result, h (hist index)}
        self.pending_fire: dict | None = None
        # endpoint rule
        self.rule_open = False; self.rule_text: list[str] = []; self.hits = 0; self.prev_bc = 0.0
        # silence ladder (audio ms)
        self.sil_anchor_ms = 0.0; self.sil_idx = 0; self.user_speaking = False; self.onset_t: float | None = None
        self.calls_log: list[dict] = []; self.tl: list[str] = []
        self.bc_cont_ms: float | None = None   # audio ms of the last <continue> on a backchannel (its late words are dropped)
        self.ov_hold: list[tuple] = []; self.ov_hold_task: asyncio.Task | None = None   # overlap input waiting (_hold)
        self.counters.update({"calls": 0, "speak": 0, "interrupt": 0, "listen": 0, "listen_muted": 0, "continue": 0, "interject": 0,
                              "yield": 0, "harness_yield": 0, "undo": 0, "overlaps": 0, "mspec_fired": 0,
                              "mspec_promoted": 0, "mspec_discarded": 0, "mspec_wasted_replies": 0, "mspec_parallel_killed": 0, "early_release": 0, "early_fired": 0, "early_confirmed": 0, "early_replaced": 0, "early_confirm_skipped": 0, "tools": 0,
                              "stale_dropped": 0, "held_overlaps": 0, "bare_complete_dropped": 0, "tool_hold_waits": 0,
                              "silence_fallback": 0, "notes_written": 0, "notes_inserted": 0, "replies_with_notes": 0,
                              "notes_unfinished": 0, "phrase_cache_hits": 0, "phrase_cache_misses": 0})

    def _extra_facts(self, start):
        """the session's own tools as facts: the model describes itself from the Facts line, not the declarations"""
        extra = []
        if self.m.tools and self.m.reset_tool: extra.append("You can wipe this chat and start over when the user asks.")
        if self.m.claude: extra.append("You can ask Claude, an AI agent working on the user's computer, questions or hand it tasks.")
        if self.m.frontier: extra.append("For hard questions you can ask a bigger, smarter model.")
        return "".join(f" {k}. {e}" for k, e in enumerate(extra, start=start))

    def _system_text(self):
        rules = " The user's standing instructions: " + self.user_rules.rstrip(".") + "." if self.user_rules else ""
        s = PERSONAS[self.m.persona].format(city=DEFAULT_CITY) + self._extra_facts(3)
        return s + rules + "\n" + P.date_line(datetime.datetime.now(), DEFAULT_TZ)

    # ------------------------------------------------------------------ lifecycle
    async def start(self):
        await super().start()
        try:                                                  # the system prompt into vLLM's prefix cache
            await self.llm.warm(self.http, self.ctx.ids)
        except Exception as e:
            self.send({"type": "error", "where": "llm", "detail": repr(e)})
        self.decider = asyncio.create_task(self._decide_loop())
        self.send({"type": "brain", "brain": "speakrail", "system": self.system, "bias": self.m.bias,
                   "end_rule": {"tau": self.m.tau, "k": self.m.k}})

    async def close(self):
        if self.closed:
            return
        if self.mspec is not None: self._spec_discard(time.monotonic(), "session closed")
        if self.claude is not None: self.claude.close()
        if self.frontier is not None: self.frontier.close()
        if self.decider: self.decider.cancel()
        for th in self.think.values():
            if th["task"] is not None and not th["task"].done(): th["task"].cancel()
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(f"{self.log_dir}/timeline.txt", "w") as f: f.write("\n".join(self.tl) + "\n")
            with open(f"{self.log_dir}/calls.jsonl", "w") as f:
                for c in self.calls_log: f.write(json.dumps(c) + "\n")
            try:
                with open(f"{self.log_dir}/context.txt", "w") as f: f.write(self.ctx.render())
            except Exception:
                pass
        await super().close()

    def _rel(self, t=None):
        return round((t if t is not None else time.monotonic()) - self.t0, 3)

    def _line(self, s: str):
        self.tl.append(f"{self._rel():8.2f}  {s}")

    # ------------------------------------------------------------------ vLLM
    async def _post(self, body):
        async with self.http.post(self.m.url, json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"llm {r.status}: {(await r.text())[:200]}")
            return await r.json()

    async def _decide_ids(self, ids, options):
        allowed = P.DECISION_SETS[options]
        if not self.m.allow_interject:                # no live tasks in this harness yet: <interject> is never offered
            allowed = [d for d in allowed if d != P.INTERJECT]
        dec, pr, _, ms = await self.llm.decide(self.http, ids, allowed, self.m.bias)
        probs = {P.DECISION_NAME[i]: round(v, 4) for i, v in pr.items()}
        return dec, {"set": options, "decision": P.DECISION_NAME[dec], "probs": probs, "ms": round(ms, 1), "ctx": len(ids)}

    async def _decide(self, options, why):
        dec, rec = await self._decide_ids(self.ctx.ids, options)
        rec.update(t=self._rel(), why=why)
        self.calls_log.append(rec); self.counters["calls"] += 1; self.counters[rec["decision"]] += 1
        self.send({"type": "decision", **rec})
        pr = " ".join(f"{k}={v:.2f}" for k, v in sorted(rec["probs"].items(), key=lambda kv: -kv[1])[:3])
        self._line(f"    CALL {why:14s} -> {rec['decision']:12s} {rec['ms']:6.1f} ms  {pr}")
        return dec

    # ------------------------------------------------------------------ input queue (serial calls)
    def _push(self, items: list[tuple]):
        if self.closed or not items:
            return
        self.inq.extend(items); self.inq_ev.set()

    async def _decide_loop(self):
        try:
            while True:
                await self.inq_ev.wait(); self.inq_ev.clear()
                while self.inq:
                    batch, self.inq = self.inq, []
                    self.busy = True
                    try:
                        await self._process(batch)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self.send({"type": "error", "where": "decide", "detail": repr(e)})
                    finally:
                        self.busy = False
        except asyncio.CancelledError:
            pass

    def _append_input(self, it):
        k = it[0]
        if k == "word":
            self.ctx.word(it[1]); self._line(f"user  {it[1]}" + ("  (peek)" if len(it) > 2 and it[2].get("peek") else ""))
        elif k == "complete":
            self.ctx.input_token("<complete>"); self._line("<complete>")
        elif k == "user_bc":
            self.ctx.input_token("<user_bc>"); self._line("<user_bc>")
        elif k == "sil":
            self.ctx.input_token(f"<sil:{it[1]}s>"); self._line(f"<sil:{it[1]}s>")

    async def _process(self, batch):
        if self.reset_due is not None and (self.R is None or self.R.cut):   # the confirming reply was cut: reset now
            self._do_reset()
        for it in [x for x in batch if x[0] == "reply_end"]:
            await self._reply_end(it[1])
        # async tool results (claude_code): into the stream now, or after our reply if we are talking (as in
        # training for a normal-urgency result); then a decision, as training calls after a result
        self.async_pending.extend(x for x in batch if x[0] == "async")
        fresh = self._flush_async() if (self.R is None or self.R.cut) else 0
        items = [x for x in batch if x[0] in ("word", "complete", "user_bc", "sil")]
        if self.bc_cont_ms is not None and self.R is not None:
            # as the tape builder: the words of a backchannel we already continued over (they arrive one ASR delay after
            # its <user_bc>) never reach the context
            late = [x for x in items if x[0] == "word" and x[2]["start"] <= self.bc_cont_ms + 200 and is_backchannel(x[1])]
            if late:
                self._line(f"(late words of a continued backchannel dropped: {' '.join(x[1] for x in late)})")
                items = [x for x in items if x not in late]
        R = self.R
        if R is not None and not R.cut and R.kind == "interrupt":
            items = self._drop_stale(R, items)
        if any(x[0] == "nudge" for x in batch) and (self.R is None or self.R.cut) and not items:
            # an async result waited unreported through result_nudge_s of quiet: the harness takes the turn.
            # only for results no reply has mentioned yet, and the result goes back in front of the model first:
            pending = [u for u in self.unreported_results if not self._result_mentioned(u)]
            self.unreported_results = []
            if not pending:
                self._line("(result nudge: the result was already mentioned, no turn taken)")
                return
            self.counters["result_nudges"] = self.counters.get("result_nudges", 0) + 1
            self._line(f"(result unreported after {self.m.result_nudge_s:.0f} s of quiet: the result goes back in, the harness takes the turn)")
            for u in pending: self.ctx.async_result(u["name"], u["result"])
            snap = self._snapshot(); self._force_speak()
            if self.m.notes:
                n = self._insert_notes(self.ctx, self.turn)
                if n: self.counters["notes_inserted"] += n; self.counters["replies_with_notes"] += 1
            self._start_reply("speak", self._close_user_turn(False), snap)
            return
        if any(x[0] == "nudge" for x in batch) and self.unreported_results:
            self.result_unreported = True                     # the nudge could not take the turn now: re-arm it
        if not items:
            if fresh: await self._idle_call("tool")
            return
        if (R is not None and not R.cut and self.overlap is None and not self.ov_hold
                and not any(x[0] in ("word", "user_bc") for x in items)):
            bare = [x for x in items if x[0] in ("complete", "sil")]
            if bare:                                          # nobody spoke over us: no overlap turn opens on it
                # a <sil:Ns> queued just before the reply started too: with the 1 s silence fallback the
                # <sil:1s> marker lands with the fallback <complete> and opened an overlap pinned at reply word 1
                self.counters["bare_complete_dropped"] += 1
                self._line(f"({' '.join('<' + x[0] + '>' for x in bare)} while we speak, no speech over us: dropped)")
                items = [x for x in items if x[0] not in ("complete", "sil")]
                if not items:
                    return
        if R is not None and not R.cut:
            if self._interrupt_holds(R) or (self.ov_hold and R.kind == "interrupt"):
                self._hold(R, items, "until the interrupt's first sentence has played"); return
            if await self._overlap(R, items):
                return
        why = "+".join(sorted({x[0] for x in items}))
        for it in items: self._append_input(it)
        await self._idle_call(why, complete="complete" in why)

    async def _idle_call(self, why, complete=False):
        snap = self._snapshot()
        dec = await self._decide("idle", why)
        self.ctx.call(dec, "idle")
        if dec == P.SPEAK and self.m.notes:
            n = self._insert_notes(self.ctx, self.turn)
            if n: self.counters["notes_inserted"] += n; self.counters["replies_with_notes"] += 1; self._line(f"    NOTES ({n}) inserted at the reply start")
        if dec in (P.SPEAK, P.INTERRUPT):
            tr = self._close_user_turn(complete)
            self._start_reply(P.DECISION_NAME[dec], tr, snap)
        elif dec == INTERJECT:                         # the user's turn goes on: no reply, no overlap, no turn close
            await self._interject()

    def _spec_confirm(self, t, frame_ms):
        """(v4) today's trigger came while an early speculation runs: peek again to check its words, unless nothing
        changed since the early peek (ANT4_CONFIRM_SKIP: the fork is built and no head frame past the early peek's
        coverage heard speech, so a second peek would read the same words)"""
        sp = self.mspec; sp.conf_sent = True
        if ANT4_CONFIRM_SKIP and sp.fork is not None and sp.cover_ms is not None and sp.speech_ms <= sp.cover_ms:
            self.counters["early_confirm_skipped"] += 1
            self.send({"type": "early_confirm", "t": self._rel(t), "same": True, "skipped": True,
                       "age_ms": round((t - sp.t) * 1000)})
            return self._early_confirmed(sp)
        sp.conf_id = f"mconf:{frame_ms}"
        n = self.cfg.peek_frames or (max(self.cfg.delay_ms, 0) // 80 + 2)
        asyncio.ensure_future(self.asr_ws.send(json.dumps({"type": "peek", "frames": n, "id": sp.conf_id})))

    def _early_confirmed(self, sp):
        sp.confirmed = True; self.counters["early_confirmed"] += 1
        if sp.promote_at is not None and sp.dec is not None:   # the rule already fired: release the reply now
            t0, fm = sp.promote_at
            if not self._spec_promote(t0): self._commit(t0, fm)

    def _on_confirm_peek(self, ev, t):
        """(v4) the confirmation peek: same words -> keep the early speculation, else replace it from this peek"""
        sp = self.mspec
        if sp is None or sp.dead or sp.conf_id != ev.get("id") or self.turn is None:
            return
        new = self._peek_new(ev)
        norm2 = normalize(self.turn.text() + "".join(" " + w["text"] for w in new))
        same = sp.fork is not None and norm2 == sp.norm
        self.send({"type": "early_confirm", "t": self._rel(t), "same": same, "age_ms": round((t - sp.t) * 1000),
                   "early_words": " ".join(sp.norm[-4:]), "now_words": " ".join(norm2[-4:])})
        if same:
            return self._early_confirmed(sp)
        self.counters["early_replaced"] += 1
        pa = sp.promote_at
        self._spec_discard(t, "early words wrong", commit_pending=False)
        if self._agent_speaking():
            if pa is not None: self._commit(*pa)
            return
        sp2 = self.mspec = Spec(t, 0, len(self.ctx.ids), len(self.turn.words)); sp2.peek_id = ev.get("id"); sp2.promote_at = pa
        sp2.early = True; sp2.confirmed = True; sp2.conf_sent = True      # today's trigger's own peek: no confirmation needed
        self.counters["mspec_fired"] += 1
        self.send({"type": "spec", "turn": self.turn.id, "t": self._rel(t), "frame_ms": None, "early": False, "replaces": True})
        self._on_spec_peek(ev, t)

    # ------------------------------------------------------------------ interjections (live tasks)
    async def _interject(self):
        """<interject> (the context already ends with it): as in training, the model writes its
        few words like a reply (decision tokens and <|channel> banned, <interject> too), stopping on </interject>; a tool
        call runs for real and generation goes on after its result. A stop on anything else (<turn|>, max tokens) is closed
        by the harness with </interject>. Awaited inside the decide loop, so the interjection sits in the context exactly
        at its call and the user's next words come after it. Its words go to TTS at once; the audio uses self.IJ's utt,
        which the pages queue after whatever plays (an "interject" message marks it), so a reply never cuts it."""
        J = self.IJ
        if J is None:
            J = self.IJ = Reply("interject", self.hub.new_utt(self), Turn(-1))
        prompt = list(self.ctx.ids); pieces = []; stop = None; t0 = time.monotonic(); ttft = None
        banned = {int(k): v for k, v in BANNED_IN_REPLY.items() if int(k) != INTERJECT_END}
        for rnd in range(self.m.tool_rounds):
            raw = ""
            st = self.llm.stream(self.http, prompt, INTERJECT_MAX_TOKENS, [INTERJECT_END, TOOL_RESPONSE_OPEN, P.SPEAK], banned)
            async for d in st:
                if ttft is None: ttft = (time.monotonic() - t0) * 1000
                raw += d
            stop = st.stop
            words = " ".join(raw.split("<|tool_call>")[0].split())     # as in training: the words joined by single spaces
            if words:
                pieces.append(("text", words)); self._interject_say(J, words)
            calls = list(CALL_RE.finditer(raw))
            if stop == TOOL_RESPONSE_OPEN and calls:
                res = []
                for m in calls:
                    name = m.group(1); args = self._parse_args(m.group(2)) or {}
                    res.append((name, args, await self._run_tool(J, name, args)))
                pieces.append(("calls", res))
                prompt = prompt + P.encode(raw[:calls[-1].end()] + "<|tool_response>") + P.encode(
                    P.tool_results_text([(n, r) for n, _, r in res]))
                continue
            break
        self._ctx_interject(pieces)
        self._hist_add("Assistant (live task, said while they kept talking): " + " ".join(x for k, x in pieces if k == "text"))
        text = " ".join(x if k == "text" else f"[{'+'.join(n for n, _, _ in x)}]" for k, x in pieces)
        closed = "model" if stop == INTERJECT_END else f"harness (stop {stop})"
        self._line(f"    INTERJECT (utt {J.utt}, ttft {ttft and round(ttft, 1)} ms, {round((time.monotonic() - t0) * 1000)} ms, "
                   f"closed by {closed}): {text!r}")
        self._log({"type": "interject", "utt": J.utt, "text": text, "closed": closed, "ttft_ms": ttft and round(ttft, 1),
                   "ms": round((time.monotonic() - t0) * 1000)})

    def _interject_say(self, J: Reply, words: str):
        J.units.extend(("w", w) for w in words.split())
        seg = strip_tags(words)
        self.send({"type": "interject", "utt": J.utt, "text": words, "t": self._rel()})
        if seg.strip():
            s = Segment(J.utt, J.n_seg, seg); J.n_seg += 1; J.segs.append(s)
            self.hub.worker.say(s)

    def _ctx_interject(self, pieces):
        """into the context exactly as training writes it: Context.interject(words, calls) = words [<|tool_call>...
        <|tool_response>response:...<tool_response|>] </interject> (no loss: the model's own output). Shapes training
        never have (words after a tool result, a second round of calls) go in the same way, in the order the model wrote
        them, then </interject>. turn_has_content: the user's next word keeps its leading space."""
        kinds = [k for k, _ in pieces]
        if kinds in ([], ["text"], ["calls"], ["text", "calls"]):
            self.ctx.interject(next((x for k, x in pieces if k == "text"), ""), next((x for k, x in pieces if k == "calls"), None), loss=False)
            return
        for k, x in pieces:
            if k == "text": self.ctx._append(x, "interject_noloss")
            else:
                self.ctx._append(P.tool_calls_text([(n, a) for n, a, _ in x]), "tool_call")
                self.ctx._append(P.tool_results_text([(n, r) for n, _, r in x]), "tool_result")
        self.ctx._append_id(INTERJECT_END, "interject_end"); self.ctx.turn_has_content = True

    # ------------------------------------------------------------------ listening notes (--notes)
    def _think_tools(self):
        out = []
        for d in (self.decls or [])[:25]:
            f = d.get("function", d)
            out.append(f"- {f.get('name')}: {(f.get('description') or '').split('. ')[0]}")
        return "\n".join(out) or "(none)"

    def _think_word(self):
        """a live word of the open user turn: new note points (protocol.note_points over the words as they arrived, the
        rule synth_notes.py used) queue a note; one thinker task per turn works through them in order"""
        tr = self.turn
        if tr is None or not (self.m.notes and THINK_SYS): return
        th = self.think.setdefault(tr.id, {"msgs": None, "notes": [], "pts": 0, "upto": 0, "queue": [], "task": None})
        pts = P.note_points([(w["text"], w["t"]) for w in tr.words])
        if len(pts) <= th["pts"]: return
        th["queue"] += pts[th["pts"]:]; th["pts"] = len(pts)
        if th["task"] is None or th["task"].done():
            th["task"] = asyncio.ensure_future(self._think_run(tr, th))

    async def _think_run(self, tr, th):
        prev_upto = th["upto"]
        try:
            if th["msgs"] is None:                       # synth_notes.py's message shape
                th["msgs"] = [{"role": "system", "content": THINK_SYS.format(system=self.system, tools=self._think_tools())},
                              {"role": "user", "content": "Conversation so far:\n" + ("\n".join(self.hist[-40:]) or "(start of the conversation)")},
                              {"role": "assistant", "content": "STATE: nothing yet, waiting for the user."}]
            while th["queue"]:
                i = th["queue"][-1]; th["queue"].clear()   # behind: everything heard up to the latest point in one piece
                prev_upto = th["upto"]
                chunk = " ".join(w["text"] for w in tr.words[th["upto"]: i + 1]); th["upto"] = i + 1
                if not chunk: continue
                th["msgs"].append({"role": "user", "content": chunk})
                t0 = time.monotonic()
                body = {"model": self.m.think_model, "messages": th["msgs"], "temperature": 0, "max_tokens": P.NOTE_MAX_TOKENS,
                        "chat_template_kwargs": {"enable_thinking": False}}
                async with self.http.post(self.m.url.replace("/v1/completions", "/v1/chat/completions"), json=body) as r:
                    if r.status != 200: raise RuntimeError(f"thinker {r.status}: {(await r.text())[:200]}")
                    d = await r.json()
                note = (d["choices"][0]["message"]["content"] or "").strip()
                th["msgs"].append({"role": "assistant", "content": note or "(no note)"})
                if note:
                    th["notes"].append(note); self.counters["notes_written"] += 1
                    self._line(f"    NOTE ({round((time.monotonic() - t0) * 1000)} ms, {len(note)} chars): {note[:160]!r}")
        except asyncio.CancelledError:
            if th["msgs"] and th["msgs"][-1]["role"] == "user":   # the unfinished note: its piece is asked again later
                th["msgs"].pop(); th["upto"] = prev_upto
        except Exception as e:
            self.send({"type": "error", "where": "notes", "detail": repr(e)})

    def _insert_notes(self, ctx, tr) -> int:
        """at a speak decision: the turn's finished notes (the most recent that fit NOTES_MAX_CHARS) as a thought block"""
        th = self.think.get(tr.id) if (tr is not None and self.m.notes) else None
        if not th or not th["notes"]: return 0
        keep, size = [], 0
        for n in reversed(th["notes"]):
            if size + len(n) > NOTES_MAX_CHARS and keep: break
            keep.insert(0, n); size += len(n) + 2
        ctx.notes("\n\n".join(keep), "thought")
        return len(keep)

    def _think_close(self, tr):
        """the turn is answered: an unfinished note is dropped (state kept: an early-start undo reopens the turn)"""
        th = self.think.get(tr.id)
        if th and th["task"] is not None and not th["task"].done():
            th["task"].cancel(); self.counters["notes_unfinished"] += 1

    def _hist_add(self, line, key=None):
        if not line: return
        if key is not None:
            k = self.__dict__.setdefault("hist_keys", {})
            if key in k: self.hist[k[key]] = line; return
            k[key] = len(self.hist)
        self.hist.append(line)

    def _hist_reply(self, R):
        buf = []
        for u in R.units:
            if u[0] == "w": buf.append(u[1]); continue
            if buf: self._hist_add("Assistant: " + " ".join(buf)); buf = []
            self._hist_add(f"Assistant called {u[1]}({json.dumps(u[2], ensure_ascii=False)[:200]}) -> {json.dumps(u[3], ensure_ascii=False)[:300]}")
        if buf: self._hist_add("Assistant: " + " ".join(buf))

    # ------------------------------------------------------------------ context snapshot / rollback (early-start undo)
    def _snapshot(self):
        c = self.ctx
        return (len(c.ids), len(c.targets), len(c.calls), len(c.spans), c.turn_has_content, c.reply_last)

    def _rollback(self, snap):
        c = self.ctx; n_ids, n_tg, n_calls, n_spans, thc, rl = snap
        del c.ids[n_ids:]; del c.targets[n_tg:]; del c.calls[n_calls:]; del c.spans[n_spans:]
        c.turn_has_content, c.reply_last = thc, rl
        if n_tg: c.targets[n_tg - 1] = -100                   # the undone call's target

    # ------------------------------------------------------------------ user turn bookkeeping (latency chain, UI)
    def _close_user_turn(self, complete) -> Turn:
        tr = self.turn
        if tr is None:                                       # a reply without new words (check-in, after a result)
            tr = Turn(self.n_turns); self.n_turns += 1; self.turn_objs.append(tr)
        tr.t_endpoint = time.monotonic(); tr.ep_reason = "complete" if complete else "call"
        tr.t_mic_offset = self.mic_last_hot
        if tr.words: self._hist_add("User: " + tr.text().strip(), key=("turn", tr.id))   # (an undo reopens it: replaced)
        self._think_close(tr)
        self.turn = None; self.last_ended = tr
        self.counters["endpoints"] += 1
        return tr

    # ------------------------------------------------------------------ ASR words
    def _on_words(self, ev, t):
        words = [w for w in ev.get("words", []) if not self._peek_duplicate(w)]
        if not words:
            return
        self.bc_idle = False
        items = [("word", w["text"], self._word_rec(w, t, peek=False)) for w in words]
        sp = self.mspec
        if sp is not None and sp.early and (sp.peek_ev is None or sp.fork is None):
            pass                                              # (v4) the early fork is built from the context at peek time
        elif sp is not None:
            if sp.peek_ev is None:
                self._spec_discard(t, "new words")
            else:
                live = normalize(self.turn.text()) if self.turn else []
                if len(live) > len(sp.norm) or live != sp.norm[:len(live)]:
                    self._spec_discard(t, "new words")
                else:
                    # the fork already has them (the spec's peek read them out of the delay line): not new speech, so
                    # they don't arm the endpoint rule (it is open anyway until it fires; once it has fired, re-arming
                    # it would send a second <complete> while the adopted reply plays)
                    sp.held_live.extend(items)
                    return
        self.rule_open = True; self.rule_text.extend(w["text"] for w in words)
        self._push(items)

    def _word_rec(self, w, t, peek):
        if self.turn is None:
            self.turn = Turn(self.n_turns); self.n_turns += 1; self.turn_objs.append(self.turn)
        w_end_wall = self.clock.wall_of(w["end"])
        lag = None if (peek or w_end_wall is None) else round((t - w_end_wall) * 1000)
        rec = {"text": w["text"], "raw": " " + w["text"], "start": w["start"], "end": w["end"], "lag_ms": lag,
               "fed_ms": round(self.clock.audio_ms), "t": self._rel(t), "turn": self.turn.id}
        if peek: rec["peek"] = True
        self.turn.words.append(rec); self.words_all.append(rec)
        if lag is not None: self.word_lags.append(lag)
        self.turn.t_last_word_ev = t
        self.turn.last_word_end_ms = w["end"]; self.turn.t_word_end = w_end_wall
        self.send({"type": "word", **rec})
        if not peek and self.m.notes: self._think_word()
        return rec

    # ------------------------------------------------------------------ turn head
    def _agent_speaking(self) -> bool:
        return self.R is not None and not self.R.cut

    def _on_head(self, ev, t, offset_ms):
        p = ev["p"]; self.head_p = p
        frame_ms = ev["t"] + offset_ms
        self.head_frame_ms = frame_ms
        agent = self._agent_speaking()
        self.send({"type": "head", "t": self._rel(t), "frame_ms": frame_ms, "p": p, "armed": self.pending_fire is not None,
                   "spec": self.mspec is not None, "agent": agent, "turn": self.turn.id if self.turn is not None else None})
        bc_onset = p[3] >= self.m.user_bc_tau and self.prev_bc < self.m.user_bc_tau
        self.prev_bc = p[3]
        # a <user_bc> only over our speech, or once inside a user turn that has words: the head's bc class firing
        # on noise while nobody spoke put runs of 131 and 69 into the context (training has at most 2 in a row)
        if bc_onset and (agent or (self.turn is not None and self.turn.words and not self.bc_idle)):
            self._push([("user_bc", frame_ms)])
            if not agent: self.bc_idle = True
        if (ANT4 and not agent and self.mspec is None and self.rule_open and self.turn is not None and self.turn.words
                and self.pending_fire is None and any(len(p) > i and p[i] >= tau for i, tau in ANT4)):
            self._spec_fire(t, frame_ms, early=True)
        if p[0] >= 0.5:                                          # the user is speaking
            if self.sil_frames > 0: self.seg_bc = 0.0
            self.sil_frames = 0; self.hits = 0; self.spk_frames += 1
            self.seg_bc = max(self.seg_bc, p[3])
            self.sil_anchor_ms = frame_ms + 80; self.sil_idx = 0
            if agent:
                if self.onset_t is None: self.onset_t = t
                if self.cfg.barge_in in ("head", "duck") and self.R.kind != "interrupt" and not self._interrupt_holds(self.R):
                    if not self.ducked: self._line("(ducking: the user speaks over us)")
                    self.ducked = True; self.send({"type": "duck", "gain": self.m.duck_gain})
            sp = self.mspec
            if sp is not None and sp.early: sp.speech_ms = max(sp.speech_ms, frame_ms)
            if sp is not None and not (sp.early and t - sp.t < ANT4_AGE_S):
                self._spec_discard(t, "speech")
            return
        self.sil_frames += 1; self.spk_frames = 0
        if (self.m.result_nudge_s > 0 and self.result_unreported and not agent and self.mspec is None
                and frame_ms - self.sil_anchor_ms >= self.m.result_nudge_s * 1000):   # quiet since the last speech of
                                                                                     # EITHER side (our reply too), not the user's
            self.result_unreported = False; self._push([("nudge",)])
        self.seg_bc = max(self.seg_bc, p[3])
        if self.ducked and self.sil_frames * 80 >= self.m.unduck_ms:   # the user stopped: full volume again
            self._unduck()
        if not agent: self.onset_t = None
        self._sil_tick(frame_ms)
        if not self.rule_open:
            return
        if agent and (self.seg_bc >= 0.5 or is_backchannel(" ".join(self.rule_text))):
            if self.sil_frames * 0.08 >= self.m.bc_close_s:
                self.rule_open = False; self.rule_text = []
            return
        pf = p[1] if agent else p[1] + p[3]
        sp4 = self.mspec
        if sp4 is not None and sp4.early and not sp4.conf_sent and not agent and pf >= self.cfg.spec_tau:
            self._spec_confirm(t, frame_ms)
        if (self.cfg.spec and self.cfg.peek and not agent and self.mspec is None and pf >= self.cfg.spec_tau
                and not self.busy and not self.inq and self.pending_fire is None):
            self._spec_fire(t, frame_ms)
        if self._early_check(t, frame_ms, pf):
            return
        self.hits = self.hits + 1 if pf >= self.m.tau else 0     # end of turn: P(fire) >= tau on k frames in a row
        fired = self.hits >= self.m.k
        if fired:
            self.rule_open = False; self.rule_text = []; self.hits = 0
            self._fire(t, frame_ms)
        elif (not agent and self.m.silence_fallback_ms and self.sil_frames * 80 >= self.m.silence_fallback_ms
              and self.pending_fire is None):
            # the head never fired on this silence: the harness sends <complete> anyway (peek first, as the rule does);
            # the model still decides whether to speak
            self.rule_open = False; self.rule_text = []; self.hits = 0
            self.counters["silence_fallback"] += 1
            self._line(f"(no end-of-turn after {self.m.silence_fallback_ms} ms of silence: fallback <complete>)")
            self._fire(t, frame_ms)

    def _sil_tick(self, frame_ms):
        """<sil:Ns> while nobody speaks: after the user's last speech frame or the end of our own playback"""
        if self._agent_speaking() or self.sil_idx >= len(SIL_LADDER_S):
            return
        L = SIL_LADDER_S[self.sil_idx]
        if frame_ms - self.sil_anchor_ms >= L * 1000:
            self.sil_idx += 1
            sp = self.mspec
            if sp is not None and sp.peek_ev is not None and not sp.dead:
                # a spec waiting for the rule survives the marker: it goes into held_live, so it reaches the
                # context only if the spec is discarded (on promotion the fork's <complete> stands in for it). Discarding
                # here threw away ready replies on every slow head ("Hi, what's up?" waited 3.5 s for the rule)
                sp.held_live.append(("sil", L)); self._line(f"(<sil:{L}s> held: speculation pending)"); return
            if sp is not None: self._spec_discard(time.monotonic(), "silence marker")
            self._push([("sil", L)])

    def _early_check(self, t, frame_ms, pf) -> bool:
        """the early rule (MicroCfg.early): True if it released the held reply (the turn is committed)"""
        sp = self.mspec
        if (self.m.early == "off" or sp is None or sp.dead or sp.dec != P.SPEAK or (sp.early and not sp.confirmed)
                or sp.reply is None or self._agent_speaking()
                or sp.promote_at is not None or self.pending_fire is not None or pf < self.cfg.spec_tau):
            return False
        p_speak = sp.rec["probs"].get(P.DECISION_NAME[P.SPEAK], 0.0)
        if p_speak < self.m.early_tau:
            return False
        if not sp.early_seen:
            sp.early_seen = True
            self.send({"type": "early", "t": self._rel(t), "frame_ms": frame_ms, "p_speak": p_speak, "pf": round(pf, 3),
                       "age_ms": round((t - sp.t) * 1000), "shadow": self.m.early != "on"})
        if self.m.early != "on":
            return False
        self.counters["early_release"] += 1
        self._line(f"(early release: fork decided speak {p_speak:.3f}, head P(fire) {pf:.2f})")
        self.rule_open = False; self.rule_text = []; self.hits = 0
        self._fire(t, frame_ms, early=True)
        return True

    def _fire(self, t, frame_ms, early=False):
        self.counters["head_fires"] += 1
        self.send({"type": "arm", "reason": "early" if early else "head", "t": self._rel(t), "frame_ms": frame_ms})
        if self.turn is not None: self.turn.t_head_fire = t
        sp = self.mspec
        if sp is not None and sp.early and not sp.confirmed:
            sp.promote_at = (t, frame_ms)                     # (v4) release once today's trigger peek confirms the words
            if not sp.conf_sent: self._spec_confirm(t, frame_ms)
            return
        if sp is not None:
            if sp.dec is None:                                # the spec is still peeking / deciding: commit when it lands
                sp.promote_at = (t, frame_ms); return
            if self._spec_promote(t, early):
                return
        self._commit(t, frame_ms)

    def _commit(self, t, frame_ms):
        if self.cfg.peek and self.asr_ws is not None:
            n = self.cfg.peek_frames or (max(self.cfg.delay_ms, 0) // 80 + 2)
            self.pending_fire = {"id": frame_ms, "t": t}
            asyncio.ensure_future(self.asr_ws.send(json.dumps({"type": "peek", "frames": n, "id": frame_ms})))
        else:
            wait = (max(self.cfg.delay_ms, 0) // 80 + self.cfg.commit_extra_frames) * 0.08
            asyncio.get_running_loop().call_later(wait, lambda: self._push([("complete",)]))

    def _on_peek(self, ev, t):
        if str(ev.get("id", "")).startswith("mconf:"):
            return self._on_confirm_peek(ev, t)
        sp = self.mspec
        if sp is not None and ev.get("id") == sp.peek_id:
            return self._on_spec_peek(ev, t)
        pf = self.pending_fire
        if pf is None or ev.get("id") != pf["id"]:
            return
        self.pending_fire = None
        new = self._peek_new(ev)
        items = [("word", w["text"], self._word_rec(w, t, peek=True)) for w in new]
        self._peek_consumed(ev)
        if self.turn is not None:
            self.turn.t_peek = t; self.turn.peek_ms = ev.get("ms"); self.turn.peek_words = len(new)
        self.send({"type": "peek", "t": self._rel(t), "ms": ev.get("ms"), "n_words": len(new), "text": ev.get("text", "")})
        self.send({"type": "endpoint", "turn": self.turn.id if self.turn else None,
                   "text": self.turn.text() if self.turn else "", "t": self._rel(t), "reason": "complete"})
        self._push(items + [("complete",)])

    def _peek_new(self, ev):
        last = self.turn.last_word_end_ms if (self.turn and self.turn.last_word_end_ms is not None) else -1
        # and never a word an earlier peek already delivered: once the turn is closed, a second fire's peek
        # (the rule re-opened by a late live word) returned the same word as new speech -> early-start undo + a duplicate
        last = max(last, self.consumed_until_ms)
        return [w for w in ev.get("words", []) if w["end"] > last]

    def _peek_consumed(self, ev):
        if ev.get("words"):
            self.consumed_until_ms = max(self.consumed_until_ms, ev["words"][-1]["end"])
            self.peek_tail = [" ".join(normalize(w["text"])) for w in ev["words"][-3:]]
            self.peek_last = (ev["words"][-1]["start"], " ".join(normalize(ev["words"][-1]["text"])))

    # ------------------------------------------------------------------ speculation
    def _spec_fire(self, t, frame_ms, early=False):
        if self.asr_ws is None or self.turn is None:
            return
        sp = self.mspec = Spec(t, frame_ms, len(self.ctx.ids), len(self.turn.words)); sp.early = early
        if early: self.counters["early_fired"] += 1
        self.counters["mspec_fired"] += 1
        self.send({"type": "spec", "turn": self.turn.id, "t": self._rel(t), "frame_ms": frame_ms, "early": early})
        n = self.cfg.peek_frames or (max(self.cfg.delay_ms, 0) // 80 + 2)
        asyncio.ensure_future(self.asr_ws.send(json.dumps({"type": "peek", "frames": n, "id": sp.peek_id})))

    def _on_spec_peek(self, ev, t):
        sp = self.mspec
        if sp.dead:
            return
        if sp.early and sp.cover_ms is None:
            sp.cover_ms = self.head_frame_ms   # frames the head reports after this may be audio the peek didn't hear
        if sp.early and (self.busy or self.inq) and not self._agent_speaking() and self.turn is not None:
            # (v4) the user is mid-turn and their words' calls are in flight: build the fork once they land
            if sp.wait_task is None:
                async def later():
                    for _ in range(60):
                        await asyncio.sleep(0.005)
                        if sp.dead or self.mspec is not sp: return
                        if not (self.busy or self.inq):
                            sp.ctx_len = len(self.ctx.ids); return self._on_spec_peek(ev, time.monotonic())
                    if self.mspec is sp: self._spec_discard(time.monotonic(), "context busy")
                sp.wait_task = asyncio.ensure_future(later())
            return
        if sp.early: sp.ctx_len = len(self.ctx.ids)
        if self.busy or self.inq or len(self.ctx.ids) != sp.ctx_len or self._agent_speaking() or self.turn is None:
            return self._spec_discard(t, "context moved")
        sp.peek_ev = ev
        sp.new_words = self._peek_new(ev)
        sp.norm = normalize(self.turn.text() + "".join(" " + w["text"] for w in sp.new_words))
        fork = copy.deepcopy(self.ctx)
        for w in sp.new_words: fork.word(w["text"])
        fork.input_token("<complete>")
        sp.fork = fork
        self.send({"type": "spec_peek", "t": self._rel(t), "ms": ev.get("ms"), "n_words": len(sp.new_words),
                   "text": ev.get("text", ""), "turn": self.turn.id})
        sp.task = asyncio.ensure_future(self._spec_run(sp))

    async def _spec_run(self, sp: Spec):
        try:
            snap_fork = copy.deepcopy(sp.fork)
            R = gen_fork = None; n_notes = 0
            if self.m.spec_parallel:
                # the reply is requested together with the decision: the fork + a speak call (+ notes) goes to
                # the LLM at once and the decision (p50 58 ms) lands while its prefill runs (TTFT 76 ms). Any other decision
                # kills it, normally before its first token; a tool call can't run before that (tool_hold_ms comes first)
                gen_fork = copy.deepcopy(sp.fork)
                gen_fork.call(P.SPEAK, "idle")
                n_notes = self._insert_notes(gen_fork, self.turn) if self.m.notes else 0
                R = Reply(P.DECISION_NAME[P.SPEAK], self.hub.new_utt(self), Turn(-1), held=True)
                R.ctx_snapshot = ("fork", snap_fork)
                sp.reply = R                                      # a discard while deciding kills it
                R.task = asyncio.ensure_future(self._generate(R, list(gen_fork.ids)))
            dec, rec = await self._decide_ids(sp.fork.ids, "idle")
            if sp.dead:
                return
            sp.dec, sp.rec = dec, rec
            self.send({"type": "spec_decided", "t": self._rel(), "decision": rec["decision"], "probs": rec["probs"],
                       "ms": rec["ms"], "age_ms": round((time.monotonic() - sp.t) * 1000)})
            if R is not None and dec == P.SPEAK:
                sp.fork = gen_fork; sp.n_notes = n_notes
            else:
                if R is not None:
                    self._kill_reply_gen(R); sp.reply = None
                    self.counters["mspec_parallel_killed"] += 1
                sp.fork.call(dec, "idle")
                if dec == P.SPEAK and self.m.notes: sp.n_notes = self._insert_notes(sp.fork, self.turn)
                if dec in (P.SPEAK, P.INTERRUPT):
                    R = Reply(P.DECISION_NAME[dec], self.hub.new_utt(self), Turn(-1), held=True)   # scratch turn until promoted
                    R.ctx_snapshot = ("fork", snap_fork)
                    sp.reply = R
                    R.task = asyncio.ensure_future(self._generate(R, list(sp.fork.ids)))
            if sp.promote_at is not None and (not sp.early or sp.confirmed):
                t, frame_ms = sp.promote_at
                if not self._spec_promote(t):
                    self._commit(t, frame_ms)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.send({"type": "error", "where": "speculation", "detail": repr(e)})
            if not sp.dead: self._spec_discard(time.monotonic(), "error")

    def _spec_discard(self, t, why, commit_pending=True):
        sp = self.mspec
        if sp is None:
            return
        self.mspec = None; sp.dead = True
        self.counters["mspec_discarded"] += 1
        if sp.task and not sp.task.done(): sp.task.cancel()
        if sp.reply is not None:
            self.counters["mspec_wasted_replies"] += 1
            self._kill_reply_gen(sp.reply)
        self.send({"type": "spec_discard", "t": self._rel(t), "why": why, "age_ms": round((t - sp.t) * 1000),
                   "decided": sp.rec["decision"] if sp.rec else None})
        self._line(f"(spec discarded: {why}{'' if not sp.rec else ', had decided ' + sp.rec['decision']})")
        if sp.held_live:
            self._push(sp.held_live)
        if commit_pending and sp.promote_at is not None:        # the rule already fired: commit the normal way
            self._commit(*sp.promote_at)

    def _spec_promote(self, t, early=False) -> bool:
        """the endpoint rule committed: adopt the fork if nothing reached the context since it was taken"""
        sp = self.mspec
        live = normalize(self.turn.text()) if self.turn else []
        if (self.busy or self.inq or len(self.ctx.ids) != sp.ctx_len or self.turn is None
                or len(live) > len(sp.norm) or live != sp.norm[:len(live)]):
            self._spec_discard(t, "mismatch", commit_pending=False)   # the caller commits
            return False
        self.mspec = None
        self.counters["mspec_promoted"] += 1
        if getattr(sp, "n_notes", 0):
            self.counters["notes_inserted"] += sp.n_notes; self.counters["replies_with_notes"] += 1
            self._line(f"    NOTES ({sp.n_notes}) inserted at the reply start (spec)")
        for w in sp.new_words:
            if not any(r["end"] == w["end"] and r["text"] == w["text"] for r in self.turn.words):
                self._word_rec(w, sp.t, peek=True)
        self._peek_consumed(sp.peek_ev)
        tr = self.turn
        tr.t_peek = sp.t; tr.peek_ms = sp.peek_ev.get("ms"); tr.peek_words = len(sp.new_words); tr.t_spec = sp.t
        for w in sp.new_words: self._line(f"user  {w['text']}  (peek)")
        self._line(f"<complete>  (speculation adopted, {round((t - sp.t) * 1000)} ms after it fired)")
        rec = dict(sp.rec, t=self._rel(sp.t), why="complete (spec)")
        self.calls_log.append(rec); self.counters["calls"] += 1; self.counters[rec["decision"]] += 1
        self.send({"type": "decision", **rec})
        self._line(f"    CALL {'complete(spec)':14s} -> {rec['decision']:12s} {rec['ms']:6.1f} ms")
        reason = "early+spec" if early else "complete+spec"
        self.send({"type": "endpoint", "turn": tr.id, "text": tr.text(), "t": self._rel(t), "reason": reason})
        self.ctx = sp.fork                                    # peek words + <complete> + the call (and model open)
        if sp.dec in (P.SPEAK, P.INTERRUPT):
            tr = self._close_user_turn(True); tr.ep_reason = reason; tr.t_spec = sp.t
            R = sp.reply
            scratch = R.turn
            for k in ("t_llm_sent", "t_first_token", "t_llm_done", "t_tts_in", "t_first_pcm", "n_tokens", "reply",
                      "search_query", "t_search_sent", "t_search_done", "search_n", "search_results"):
                setattr(tr, k, getattr(scratch, k))
            R.turn = tr; R.ctx_snapshot = self._snapshot_of(sp)
            self._adopt_reply(R)
        return True

    def _snapshot_of(self, sp: Spec):
        f = sp.reply.ctx_snapshot[1]                          # the fork before the call
        return (len(f.ids), len(f.targets), len(f.calls), len(f.spans), f.turn_has_content, f.reply_last)

    # ------------------------------------------------------------------ replies
    def _start_reply(self, kind, tr: Turn, snap):
        # a reply no longer clears result_unreported by itself: only one that MENTIONS the result counts as its
        # report (_result_mentioned, checked when the nudge fires): an unrelated reply would have hidden it.
        R = Reply(kind, self.hub.new_utt(self), tr)
        R.ctx_snapshot = snap
        self._adopt_reply(R)
        R.task = asyncio.ensure_future(self._generate(R, list(self.ctx.ids)))

    def _adopt_reply(self, R: Reply):
        """R becomes the live reply: UI, playback bookkeeping; a held (speculative) reply releases what it buffered"""
        if self.R is not None and not self.R.cut and self.R is not R:
            self._stop_reply(self.R, "replaced")
        self.R = R; self.overlap = None; self.onset_t = None; self.bc_cont_ms = None
        self.IJ = None                                        # the next interjection opens a new utt (after this reply's)
        R.start_ms = self.clock.audio_ms
        self.reply_turn = R.turn; self.utt = R.utt; self.speaking = True
        self.sil_idx = 0
        self.send({"type": "reply_start", "turn": R.turn.id, "utt": R.utt, "kind": R.kind})
        if R.held:
            R.held = False
            if R.gen_done: self._hist_reply(R)
            if R.held_deltas: self.send({"type": "reply_delta", "turn": R.turn.id, "text": "".join(R.held_deltas)})
            for pcm in R.held_pcm: self._send_pcm(R.utt, pcm)
            R.held_deltas, R.held_pcm = [], []
            if R.gen_done: self._maybe_finished(R)
        self._line(f"    ASSISTANT {R.kind} starts (utt {R.utt})")

    async def _generate(self, R: Reply, prompt: list[int]):
        R.turn.t_llm_sent = time.monotonic()
        try:
            for rnd in range(self.m.tool_rounds + 1):
                raw = ""; spoken_upto = 0; cut = -1
                st = self.llm.stream(self.http, prompt, self.m.reply_max_tokens, [P.SPEAK, TOOL_RESPONSE_OPEN],
                                     {int(k): v for k, v in BANNED_IN_REPLY.items()})
                async for d in st:
                    if R.turn.t_first_token is None: R.turn.t_first_token = time.monotonic()
                    R.turn.n_tokens += 1
                    raw += d
                    cut = raw.find("<|tool_call>")
                    safe = len(raw) if cut < 0 else cut
                    tail = raw[max(spoken_upto, safe - 12):safe]      # a "<|tool" still streaming stays back
                    if cut < 0 and "<" in tail:
                        safe = max(spoken_upto, safe - (len(tail) - tail.rfind("<")))
                    if safe > spoken_upto:
                        self._speak_text(R, raw[spoken_upto:safe]); spoken_upto = safe
                stop = st.stop
                calls = list(CALL_RE.finditer(raw))              # every call in this generation (Gemma may emit several)
                if cut < 0 and spoken_upto < len(raw):
                    self._speak_text(R, raw[spoken_upto:]); spoken_upto = len(raw)
                self._flush(R, final=True)
                R.raw += " "                                        # the next round's text starts a new word
                if stop == TOOL_RESPONSE_OPEN and calls and rnd < self.m.tool_rounds:
                    if rnd == 0 and self.m.tool_hold_ms > 0: await self._tool_hold(R)
                    results = ""
                    for k, m in enumerate(calls):                # run them in order; one response block per call, as
                        name = m.group(1); args = self._parse_args(m.group(2)) or {}     # Gemma's template renders them
                        result = await self._run_tool(R, name, args)
                        R.units.append(("tool", name, args, result))
                        results += P.tool_result_text(name, result, opened=(k == 0))
                    if R.held: R.held_deltas.append(" ")                # the subtitle's text resumes as a new word
                    else: self.send({"type": "reply_delta", "turn": R.turn.id, "text": " "})
                    prompt = prompt + P.encode(raw[:calls[-1].end()] + "<|tool_response>") + P.encode(results)
                    continue
                break
            tr = R.turn
            tr.t_llm_done = time.monotonic()
            R.gen_done = True
            if not R.held: self._hist_reply(R)
            R.text = " ".join(u[1] if u[0] == "w" else f"[{u[1]}]" for u in R.units)
            tr.reply = " ".join(u[1] for u in R.units if u[0] == "w")
            self._log({"type": "llm_done", "turn": tr.id, "utt": R.utt, "reply": R.text, "n_tokens": tr.n_tokens,
                       "ms": round((tr.t_llm_done - tr.t_llm_sent) * 1000), "held": R.held})
            if not R.held:
                self._line(f"    ASSISTANT ({R.kind}): {R.text[:300]}")
            self._maybe_finished(R)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.send({"type": "error", "where": "reply", "detail": repr(e)})
            R.gen_done = True
            self._maybe_finished(R)

    def _speak_text(self, R: Reply, text: str):
        """spoken text as it streams: words into the reply's units, deltas to the UI, clauses to TTS"""
        if not text:
            return
        R.raw += text
        done_words = R.raw.split()
        if not R.raw[-1:].isspace() and done_words:
            done_words = done_words[:-1]                          # the last word may still be growing
        for w in done_words[R.n_words():]:
            R.units.append(("w", w))
        if R.held: R.held_deltas.append(text)
        else: self.send({"type": "reply_delta", "turn": R.turn.id, "text": text})
        R.out_buf += text
        self._flush(R, final=False)

    def _flush(self, R: Reply, final: bool):
        if final:                                                 # the last word of the stretch is complete now
            ws = R.raw.split()
            have = R.n_words()
            for w in ws[have:]: R.units.append(("w", w))
        while True:
            seg = self._cut(R, final)
            if seg is None:
                return
            seg = strip_tags(seg)
            if not seg.strip():
                continue
            if R.turn.t_tts_in is None: R.turn.t_tts_in = time.monotonic()
            s = Segment(R.utt, R.n_seg, seg); R.n_seg += 1; R.segs.append(s)
            pc = self._phrases() if s.seg == 0 else None
            pcm = pc.get(seg) if pc is not None else None
            if pcm is not None:                                   # the first clause plays at once from the cache; the
                self.counters["phrase_cache_hits"] += 1           # rest of the reply goes to TTS as usual and queues behind it
                s.samples = len(pcm) // 2; s.t_first = time.monotonic(); s.done = True
                loop = asyncio.get_running_loop()
                loop.call_soon(self.on_tts_audio, s, pcm); loop.call_soon(self.on_tts_done, s)
                self._line(f"    (first clause {seg!r} from the phrase cache)")
                continue
            if pc is not None: self.counters["phrase_cache_misses"] += 1
            self.hub.worker.say(s)

    def _phrases(self):
        """the hub's phrase cache for its exact engine, voice and render settings (loaded once per process); None if off"""
        h = self.hub
        if not hasattr(h, "phrase_cache"):
            h.phrase_cache = None
            if h.engine in ("breeze", "pocket") and self.m.phrase_cache:
                try:
                    from .phrase_cache import PhraseCache, PocketPhraseCache
                    w = h.worker
                    h.phrase_cache = (PhraseCache(w.ref, w.seed, w.cfg_scale, w.instruction) if h.engine == "breeze"
                                      else PocketPhraseCache(w.voice, w.model)).load()
                    print(f"[phrase cache] {len(h.phrase_cache.pcm)} clips ({h.phrase_cache.dir.name})", flush=True)
                except Exception as e:
                    print(f"[phrase cache] off: {e!r}", flush=True)
        return h.phrase_cache if self.m.phrase_cache else None

    def _cut(self, R: Reply, final: bool):
        buf = R.out_buf
        if not buf.strip():
            R.out_buf = ""; return None
        if final:
            R.out_buf = ""; return buf.strip()
        first = R.n_seg == 0
        need = self.m.first_clause_chars if first else self.m.next_clause_chars
        seen = 0
        for i, ch in enumerate(buf):
            seen += 1
            if seen >= need and (ch in ".?!;:\n" or (ch == "," and first)):
                seg, R.out_buf = buf[:i + 1].strip(), buf[i + 1:]
                return seg or None
        if seen >= 160:
            sp = buf.rfind(" ", 0, 160)
            cut = sp if sp > 0 else 160
            seg, R.out_buf = buf[:cut].strip(), buf[cut:]
            return seg or None
        return None

    @staticmethod
    def _parse_args(s):
        """Gemma call arguments '{k:<|"|>v<|"|>,n:3}' -> dict (best effort)"""
        strs = []
        s2 = re.sub(r'<\|"\|>(.*?)<\|"\|>', lambda m: (strs.append(m.group(1)), f'"\x00{len(strs) - 1}"')[1], s, flags=re.S)
        s2 = re.sub(r'([{,]\s*)([A-Za-z_]\w*)\s*:', r'\1"\2":', s2)
        try:
            v = json.loads(s2, strict=False)                  # the \x00 placeholders are control characters
        except Exception:
            return None
        def fix(o):
            if isinstance(o, str) and o.startswith("\x00"): return strs[int(o[1:])]
            if isinstance(o, dict): return {k: fix(x) for k, x in o.items()}
            if isinstance(o, list): return [fix(x) for x in o]
            return o
        return fix(v)

    async def _tool_hold(self, R: Reply):
        """wait until the user has been quiet tool_hold_ms (turn-head frames, 80 ms each) before the first tool
        call of a reply; a reply cut meanwhile (undo / yield) is cancelled by its owner, so this only waits. Capped at 6 s."""
        t0 = time.monotonic()
        while (self.spk_frames > 0 or self.sil_frames * 80 < self.m.tool_hold_ms) and not R.cut:
            if time.monotonic() - t0 > 6.0: break
            await asyncio.sleep(0.02)
        ms = (time.monotonic() - t0) * 1000
        if ms > 30:
            self.counters["tool_hold_waits"] += 1
            self._line(f"    (tool call held {ms:.0f} ms until the user had been quiet {self.m.tool_hold_ms} ms)")

    async def _run_tool(self, R: Reply, name: str, args: dict):
        tr = R.turn
        if name in self.client_tool_names:
            while R.held:                                     # a speculative reply calls out only once adopted (a discard cancels this)
                await asyncio.sleep(0.02)
            t0 = time.monotonic(); self.counters["tools"] += 1
            result = await self.m.client_tool_runner(name, args)
            self.send({"type": "tool", "turn": tr.id, "name": name, "args": args, "result": result, "t": self._rel(), "ms": round((time.monotonic() - t0) * 1000), "client": True})
            self._line(f"    TOOL {name}{json.dumps(args)} -> {json.dumps(result)}  (client)")
            return result
        if name == RESET_TOOL and self.m.tools and self.m.reset_tool:
            return await self._reset_tool(R, args)
        if (name == CLAUDE_TOOL and self.claude is not None) or (name == FRONTIER_TOOL and self.frontier is not None) or \
                (name in ("task_status", "cancel_task") and (self.claude is not None or self.frontier is not None)):
            return await self._task_tool(R, name, args)
        if name in LOCAL_TOOLS and self.m.tools:
            if name in STATEFUL_TOOLS:                            # a speculative reply writes only once adopted
                while R.held:                                     # (a discard cancels this task here)
                    await asyncio.sleep(0.02)
            t0 = time.monotonic(); self.toolbox.http = self.http; self.counters["tools"] += 1
            result = await self.toolbox.call(name, args)
            tb = self.toolbox
            self.send({"type": "tool", "turn": tr.id, "name": name, "args": args, "result": result, "t": self._rel(),
                       "ms": round((time.monotonic() - t0) * 1000),
                       "store": {"notes": tb.notes, "lists": tb.lists, "todos": tb.todos, "counters": tb.counters}})
            self._line(f"    TOOL {name}{json.dumps(args)} -> {json.dumps(result)}")
            return result
        if name != "web_search" or self.cfg.search == "off":
            self._line(f"    TOOL {name}{args} -> unavailable")
            return {"status": "unavailable"}
        query = str(args.get("query") or "")
        tr.search_query = query; tr.t_search_sent = time.monotonic(); self.counters["searches"] += 1
        self.send({"type": "search", "turn": tr.id, "query": query, "t": self._rel()})
        await self._web_search(tr, query)                         # fills tr.search_results / search_n (cascade)
        tr.t_search_done = time.monotonic()
        self.send({"type": "search_done", "turn": tr.id, "n": tr.search_n, "results": tr.search_results,
                   "ms": round((tr.t_search_done - tr.t_search_sent) * 1000)})
        self._line(f"    TOOL web_search({query!r}) -> {tr.search_n} results")
        if not tr.search_results:
            return {"results": [], "note": "no results" if (tr.search_n or 0) >= 0 else "search failed"}
        return {"results": [{"title": x["title"], "snippet": x["snippet"]} for x in tr.search_results]}

    # ------------------------------------------------------------------ background tasks: claude_code, ask_frontier 
    def _registry(self, task_id):
        tid = str(task_id or "")
        return self.claude if tid.startswith("c") else self.frontier if tid.startswith("f") else None

    async def _task_tool(self, R: Reply, name: str, args: dict):
        while R.held:                                             # a speculative reply starts a task only once adopted
            await asyncio.sleep(0.02)
        t0 = time.monotonic()
        conv = "\n".join(self.hist[-30:])
        if name in (CLAUDE_TOOL, FRONTIER_TOOL):
            text = str(args.get("task") or args.get("question") or "").strip()
            if not text:
                result = {"error": "empty " + ("task" if name == CLAUDE_TOOL else "question")}
            elif name == CLAUDE_TOOL:
                result = {"task_id": self.claude.start(text, str(args.get("context") or ""), conv)}
            else:
                result = {"task_id": self.frontier.start(text, str(args.get("context") or ""), str(args.get("depth") or "standard"), conv)}
        elif name == "task_status":
            tid = args.get("task_id")
            reg = self._registry(tid) if tid else None
            if tid and reg is None:
                result = {"error": "not found"}
            elif tid:
                result = {"tasks": [x for x in self._all_tasks() if x["task_id"] == str(tid)]} or {"error": "not found"}
                if not result.get("tasks"): result = {"error": "not found"}
            else:
                result = {"tasks": self._all_tasks()}
        else:
            reg = self._registry(args.get("task_id"))
            result = reg.cancel(args.get("task_id")) if reg is not None else {"error": "not found"}
        self.send({"type": "tool", "turn": R.turn.id, "name": name, "args": args, "result": result, "t": self._rel(),
                   "ms": round((time.monotonic() - t0) * 1000), "store": {}})
        self._line(f"    TOOL {name}{json.dumps(args, ensure_ascii=False)} -> {json.dumps(result, ensure_ascii=False)}")
        return result

    def _all_tasks(self):
        out = self.claude.status().get("tasks", []) if self.claude is not None else []
        return out + (self.frontier.status_list() if self.frontier is not None else [])

    def _task_done(self, name, tid, result):
        reg = self.claude if name == CLAUDE_TOOL else self.frontier
        self.send({"type": "tool", "turn": None, "name": name, "args": {"task_id": tid}, "result": result, "t": self._rel(),
                   "ms": round(reg.tasks[tid].get("elapsed_s", 0) * 1000), "store": {}})
        self._line(f"    ({name} task {tid} finished: {json.dumps(result, ensure_ascii=False)[:300]})")
        self._push([("async", name, result)])

    _MENTION_STOP = set("that this with from have your about which there their were what when will would could should also than "
                        "then they them just like been into more most some only over such very each does here after before "
                        "because where while these those being answer result task status said says asked agent".split())

    def _result_mentioned(self, u) -> bool:
        """did a reply after this async result mention it? >= 3 shared content words (4+ letters) between the
        result and the assistant's lines since it arrived"""
        res = u["result"]
        if isinstance(res, dict): res = {k: v for k, v in res.items() if k != "task_id"}
        key = set(re.findall(r"[a-z]{4,}", json.dumps(res, ensure_ascii=False).lower())) - self._MENTION_STOP
        said = " ".join(l for l in self.hist[u["h"]:] if l.startswith("Assistant:")).lower()
        return len(key & set(re.findall(r"[a-z]{4,}", said))) >= 3

    def _force_speak(self):
        """the harness takes the turn right where the model's last call said listen: that call becomes speak. A
        listen is not saved in the ids, so the position is still free for the speak token; a second call there used to raise
        "call at N conflicts with an existing target" (the result nudge, live 10-05 11:29)."""
        c = self.ctx; pos = len(c.ids) - 1
        if c.calls and c.calls[-1][0] == pos and c.calls[-1][1] in (P.LISTEN, P.LISTEN_MUTED):
            c.calls.pop(); c.targets[pos] = -100
        c.call(P.SPEAK, "idle")

    def _flush_async(self) -> int:
        """the waiting async results into the context (the user stream, as training's async_result); returns how many"""
        n = len(self.async_pending)
        if not n: return 0
        if self.mspec is not None: self._spec_discard(time.monotonic(), "tool result")   # its fork lacks the result
        for _, name, result in self.async_pending:
            self.ctx.async_result(name, result)
            self._hist_add(f"Result of {name}: {json.dumps(result, ensure_ascii=False)[:600]}")
            self.unreported_results.append({"name": name, "result": result, "h": len(self.hist)})
            self._line(f"tool result {name} (async)")
        self.async_pending = []
        self.result_unreported = True                          # the nudge watches it; cleared when it fires
        return n

    # ------------------------------------------------------------------ reset_chat 
    async def _reset_tool(self, R: Reply, args: dict):
        """reset_chat, always confirmed by the harness: it resets only on a call with confirmed=true whose user turn says yes,
        after we asked (an earlier reset_chat call, or our previous reply asked about a reset). Anything else: nothing is
        reset and the result tells the model to ask. The reset itself happens when this reply ends."""
        while R.held:                                             # a speculative reply acts only once adopted
            await asyncio.sleep(0.02)
        t0 = time.monotonic()
        confirmed = str(args.get("confirmed", "")).strip().lower() in ("true", "1", "yes")
        instr = str(args.get("instructions") or "").strip()
        ask = self.reset_ask if (self.reset_ask and self.reset_ask["utt"] != R.utt and t0 - self.reset_ask["t"] < RESET_ASK_S) else None
        prev = next((h for h in reversed(self.hist[:-1]) if h.startswith("Assistant: ")), "") if self.hist else ""
        asked = ask is not None or ("?" in prev and bool(RESET_ASKED_RE.search(prev)))
        said = R.turn.text() if R.turn is not None else ""
        yes = bool(RESET_YES_RE.search(said)) and not RESET_NO_RE.search(said)
        if confirmed and asked and yes:
            self.reset_due = instr or (ask or {}).get("instructions", ""); self.reset_ask = None
            result = {"status": "done", "note": "the conversation is wiped when you finish this reply"}
        else:
            self.reset_ask = {"t": t0, "utt": R.utt, "instructions": instr or (ask or {}).get("instructions", "")}
            result = {"status": "needs_confirmation",
                      "note": "nothing was reset; ask the user whether to wipe the conversation and start over, and call "
                              "again with confirmed=true once they say yes"}
        self.send({"type": "tool", "turn": R.turn.id, "name": RESET_TOOL, "args": args, "result": result, "t": self._rel(),
                   "ms": round((time.monotonic() - t0) * 1000), "store": {}})
        self._line(f"    TOOL {RESET_TOOL}{json.dumps(args)} -> {json.dumps(result)}  (asked={asked}, user said {said!r})")
        return result

    def _do_reset(self):
        """wipe the conversation: a fresh context (system prompt + the standing instructions + the tools), empty thinker
        history; the tool store (notes, lists, todos) stays. Words of a user turn already in progress carry over."""
        rules, self.reset_due = self.reset_due or "", None
        self.n_resets += 1
        if self.log_dir:
            try:
                with open(f"{self.log_dir}/context_before_reset_{self.n_resets}.txt", "w") as f: f.write(self.ctx.render())
            except Exception:
                pass
        if self.mspec is not None: self._spec_discard(time.monotonic(), "chat reset")
        for th in self.think.values():
            if th.get("task") and not th["task"].done(): th["task"].cancel()
        self.think = {}; self.hist = []; self.__dict__.pop("hist_keys", None)
        self.overlap = None; self.IJ = None; self.reset_ask = None
        self.user_rules = rules
        if self.claude is not None: self.claude.reset()          # the next Claude task starts a new Claude Code session
        self.system = self._system_text()
        self.ctx = P.Context(); self.ctx.start(self.system, self.decls)
        if self.turn is not None:
            for w in self.turn.words: self.ctx.word(w["text"])
        self._line(f"(chat reset #{self.n_resets}: context wiped" + (f"; standing instructions: {rules!r})" if rules else ")"))
        self.send({"type": "reset", "n": self.n_resets, "instructions": rules})
        self.send({"type": "brain", "brain": "speakrail", "system": self.system, "bias": self.m.bias,
                   "end_rule": {"tau": self.m.tau, "k": self.m.k}})
        async def warm():
            try: await self.llm.warm(self.http, self.ctx.ids)
            except Exception as e: self.send({"type": "error", "where": "llm", "detail": repr(e)})
        asyncio.ensure_future(warm())

    # ------------------------------------------------------------------ TTS / playback
    def on_tts_audio(self, seg, pcm):
        R = self._reply_of(seg.utt)
        if R is None or R.cut or self.closed:
            return
        if R.turn.t_first_pcm is None: R.turn.t_first_pcm = time.monotonic()
        if R.held:
            R.held_pcm.append(pcm); return
        self._send_pcm(seg.utt, pcm)

    def on_tts_done(self, seg):
        self._log({"type": "tts_seg", "utt": seg.utt, "seg": seg.seg, "text": seg.text, "samples": seg.samples})
        R = self._reply_of(seg.utt)
        if R is not None and R is not self.IJ: self._maybe_finished(R)

    def _reply_of(self, utt):
        if self.R is not None and self.R.utt == utt: return self.R
        if self.IJ is not None and self.IJ.utt == utt: return self.IJ
        sp = self.mspec
        if sp is not None and sp.reply is not None and sp.reply.utt == utt: return sp.reply
        return None

    def note_play_start(self, utt, delay_ms):
        self._log({"type": "play_start", "utt": utt, "delay_ms": delay_ms})
        t_play = time.monotonic() + delay_ms / 1000
        if self.tape: self.tape.play_start(utt, t_play)
        R = self.R
        if R is not None and R.utt == utt and R.t_play is None:
            R.t_play = t_play; R.play_ms = self.clock.ms_at(t_play)
            if R.turn.t_play is None: R.turn.t_play = t_play; self._log_turn(R.turn)
            self._maybe_finished(R)

    def _maybe_finished(self, R: Reply):
        """generation and TTS done: tell the UI, and schedule the end of playback (the reply_end input)"""
        if R.held or R.cut or not R.gen_done or any(not s.done for s in R.segs):
            return
        if not getattr(R, "ui_ended", False):
            R.ui_ended = True
            self.speaking = False
            self.send({"type": "reply_end", "turn": R.turn.id})
            self._log_turn(R.turn, again=True)
        if R.end_handle is not None:
            return
        if not R.segs:                                            # nothing was said (a bare tool call, an empty reply)
            R.end_handle = True; self._push([("reply_end", R)]); return
        if R.t_play is None:
            return                                               # note_play_start will call again
        delay = max(0.0, R.t_play + R.audio_s() - time.monotonic())
        R.end_handle = asyncio.get_running_loop().call_later(delay, lambda: self._push([("reply_end", R)]))

    async def _reply_end(self, R: Reply):
        if R is not self.R or R.cut:
            return
        if self.reset_due is not None:                            # reset_chat was confirmed in this reply
            self.R = None; self.overlap = None; self._after_speech(); self._unduck()
            self._do_reset(); return
        if self.overlap is not None:                              # the model kept listening and the audio played out
            self.ctx.resume_reply(); self._emit(R, 10 ** 9); self.ctx.end_reply(); self.overlap = None
            self._line("(reply finished playing while the overlap turn was open: rest saved after it)")
            self.R = None; self._after_speech(); return
        self._emit(R, 10 ** 9); self.ctx.end_reply()
        self._line("(reply ends)")
        self.R = None; self._after_speech()
        self._unduck()
        n = self._flush_async()                                   # results that arrived while we talked
        await self._idle_call("tool" if n else "after_reply")

    def _after_speech(self):
        self.speaking = False; self.onset_t = None
        self.sil_anchor_ms = self.clock.audio_ms; self.sil_idx = 0

    def _emit(self, R: Reply, upto_words: int):
        """save R's units into the context up to spoken word `upto_words` (tool units right after included)"""
        buf = []
        def flush():
            if buf: self.ctx.reply_text(" ".join(buf), loss=False); buf.clear()
        while R.emitted_u < len(R.units):
            u = R.units[R.emitted_u]
            if u[0] == "w":
                if R.emitted_w >= upto_words: break
                buf.append(u[1]); R.emitted_w += 1
            else:
                flush(); self.ctx.reply_tool(u[1], u[2], u[3], loss=False)
            R.emitted_u += 1
        flush()

    def _kill_reply_gen(self, R: Reply):
        if R.task and not R.task.done(): R.task.cancel()
        self.hub.worker.kill(R.utt)
        if R.end_handle not in (None, True): R.end_handle.cancel()

    def _stop_reply(self, R: Reply, reason: str):
        R.cut = True
        self._kill_reply_gen(R)
        if self.tape: self.tape.stop(R.utt, time.monotonic())
        R.turn.cut = R.turn.cut or reason
        self.send({"type": "stop_audio", "reason": reason, "turn": R.turn.id})
        self._log_turn(R.turn, again=True)
        if self.R is R:
            self.R = None; self.utt = None; self._after_speech()
        self._unduck()

    # ------------------------------------------------------------------ overlap (the user is heard while we speak)
    async def _overlap(self, R: Reply, items) -> bool:
        """True if handled here; False -> the items go through the idle path (after an early-start undo)"""
        words = [x[1] for x in items if x[0] == "word"]
        text = ((self.overlap or {}).get("text", "") + " " + " ".join(words)).strip()
        late = [x for x in items if x[0] == "word" and x[2]["start"] > R.start_ms]      # started after our reply did
        content = bool(words) and not is_backchannel(text) and (bool(late) or R.kind == "speak")
        onset = self.onset_t or time.monotonic()
        played = max(R.played_words(onset), R.min_pin)
        if content and self.overlap is None and R.kind == "speak" and played <= self.m.undo_words:
            # early start: the user was not done. Undo the reply as if it never started; the words join the turn.
            self.counters["undo"] += 1
            self._stop_reply(R, "undo (early start)")
            self._rollback(R.ctx_snapshot)
            self._line(f"(early start undone: {played} reply words had played; context rolled back)")
            tr = R.turn; tr.cut = "undo"
            self.turn = self.turn or tr                          # the same user turn goes on
            return False
        if self.overlap is None:
            if R.n_words() == 0 and not R.gen_done:           # nothing said yet: no empty model turn, wait for a word
                self._hold(R, items, "the reply has no words yet"); return True
            k = max(1, played, R.emitted_w + 1)               # an earlier overlap already split here: at least one more word
            if k > R.n_words() and R.gen_done:
                self._line("(overlap at the very end of the reply: left to the reply end)")
                return True
            self._emit(R, k); self.ctx.cut_reply()
            self.overlap = {"t": onset, "k": k, "text": ""}; self.counters["overlaps"] += 1
            self._line(f"(overlap turn pinned after reply word {k} of {R.n_words()})")
        self.overlap["text"] = text
        ws = [x for x in items if x[0] == "word"]
        if ws:
            if self.overlap.get("c0") is None and not is_backchannel(text): self.overlap["c0"] = ws[0][2]["start"]
            self.overlap["c1"] = max(self.overlap.get("c1", 0), max(x[2]["end"] for x in ws))
        for it in items: self._append_input(it)
        if content and self.cfg.barge_in in ("word", "head"):
            dec = P.YIELD; self.counters["harness_yield"] += 1
            self._line(f"    RULE barge-in on {text!r} -> yield")
            self.send({"type": "decision", "t": self._rel(), "why": "barge rule", "set": "overlap", "decision": "yield", "probs": {}, "ms": 0})
        else:
            dec = await self._decide("overlap", "overlap")
            if (dec == P.LISTEN and self.m.safety_yield_s and R.kind != "interrupt" and self.overlap.get("c0") is not None
                    and self.overlap.get("c1", 0) - self.overlap["c0"] >= self.m.safety_yield_s * 1000):
                dec = P.YIELD; self.counters["safety_yield"] = self.counters.get("safety_yield", 0) + 1
                self._line(f"    SAFETY NET: {(self.overlap['c1'] - self.overlap['c0']) / 1000:.1f} s of speech over us and the model "
                           f"still listens -> yield")
                self.send({"type": "decision", "t": self._rel(), "why": "safety net", "set": "overlap", "decision": "yield", "probs": {}, "ms": 0})
        self.ctx.call(dec, "overlap")
        if dec == P.CONTINUE:
            self.ctx.resume_reply(); self.overlap = None; self.onset_t = None
            self.bc_cont_ms = self.clock.audio_ms
            if not words or is_backchannel(text):
                self.counters["backchannels"] += 1
                self.send({"type": "backchannel", "turn": R.turn.id, "text": text, "t": self._rel(), "reason": "continue"})
            self._unduck()
        elif dec == P.YIELD:
            self._line(f"(yield: reply cut after word {R.emitted_w})")
            self._stop_reply(R, "yield"); self.overlap = None
            self.turn = self.turn or Turn(self.n_turns)
        return True

    def _unduck(self):
        if self.ducked or self.cfg.barge_in in ("head", "duck"):
            self.ducked = False; self.send({"type": "duck", "gain": 1.0})

    def _drop_stale(self, R: Reply, items):
        """interrupt: what the user said before they could hear us is the tail of the turn we cut into. As in the tape
        builder it never reaches the context, nor does a <complete> / <user_bc> of it."""
        edge = None if R.play_ms is None else R.play_ms + self.m.react_ms
        keep, drop = [], []
        for x in items:
            if x[0] == "word":
                stale = edge is None or x[2]["start"] < edge
                if not stale: R.fresh_words += 1
            elif x[0] == "user_bc":
                stale = edge is None or (len(x) > 1 and x[1] < edge)
            elif x[0] == "complete":
                stale = R.fresh_words == 0
            else:
                stale = False
            (drop if stale else keep).append(x)
        if drop:
            self.counters["stale_dropped"] += len(drop)
            self._line("(said before our interrupt could be heard, dropped: "
                       + " ".join(x[1] if x[0] == "word" else f"<{x[0]}>" for x in drop) + ")")
            if self.onset_t is not None and (R.t_play is None or self.onset_t < R.t_play + self.m.react_ms / 1000):
                self.onset_t = None                               # that speech is not an overlap onset
        return keep

    def _interrupt_holds(self, R: Reply | None) -> bool:
        """an interrupt keeps the floor until its first sentence has played (hold_max_s of playback at most)"""
        if R is None or R.cut or R.kind != "interrupt" or not self.m.interrupt_hold:
            return False
        now = time.monotonic()
        if now - R.t0 > self.m.hold_max_s + 2.0 or (R.t_play is not None and now - R.t_play >= self.m.hold_max_s):
            return False
        if R.t_play is None:
            return True
        n = R.first_sentence_words()
        return n is None or R.played_words(now) < n

    def _hold(self, R: Reply, items, why):
        """overlap input waits, in order, until `why` is over; then it goes through _process again"""
        if not self.ov_hold:
            self.counters["held_overlaps"] += 1
            self._line(f"(overlap held {why})")
        self.ov_hold.extend(items)
        if self.ov_hold_task is None or self.ov_hold_task.done():
            self.ov_hold_task = asyncio.ensure_future(self._hold_release(R))

    async def _hold_release(self, R: Reply):
        while R is self.R and not R.cut and (self._interrupt_holds(R) or (R.n_words() == 0 and not R.gen_done)):
            await asyncio.sleep(0.04)
        items, self.ov_hold = self.ov_hold, []
        live = R is self.R and not R.cut
        if live: R.min_pin = R.played_words(time.monotonic())
        if items and not self.closed:
            self._line(f"(held input released {'after reply word ' + str(R.min_pin) if live else 'after the reply'}: "
                       + " ".join(x[1] if x[0] == "word" else f"<{x[0]}>" for x in items) + ")")
            self.inq[:0] = items; self.inq_ev.set()

    def _cancel_reply(self, reason: str):
        """the UI's interrupt button / session close: cut the live reply where it is (as a yield)"""
        R = self.R
        if R is None or R.cut:
            return
        if reason != "session closed" and self.overlap is None:
            k = max(1, R.played_words(time.monotonic())) if R.segs else 0
            self._emit(R, min(k, R.n_words()) if R.n_words() else k); self.ctx.cut_reply()
            self._line(f"(cut by {reason} after reply word {R.emitted_w})")
        self._stop_reply(R, reason)
