#!/usr/bin/env python3
"""Offline test of session.MicroSession's control flow: no GPU, no ASR, no TTS.

A mock vLLM (aiohttp on 127.0.0.1:18001) answers decision calls with scripted rules (speak after <complete>, continue after
<user_bc>, yield on overlap words, listen otherwise) and streams a canned reply; a fake TTS turns each clause into silence
(60 ms per character); a fake ASR delivers words, head frames and peeks in real time; a fake browser reports play_start.
Scenarios: a turn taken through speculation, a barge-in (model yield and harness rule), a backchannel (continue, resumed
reply saved at its end), an early start (undo + context rollback), the silence ladder after a reply.

    python tests/test_session.py
"""
import asyncio, json, os, sys, tempfile, time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "src")]
import aiohttp
from aiohttp import web
from speakrail.harness import protocol as P
from speakrail.harness.cascade import Cfg
from speakrail.harness import session as MS

PORT = 18001
DECIDE_DELAY = [0.03]                                      # seconds per decision on a <complete> (a scenario slows it)
REPLY = "Sure, it is fourteen degrees and cloudy in Berlin right now."
INTERRUPT_REPLY = "Sorry, it's ten thousand. You said ten before, so which one is it?"


# ------------------------------------------------------------------ mock vLLM
async def completions(request):
    b = await request.json()
    ids = b["prompt"]
    if b.get("stream"):
        resp = web.StreamResponse(); await resp.prepare(request)
        tail = P.decode(ids[-60:])
        if "slowly" in tail: await asyncio.sleep(0.6)
        near = P.decode(ids[-14:])                         #reset_chat: only the current user turn decides the call
        if "<|turn>model" in near and ("Claude" in near or "bigger" in near):   #claude_code / ask_frontier
            call = ('<|tool_call>call:claude_code{task:<|"|>check the disk<|"|>}<tool_call|>' if "Claude" in near else
                    '<|tool_call>call:ask_frontier{depth:<|"|>quick<|"|>,question:<|"|>is a pension worth it<|"|>}<tool_call|>')
            for i in range(0, len(call), 9):
                await resp.write(f"data: {json.dumps({'choices': [{'text': call[i:i + 9], 'finish_reason': None}]})}\n\n".encode())
            await resp.write(f"data: {json.dumps({'choices': [{'text': '', 'finish_reason': 'stop', 'stop_reason': 50}]})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n"); return resp
        if "<|turn>model" in near and ("Yes." in near or "eset" in near):
            call = ('<|tool_call>call:reset_chat{confirmed:true,instructions:<|"|>Be brief.<|"|>}<tool_call|>' if "Yes." in near
                    else '<|tool_call>call:reset_chat{confirmed:true}<tool_call|>' if "Reset now." in near   # skips the question
                    else '<|tool_call>call:reset_chat{}<tool_call|>')
            for i in range(0, len(call), 9):
                await resp.write(f"data: {json.dumps({'choices': [{'text': call[i:i + 9], 'finish_reason': None}]})}\n\n".encode())
            await resp.write(f"data: {json.dumps({'choices': [{'text': '', 'finish_reason': 'stop', 'stop_reason': 50}]})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n"); return resp
        if ("dice" in tail or "note" in tail) and "<|turn>model" in tail[-20:]:   # a tool call, then the reply after its result
            call = ('<|tool_call>call:dice_roll{expr:<|"|>1d6<|"|>}<tool_call|><|tool_call>call:record_note{text:<|"|>rolled<|"|>}<tool_call|>'
                    if "and note" in tail                                        # two calls in ONE message
                    else '<|tool_call>call:dice_roll{expr:<|"|>2d6<|"|>}<tool_call|>' if "dice" in tail
                    else '<|tool_call>call:record_note{text:<|"|>buy milk<|"|>}<tool_call|>')
            for i in range(0, len(call), 9):
                await resp.write(f"data: {json.dumps({'choices': [{'text': call[i:i + 9], 'finish_reason': None}]})}\n\n".encode())
            await resp.write(f"data: {json.dumps({'choices': [{'text': '', 'finish_reason': 'stop', 'stop_reason': 50}]})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n"); return resp
        words = ("The bigger model says yes." if "response:ask_frontier{answer" in tail
                 else "Asking the bigger model." if "response:ask_frontier{task_id" in tail
                 else "Claude says the disk is fine." if "response:claude_code{answer" in tail
                 else "Asking Claude." if "response:claude_code{task_id" in tail
                 else "Should I wipe our chat and start over?" if "needs_confirmation" in tail
                 else "Done, starting fresh." if "response:reset_chat" in tail
                 else f"You rolled {tail.split('total:')[1].split('}')[0]}." if "response:dice_roll" in tail
                 else "Saved." if "response:record_note" in tail
                 else INTERRUPT_REPLY if "<interrupt>" in tail[-40:] else REPLY).split(" ")
        for w in words:
            chunk = {"choices": [{"text": w + " ", "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode()); await asyncio.sleep(0.01)
        await resp.write(f"data: {json.dumps({'choices': [{'text': '', 'finish_reason': 'stop', 'stop_reason': P.SPEAK}]})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n"); return resp
    allowed = b.get("allowed_token_ids")
    if not allowed:
        return web.json_response({"choices": [{"text": "", "logprobs": None}]})
    last = ids[-1]
    if P.SPEAK in allowed:                                 # idle set
        dec = (P.SPEAK if last == P.IDS["<complete>"] or P.decode(ids[-2:]).endswith("<tool_response|>") else P.INTERRUPT if P.decode(ids[-6:]).rstrip().endswith("8,000")
               else P.LISTEN)
        if last == P.IDS["<complete>"] and "maybe" in P.decode(ids[-8:]): dec = P.LISTEN      #a spec that decides listen
    else:                                                  # overlap set
        dec = P.CONTINUE if last == P.IDS["<user_bc>"] or P.decode([last]).strip().lower().strip(".,") in ("mm", "mmhmm", "mm-hmm", "yeah") else P.YIELD
        if "electrical" in P.decode(ids[-30:]).lower(): dec = P.LISTEN          #a model that never yields: the safety net must
    await asyncio.sleep(DECIDE_DELAY[0] if last == P.IDS["<complete>"] else 0.03)
    top = {f"token_id:{i}": (-0.01 if i == dec else -5.0) for i in allowed}
    return web.json_response({"choices": [{"text": "", "logprobs": {"tokens": [f"token_id:{dec}"], "top_logprobs": [top]}}]})


async def fireworks(request):                             #mock GLM for ask_frontier
    b = await request.json()
    ok = b["model"].endswith("glm-5p3-flash") and request.headers.get("Authorization") == "Bearer test-fw" and "pension" in b["messages"][1]["content"]
    await asyncio.sleep(0.8)
    return web.json_response({"choices": [{"message": {"content": ("Yes, for most people a pension is worth it, mainly for the "
                              "employer match.\nConfidence: high") if ok else "bad request"}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 900, "completion_tokens": 300}})


# ------------------------------------------------------------------ fake TTS / hub
class FakeWorker:
    def __init__(self, hub): self.hub = hub; self.killed = set(); self.stats = {"segments": 0, "ttfa_ms": [], "rtf": [], "silent_lead": []}; self.device = "fake"
    def say(self, seg):
        self.said = getattr(self, "said", []) + [seg.text]
        async def run():
            await asyncio.sleep(0.05)
            if seg.utt in self.killed: return
            n = int(0.06 * len(seg.text) * MS.TTS_RATE)
            seg.samples = n; self.hub._on_audio(seg, b"\x00\x00" * n)
            seg.done = True; self.stats["segments"] += 1; self.hub._on_done(seg)
        asyncio.get_running_loop().create_task(run())
    def kill(self, utt): self.killed.add(utt)


class FakeHub:
    engine = "breeze"
    def __init__(self): self.owners = {}; self.next_utt = 1; self.worker = FakeWorker(self)
    def new_utt(self, s): u = self.next_utt; self.next_utt += 1; self.owners[u] = s; return u
    def idle(self): return True
    def _on_audio(self, seg, pcm):
        s = self.owners.get(seg.utt)
        if s: s.on_tts_audio(seg, pcm)
    def _on_done(self, seg):
        s = self.owners.get(seg.utt)
        if s: s.on_tts_done(seg)


class FakeASR:
    """collects peek requests; answers them 65 ms later with the words the test says are still in the delay line"""
    def __init__(self): self.sess = None; self.pending_words = []; self.peeks = []
    async def send(self, msg):
        m = json.loads(msg)
        if m.get("type") == "peek":
            self.peeks.append(m["id"])
            async def answer():
                await asyncio.sleep(0.065)
                words, self.pending_words = self.pending_words, []
                self.sess._on_peek({"type": "transcript.peek", "id": m["id"], "ms": 40, "words": words,
                                    "text": " ".join(w["text"] for w in words)}, time.monotonic())
            asyncio.get_running_loop().create_task(answer())


# ------------------------------------------------------------------ driver
class Driver:
    def __init__(self, barge="off", **mkw):
        self.events = []; self.played = set()
        cfg = Cfg(delay_ms=480, barge_in=barge, peek=True, spec=True, spec_tau=0.5, search="off")
        self.dir = tempfile.mkdtemp(prefix="mt_test_")
        # the control-flow scenarios below are timed for the end rule: P >= 0.9 on 3 frames in a row
        mc = MS.MicroCfg(url=f"http://127.0.0.1:{PORT}/v1/completions", bias={}, **mkw)
        self.s = MS.MicroSession(cfg, FakeHub(), self.send, log_dir=self.dir, mcfg=mc)
        self.asr = FakeASR(); self.asr.sess = self.s; self.s.asr_ws = self.asr
        self.audio_ms = 0.0; self.frame = 0

    def send(self, obj):
        if isinstance(obj, bytes):
            utt = int.from_bytes(obj[:2], "little")
            if utt not in self.played:                       # the browser starts playing 20 ms later
                self.played.add(utt); asyncio.get_running_loop().call_soon(self.s.note_play_start, utt, 20)
            return
        self.events.append(obj)

    async def start(self):
        self.s.http = aiohttp.ClientSession()
        self.s.decider = asyncio.create_task(self.s._decide_loop())

    async def frames(self, n, p):
        """n head frames (80 ms each, real time) with probabilities p = [speaking, complete, incomplete, bc, wait]"""
        for _ in range(n):
            self.audio_ms += 80; self.frame += 80
            self.s.clock.push(1280, time.monotonic())
            self.s._on_head({"type": "turn", "t": self.frame, "p": p}, time.monotonic(), 0.0)
            await asyncio.sleep(0.08)

    def words(self, *ws, start=None):
        out = []
        for w in ws:
            st = self.audio_ms - 300 if start is None else start
            out.append({"text": w, "start": st, "end": st + 200})
        self.s._on_words({"type": "transcript", "text": " ".join(ws), "words": out}, time.monotonic())

    def decisions(self): return [(e["why"], e["decision"]) for e in self.events if e.get("type") == "decision"]
    def types(self, k): return [e for e in self.events if e.get("type") == k]


SPEAK = [0.9, 0.02, 0.05, 0.01, 0.02]; SIL_INC = [0.02, 0.3, 0.6, 0.02, 0.06]; SIL_HALF = [0.02, 0.6, 0.3, 0.02, 0.06]
SIL_DONE = [0.02, 0.95, 0.02, 0.01, 0.0]; BC = [0.3, 0.02, 0.02, 0.97, 0.0]


async def turn(d, *words, peek_tail=()):
    await d.frames(4, SPEAK); d.words(*words)
    d.asr.pending_words = [{"text": w, "start": d.audio_ms, "end": d.audio_ms + 200} for w in peek_tail]
    await d.frames(1, SIL_HALF); await d.frames(4, SIL_DONE)


async def scenario_spec_turn():
    d = Driver(); await d.start()
    await turn(d, "What's", "the", "weather", peek_tail=("in", "Berlin?"))
    await asyncio.sleep(0.4)
    await d.frames(60, [0.02, 0.1, 0.1, 0.02, 0.76])          # reply plays out, then the ladder
    await asyncio.sleep(0.3)
    ctx = d.s.ctx.render()
    ok = (d.s.counters["mspec_promoted"] == 1 and ("complete (spec)", "speak") in d.decisions()
          and ("after_reply", "listen") in d.decisions() and "Berlin?<|complete|>" not in ctx and "<sil:1s>" in ctx
          and REPLY.split()[0] in ctx and ctx.count("<|turn>model") == 1)
    await d.s.http.close()
    return "spec turn", ok, d


async def scenario_phrase_cache():
    """the reply's first clause ("Sure,") is in the phrase cache: its clip is played without a TTS request, the rest
    of the reply goes to TTS, the reply plays out and ends normally"""
    class Stub:
        pcm = {"Sure,": b"\x00\x00" * int(0.4 * MS.TTS_RATE)}
        def get(self, seg): return self.pcm.get(" ".join(seg.split()))
    d = Driver(); d.s.hub.phrase_cache = Stub(); await d.start()
    await turn(d, "What's", "the", "weather", peek_tail=("in", "Berlin?"))
    await asyncio.sleep(0.4)
    await d.frames(60, [0.02, 0.1, 0.1, 0.02, 0.76])
    await asyncio.sleep(0.3)
    said = getattr(d.s.hub.worker, "said", [])
    ok = (d.s.counters["phrase_cache_hits"] == 1 and "Sure," not in said and any("fourteen" in x for x in said)
          and ("after_reply", "listen") in d.decisions() and REPLY in " ".join(d.s.ctx.render().split()))
    await d.s.http.close()
    return "phrase cache: first clause from a clip", ok, d


async def scenario_safety_net():
    """the user talks over a reply and the model keeps listening: after 1.5 s of non-backchannel speech over us the
    harness yields (a live miss: the model read a long answer to the end while the user tried to cut in)"""
    d = Driver(); await d.start()
    await turn(d, "Tell", "me", "about", "Berlin.")
    await asyncio.sleep(1.2)
    t0 = d.audio_ms
    for k, w in enumerate(["Electrical,", "electrical", "electrical", "electrical."]):   # the mock listens on every one
        await d.frames(3, SPEAK); d.words(w, start=t0 + k * 600); await asyncio.sleep(0.15)
    await d.frames(3, SPEAK); await asyncio.sleep(0.3)
    stops = [e["reason"] for e in d.types("stop_audio")]
    ok = d.s.counters.get("safety_yield") == 1 and "yield" in stops and d.s.R is None
    await d.s.http.close()
    return "safety net: model keeps listening -> harness yields", ok, d


async def scenario_barge(barge):
    d = Driver(barge); await d.start()
    await turn(d, "Tell", "me", "about", "Berlin.")
    await asyncio.sleep(1.2)                                   # a few reply words play
    await d.frames(3, SPEAK); d.words("Wait,", "stop."); await d.frames(3, SPEAK); await asyncio.sleep(0.3)
    stops = [e["reason"] for e in d.types("stop_audio")]
    want = "yield"
    ok = want in stops and d.s.R is None and "<turn|>" in d.s.ctx.render()
    if barge == "word": ok = ok and d.s.counters["harness_yield"] == 1
    else: ok = ok and ("overlap", "yield") in d.decisions()
    await d.s.http.close()
    return f"barge ({barge})", ok, d


async def scenario_backchannel():
    d = Driver(); await d.start()
    await turn(d, "Tell", "me", "about", "Berlin.")
    await asyncio.sleep(1.0)
    await d.frames(2, BC); d.words("Mm-hmm."); await d.frames(20, [0.02, 0.1, 0.1, 0.05, 0.73])
    await asyncio.sleep(3.5); await d.frames(5, [0.02, 0.1, 0.1, 0.02, 0.76])
    ctx = d.s.ctx.render()
    ok = ("overlap", "continue") in d.decisions() and not d.types("stop_audio") and d.s.R is None and ctx.count("<|turn>model") == 2
    await d.s.http.close()
    return "backchannel", ok, d


async def scenario_early_start():
    d = Driver(); await d.start()
    await turn(d, "I", "need")
    await asyncio.sleep(0.15)                                  # we start speaking...
    n_before = d.s.ctx.render().count("<|turn>model")
    await d.frames(2, SPEAK); d.words("a", "flight", "to", "Rome."); await asyncio.sleep(0.3)
    ctx = d.s.ctx.render()
    ok = d.s.counters["undo"] == 1 and ctx.count("<|turn>model") == 0 and "<complete> a flight to Rome." in ctx
    await d.s.http.close()
    return f"early start (model turns before undo: {n_before})", ok, d


async def scenario_tool():
    d = Driver(); await d.start()
    await turn(d, "Roll", "two", "dice.")
    await asyncio.sleep(2.0); await d.frames(10, [0.02, 0.1, 0.1, 0.02, 0.76]); await asyncio.sleep(0.3)
    ctx = d.s.ctx.render(); tools = d.types("tool")
    ok = (len(tools) == 1 and tools[0]["name"] == "dice_roll" and len(tools[0]["result"]["dice"]) == 2
          and "call:dice_roll" in ctx and "response:dice_roll{dice:[" in ctx and "You rolled" in ctx and d.s.R is None
          and ctx.count("call:dice_roll") == 1)              # one call, no repeat
    await d.s.http.close()
    return "tool call (dice_roll)", ok, d


async def scenario_two_calls():
    """two calls in one generation: both run (the greedy regex used to swallow the second into the first's arguments)"""
    d = Driver(); await d.start()
    await turn(d, "Roll", "dice", "and", "note.")
    await asyncio.sleep(2.0); await d.frames(10, [0.02, 0.1, 0.1, 0.02, 0.76]); await asyncio.sleep(0.3)
    ctx = d.s.ctx.render(); tools = d.types("tool")
    ok = ([t["name"] for t in tools] == ["dice_roll", "record_note"] and len(tools[0]["result"]["dice"]) == 1
          and d.s.toolbox.notes == ["rolled"] and ctx.count("response:dice_roll{") == 1 and ctx.count("response:record_note{") == 1
          and ctx.count("<|tool_response>") == 2)
    await d.s.http.close()
    return "two tool calls in one message", ok, d


async def scenario_reset():
    """reset_chat: a confirmed call without asking first is refused; a bare call asks; the confirmed call after the
    user's yes wipes the context when its reply ends, with the standing instructions in the new system prompt"""
    d = Driver(); await d.start()
    idle = [0.02, 0.1, 0.1, 0.02, 0.76]
    for ws in (("Reset", "now."), ("Please", "reset", "the", "chat."), ("Yes.",)):
        await turn(d, *ws)
        await asyncio.sleep(2.5); await d.frames(10, idle); await asyncio.sleep(0.3)
    st = [t["result"]["status"] for t in d.types("tool")]
    ctx = d.s.ctx.render(); before = Path(d.dir, "context_before_reset_1.txt")
    ok = (st == ["needs_confirmation", "needs_confirmation", "done"] and len(d.types("reset")) == 1 and d.s.n_resets == 1
          and "standing instructions: Be brief." in ctx and "reset" not in ctx.split("<|turn>user", 1)[-1].lower()
          and "Done, starting fresh." not in ctx and d.s.hist == [] and d.s.R is None
          and before.exists() and "Please reset the chat." in before.read_text())
    await d.s.http.close()
    return "reset_chat (confirmed by the harness)", ok, d


async def scenario_peek_twice():
    """the rule fires again just after our reply starts and its peek returns the word the first peek already gave
    (seen live: the same word twice -> early-start undo -> a doubled word and <complete>): no new word, no undo"""
    d = Driver(); await d.start()
    await d.frames(4, SPEAK); d.words("Can", "you", "Google")
    tail = [{"text": "it?", "start": d.audio_ms, "end": d.audio_ms + 200}]
    d.asr.pending_words = [dict(w) for w in tail]
    await d.frames(1, SIL_HALF); await d.frames(4, SIL_DONE); await asyncio.sleep(0.1)
    d.asr.pending_words = [dict(w) for w in tail]             # the delay line still holds it: the next peek returns it again
    d.s.rule_open = True                                      # live: "Google" arrived after the first fire and re-opened the rule
    await d.frames(4, SIL_DONE); await asyncio.sleep(0.3)
    ctx = d.s.ctx.render()
    ok = d.s.counters["undo"] == 0 and ctx.count("it?") == 1 and ctx.count("<complete>") == 1
    await d.s.http.close()
    return "second peek repeats a peeked word", ok, d


async def scenario_bc_flood():
    """the head's backchannel class firing on noise while nobody speaks and we are silent (runs of 131 and 69
    <user_bc> were seen live; training has at most 2 in a row): none without speech"""
    d = Driver(); await d.start()
    for _ in range(6):
        await d.frames(1, BC); await d.frames(1, [0.02, 0.1, 0.1, 0.5, 0.28])
    await asyncio.sleep(0.2)
    ok = d.s.ctx.render().count("<user_bc>") == 0
    await d.s.http.close()
    return "no <user_bc> flood while idle", ok, d


async def scenario_claude():
    """claude_code with a fake `claude` CLI: the call returns a task_id, the reply goes on; the result arrives later
    as an async result in the user stream (the trained async-result shape), a decision follows and the model reports it; the
    child gets no API key and no parent Claude Code variables; the next task resumes the same Claude session"""
    d = Driver(claude=True); await d.start()
    fake = Path(d.dir, "claude"); envf = Path(d.dir, "env.txt"); argf = Path(d.dir, "args.txt")
    fake.write_text(f"#!/bin/bash\nenv > {envf}\necho RUN >> {argf}\n"
                    f"for a in \"$@\"; do [ \"$prev\" = --resume ] && echo \"RESUME $a\" >> {argf}; prev=$a; done\nsleep 1.5\n"
                    "echo '[{\"type\":\"system\",\"subtype\":\"init\"},{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,"
                    "\"result\":\"The disk is 41% full.\",\"session_id\":\"s-test\",\"num_turns\":2}]'\n")   # the CLI's list form
    fake.chmod(0o755); d.s.claude.bin = str(fake)
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY") or "test-key"; os.environ["CLAUDECODE"] = "1"
    await turn(d, "Ask", "Claude", "about", "the", "disk.")
    idle = [0.02, 0.1, 0.1, 0.02, 0.76]
    await asyncio.sleep(1.0); await d.frames(30, idle); await asyncio.sleep(1.5); await d.frames(10, idle)
    first = d.s.claude.session_id
    d.s.claude.start("and the memory?")                           # a second task resumes the Claude session
    await asyncio.sleep(2.2)
    ctx = d.s.ctx.render(); env = envf.read_text() if envf.exists() else ""; argv = argf.read_text().splitlines() if argf.exists() else []
    tools = [t["name"] for t in d.types("tool")]
    ok = ("response:claude_code{task_id:" in ctx and "<|turn>user\n<|tool_response>response:claude_code{answer:" in ctx.replace(" [->listen]", "")
          and "Asking Claude." in ctx and "Claude says the disk is fine." in ctx and tools.count("claude_code") >= 2
          and "ANTHROPIC_API_KEY" not in env and "CLAUDECODE" not in env and first == "s-test"
          and argv == ["RUN", "RUN", "RESUME s-test"]
          and Path(d.dir, "claude_tasks.jsonl").exists())
    await d.s.http.close()
    return "claude_code: async task, result reported", ok, d


async def scenario_frontier():
    """ask_frontier on a mock GLM: task_id at once, the answer later as the trained async-result shape {answer, confidence,
    task_id}, then the report; the cost goes to frontier_tasks.jsonl; the Facts line names the session's tools"""
    from speakrail.harness import frontier_tool as FT
    FT.ENDPOINT = f"http://127.0.0.1:{PORT}/v1/chat/completions"; os.environ["FIREWORKS_API_KEY"] = "test-fw"
    d = Driver(frontier=True); await d.start()
    await turn(d, "Ask", "the", "bigger", "model.")
    idle = [0.02, 0.1, 0.1, 0.02, 0.76]
    await asyncio.sleep(1.0); await d.frames(20, idle); await asyncio.sleep(0.5)
    ctx = d.s.ctx.render(); log = Path(d.dir, "frontier_tasks.jsonl")
    rec = json.loads(log.read_text().splitlines()[0]) if log.exists() else {}
    ok = ("response:ask_frontier{task_id:<|\"|>f1<|\"|>}" in ctx and "Asking the bigger model." in ctx
          and "response:ask_frontier{answer:<|\"|>Yes, for most people" in ctx and "confidence:<|\"|>high<|\"|>" in ctx
          and "The bigger model says yes." in ctx and rec.get("usd") == round((900 * 0.15 + 300 * 0.50) / 1e6, 5)
          and ". For hard questions you can ask a bigger, smarter model." in d.s.system and "declaration:task_status" in ctx)
    await d.s.http.close()
    return "ask_frontier: GLM answer reported", ok, d


async def scenario_stateful_spec():
    """record_note asked during speculation: runs once, only after the rule commits (held until then)"""
    d = Driver(); await d.start()
    await turn(d, "Make", "a", "note.")
    await asyncio.sleep(2.0); await d.frames(10, [0.02, 0.1, 0.1, 0.02, 0.76]); await asyncio.sleep(0.3)
    tl = "\n".join(d.s.tl); i_tool = tl.find("TOOL record_note"); i_adopt = tl.find("speculation adopted")
    ok = d.s.toolbox.notes == ["buy milk"] and 0 <= i_adopt < i_tool and "Saved." in d.s.ctx.render()
    await d.s.http.close()
    return "stateful tool in a spec (record_note)", ok, d


IDLE = [0.02, 0.1, 0.1, 0.02, 0.76]


async def scenario_interrupt_stale():
    """the words the user said before our interrupt could be heard (ASR delay) are dropped: the reply plays in full"""
    d = Driver(); await d.start()
    await d.frames(4, SPEAK); d.words("I", "have", "like", "$8,000")
    await asyncio.sleep(0.25)                                   # the interrupt is decided and starts playing ...
    d.words("in", "my")                                          # ... and these arrive, spoken 300 ms before
    await d.frames(55, IDLE); await asyncio.sleep(0.3)
    ctx = d.s.ctx.render()
    ok = (("word", "interrupt") in d.decisions() and not any(w == "overlap" for w, _ in d.decisions())
          and d.s.counters["stale_dropped"] == 2 and INTERRUPT_REPLY in ctx and " in my" not in ctx and not d.types("stop_audio"))
    await d.s.http.close()
    return "interrupt: stale words dropped", ok, d


async def scenario_interrupt_hold():
    """the user talks over our interrupt (after they could hear it): held until the first sentence has played, then the
    overlap opens there and the model yields; the first sentence is out and saved"""
    d = Driver(); await d.start()
    await d.frames(4, SPEAK); d.words("I", "have", "like", "$8,000")
    await d.frames(10, SPEAK)                                   # our audio has been playing for a while
    R = d.s.R; fresh = (R.play_ms or 0) + 400
    d.words("no", "wait,", "listen", start=fresh)
    await asyncio.sleep(0.2)
    held_while_playing = d.s.R is R and not R.cut and d.s.counters["held_overlaps"] == 1 and not any(w == "overlap" for w, _ in d.decisions())
    await d.frames(40, IDLE); await asyncio.sleep(0.3)
    ctx = d.s.ctx.render()
    ok = (held_while_playing and ("overlap", "yield") in d.decisions() and "Sorry, it's ten thousand.<turn|>" in ctx
          and "no wait, listen [->yield]" in ctx.replace("\n", " ") or False)
    if not ok: print("held_while_playing", held_while_playing)
    await d.s.http.close()
    return "interrupt: holds its first sentence", ok, d


async def scenario_no_empty_pin():
    """a backchannel before a (slow) reply has any word waits for the first word: no overlap with an empty model turn"""
    d = Driver(); d.s.cfg.spec = False; await d.start()
    await turn(d, "Tell", "me", "slowly.")
    await asyncio.sleep(0.3)                                    # speak decided; the reply's first token is 0.6 s away
    words_then = d.s.R.n_words() if d.s.R else -1
    d.words("yeah")
    await d.frames(50, IDLE); await asyncio.sleep(0.3)
    ctx = d.s.ctx.render(); tl = "\n".join(d.s.tl)
    ok = (words_then == 0 and "held the reply has no words yet" in tl and "<|turn>model\n<turn|>" not in ctx
          and ("overlap", "continue") in d.decisions() and "<|turn>model\nSure,<turn|>" in ctx
          and "it is fourteen degrees and cloudy in Berlin right now.<turn|>" in ctx)
    await d.s.http.close()
    return "no overlap pin before the first word", ok, d


async def scenario_spec_live_duplicate():
    """the spec's peek already had the last word; its live copy arrives after the endpoint rule fired but before the spec's
    decision landed: absorbed by the spec, it must not re-arm the rule (no second <complete>, no overlap while we speak)"""
    d = Driver(); await d.start()
    DECIDE_DELAY[0] = 0.4
    try:
        await d.frames(4, SPEAK); d.words("Jarvis,", "can", "you")
        st = d.audio_ms
        d.asr.pending_words = [{"text": "answer?", "start": st, "end": st + 200}]
        await d.frames(1, SIL_HALF)                           # spec fires, peeks "answer?" (65 ms), decides (400 ms)
        await d.frames(3, SIL_DONE)                           # the rule fires while the decision is pending
        d.s._on_words({"type": "transcript", "text": "answer?", "words": [{"text": "answer?", "start": st, "end": st + 200}]},
                      time.monotonic())                       # the live copy of the peeked word
        await asyncio.sleep(0.3)
        await d.frames(8, [0.02, 0.95, 0.02, 0.01, 0.0])      # we speak; the head keeps saying "complete"
    finally:
        DECIDE_DELAY[0] = 0.03
    await d.frames(40, IDLE); await asyncio.sleep(0.3)
    ctx = d.s.ctx.render()
    ok = (d.s.counters["mspec_promoted"] == 1 and d.s.counters["overlaps"] == 0 and ctx.count("<complete>") == 1
          and not d.types("stop_audio") and REPLY in ctx.replace("\n", " "))
    await d.s.http.close()
    return "spec: live copy of a peeked word", ok, d


async def scenario_silence_fallback():
    """the head never reaches the endpoint threshold: after silence_fallback_ms of silence the harness sends <complete>
    and the model decides (the mock speaks)"""
    d = Driver(); d.s.m.silence_fallback_ms = 2000; await d.start()
    await d.frames(4, SPEAK); d.words("So", "it", "hasn't", "been", "released", "yet.")
    await d.frames(30, SIL_INC)                                # 2.4 s of silence, P(complete) 0.3: the rule never fires
    await asyncio.sleep(0.8)
    tl = "\n".join(d.s.tl)
    #the reply is still playing here, so it is checked in the timeline, not the context; the <sil:2s> marker that
    # lands with the fallback must not open an overlap (it used to, pinning the reply at word 1)
    ok = (d.s.counters["silence_fallback"] == 1 and "fallback <complete>" in tl and ("complete", "speak") in d.decisions()
          and f"ASSISTANT (speak): {REPLY.split()[0]}" in tl and d.s.counters["overlaps"] == 0)
    await d.s.http.close()
    return "silence fallback (2 s)", ok, d


async def scenario_spec_parallel_listen():
    """spec_parallel: the fork decides listen -> the reply requested with the decision is killed, none of it
    (text, audio, context) gets out; the next turn speaks through a new speculation"""
    d = Driver(); await d.start()
    DECIDE_DELAY[0] = 0.12                                     # slower than the reply's first tokens: they exist, held
    await turn(d, "Well", "maybe")
    await asyncio.sleep(0.5)
    killed = d.s.counters["mspec_parallel_killed"]
    leaked = bool(d.types("reply_start") or d.types("reply_delta") or d.played)   # (a bool: d.played grows later)
    if leaked: print("LEAKED:", d.types("reply_start"), d.types("reply_delta")[:3], d.played, [e for e in d.events if e.get("type") in ("tts_seg", "spec_discard", "error")][:6])
    DECIDE_DELAY[0] = 0.03
    await d.frames(6, SPEAK); await turn(d, "What's", "the", "weather?")
    await asyncio.sleep(0.4); await d.frames(30, [0.02, 0.1, 0.1, 0.02, 0.76]); await asyncio.sleep(0.3)
    ctx = d.s.ctx.render()
    conds = {"killed": killed == 1, "nothing leaked": not leaked, "spec listen": ("complete (spec)", "listen") in d.decisions(),
             "spec speak": ("complete (spec)", "speak") in d.decisions(), "one model turn": ctx.count("<|turn>model") == 1,
             "one reply": len(d.types("reply_start")) == 1}
    ok = all(conds.values())
    if not ok: print("conditions:", conds)
    await d.s.http.close()
    return "spec decides listen: parallel reply killed", ok, d


async def scenario_nudge_after_listen():
    """an async result waited unreported; the nudge fired right after a call that said listen.
    It must take over that listen (no "conflicts with an existing target"), put the result back in front of the model and
    the forced reply must report it"""
    d = Driver(); await d.start()
    await d.frames(4, SPEAK); d.words("Hmm")                  # a word -> a call -> listen (the context ends on that call)
    await asyncio.sleep(0.3)
    d.s.unreported_results = [{"name": "claude_code", "result": {"task_id": "c1", "answer": "the disk is fine"}, "h": len(d.s.hist)}]
    d.s.result_unreported = True; d.s._push([("nudge",)])
    await asyncio.sleep(0.6)
    said = "".join(e.get("text", "") for e in d.types("reply_delta"))
    errs = d.types("error")
    ok = (not errs and len(d.types("reply_start")) == 1 and d.s.counters.get("result_nudges") == 1
          and ("word", "listen") in d.decisions() and "disk is fine" in said and d.s.ctx.render().count("the disk is fine") == 1)
    if not ok: print("ERRORS:", errs, "| said:", said, "| ctx tail:", d.s.ctx.render()[-300:])
    await d.s.http.close()
    return "result nudge: takes over a listen, re-inserts the result, reports it", ok, d


async def scenario_nudge_already_mentioned():
    """a reply already told the user the result: the nudge must not take the turn"""
    d = Driver(); await d.start()
    await d.frames(4, SPEAK); d.words("Hmm"); await asyncio.sleep(0.3)
    h = len(d.s.hist)
    d.s.unreported_results = [{"name": "claude_code", "result": {"task_id": "c1", "answer": "the backup finished and the disk is healthy"}, "h": h}]
    d.s.hist.append("Assistant: Claude says the backup finished and your disk is healthy.")
    d.s.result_unreported = True; d.s._push([("nudge",)])
    await asyncio.sleep(0.5)
    ok = not d.types("reply_start") and not d.types("error") and not d.s.counters.get("result_nudges")
    await d.s.http.close()
    return "result nudge: no turn when the result was already mentioned", ok, d


async def scenario_peek_respelled():
    """the peek had 'FDBV3', the live copy of the same word came back as 'FDBB3?' (same start, 80 ms
    later end) after the speculative reply started: it is the same word, not new speech (no early-start undo)"""
    d = Driver(); await d.start()
    await d.frames(4, SPEAK); d.words("what", "about")
    st = d.audio_ms
    d.asr.pending_words = [{"text": "FDBV3", "start": st, "end": st + 320}]
    await d.frames(1, SIL_HALF); await d.frames(4, SIL_DONE)  # spec peeks "FDBV3", the rule commits, the reply starts
    await asyncio.sleep(0.2)
    d.s._on_words({"type": "transcript", "text": "FDBB3?", "words": [{"text": "FDBB3?", "start": st, "end": st + 400}]},
                  time.monotonic())
    await asyncio.sleep(0.3); await d.frames(30, IDLE); await asyncio.sleep(0.3)
    ok = (d.s.counters["mspec_promoted"] == 1 and not d.s.counters.get("undo") and not d.types("stop_audio")
          and "FDBB3" not in d.s.ctx.render())
    if not ok: print("counters:", {k: v for k, v in d.s.counters.items() if v}, "| stop:", d.types("stop_audio"))
    await d.s.http.close()
    return "spec: a re-spelled live copy of the peeked word", ok, d


def ant(p, a80, a160, a240):
    """an all_ant80 head frame: the 5 softmax classes + 3 anticipation sigmoids (P(turn ends within 80/160/240 ms))"""
    return p + [a80, a160, a240]


async def ant4_turn(confirm_tail):
    """ANT4=6:0.7: ant160 crosses 0.7 while the user still speaks -> early speculation (peek: "in Berlin?"); today's
    trigger then re-peeks and sees `confirm_tail`"""
    old = MS.ANT4; MS.ANT4 = [(6, 0.7)]
    try:
        d = Driver(); await d.start()
        await d.frames(4, ant(SPEAK, 0.02, 0.05, 0.1)); d.words("What's", "the", "weather")
        d.asr.pending_words = [{"text": w, "start": d.audio_ms, "end": d.audio_ms + 200} for w in ("in", "Berlin?")]
        await d.frames(2, ant([0.9, 0.04, 0.03, 0.01, 0.02], 0.6, 0.85, 0.9))   # still speaking, about to stop
        d.asr.pending_words = [{"text": w, "start": d.audio_ms, "end": d.audio_ms + 200} for w in confirm_tail]
        await d.frames(1, ant(SIL_HALF, 0.8, 0.9, 0.95)); await d.frames(4, ant(SIL_DONE, 0.9, 0.95, 0.97))
        await asyncio.sleep(0.4)
        await d.frames(60, ant([0.02, 0.1, 0.1, 0.02, 0.76], 0.01, 0.01, 0.01))
        await asyncio.sleep(0.3)
        return d
    finally:
        MS.ANT4 = old


async def scenario_ant4_confirmed():
    d = await ant4_turn(("in", "Berlin?"))
    c, ctx = d.s.counters, d.s.ctx.render()
    early = [e for e in d.types("spec") if e.get("early")]
    ok = (c["early_fired"] == 1 and c["early_confirmed"] == 1 and c["early_replaced"] == 0 and c["mspec_fired"] == 1
          and c["mspec_promoted"] == 1 and len(early) == 1 and ("complete (spec)", "speak") in d.decisions()
          and " in Berlin?" in ctx and REPLY.split()[0] in ctx and ctx.count("<|turn>model") == 1)
    await d.s.http.close()
    return "ANT4: early speculation confirmed by today's trigger", ok, d


async def scenario_ant4_replaced():
    d = await ant4_turn(("in", "Paris?"))
    c, ctx = d.s.counters, d.s.ctx.render()
    ok = (c["early_fired"] == 1 and c["early_replaced"] == 1 and c["early_confirmed"] == 0 and c["mspec_fired"] == 2
          and c["mspec_promoted"] == 1 and " in Paris?" in ctx and "Berlin?" not in ctx and ctx.count("<|turn>model") == 1)
    await d.s.http.close()
    return "ANT4: early speculation replaced (the words changed)", ok, d


async def scenario_ant4_confirm_skipped():
    """ant160 crosses on the last speech frame, then only silence: nothing the early peek missed, so today's trigger
    confirms without a second peek (ANT4_CONFIRM_SKIP)"""
    old = MS.ANT4; MS.ANT4 = [(6, 0.7)]
    try:
        d = Driver(); await d.start()
        await d.frames(4, ant(SPEAK, 0.02, 0.05, 0.1)); d.words("What's", "the", "weather")
        d.asr.pending_words = [{"text": w, "start": d.audio_ms, "end": d.audio_ms + 200} for w in ("in", "Berlin?")]
        await d.frames(1, ant([0.9, 0.04, 0.03, 0.01, 0.02], 0.6, 0.85, 0.9))   # the last speech frame
        await d.frames(1, ant(SIL_HALF, 0.8, 0.9, 0.95)); await d.frames(4, ant(SIL_DONE, 0.9, 0.95, 0.97))
        await asyncio.sleep(0.4)
        await d.frames(60, ant([0.02, 0.1, 0.1, 0.02, 0.76], 0.01, 0.01, 0.01))
        await asyncio.sleep(0.3)
    finally:
        MS.ANT4 = old
    c, ctx = d.s.counters, d.s.ctx.render()
    spec_peeks = [i for i in d.asr.peeks if str(i).startswith(("mspec", "mconf"))]
    ok = (c["early_fired"] == 1 and c["early_confirmed"] == 1 and c["early_confirm_skipped"] == 1 and c["early_replaced"] == 0
          and len(spec_peeks) == 1 and c["mspec_promoted"] == 1 and " in Berlin?" in ctx and ctx.count("<|turn>model") == 1)
    if not ok: print("peeks:", d.asr.peeks)
    await d.s.http.close()
    return "ANT4: confirmed without a second peek (nothing changed)", ok, d


async def scenario_ant4_old_head():
    old = MS.ANT4; MS.ANT4 = [(6, 0.7)]
    try:
        name, ok, d = await scenario_spec_turn()               # 5-value head: p[6] doesn't exist, nothing fires early
    finally:
        MS.ANT4 = old
    return "ANT4 on with a 5-class head: unchanged", ok and d.s.counters["early_fired"] == 0, d


async def main():
    app = web.Application(); app.router.add_post("/v1/completions", completions); app.router.add_post("/v1/chat/completions", fireworks)
    runner = web.AppRunner(app); await runner.setup(); await web.TCPSite(runner, "127.0.0.1", PORT).start()
    results = []
    for sc in (scenario_spec_turn, lambda: scenario_barge("off"), lambda: scenario_barge("word"), scenario_backchannel, scenario_early_start, scenario_tool, scenario_two_calls, scenario_stateful_spec,
               scenario_interrupt_stale, scenario_interrupt_hold, scenario_no_empty_pin,
               scenario_spec_live_duplicate, scenario_silence_fallback, scenario_phrase_cache, scenario_safety_net, scenario_reset, scenario_peek_twice, scenario_bc_flood, scenario_claude, scenario_frontier, scenario_spec_parallel_listen, scenario_nudge_after_listen, scenario_nudge_already_mentioned, scenario_peek_respelled,
               scenario_ant4_confirmed, scenario_ant4_replaced, scenario_ant4_confirm_skipped,
               scenario_ant4_old_head):
        name, ok, d = await sc()
        results.append((name, ok))
        print(f"\n=== {name}: {'PASS' if ok else 'FAIL'}")
        print("decisions:", d.decisions())
        print("counters:", {k: v for k, v in d.s.counters.items() if v})
        if not ok or os.environ.get("VERBOSE"):
            print("\n".join(d.s.tl[-40:])); print("--- context tail:\n" + d.s.ctx.render()[-900:])
            errs = d.types("error")
            if errs: print("ERRORS:", errs)
    await runner.cleanup()
    print("\n" + " | ".join(f"{n}: {'PASS' if o else 'FAIL'}" for n, o in results))
    return all(o for _, o in results)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
