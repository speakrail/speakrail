"""Speakrail protocol: the ONE definition of how the live context is built, shared by the tape builder (training)
and the serving harness (session.py). If training and serving build the context differently, the model sees
inputs it never trained on, so both sides must only ever call this module.

The context is a token-id list built chunk by chunk. Every chunk (the system turn, one word, one input token, one
harness insertion, one reply span) is tokenized on its own, so the ids are identical whether the chunks arrive
live one at a time (serving) or are laid out at once (training).

Layout (Gemma 4 native turns; decisions from tokens_v1):
  <bos><|turn>system\\n{system prompt}<|tool>declaration:...<tool|>...<turn|>\\n<|turn>user\\n
  user turn: words (" word" after the first), input tokens (<complete>, <sil:Ns>, <user_bc>, no space),
             async tool results (<|tool_response>response:name{...}<tool_response|>)
  a call   = one decision token predicted after the last context token; decision sets below
  speak    : model emits <turn|> (saved); harness appends "\\n<|turn>model\\n"; the reply is generated
  interrupt: model emits <interrupt> (saved); harness appends "<turn|>\\n<|turn>model\\n"; the reply opens with an apology
  reply    : text, tool calls "<|tool_call>call:name{...}<tool_call|>" + "<|tool_response>" (model stops here) +
             harness "response:name{...}<tool_response|>", more text, "<turn|>"; harness appends "\\n<|turn>user\\n"
             calls issued together (v3): all "<|tool_call>...<tool_call|>" back to back, one "<|tool_response>", then
             one result block per call in call order, every block after the first with its own "<|tool_response>"
  overlap  : the user speaks while the assistant talks. The harness pins the played word k at speech onset and opens
             an overlap turn: reply[:k] + "<turn|>\\n<|turn>user\\n" + overlapping words -> calls with the overlap set.
             continue: harness closes it ("<turn|>\\n<|turn>model\\n") and saves the rest of the reply after it.
             yield   : the reply stays cut at k; the overlap turn simply is the user's turn from then on.
  urgent event while the assistant talks: same pinning, the tool result goes in the overlap turn, choices
             {continue, interrupt}.
  Not saved: <listen>, <listen_muted>, <continue>, <yield>. No thinking channel is ever opened (<|channel> banned).
"""
from __future__ import annotations
import json, re
from pathlib import Path

HERE = Path(__file__).parent
# tokenizer v1.2 (Gemma 4 + the 22 speakrail tokens), which the installer downloads next to the model
from ..config import CFG                                                            # noqa: E402
TOKENIZER_DIR = Path(CFG.tokenizer_dir)
from .tokens_v1 import IDS, TURN_CLOSE, TURN_OPEN, TOOL_RESPONSE_OPEN, sil_token   # noqa: E402
from . import toolschema                                                                   # noqa: E402

LISTEN, LISTEN_MUTED, INTERRUPT, CONTINUE, YIELD = (IDS[t] for t in ("<listen>", "<listen_muted>", "<interrupt>", "<continue>", "<yield>"))
INTERJECT, INTERJECT_END = IDS["<interject>"], IDS["</interject>"]   # v3: live tasks (1-4 words, the speaker keeps the turn)
SPEAK = TURN_CLOSE
IDLE_DECISIONS = [SPEAK, INTERRUPT, LISTEN, LISTEN_MUTED, INTERJECT]
OVERLAP_DECISIONS = [CONTINUE, YIELD, LISTEN]
EVENT_DECISIONS = [CONTINUE, INTERRUPT]            # urgent event while the assistant is talking
DECISION_SETS = {"idle": IDLE_DECISIONS, "overlap": OVERLAP_DECISIONS, "event": EVENT_DECISIONS}
DECISION_NAME = {SPEAK: "speak", INTERRUPT: "interrupt", LISTEN: "listen", LISTEN_MUTED: "listen_muted",
                 CONTINUE: "continue", YIELD: "yield", INTERJECT: "interject"}
USER_OPEN = "<|turn>user\n"
MODEL_OPEN = "<|turn>model\n"
Q = '<|"|>'                                         # Gemma's string quote token

# ------------------------------------------------------------------ Gemma formatting (ports of the template macros)
def format_argument(v, escape_keys=True):
    if v is None: return "null"
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, str): return Q + v + Q
    if isinstance(v, dict):
        return "{" + ",".join((Q + k + Q if escape_keys else k) + ":" + format_argument(v[k], escape_keys) for k in sorted(v)) + "}"
    if isinstance(v, (list, tuple)): return "[" + ",".join(format_argument(x, escape_keys) for x in v) + "]"
    return str(v)

def tool_call_text(name, args):
    """what the model generates for a call, including the <|tool_response> it stops on"""
    return tool_calls_text([(name, args)])

def tool_calls_text(calls):
    """calls issued together [(name, args), ...]: back to back, then the one <|tool_response> the model stops on"""
    def one(name, args):
        body = ",".join(k + ":" + format_argument(args[k], escape_keys=False) for k in sorted(args or {}))
        return "<|tool_call>call:" + name + "{" + body + "}<tool_call|>"
    return "".join(one(n, a) for n, a in calls) + "<|tool_response>"

def tool_results_text(results):
    """the harness's answer to calls issued together [(name, result), ...]: the first block reuses the model's
    <|tool_response>, every later one opens its own (as Gemma's template and the live harness write them)"""
    return "".join(tool_result_text(n, r, opened=(k == 0)) for k, (n, r) in enumerate(results))

def tool_result_text(name, result, opened=True):
    """harness insertion after the model's <|tool_response>; opened=False adds the opener (async results)"""
    if isinstance(result, dict):
        body = "response:" + name + "{" + ",".join(k + ":" + format_argument(result[k], escape_keys=False) for k in sorted(result)) + "}"
    else:
        body = "response:" + name + "{value:" + format_argument(result, escape_keys=False) + "}"
    return ("" if opened else "<|tool_response>") + body + "<tool_response|>"

# ------------------------------------------------------------------ tool declarations (catalog -> JSON schema)
def tool_declaration(name, spec):
    """catalog tool entry -> OpenAI-style function declaration for the chat template. Arguments come from the shared
    schema module (v3): the v2 shorthand renders exactly as before; nested objects, formats (date, phone, money...) and
    ready JSON Schemas (`parameters`, e.g. imported tools) are new."""
    fn = {"name": name, "description": spec.get("description", ""), "parameters": toolschema.parameters(spec)}
    return {"type": "function", "function": fn}

# ------------------------------------------------------------------ listening notes (v3): when the thinker writes, how notes look
NOTE_MIN_CHUNK_S = 1.2        # a note after a sentence end once the chunk since the last note covers this much speech
NOTE_RUNON_S = 3.0            # ... or after this much speech with no sentence end (rambling, run-on speech)
NOTE_TOK_S, NOTE_MAX_TOKENS = 75, 400   # the thinker must finish before the next chunk arrives: budget = next chunk's speech time x 75 tok/s
NOTE_FORMATS = {"thought": "<|channel>thought\n{text}\n<channel|>",          # Gemma's own thinking channel (chat template format)
                "notes": "<notes>\n{text}\n</notes>\n"}                      # a plain-text block (ablation arm)


def note_points(words):
    """indices of the words after which the thinker writes a note, from the words of one user turn as they arrive:
    [(text, arrival_time_s), ...]. Live harness and tape builder use the same rule."""
    pts, last_t, start_t = [], None, None
    for i, (w, t) in enumerate(words):
        if start_t is None: start_t = t
        since = t - (last_t if last_t is not None else start_t)
        sentence_end = bool(re.search(r"[.?!:][\"')\]]*$", w))
        if (sentence_end and since >= NOTE_MIN_CHUNK_S) or since >= NOTE_RUNON_S:
            pts.append(i); last_t = t
    return pts


def date_line(now, tz=None):
    """the system prompt's last line with the current local date and time (v3; the live harness must write the same)"""
    return f"Current date and time: {now.strftime('%A')}, {now.day} {now.strftime('%B %Y, %H:%M')}" + (f" ({tz})" if tz else "") + "."

_TEMPLATE = None
def system_text(system_prompt, tools):
    """<bos> + the system turn exactly as Gemma's chat template renders it (tools = list of declarations)"""
    global _TEMPLATE
    if _TEMPLATE is None:
        import jinja2
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        _TEMPLATE = env.from_string((TOKENIZER_DIR / "chat_template.jinja").read_text())
    txt = _TEMPLATE.render(messages=[{"role": "system", "content": system_prompt}], tools=tools or None,
                           bos_token="<bos>", add_generation_prompt=False)
    assert txt.endswith("<turn|>\n"), txt[-40:]
    return txt

# ------------------------------------------------------------------ tokenizer
_TOK = None
def tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    return _TOK

def encode(text):
    return tokenizer().encode(text, add_special_tokens=False).ids

def decode(ids):
    return tokenizer().decode(ids, skip_special_tokens=False)

# ------------------------------------------------------------------ the context
class Context:
    """Token ids plus, for training, the target predicted after each position (-100 = no loss) and the decision
    class at call positions. The serving harness uses the same methods and ignores the targets."""

    def __init__(self):
        self.ids: list[int] = []
        self.targets: list[int] = []
        self.calls: list[tuple[int, int, str]] = []   # (position, decision id, decision set) at every call
        self.spans: list[tuple[int, int, str]] = []   # (start, end, kind) for rendering
        self.turn_has_content = False                  # inside the current user/overlap turn
        self.reply_last = None                         # last chunk kind in the current reply: None / "text" / "tool"

    # ---- low level
    def _append(self, text, kind, loss=False):
        ids = encode(text)
        start = len(self.ids)
        if loss and start > 0:                          # predict every token of the chunk from the one before it
            self.targets[start - 1] = ids[0]
        self.ids += ids
        self.targets += [ids[i + 1] if (loss and i + 1 < len(ids)) else -100 for i in range(len(ids))]
        self.spans.append((start, len(self.ids), kind))
        return ids

    def _append_id(self, tid, kind, loss=False):
        start = len(self.ids)
        if loss and start > 0: self.targets[start - 1] = tid
        self.ids.append(tid); self.targets.append(-100)
        self.spans.append((start, start + 1, kind))

    # ---- session start
    def start(self, system_prompt, tools):
        self._append(system_text(system_prompt, tools), "system")
        self._append(USER_OPEN, "harness"); self.turn_has_content = False

    # ---- user stream (user turn or overlap turn)
    def word(self, w):
        self._append(("" if not self.turn_has_content else " ") + w, "word"); self.turn_has_content = True

    def input_token(self, tok):
        """<complete>, <sil:Ns>, <user_bc>"""
        self._append_id(IDS[tok], "input"); self.turn_has_content = True

    def async_result(self, name, result):
        self._append(tool_result_text(name, result, opened=False), "tool_result"); self.turn_has_content = True

    # ---- a call: the decision predicted after the current last token
    def call(self, decision, options="idle"):
        """options = the decision set the harness allows at this call: "idle", "overlap" or "event" (DECISION_SETS)"""
        pos = len(self.ids) - 1
        if decision not in DECISION_SETS[options]: raise ValueError(f"{DECISION_NAME[decision]} not in the {options} set")
        self.calls.append((pos, decision, options))
        if self.targets[pos] not in (-100, decision):
            raise ValueError(f"call at {pos} conflicts with an existing target")
        self.targets[pos] = decision
        if decision == SPEAK:
            self._append_id(SPEAK, "decision")               # saved (its target was set above)
            self._append("\n" + MODEL_OPEN, "harness"); self.reply_last = None
        elif decision == INTERRUPT:
            self._append_id(INTERRUPT, "decision")
            self._append("<turn|>\n" + MODEL_OPEN, "harness"); self.reply_last = None
        elif decision == INTERJECT:                          # v3: saved; the words follow inside the speaker's turn (interject())
            self._append_id(INTERJECT, "decision")
        # listen / listen_muted / continue / yield: not saved

    # ---- the assistant's reply (loss on what the model generates)
    def reply_text(self, text, loss=True):
        """a space only between text chunks; none at the turn start or right after a tool result (as the template)"""
        if not text: return
        self._append((" " if self.reply_last == "text" else "") + text, "reply" if loss else "reply_noloss", loss=loss)
        self.reply_last = "text"

    def reply_tool(self, name, args, result, loss=True):
        self.reply_tools([(name, args, result)], loss=loss)

    def reply_tools(self, calls, loss=True):
        """one or more calls issued together [(name, args, result), ...]: the model writes them back to back and stops on
        one <|tool_response>; the harness answers with one result block per call, in call order (Gemma's template)"""
        self._append(tool_calls_text([(n, a) for n, a, _ in calls]), "tool_call", loss=loss)
        self._append(tool_results_text([(n, r) for n, _, r in calls]), "tool_result")
        self.reply_last = "tool"

    def interject(self, text, calls=None, loss=True):
        """v3 live tasks, right after call(INTERJECT): the model's 1-4 words, optionally the calls of an act task (the model
        stops on <|tool_response>, the harness answers at once), then the model's </interject>. No turn is opened or closed:
        the speaker's words go on in the same turn after it (word() keeps its leading space)."""
        if text: self._append(text, "interject" if loss else "interject_noloss", loss=loss)
        if calls:
            self._append(tool_calls_text([(n, a) for n, a, _ in calls]), "tool_call", loss=loss)
            self._append(tool_results_text([(n, r) for n, _, r in calls]), "tool_result")
        self._append_id(INTERJECT_END, "interject_end", loss=loss)
        self.turn_has_content = True

    def notes(self, text, fmt="thought"):
        """v3 listening notes from the thinker, inserted by the harness right after the model turn opens (no loss).
        fmt "thought": Gemma's native thinking channel; fmt "notes": a plain-text <notes> block (ablation)"""
        text = (text or "").strip()
        if not text: return
        block = NOTE_FORMATS[fmt].format(text=text)
        self._append(block, "notes")
        self.reply_last = None                               # the reply starts right after, no space

    def end_reply(self, loss=True):
        """natural end of the reply: the model's <turn|>, then the harness opens the user turn"""
        self._append_id(TURN_CLOSE, "reply_end", loss=loss)
        self._append("\n" + USER_OPEN, "harness"); self.turn_has_content = False

    def cut_reply(self):
        """the harness closes the reply at the pinned word and opens a user (overlap) turn; not a model output"""
        self._append_id(TURN_CLOSE, "harness")
        self._append("\n" + USER_OPEN, "harness"); self.turn_has_content = False

    def resume_reply(self):
        """after <continue>: close the overlap turn and reopen the model turn for the rest of the reply"""
        self._append_id(TURN_CLOSE, "harness")
        self._append("\n" + MODEL_OPEN, "harness"); self.reply_last = None

    # ---- rendering for humans
    def render(self):
        """decoded context with the call decisions shown inline as [->decision]"""
        call_at = {p: d for p, d, _ in self.calls}
        out = []
        for s, e, kind in self.spans:
            txt = decode(self.ids[s:e])
            out.append(txt)
            if e - 1 in call_at: out.append(f" [->{DECISION_NAME[call_at[e - 1]]}]")
        return "".join(out)
