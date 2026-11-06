"""GET /v1/live/sessions: the WebSocket form of OpenAI's GPT-Live protocol, on one harness session.

Supported: session.start, session.close, session.input_audio.append/mute/unmute, function tools in delegation.responses.tools (the call
and its result as response.event / response.item.create / response.create), session.instructions.
Not here yet, each answered with an error rather than ignored: other delegation modes, hosted tools, session.update, the
thinking/commentary/instructions appends, G.711.

Extensions, opt in with session.speakrail = {...}:
  default_tools: true   our tools (weather, calculator, lists and so on) in the model's tool list, run here
  playback_ack: true    the client says when audio really starts playing, with {"type":"speakrail.played","utt":N,"delay_ms":M}
  events: true          harness events as speakrail.<type>; output audio then carries its `utt`

The engines may have been unloaded while nobody was connected. session.start then loads them, with an info event (code "loading")
in the meantime, and session.started once they're up. Send no audio before session.started.

Turn-taking is the model's own, so there are no turn-detection settings, and output audio keeps flowing as the session decides. The server
sends audio close to real time, so when a reply is cut, there is little left at the client to play.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import secrets
import time
from typing import Awaitable, Callable

from aiohttp import WSMsgType, web

from ..config import secret
from .audio import Downsampler, Pacer

CLIENT_TOOL_TIMEOUT_S = 60.0
USAGE_EVERY_S = 5.0
MAX_CONTEXT_TOKENS = 16384
SESSION_SECONDS = 4 * 3600
ACTIVE: list = []                                      # one session at a time, across the API and the web page: they share one set of engines
RELEASE_WAIT_S = 2.0                                   # Stop then Start: the old session may still be closing
BUSY_MESSAGE = "Another session is active, and this server runs one at a time. End it first."
REFUSALS = ("engines_offloaded", "engines_failed", "not_enough_memory", "host_unavailable")    # the engines won't come up for this session


class Rejected(Exception):
    def __init__(self, code: str, message: str, param: str | None = None):
        super().__init__(message)
        self.code, self.message, self.param = code, message, param


async def claim(owner) -> bool:
    """the one session's place -> whether this got it"""
    deadline = time.monotonic() + RELEASE_WAIT_S
    while ACTIVE:
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.05)
    ACTIVE.append(owner)
    return True


def origin_allowed(request) -> bool:
    """any web page can open a WebSocket to localhost, so a browser's Origin has to be this server's own"""
    origin = request.headers.get("Origin")
    if origin is None:                                 # not a browser
        return True
    return origin.split("://", 1)[-1].rstrip("/") == request.host


def token_ok(request) -> bool:
    expected = secret("SPEAKRAIL_API_TOKEN")
    if not expected:
        return True
    given = request.headers.get("Authorization", "")
    return given.startswith("Bearer ") and secrets.compare_digest(given[7:].strip(), expected)


def parse_tools(session: dict) -> tuple[list[dict], str]:
    """-> (function tools, extra instructions from delegation.responses)"""
    d = session.get("delegation")
    if not d or d.get("type") == "client":
        return [], ""
    if d.get("type") != "responses":
        raise Rejected("unsupported_delegation", f"delegation type {d.get('type')!r} isn't supported", "session.delegation.type")
    r = d.get("responses") or {}
    tools = []
    for i, t in enumerate(r.get("tools") or []):
        if t.get("type") != "function":
            raise Rejected("unsupported_tool", f"only function tools are supported, not {t.get('type')!r}", f"session.delegation.responses.tools[{i}]")
        if not t.get("name"):
            raise Rejected("invalid_tool", "a function tool needs a name", f"session.delegation.responses.tools[{i}]")
        tools.append({"name": t["name"], "description": t.get("description") or "", "parameters": t.get("parameters") or {"type": "object", "properties": {}}})
    return tools, str(r.get("instructions") or "")


def parse_format(session: dict) -> int:
    fmt = (session.get("audio") or {}).get("format")
    if fmt is None:
        return 24000
    if fmt.get("type") != "audio/pcm" or fmt.get("rate") not in (16000, 24000):
        raise Rejected("unsupported_audio_format", "audio/pcm at 16000 or 24000 Hz is supported", "session.audio.format")
    return int(fmt["rate"])


class LiveSession:
    def __init__(self, ws, make_session: Callable, gate=None):
        self.ws, self.make_session, self.gate = ws, make_session, gate
        self.entered = False
        self.sess = None
        self.out: asyncio.Queue = asyncio.Queue()
        self.started_at = time.monotonic()
        self.id = "sess_" + secrets.token_hex(8)
        self.n = 0
        self.rate = 24000
        self.down = None
        self.pacer = None
        self.muted = False
        self.ext = {}
        self.resource = {}
        self.pending: dict[str, asyncio.Future] = {}
        self.results: dict[str, object] = {}
        self.deleg = None
        self.new_reply = False
        self.said_any = False

    # ------------------------------------------------------------------ sending
    def event(self, kind: str, **fields) -> dict:
        self.n += 1
        return {"type": kind, "event_id": f"event_{self.n}", **fields}

    def emit(self, kind: str, **fields) -> None:
        self.out.put_nowait(self.event(kind, **fields))

    def error(self, code: str, message: str, client_event_id=None, param=None, kind="invalid_request_error") -> None:
        err = {"code": code, "message": message, "type": kind}
        if client_event_id:
            err["client_event_id"] = client_event_id
        if param:
            err["param"] = param
        fields = {"error": err}
        if client_event_id:
            fields["client_event_id"] = client_event_id
        self.emit("error", **fields)

    def now_ms(self) -> int:
        return round((time.monotonic() - self.started_at) * 1000)

    async def writer(self) -> None:
        while True:
            e = await self.out.get()
            if e is None:
                return
            await self.ws.send_str(json.dumps(e, ensure_ascii=False))

    # ------------------------------------------------------------------ from the harness
    def on_harness(self, obj) -> None:
        if obj is None:
            return
        if isinstance(obj, bytes):
            self.pacer.push(int.from_bytes(obj[:2], "little"), obj[4:])
            return
        kind = obj.get("type")
        if kind == "word":
            self.emit("session.input_transcript.delta", delta=obj.get("raw") or " " + obj.get("text", ""), start_ms=int(obj.get("start", 0)), end_ms=int(obj.get("end", 0)))
        elif kind == "reply_start":
            self.new_reply = True                       # a reply cut short can't be taken back: at least keep the words apart
        elif kind == "reply_delta":
            if obj.get("text"):
                t = self.now_ms()
                text = obj["text"]
                if self.new_reply and self.said_any and not text.startswith(" "):
                    text = " " + text
                self.new_reply, self.said_any = False, True
                self.emit("session.output_transcript.delta", delta=text, start_ms=t, end_ms=t)
        elif kind == "stop_audio":
            self.pacer.clear()
        elif kind == "duck":
            self.pacer.set_gain(float(obj.get("gain", 1.0)))
        elif kind == "error":
            self.error("engine_error", f"{obj.get('where')}: {str(obj.get('detail'))[:300]}", kind="server_error")
        if self.ext.get("events"):
            self.out.put_nowait({"type": f"speakrail.{kind}", **{k: v for k, v in obj.items() if k != "type"}})

    async def send_audio(self, utt: int, pcm: bytes) -> None:
        e = self.event("session.output_audio.delta", delta=base64.b64encode(pcm).decode())
        if self.ext.get("events"):
            e["utt"] = utt
        self.out.put_nowait(e)

    def first_audio(self, utt: int) -> None:
        if not self.ext.get("playback_ack") and self.sess:
            self.sess.note_play_start(utt, 0.0)

    # ------------------------------------------------------------------ client tools
    async def run_client_tool(self, name: str, args: dict):
        """called by the harness when the model calls a declared tool: ask the client, and wait for its answer"""
        call_id = "call_" + secrets.token_hex(8)
        resp_id, deleg_id = "resp_" + secrets.token_hex(8), "item_" + secrets.token_hex(8)
        self.emit("session.delegation.created", offset_ms=self.now_ms(), delegation={"id": deleg_id, "target": "responses", "type": "delegation", "response_id": resp_id})
        nested = lambda e: self.emit("response.event", delegation_id=deleg_id, event=e)
        nested({"type": "response.created", "response": {"id": resp_id, "status": "in_progress", "output": []}})
        nested({"type": "response.output_item.done", "output_index": 0, "sequence_number": 1,
                "item": {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": name, "arguments": json.dumps(args), "status": "completed"}})
        fut = asyncio.get_running_loop().create_future()
        self.pending[call_id] = fut
        try:
            result = await asyncio.wait_for(fut, CLIENT_TOOL_TIMEOUT_S)
        except asyncio.TimeoutError:
            result = {"error": "the tool didn't answer in time"}
            self.error("tool_timeout", f"no result for {call_id} in {CLIENT_TOOL_TIMEOUT_S:.0f} s", kind="server_error")
        finally:
            self.pending.pop(call_id, None)
            self.results.pop(call_id, None)
        nested({"type": "response.completed", "response": {"id": resp_id, "status": "completed", "output": []}})
        return result

    def on_item_create(self, m: dict, eid) -> None:
        item = m.get("item") or {}
        if item.get("type") != "function_call_output":
            raise Rejected("unsupported_item", "only function_call_output items are supported", "item.type")
        call_id = item.get("call_id")
        if call_id not in self.pending:
            raise Rejected("unknown_call_id", f"no pending call {call_id!r}", "item.call_id")
        if call_id in self.results:
            raise Rejected("duplicate_tool_output", f"{call_id} already has a result", "item.call_id")
        output = item.get("output")
        if not isinstance(output, str):
            raise Rejected("invalid_tool_output", "output must be a string", "item.output")
        try:
            value = json.loads(output)
        except ValueError:
            value = output
        self.results[call_id] = value if isinstance(value, dict) else {"result": value}

    def on_response_create(self) -> None:
        if not self.pending:
            raise Rejected("no_pending_response", "no tool call is waiting for a result")
        missing = [c for c in self.pending if c not in self.results]
        if missing:
            raise Rejected("tool_outputs_required", f"missing results for {', '.join(missing)}")
        for call_id, fut in list(self.pending.items()):
            if not fut.done():
                fut.set_result(self.results[call_id])

    # ------------------------------------------------------------------ from the client
    async def start(self, m: dict) -> None:
        s = m.get("session")
        if not isinstance(s, dict) or not s.get("model"):
            raise Rejected("missing_parameter", "session.start needs session.model", "session.model")
        self.rate = parse_format(s)
        tools, extra = parse_tools(s)
        self.ext = dict(s.get("speakrail") or {})
        if unknown := set(self.ext) - {"default_tools", "playback_ack", "events"}:
            raise Rejected("unknown_parameter", f"unknown speakrail options: {sorted(unknown)}", "session.speakrail")
        instructions = "\n".join(x for x in (str(s.get("instructions") or ""), extra) if x)
        if not await claim(self):
            raise Rejected("session_busy", BUSY_MESSAGE)
        if self.gate:
            await self.gate.enter(self.ws, lambda text: self.emit("info", code="loading", message=text))
            self.entered = True
        self.down = Downsampler() if self.rate == 24000 else None
        self.pacer = Pacer(self.send_audio, self.rate, on_first=self.first_audio)
        self.sess = self.make_session(self.on_harness, tools=tools, default_tools=bool(self.ext.get("default_tools")), instructions=instructions, run_client_tool=self.run_client_tool)
        await self.sess.start()
        self.resource = {"id": self.id, "expires_at": int(time.time()) + SESSION_SECONDS, "model": s["model"], "status": "active",
                         "audio": {"format": {"type": "audio/pcm", "rate": self.rate}},
                         "instructions": instructions or None}
        if s.get("delegation"):
            self.resource["delegation"] = s["delegation"]
        self.emit("session.started", session=self.resource, **({"client_event_id": m["event_id"]} if m.get("event_id") else {}))
        voice = ((s.get("audio") or {}).get("output") or {}).get("voice")
        if voice:
            self.emit("info", code="voice_ignored", message=f"voice {voice!r} isn't available on this server; using its configured voice")
        ignored = sorted(k for k in ("input", "client", "triggers") if s.get(k))
        if ignored:
            self.emit("info", code="ignored_parameter", message=f"not supported yet, ignored: {', '.join(ignored)}")

    def on_audio(self, m: dict) -> None:
        try:
            pcm = base64.b64decode(m.get("audio", ""), validate=True)
        except (binascii.Error, ValueError):
            raise Rejected("invalid_audio", "audio must be base64", "audio") from None
        if self.down:
            pcm = self.down.process(pcm)
        if pcm:
            self.sess.feed_audio(bytes(len(pcm)) if self.muted else pcm)   # silence, not nothing: the recognizer keeps running

    async def handle(self, m: dict) -> bool:
        """-> False once the session is over"""
        kind, eid = m.get("type"), m.get("event_id")
        ack = {"client_event_id": eid} if eid else {}
        try:
            if kind == "session.close":
                return False
            if kind == "session.start":
                if self.sess is not None:
                    raise Rejected("already_started", "the session has already started")
                await self.start(m)
            elif self.sess is None:
                raise Rejected("session_not_started", "send session.start and wait for session.started first")
            elif kind == "session.input_audio.append":
                self.on_audio(m)
            elif kind == "session.input_audio.mute":
                self.muted = True
                self.emit("session.input_audio.muted", **ack)
            elif kind == "session.input_audio.unmute":
                self.muted = False
                self.emit("session.input_audio.unmuted", **ack)
            elif kind == "response.item.create":
                self.on_item_create(m, eid)
            elif kind == "response.create":
                self.on_response_create()
            elif kind == "speakrail.played":
                self.sess.note_play_start(int(m["utt"]), float(m.get("delay_ms", 0)))
            elif kind in ("session.update", "session.thinking.append", "session.commentary.append", "session.instructions.append", "decision.create", "decision.item.create", "decision.item.delete", "output_audio.fixture.play"):
                raise Rejected("unsupported_event", f"{kind} isn't supported yet")
            else:
                raise Rejected("unknown_event", f"unknown event type {kind!r}", "type")
        except Rejected as e:
            self.error(e.code, e.message, eid, e.param)
            if e.code in ("session_busy", *REFUSALS):
                return False
        return True

    async def usage(self) -> None:
        while True:
            await asyncio.sleep(USAGE_EVERY_S)
            if self.sess is not None:
                ids = getattr(getattr(self.sess, "ctx", None), "ids", ())
                self.emit("session.usage.updated", usage={"seconds": time.monotonic() - self.started_at}, context_window={"usage_ratio": min(1.0, len(ids) / MAX_CONTEXT_TOKENS)})

    async def serve(self) -> None:
        writer = asyncio.create_task(self.writer())
        usage = asyncio.create_task(self.usage())
        pacer = None
        reason = "connection_lost"
        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        m = json.loads(msg.data)
                    except ValueError:
                        self.error("invalid_json", "this message isn't JSON")
                        continue
                    if not isinstance(m, dict):
                        self.error("invalid_event", "an event must be a JSON object")
                        continue
                    try:
                        go_on = await self.handle(m)
                    except Exception as e:                # a failure to start: nothing else can run
                        self.error("engine_error", f"{type(e).__name__}: {e}"[:300], m.get("event_id"), kind="server_error")
                        break
                    if self.pacer is not None and pacer is None:
                        pacer = asyncio.create_task(self.pacer.run())
                    if not go_on:
                        reason = "close_requested"
                        break
                elif msg.type == WSMsgType.BINARY:
                    self.error("binary_not_supported", "send audio as session.input_audio.append")
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            usage.cancel()
            if pacer:
                pacer.cancel()
            for fut in self.pending.values():
                if not fut.done():
                    fut.cancel()
            if self.sess is not None:
                await self.sess.close()
            if self in ACTIVE:
                ACTIVE.remove(self)
            if self.entered:
                await self.gate.leave(self.ws)
            if self.resource:
                self.emit("session.closed", reason=reason, session=self.resource, usage={"seconds": time.monotonic() - self.started_at})
            self.out.put_nowait(None)
            try:
                await writer
            except Exception:
                pass


def handler(make_session: Callable, gate=None) -> Callable[..., Awaitable]:
    async def live(request):
        if not origin_allowed(request):
            raise web.HTTPForbidden(text="this origin may not use the API")
        if not token_ok(request):
            raise web.HTTPUnauthorized(text="a valid Authorization: Bearer token is required")
        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=20)
        await ws.prepare(request)
        await LiveSession(ws, make_session, gate).serve()
        if not ws.closed:
            await ws.close()
        return ws
    return live
