#!/usr/bin/env python3
"""The public API on the stub engines, with no GPU and no models: the GPT-Live protocol over raw WebSockets, and through OpenAI's own SDK
if it's installed. Needs the tokenizer, as test_session.py does.

    SPEAKRAIL_TOKENIZER_DIR=... python tests/test_gateway.py
"""
import asyncio, base64, json, math, os, sys, time, urllib.request
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
if Path("/tmp/oai").is_dir():
    sys.path.append("/tmp/oai")
import test_run as TR
from test_run import HOME, Run, wait_ports_free
from speakrail.gateway.audio import Downsampler, Pacer

URL = f"ws://127.0.0.1:{TR.UI}/v1/live/sessions"


def token():
    return Path(HOME, "config/api_token").read_text().strip()


def b64(pcm):
    return base64.b64encode(pcm).decode()


def tone(seconds, rate=24000, hz=300.0, amp=6000):
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype("<i2").tobytes()


class Client:
    def __init__(self, headers=None):
        self.headers = {"Authorization": f"Bearer {token()}"} if headers is None else headers
        self.events = []

    async def __aenter__(self):
        self.ws = await websockets.connect(URL, max_size=None, additional_headers=self.headers)
        self.task = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *a):
        self.task.cancel()
        await self.ws.close()

    async def _read(self):
        async for m in self.ws:
            self.events.append(json.loads(m))

    async def send(self, **e):
        await self.ws.send(json.dumps(e))

    async def until(self, pred, timeout=15.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            for e in self.events:
                if pred(e):
                    return e
            await asyncio.sleep(0.01)
        raise AssertionError(f"timed out; saw {[e['type'] for e in self.events][-12:]}")

    def of(self, kind):
        return [e for e in self.events if e["type"] == kind]

    async def start(self, **session):
        await self.send(type="session.start", event_id="start1", session={"model": "gpt-live-1", **session})
        return await self.until(lambda e: e["type"] in ("session.started", "error"))

    async def speak(self, seconds=2.0, rate=24000):
        pcm = tone(seconds, rate)
        chunk = int(rate * 0.08) * 2
        for i in range(0, len(pcm), chunk):
            await self.send(type="session.input_audio.append", audio=b64(pcm[i:i + chunk]))
            await asyncio.sleep(0.08)

    async def silence(self, seconds, rate=24000, stop=None):
        chunk = bytes(int(rate * 0.08) * 2)
        for _ in range(int(seconds / 0.08)):
            await self.send(type="session.input_audio.append", audio=b64(chunk))
            await asyncio.sleep(0.08)
            if stop and stop():
                return

    def audio_bytes(self):
        return sum(len(base64.b64decode(e["delta"])) for e in self.of("session.output_audio.delta"))


def with_run(*args):
    def deco(fn):
        def wrapped():
            r = Run("--no-bench", *args)
            try:
                r.wait_up()
                asyncio.run(fn())
            finally:
                r.stop()
            assert wait_ports_free(), "ports left open"
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


# ---------------------------------------------------------------------- no server needed
def test_resampler_keeps_a_tone_and_its_length():
    pcm = tone(2.0, 24000, 1000.0)
    out = np.frombuffer(Downsampler().process(pcm), dtype="<i2").astype(np.float64)
    assert abs(len(out) - 32000) <= 2, len(out)
    spec = np.abs(np.fft.rfft(out[1000:17000] * np.hanning(16000)))
    assert abs(np.argmax(spec) - 1000) <= 1, np.argmax(spec)          # 16000 samples at 16 kHz: bin = Hz
    rms = np.sqrt(np.mean(out[2000:] ** 2))
    assert abs(rms - 6000 / math.sqrt(2)) < 60, rms                   # same level


def test_resampler_matches_the_filter_the_pipeline_was_measured_on():
    from scipy.signal import resample_poly
    rng = np.random.default_rng(0)
    speechlike = np.convolve(rng.normal(0, 4000, 24000 * 3), np.ones(4) / 4, "same").astype("<i2")
    ours = np.frombuffer(Downsampler().process(speechlike.tobytes()), dtype="<i2").astype(np.float64)
    ref = resample_poly(speechlike.astype(np.float64), 2, 3)
    n = min(len(ours), len(ref)) - 100
    assert abs(len(ref) - len(ours)) <= 1, (len(ref), len(ours))
    shift = Downsampler.HALF_LEN // Downsampler.DOWN                  # the filter's delay is kept, not dropped
    assert np.abs(ours[100 + shift:n + shift] - ref[100:n]).max() <= 1.0, np.abs(ours[100 + shift:n + shift] - ref[100:n]).max()   # rounding only


def test_80_ms_in_gives_exactly_80_ms_out():
    d = Downsampler()                                                 # the harness times words by how much audio has arrived
    chunks = [len(d.process(bytes(3840))) for _ in range(50)]
    assert set(chunks) == {2560}, set(chunks)


def test_resampler_is_independent_of_chunking():
    pcm = tone(1.0, 24000, 700.0)
    whole = Downsampler().process(pcm)
    d = Downsampler()
    parts = b"".join(d.process(pcm[i:i + n]) for i, n in zip(np.cumsum([0, 7, 101, 1000, 1, 2000, 3, 5000]), [7, 101, 1000, 1, 2000, 3, 5000, len(pcm)]))
    assert parts == whole


def test_resampler_removes_what_16k_cannot_hold():
    out = np.frombuffer(Downsampler().process(tone(1.0, 24000, 11000.0)), dtype="<i2").astype(np.float64)
    assert np.sqrt(np.mean(out[1000:] ** 2)) < 400                    # an 11 kHz tone would alias into speech


def test_pacer_sends_in_real_time_and_can_be_cleared():
    async def run():
        sent = []

        async def send(utt, pcm):
            sent.append((time.monotonic(), utt, len(pcm)))
        first = []
        p = Pacer(send, 24000, lead_s=0.1, on_first=first.append)
        task = asyncio.create_task(p.run())
        t0 = time.monotonic()
        p.push(7, bytes(2 * 24000))                                   # 1 s of audio, all at once
        await asyncio.sleep(0.3)
        n = sum(s[2] for s in sent) / 2 / 24000
        assert 0.3 <= n <= 0.55, n                                    # 0.3 s played + the 0.1 lead, not the whole second
        assert first == [7]
        p.clear()
        await asyncio.sleep(0.05)
        before = sum(s[2] for s in sent)
        await asyncio.sleep(0.5)
        assert sum(s[2] for s in sent) == before, "kept sending after an interruption"
        task.cancel()
    asyncio.run(run())


# ---------------------------------------------------------------------- the protocol
@with_run()
async def test_session_started_and_closed():
    async with Client() as c:
        e = await c.start(instructions="Be brief.")
        assert e["type"] == "session.started" and e["client_event_id"] == "start1", e
        s = e["session"]
        assert s["model"] == "gpt-live-1" and s["status"] == "active" and s["audio"]["format"] == {"type": "audio/pcm", "rate": 24000} and s["id"]
        await c.send(type="session.close")
        closed = await c.until(lambda x: x["type"] == "session.closed")
        assert closed["reason"] == "close_requested" and closed["usage"]["seconds"] > 0


@with_run()
async def test_a_conversation_with_transcripts_and_audio():
    async with Client() as c:
        await c.start()
        await c.speak(2.0)
        await c.silence(5.0, stop=lambda: c.audio_bytes() > 24000)
        said = "".join(e["delta"] for e in c.of("session.input_transcript.delta"))
        heard = "".join(e["delta"] for e in c.of("session.output_transcript.delta"))
        assert "stub" in said or "hello" in said, said
        assert "stub engine" in heard, heard
        pcm = b"".join(base64.b64decode(e["delta"]) for e in c.of("session.output_audio.delta"))
        assert len(pcm) > 24000 and len(pcm) % 2 == 0 and np.abs(np.frombuffer(pcm, "<i2")).max() > 300
        assert all(e["end_ms"] >= e["start_ms"] for e in c.of("session.input_transcript.delta"))
        assert not c.of("error"), c.of("error")
        assert not [e for e in c.events if e["type"].startswith("speakrail.")], "extensions without asking"


@with_run()
async def test_16k_audio_both_ways():
    async with Client() as c:
        e = await c.start(audio={"format": {"type": "audio/pcm", "rate": 16000}})
        assert e["session"]["audio"]["format"]["rate"] == 16000
        await c.speak(2.0, 16000)
        await c.silence(5.0, 16000, stop=lambda: c.audio_bytes() > 16000)
        assert c.audio_bytes() > 8000 and not c.of("error")


@with_run()
async def test_output_is_paced_close_to_real_time():
    async with Client() as c:
        await c.start()
        await c.speak(2.0)
        await c.silence(3.0, stop=lambda: c.audio_bytes() > 2000)
        t0, b0 = time.monotonic(), c.audio_bytes()
        await c.silence(1.5)
        t1, b1 = time.monotonic(), c.audio_bytes()
        rate = (b1 - b0) / 2 / 24000 / (t1 - t0)
        assert 0.0 <= rate < 1.4, rate                                # the stub makes audio 4x faster than this


def test_a_new_reply_keeps_its_words_apart():
    from speakrail.gateway.live import LiveSession
    s = LiveSession(None, None)
    s.pacer = None
    for e in ({"type": "reply_start"}, {"type": "reply_delta", "text": "I need your"}, {"type": "reply_start"}, {"type": "reply_delta", "text": "Let me look."}):
        s.on_harness(e)
    out = []
    while not s.out.empty():
        out.append(s.out.get_nowait()["delta"])
    assert "".join(out) == "I need your Let me look.", out


@with_run()
async def test_mistakes_get_errors():
    async with Client() as c:
        await c.send(type="session.input_audio.append", audio="AAAA", event_id="early")
        e = await c.until(lambda x: x["type"] == "error")
        assert e["error"]["code"] == "session_not_started" and e["client_event_id"] == "early" and e["error"]["client_event_id"] == "early"
        for session, code in (({"audio": {"format": {"type": "audio/pcmu", "rate": 8000}}}, "unsupported_audio_format"),
                              ({"delegation": {"type": "responses", "responses": {"model": "x", "tools": [{"type": "web_search"}]}}}, "unsupported_tool"),
                              ({"delegation": {"type": "decisions", "decisions": {}}}, "unsupported_delegation"),
                              ({"speakrail": {"nope": True}}, "unknown_parameter")):
            c.events.clear()
            e = await c.start(**session)
            assert e["type"] == "error" and e["error"]["code"] == code, (code, e)
        c.events.clear()
        await c.send(type="session.start", session={})
        assert (await c.until(lambda x: x["type"] == "error"))["error"]["code"] == "missing_parameter"
        c.events.clear()
        assert (await c.start())["type"] == "session.started"
        for event, code in (({"type": "session.start", "session": {"model": "x"}}, "already_started"), ({"type": "nope"}, "unknown_event"), ({"type": "session.update", "session": {}}, "unsupported_event"),
                            ({"type": "session.input_audio.append", "audio": "!!not base64"}, "invalid_audio"), ({"type": "response.create"}, "no_pending_response"),
                            ({"type": "response.item.create", "item": {"type": "function_call_output", "call_id": "nope", "output": "{}"}}, "unknown_call_id"),
                            ({"type": "response.item.create", "item": {"type": "message"}}, "unsupported_item")):
            c.events.clear()
            await c.send(**event)
            e = await c.until(lambda x: x["type"] == "error")
            assert e["error"]["code"] == code, (code, e)
        c.events.clear()
        await c.ws.send("not json")
        assert (await c.until(lambda x: x["type"] == "error"))["error"]["code"] == "invalid_json"
        c.events.clear()
        await c.ws.send(b"\x00\x01")
        assert (await c.until(lambda x: x["type"] == "error"))["error"]["code"] == "binary_not_supported"


@with_run()
async def test_one_session_at_a_time_then_free_again():
    async with Client() as a:
        await a.start()
        async with Client() as b:
            e = await b.start()
            assert e["type"] == "error" and e["error"]["code"] == "session_busy", e
        await a.send(type="session.close")
        await a.until(lambda x: x["type"] == "session.closed")
    await asyncio.sleep(0.3)
    async with Client() as c:
        assert (await c.start())["type"] == "session.started"


@with_run()
async def test_a_dropped_connection_frees_the_session():
    c = Client()
    await c.__aenter__()
    await c.start()
    await c.__aexit__()
    await asyncio.sleep(0.5)
    async with Client() as d:
        assert (await d.start())["type"] == "session.started"


async def _rejected(headers, code):
    try:
        ws = await websockets.connect(URL, additional_headers=headers)
    except websockets.exceptions.InvalidStatus as e:
        assert e.response.status_code == code, e.response.status_code
        return
    await ws.close()
    raise AssertionError(f"connected with {headers}")


@with_run()
async def test_token_and_origin_are_enforced():
    await _rejected({}, 401)
    await _rejected({"Authorization": "Bearer wrong"}, 401)
    await _rejected({"Authorization": f"Bearer {token()}", "Origin": "https://evil.example"}, 403)
    async with Client({"Authorization": f"Bearer {token()}", "Origin": f"http://127.0.0.1:{TR.UI}"}) as c:
        assert (await c.start())["type"] == "session.started"
        await c.send(type="session.close")
        await c.until(lambda x: x["type"] == "session.closed")
    await asyncio.sleep(0.3)
    try:                                                              # the private socket hears the microphone too
        ws = await websockets.connect(f"ws://127.0.0.1:{TR.UI}/ws", additional_headers={"Origin": "https://evil.example"})
    except websockets.exceptions.InvalidStatus as e:
        assert e.response.status_code == 403
    else:
        await ws.close()
        raise AssertionError("/ws accepted a foreign origin")
    ws = await websockets.connect(f"ws://127.0.0.1:{TR.UI}/ws", additional_headers={"Origin": f"http://127.0.0.1:{TR.UI}"})
    await ws.close()


def _stub_prompt():
    from speakrail.harness import protocol as P
    with urllib.request.urlopen("http://127.0.0.1:18000/stub/first_prompt") as r:
        return P.decode(json.load(r)["prompt"])


LOOKUP = {"type": "function", "name": "lookup_city", "description": "Look a city up.", "strict": False, "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


@with_run()
async def test_tools_and_instructions_reach_the_model_prompt():
    async with Client() as c:
        await c.start(instructions="Always call the user Captain.", delegation={"type": "responses", "responses": {"model": "ignored", "tools": [LOOKUP, {**LOOKUP, "name": "get_weather", "description": "Their own weather."}]}})
        await asyncio.sleep(0.5)
        prompt = _stub_prompt()
        assert "lookup_city" in prompt and "Look a city up." in prompt, prompt[:600]
        assert "Their own weather." in prompt, "a client's tool with one of our names wins"
        assert "calculator" not in prompt and "reset_chat" not in prompt, "our tools without asking"
        locked = prompt.index("You are Nova")
        assert locked < prompt.index("Always call the user Captain."), "the turn-taking sentences come first"


@with_run()
async def test_default_tools_when_asked_for():
    async with Client() as c:
        await c.start(speakrail={"default_tools": True}, delegation={"type": "responses", "responses": {"model": "x", "tools": [{**LOOKUP, "name": "get_weather", "description": "Their own weather."}]}})
        await asyncio.sleep(0.5)
        prompt = _stub_prompt()
        assert "calculator" in prompt and "reset_chat" in prompt
        assert "Their own weather." in prompt and prompt.count("get_weather") == prompt.count("Their own weather."), "get_weather declared twice"


def _call_of(e):
    return e["type"] == "response.event" and e["event"].get("type") == "response.output_item.done"


@with_run("--param", "llm/stub:flags=tool_call=lookup_city")
async def test_a_function_call_and_its_result():
    async with Client() as c:
        await c.start(delegation={"type": "responses", "responses": {"model": "x", "tools": [LOOKUP]}})
        await c.speak(2.0)
        speaking = asyncio.create_task(c.silence(9.0, stop=lambda: "stub engine" in "".join(e["delta"] for e in c.of("session.output_transcript.delta"))))
        call = await c.until(_call_of)
        item = call["event"]["item"]
        assert item["type"] == "function_call" and item["name"] == "lookup_city" and json.loads(item["arguments"]) == {"city": "Rome"}, item
        created = c.of("session.delegation.created")
        assert created and created[0]["delegation"]["target"] == "responses" and created[0]["delegation"]["id"] == call["delegation_id"]
        await c.send(type="response.create", event_id="early")        # the result hasn't been sent yet
        e = await c.until(lambda x: x["type"] == "error")
        assert e["error"]["code"] == "tool_outputs_required", e
        c.events.remove(e)
        await c.send(type="response.item.create", item={"type": "function_call_output", "call_id": item["call_id"], "output": json.dumps({"found": "yes"})})
        await c.send(type="response.item.create", item={"type": "function_call_output", "call_id": item["call_id"], "output": "{}"})
        e = await c.until(lambda x: x["type"] == "error")
        assert e["error"]["code"] == "duplicate_tool_output", e
        c.events.remove(e)
        assert not [x for x in c.of("session.output_transcript.delta") if "stub engine" in x["delta"]], "resumed before response.create"
        await c.send(type="response.create")
        await speaking
        heard = "".join(e["delta"] for e in c.of("session.output_transcript.delta"))
        assert "stub engine" in heard, heard
        assert [e for e in c.of("response.event") if e["event"]["type"] == "response.completed"]
        assert not c.of("error"), c.of("error")


@with_run()
async def test_extensions_only_when_asked_for():
    async with Client() as c:
        await c.start(speakrail={"events": True, "playback_ack": True})
        await c.speak(2.0)
        await c.silence(5.0, stop=lambda: c.audio_bytes() > 2000)
        assert [e for e in c.events if e["type"] == "speakrail.word"] and [e for e in c.events if e["type"] == "speakrail.head"]
        utts = {e.get("utt") for e in c.of("session.output_audio.delta")}
        assert utts and None not in utts
        await c.send(type="speakrail.played", utt=next(iter(utts)), delay_ms=120)
        await asyncio.sleep(0.2)
        assert not c.of("error"), c.of("error")


@with_run()
async def test_mute_sends_silence_and_is_acknowledged():
    async with Client() as c:
        await c.start()
        await c.send(type="session.input_audio.mute", event_id="m1")
        assert (await c.until(lambda x: x["type"] == "session.input_audio.muted"))["client_event_id"] == "m1"
        await c.speak(1.5)
        assert not c.of("session.input_transcript.delta"), "heard speech while muted"
        await c.send(type="session.input_audio.unmute")
        await c.until(lambda x: x["type"] == "session.input_audio.unmuted")
        await c.speak(1.5)
        await c.until(lambda x: x["type"] == "session.input_transcript.delta", timeout=3)


@with_run("--param", "llm/stub:flags=tool_call=lookup_city")
async def test_official_sdk_and_every_event_fits_its_schema():
    try:
        from openai import AsyncOpenAI
        from openai.types.live.server_event import ServerEvent
        from pydantic import TypeAdapter
    except ImportError as e:
        print(f"SKIP official SDK: {e}")
        return
    adapter = TypeAdapter(ServerEvent)
    client = AsyncOpenAI(api_key=token(), base_url=f"http://127.0.0.1:{TR.UI}/v1")
    seen, heard = {}, ""
    async with client.live.connect() as conn:
        await conn.send({"type": "session.start", "session": {"model": "gpt-live-1", "instructions": "Be brief.", "audio": {"format": {"type": "audio/pcm", "rate": 24000}},
                                                               "delegation": {"type": "responses", "responses": {"model": "gpt-x", "tools": [LOOKUP]}}}})
        speech, chunk = tone(2.0), 3840

        async def mic():
            for i in range(0, len(speech), chunk):
                await conn.send({"type": "session.input_audio.append", "audio": b64(speech[i:i + chunk])})
                await asyncio.sleep(0.08)
            for _ in range(120):
                await conn.send({"type": "session.input_audio.append", "audio": b64(bytes(chunk))})
                await asyncio.sleep(0.08)
        task = None
        t0 = time.monotonic()
        async for ev in conn:
            raw = ev.model_dump(exclude_none=True, by_alias=True)
            adapter.validate_python(raw)                               # strict enough to catch a wrong field or type
            seen[ev.type] = seen.get(ev.type, 0) + 1
            if ev.type == "session.started":
                task = asyncio.create_task(mic())
            elif ev.type == "response.event" and ev.event.get("type") == "response.output_item.done":
                await conn.send({"type": "response.item.create", "item": {"type": "function_call_output", "call_id": ev.event["item"]["call_id"], "output": json.dumps({"ok": True})}})
                await conn.send({"type": "response.create"})
            elif ev.type == "session.output_transcript.delta":
                heard += ev.delta
            elif ev.type == "error":
                raise AssertionError(ev)
            if "stub engine" in heard and seen.get("session.output_audio.delta", 0) > 3 and "closing" not in seen:
                seen["closing"] = 1
                await conn.send({"type": "session.close"})
            if ev.type == "session.closed":
                break
            assert time.monotonic() - t0 < 30, seen
        if task:
            task.cancel()
    for kind in ("session.started", "session.input_transcript.delta", "session.output_transcript.delta", "session.output_audio.delta", "session.delegation.created", "response.event", "session.closed"):
        assert seen.get(kind), (kind, seen)


if __name__ == "__main__":
    assert os.environ.get("SPEAKRAIL_TOKENIZER_DIR"), "set SPEAKRAIL_TOKENIZER_DIR"
    only = sys.argv[1:]
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and (not only or any(o in k for o in only)) and callable(v)]
    tests = [t for t in tests if t.__module__ == "__main__"]
    bad = 0
    for t in tests:
        try:
            t0 = time.monotonic(); t()
            print(f"PASS {t.__name__} ({time.monotonic() - t0:.0f} s)", flush=True)
        except Exception as e:
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {str(e)[:700]}", flush=True)
    print(f"{len(tests) - bad} of {len(tests)} passed")
    raise SystemExit(1 if bad else 0)
