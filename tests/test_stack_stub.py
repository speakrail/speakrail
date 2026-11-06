#!/usr/bin/env python3
"""The whole stack on stub engines, with no GPU and no models: the supervisor starts the engines, the harness runs against them,
and a client talks to it the way a browser does. Needs the tokenizer (SPEAKRAIL_TOKENIZER_DIR), as test_session.py does.

    SPEAKRAIL_TOKENIZER_DIR=... python tests/test_stack_stub.py
"""
import asyncio, json, math, os, signal, struct, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]
os.environ.setdefault("SPEAKRAIL_HOME", tempfile.mkdtemp(prefix="speakrail-stack-"))
import aiohttp
from speakrail.manifest import discover
from speakrail.supervisor import FAILED, Supervisor, check_health

UI_PORT = 18080
FRAME = 2560


def tone(seconds):
    n = int(16000 * seconds)
    return b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 300 * i / 16000))) for i in range(n))


def start_harness(sup, sessions):
    tts = sup.running["tts/stub"].engine
    conf = sessions / "config.json"
    conf.write_text(json.dumps({"turn_k": 1}))        # the stub head is built for K=1, and only a config can set that
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1", "ASR_URL": sup.url("asr/stub"), "LLM_URL": sup.url("llm/stub"), "TTS_URL": sup.url("tts/stub"),
           "TTS_ENGINE": tts.harness["worker"], "TTS_MODEL": tts.harness["model"], "SPEAKRAIL_CONFIG": str(conf), "UI_PORT": str(UI_PORT), "SESSIONS_DIR": str(sessions), "VOICE": "", "PHRASE_CACHE_DIR": str(sessions / "phrases")}
    return subprocess.Popen([sys.executable, "-u", "-m", "speakrail.harness.server", "--no-phrase-cache"], env=env, stdout=open(sessions / "harness.log", "wb"), stderr=subprocess.STDOUT, start_new_session=True)


async def talk():
    events, audio = [], 0
    async with aiohttp.ClientSession() as http, http.ws_connect(f"http://127.0.0.1:{UI_PORT}/ws", max_msg_size=0) as ws:
        ready, spoke, acked = asyncio.Event(), asyncio.Event(), set()

        async def reader():
            nonlocal audio
            async for m in ws:
                if m.type == aiohttp.WSMsgType.TEXT:
                    e = json.loads(m.data)
                    events.append(e)
                    if e.get("type") == "brain":
                        ready.set()
                elif m.type == aiohttp.WSMsgType.BINARY:
                    audio += len(m.data) - 4
                    spoke.set()
                    utt = int.from_bytes(m.data[:2], "little")
                    if utt not in acked:
                        acked.add(utt)
                        await ws.send_json({"cmd": "play_start", "utt": utt, "delay_ms": 0})

        task = asyncio.create_task(reader())
        await asyncio.wait_for(ready.wait(), 30)
        speech = tone(2.0)
        for i in range(0, len(speech) - FRAME, FRAME):            # a short utterance...
            await ws.send_bytes(speech[i:i + FRAME])
            await asyncio.sleep(0.08)
        for _ in range(60):                                       # ...then silence until the reply has played
            await ws.send_bytes(bytes(FRAME))
            await asyncio.sleep(0.08)
            if spoke.is_set() and audio > 48000:
                break
        await ws.send_json({"cmd": "stop"})
        await asyncio.sleep(0.5)
        task.cancel()
    return events, audio


def main():
    assert os.environ.get("SPEAKRAIL_TOKENIZER_DIR"), "set SPEAKRAIL_TOKENIZER_DIR"
    engines = discover([ROOT / "engines"])
    stubs = [engines[k] for k in ("asr/stub", "tts/stub", "llm/stub")]
    sessions = Path(tempfile.mkdtemp(prefix="speakrail-sessions-"))
    sup = Supervisor()
    harness = None
    try:
        sup.start_all(stubs)
        harness = start_harness(sup, sessions)
        t0 = time.monotonic()
        while not check_health(f"http://127.0.0.1:{UI_PORT}/"):
            assert harness.poll() is None, (sessions / "harness.log").read_text()[-800:]
            assert time.monotonic() - t0 < 30, "harness not up"
            time.sleep(0.2)
        events, audio = asyncio.run(talk())
        kinds = [e.get("type") for e in events]
        errors = [e for e in events if e.get("type") == "error"]
        decisions = [(e.get("why"), e.get("decision")) for e in events if e.get("type") == "decision"]
        words = [e for e in events if e.get("type") == "word"]
        reply = "".join(e.get("text", "") for e in events if e.get("type") == "reply_delta")
        checks = {"no session errors": not errors, "words reached the harness": len(words) >= 2, "the head reached the harness": kinds.count("head") > 10,
                  "the LLM decided to speak": any(d == "speak" for _, d in decisions), "the reply text came through": "stub engine" in reply,
                  "audio came back": audio > 2000}
        print("decisions:", decisions[:8]); print("reply:", reply); print("audio bytes:", audio)
        # an engine dying mid-session is seen by the supervisor
        os.killpg(sup.pids("tts/stub")["server"], signal.SIGKILL)
        t0 = time.monotonic()
        while sup.state("tts/stub") != FAILED and time.monotonic() - t0 < 3:
            time.sleep(0.05)
        checks["a killed engine is noticed"] = sup.state("tts/stub") == FAILED
        for name, ok in checks.items():
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if errors:
            print("errors:", errors[:3])
        if not all(checks.values()):
            print((sessions / "harness.log").read_text()[-1500:])
        return 0 if all(checks.values()) else 1
    finally:
        if harness and harness.poll() is None:
            os.killpg(harness.pid, signal.SIGTERM)
            harness.wait(timeout=10)
        sup.close()


if __name__ == "__main__":
    raise SystemExit(main())
