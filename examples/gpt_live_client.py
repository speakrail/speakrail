#!/usr/bin/env python3
"""Talk to a running Speakrail over its public API: OpenAI's GPT-Live protocol, on a WebSocket at /v1/live/sessions.

    speakrail start                                         # in another terminal, or as a service
    python examples/gpt_live_client.py                      # plays verify/tapes/everyday.wav, saves what Speakrail said
    python examples/gpt_live_client.py --wav question.wav --out answer.wav

The input is a 16 kHz mono 16-bit WAV, sent at real time as a microphone would be; a real app sends its microphone frames the same way.
The script prints both sides of the conversation as they happen, answers a tool call (a calculator that runs here, in the client), and
writes Speakrail's voice to --out. It needs only aiohttp, which Speakrail itself depends on.

The protocol, in the order this script uses it:
    -> session.start               {"session": {"model", "audio": {"format": {"type": "audio/pcm", "rate": 16000 or 24000}},
                                                "delegation": {"type": "responses", "responses": {"tools": [...]}}}}
    <- session.started             (an `info` with code "loading" first, while the engines load)
    -> session.input_audio.append  {"audio": base64 PCM}, in small chunks, at real time
    <- session.input_transcript.delta / session.output_transcript.delta    what each side said, as it is said
    <- session.output_audio.delta  {"delta": base64 PCM}, at the rate asked for, close to real time
    <- response.event              a function_call item: the model wants one of your tools
    -> response.item.create        {"item": {"type": "function_call_output", "call_id", "output": "<json>"}}, then response.create
    -> session.close               <- session.closed
Turn-taking is Speakrail's own: there is no turn-detection setting and no "respond now" button; it speaks when the user is done.
src/speakrail/gateway/live.py lists what is supported, and the Speakrail extensions (session.speakrail).
"""
import argparse
import asyncio
import base64
import json
import os
import random
import re
import struct
import time
import wave
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
RATE = 16000
CHUNK_MS = 80

TOOLS = [{"type": "function", "name": "calculate", "description": "Evaluate an arithmetic expression, like 17 * 23.",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}]


def calculate(expression: str) -> dict:
    if not re.fullmatch(r"[\d\s.+\-*/()]+", expression):
        return {"error": "only numbers and + - * / ( )"}
    return {"result": eval(expression, {"__builtins__": {}})}       # safe: the pattern above allows arithmetic only


def token(arg: str | None) -> str:
    """the API token `speakrail run` made: the server takes no session without it"""
    if arg:
        return arg
    if os.environ.get("SPEAKRAIL_API_TOKEN"):
        return os.environ["SPEAKRAIL_API_TOKEN"]
    f = Path(os.environ.get("SPEAKRAIL_HOME", Path.home() / ".speakrail")) / "config" / "api_token"
    return f.read_text().strip() if f.exists() else ""


async def main(a) -> None:
    with wave.open(a.wav) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            raise SystemExit(f"{a.wav}: needs 16 kHz mono 16-bit")
        pcm = w.readframes(w.getnframes())
    heard = bytearray()
    t0 = time.monotonic()
    stamp = lambda: f"{time.monotonic() - t0:6.2f}s"
    headers = {"Authorization": f"Bearer {token(a.token)}"}
    async with aiohttp.ClientSession() as http, http.ws_connect(a.url, headers=headers, max_msg_size=0) as ws:
        send = lambda **m: ws.send_json(m)
        await send(type="session.start", session={"model": "gpt-live-1", "audio": {"format": {"type": "audio/pcm", "rate": RATE}},
                                                   "delegation": {"type": "responses", "responses": {"tools": TOOLS}}})
        started, closed = asyncio.Event(), asyncio.Event()
        line = {"who": None, "text": "", "at": 0.0}                     # transcripts arrive a word or less at a time: print whole lines

        def say(who: str, delta: str):
            if who != line["who"] or time.monotonic() - line["at"] > 1.5:
                flush()
                line.update(who=who, text="", start=stamp())
            line["text"] += delta
            line["at"] = time.monotonic()

        def flush():
            if line["text"].strip():
                print(f"{line['start']} {line['who']:>9}: {' '.join(line['text'].split())}")
            line["text"], line["who"] = "", None                       # what comes next starts a line with its own time

        async def receive():
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                e = json.loads(msg.data)
                kind = e["type"]
                if kind == "session.started":
                    started.set()
                elif kind == "info":
                    print(f"{stamp()} info: {e.get('message')}")
                elif kind == "error":
                    print(f"{stamp()} error: {e['error']['code']}: {e['error']['message']}")
                    started.set()
                elif kind == "session.input_transcript.delta":
                    say("you", e["delta"])
                elif kind == "session.output_transcript.delta":
                    say("speakrail", e["delta"])
                elif kind == "session.output_audio.delta":
                    heard.extend(base64.b64decode(e["delta"]))
                elif kind == "response.event" and e["event"].get("type") == "response.output_item.done":
                    item = e["event"]["item"]
                    if item.get("type") == "function_call":                        # one of our tools: run it, send the result back
                        result = calculate(**json.loads(item["arguments"])) if item["name"] == "calculate" else {"error": "unknown tool"}
                        flush()
                        print(f"{stamp()} tool call: {item['name']}({item['arguments']}) -> {result}")
                        await send(type="response.item.create", item={"type": "function_call_output", "call_id": item["call_id"], "output": json.dumps(result)})
                        await send(type="response.create")
                elif kind == "session.closed":
                    flush()
                    closed.set()
                    return

        reader = asyncio.create_task(receive())
        await asyncio.wait_for(started.wait(), 600)                  # the first session may load the engines: a minute or two
        step = RATE * 2 * CHUNK_MS // 1000
        rng = random.Random(0)                                        # a quiet room, not digital zeros: the recognizer treats those differently
        silence = struct.pack(f"<{step // 2}h", *(rng.randint(-3, 3) for _ in range(step // 2)))
        chunks = [pcm[i:i + step] for i in range(0, len(pcm), step)] + [silence] * int(a.tail * 1000 / CHUNK_MS)
        t_audio = time.monotonic()
        for n, chunk in enumerate(chunks, 1):                         # real time: a microphone doesn't send faster than it hears
            await send(type="session.input_audio.append", audio=base64.b64encode(chunk).decode())
            await asyncio.sleep(max(0.0, t_audio + n * CHUNK_MS / 1000 - time.monotonic()))
        await send(type="session.close")
        await asyncio.wait_for(closed.wait(), 30)
        reader.cancel()
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(bytes(heard))
    print(f"\nwrote {a.out}: {len(heard) / 2 / RATE:.1f} s of Speakrail's voice")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="ws://127.0.0.1:8080/v1/live/sessions")
    ap.add_argument("--wav", default=str(ROOT / "verify" / "tapes" / "everyday.wav"), help="16 kHz mono 16-bit")
    ap.add_argument("--out", default="speakrail_reply.wav")
    ap.add_argument("--tail", type=float, default=6.0, help="seconds of silence after the input, for the last answer")
    ap.add_argument("--token", help="the API token; by default ~/.speakrail/config/api_token")
    asyncio.run(main(ap.parse_args()))
