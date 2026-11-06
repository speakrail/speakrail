#!/usr/bin/env python3
"""The reference ASR engine. It has no model: words and turn probabilities come from audio energy, but the protocol is real.

WebSocket /v1/transcribe (16 kHz int16 PCM in):
  first message   {"type":"session.created", ...}
  words           {"type":"transcript","text":..,"words":[{"text","start","end"}]}, times in ms
  every 80 ms     {"type":"turn","frame":n,"t":ms,"p":[speaking,complete,incomplete,backchannel,wait,ant80,ant160,ant240]}
  client: {"type":"peek","frames":N,"id":..}  -> {"type":"transcript.peek","id":..,"text":..,"words":[..],"ms":..}, only while audio flows
  client: {"type":"input_audio.end"}          -> {"type":"session.ended"}
"""
import argparse
import array
import asyncio
import json
import math

from aiohttp import WSMsgType, web

FRAME_BYTES, FRAME_MS, WORD_FRAMES = 2560, 80, 5
WORDS = "hello this is the stub speech engine talking".split()
RMS_SPEECH = 300.0
FLAGS = set()


def rms(pcm):
    a = array.array("h")
    a.frombytes(pcm[:len(pcm) & ~1])
    return math.sqrt(sum(x * x for x in a) / len(a)) if len(a) else 0.0


async def transcribe(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    await ws.send_json({"type": "session.created", "session_id": "stub", "delay_ms": 480, "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}})
    buf, frame, voiced, quiet, n_words, peeks = b"", 0, 0, 99, 0, []
    async for m in ws:
        if m.type == WSMsgType.BINARY:
            buf += m.data
            while len(buf) >= FRAME_BYTES:
                pcm, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
                speech = rms(pcm) >= RMS_SPEECH
                quiet = 0 if speech else quiet + 1
                if speech:
                    voiced += 1
                    if voiced % WORD_FRAMES == 0:
                        w = WORDS[n_words % len(WORDS)]
                        n_words += 1
                        t = (frame + 1) * FRAME_MS
                        await ws.send_json({"type": "transcript", "text": " " + w, "words": [{"text": w, "start": t - WORD_FRAMES * FRAME_MS, "end": t}]})
                if speech:
                    p = [0.95, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.02]
                elif "vad_head" in FLAGS:             # only a VAD, as an adapter without a real head would send: never says "complete"
                    p = [0.01, 0.02, 0.95, 0.01, 0.01, 0.02, 0.02, 0.02]
                elif quiet >= 3:
                    p = [0.01, 0.97, 0.01, 0.0, 0.01, 0.9, 0.9, 0.9]
                else:
                    p = [0.3, 0.2, 0.48, 0.01, 0.01, 0.3, 0.3, 0.3]
                await ws.send_json({"type": "turn", "frame": frame, "t": frame * FRAME_MS, "p": p})
                frame += 1
                for ev in peeks:                      # a peek is served between frames, so only while audio keeps coming
                    await ws.send_json({"type": "transcript.peek", "text": "", "words": [], "ms": 5.0, "n_frames": ev.get("frames"), "id": ev.get("id")})
                peeks = []
        elif m.type == WSMsgType.TEXT:
            ev = json.loads(m.data)
            if ev.get("type") == "peek" and "no_peek" not in FLAGS:
                peeks.append(ev)
            elif ev.get("type") == "input_audio.end":
                await ws.send_json({"type": "session.ended", "session_id": "stub"})
                break
        else:
            break
    await ws.close()
    return ws


async def health(_):
    return web.json_response({"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--flags", default="")
    a = ap.parse_args()
    FLAGS.update(f for f in a.flags.split(",") if f)
    app = web.Application()
    app.router.add_get("/v1/transcribe", transcribe)
    app.router.add_get("/health", health)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
