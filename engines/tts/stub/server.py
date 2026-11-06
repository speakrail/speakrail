#!/usr/bin/env python3
"""The reference TTS engine. It has no model: it streams a tone for as long as the text would take to say.

POST /v1/audio/speech  {"model":..,"input":"text","response_format":"pcm","stream_format":"audio"}
  -> 24 kHz mono int16 PCM over chunked transfer. Closing the connection stops generation.
  -> 503 while the engine is busy with another request; the harness retries.
"""
import argparse
import array
import asyncio
import math

from aiohttp import web

RATE, CHUNK_S, SECONDS_PER_CHAR = 24000, 0.08, 0.06
FLAGS = set()


def tone(start, n, freq=220.0):
    return array.array("h", (int(8000 * math.sin(2 * math.pi * freq * (start + i) / RATE)) for i in range(n))).tobytes()


async def speech(request):
    b = await request.json()
    text = b.get("input", "")
    if not text:
        return web.json_response({"error": {"message": "empty input"}}, status=400)
    resp = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
    resp.enable_chunked_encoding()
    await resp.prepare(request)
    total, sent, chunk = int(len(text) * SECONDS_PER_CHAR * RATE), 0, int(CHUNK_S * RATE)
    while sent < total:
        n = min(chunk, total - sent)
        await resp.write(bytes(n * 2) if "silent" in FLAGS else tone(sent, n))
        sent += n
        await asyncio.sleep(CHUNK_S / 4)              # faster than real time, as a healthy engine is
    await resp.write_eof()
    return resp


async def health(_):
    return web.json_response({"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--flags", default="")
    a = ap.parse_args()
    FLAGS.update(f for f in a.flags.split(",") if f)
    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    app.router.add_get("/health", health)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
