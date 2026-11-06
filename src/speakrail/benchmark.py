"""A short benchmark of the running engines, under a minute at first launch: how much this machine can afford.

The LLM is measured while the recognizer runs in real time, because they share the GPU, and that sharing is what limits a low-bandwidth machine.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp
import websockets

from .contracts.conformance import speech_16k

QUERY = "?encoding=pcm_s16le&sample_rate=16000&channels=1&delay_ms=480&turn_detection=none&client_id=bench&turn_probs=true"
FRAME = 2560
PROMPT = [2, 105] + list(range(1000, 2000))


async def _asr_speed(url, seconds=6.0, peeks=3):
    """frames fed faster than real time: how long one takes. Peeks are measured with audio still flowing."""
    pcm = speech_16k(seconds)
    frames = [pcm[i:i + FRAME] for i in range(0, len(pcm) - FRAME + 1, FRAME)]
    stamps, peek_ms, ended = [], [], asyncio.Event()
    async with websockets.connect(url + QUERY, max_size=None) as ws:
        await ws.recv()

        async def reader():
            async for m in ws:
                e = json.loads(m)
                if e.get("type") == "turn":
                    stamps.append(time.perf_counter())
                elif e.get("type") == "transcript.peek" and e.get("ms") is not None and not e.get("error"):
                    peek_ms.append(float(e["ms"]))
                elif e.get("type") == "session.ended":
                    ended.set()

        task = asyncio.create_task(reader())
        for f in frames:
            await ws.send(f)
        await asyncio.sleep(0)
        t0 = time.monotonic()
        while len(stamps) < len(frames) * 0.9 and time.monotonic() - t0 < 30:
            await asyncio.sleep(0.02)
        for i, f in enumerate(frames[:peeks * 6]):           # now at real time, peeking every half second
            await ws.send(f)
            if i % 6 == 0:
                await ws.send(json.dumps({"type": "peek", "frames": 8, "id": f"bench{i}"}))
            await asyncio.sleep(0.08)
        await ws.send(json.dumps({"type": "input_audio.end"}))
        try:
            await asyncio.wait_for(ended.wait(), 10)
        except asyncio.TimeoutError:
            pass
        task.cancel()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])][:len(frames)]
    frame_ms = statistics.median(gaps) * 1000 if gaps else None
    return frame_ms, (statistics.median(peek_ms) if peek_ms else None)


async def _realtime_feed(url, stop: asyncio.Event, lags: list):
    pcm = speech_16k(8.0)
    async with websockets.connect(url + QUERY, max_size=None) as ws:
        await ws.recv()
        fed = [0]

        async def reader():
            async for m in ws:
                e = json.loads(m)
                if e.get("type") == "turn":
                    lags.append((fed[0] * 80) - e.get("t", 0))

        task = asyncio.create_task(reader())
        i = 0
        start = time.monotonic()
        while not stop.is_set():
            chunk = pcm[i:i + FRAME] if i + FRAME <= len(pcm) else bytes(FRAME)
            await ws.send(chunk)
            fed[0] += 1
            i += FRAME
            await asyncio.sleep(max(0, start + fed[0] * 0.08 - time.monotonic()))
        await ws.send(json.dumps({"type": "input_audio.end"}))
        await asyncio.sleep(0.2)
        task.cancel()


async def _llm_speed(url, lora="speakrail"):
    decide = {"model": lora, "max_tokens": 1, "temperature": 0, "logprobs": 20, "return_tokens_as_token_ids": True, "allowed_token_ids": [106, 16, 17], "cache_salt": "bench"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http:
        async with http.post(url, json={**decide, "prompt": PROMPT}) as r:
            await r.read()
        ms = []
        for k in range(1, 9):
            t0 = time.perf_counter()
            async with http.post(url, json={**decide, "prompt": PROMPT + list(range(2000, 2000 + k))}) as r:
                await r.read()
            ms.append((time.perf_counter() - t0) * 1000)
        body = {"model": lora, "prompt": PROMPT, "max_tokens": 48, "temperature": 0, "stream": True, "ignore_eos": True, "skip_special_tokens": False, "cache_salt": "bench",
                "stream_options": {"include_usage": True}}
        t0, first, n = time.perf_counter(), None, None
        async with http.post(url, json=body) as r:
            async for line in r.content:
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                j = json.loads(line[5:])
                if j.get("usage"):
                    n = j["usage"].get("completion_tokens")
                if first is None and j.get("choices"):
                    first = time.perf_counter()
        total = time.perf_counter() - (first or t0)
    return statistics.median(ms), ((n - 1) / total if n and n > 1 and total > 0 else None)


async def _tts_speed(url, model):
    body = {"model": model, "input": "This is a short test of how fast the speech engine runs on this machine.", "response_format": "pcm", "stream_format": "audio"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as http:
        for _ in range(100):
            t0 = time.perf_counter()
            r = await http.post(url, json=body)
            if r.status != 503:
                break
            r.release()
            await asyncio.sleep(0.1)
        first, size = None, 0
        async for chunk in r.content.iter_any():
            first = first or time.perf_counter()
            size += len(chunk)
        total = time.perf_counter() - t0
    audio_s = size / 2 / 24000
    return ((first - t0) * 1000 if first else None), (audio_s / total if total > 0 else None)


async def run(urls: dict, tts_model: str, capabilities: dict | None = None) -> dict:
    t0 = time.monotonic()
    frame_ms, peek_ms = await _asr_speed(urls["asr"])
    stop, lags = asyncio.Event(), []
    feed = asyncio.create_task(_realtime_feed(urls["asr"], stop, lags))
    await asyncio.sleep(2.5)                                 # a new session's first frames take about a second: that isn't load
    lags.clear()
    try:
        decision_ms, tok_s = await _llm_speed(urls["llm"])
    finally:
        stop.set()
        await feed
    ttfa, speed = await _tts_speed(urls["tts"], tts_model)
    lags.sort()
    return {"asr_frame_ms": frame_ms, "asr_peek_ms": peek_ms, "asr_lag_p90_ms": lags[int(.9 * len(lags))] if lags else None, "llm_decision_ms": decision_ms, "llm_decode_tok_s": tok_s,
            "tts_ttfa_ms": ttfa, "tts_realtime_x": speed, "took_s": round(time.monotonic() - t0, 1), "date": time.strftime("%Y-%m-%d %H:%M")}


def fingerprint(hw, config) -> str:
    return f"{hw.platform}|{hw.name}|{hw.memory_gib:.0f}|{'|'.join(sorted(config.engines.values()))}"


def load(path: Path, key: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except ValueError:
        return None
    return data.get("result") if data.get("fingerprint") == key else None


def save(path: Path, key: str, result: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"fingerprint": key, "result": result}, indent=1) + "\n")
