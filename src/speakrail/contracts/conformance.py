"""Conformance checks: does an engine at this URL follow its role's contract, and can it do what its manifest says?

Everything that applies to every engine of a role must pass. A capability check only runs when the manifest declares the capability.
"""
from __future__ import annotations

import array
import asyncio
import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import numpy as np
import websockets
from scipy.signal import resample_poly

VOICE = Path(__file__).resolve().parent.parent / "harness" / "voices" / "female_a.wav"
PROMPT = [2, 105] + list(range(1000, 1062))           # token ids that exist in any large vocabulary
DECISIONS = [106, 16, 17]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


class Checks:
    def __init__(self):
        self.results: list[Check] = []

    def add(self, name, ok, detail=""):
        self.results.append(Check(name, bool(ok), str(detail)[:300]))
        return ok

    async def run(self, name, coro):
        try:
            await coro
        except Exception as e:
            self.add(name, False, f"{type(e).__name__}: {e}")


async def _post(http, url, body):
    async with http.post(url, json=body) as r:
        text = await r.text()
        try:
            return r.status, json.loads(text)
        except ValueError:
            return r.status, {"raw": text[:200]}


# ---------------------------------------------------------------------- LLM
async def check_llm(url: str, capabilities, lora="speakrail", base="speakrail-base", timeout=120.0) -> list[Check]:
    c = Checks()
    caps = set(capabilities)
    chat_url = url.replace("/v1/completions", "/v1/chat/completions")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as http:
        decide = {"model": lora, "max_tokens": 1, "temperature": 0, "logprobs": 20, "return_tokens_as_token_ids": True}

        async def token_ids():
            st, j = await _post(http, url, {"model": lora, "prompt": PROMPT, "max_tokens": 4, "temperature": 0})
            ch = (j.get("choices") or [{}])[0]
            c.add("completion with token-id prompt", st == 200 and isinstance(ch.get("text"), str) and ch.get("finish_reason") in ("stop", "length")
                  and j.get("usage", {}).get("completion_tokens", 0) >= 1, f"status {st}: {str(j)[:150]}")

        async def allowed():
            for name, allowed_ids in (("idle", DECISIONS), ("overlap", [19, 20, 16])):
                st, j = await _post(http, url, {**decide, "prompt": PROMPT, "allowed_token_ids": allowed_ids})
                lp = j["choices"][0]["logprobs"]
                tok = int(lp["tokens"][0].split(":")[1])
                top = lp["top_logprobs"][0]
                ok = st == 200 and tok in allowed_ids and all(k.startswith("token_id:") and isinstance(v, float) for k, v in top.items())
                c.add(f"decision stays in the allowed set ({name})", ok, f"token {tok}, allowed {allowed_ids}")

        async def eos_decision():
            # the speak token is an end-of-turn token as well; a decision must still report it
            st, j = await _post(http, url, {**decide, "prompt": PROMPT, "allowed_token_ids": [106]})
            tok = j["choices"][0]["logprobs"]["tokens"][0]
            c.add("decision can answer with an end-of-turn token", st == 200 and tok == "token_id:106", tok)

        async def first_token():
            st, j = await _post(http, url, {**decide, "prompt": PROMPT, "skip_special_tokens": False})
            return int(j["choices"][0]["logprobs"]["tokens"][0].split(":")[1])

        async def bias():
            t = await first_token()
            st, j = await _post(http, url, {**decide, "prompt": PROMPT, "logit_bias": {str(t): -100}})
            t2 = int(j["choices"][0]["logprobs"]["tokens"][0].split(":")[1])
            c.add("logit_bias bans a token", st == 200 and t2 != t, f"{t} -> {t2}")

        async def stop():
            t = await first_token()
            st, j = await _post(http, url, {"model": lora, "prompt": PROMPT, "max_tokens": 8, "temperature": 0, "stop_token_ids": [t], "skip_special_tokens": False})
            ch = j["choices"][0]
            c.add("stop_token_ids ends with stop_reason", st == 200 and ch.get("stop_reason") == t and not ch.get("text"), str(ch)[:150])

        async def stream():
            body = {"model": lora, "prompt": PROMPT, "max_tokens": 5, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                    "ignore_eos": True, "skip_special_tokens": False}
            text, finish, usage = "", None, None
            async with http.post(url, json=body) as r:
                status = r.status
                async for line in r.content:
                    line = line.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    j = json.loads(line[5:])
                    usage = j.get("usage") or usage
                    for ch in j.get("choices", []):
                        text += ch.get("text") or ""
                        finish = ch.get("finish_reason") or finish
            c.add("streaming with usage", status == 200 and finish and usage and 1 <= usage.get("completion_tokens", 0) <= 5, f"finish {finish}, usage {usage}, text {text!r}")

        async def errors():
            st, _ = await _post(http, url, {**decide, "model": "no-such-model", "prompt": PROMPT, "allowed_token_ids": DECISIONS})
            c.add("unknown model is an error", 400 <= st < 500, f"status {st}")
            st, _ = await _post(http, url, {"model": lora, "prompt": PROMPT * 5000, "max_tokens": 1, "temperature": 0})   # 320k tokens: past any context a config sets (Gemma 4 stops at 262k)
            c.add("an over-long prompt is an error", 400 <= st < 500, f"status {st}")

        for name, fn in (("completion", token_ids), ("allowed", allowed), ("eos decision", eos_decision), ("logit_bias", bias), ("stop", stop), ("stream", stream), ("errors", errors)):
            await c.run(name, fn())

        async def adapter():
            tops = []
            for m in (lora, base):
                st, j = await _post(http, url, {**decide, "model": m, "prompt": PROMPT, "allowed_token_ids": DECISIONS})
                tops.append(j["choices"][0]["logprobs"]["top_logprobs"][0])
            shared = set(tops[0]) & set(tops[1])
            c.add("capability per_request_adapter: base and LoRA differ", any(abs(tops[0][t] - tops[1][t]) > 1e-3 for t in shared), f"{len(shared)} shared tokens")

        async def chat():
            msgs = [{"role": "user", "content": "Say one short sentence."}]
            st, j = await _post(http, chat_url, {"model": base, "messages": msgs, "temperature": 0, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}})
            content = (j.get("choices") or [{}])[0].get("message", {}).get("content")
            c.add("capability chat", st == 200 and isinstance(content, str) and content.strip(), f"status {st}: {str(j)[:150]}")

        async def cancel():
            body = {"model": lora, "prompt": PROMPT, "max_tokens": 200, "temperature": 0, "stream": True, "ignore_eos": True, "skip_special_tokens": False}
            async with http.post(url, json=body) as r:
                async for line in r.content:
                    if line.startswith(b"data:"):
                        break
            t0 = time.monotonic()                     # leaving the block closed the stream; the engine must be free soon after
            st, _ = await _post(http, url, {**decide, "prompt": PROMPT, "allowed_token_ids": DECISIONS})
            c.add("capability cancel: free after a dropped stream", st == 200 and time.monotonic() - t0 < 10, f"{time.monotonic() - t0:.2f} s")

        async def parallel():
            body = {"model": lora, "prompt": PROMPT, "max_tokens": 12, "temperature": 0, "stream": True, "ignore_eos": True, "skip_special_tokens": False}
            marks = {}

            async def one(k):
                async with http.post(url, json=body) as r:
                    async for line in r.content:
                        if line.startswith(b"data:") and b"[DONE]" not in line:
                            marks.setdefault(k, []).append(time.monotonic())
            await asyncio.gather(one(0), one(1))
            ok = len(marks) == 2 and marks[0][0] < marks[1][-1] and marks[1][0] < marks[0][-1]
            c.add("capability parallel: two streams overlap", ok, {k: len(v) for k, v in marks.items()})

        async def prefix():
            long = [2, 105] + list(range(2000, 2800))
            salt = f"conformance{time.time()}"        # a salt no earlier request used, so the first call is really cold
            t0 = time.monotonic()
            await _post(http, url, {**decide, "prompt": long, "allowed_token_ids": DECISIONS, "cache_salt": salt})
            cold = time.monotonic() - t0
            t0 = time.monotonic()
            await _post(http, url, {**decide, "prompt": long + [3000], "allowed_token_ids": DECISIONS, "cache_salt": salt})
            warm = time.monotonic() - t0
            c.add("capability prefix_cache: a shared prefix is faster", warm < cold, f"cold {cold * 1000:.0f} ms, then {warm * 1000:.0f} ms")

        async def sampling():
            st, j = await _post(http, url, {"model": lora, "prompt": PROMPT, "max_tokens": 4, "temperature": 0.7, "top_p": 0.95, "top_k": 64})
            c.add("capability sampling: temperature, top_p and top_k are accepted", st == 200 and j.get("usage", {}).get("completion_tokens", 0) >= 1, f"status {st}: {str(j)[:150]}")

        for cap, fn in (("per_request_adapter", adapter), ("chat", chat), ("cancel", cancel), ("parallel", parallel), ("prefix_cache", prefix), ("sampling", sampling)):
            if cap in caps:
                await c.run(f"capability {cap}", fn())
    return c.results


# ---------------------------------------------------------------------- ASR
def speech_16k(seconds=8.0) -> bytes:
    with wave.open(str(VOICE)) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(int(seconds * rate)), dtype="<i2").reshape(-1, ch)[:, 0].astype(np.float32)
    y = resample_poly(x, 16000, rate) if rate != 16000 else x
    return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()


async def check_asr(url: str, capabilities, seconds=8.0, frame_s=0.05) -> list[Check]:
    c = Checks()
    caps = set(capabilities)
    pcm = speech_16k(seconds) + bytes(2 * 16000)
    query = "?encoding=pcm_s16le&sample_rate=16000&channels=1&delay_ms=480&turn_detection=none&client_id=conformance&turn_probs=true"
    frames = [pcm[i:i + 2560] for i in range(0, len(pcm) - 2559, 2560)]

    async def session():
        async with websockets.connect(url + query, max_size=None) as ws:
            created = json.loads(await asyncio.wait_for(ws.recv(), 30))
            c.add("session.created comes first", created.get("type") == "session.created", created)
            words, turns, peeks, ended = [], [], [], asyncio.Event()

            async def reader():
                async for m in ws:
                    e = json.loads(m)
                    k = e.get("type")
                    if k == "transcript":
                        words.extend(e.get("words", []))
                    elif k == "turn":
                        turns.append(e)
                    elif k == "transcript.peek":
                        peeks.append(e)
                    elif k == "session.ended":
                        ended.set()
                    elif k == "error":
                        c.add("no error events", False, e)

            task = asyncio.create_task(reader())
            peek_at = len(frames) // 2
            for i, f in enumerate(frames):
                await ws.send(f)
                if i == peek_at and "peek" in caps:   # audio keeps flowing: a decoder may serve peeks only between frames
                    await ws.send(json.dumps({"type": "peek", "frames": 8, "id": "conformance"}))
                await asyncio.sleep(frame_s)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5 and "peek" in caps and not peeks:
                await asyncio.sleep(0.05)
            await ws.send(json.dumps({"type": "input_audio.end"}))
            await asyncio.wait_for(ended.wait(), 30)
            task.cancel()
            c.add("input_audio.end gets session.ended", True)
            if "transcribe" in caps:
                ok = len(words) >= 3 and all(isinstance(w.get("text"), str) and w["end"] >= w["start"] for w in words)
                c.add("capability transcribe: timed words", ok, " ".join(w.get("text", "") for w in words)[:160])
            if "turn_head" in caps:
                ok = len(turns) >= len(frames) * 0.5 and all(isinstance(t.get("p"), list) and len(t["p"]) >= 5 and "t" in t for t in turns)
                c.add("capability turn_head: a frame every 80 ms", ok, f"{len(turns)} turn events for {len(frames)} frames")
            if "anticipation" in caps:
                c.add("capability anticipation: eight outputs", bool(turns) and all(len(t["p"]) >= 8 for t in turns), f"{len(turns[0]['p']) if turns else 0} outputs")
            if "peek" in caps:
                ok = len(peeks) == 1 and peeks[0].get("id") == "conformance" and not peeks[0].get("error") and isinstance(peeks[0].get("ms"), (int, float))
                c.add("capability peek: answered while audio flows", ok, peeks[0] if peeks else "no answer")

    await c.run("ASR session", session())
    return c.results


# ---------------------------------------------------------------------- TTS
TTS_BUSY = (409, 503)                                 # the engine is still releasing a killed request


def speech_request(worker: str, model: str, text: str) -> dict:
    """arguments for one speech POST, in the form this engine's harness worker speaks: Breeze takes a form with a reference voice, the others JSON"""
    if worker == "breeze":
        form = aiohttp.FormData()
        form.add_field("text", text)
        form.add_field("ref_text", VOICE.with_suffix(".txt").read_text().strip())
        form.add_field("seed", "42")
        form.add_field("ref_audio", VOICE.read_bytes(), filename="ref.wav", content_type="audio/wav")
        return {"data": form}
    return {"json": {"model": model, "input": text, "response_format": "pcm", "stream_format": "audio"}}


async def check_tts(url: str, capabilities, model="", worker="pocket") -> list[Check]:
    c = Checks()
    caps = set(capabilities)
    text = "This is a short test of the speech engine."

    async def post_ready(http):
        for _ in range(100):
            r = await http.post(url, **speech_request(worker, model, text))
            if r.status not in TTS_BUSY:
                return r
            r.release()
            await asyncio.sleep(0.1)
        return r

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http:
        async def speech():
            t0 = time.monotonic()
            r = await post_ready(http)
            pcm, first, chunks = bytearray(), None, 0
            async for chunk in r.content.iter_any():
                first = first or time.monotonic() - t0
                pcm += chunk
                chunks += 1
            total = time.monotonic() - t0
            c.add("speech returns PCM", r.status == 200 and len(pcm) > 4800 and len(pcm) % 2 == 0, f"status {r.status}, {len(pcm)} bytes")
            a = array.array("h")
            a.frombytes(bytes(pcm[:len(pcm) & ~1]))
            peak = max((abs(x) for x in a), default=0)
            c.add("speech isn't silent", peak > 300, f"peak {peak}")
            if "stream" in caps:
                c.add("capability stream: audio before the end", first is not None and chunks > 1 and first < total * 0.8, f"first chunk {first:.2f} s of {total:.2f} s, {chunks} chunks")

        async def cancel():
            r = await post_ready(http)
            async for _ in r.content.iter_any():
                break
            r.close()
            t0 = time.monotonic()
            r2 = await post_ready(http)
            await r2.content.read(1000)
            r2.close()
            c.add("capability cancel: free after a dropped request", r2.status == 200 and time.monotonic() - t0 < 10, f"{time.monotonic() - t0:.2f} s")

        await c.run("speech", speech())
        if "cancel" in caps:
            await c.run("capability cancel", cancel())
    return c.results


async def check(engine, url: str) -> list[Check]:
    if engine.role == "llm":
        return await check_llm(url, engine.capabilities)
    if engine.role == "asr":
        return await check_asr(url, engine.capabilities)
    return await check_tts(url, engine.capabilities, engine.harness.get("model", ""), engine.harness.get("worker", "pocket"))
