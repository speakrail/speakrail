"""The two HTTP calls the harness makes to vLLM.

  decide(ids, allowed, bias)  one decision: prefill the context, pick one of `allowed` token ids.
      /v1/completions, max_tokens 1, allowed_token_ids + logit_bias server side, top-20 raw logprobs
      (return_tokens_as_token_ids) for the probs.
  stream(prompt, ...)        one streamed reply: yields text deltas; .stop is the stop token id that ended it
      (`stop_reason`), .n_tokens the completion tokens.
      /v1/completions with token-id prompts, stop_token_ids, skip_special_tokens false, logit_bias. The adapter is
      picked by `model`: the LoRA name, or the base model's served name when lora is None.
"""
from __future__ import annotations

import json
import math
import time


class Engine:
    def __init__(self, base: str = "http://127.0.0.1:8001", lora: str | None = "speakrail", served: str = "speakrail-base"):
        self.base, self.lora, self.served = base.rstrip("/"), lora, served
        self.url = self.base + "/v1/completions"
        self.stop = None; self.n_tokens = None; self.cached = None

    @property
    def model(self):
        return self.served if self.lora is None else self.lora

    async def _post(self, http, url, body):
        async with http.post(url, json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"vllm {r.status}: {(await r.text())[:300]}")
            return await r.json()

    async def warm(self, http, ids):
        """prefill a context into the prefix cache (session start: the system prompt)"""
        await self._post(http, self.url, {"model": self.model, "prompt": list(ids), "max_tokens": 1, "temperature": 0.0})

    async def decide(self, http, ids, allowed, bias=None, cache_salt=None):
        """-> (decision id, {id: normalized prob over allowed}, {id: raw logprob}, ms)"""
        bias = {int(k): v for k, v in (bias or {}).items() if int(k) in allowed}
        t0 = time.perf_counter()
        body = {"model": self.model, "prompt": list(ids), "max_tokens": 1, "temperature": 0.0,
                "allowed_token_ids": list(allowed), "logprobs": 20, "return_tokens_as_token_ids": True}
        if bias: body["logit_bias"] = {str(k): v for k, v in bias.items()}
        if cache_salt: body["cache_salt"] = cache_salt
        j = await self._post(http, self.url, body)
        ms = (time.perf_counter() - t0) * 1000
        lp = j["choices"][0].get("logprobs") or {}
        tok = int(lp["tokens"][0].split(":")[1]) if lp.get("tokens") else None
        top = {int(k.split(":")[1]): v for k, v in (lp.get("top_logprobs") or [{}])[0].items()}
        raw = {i: top[i] for i in allowed if i in top}
        dec = tok if tok in allowed else max(allowed, key=lambda i: top.get(i, -1e9))
        self.cached = None
        z = sum(math.exp(v) for v in raw.values()) or 1.0
        probs = {i: math.exp(v) / z for i, v in raw.items()}
        return dec, probs, raw, ms

    def stream(self, http, prompt, max_tokens, stop_ids, logit_bias=None, ignore_eos=False, cache_salt=None):
        """-> a Stream: `async for delta in it`; afterwards it.stop / it.n_tokens (also copied to self.stop / self.n_tokens)"""
        body = {"model": self.model, "prompt": list(prompt), "max_tokens": max_tokens, "temperature": 0.0,
                "stop_token_ids": list(stop_ids), "skip_special_tokens": False, "stream": True,
                "stream_options": {"include_usage": True}}
        if logit_bias: body["logit_bias"] = {str(k): v for k, v in logit_bias.items()}
        if ignore_eos: body["ignore_eos"] = True
        if cache_salt: body["cache_salt"] = cache_salt
        return Stream(self, http, body)


class Stream:
    """one streamed completion (per call state, so concurrent replies don't share it)"""
    def __init__(self, eng, http, body):
        self.eng, self.http, self.body = eng, http, body
        self.stop = None; self.n_tokens = None

    async def __aiter__(self):
        e = self.eng
        async with self.http.post(e.url, json=self.body) as r:
            if r.status != 200:
                raise RuntimeError(f"vllm {r.status}: {(await r.text())[:300]}")
            async for line in r.content:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                j = json.loads(line[5:])
                if j.get("usage"): self.n_tokens = j["usage"].get("completion_tokens")
                for c in j.get("choices", []):
                    if c.get("finish_reason"):
                        self.stop = c.get("stop_reason")
                    d = c.get("text") or ""
                    if d:
                        yield d
        e.stop, e.n_tokens = self.stop, self.n_tokens
