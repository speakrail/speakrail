"""Fidelity: does this LLM engine make the same decisions as a reference? A fast engine with drifted turn-taking is worse than none.

A reference is a JSON list of {"model", "prompt": [ids], "allowed": [ids], "token": "token_id:N", "top": {token: logprob}}.
Logprobs move a little between engines through bf16 rounding (up to about 1 between two chunk sizes of one engine), so the
pass mark is the decision itself; the logprob differences are reported for a person to read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import aiohttp


@dataclass
class FidelityResult:
    total: int
    same: int
    max_delta: float
    max_delta_likely: float
    differing: list

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.same == self.total


async def decide(http, url, item, salt):
    body = {"model": item["model"], "prompt": item["prompt"], "max_tokens": 1, "temperature": 0, "logprobs": 20, "return_tokens_as_token_ids": True,
            "allowed_token_ids": item["allowed"], "cache_salt": salt}
    async with http.post(url, json=body) as r:
        j = await r.json()
    lp = j["choices"][0]["logprobs"]
    return lp["tokens"][0], lp["top_logprobs"][0]


async def compare(url: str, reference: Path) -> FidelityResult:
    ref = json.loads(Path(reference).read_text())
    same, deltas, likely, differing = 0, [], [], []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as http:
        for i, item in enumerate(ref):
            tok, top = await decide(http, url, item, f"fidelity{i}")
            same += tok == item["token"]
            if tok != item["token"]:
                differing.append({"index": i, "expected": item["token"], "got": tok})
            for t in set(top) & set(item["top"]):
                deltas.append(abs(top[t] - item["top"][t]))
                if item["top"][t] > -5:
                    likely.append(deltas[-1])
    return FidelityResult(len(ref), same, max(deltas, default=0.0), max(likely, default=0.0), differing)


async def record(url: str, prompts: list[dict], out: Path) -> int:
    """prompts: [{"model", "prompt", "allowed"}]; writes the reference from the engine at url"""
    items = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as http:
        for i, item in enumerate(prompts):
            tok, top = await decide(http, url, item, f"record{i}")
            items.append({**item, "token": tok, "top": top})
    Path(out).write_text(json.dumps(items))
    return len(items)
