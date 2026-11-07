#!/usr/bin/env python3
"""Record a decision reference for `speakrail engine fidelity` and `speakrail verify`, from a config whose LLM engine is the reference.

    python verify/record_fidelity.py nvidia-12b-24gb [--items 120]

It starts the config as `speakrail run` does, puts a recording proxy in front of its LLM, and plays the test tapes through a second
harness that talks to the proxy: so the prompts are exactly what the harness sends in those conversations (contexts, allowed decision
sets), not made up. Then it asks the LLM engine itself for each decision, the way the fidelity check will (fidelity.record), keeps a
mix of them (every decision that isn't listen, listens to fill up), checks the engine against what it just wrote, and stops.

The reference is verify/fidelity/<config>.json: {"meta": {...the model files it came from...}, "items": [...]}. Another engine or
another machine running the same model files should make the same decisions; `verify` checks that when a reference's model files are
all in the config it verifies.
"""
import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]

import aiohttp
from aiohttp import web

from speakrail import verify
from speakrail.config import CFG
from speakrail.contracts.fidelity import compare, record
from speakrail.resolver import discover_configs

OUT = ROOT / "verify" / "fidelity"


async def proxy(real_url: str, captured: list, port: int) -> web.AppRunner:
    """forwards everything to the real LLM, streams included; keeps the decision requests (one token from an allowed set)"""
    base = real_url.split("/v1/")[0]
    http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    async def forward(request: web.Request):
        body = await request.read()
        try:
            j = json.loads(body) if body else {}
        except ValueError:
            j = {}
        if j.get("allowed_token_ids") and j.get("max_tokens") == 1:
            captured.append({"model": j["model"], "prompt": j["prompt"], "allowed": j["allowed_token_ids"]})
        async with http.request(request.method, base + request.path_qs, data=body, headers={"Content-Type": "application/json"}) as r:
            out = web.StreamResponse(status=r.status, headers={"Content-Type": r.headers.get("Content-Type", "application/json")})
            await out.prepare(request)
            try:
                async for chunk in r.content.iter_any():
                    await out.write(chunk)
                await out.write_eof()
            except (ConnectionResetError, aiohttp.ClientConnectionResetError):
                pass                                  # the harness cancelled the reply (an interruption): leaving here closes the upstream too
            return out

    app = web.Application(client_max_size=1 << 28)
    app.router.add_route("*", "/{tail:.*}", forward)
    app.on_cleanup.append(lambda _: http.close())
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def pick(items: list, n: int, seed: int = 0) -> list:
    """every decision other than the most common one (listen), then that one to fill up to n, spread over the conversation"""
    rng = random.Random(seed)
    counts = {}
    for it in items:
        counts[it["token"]] = counts.get(it["token"], 0) + 1
    common = max(counts, key=counts.get)
    rare = [it for it in items if it["token"] != common]
    rest = [it for it in items if it["token"] == common]
    keep = rare[:n] + rng.sample(rest, max(0, min(len(rest), n - len(rare))))
    return sorted(keep, key=items.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--items", type=int, default=120)
    a = ap.parse_args()
    config = discover_configs(home=CFG.home)[a.config]
    work = Path(CFG.home) / "verify" / f"record-{a.config}-{time.strftime('%Y%m%d-%H%M%S')}"
    work.mkdir(parents=True)
    stack = harness = None
    try:
        print(f"starting {a.config}...", flush=True)
        stack = verify.start_stack(argparse.Namespace(config=a.config, force=False), work, verify._free_port())
        conf = json.loads((Path(CFG.home) / "config" / "resolved.json").read_text())
        real = conf["llm_url"]
        captured, pport, hport = [], verify._free_port(), verify._free_port()
        loop = asyncio.new_event_loop()
        runner = loop.run_until_complete(proxy(real, captured, pport))
        conf["llm_url"] = real.replace(real.split("/v1/")[0], f"http://127.0.0.1:{pport}")
        (work / "conf.json").write_text(json.dumps(conf))
        env = {**os.environ, "SPEAKRAIL_CONFIG": str(work / "conf.json"), "UI_PORT": str(hport), "SESSIONS_DIR": str(work / "sessions"),
               "PYTHONPATH": str(ROOT / "src")}
        harness = subprocess.Popen([sys.executable, "-u", "-m", "speakrail.harness.server"], env=env, cwd=ROOT, stdout=open(work / "harness.log", "wb"),
                                   stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(120):
            try:
                loop.run_until_complete(asyncio.wait_for(asyncio.open_connection("127.0.0.1", hport), 1))
                break
            except (OSError, asyncio.TimeoutError):
                time.sleep(1)
        for wav in sorted(verify.TAPES.glob("*.wav")):
            print(f"playing {wav.stem}...", flush=True)
            loop.run_until_complete(verify.play(hport, wav))
        harness.terminate(); harness.wait(30); harness = None
        loop.run_until_complete(runner.cleanup())
        print(f"{len(captured)} decisions captured; asking the engine for each...", flush=True)
        everything = work / "all.json"
        loop.run_until_complete(record(real, captured, everything))
        items = pick(json.loads(everything.read_text()), a.items)
        # the files that decide: the Gemma weights and the Speakrail LoRA (with its token rows). A config with all of them uses this reference
        meta = {"config": a.config, "recorded": time.strftime("%Y-%m-%d"), "engine": config.engines["llm"],
                "models": sorted({f"{d['repo']}@{d['revision']}" for d in config.downloads if "gemma" in d["repo"].lower()})}
        OUT.mkdir(exist_ok=True)
        out = OUT / f"{a.config}.json"
        out.write_text(json.dumps({"meta": meta, "items": items}) + "\n")
        r = loop.run_until_complete(compare(real, out))
        tokens = {}
        for it in items:
            tokens[it["token"]] = tokens.get(it["token"], 0) + 1
        print(f"wrote {out}: {len(items)} decisions {tokens}; the engine against it: {r.same} of {r.total} the same, largest logprob difference {r.max_delta:.3f}")
        return 0 if r.ok else 1
    finally:
        if harness is not None:
            harness.terminate()
        if stack is not None:
            verify.stop_stack(stack)


if __name__ == "__main__":
    raise SystemExit(main())
