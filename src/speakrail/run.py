"""`speakrail run`: from "what is this machine" to a running assistant.

Detect, resolve, start the engines, benchmark them if this machine hasn't been measured, render the phrase cache if it's
missing, then start the harness. Any engine or the harness dying stops everything, and the exit code says so.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import benchmark as bench
from .config import CFG, REPO_ROOT
from .hardware import detect
from .manifest import discover
from .resolver import ResolveError, discover_configs, explain, resolve, to_file
from .supervisor import EngineError, Supervisor, check_health, port_free


def paths():
    home = Path(CFG.home)
    return home, Path(CFG.models_dir or home / "models")


def setup(args, benchmark=None):
    """-> (resolved, configs, engines). Raises ResolveError if nothing fits."""
    from .resolver import parse_set
    hw = detect()
    if hw is None:
        raise ResolveError(f"No supported hardware found. Known platforms are in {REPO_ROOT / 'platforms'}: Apple Silicon, and NVIDIA through nvidia-smi.")
    home, models = paths()
    configs, engines = discover_configs(home=home), discover()
    params = {}
    for item in getattr(args, "param", None) or []:
        target, _, rest = item.partition(":")
        key, eq, val = rest.partition("=")
        if not eq or not target:
            raise ResolveError(f"--param wants engine:key=value, like asr/audiocpp:threads=2; got {item!r}")
        params.setdefault(target, {})[key] = val
    chosen = resolve(configs, engines, hw, pinned=getattr(args, "config", None), sets=parse_set(getattr(args, "set", None)), params=params, benchmark=benchmark,
                     force=getattr(args, "force", False), models=models, home=home)
    return chosen, configs, engines


def stored_benchmark(resolved):
    home, _ = paths()
    return bench.load(home / "config" / "benchmark.json", bench.fingerprint(resolved.hardware, resolved.config))


def say(msg):
    print(msg, flush=True)


def render_phrase_cache(harness_values, env, url):
    """pre-rendered first clauses, once per voice. Without them replies still work, only their first clause starts later."""
    t0 = time.monotonic()
    cmd = [sys.executable, "-m", "speakrail.harness.phrase_cache", "render", "--engine", harness_values["tts_engine"], "--voice", harness_values.get("voice", ""), "--urls", url]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    last = (out.stdout.strip().splitlines() or [""])[-1]
    if out.returncode != 0:
        say(f"Phrase cache not rendered, replies will start without it: {(out.stderr.strip().splitlines() or [''])[-1][:160]}")
    else:
        say(f"Phrase cache ready in {time.monotonic() - t0:.0f} s: {last[:120]}")


def main(args) -> int:
    try:
        resolved, _, _ = setup(args)
        stored = stored_benchmark(resolved)
        if stored:
            resolved, _, _ = setup(args, benchmark=stored)
    except (ResolveError, ValueError) as e:
        say(str(e))
        return 2
    home, _ = paths()
    say(explain(resolved))
    if not port_free(CFG.ui_port):
        say(f"Port {CFG.ui_port} is already in use. Pick another with UI_PORT, or stop what's using it.")
        return 2
    failed = threading.Event()
    stop = threading.Event()
    reason = []

    def on_event(engine, state, detail):
        say(f"[{engine}] {state}{': ' + detail if detail else ''}")
        if state == "failed":
            reason.append(f"{engine}: {detail}")
            failed.set()

    sup = Supervisor(on_event=on_event)
    harness = None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        engines = list(resolved.engines.values())
        sup.start_all(engines, params={e.id: resolved.params.get(e.id, {}) for e in engines})
        urls = {role: sup.url(e.id) for role, e in resolved.engines.items()}
        if not stored and not args.no_bench:
            say("Measuring this machine, under a minute...")
            try:
                result = asyncio.run(bench.run(urls, resolved.engines["tts"].harness.get("model", "")))
                bench.save(home / "config" / "benchmark.json", bench.fingerprint(resolved.hardware, resolved.config), result)
                before = dict(resolved.harness)
                resolved, _, _ = setup(args, benchmark=result)
                changed = [d for d in resolved.decisions if before.get(d.setting) != resolved.harness.get(d.setting)]
                say(f"Measured: ASR {result['asr_frame_ms']:.0f} ms per frame, peek {result['asr_peek_ms']:.0f} ms, LLM {result['llm_decode_tok_s']:.0f} tok/s, first audio {result['tts_ttfa_ms']:.0f} ms ({result['took_s']:.0f} s)")
                for d in changed:
                    say(f"Changed: {d.setting} = {d.value}: {d.why}")
                for w in resolved.warnings:
                    say(f"Warning: {w}")
            except Exception as e:                   # a benchmark that can't finish shouldn't stop someone talking to it
                say(f"Benchmark failed, keeping the config's settings: {type(e).__name__}: {e}")
        conf = home / "config" / "resolved.json"
        to_file(resolved, urls, conf)
        token_file = home / "config" / "api_token"
        if not token_file.exists():
            token_file.touch(mode=0o600)
            token_file.write_text(secrets.token_urlsafe(32) + "\n")
        env = {**os.environ, "SPEAKRAIL_CONFIG": str(conf), "SPEAKRAIL_API_TOKEN": token_file.read_text().strip(),
               "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]))}
        if resolved.config.phrase_cache and resolved.harness.get("phrase_cache_dir") and not _phrases_ready(env, resolved):
            say("Rendering the phrase cache, once per voice...")
            render_phrase_cache(resolved.harness, env, urls["tts"])
        log_dir = home / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "speakrail.harness.server"] + ([] if resolved.config.phrase_cache else ["--no-phrase-cache"])
        with open(log_dir / "harness.log", "ab", buffering=0) as log:
            harness = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        url = f"http://{CFG.ui_host}:{CFG.ui_port}/"
        t0 = time.monotonic()
        while not check_health(url):
            if harness.poll() is not None:
                say(f"The harness exited with code {harness.returncode}; see {log_dir / 'harness.log'}")
                return 1
            if stop.is_set():
                return 0
            if time.monotonic() - t0 > 120:
                say("The harness didn't come up in 120 s")
                return 1
            time.sleep(0.2)
        say(f"\nSpeakrail is ready: {url}\nAPI (GPT-Live over WebSocket): ws://{CFG.ui_host}:{CFG.ui_port}/v1/live/sessions, with the token in {token_file}\nCtrl-C stops everything.")
        while not stop.is_set() and not failed.is_set():
            if harness.poll() is not None:
                reason.append(f"the harness exited with code {harness.returncode}; see {log_dir / 'harness.log'}")
                failed.set()
                break
            stop.wait(0.5)
        if failed.is_set():
            say(f"Stopping: {'; '.join(reason)}")
            return 1
        return 0
    except EngineError as e:
        say(str(e))
        return 1
    finally:
        if harness and harness.poll() is None:
            try:
                os.killpg(harness.pid, signal.SIGTERM)
                harness.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(harness.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        sup.close()
        say("Stopped.")


def _phrases_ready(env, resolved) -> bool:
    cmd = [sys.executable, "-m", "speakrail.harness.phrase_cache", "status", "--engine", resolved.harness["tts_engine"], "--voice", resolved.harness.get("voice", "")]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    text = out.stdout.strip()
    if out.returncode != 0 or ":" not in text:
        return False
    done = text.rsplit(":", 1)[1].strip().split(" ")[0]         # "/path: 30/30 phrases rendered"
    a, _, b = done.partition("/")
    return a.isdigit() and a == b and int(b) > 0


def bench_main(args) -> int:
    """measure the engines of the chosen config again"""
    try:
        resolved, _, _ = setup(args)
    except (ResolveError, ValueError) as e:
        say(str(e))
        return 2
    home, _ = paths()
    sup = Supervisor(on_event=lambda e, s, d: say(f"[{e}] {s}{': ' + d if d else ''}"))
    try:
        engines = list(resolved.engines.values())
        sup.start_all(engines, params={e.id: resolved.params.get(e.id, {}) for e in engines})
        urls = {role: sup.url(e.id) for role, e in resolved.engines.items()}
        result = asyncio.run(bench.run(urls, resolved.engines["tts"].harness.get("model", "")))
    except EngineError as e:
        say(str(e))
        return 1
    finally:
        sup.close()
    bench.save(home / "config" / "benchmark.json", bench.fingerprint(resolved.hardware, resolved.config), result)
    say(json.dumps(result, indent=1))
    return 0
