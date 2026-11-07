"""`speakrail verify`: does this machine run a config properly?

For anyone who wants to know, and for a pull request that marks a config `tested`: it starts the config, checks every engine against
its role's contract, plays the synthetic tapes in verify/tapes/ through the whole stack at real time, the way a browser would, and
checks what the assistant did against each tape's labels: a reply where the user finished, none in a pause mid-sentence, no stopping
for a backchannel, stopping for an interruption, the right tool, and an answer that says what it should (17 x 23 has to come back as
391). Pass or fail is about that behaviour. Latency and memory depend on the hardware, so they are reported, never judged.

The report (report.md and report.json, with this machine's details and the exact versions) goes with the pull request.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

from . import __version__
from .config import CFG, REPO_ROOT
from .ui import say

TAPES = REPO_ROOT / "verify" / "tapes"
REFERENCES = REPO_ROOT / "verify" / "fidelity"
QUICK = ["everyday"]
FRAME = 2560                    # 80 ms of 16 kHz int16, the browser's chunk
TAIL_S = 8.0                    # quiet after the tape, for the last reply
START_TIMEOUT_S = 1800
EARLY_S = 0.3                   # a reply may start this much before a label's time: labels are cut at the last loud sample
STOP_WITHIN_S = 3.0             # an interruption should stop the reply within this
BACKCHANNEL_S = 2.0             # a backchannel must not stop the reply within this


# ---------------------------------------------------------------------- this machine
def _out(cmd: list[str], timeout: float = 10) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def system_info() -> dict:
    from .hardware import detect
    info = {"os": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    try:
        info["ram_gib"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        info["macos"] = _out(["sw_vers", "-productVersion"])
        info["cpu"] = _out(["sysctl", "-n", "machdep.cpu.brand_string"])
    else:
        try:
            info["cpu"] = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
        except (OSError, StopIteration):
            info["cpu"] = platform.processor()
        try:
            info["wsl"] = "microsoft" in open("/proc/version").read().lower()
        except OSError:
            pass
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else None)
    if smi:
        info["gpus"] = [dict(zip(("name", "memory_mib", "driver"), (x.strip() for x in line.split(","))))
                        for line in _out([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]).splitlines() if line]
        m = re.search(r"CUDA Version:\s*([\d.]+)", _out([smi]))
        info["cuda"] = m[1] if m else ""
    if shutil.which("docker"):
        info["docker"] = _out(["docker", "version", "--format", "{{.Server.Version}}"])
        if shutil.which("nvidia-ctk"):
            info["nvidia_container_toolkit"] = (_out(["nvidia-ctk", "--version"]).splitlines() or [""])[0]
    hw = detect()
    info["speakrail_sees"] = hw.describe() if hw else "no supported hardware"
    return info


def install_info(resolved) -> dict:
    from .lifecycle import app_commit
    from .manifest import render, values_for
    from .run import paths
    from .supervisor import images
    home, models = paths()
    commit = app_commit(REPO_ROOT)
    dirty = bool(_out(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"])) if commit else None
    c = resolved.config
    engines = {}
    for role, e in resolved.engines.items():
        manifest = e.dir / "engine.toml"
        entry = {"id": e.id, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()[:16], "params": resolved.params.get(e.id, {})}
        imgs = sorted(images(e, home, resolved.params.get(e.id)))
        if imgs:
            entry["images"] = {i: _out(["docker", "image", "inspect", "--format", "{{.Id}}", i])[:19] or "not on this machine" for i in imgs}
        engines[role] = entry
    return {"speakrail": __version__, "commit": commit, "uncommitted_changes": dirty,
            "config": c.name, "config_status": c.status, "config_sha256": hashlib.sha256(c.path.read_bytes()).hexdigest()[:16] if c.path else None,
            "engines": engines, "models": [f"{d['repo']}@{d['revision'][:12]}" + (f" ({d['from']})" if d.get("from") else "") for d in c.downloads]}


# ---------------------------------------------------------------------- the stack
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Memory:
    """the machine's used GPU (or unified) memory every two seconds, as Speakrail itself measures it, from before the start to the end"""
    def __init__(self):
        from .hardware import detect
        self.detect, self.baseline, self.peak, self._stop = detect, None, 0.0, threading.Event()
        hw = detect()
        if hw:
            self.baseline = hw.memory_gib - hw.free_memory_gib
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.wait(2.0):
            hw = self.detect()
            if hw:
                self.peak = max(self.peak, hw.memory_gib - hw.free_memory_gib)

    def stop(self) -> dict:
        self._stop.set()
        if self.baseline is None:
            return {}
        return {"before_gib": round(self.baseline, 2), "peak_gib": round(self.peak, 2), "peak_above_before_gib": round(max(0.0, self.peak - self.baseline), 2)}


def start_stack(a, out: Path, port: int) -> subprocess.Popen:
    from .service import READY_LINE
    cmd = [sys.executable, "-u", "-m", "speakrail", "run"] + (["--config", a.config] if a.config else []) + (["--force"] if a.force else [])
    env = {**os.environ, "UI_PORT": str(port), "SESSIONS_DIR": str(out / "sessions"), "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]))}
    log = open(out / "run.log", "wb", buffering=0)
    p = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < START_TIMEOUT_S:
        text = (out / "run.log").read_text(errors="replace")
        if READY_LINE in text:
            return p
        if p.poll() is not None:
            raise RuntimeError("Speakrail stopped before it was ready:\n" + "\n".join(text.splitlines()[-15:]))
        time.sleep(1)
    raise RuntimeError(f"Speakrail wasn't ready after {START_TIMEOUT_S // 60} minutes; the log is {out / 'run.log'}")


def stop_stack(p: subprocess.Popen) -> None:
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


async def conformance(resolved) -> list[dict]:
    """each engine against its role's contract, while the stack is up and before any session holds the recognizer"""
    from .contracts.conformance import check
    from .manifest import render, values_for
    from .run import paths
    home, models = paths()
    out = []
    for role, e in resolved.engines.items():
        url = render(e.url, values_for(e, home, models, home / "run" / e.slug, resolved.params.get(e.id)), e.id)
        try:
            checks = await check(e, url)
        except Exception as err:                       # an engine that breaks the checker fails, it doesn't stop the report
            checks = []
            out.append({"engine": e.id, "check": "conformance", "ok": False, "detail": f"{type(err).__name__}: {err}"})
        out += [{"engine": e.id, "check": c.name, "ok": c.ok, "detail": c.detail} for c in checks]
    return out


def reference_for(config) -> Path | None:
    """the decision reference for this config's model: the one its [fidelity] names (a port of a model to other files, a GGUF for
    llama.cpp say), or else the one recorded from exactly the model files it downloads. None: not checked."""
    named = (getattr(config, "fidelity", None) or {}).get("reference")
    if named:
        return REFERENCES / f"{named}.json"
    have = {f"{d['repo']}@{d['revision']}" for d in config.downloads}
    for f in sorted(REFERENCES.glob("*.json")):
        meta = json.loads(f.read_text()).get("meta", {})
        if meta.get("models") and set(meta["models"]) <= have:
            return f
    return None


async def fidelity(resolved) -> list[dict]:
    """the LLM engine's decisions against the reference for its model: the same model should decide the same, on any engine and GPU"""
    from .contracts.fidelity import AGREE, compare
    from .manifest import render, values_for
    from .run import paths
    ref = reference_for(resolved.config)
    e = resolved.engines["llm"]
    if ref is None:
        return [{"engine": e.id, "check": "fidelity", "ok": True, "detail": "no reference for this model (a config can name one in [fidelity]); not checked"}]
    if not ref.exists():
        return [{"engine": e.id, "check": "fidelity", "ok": False, "detail": f"[fidelity] names {ref.stem}, and there is no {ref}"}]
    agree = float(resolved.config.fidelity.get("agree", AGREE))
    home, models = paths()
    url = render(e.url, values_for(e, home, models, home / "run" / e.slug, resolved.params.get(e.id)), e.id)
    r = await compare(url, ref, agree)
    return [{"engine": e.id, "check": f"fidelity: the same decisions as {ref.stem}", "ok": r.ok,
             "detail": f"{r.same} of {r.total} the same (pass: {agree:.0%}); largest logprob difference {r.max_delta:.2f}, {r.max_delta_likely:.2f} where likely"}]


# ---------------------------------------------------------------------- one tape, as a browser plays it
async def play(port: int, wav: Path) -> dict:
    """the tape at real time into /ws; every event and every reply's audio stamped with the tape time it arrived at"""
    import aiohttp
    import numpy as np
    with wave.open(str(wav)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{wav.name}: tapes are 16 kHz mono 16-bit")
        raw = w.readframes(w.getnframes())
    events, audio, clock = [], {}, {"t0": None}
    now = lambda: time.monotonic() - clock["t0"] if clock["t0"] is not None else -1.0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http, http.ws_connect(f"http://127.0.0.1:{port}/ws", max_msg_size=0) as ws:
        ready = asyncio.Event()

        async def reader():
            async for m in ws:
                if m.type == aiohttp.WSMsgType.TEXT:
                    e = json.loads(m.data)
                    e["_t"] = now()
                    events.append(e)
                    if e.get("type") == "brain":
                        ready.set()
                elif m.type == aiohttp.WSMsgType.BINARY and len(m.data) > 4:
                    utt = int.from_bytes(m.data[:2], "little")
                    t, n = now(), (len(m.data) - 4) // 2
                    if utt not in audio:
                        audio[utt] = {"first": t, "last": t, "samples": 0}
                        await ws.send_json({"cmd": "play_start", "utt": utt, "delay_ms": 0})
                    audio[utt]["last"] = t + n / 24000
                    audio[utt]["samples"] += n
        task = asyncio.create_task(reader())
        await asyncio.wait_for(ready.wait(), 120)
        rng = np.random.default_rng(0)
        clock["t0"] = t0 = time.monotonic()
        n = 0
        for i in range(0, len(raw) - FRAME + 1, FRAME):
            await ws.send_bytes(raw[i:i + FRAME]); n += 1
            await asyncio.sleep(max(0, t0 + n * 0.08 - time.monotonic()))
        for _ in range(int(TAIL_S / 0.08)):           # a quiet room after the tape, not digital zeros
            await ws.send_bytes(rng.normal(0, 3, FRAME // 2).astype("<i2").tobytes()); n += 1
            await asyncio.sleep(max(0, t0 + n * 0.08 - time.monotonic()))
        await ws.send_json({"cmd": "stop"})
        await asyncio.sleep(1.5)
        task.cancel()
    return {"events": events, "audio": audio}


# ---------------------------------------------------------------------- what the assistant did, against the labels
def replies_of(rec: dict) -> list[dict]:
    """every reply the user heard: when its audio started and stopped playing on the tape, and its text. Audio arrives faster than it
    plays, so a reply plays until its start plus its length, or until it was stopped. The text is what the browser got: the reply's
    deltas, joined to its audio through reply_start's turn."""
    ev = rec["events"]
    turn_of = {e["utt"]: e.get("turn") for e in ev if e.get("type") == "reply_start" and "utt" in e}
    text: dict = {}
    for e in ev:
        if e.get("type") == "reply_delta":
            text[e.get("turn")] = text.get(e.get("turn"), "") + e.get("text", "")
    out = []
    for u, a in rec["audio"].items():
        audio = {**a, "utt": int(u)} if isinstance(u, str) else {**a, "utt": u}
        if audio["samples"] <= 2400:
            continue
        turn = turn_of.get(audio["utt"])
        end = audio["first"] + audio["samples"] / 24000
        stopped = [e["_t"] for e in ev if e.get("type") == "stop_audio" and e.get("turn") == turn and e["_t"] > audio["first"]]
        if stopped:
            end = min(end, min(stopped))
        out.append({"utt": audio["utt"], "turn": turn, "start": round(audio["first"], 2), "end": round(end, 2), "text": text.get(turn, "")})
    return sorted(out, key=lambda r: r["start"])


def judge(tape: dict, rec: dict) -> list[dict]:
    replies = replies_of(rec)
    tools = [(e["_t"], e.get("name")) for e in rec["events"] if e.get("type") == "tool"]
    stops = [(e["_t"], e.get("reason", "")) for e in rec["events"] if e.get("type") == "stop_audio" and e.get("reason") != "session closed"]
    starts = [s for s, _, _ in tape["speech"]] + [tape["seconds"] + TAIL_S]
    speaking = lambda t: any(r["start"] < t < r["end"] - 0.3 for r in replies)
    checks = []

    def add(label, name, ok, detail):
        checks.append({"tape": tape["tape"], "at": label.get("at", label.get("from")), "check": name, "ok": ok, "detail": detail})

    for lb in tape["labels"]:
        k = lb["kind"]
        if k == "reply":
            until = min(s for s in starts if s > lb["at"])                 # the user's next utterance
            heard = [r for r in replies if lb["at"] - EARLY_S <= r["start"] < until]
            first = next((r for r in heard if r["start"] <= lb["at"] + lb["within"]), None)
            said = " ".join(r["text"] for r in heard)
            add(lb, "reply", first is not None,
                f"after \"{lb['text']}\": " + (f"started {first['start'] - lb['at']:+.2f} s after the last word" if first else
                                               (f"no reply within {lb['within']:.0f} s" + (f" (one at {heard[0]['start'] - lb['at']:+.1f} s)" if heard else ""))))
            if lb.get("contains_any"):
                norm = lambda x: " ".join(re.sub(r"[-,]", " ", x.lower()).split())      # "forty-one" = "forty one"
                ok = any(norm(w) in norm(said) for w in lb["contains_any"])
                add(lb, "answer", ok, f"wanted one of {lb['contains_any']}; said: \"{said.strip()[:160]}\"")
            if lb.get("tools_any"):
                called = [n for t, n in tools if lb["at"] - EARLY_S <= t < until]
                add(lb, "tool", any(n in lb["tools_any"] for n in called), f"wanted {' or '.join(lb['tools_any'])}; called {called or 'nothing'}")
        elif k == "hold":
            cut = [r for r in replies if lb["from"] - EARLY_S < r["start"] <= lb["to"] + EARLY_S]
            add(lb, "hold", not cut, f"pause {lb['from']:.1f}-{lb['to']:.1f} s mid-sentence: " + (f"a reply started at {cut[0]['start']:.1f} s" if cut else "no reply started"))
        elif k == "backchannel":
            if not speaking(lb["at"]):
                add(lb, "backchannel", None, "the assistant wasn't speaking then, so there was nothing to keep going")
                continue
            hit = [(t, r) for t, r in stops if lb["at"] <= t <= lb["at"] + BACKCHANNEL_S]
            add(lb, "backchannel", not hit, "kept speaking" if not hit else f"stopped ({hit[0][1]}) {hit[0][0] - lb['at']:.1f} s after the backchannel")
        elif k == "interrupt":
            if not speaking(lb["at"]):
                add(lb, "interrupt", None, "the assistant wasn't speaking then, so there was nothing to stop")
                continue
            hit = [(t, r) for t, r in stops if lb["at"] <= t <= lb["at"] + STOP_WITHIN_S]
            add(lb, "interrupt", bool(hit), f"stopped ({hit[0][1]}) {hit[0][0] - lb['at']:.1f} s after the user cut in" if hit else f"still speaking {STOP_WITHIN_S:.0f} s after the user cut in")
    errors = [e for e in rec["events"] if e.get("type") == "error"]
    checks.append({"tape": tape["tape"], "at": None, "check": "errors", "ok": not errors,
                   "detail": "none" if not errors else "; ".join(str(e.get("message") or e.get("error") or e)[:120] for e in errors[:3])})
    return checks


# ---------------------------------------------------------------------- the report
def pctl(xs: list, p: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def write_report(out: Path, r: dict) -> None:
    (out / "report.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    s, i = r["system"], r["install"]
    gpus = ", ".join(f"{g['name']} ({int(g['memory_mib']) / 1024:.0f} GB, driver {g['driver']})" for g in s.get("gpus", [])) or s.get("speakrail_sees", "")
    checks = [c for c in r["checks"] if c["ok"] is not None]
    passed = sum(1 for c in checks if c["ok"])
    L = [f"# speakrail verify: {i['config']} — {'PASS' if r['pass'] else 'FAIL'}", "",
         f"{passed} of {len(checks)} checks passed on {len(r['tapes'])} tape(s), {r['date']}.", "",
         "## This machine", "",
         f"- GPU: {gpus}" + (f", CUDA {s['cuda']}" if s.get("cuda") else ""),
         f"- CPU: {s.get('cpu', '?')}, {s.get('ram_gib', '?')} GB RAM",
         f"- OS: {s['os']}" + (f", macOS {s['macos']}" if s.get("macos") else "") + (", under WSL2" if s.get("wsl") else ""),
         f"- Docker: {s['docker']}" + (f", {s['nvidia_container_toolkit']}" if s.get("nvidia_container_toolkit") else "") if s.get("docker") else "- Docker: not installed",
         f"- Speakrail sees: {s['speakrail_sees']}", "",
         "## What ran", "",
         f"- Speakrail {i['speakrail']}, commit {(i['commit'] or 'unknown')[:12]}" + (" **with uncommitted changes**" if i["uncommitted_changes"] else ""),
         f"- Config `{i['config']}` ({i['config_status']}), file sha256 {i['config_sha256']}"]
    for role, e in i["engines"].items():
        L.append(f"- {role}: `{e['id']}` (manifest {e['manifest_sha256']})" + (f", params {e['params']}" if e["params"] else "")
                 + "".join(f", image {k} {v}" for k, v in e.get("images", {}).items()))
    L += [f"- Models: {', '.join(i['models']) or 'none'}", "",
          "## Engines", "", "| engine | check | | detail |", "|---|---|---|---|"]
    L += [f"| {c['engine']} | {c['check']} | {'ok' if c['ok'] else '**FAIL**'} | {c['detail'][:120]} |" for c in r["conformance"]]
    L += ["", "## Tapes", "", "| tape | at | check | | detail |", "|---|---|---|---|---|"]
    for c in r["checks"]:
        mark = "n/a" if c["ok"] is None else ("ok" if c["ok"] else "**FAIL**")
        at = "" if c["at"] is None else f"{c['at']:.1f} s"
        L.append(f"| {c['tape']} | {at} | {c['check']} | {mark} | {c['detail'].replace('|', '/')} |")
    m, lat = r["memory"], r["latency"]
    L += ["", "## Measured (reported, not judged)", "",
          f"- Ready after {r['startup_s']:.0f} s",
          f"- Last word to the reply's first audio, p50/p90: {lat['p50_ms']}/{lat['p90_ms']} ms over {lat['n']} replies" if lat["n"] else "- No replies to time",
          (f"- GPU/unified memory: {m['peak_gib']} GB at the peak, {m['peak_above_before_gib']} GB above the {m['before_gib']} GB used before the start" if m else "- Memory: not measured on this platform"),
          "", "## For the config's [evidence]", "", "```toml", "[evidence]",
          f'machine = "{(s.get("gpus") or [{"name": s["speakrail_sees"]}])[0]["name"]}, {s["os"].split("-")[0]}{", WSL2" if s.get("wsl") else ""}"',
          f'date = "{r["date"][:10]}"',
          f'tapes = "speakrail verify {i["speakrail"]}: {len(r["tapes"])} synthetic tapes, {passed} of {len(checks)} checks, {r["replies"]} replies"',
          f'last_word_to_ear_ms = "{lat["p50_ms"]}/{lat["p90_ms"]}"' if lat["n"] else "", "```", ""]
    (out / "report.md").write_text("\n".join(x for x in L if x is not None) + "\n")


# ---------------------------------------------------------------------- the command
def main(a) -> int:
    from . import service
    from .resolver import ResolveError
    from .run import setup
    if service.find():
        say("Speakrail is running. Stop it first (speakrail stop): verify starts the config itself, in a known state.")
        return 1
    names = QUICK if a.quick else sorted(p.stem for p in (Path(a.tapes) if a.tapes else TAPES).glob("*.json"))
    tapes = [json.loads(((Path(a.tapes) if a.tapes else TAPES) / f"{n}.json").read_text()) for n in names]
    if not tapes:
        say(f"No tapes in {a.tapes or TAPES}")
        return 1
    try:
        resolved, _, _ = setup(argparse.Namespace(config=a.config, set=None, param=None, force=a.force))
    except (ResolveError, ValueError) as e:
        say(str(e))
        return 2
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) if a.out else Path(CFG.home) / "verify" / f"{resolved.config.name}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    say(f"Verifying {resolved.config.name} with {len(tapes)} tape{'s' if len(tapes) != 1 else ''}; the report goes to {out}")
    r = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "system": system_info(), "install": install_info(resolved), "tapes": names}
    port, mem, p, recs = _free_port(), Memory(), None, {}
    try:
        say("Starting it, as `speakrail run` would (the first start also measures this machine)...")
        t0 = time.monotonic()
        p = start_stack(a, out, port)
        r["startup_s"] = time.monotonic() - t0
        say(f"Ready after {r['startup_s']:.0f} s. Checking each engine against its contract...")
        r["conformance"] = asyncio.run(conformance(resolved))
        say("Comparing its decisions with the reference for its model...")
        r["conformance"] += asyncio.run(fidelity(resolved))
        checks = []
        for tape in tapes:
            say(f"Playing {tape['tape']} ({tape['seconds']:.0f} s, voice {tape['voice']})...")
            rec = asyncio.run(play(port, (Path(a.tapes) if a.tapes else TAPES) / f"{tape['tape']}.wav"))
            recs[tape["tape"]] = rec
            got = judge(tape, rec)
            checks += got
            for c in got:
                if c["ok"] is False:
                    say(f"  FAIL {c['check']}: {c['detail']}")
        r["checks"] = checks
    except (RuntimeError, OSError, asyncio.TimeoutError) as e:
        say(f"Verify couldn't finish: {e}")
        r.setdefault("startup_s", 0.0); r.setdefault("conformance", [])
        r["checks"] = r.get("checks", []) + [{"tape": "-", "at": None, "check": "run", "ok": False, "detail": str(e)[:300]}]
    finally:
        if p is not None:
            say("Stopping it...")
            stop_stack(p)
        r["memory"] = mem.stop()
    lat = []
    for tape in tapes:
        rec = recs.get(tape["tape"])
        if not rec:
            continue
        rs = replies_of(rec)
        for lb in (x for x in tape["labels"] if x["kind"] == "reply"):
            first = next((x for x in rs if lb["at"] - EARLY_S <= x["start"] <= lb["at"] + lb["within"]), None)
            if first:
                lat.append(round((first["start"] - lb["at"]) * 1000))
        (out / f"{tape['tape']}.events.json").write_text(json.dumps({"replies": rs, "audio": rec["audio"], "events": rec["events"]}, default=str) + "\n")
    r["latency"] = {"n": len(lat), "p50_ms": pctl(lat, 0.5), "p90_ms": pctl(lat, 0.9)}
    r["replies"] = sum(len(replies_of(x)) for x in recs.values())
    r["pass"] = bool(r["checks"]) and all(c["ok"] is not False for c in r["checks"]) and all(c["ok"] for c in r["conformance"])
    write_report(out, r)
    n_ok = sum(1 for c in r["checks"] if c["ok"])
    n_all = sum(1 for c in r["checks"] if c["ok"] is not None)
    conf_bad = [c for c in r["conformance"] if not c["ok"]]
    say(f"\n{'PASS' if r['pass'] else 'FAIL'}: {n_ok} of {n_all} tape checks" + (f", {len(conf_bad)} engine check(s) failed" if conf_bad else ", every engine check passed")
        + (f"; last word to ear p50/p90 {r['latency']['p50_ms']}/{r['latency']['p90_ms']} ms" if lat else "")
        + (f"; peak memory {r['memory']['peak_gib']} GB" if r["memory"] else ""))
    say(f"The report: {out / 'report.md'}" + ("" if a.quick else ". Attach it to a pull request that marks this config tested."))
    return 0 if r["pass"] else 1
