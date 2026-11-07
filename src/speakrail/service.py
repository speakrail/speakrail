"""speakrail start | stop | status | logs | load | offload: a running Speakrail, from any terminal.

`run` leaves <home>/run/host.json saying which process it is and where its control port is, so these find it from there.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import ui
from .config import CFG, REPO_ROOT
from .ui import say

NOT_RUNNING = 3                       # what `status` exits with, like an init script
READY_LINE = "Speakrail is ready"


def home() -> Path:
    return Path(CFG.home)


def host_file() -> Path:
    return home() / "run" / "host.json"


def token() -> str:
    f = home() / "config" / "api_token"
    return f.read_text().strip() if f.exists() else ""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def call(info: dict, method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    req = urllib.request.Request(f"http://127.0.0.1:{info['port']}{path}", method=method, data=json.dumps(body or {}).encode() if method == "POST" else None,
                                 headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # a local port, whatever proxy is set
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def find() -> dict | None:
    """the running host, or None. A file left behind by a crash doesn't count, and is removed."""
    try:
        info = json.loads(host_file().read_text())
    except (OSError, ValueError):
        return None
    if alive(info["pid"]):
        try:
            call(info, "GET", "/status", timeout=3)
            return info
        except (OSError, ValueError):
            pass
    host_file().unlink(missing_ok=True)
    return None


def write_host_file(port: int, url: str) -> None:
    f = host_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch(mode=0o600)
    f.write_text(json.dumps({"pid": os.getpid(), "port": port, "url": url}))


def remove_host_file() -> None:
    try:
        if json.loads(host_file().read_text())["pid"] == os.getpid():
            host_file().unlink()
    except (OSError, ValueError, KeyError):
        pass


def _run_args(a) -> list[str]:
    out = []
    if a.config:
        out += ["--config", a.config]
    for s in a.set or []:
        out += ["--set", s]
    for p in a.param or []:
        out += ["--param", p]
    if a.force:
        out.append("--force")
    if a.no_bench:
        out.append("--no-bench")
    return out


def start_main(a) -> int:
    info = find()
    if info:
        say(f"Already running (pid {info['pid']}): {info['url']}")
        return 0
    log = home() / "logs" / "speakrail.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    offset = log.stat().st_size if log.exists() else 0
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]))}
    with open(log, "ab", buffering=0) as out:
        p = subprocess.Popen([sys.executable, "-u", "-m", "speakrail", "run", *_run_args(a)], env=env, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            partial = ""
            while True:
                partial += f.read().decode(errors="replace")
                ready = READY_LINE in partial
                lines, partial = partial.rsplit("\n", 1) if "\n" in partial else ("", partial)
                if lines:
                    say(lines)
                if ready:
                    return 0
                code = p.poll()
                if code is not None:
                    say(f"\nIt stopped with code {code}; the log is {log}")
                    return code or 1
                time.sleep(0.2)
    except KeyboardInterrupt:
        say("\nStill starting in the background: speakrail status shows how it's going, speakrail stop stops it.")
        return 0


def stop_main(_) -> int:
    info = find()
    if not info:
        say("Not running")
        return 0
    call(info, "POST", "/shutdown")
    t0 = time.monotonic()
    while alive(info["pid"]):
        if time.monotonic() - t0 > 60:
            os.kill(info["pid"], signal.SIGTERM)
            say(f"Still stopping after 60 s (pid {info['pid']}); asked it again. The log is {home() / 'logs'}")
            return 1
        time.sleep(0.1)
    say("Stopped")
    return 0


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def describe(s: dict) -> list[str]:
    why = {"manual": "offloaded on request, stays that way until loaded", "idle": "offloaded after no session, the next session loads it", "memory": "offloaded under memory pressure, the next session loads it if it fits"}
    state = s["state"] + (f" ({why[s['reason']]})" if s["state"] == "stopped" and s["reason"] in why else "")
    out = [f"Engines   {state}; {s['sessions']} session{'' if s['sessions'] == 1 else 's'} open"]
    if s["idle_in_s"] is not None:
        out.append(f"Idle      unloads in {_minutes(s['idle_in_s'])} if no session opens")
    elif s["state"] == "ready" and not s["idle_offload_s"]:
        out.append("Idle      stays loaded")
    for name, e in s["engines"].items():
        mem = f"  {e['memory_gib']:.1f} GB" if "memory_gib" in e else ""
        out.append(f"  {name:18} {e['state']}{mem}{'  ' + e['error'] if e['error'] else ''}")
    if s["machine"]:
        m = s["machine"]
        out.append(f"Memory    {m['free_gib']:.1f} of {m['total_gib']:.0f} GB free, pressure {m['pressure']}; the stack peaks at {s['peak_gib']:.1f} GB")
    if s["warning"]:
        out.append(f"Warning   {s['warning']}")
    if s["error"]:
        out.append(f"Error     {s['error']}")
    return out


def status_main(_) -> int:
    from .lifecycle import update_notice
    info = find()
    if not info:
        say("Not running")
        code = NOT_RUNNING
    else:
        _, s = call(info, "GET", "/status?memory=1")
        say(f"Running (pid {info['pid']}): {info['url']}")
        say("\n".join(describe(s)))
        code = 0
    notice = update_notice(home())
    if notice:
        say(notice)
    return code


def load_main(a) -> int:
    info = find()
    if not info:
        say("Not running; start it with: speakrail start")
        return NOT_RUNNING
    code, r = call(info, "POST", "/load", {"force": a.force})
    if code >= 400:
        say(r["error"]["message"])
        return 1
    while True:
        _, s = call(info, "GET", "/status")
        if s["state"] == "ready":
            say("Loaded")
            return 0
        if s["state"] in ("failed", "stopped") and s["error"]:
            say(s["error"])
            return 1
        time.sleep(0.2)


def offload_main(a) -> int:
    info = find()
    if not info:
        say("Not running")
        return NOT_RUNNING
    code, r = call(info, "POST", "/offload", {"force": a.force}, timeout=120)
    if code >= 400:
        say(r["error"]["message"])
        return 1
    say("Offloaded; it stays that way until speakrail load")
    return 0


def log_files() -> dict[str, Path]:
    root = home() / "logs"
    return {str(p.relative_to(root).with_suffix("")): p for p in sorted(root.rglob("*.log"))} if root.is_dir() else {}


def logs_main(a) -> int:
    files = log_files()
    name = a.name or ("speakrail" if "speakrail" in files else "harness")
    match = [k for k in files if k == name] or [k for k in files if k.startswith(name.replace("/", "-"))]
    if len(match) != 1:
        say(f"No single log called {name!r}. There are: {', '.join(files) or 'none yet'}")
        return 1
    path = files[match[0]]
    lines = path.read_bytes().decode(errors="replace").splitlines()
    say("\n".join(lines[-a.lines:]))
    if not a.follow:
        return 0
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            while True:
                text = f.read().decode(errors="replace")
                if text:
                    print(ui.paint(text), end="", flush=True)
                time.sleep(0.2)
    except KeyboardInterrupt:
        return 0
