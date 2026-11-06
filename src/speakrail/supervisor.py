"""Starts engines, watches them and stops them. The caller sees one state per engine: stopped, starting, ready or failed.

Each process runs in its own process group, so stopping an engine also stops whatever it started. Failure is detected within
a second and reported through on_event; the supervisor doesn't restart anything by itself, that decision belongs to the caller.
"""
from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import CFG
from .contracts import ROLES
from .manifest import Engine, render, values_for

STOPPED, STARTING, READY, FAILED = "stopped", "starting", "ready", "failed"
SECRET_ENV = re.compile(r"(_API_KEY|_TOKEN|_SECRET|_ACCESS_KEY)$|^(HF_TOKEN|ANTHROPIC_|OPENAI_)", re.I)   # engines don't get the user's keys
STOP_TIMEOUT_S = 12.0


class EngineError(RuntimeError):
    pass


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) != 0


def check_health(target: str, timeout: float = 1.0) -> bool:
    if target.startswith("tcp://"):
        host, port = target[6:].rsplit(":", 1)
        with socket.socket() as s:
            s.settimeout(timeout)
            return s.connect_ex((host, int(port))) == 0
    try:
        with urllib.request.urlopen(target, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


@dataclass
class Running:
    engine: Engine
    values: dict
    state: str = STARTING
    procs: dict = field(default_factory=dict)            # process name -> Popen
    error: str = ""
    started_s: float | None = None


class Supervisor:
    def __init__(self, home: Path | str | None = None, models: Path | str | None = None, on_event: Callable[[str, str, str], None] | None = None):
        self.home = Path(home or CFG.home)
        self.models = Path(models or CFG.models_dir or self.home / "models")
        self.on_event = on_event or (lambda engine, state, detail: None)
        self.running: dict[str, Running] = {}
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._watcher = threading.Thread(target=self._watch, daemon=True, name="engine-watch")
        self._watcher.start()

    # ------------------------------------------------------------------ state
    def state(self, engine_id: str) -> str:
        r = self.running.get(engine_id)
        return r.state if r else STOPPED

    def error(self, engine_id: str) -> str:
        r = self.running.get(engine_id)
        return r.error if r else ""

    def url(self, engine_id: str) -> str:
        r = self.running[engine_id]
        return render(r.engine.url, r.values, engine_id)

    def status(self) -> dict:
        return {k: {"state": r.state, "error": r.error, "started_s": r.started_s} for k, r in self.running.items()}

    def pids(self, engine_id: str) -> dict:
        r = self.running.get(engine_id)
        return {n: p.pid for n, p in r.procs.items()} if r else {}

    def _set(self, r: Running, state: str, detail: str = ""):
        r.state = state
        if state == FAILED:
            r.error = detail
        self.on_event(r.engine.id, state, detail)

    # ------------------------------------------------------------------ start
    def start(self, engine: Engine, params: dict | None = None, python: str | None = None) -> Running:
        """start one engine and wait until every process is healthy. Raises EngineError on failure, with nothing left running."""
        with self.lock:
            prev = self.running.get(engine.id)
            if prev and prev.state in (STARTING, READY):
                raise EngineError(f"{engine.id} is already {prev.state}")
        if not engine.runs_on():
            raise EngineError(f"{engine.id} doesn't run on {sys.platform} (it runs on {', '.join(engine.platforms)})")
        for name, port in engine.ports.items():
            if not port_free(int(port)):
                raise EngineError(f"{engine.id}: port {port} ({name}) is already in use")
        run_dir = self.home / "run" / engine.slug
        log_dir = self.home / "logs" / engine.slug
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        values = values_for(engine, self.home, self.models, run_dir, params, python)
        if engine.python and not Path(values["python"]).exists():
            raise EngineError(f"{engine.id} isn't installed: no environment at {values['python']} (speakrail engine install {engine.id})")
        missing = [p for p in (render(n, values, engine.id) for n in engine.needs) if not Path(p).exists()]
        if missing:
            raise EngineError(f"{engine.id} is missing {', '.join(missing)}")
        for f in engine.files:
            (run_dir / f["name"]).write_text(render((engine.dir / f["template"]).read_text(), values, f"{engine.id} file {f['name']}"))
        r = Running(engine, values)
        with self.lock:
            self.running[engine.id] = r
        self._set(r, STARTING)
        t0 = time.monotonic()
        deadline = t0 + engine.startup_timeout_s
        env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
        try:
            for proc in engine.processes:
                cmd = [render(c, values, f"{engine.id} {proc.name}") for c in proc.cmd]
                cwd = render(proc.cwd, values, f"{engine.id} {proc.name}") if proc.cwd else None
                penv = {**env, **{k: render(str(v), values, f"{engine.id} {proc.name}") for k, v in proc.env.items()}}
                log = open(log_dir / f"{proc.name}.log", "ab", buffering=0)
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
                try:
                    r.procs[proc.name] = subprocess.Popen(cmd, cwd=cwd, env=penv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as e:
                    raise EngineError(f"{engine.id}: can't start {proc.name}: {e}") from e
                finally:
                    log.close()
                if proc.health:
                    target = render(proc.health, values, f"{engine.id} {proc.name}")
                    while not check_health(target):
                        self._check_alive(r)
                        if r.state == FAILED:
                            raise EngineError(r.error)
                        if time.monotonic() > deadline:
                            raise EngineError(f"{engine.id}: {proc.name} not healthy after {engine.startup_timeout_s:.0f} s ({target})")
                        time.sleep(0.2)
            self._check_alive(r)
            if r.state == FAILED:
                raise EngineError(r.error)
        except Exception as e:
            self._kill(r)
            if r.state != FAILED:
                self._set(r, FAILED, str(e))
            raise EngineError(str(e)) from None
        r.started_s = time.monotonic() - t0
        self._set(r, READY, f"{r.started_s:.1f} s")
        return r

    def start_all(self, engines: list[Engine], params: dict | None = None, python: str | None = None) -> None:
        """start in role order (ASR, TTS, LLM); on failure the ones already started are stopped again"""
        started = []
        try:
            for e in sorted(engines, key=lambda x: ROLES.index(x.role)):
                self.start(e, (params or {}).get(e.id), python)
                started.append(e.id)
        except EngineError:
            for k in reversed(started):
                self.stop(k)
            raise

    # ------------------------------------------------------------------ watch
    def _check_alive(self, r: Running):
        for name, p in r.procs.items():
            code = p.poll()
            if code is not None and r.state in (STARTING, READY):
                self._set(r, FAILED, f"{name} exited with code {code}; see {self.home / 'logs' / r.engine.slug / (name + '.log')}")
                self._kill(r)
                return

    def _watch(self):
        while not self._stop.wait(0.5):
            with self.lock:
                items = list(self.running.values())
            for r in items:
                if r.state == READY:
                    self._check_alive(r)

    # ------------------------------------------------------------------ stop
    @staticmethod
    def _signal(p: subprocess.Popen, sig) -> None:
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _kill(self, r: Running) -> None:
        procs = list(r.procs.values())
        for p in reversed(procs):
            self._signal(p, signal.SIGTERM)                # also when the leader is gone: its group may outlive it
        t0 = time.monotonic()
        while time.monotonic() - t0 < STOP_TIMEOUT_S and any(p.poll() is None for p in procs):
            time.sleep(0.05)
        for p in procs:
            if p.poll() is None:
                self._signal(p, signal.SIGKILL)
                p.wait(timeout=5)

    def stop(self, engine_id: str) -> None:
        r = self.running.get(engine_id)
        if r is None:
            return
        was_failed = r.state == FAILED
        r.state = STOPPED if not was_failed else FAILED    # so the watcher doesn't report the exit as a failure
        self._kill(r)
        if not was_failed:
            self._set(r, STOPPED)

    def stop_all(self) -> None:
        with self.lock:
            items = sorted(self.running.values(), key=lambda r: ROLES.index(r.engine.role), reverse=True)
        for r in items:
            self.stop(r.engine.id)

    def close(self) -> None:
        self.stop_all()
        self._stop.set()


def install(engine: Engine, home: Path | str | None = None, uv: str = "uv") -> Path | None:
    """create the engine's own Python environment. Engines without one have nothing to install here."""
    if not engine.python:
        return None
    home = Path(home or CFG.home)
    env = engine.env_dir(home)
    env.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([uv, "-q", "venv", "--allow-existing", "--python", str(engine.python.get("version", "3.12")), str(env)], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(env / "bin" / "python"), "-r", str(engine.dir / engine.python["requirements"])], check=True)
    return env
