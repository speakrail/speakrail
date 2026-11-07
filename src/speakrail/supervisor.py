"""Starts engines, watches them and stops them. The caller sees one state per engine: stopped, starting, ready or failed.

Each process runs in its own process group, so stopping an engine also stops whatever it started. Failure is detected within
a second and reported through on_event; the supervisor doesn't restart anything by itself, that decision belongs to the caller.
"""
from __future__ import annotations

import os
import re
import shutil
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

from . import containers
from .config import CFG
from .contracts import ROLES
from .manifest import Engine, ports_of, render, values_for

STOPPED, STARTING, READY, FAILED = "stopped", "starting", "ready", "failed"
SECRET_ENV = re.compile(r"(_API_KEY|_TOKEN|_SECRET|_ACCESS_KEY)$|^(HF_TOKEN|ANTHROPIC_|OPENAI_)", re.I)   # engines don't get the user's keys
STOP_TIMEOUT_S = 12.0


class EngineError(RuntimeError):
    pass


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) != 0


def who_uses(port: int) -> str:
    """the program listening on a local port, as "name (pid N)", or "" when the system won't say"""
    try:
        if shutil.which("lsof"):
            out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"], capture_output=True, text=True, timeout=5).stdout
            pid = next((x[1:] for x in out.splitlines() if x.startswith("p")), "")
            name = next((x[1:] for x in out.splitlines() if x.startswith("c")), "")
            if name:
                return f"{name} (pid {pid})"
        if shutil.which("ss"):
            out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r'users:\(\("([^"]+)",pid=(\d+)', out)
            if m:
                return f"{m[1]} (pid {m[2]})"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def ui_port_taken(port: int) -> str:
    who = who_uses(port)
    free = next((p for p in range(port + 100, port + 200) if port_free(p)), port + 100)
    return (f"The page needs port {port}, and {who or 'another program'} is already using it. Either close that program, "
            f"or start Speakrail on a free port: UI_PORT={free} speakrail start")


def port_taken(engine_id: str, name: str, port: int) -> str:
    """what to say when an engine's port is taken: who has it, and the two ways out"""
    who = who_uses(port)
    free = next((p for p in range(port + 100, port + 200) if port_free(p)), port + 100)
    return (f"{engine_id} needs port {port}, and {who or 'another program'} is already using it. Either close that program, "
            f"or move this engine to a free port: speakrail start --param {engine_id}:port_{name}={free}")


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
    containers: list = field(default_factory=list)       # names of the containers it started, in order
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
        run_dir = self.home / "run" / engine.slug
        log_dir = self.home / "logs" / engine.slug
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        values = values_for(engine, self.home, self.models, run_dir, params, python)
        ports = ports_of(engine, values)
        leftovers = [containers.name(engine.slug, p.name) for p in engine.processes if p.image]
        for name in leftovers:                             # a run that was killed leaves its containers holding their ports and GPU memory
            containers.remove(name)
        if leftovers:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not all(port_free(port) for port in ports.values()):
                time.sleep(0.1)
        for name, port in ports.items():
            if not port_free(port):
                raise EngineError(port_taken(engine.id, name, port))
        if engine.python and not Path(values["python"]).exists():
            raise EngineError(f"{engine.id} isn't installed: no environment at {values['python']} (speakrail engine install {engine.id})")
        missing = [p for p in (render(n, values, engine.id) for n in engine.needs) if not Path(p).exists()]
        if missing:
            raise EngineError(f"{engine.id} is missing {', '.join(missing)}")
        if engine.in_containers:
            if (why := containers.problem(gpu=engine.needs_gpu_runtime)):
                raise EngineError(f"{engine.id}: {why}")
            for image in {render(p.image, values, engine.id) for p in engine.processes if p.image}:
                if not containers.image_present(image):
                    raise EngineError(f"{engine.id}: the image {image} isn't on this machine (docker pull {image})")
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
                where = f"{engine.id} {proc.name}"
                cmd = [render(c, values, where) for c in proc.cmd]
                cwd = render(proc.cwd, values, where) if proc.cwd else None
                own_env = {k: render(str(v), values, where) for k, v in proc.env.items()}
                penv = {**env, **own_env}
                if proc.image:
                    name = containers.name(engine.slug, proc.name)
                    r.containers.append(name)
                    cmd = containers.run_command(name, render(proc.image, values, where), cmd, proc.entrypoint and render(proc.entrypoint, values, where), proc.gpu,
                                                 [render(m, values, where) for m in proc.mounts], own_env, [render(a, values, where) for a in proc.docker_args])
                    penv = env
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
        for name in reversed(r.containers):                # by name: their `docker run` clients below don't reliably take them along
            containers.stop(name)
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


def images(engine: Engine, home: Path | str | None = None, params: dict | None = None) -> set[str]:
    """the images the engine's containers run, with the config's params filled in: a config can pick the image"""
    if not engine.in_containers:
        return set()
    home = Path(home or CFG.home)
    values = values_for(engine, home, Path(CFG.models_dir or home / "models"), home / "run" / engine.slug, params)
    return {render(p.image, values, engine.id) for p in engine.processes if p.image}


def install(engine: Engine, home: Path | str | None = None, uv: str = "uv", params: dict | None = None) -> Path | None:
    """get what the engine runs from: its images, or its own Python environment. Engines with neither have nothing to install here."""
    home = Path(home or CFG.home)
    if engine.in_containers:
        if (why := containers.problem(gpu=engine.needs_gpu_runtime)):
            raise EngineError(f"{engine.id}: {why}")
        for image in images(engine, home, params):
            if not containers.image_present(image):
                containers.pull(image)
    if not engine.python:
        return None
    env = engine.env_dir(home)
    env.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([uv, "-q", "venv", "--allow-existing", "--python", str(engine.python.get("version", "3.12")), str(env)], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(env / "bin" / "python"), "-r", str(engine.dir / engine.python["requirements"])], check=True)
    return env
