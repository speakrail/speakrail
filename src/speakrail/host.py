"""The host: the engines as one stack that can load and unload, and a private local port to control it.

The stack is stopped, loading, ready, unloading or failed. The harness outlives all of these, so the port stays open and a
new session can wake the engines. Why the stack is stopped decides that: after an idle or memory offload a session loads
it again, while an offload someone asked for lasts until they load it.

An engine dying stops the others and leaves the stack failed, with the reason in the status; only a load brings it back.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .hardware import Hardware, detect, engine_memory
from .supervisor import EngineError

STOPPED, LOADING, READY, UNLOADING, FAILED = "stopped", "loading", "ready", "unloading", "failed"
MANUAL, IDLE, MEMORY = "manual", "idle", "memory"
PRESSURE_CHECKS = 3                   # in a row, so a short spike doesn't unload anything
MEMORY_CHECK_EVERY_S = 5


class Refused(Exception):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def fit_problem(peak_gib: float, hw: Hardware | None) -> tuple[str, str]:
    """-> (why it can't load, what to warn about). Free memory isn't the whole story on unified memory: macOS can compress other
    apps to make room, so being a little short only earns a warning there. Its pressure level "warn" doesn't count against loading:
    a 24 GB Mac reaches it with this stack and everyday apps open, and replies stay on time. "critical" does."""
    if hw is None:
        return "", ""
    short = f"the stack peaks at {peak_gib:.1f} GB and {hw.free_memory_gib:.1f} GB is free"
    if hw.memory_pressure == "critical":
        return f"This machine is critically short of memory, and {short}. Close other apps and try again.", ""
    if hw.free_memory_gib >= peak_gib:
        return "", ""
    if hw.memory_kind == "unified" and hw.free_memory_gib >= peak_gib / 2:
        return "", f"Tight on memory: {short}, so other apps will be compressed to make room"
    return f"Not enough memory: {short}. Close other apps and try again.", ""


class Stack:
    def __init__(self, sup, engines, params, peak_gib: float, idle_s: float = 0.0, hardware=detect, say=print, clock=time.monotonic):
        self.sup, self.engines, self.params, self.peak_gib, self.idle_s = sup, engines, params, peak_gib, idle_s
        self.hardware, self.say, self.clock = hardware, say, clock
        self.state = STOPPED
        self.reason = None                   # why it is stopped: manual, idle or memory
        self.error = ""
        self.fit_warning = ""                # from the last load
        self.pressure_warning = ""
        self.sessions = 0
        self.generation = 0                  # counts loads; the harness warms its TTS voice again when this changes
        self.idle_since = clock()
        self.hw = None
        self.op = threading.RLock()          # one load or unload at a time
        self.pressure_checks = 0
        self.closed = threading.Event()
        self.monitor = None

    @property
    def warning(self) -> str:
        return self.pressure_warning or self.fit_warning

    # ------------------------------------------------------------------ load and unload
    def require_fit(self) -> str:
        self.hw = self.hardware()
        problem, warning = fit_problem(self.peak_gib, self.hw)
        if problem:
            raise Refused("not_enough_memory", problem)
        return warning

    def load(self, force: bool = False) -> None:
        with self.op:
            if self.state != READY:
                self._start("" if force else self.require_fit())

    def load_in_background(self, force: bool = False) -> None:
        """the answer can't wait for the engines, but it does say at once if they won't fit, and the state has moved on by the time it returns"""
        with self.op:
            if self.state == READY:
                return
            warning = "" if force else self.require_fit()
            self.state, self.reason, self.error = LOADING, None, ""

        def run():
            try:
                with self.op:
                    self._start(warning)
            except Refused:
                pass                         # the status has the error
        threading.Thread(target=run, daemon=True, name="stack-load").start()

    def _start(self, warning: str) -> None:
        if self.state == READY:              # a session got in first
            return
        if warning:
            self.say(warning)
        self.state, self.error, self.fit_warning, self.pressure_warning = LOADING, "", warning, ""
        try:
            self.sup.start_all(self.engines, params=self.params)
        except EngineError as e:
            self.sup.stop_all()
            self.state, self.error = FAILED, str(e)
            raise Refused("engines_failed", str(e), 500) from None
        self.state, self.reason = READY, None
        self.generation += 1
        self.idle_since = self.clock()
        self.pressure_checks = 0

    def offload(self, reason: str = MANUAL, force: bool = False) -> None:
        with self.op:
            if self.state != READY:
                if self.state == STOPPED or reason == MANUAL:
                    self.state, self.error = STOPPED, ""
                    self.reason = MANUAL if reason == MANUAL else self.reason
                return
            if self.sessions and not force:
                raise Refused("session_active", "a session is open; end it first, or use force to end it")
            self.state = UNLOADING
            self.sup.stop_all()
            self.state, self.reason, self.fit_warning, self.pressure_warning = STOPPED, reason, "", ""

    def ensure_ready(self) -> int:
        """a session wants the engines -> the generation that's running. Waits for a load, or starts one if that's allowed."""
        with self.op:
            if self.state == READY:
                return self.generation
            if self.state == FAILED:
                raise Refused("engines_failed", f"The engines failed and need loading again: {self.error}")
            if self.reason == MANUAL:
                raise Refused("engines_offloaded", "The engines were offloaded on request. Load them first: the Engines menu, or speakrail load.")
            self.load()
            return self.generation

    def close(self) -> None:
        self.closed.set()
        with self.op:
            self.sup.stop_all()

    # ------------------------------------------------------------------ what happens by itself
    def on_engine_event(self, engine: str, state: str, detail: str) -> None:
        """from the supervisor's watcher. While loading, the failure comes back as an exception from start_all instead."""
        if state == "failed" and self.state == READY:
            self.state, self.error = FAILED, f"{engine}: {detail}"
            self.say(f"Engine failed, stopping the others: {self.error}. Load again from the Engines menu, or with speakrail load.")
            threading.Thread(target=self._stop_the_rest, daemon=True, name="stack-stop").start()

    def _stop_the_rest(self) -> None:
        with self.op:
            if self.state == FAILED:
                self.sup.stop_all()

    def set_sessions(self, n: int) -> None:
        self.sessions = n
        if n == 0:
            self.idle_since = self.clock()

    def tick(self, check_memory: bool = True) -> None:
        if self.state != READY:
            return
        if not self.sessions and self.idle_s and self.clock() - self.idle_since >= self.idle_s:
            self._auto_offload(IDLE, f"No session for {self.idle_s / 60:g} min: engines offloaded, the next session loads them again.")
        elif check_memory:
            self.hw = self.hardware()
            if self.hw is None or self.hw.memory_pressure != "critical":
                self.pressure_checks, self.pressure_warning = 0, ""
                return
            self.pressure_checks += 1
            if self.sessions:                # never a live conversation
                if not self.pressure_warning:
                    self.pressure_warning = "This machine is critically short of memory; replies may slow down."
                    self.say(self.pressure_warning)
            elif self.pressure_checks >= PRESSURE_CHECKS:
                self._auto_offload(MEMORY, "Critically short of memory with no session: engines offloaded, the next session loads them again if they fit.")

    def _auto_offload(self, reason: str, message: str) -> None:
        with self.op:
            if self.sessions or self.state != READY:          # someone connected while we were deciding
                return
            self.offload(reason)
        self.say(message)

    def start_monitor(self) -> None:
        def run():
            n = 0
            while not self.closed.wait(1.0):
                n += 1
                try:
                    self.tick(check_memory=n % MEMORY_CHECK_EVERY_S == 0)
                except Exception as e:       # a failed measurement must not take the stack down
                    self.say(f"Stack monitor: {type(e).__name__}: {e}")
        self.monitor = threading.Thread(target=run, daemon=True, name="stack-monitor")
        self.monitor.start()

    # ------------------------------------------------------------------ status
    def status(self, memory: bool = False) -> dict:
        hw = self.hw
        if memory or hw is None:
            hw = self.hw = self.hardware()
        engines = {e.id: {"role": e.role, "state": self.sup.state(e.id), "error": self.sup.error(e.id)} for e in self.engines}
        if memory and hw:
            groups = {e.id: list(self.sup.pids(e.id).values()) for e in self.engines if self.sup.state(e.id) == "ready"}
            for name, gib in engine_memory(groups, hw.platform).items():
                engines[name]["memory_gib"] = round(gib, 2)
        idle_in = None
        if self.state == READY and self.idle_s and not self.sessions:
            idle_in = max(0.0, self.idle_s - (self.clock() - self.idle_since))
        return {"state": self.state, "reason": self.reason, "wakes_on_session": self.state == STOPPED and self.reason != MANUAL, "error": self.error, "warning": self.warning,
                "sessions": self.sessions, "generation": self.generation, "idle_offload_s": self.idle_s, "idle_in_s": idle_in, "peak_gib": self.peak_gib, "engines": engines,
                "machine": {"free_gib": round(hw.free_memory_gib, 1), "total_gib": round(hw.memory_gib, 1), "kind": hw.memory_kind, "pressure": hw.memory_pressure} if hw else None}


def serve_control(stack: Stack, token: str, on_shutdown) -> ThreadingHTTPServer:
    """on 127.0.0.1, an ephemeral port, the API token required. The harness forwards /v1/control/* here; the CLI comes straight here."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self) -> bool:
            given = self.headers.get("Authorization", "")
            if given.startswith("Bearer ") and secrets.compare_digest(given[7:].strip(), token):
                return True
            self.reply(401, {"error": {"code": "unauthorized", "message": "a valid Authorization: Bearer token is required"}})
            return False

        def do_GET(self):
            if not self.authorized():
                return
            url = urlparse(self.path)
            if url.path == "/status":
                return self.reply(200, stack.status(memory=parse_qs(url.query).get("memory") == ["1"]))
            self.reply(404, {"error": {"code": "not_found", "message": url.path}})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                force = body.get("force") is True
                if self.path == "/load":
                    stack.load_in_background(force)
                    return self.reply(202, {"state": stack.state})
                if self.path == "/offload":
                    stack.offload(MANUAL, force)
                    return self.reply(200, stack.status())
                if self.path == "/ensure_ready":
                    return self.reply(200, {"generation": stack.ensure_ready()})
                if self.path == "/sessions":
                    stack.set_sessions(int(body["open"]))
                    return self.reply(200, {})
                if self.path == "/shutdown":
                    self.reply(200, {})
                    return on_shutdown()
                self.reply(404, {"error": {"code": "not_found", "message": self.path}})
            except Refused as e:
                self.reply(e.status, {"error": {"code": e.code, "message": e.message}})
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                self.reply(400, {"error": {"code": "invalid_request", "message": str(e)}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="control").start()
    return server
