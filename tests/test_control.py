#!/usr/bin/env python3
"""Loading and unloading: the stack's rules on a fake supervisor, then the whole thing on the stub engines, with no GPU and no models.
Needs the tokenizer, as test_session.py does.

    SPEAKRAIL_TOKENIZER_DIR=... python tests/test_control.py
"""
import asyncio, json, math, os, signal, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
HOME = tempfile.mkdtemp(prefix="speakrail-control-")
os.environ["SPEAKRAIL_HOME"] = HOME
import aiohttp
from speakrail.hardware import Hardware
from speakrail.host import FAILED, IDLE, LOADING, MANUAL, MEMORY, PRESSURE_CHECKS, READY, STOPPED, Refused, Stack, fit_problem
from speakrail.supervisor import EngineError, port_free
from test_stack_stub import talk

UI = 18081
ENGINE_PORTS = (18000, 18091, 18765)
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1", "SPEAKRAIL_HOME": HOME, "UI_PORT": str(UI)}
BASE = f"http://127.0.0.1:{UI}"


# ---------------------------------------------------------------------- the rules, with nothing running
def hw(free=20.0, pressure="normal", kind="unified"):
    return Hardware("apple", "test", 24.0, free, kind, 4.0, memory_pressure=pressure)


class FakeSup:
    def __init__(self):
        self.up, self.fail, self.starts = False, False, 0
    def start_all(self, engines, params=None):
        self.starts += 1
        if self.fail:
            raise EngineError("asr: no model")
        self.up = True
    def stop_all(self):
        self.up = False
    def state(self, _): return "ready" if self.up else "stopped"
    def error(self, _): return ""
    def pids(self, _): return {}


def stack(machine=None, idle_s=600.0, peak=10.0):
    now = [0.0]
    box = {"hw": machine or hw()}
    s = Stack(FakeSup(), [], {}, peak, idle_s=idle_s, hardware=lambda: box["hw"], say=lambda _: None, clock=lambda: now[0])
    s.now, s.box = now, box
    return s


def refused_with(code, fn, *a, **k):
    try:
        fn(*a, **k)
    except Refused as e:
        assert e.code == code, e.code
        return e
    raise AssertionError(f"not refused, wanted {code}")


def test_it_fits_or_it_doesnt():
    assert fit_problem(10, hw(free=12)) == ("", "")
    problem, warning = fit_problem(10, hw(free=9))                    # macOS can compress other apps to make room
    assert not problem and "Tight" in warning
    assert "Not enough" in fit_problem(10, hw(free=4))[0]
    assert "Not enough" in fit_problem(10, hw(free=9, kind="dedicated"))[0]    # a GPU can't
    assert fit_problem(10, hw(free=20, pressure="warn")) == ("", ""), "everyday on a 24 GB Mac, replies stay on time"
    assert "Tight" in fit_problem(10, hw(free=9, pressure="warn"))[1]
    assert "critically short" in fit_problem(10, hw(free=20, pressure="critical"))[0]


def test_idle_offloads_only_with_no_session():
    s = stack(); s.load()
    s.now[0] = 599; s.tick(); assert s.state == READY
    s.now[0] = 601; s.tick(); assert s.state == STOPPED and s.reason == IDLE and not s.sup.up
    assert s.ensure_ready() == 2 and s.state == READY, "a session wakes it"
    s.set_sessions(1); s.now[0] += 5000; s.tick(); assert s.state == READY, "not with a session open"
    s.set_sessions(0); s.now[0] += 599; s.tick(); assert s.state == READY, "the clock starts when the last session ends"
    s.now[0] += 2; s.tick(); assert s.state == STOPPED


def test_idle_zero_stays_loaded():
    s = stack(idle_s=0.0); s.load(); s.now[0] = 10**7; s.tick(); assert s.state == READY


def test_a_manual_offload_lasts_until_a_load():
    s = stack(); s.load(); s.offload(MANUAL)
    assert s.reason == MANUAL and not s.status()["wakes_on_session"]
    refused_with("engines_offloaded", s.ensure_ready)
    assert s.sup.starts == 1
    s.load(); assert s.state == READY and s.reason is None and s.ensure_ready() == 2


def test_offload_is_refused_with_a_session_unless_forced():
    s = stack(); s.load(); s.set_sessions(1)
    refused_with("session_active", s.offload)
    assert s.state == READY
    s.offload(MANUAL, force=True); assert s.state == STOPPED


def test_load_is_refused_when_it_wont_fit():
    s = stack(hw(free=2)); e = refused_with("not_enough_memory", s.load)
    assert "2.0 GB" in e.message and s.state == STOPPED and s.sup.starts == 0
    s.load(force=True); assert s.state == READY


def test_a_wake_is_refused_when_it_wont_fit():
    s = stack(); s.load(); s.now[0] = 601; s.tick(); assert s.reason == IDLE
    s.box["hw"] = hw(free=2)
    refused_with("not_enough_memory", s.ensure_ready)
    assert s.state == STOPPED and s.sup.starts == 1, "and it can try again"
    s.box["hw"] = hw(free=20); assert s.ensure_ready() and s.state == READY


def test_critical_pressure_offloads_when_idle_and_warns_in_a_session():
    s = stack(idle_s=0.0); s.load(); s.box["hw"] = hw(pressure="critical")
    s.set_sessions(1)
    for _ in range(PRESSURE_CHECKS + 3):
        s.tick()
    assert s.state == READY and "critically short" in s.warning, "never a live conversation"
    s.set_sessions(0)
    s.pressure_checks = 0
    for _ in range(PRESSURE_CHECKS - 1):
        s.tick()
    assert s.state == READY, "a short spike doesn't count"
    s.tick(); assert s.state == STOPPED and s.reason == MEMORY and s.status()["wakes_on_session"]


def test_warn_pressure_leaves_a_loaded_stack_alone():
    """seen on a 24 GB Mac: "warn" 15 s after loading, with replies on time. Acting on it would make the idle timer pointless."""
    s = stack(idle_s=600.0); s.load(); s.box["hw"] = hw(free=3, pressure="warn")
    for _ in range(PRESSURE_CHECKS * 10):
        s.tick()
    assert s.state == READY and not s.warning


def test_pressure_going_away_clears_its_warning_but_not_the_tight_fit():
    s = stack(hw(free=9), idle_s=0.0); s.load(); assert "Tight" in s.warning
    s.set_sessions(1); s.box["hw"] = hw(pressure="critical"); s.tick(); assert "critically" in s.warning
    s.box["hw"] = hw(free=9); s.tick(); assert "Tight" in s.warning and s.pressure_checks == 0
    s.set_sessions(0); s.offload(); assert not s.warning


def test_a_failed_load_says_why_and_can_retry():
    s = stack(); s.sup.fail = True
    refused_with("engines_failed", s.load)
    assert s.state == FAILED and "no model" in s.error and not s.sup.up
    refused_with("engines_failed", s.ensure_ready)
    s.sup.fail = False; s.load(); assert s.state == READY and not s.error


def test_an_engine_crash_needs_a_load():
    s = stack(); s.load(); s.on_engine_event("llm/x", "failed", "exited with code 9")
    assert s.state == FAILED and "llm/x" in s.error
    t0 = time.monotonic()
    while s.sup.up and time.monotonic() - t0 < 2:
        time.sleep(0.01)
    assert not s.sup.up, "the others stop"
    refused_with("engines_failed", s.ensure_ready)
    s.on_engine_event("llm/x", "failed", "ignored"); s.load(); assert s.state == READY


def test_a_failure_while_loading_isnt_a_crash():
    s = stack(); s.state = LOADING; s.on_engine_event("llm/x", "failed", "boom"); assert s.state == LOADING


def test_background_load_shows_loading_at_once_and_clears_the_old_error():
    s = stack(); s.sup.fail = True; refused_with("engines_failed", s.load); s.sup.fail = False
    release = threading.Event()
    real = s.sup.start_all
    s.sup.start_all = lambda *a, **k: (release.wait(5), real(*a, **k))
    s.load_in_background()
    assert s.state == LOADING and not s.error, (s.state, s.error)
    release.set()
    t0 = time.monotonic()
    while s.state != READY and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert s.state == READY


def test_a_session_during_an_idle_decision_keeps_the_engines():
    s = stack(); s.load(); s.now[0] = 601; s.set_sessions(1); s._auto_offload(IDLE, "x"); assert s.state == READY


def test_apple_parsers():
    from speakrail.hardware import _load_detect
    apple = _load_detect(ROOT / "platforms/apple")
    assert apple.parse_pressure("1\n") == "normal" and apple.parse_pressure("2") == "warn" and apple.parse_pressure("4") == "critical" and apple.parse_pressure("") == "normal"
    assert apple.parse_top("PID MEM\n123 5120M\n77* 2G+\nxx\n9 10K-\n") == {123: 5.0, 77: 2.0, 9: 10 / 2**20}


def test_colors_only_for_people():
    from speakrail.ui import paint
    text = "Speakrail is ready: http://127.0.0.1:8080/\n[llm/mlx] failed: boom\nok    Python 3.12\nFAIL  no model\n      -> fix it\n  llm/mlx            ready  4.1 GB\nsomething else"
    assert paint(text, on=False) == text
    colored = paint(text, on=True)
    assert "\033[" in colored and "\033[38;5;210m" in colored.split("\n")[1] and "\033[38;5;114m" in colored.split("\n")[5]
    assert __import__("re").sub(r"\033\[[0-9;]*m", "", colored) == text, "color never changes the words"
    assert colored.split("\n")[-1] == "something else"
    assert paint("x") == "x", "not a terminal here"


def test_a_voice_name_stays_a_name_and_must_exist():
    from speakrail.manifest import discover
    from speakrail.resolver import discover_configs, resolve
    from speakrail.run import missing_voice
    configs, engines = discover_configs(), discover()
    mac = Hardware("apple", "test", 24.0, 20.0, "unified", 4.0)
    if "mac-e4b" not in configs or sys.platform != "darwin":
        return
    with tempfile.TemporaryDirectory() as d:
        os.environ["MODELS_DIR"] = d
        import speakrail.run as run
        keep = run.paths
        run.paths = lambda: (Path(d), Path(d))
        try:
            r = resolve(configs, engines, mac, pinned="mac-e4b", models=d, home=d)
            assert r.harness["voice"] == "jane"
            assert "isn't installed" in missing_voice(r) and "jane.safetensors" in missing_voice(r)
            f = Path(d, "pocket/english/embeddings/jane.safetensors"); f.parent.mkdir(parents=True); f.touch()
            assert missing_voice(r) == ""
            clone = resolve(configs, engines, mac, pinned="mac-e4b", models=d, home=d, sets={"voice": "female_a.wav"})
            assert clone.harness["voice"].endswith("harness/voices/female_a.wav") and missing_voice(clone) == "", "a clip is still a clone"
        finally:
            run.paths = keep
            del os.environ["MODELS_DIR"]


# ---------------------------------------------------------------------- the whole thing on the stub engines
class Run:
    def __init__(self, *args, idle_min=None):
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        env = {**ENV, **({"IDLE_OFFLOAD_MIN": str(idle_min)} if idle_min is not None else {"IDLE_OFFLOAD_MIN": "0"})}
        self.p = subprocess.Popen([sys.executable, "-u", "-m", "speakrail", "run", "--config", "stub", "--no-bench", *args], env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)

    def text(self):
        return Path(self.log.name).read_text()

    def __enter__(self):
        t0 = time.monotonic()
        while "Speakrail is ready" not in self.text():
            assert self.p.poll() is None, self.text()[-1200:]
            assert time.monotonic() - t0 < 90, "not up: " + self.text()[-800:]
            time.sleep(0.1)
        return self

    def __exit__(self, *_):
        if self.p.poll() is None:
            os.kill(self.p.pid, signal.SIGTERM)
        self.p.wait(timeout=40)
        wait_until(lambda: engines_gone() and port_free(UI), "ports left open")


def token():
    return Path(HOME, "config/api_token").read_text().strip()


def http(method, path, body=None, headers=None, auth=True):
    h = {"Content-Type": "application/json", **(headers or {})}
    if auth:
        h["Authorization"] = f"Bearer {token()}"
    req = urllib.request.Request(BASE + path, method=method, data=None if body is None else json.dumps(body).encode(), headers=h)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def status():
    return http("GET", "/v1/control/status")[1]


def wait_until(cond, what, timeout=30):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, what
        time.sleep(0.1)


def engines_gone():
    return all(port_free(p) for p in ENGINE_PORTS)


def _talk():
    import test_stack_stub
    test_stack_stub.UI_PORT = UI
    return asyncio.run(talk())


def speaks(result):
    events, audio = result
    assert not [e for e in events if e.get("type") == "error"], [e for e in events if e.get("type") == "error"]
    assert any(e.get("decision") == "speak" for e in events if e.get("type") == "decision") and audio > 2000, audio


async def _first_events(url, first_message=None, want=("session.started", "error", "ws_error", "ready"), auth=True, timeout=40):
    """what the server says up to the first of `want`, on /ws or the Live API"""
    seen = []
    headers = {"Authorization": f"Bearer {token()}"} if auth else {}
    async with aiohttp.ClientSession() as c, c.ws_connect(BASE.replace("http", "ws") + url, headers=headers) as ws:
        if first_message:
            await ws.send_str(json.dumps(first_message))
        while True:
            m = await ws.receive(timeout=timeout)
            if m.type != aiohttp.WSMsgType.TEXT:
                seen.append({"type": "closed"})
                return seen
            e = json.loads(m.data)
            seen.append(e)
            if e.get("type") in want:
                return seen


START = {"type": "session.start", "session": {"model": "gpt-live-1"}}


def test_status_offload_and_load_through_the_api():
    with Run() as r:
        s = status()
        assert s["state"] == READY and set(s["engines"]) == {"asr/stub", "tts/stub", "llm/stub"} and s["sessions"] == 0, s
        assert all(e["state"] == "ready" for e in s["engines"].values())
        code, m = http("GET", "/v1/control/status?memory=1")
        assert code == 200 and all(e["memory_gib"] > 0 for e in m["engines"].values()), m
        code, s = http("POST", "/v1/control/offload", {})
        assert code == 200 and s["state"] == STOPPED and s["reason"] == MANUAL, s
        assert engines_gone(), "offloading frees the ports: the processes are gone"
        assert urllib.request.urlopen(BASE + "/").status == 200, "the page stays up"
        code, s = http("POST", "/v1/control/load", {})
        assert code == 202 and s["state"] == LOADING, s
        wait_until(lambda: status()["state"] == READY, "didn't load")
        speaks(_talk())
        assert r.p.poll() is None


def test_who_may_control_it():
    with Run():
        path = "/v1/control/status"
        assert http("GET", path, auth=False)[0] == 401, "curl with no token"
        assert http("GET", path, headers={"Authorization": "Bearer wrong"}, auth=False)[0] == 401
        assert http("POST", "/v1/control/offload", {}, auth=False)[0] == 401
        assert http("GET", path, headers={"Sec-Fetch-Site": "same-origin"}, auth=False)[0] == 200, "our page"
        assert http("POST", "/v1/control/offload", {}, headers={"Origin": "http://evil.example"}, auth=False)[0] == 403
        assert http("POST", "/v1/control/offload", {}, headers={"Origin": "http://evil.example", "Authorization": f"Bearer {token()}"}, auth=False)[0] == 403
        rebound = {"Host": "evil.example:18081", "Origin": "http://evil.example:18081"}
        assert http("POST", "/v1/control/offload", {}, headers=rebound, auth=False)[0] == 401, "DNS rebinding: it looks same-origin, the host gives it away"
        assert status()["state"] == READY, "none of those did anything"
        assert http("POST", "/v1/control/offload", {}, headers={"Origin": BASE}, auth=False)[0] == 200, "our page"
        assert status()["state"] == STOPPED
        assert http("POST", "/v1/control/load", "not an object")[0] == 400


def test_the_control_port_itself_wants_the_token():
    with Run():
        info = json.loads(Path(HOME, "run/host.json").read_text())
        req = urllib.request.Request(f"http://127.0.0.1:{info['port']}/status")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            opener.open(req, timeout=5); raise AssertionError("no token got in")
        except urllib.error.HTTPError as e:
            assert e.code == 401
        assert oct(Path(HOME, "run/host.json").stat().st_mode & 0o777) == "0o600"


def test_idle_offload_then_a_session_wakes_it():
    with Run(idle_min=0.05) as r:                                     # 3 s
        wait_until(lambda: status()["state"] == STOPPED, "didn't offload when idle", 15)
        s = status()
        assert s["reason"] == IDLE and s["wakes_on_session"] and engines_gone(), s
        assert "No session for" in r.text()
        speaks(_talk())                                               # the browser's /ws
        wait_until(lambda: status()["state"] == STOPPED, "didn't offload again", 15)
        seen = asyncio.run(_first_events("/v1/live/sessions", START))  # the API
        assert [e["type"] for e in seen] == ["info", "session.started"] and seen[0]["code"] == "loading", seen
        assert status()["generation"] == 3


def test_idle_doesnt_touch_an_open_session():
    with Run(idle_min=0.03):                                          # 2 s
        async def go():
            async with aiohttp.ClientSession() as c, c.ws_connect(BASE.replace("http", "ws") + "/v1/live/sessions", headers={"Authorization": f"Bearer {token()}"}) as ws:
                await ws.send_str(json.dumps(START))
                await asyncio.sleep(5)
                assert status()["state"] == READY and status()["sessions"] == 1, status()
        asyncio.run(go())
        wait_until(lambda: status()["state"] == STOPPED, "the clock didn't start when it ended", 15)


def test_the_browsers_mic_during_a_wake_is_dropped():
    """30 s of microphone sent while the engines load. Fed in afterwards, the recognizer would hear it all before anything said now,
    and the first word would date from 30 s of audio in. `fed_ms` is how much audio had reached the session when a word came."""
    with Run(idle_min=0.03):                                          # 2 s
        wait_until(lambda: status()["state"] == STOPPED, "didn't offload when idle", 15)
        async def go():
            async with aiohttp.ClientSession() as c, c.ws_connect(BASE.replace("http", "ws") + "/ws", max_msg_size=0) as ws:
                await ws.send_bytes(bytes(32000 * 30))
                words, got_info = [], False
                async def reader():
                    nonlocal got_info
                    async for m in ws:
                        if m.type == aiohttp.WSMsgType.TEXT:
                            e = json.loads(m.data)
                            got_info = got_info or (e.get("type") == "info" and e.get("code") == "loading")
                            if e.get("type") == "brain":
                                ready.set()
                            elif e.get("type") == "word":
                                words.append(e)
                ready = asyncio.Event()
                task = asyncio.create_task(reader())
                await asyncio.wait_for(ready.wait(), 30)
                tone = b"".join(int(6000 * math.sin(2 * math.pi * 300 * i / 16000)).to_bytes(2, "little", signed=True) for i in range(32000))
                for i in range(0, len(tone), 2560):
                    await ws.send_bytes(tone[i:i + 2560])
                    await asyncio.sleep(0.08)
                t0 = time.monotonic()
                while not words and time.monotonic() - t0 < 5:
                    await asyncio.sleep(0.05)
                task.cancel()
                return got_info, words
        got_info, words = asyncio.run(go())
        assert got_info, "no word that it was loading"
        assert words, "the stub said nothing"
        assert words[0]["fed_ms"] < 5000, f"late audio reached the recognizer: first word at {words[0]['fed_ms']} ms of audio"


def test_a_refused_browser_session_says_why_and_isnt_counted():
    with Run():
        http("POST", "/v1/control/offload", {})
        async def go():
            async with aiohttp.ClientSession() as c, c.ws_connect(BASE.replace("http", "ws") + "/ws") as ws:
                await ws.send_bytes(bytes(3200))
                out = []
                while True:
                    m = await ws.receive(timeout=20)
                    if m.type != aiohttp.WSMsgType.TEXT:
                        return out
                    out.append(json.loads(m.data))
        events = asyncio.run(go())
        assert events and events[-1]["type"] == "error" and events[-1]["code"] == "engines_offloaded", events
        assert status()["sessions"] == 0


def test_a_manual_offload_refuses_sessions_with_a_clear_error():
    with Run():
        http("POST", "/v1/control/offload", {})
        seen = asyncio.run(_first_events("/v1/live/sessions", START))
        err = next(e for e in seen if e["type"] == "error")
        assert err["error"]["code"] == "engines_offloaded" and "Load them" in err["error"]["message"], seen
        assert engines_gone() and status()["sessions"] == 0
        http("POST", "/v1/control/load", {})
        wait_until(lambda: status()["state"] == READY, "didn't load")
        assert [e["type"] for e in asyncio.run(_first_events("/v1/live/sessions", START))] == ["session.started"]


def test_offload_refused_with_a_session_then_forced():
    with Run():
        async def go():
            async with aiohttp.ClientSession() as c, c.ws_connect(BASE.replace("http", "ws") + "/v1/live/sessions", headers={"Authorization": f"Bearer {token()}"}) as ws:
                await ws.send_str(json.dumps(START))
                while json.loads((await ws.receive(timeout=20)).data)["type"] != "session.started":
                    pass
                code, r = await asyncio.to_thread(http, "POST", "/v1/control/offload", {})
                assert code == 409 and r["error"]["code"] == "session_active", r
                assert (await asyncio.to_thread(status))["state"] == READY
                code, r = await asyncio.to_thread(http, "POST", "/v1/control/offload", {"force": True})
                assert code == 200 and r["state"] == STOPPED, r
                t0 = time.monotonic()
                while (await ws.receive(timeout=20)).type not in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    assert time.monotonic() - t0 < 20
        asyncio.run(go())
        wait_until(lambda: status()["sessions"] == 0, "the session stayed counted")


def test_a_crash_reports_and_load_recovers():
    with Run() as r:
        pid = int(subprocess.run(["lsof", "-nP", "-iTCP:18000", "-sTCP:LISTEN", "-t"], capture_output=True, text=True).stdout.split()[0])
        os.kill(pid, signal.SIGKILL)
        wait_until(lambda: status()["state"] == FAILED, "the crash wasn't noticed", 15)
        s = status()
        assert "llm/stub" in s["error"] and "exited" in s["error"], s
        wait_until(engines_gone, "the others kept running")
        seen = asyncio.run(_first_events("/v1/live/sessions", START))
        assert next(e for e in seen if e["type"] == "error")["error"]["code"] == "engines_failed", seen
        http("POST", "/v1/control/load", {})
        wait_until(lambda: status()["state"] == READY, "didn't load")
        assert not status()["error"]
        speaks(_talk())
        assert r.p.poll() is None


def test_a_failed_load_leaves_a_failed_stack_not_a_dead_server():
    """an engine won't start again: its port is taken"""
    import socket
    with Run():
        http("POST", "/v1/control/offload", {})
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", 18000)); s.listen(1)
        try:
            http("POST", "/v1/control/load", {})
            wait_until(lambda: status()["state"] == FAILED, "the failure wasn't reported", 40)
            assert "18000" in status()["error"], status()
            assert all(port_free(p) for p in (18091, 18765)), "the engines already started were stopped again"
        finally:
            s.close()
        http("POST", "/v1/control/load", {})
        wait_until(lambda: status()["state"] == READY, "didn't recover")


async def _hold(url, first=None):
    """open a session and wait until it's in, leaving it open -> (client, ws)"""
    c = aiohttp.ClientSession()
    ws = await c.ws_connect(BASE.replace("http", "ws") + url, headers={"Authorization": f"Bearer {token()}"}, max_msg_size=0)
    if first:
        await ws.send_str(json.dumps(first))
    while True:
        m = await ws.receive(timeout=30)
        assert m.type == aiohttp.WSMsgType.TEXT, m
        if json.loads(m.data).get("type") in ("session.started", "brain"):
            return c, ws


def test_one_session_across_the_page_and_the_api():
    with Run():
        async def go():
            async def refused(url, first=None):
                seen = await _first_events(url, first, want=("error",))
                e = next(x for x in seen if x["type"] == "error")
                return e.get("code") or e["error"]["code"]
            c, ws = await _hold("/ws")
            assert await refused("/ws") == "session_busy", "a second tab"
            assert await refused("/v1/live/sessions", START) == "session_busy", "the API, with a tab open"
            assert (await asyncio.to_thread(status))["sessions"] == 1
            await ws.close(); await c.close()
            c, ws = await _hold("/v1/live/sessions", START)
            assert await refused("/ws") == "session_busy", "a tab, with the API open"
            assert not ws.closed, "the session already in keeps going"
            await ws.close(); await c.close()
            c, ws = await _hold("/debug/ws")                       # the place was given back, and this page counts too
            assert await refused("/ws") == "session_busy"
            await ws.close(); await c.close()
        asyncio.run(go())
        wait_until(lambda: status()["sessions"] == 0, "the session stayed counted")


def test_stop_then_start_straight_away_works():
    with Run():
        async def go():
            for _ in range(3):
                c, ws = await _hold("/ws")
                await ws.close(); await c.close()
        asyncio.run(go())


def test_the_tuned_settings_cant_be_set():
    r = subprocess.run([sys.executable, "-m", "speakrail", "run", "--config", "stub", "--set", "turn_k=2"], env=ENV, capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert r.returncode == 2 and "tuned" in r.stdout and "[asr/stub]" not in r.stdout, r.stdout[-400:]
    r = subprocess.run([sys.executable, "-m", "speakrail.harness.server", "--tool-hold-ms", "1"], env=ENV, capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert r.returncode == 2 and "unrecognized" in r.stderr, r.stderr[-300:]
    with Run() as run:
        conf = json.loads(Path(HOME, "config/resolved.json").read_text())
        assert conf["turn_k"] == 1


# ---------------------------------------------------------------------- the commands
def cli(*args, timeout=120):
    return subprocess.run([sys.executable, "-m", "speakrail", *args], env={**ENV, "IDLE_OFFLOAD_MIN": "0"}, capture_output=True, text=True, timeout=timeout, cwd=ROOT)


def test_start_status_logs_load_offload_stop():
    assert cli("status").returncode == 3
    assert "Not running" in cli("stop").stdout
    r = cli("start", "--config", "stub", "--no-bench")
    try:
        assert r.returncode == 0 and "Speakrail is ready" in r.stdout, r.stdout[-600:] + r.stderr[-300:]
        s = cli("status")
        assert s.returncode == 0 and "Running (pid" in s.stdout and "asr/stub" in s.stdout and "ready" in s.stdout and " GB" in s.stdout, s.stdout
        again = cli("start", "--config", "stub")
        assert again.returncode == 0 and "Already running" in again.stdout
        assert cli("run", "--config", "stub").returncode == 2, "a second one in the foreground is refused too"
        assert "[asr/stub]" in cli("logs").stdout
        assert cli("logs", "harness", "-n", "5").returncode == 0
        assert cli("logs", "asr/stub").returncode == 0
        missing = cli("logs", "nothing")
        assert missing.returncode == 1 and "harness" in missing.stdout
        assert cli("offload").returncode == 0 and engines_gone()
        assert "offloaded on request" in cli("status").stdout
        assert cli("load").returncode == 0 and not engines_gone()
        speaks(_talk())
    finally:
        out = cli("stop")
    assert out.returncode == 0 and "Stopped" in out.stdout, out.stdout
    assert engines_gone() and port_free(UI) and not Path(HOME, "run/host.json").exists()
    assert cli("status").returncode == 3


def test_a_stale_host_file_is_ignored():
    f = Path(HOME, "run/host.json"); f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"pid": 999999, "port": 1, "url": "x"}))
    assert cli("status").returncode == 3 and not f.exists()


if __name__ == "__main__":
    assert os.environ.get("SPEAKRAIL_TOKENIZER_DIR"), "set SPEAKRAIL_TOKENIZER_DIR"
    only = sys.argv[1:]
    tests = [v for k, v in globals().items() if k.startswith("test_") and (not only or any(o in k for o in only))]
    bad = 0
    for t in tests:
        try:
            t0 = time.monotonic(); t()
            print(f"PASS {t.__name__} ({time.monotonic() - t0:.0f} s)", flush=True)
        except Exception as e:
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {str(e)[:900]}", flush=True)
    print(f"{len(tests) - bad} of {len(tests)} passed")
    raise SystemExit(1 if bad else 0)
