#!/usr/bin/env python3
"""`speakrail run` end to end on the stub engines, with no GPU and no models. Needs the tokenizer, as test_session.py does.

    SPEAKRAIL_TOKENIZER_DIR=... python tests/test_run.py
"""
import asyncio, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
HOME = tempfile.mkdtemp(prefix="speakrail-run-")
os.environ["SPEAKRAIL_HOME"] = HOME
from speakrail.supervisor import check_health, port_free
from test_stack_stub import talk

UI = 8080
PORTS = (UI, 18000, 18091, 18765)
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1", "SPEAKRAIL_HOME": HOME, "UI_PORT": "18080"}
UI = 18080


class Run:
    def __init__(self, *args):
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        self.p = subprocess.Popen([sys.executable, "-u", "-m", "speakrail", "run", "--config", "stub", *args], env=ENV, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)

    def text(self):
        return Path(self.log.name).read_text()

    def wait_up(self, timeout=90):
        t0 = time.monotonic()
        while "Speakrail is ready" not in self.text():
            assert self.p.poll() is None, self.text()[-1200:]
            assert time.monotonic() - t0 < timeout, "not up: " + self.text()[-800:]
            time.sleep(0.1)
        assert check_health(f"http://127.0.0.1:{UI}/")

    def stop(self):
        if self.p.poll() is None:
            os.kill(self.p.pid, signal.SIGTERM)
        return self.p.wait(timeout=30)


def wait_ports_free():
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5:
        if all(port_free(p) for p in PORTS):
            return True
        time.sleep(0.1)
    return False


def _talk():
    import test_stack_stub
    test_stack_stub.UI_PORT = UI
    return asyncio.run(talk())


def test_first_launch_measures_and_second_reuses():
    r = Run()
    try:
        r.wait_up()
        out = r.text()
        assert "Measuring this machine" in out and "Measured:" in out, out[-800:]
        saved = json.loads(Path(HOME, "config/benchmark.json").read_text())
        assert saved["result"]["llm_decode_tok_s"] > 0 and saved["result"]["asr_frame_ms"] > 0, saved
        conf = json.loads(Path(HOME, "config/resolved.json").read_text())
        assert conf["asr_url"].endswith(":18765/v1/transcribe") and conf["tts_engine"] == "pocket", conf
    finally:
        assert r.stop() == 0, r.text()[-500:]
    assert wait_ports_free(), "ports left open"
    r = Run()
    try:
        r.wait_up()
        assert "Measuring this machine" not in r.text()
        assert "Benchmark ASR" in r.text(), r.text()[:700]
    finally:
        r.stop()
    assert wait_ports_free()


def test_a_conversation_through_run():
    r = Run("--no-bench")
    try:
        r.wait_up()
        events, audio = _talk()
        assert not [e for e in events if e.get("type") == "error"]
        assert any(e.get("decision") == "speak" for e in events if e.get("type") == "decision")
        assert audio > 2000, audio
    finally:
        r.stop()
    assert wait_ports_free()


def test_a_vad_only_asr_still_ends_turns():
    """no real turn head: the frames say speech or not, and the harness's silence fallback ends the turn"""
    r = Run("--no-bench", "--param", "asr/stub:flags=vad_head")
    try:
        r.wait_up()
        events, audio = _talk()
        assert not [e for e in events if e.get("type") == "error"]
        decisions = [(e.get("why"), e.get("decision")) for e in events if e.get("type") == "decision"]
        assert any(d == "speak" for _, d in decisions), decisions
        assert audio > 2000, audio
        sessions = [e for e in events if e.get("type") == "summary"]
    finally:
        r.stop()
    assert wait_ports_free()


def _pid_on(port):
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True).stdout.split()
    return int(out[0])


def test_an_engine_dying_stops_everything():
    r = Run("--no-bench")
    try:
        r.wait_up()
        os.kill(_pid_on(18000), signal.SIGKILL)
        code = r.p.wait(timeout=15)
        assert code == 1, (code, r.text()[-500:])
        assert "llm/stub" in r.text() and "Stopping" in r.text()
    finally:
        r.stop()
    assert wait_ports_free(), "ports left open"


def test_the_harness_dying_stops_everything():
    r = Run("--no-bench")
    try:
        r.wait_up()
        os.kill(_pid_on(UI), signal.SIGKILL)
        assert r.p.wait(timeout=15) == 1, r.text()[-500:]
        assert "the harness exited" in r.text()
    finally:
        r.stop()
    assert wait_ports_free()


def test_a_busy_ui_port_is_refused_before_loading_anything():
    import socket
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", UI)); s.listen(1)
    r = Run("--no-bench")
    try:
        assert r.p.wait(timeout=20) == 2, r.text()[-400:]
        assert "already in use" in r.text() and "[asr/stub]" not in r.text()
    finally:
        s.close()


def test_setting_a_value_reaches_the_harness():
    r = Run("--no-bench", "--set", "notes=true", "--set", "silence_ms=700")
    try:
        r.wait_up()
        conf = json.loads(Path(HOME, "config/resolved.json").read_text())
        assert conf["notes"] is True and conf["silence_ms"] == 700, conf
    finally:
        r.stop()
    assert wait_ports_free()


if __name__ == "__main__":
    assert os.environ.get("SPEAKRAIL_TOKENIZER_DIR"), "set SPEAKRAIL_TOKENIZER_DIR"
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
