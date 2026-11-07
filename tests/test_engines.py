#!/usr/bin/env python3
"""The engine foundation, with no GPU and no models: manifests, the supervisor, the stub engines' conformance, and that the
conformance checks really catch a broken engine.

    python tests/test_engines.py
"""
import asyncio, os, re, signal, sys, tempfile, textwrap, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]
TMP = tempfile.mkdtemp(prefix="speakrail-test-")
os.environ["SPEAKRAIL_HOME"] = TMP
from speakrail.contracts.conformance import check
from speakrail.manifest import ManifestError, discover, load_manifest, render, values_for
from speakrail.supervisor import EngineError, FAILED, READY, STOPPED, Supervisor, check_health, port_free

ENGINES = discover([ROOT / "engines"])
STUBS = [ENGINES[k] for k in ("asr/stub", "tts/stub", "llm/stub")]


def wait_for(fn, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.05)
    return False


def write_manifest(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "engine.toml").write_text(textwrap.dedent(text))
    return folder / "engine.toml"


GOOD = """
    contract = 1
    role = "tts"
    name = "x"
    capabilities = []
    platforms = ["darwin", "linux"]
    url = "http://127.0.0.1:${port_main}/"
    [ports]
    main = 18999
    [[process]]
    name = "server"
    cmd = ["sleep", "60"]
    health = "tcp://127.0.0.1:${port_main}"
"""


def test_every_shipped_manifest_loads():
    assert {"llm/stub", "asr/stub", "tts/stub", "llm/mlx", "asr/audiocpp", "tts/pocket"} <= set(ENGINES), sorted(ENGINES)


def test_the_audiocpp_pin_is_a_commit_and_builds_both_engines():
    import tomllib
    pin = tomllib.loads((ROOT / "engines/audiocpp.toml").read_text())
    assert re.fullmatch(r"[0-9a-f]{8,40}", pin["source"]["commit"]), pin["source"]["commit"]
    assert "-DAUDIOCPP_MODELS=voxtral_realtime;pocket_tts" in pin["build"]["darwin"]["args"]


def test_manifest_mistakes_are_errors():
    bad = {"role": GOOD.replace('role = "tts"', 'role = "video"'), "contract": GOOD.replace("contract = 1", "contract = 99"),
           "capability": GOOD.replace("capabilities = []", 'capabilities = ["flying"]'), "unknown key": '    colour = "red"\n' + GOOD,
           "no main port": GOOD.replace("main = 18999", "other = 18999"), "no health": GOOD.replace('health = "tcp://127.0.0.1:${port_main}"', ""),
           "syntax": GOOD + "\n    this is not toml\n"}
    for name, text in bad.items():
        try:
            load_manifest(write_manifest(Path(TMP) / "m" / name.replace(" ", "_"), text))
        except ManifestError:
            continue
        raise AssertionError(f"accepted a manifest with a bad {name}")


def test_folder_must_match_the_manifest():
    root = Path(TMP) / "mismatch"
    write_manifest(root / "tts" / "wrong", GOOD)
    try:
        discover([root])
    except ManifestError:
        return
    raise AssertionError("accepted tts/wrong declaring tts/x")


def test_a_local_engine_overrides_a_shipped_one():
    root = Path(TMP) / "local"
    write_manifest(root / "tts" / "stub", GOOD.replace('name = "x"', 'name = "stub"').replace("capabilities = []", 'description = "mine"\n    capabilities = []'))
    assert discover([ROOT / "engines", root])["tts/stub"].description == "mine"


def test_templates_and_params():
    e = ENGINES["llm/mlx"]
    v = values_for(e, Path("/h"), Path("/m"), Path("/r"), {"max_ctx": "4096"}, python="/py")
    assert (v["model"], v["max_ctx"], v["port_main"], v["python"]) == ("/m/speakrail-e4b-mlx", "4096", "8000", "/py")
    assert v["request_log"] == "/h/logs/llm-mlx/requests.jsonl"
    for overrides in ({"nope": "1"},):
        try:
            values_for(e, Path("/h"), Path("/m"), Path("/r"), overrides)
        except ManifestError:
            break
    else:
        raise AssertionError("accepted an unknown param")
    try:
        render("${missing}", {})
    except ManifestError:
        return
    raise AssertionError("accepted an unknown variable")


def test_start_order_and_stop():
    events = []
    sup = Supervisor(on_event=lambda e, s, d: events.append((e, s)))
    try:
        sup.start_all([STUBS[2], STUBS[1], STUBS[0]])              # given in the wrong order on purpose
        ready = [e for e, s in events if s == READY]
        assert ready == ["asr/stub", "tts/stub", "llm/stub"], ready
        assert all(sup.state(s.id) == READY for s in STUBS)
        assert check_health(sup.url("tts/stub").replace("/v1/audio/speech", "/health"))
        pids = [pid for s in STUBS for pid in sup.pids(s.id).values()]
        sup.stop_all()
        assert all(sup.state(s.id) == STOPPED for s in STUBS)
        assert all(not _alive(p) for p in pids)
        assert all(port_free(int(s.ports["main"])) for s in STUBS)
        sup.start(STUBS[2])                                        # and it can start again
        assert sup.state("llm/stub") == READY
    finally:
        sup.close()


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_a_killed_engine_is_caught():
    events = []
    sup = Supervisor(on_event=lambda e, s, d: events.append((e, s, d)))
    try:
        sup.start(STUBS[0])
        os.kill(sup.pids("asr/stub")["server"], signal.SIGKILL)
        t0 = time.monotonic()
        assert wait_for(lambda: sup.state("asr/stub") == FAILED), "not noticed"
        took = time.monotonic() - t0
        assert took < 2, took
        assert "exited with code" in sup.error("asr/stub") and any(s == FAILED for _, s, _ in events)
    finally:
        sup.close()


def test_a_busy_port_is_refused():
    import socket
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 18000)); s.listen(1)
    sup = Supervisor()
    try:
        sup.start(STUBS[2])
    except EngineError as e:
        msg = str(e)
        assert "already using it" in msg and "close that program" in msg, msg
        assert f"--param {STUBS[2].id}:port_main=" in msg, "it names a free port to move to"
        moved = int(re.search(r"port_main=(\d+)", msg)[1])
        r = sup.start(STUBS[2], {"port_main": str(moved)})        # and that works
        assert r.state == READY and str(moved) in sup.url(STUBS[2].id), sup.url(STUBS[2].id)
        return
    finally:
        s.close(); sup.close()
    raise AssertionError("started on a busy port")


def test_a_port_is_overridden_like_a_param():
    e = STUBS[2]
    v = values_for(e, Path(TMP), Path(TMP), Path(TMP), {"port_main": "18555"})
    assert v["port_main"] == "18555" and "18555" in render(e.url, v, e.id)
    for bad in ({"port_main": "http"}, {"port_main": "70000"}, {"port_nope": "1"}):
        try:
            values_for(e, Path(TMP), Path(TMP), Path(TMP), bad)
            raise AssertionError(f"accepted {bad}")
        except ManifestError:
            pass


def test_a_process_dying_during_startup_fails_and_cleans_up():
    root = Path(TMP) / "dying"
    two = GOOD.replace('name = "x"', 'name = "dying"').replace('cmd = ["sleep", "60"]', 'cmd = ["sh", "-c", "exit 3"]').replace('health = "tcp://127.0.0.1:${port_main}"', 'health = "http://127.0.0.1:1/never"')
    e = load_manifest(write_manifest(root / "tts" / "dying", two))
    sup = Supervisor()
    try:
        t0 = time.monotonic()
        sup.start(e)
    except EngineError as err:
        assert "code 3" in str(err), err
        assert time.monotonic() - t0 < 3
        assert sup.state(e.id) == FAILED
        return
    finally:
        sup.close()
    raise AssertionError("no failure")


def test_a_never_healthy_engine_times_out():
    root = Path(TMP) / "slow"
    text = GOOD.replace('name = "x"', 'name = "slow"').replace("capabilities = []", "capabilities = []\n    startup_timeout_s = 1")
    e = load_manifest(write_manifest(root / "tts" / "slow", text))
    sup = Supervisor()
    try:
        sup.start(e)
    except EngineError as err:
        assert "not healthy" in str(err), err
        pid_gone = wait_for(lambda: all(p.poll() is not None for p in sup.running[e.id].procs.values()))
        assert pid_gone
        return
    finally:
        sup.close()
    raise AssertionError("no timeout")


def test_secrets_stay_out_of_engines():
    root = Path(TMP) / "env"
    out = Path(TMP) / "env.txt"
    text = GOOD.replace('name = "x"', 'name = "envdump"').replace('cmd = ["sleep", "60"]', f'cmd = ["sh", "-c", "env > {out}; sleep 60"]')
    text = text.replace('health = "tcp://127.0.0.1:${port_main}"', 'health = ""').replace("[[process]]", "[[process]]", 1)
    text += '\n    [[process]]\n    name = "up"\n    cmd = ["python3", "-m", "http.server", "18999", "--bind", "127.0.0.1"]\n    health = "tcp://127.0.0.1:${port_main}"\n'
    e = load_manifest(write_manifest(root / "tts" / "envdump", text))
    os.environ["FIREWORKS_API_KEY"] = "secret-a"; os.environ["HF_TOKEN"] = "secret-b"; os.environ["SPEAKRAIL_PLAIN"] = "visible"
    sup = Supervisor()
    try:
        sup.start(e)
        assert wait_for(lambda: out.exists() and "SPEAKRAIL_PLAIN" in out.read_text())
        dumped = out.read_text()
        assert "secret-a" not in dumped and "secret-b" not in dumped
    finally:
        sup.close()
        del os.environ["FIREWORKS_API_KEY"], os.environ["HF_TOKEN"], os.environ["SPEAKRAIL_PLAIN"]


def _conformance(engine, flags=""):
    sup = Supervisor()
    try:
        sup.start(engine, {"flags": flags})
        return asyncio.run(check(engine, sup.url(engine.id)))
    finally:
        sup.close()


def test_stubs_pass_conformance():
    for e in STUBS:
        results = _conformance(e)
        failed = [r for r in results if not r.ok]
        assert results and not failed, (e.id, failed)


def test_conformance_catches_broken_engines():
    for engine, flag, expected in ((STUBS[2], "ignore_allowed", "allowed set"), (STUBS[2], "none_on_eos", "end-of-turn"), (STUBS[2], "ignore_bias", "logit_bias"),
                                   (STUBS[2], "no_prefix_cache", "prefix_cache"), (STUBS[2], "greedy_only", "sampling"), (STUBS[0], "no_peek", "peek"), (STUBS[1], "silent", "silent")):
        results = _conformance(engine, flag)
        failed = [r.name for r in results if not r.ok]
        assert any(expected in n for n in failed) or any(expected in r.name for r in results if not r.ok), (flag, failed)


def test_requests_carry_the_sampling_and_the_bias():
    from speakrail.harness.llm_engine import Engine

    def sent(engine):
        return engine.stream(None, [1, 2], 8, [3]).body
    greedy = sent(Engine())
    assert greedy["temperature"] == 0.0 and "top_p" not in greedy and "top_k" not in greedy
    off = sent(Engine(reply_temp=0.0, reply_top_p=0.95, reply_top_k=64))    # top_p and top_k only mean something when sampling
    assert "top_p" not in off and "top_k" not in off
    body = sent(Engine(reply_temp=0.7, reply_top_p=0.95, reply_top_k=64))
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.7, 0.95, 64)

    class Decide(Engine):
        async def _post(self, http, url, body):
            self.body = body
            return {"choices": [{"logprobs": {"tokens": ["token_id:106"], "top_logprobs": [{"token_id:106": -0.1, "token_id:16": -3.0}]}}]}
    e = Decide(reply_temp=0.7, reply_top_p=0.95, reply_top_k=64)
    asyncio.run(e.decide(None, [1], [106, 16], {106: 1.0, 20: 5.0}))
    assert e.body["temperature"] == 0.0 and "top_k" not in e.body           # decisions stay greedy
    assert e.body["logit_bias"] == {"106": 1.0}                             # only for tokens this decision allows


def test_the_harness_passes_the_settings_on():
    from dataclasses import replace
    from types import SimpleNamespace
    from speakrail.config import load
    from speakrail.harness import server
    args = SimpleNamespace(mt_url="http://x/v1/completions", undo_words=2, persona="nova", no_interject=False, notes=False, no_phrase_cache=True, safety_yield_s=1.5,
                           claude=False, claude_model="", claude_mode="auto", claude_cwd="~", frontier=False)
    was = server.ARGS, server.CFG
    server.ARGS, server.CFG = args, replace(load({}), decision_bias="106:1.0", reply_temp=0.7, reply_top_p=0.95, reply_top_k=64)
    try:
        m = server.micro_cfg({}, SimpleNamespace(silence_ms=1000))
    finally:
        server.ARGS, server.CFG = was
    assert m.bias == {106: 1.0} and (m.reply_temp, m.reply_top_p, m.reply_top_k) == (0.7, 0.95, 64)



FAKE_DOCKER = """#!/bin/sh
# stands in for docker: records every call, and `run` starts the image's "server", a small HTTP server on the port in its last argument
echo "$@" >> "$DOCKER_LOG"
case "$1" in
  info) echo '{"nvidia":{},"runc":{}}' ;;
  image) [ "$MISSING_IMAGE" != 1 ] ;;
  stop) [ -f "$DOCKER_PID" ] && kill "$(cat "$DOCKER_PID")" 2>/dev/null; rm -f "$DOCKER_PID" ;;
  rm) [ -f "$DOCKER_PID" ] && kill -9 "$(cat "$DOCKER_PID")" 2>/dev/null; rm -f "$DOCKER_PID" ;;
  run) for last; do :; done; echo $$ > "$DOCKER_PID"; exec "$FAKE_PYTHON" -m http.server "$last" --bind 127.0.0.1 ;;
esac
"""


def _container_engine(root, port=18790):
    d = Path(root) / "x" / "box"
    d.mkdir(parents=True)
    (d / "engine.toml").write_text(textwrap.dedent(f"""\
        contract = 1
        role = "tts"
        name = "box"
        capabilities = ["stream"]
        platforms = ["{sys.platform}"]
        url = "http://127.0.0.1:${{port_main}}/"
        startup_timeout_s = 15
        [ports]
        main = {port}
        [params]
        image = "example/image:1"
        [[process]]
        name = "server"
        image = "${{image}}"
        entrypoint = "/app/server"
        gpu = true
        mounts = ["${{models}}:/models:ro"]
        docker_args = ["--ipc=host"]
        env = {{ A = "${{port_main}}" }}
        cmd = ["--port", "${{port_main}}"]
        health = "http://127.0.0.1:${{port_main}}/"
        """))
    return load_manifest(d / "engine.toml")


def test_a_container_engine_starts_and_stops_by_name():
    root = tempfile.mkdtemp(prefix="speakrail-docker-")
    bin_dir, log, pid = Path(root) / "bin", Path(root) / "docker.log", Path(root) / "docker.pid"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    (bin_dir / "docker").chmod(0o755)
    saved = {k: os.environ.get(k) for k in ("PATH", "DOCKER_LOG", "DOCKER_PID", "FAKE_PYTHON", "MISSING_IMAGE")}
    os.environ.update(PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", DOCKER_LOG=str(log), DOCKER_PID=str(pid), FAKE_PYTHON=sys.executable)
    os.environ.pop("MISSING_IMAGE", None)
    e = _container_engine(root)
    sup = Supervisor(home=root, models=Path(root) / "models")
    try:
        sup.start(e)
        assert sup.state(e.id) == READY
        run = next(l for l in log.read_text().splitlines() if l.startswith("run "))
        for part in ("--name speakrail-tts-box-server", "--network host", "--no-healthcheck", "--gpus all", f"-v {Path(root) / 'models'}:/models:ro", "-e A=18790", "--entrypoint /app/server", "--ipc=host", "example/image:1 --port 18790"):
            assert part in run, (part, run)
        assert "rm -f speakrail-tts-box-server" in log.read_text()         # clears a leftover first
        calls_before = len(log.read_text().splitlines())
        sup.stop(e.id)
        after = log.read_text().splitlines()[calls_before:]
        assert any(l.startswith("stop ") and "speakrail-tts-box-server" in l for l in after) and any(l == "rm -f speakrail-tts-box-server" for l in after), after
        assert wait_for(lambda: port_free(18790)) and sup.state(e.id) == STOPPED
        sup.start(e)                                      # a supervisor that was killed leaves its container holding the port: the next start clears it
        crashed = Supervisor(home=root, models=Path(root) / "models")
        sup.running.clear()
        try:
            crashed.start(e)
            assert crashed.state(e.id) == READY
        finally:
            crashed.close()
        assert wait_for(lambda: port_free(18790))
        os.environ["MISSING_IMAGE"] = "1"                 # a clear message, not a hung start
        try:
            sup.start(e)
            raise AssertionError("started without its image")
        except EngineError as err:
            assert "isn't on this machine" in str(err), err
    finally:
        sup.close()
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_container_fields_need_an_image():
    d = Path(tempfile.mkdtemp(prefix="speakrail-manifest-")) / "engine.toml"
    d.write_text(textwrap.dedent("""\
        contract = 1
        role = "tts"
        name = "x"
        capabilities = []
        platforms = ["linux"]
        url = "http://x/"
        [ports]
        main = 1
        [[process]]
        name = "a"
        cmd = ["b"]
        gpu = true
        health = "tcp://x:1"
        """))
    try:
        load_manifest(d)
        raise AssertionError("gpu without an image was accepted")
    except ManifestError as e:
        assert "only mean something" in str(e), e


def test_every_nvidia_engine_is_a_container_with_a_health_check():
    for key in ("asr/audiocpp-docker", "tts/pocket-docker", "tts/breeze-docker", "llm/vllm-docker"):
        e = ENGINES[key]
        assert e.in_containers and all(p.image for p in e.processes) and any(p.health for p in e.processes), key
    assert ENGINES["llm/vllm-docker"].needs_gpu_runtime and not ENGINES["tts/pocket-docker"].needs_gpu_runtime


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for t in tests:
        try:
            t0 = time.monotonic(); t()
            print(f"PASS {t.__name__} ({time.monotonic() - t0:.1f} s)")
        except Exception as e:
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {str(e)[:400]}")
    print(f"{len(tests) - bad} of {len(tests)} passed")
    raise SystemExit(1 if bad else 0)
