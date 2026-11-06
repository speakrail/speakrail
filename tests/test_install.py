#!/usr/bin/env python3
"""The installer's own decisions, with nothing downloaded or built. A real install is a separate, much longer run.

    python tests/test_install.py
"""
import argparse, hashlib, os, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]
HOME = tempfile.mkdtemp(prefix="speakrail-install-")
os.environ["SPEAKRAIL_HOME"] = HOME
from speakrail.hardware import Hardware, _load_detect
from speakrail.installer import Choices, File, InstallError, RC_MARK, add_to_path, missing, rc_file, select, wizard, write_command
from speakrail.resolver import discover_configs, load_config

MAC = Hardware("apple", "test", 24.0, 20.0, "unified", 4.0)
apple = _load_detect(ROOT / "platforms/apple")


def test_flags_follow_what_the_chip_reports():
    m4 = {"FEAT_BF16": "1", "FEAT_I8MM": "1", "FEAT_DotProd": "1", "FEAT_FP16": "1"}
    assert apple.parse_arch(m4) == "armv8.6-a+dotprod+i8mm+fp16", "what the measurements were built with"
    m1 = {"FEAT_BF16": "0", "FEAT_I8MM": "0", "FEAT_DotProd": "1", "FEAT_FP16": "1"}
    assert apple.parse_arch(m1) == "armv8.4-a+dotprod+fp16", "armv8.6 would turn on BF16 and I8MM, which it doesn't have"
    assert apple.parse_arch({"FEAT_I8MM": "1", "FEAT_DotProd": "1"}) == "armv8.4-a+dotprod+i8mm", "half of them isn't 8.6"
    assert apple.parse_arch({}) == "armv8.4-a"


def test_a_config_download_is_checked():
    with tempfile.TemporaryDirectory() as d:
        base = 'name="x"\nstatus="experimental"\nplatform="any"\nmemory_peak_gib=1\n[engines]\nasr="a"\ntts="t"\nllm="l"\n'
        for bad, word in (('[[download]]\nrepo="r"\n', "missing"), ('[[download]]\nrepo="r"\nrevision="v"\ninclude=["x"]\nto="y"\nnope=1\n', "unknown")):
            f = Path(d, "c.toml"); f.write_text(base + bad)
            try:
                load_config(f); raise AssertionError("accepted: " + bad)
            except Exception as e:
                assert word in str(e), e


def test_the_mac_config_fetches_what_the_engines_need():
    from speakrail.manifest import discover, render, values_for
    c = discover_configs()["mac-e4b"]
    assert all(__import__("re").fullmatch(r"[0-9a-f]{40}", d["revision"]) for d in c.downloads), "every download is pinned to a commit"
    folders = {d["to"] for d in c.downloads}
    engines = discover()
    with tempfile.TemporaryDirectory() as d:
        for eid in c.engines.values():
            e = engines[eid]
            values = values_for(e, Path(d), Path(d, "m"), Path(d, "r"), {})
            for need in e.needs:
                path = Path(render(need, values, eid))
                if str(path).startswith(str(Path(d, "m"))):
                    rel = str(path.relative_to(Path(d, "m")))
                    assert any(rel.startswith(f + "/") for f in folders), f"{eid} needs {rel}, and nothing fetches it"


def listing():
    return [("base/a.gguf", 10, "s"), ("base/sub/b.bin", 5, None), ("base/skip.txt", 1, None), ("base/.gitattributes", 1, None)]


def test_files_land_where_the_engines_look():
    entry = {"repo": "r", "revision": "v", "from": "base", "include": ["a.gguf", "sub/*"], "to": "here"}
    files = select(listing(), entry, Path("/m"))
    assert [str(f.dest) for f in files] == ["/m/here/a.gguf", "/m/here/sub/b.bin"] and files[0].path == "base/a.gguf"
    every = select(listing(), {**entry, "include": ["*"]}, Path("/m"))
    assert ".gitattributes" not in [f.dest.name for f in every]
    root = select([("t.vxth", 1, None)], {"repo": "r", "revision": "v", "include": ["t.vxth"], "to": "h"}, Path("/m"))
    assert str(root[0].dest) == "/m/h/t.vxth"


def test_a_missing_upload_says_so():
    try:
        select([], {"repo": "speakrail/x", "revision": "main", "from": "mlx/model", "include": ["*"], "to": "y"}, Path("/m"))
        raise AssertionError("no error")
    except InstallError as e:
        assert "speakrail/x/mlx/model" in str(e) and "may not have these files yet" in str(e)


class Server:
    """a file over HTTP with range requests, which can fail on chosen chunks"""
    def __init__(self, data):
        self.data, self.requests, self.fail, self.cut = data, [], set(), False
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                a, b = (int(x) for x in self.headers["Range"].removeprefix("bytes=").split("-"))
                outer.requests.append(a)
                if a in outer.fail:
                    self.send_response(503); self.send_header("Content-Length", "0"); self.end_headers(); return
                self.send_response(206)
                body = outer.data[a:b + 1]
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}/f"


def test_a_download_picks_up_where_it_stopped():
    import speakrail.installer as inst
    keep = inst.CHUNK
    inst.CHUNK = 1000
    try:
        data = os.urandom(10_500)
        sha = hashlib.sha256(data).hexdigest()
        s = Server(data)
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d, "sub/model.bin")
            s.fail = {4000, 7000}
            try:
                inst.download(s.url, dest, len(data), sha, retry_wait=0)
                raise AssertionError("two chunks can't be fetched, and it says nothing")
            except InstallError as e:
                assert "model.bin" in str(e)
            assert not dest.exists() and Path(d, "sub/model.bin.part").exists(), "not under its name until it's whole"
            assert sorted(int(x) for x in Path(d, "sub/model.bin.part.done").read_text().split()) == [i for i in range(11) if i not in (4, 7)]
            s.fail, s.requests = set(), []
            inst.download(s.url, dest, len(data), sha, retry_wait=0)
            assert sorted(s.requests) == [4000, 7000], f"only what was missing, not {s.requests}"
            assert dest.read_bytes() == data and sorted(p.name for p in dest.parent.iterdir()) == ["model.bin"], "nothing else left"
    finally:
        inst.CHUNK = keep


def test_a_damaged_download_is_caught_and_starts_over():
    data = os.urandom(5000)
    s = Server(data)
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d, "m.bin")
        try:
            __import__("speakrail.installer").installer.download(s.url, dest, len(data), "0" * 64)
            raise AssertionError("damage not caught")
        except InstallError as e:
            assert "damaged" in str(e)
        assert list(Path(d).iterdir()) == [], "a damaged file never reaches its place"


def test_a_short_answer_is_not_a_finished_chunk():
    import speakrail.installer as inst
    data = os.urandom(3000)
    s = Server(data)
    calls = []
    real = inst.requests.get
    class Short:
        status_code = 206
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_content(self, n): yield b"x" * 10
    def fake(url, **k):
        calls.append(url); return Short() if len(calls) == 1 else real(url, **k)
    inst.requests.get = fake
    try:
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d, "m.bin")
            inst.download(s.url, dest, len(data), hashlib.sha256(data).hexdigest(), retry_wait=0)
            assert dest.read_bytes() == data and len(calls) == 2
    finally:
        inst.requests.get = real


def test_finished_files_are_skipped():
    from speakrail.installer import fetch
    with tempfile.TemporaryDirectory() as d:
        f = Path(d, "a"); f.write_bytes(b"hello")
        files = [File("r", "v", "x/a", f, 5, None)]
        assert not missing(files)
        fetch(files)                        # nothing to do, so nothing tries the network


def _config():
    return discover_configs()["mac-e4b"]


def answers(*replies):
    it = iter(replies)
    return lambda prompt: next(it)


def test_the_wizard():
    c = Choices(Path("/m"), True)
    rc = Path("/u/.zshrc")
    assert wizard(MAC, _config(), Path("/h"), c, rc, 12.3, ask=answers("1")) is True
    assert wizard(MAC, _config(), Path("/h"), c, rc, 12.3, ask=answers("")) is True, "Enter takes the default"
    assert wizard(MAC, _config(), Path("/h"), c, rc, 12.3, ask=answers("3")) is False
    assert wizard(MAC, _config(), Path("/h"), c, rc, 12.3, ask=answers("what", "3")) is False
    assert wizard(MAC, _config(), Path("/h"), c, rc, 12.3, ask=answers("2", "/Volumes/ext/models", "n", "1")) is True
    assert c.models == Path("/Volumes/ext/models").resolve() and c.modify_path is False
    assert wizard(MAC, _config(), Path("/h"), c, rc, 12.3, ask=answers("2", "", "", "1")) is True
    assert c.modify_path is False, "Enter keeps the choice"
    assert wizard(MAC, _config(), Path("/h"), c, None, 12.3, ask=answers("2", "", "1")) is True, "no startup file to offer"


def test_path_is_one_marked_line():
    with tempfile.TemporaryDirectory() as d:
        rc = Path(d, ".zshrc"); rc.write_text("alias x=y")
        assert add_to_path(rc, Path("/my home/bin")) is True
        text = rc.read_text()
        assert text.startswith("alias x=y\nexport PATH=") and RC_MARK in text and "'/my home/bin'" in text
        assert add_to_path(rc, Path("/my home/bin")) is False and rc.read_text() == text, "twice changes nothing"
        new = Path(d, ".bash_profile")
        assert add_to_path(new, Path("/b")) is True and new.read_text().startswith("export PATH=/b:")
        r = subprocess.run(["sh", "-c", f'. "{rc}" 2>/dev/null; printf %s "$PATH"'], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
        assert r.stdout == "/my home/bin:/usr/bin:/bin", r.stdout


def test_progress_lines_keep_their_words_together():
    from speakrail.ui import paint
    line = paint("[3/8] voxtral-mini-4b.gguf, 3.10 GB", on=True)
    assert "[3/8] voxtral-mini-4b.gguf, 3.10 GB" in line and "\033[38;5;218m" in line
    assert "\033[38;5;114m" in paint("[llm/mlx] ready: 3.8 s", on=True), "an engine's state still has its color"


def test_which_startup_file():
    u = Path("/u")
    assert rc_file("/bin/zsh", u) == u / ".zshrc" and rc_file("/opt/x/bash", u).name in (".bash_profile", ".bashrc") and rc_file("/bin/fish", u) is None and rc_file("", u) is None


def test_the_command_runs_and_finds_its_home():
    with tempfile.TemporaryDirectory(prefix="my home ") as d:
        home = Path(d)
        cmd = write_command(home, home / "models")
        assert "MODELS_DIR" not in cmd.read_text() and os.access(cmd, os.X_OK)
        cmd = write_command(home, Path("/other/models"))
        assert "export MODELS_DIR=/other/models" in cmd.read_text()
        env = {k: v for k, v in os.environ.items() if k != "SPEAKRAIL_HOME"}
        env["PYTHONPATH"] = str(ROOT / "src")
        r = subprocess.run([str(cmd), "status"], capture_output=True, text=True, env=env)
        assert r.returncode == 3 and "Not running" in r.stdout, r.stdout + r.stderr


def test_nothing_fits_means_nothing_happens():
    """a config that isn't there stops before any download, and Cancel takes away a folder this run made"""
    with tempfile.TemporaryDirectory() as d:
        made = Path(d, "h"); made.mkdir()
        env = {**os.environ, "SPEAKRAIL_HOME": str(made), "PYTHONPATH": str(ROOT / "src")}
        r = subprocess.run([sys.executable, "-m", "speakrail", "install", "--yes", "--fresh", "--config", "nope"], capture_output=True, text=True, env=env)
        assert r.returncode == 2 and not made.exists(), (r.returncode, r.stdout, r.stderr)


def test_install_sh():
    sh = ROOT / "install.sh"
    assert subprocess.run(["sh", "-n", str(sh)]).returncode == 0
    r = subprocess.run(["sh", str(sh), "--nonsense"], capture_output=True, text=True)
    assert r.returncode != 0 and "Unknown option" in r.stderr
    r = subprocess.run(["sh", str(sh), "--models", "a b; rm -rf x"], capture_output=True, text=True)
    assert r.returncode != 0 and "without spaces" in r.stderr
    r = subprocess.run(["sh", str(sh), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--yes" in r.stdout and "curl" in r.stdout
    text = sh.read_text()
    for private in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR"):
        assert f'{private}="$home/' in text, f"{private} would write outside the install folder"
    assert text.rstrip().endswith('main "$@"'), "all in one function: a download cut short runs nothing"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    bad = 0
    for t in tests:
        try:
            t0 = time.monotonic(); t()
            print(f"PASS {t.__name__} ({time.monotonic() - t0:.0f} s)", flush=True)
        except Exception as e:
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {str(e)[:700]}", flush=True)
    print(f"{len(tests) - bad} of {len(tests)} passed")
    raise SystemExit(1 if bad else 0)
