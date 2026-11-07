#!/usr/bin/env python3
"""update, rollback, uninstall and the new-version notice, against a local git repository standing in for GitHub.
Nothing is downloaded or installed, and no container is touched: the installer and Docker are stand-ins.

    python tests/test_lifecycle.py
"""
import argparse, json, os, subprocess, sys, tempfile, time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]
os.environ.setdefault("SPEAKRAIL_HOME", tempfile.mkdtemp(prefix="speakrail-lifecycle-"))
from speakrail import lifecycle
from speakrail.config import CFG
from speakrail.installer import RC_MARK, File, missing

lifecycle._ask = lambda prompt, default: False      # a question nobody answers is a no; the tests never wait on a terminal


def sh(*cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, (cmd, r.stderr)
    return r.stdout.strip()


def origin_with_two_commits(d: Path) -> tuple[str, str, str]:
    """-> (url, first, second): a bare repo whose main has two commits, as GitHub would"""
    work, bare = d / "work", d / "origin.git"
    work.mkdir()
    sh("git", "init", "-q", "-b", "main", cwd=work)
    sh("git", "config", "user.email", "t@t", cwd=work); sh("git", "config", "user.name", "t", cwd=work)
    (work / "src/speakrail").mkdir(parents=True)
    (work / "src/speakrail/__init__.py").write_text('__version__ = "1.0"\n')
    sh("git", "add", "-A", cwd=work); sh("git", "commit", "-qm", "one", cwd=work)
    first = sh("git", "rev-parse", "HEAD", cwd=work)
    (work / "src/speakrail/__init__.py").write_text('__version__ = "2.0"\n')
    sh("git", "commit", "-qam", "two", cwd=work)
    second = sh("git", "rev-parse", "HEAD", cwd=work)
    sh("git", "clone", "-q", "--bare", str(work), str(bare))
    return f"file://{bare}", first, second


def installed_home(d: Path, url: str, commit: str) -> Path:
    """a home as install.sh leaves it: the app cloned at `commit`, the command, and state.json for that commit"""
    home = d / "home"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "speakrail").write_text("#!/bin/sh\n")
    sh("git", "clone", "-q", url, str(home / "app"))
    sh("git", "checkout", "-q", commit, cwd=home / "app")
    lifecycle.save_state(home, {"current": entry(home, "c", [("m/a.bin", "r1")]), "previous": None})
    return home


def entry(home: Path, config: str, files, models: Path | None = None, images=(), pinned=False) -> dict:
    models = models or home / "models"
    return {"app": {"commit": lifecycle.app_commit(home / "app"), "branch": "main", "version": lifecycle.version_of(home / "app")},
            "config": config, "pinned": pinned, "models": str(models),
            "files": [{"repo": "org/repo", "revision": rev, "path": dest, "dest": dest, "size": 1, "sha256": None} for dest, rev in files],
            "patches": [], "images": list(images), "installed_at": "now"}


def fake_installer(files):
    """stands in for `speakrail install` of the version just checked out: records what it would have installed"""
    calls = []

    def run(home, app, commit, branch, config, chosen, models, force=False):
        calls.append((commit, config, chosen))
        sh("git", "checkout", "-q", commit, cwd=app)
        lifecycle.record(home, entry(home, config or "c", files(commit)))
        return 0
    return run, calls


def use_home(home: Path, **kw):
    lifecycle.CFG = replace(CFG, home=str(home), **kw)


def args(**kw):
    return argparse.Namespace(**{"yes": True, "check": False, "keep_models": False, "keep_images": True, **kw})


# ---------------------------------------------------------------------- state
def test_installing_the_same_thing_again_keeps_the_history():
    with tempfile.TemporaryDirectory() as d:
        url, first, second = origin_with_two_commits(Path(d))
        home = installed_home(Path(d), url, first)
        before = lifecycle.load_state(home)
        assert lifecycle.record(home, entry(home, "c", [("m/a.bin", "r1")])) is None
        assert lifecycle.load_state(home)["previous"] is None and lifecycle.load_state(home)["current"]["app"] == before["current"]["app"]
        sh("git", "checkout", "-q", second, cwd=home / "app")
        dropped = lifecycle.record(home, entry(home, "c", [("m/a.bin", "r2")]))
        st = lifecycle.load_state(home)
        assert dropped is None and st["previous"]["app"]["commit"] == first and st["current"]["app"]["commit"] == second
        dropped = lifecycle.record(home, entry(home, "other", []))
        assert dropped["app"]["commit"] == first, "a third version pushes the first one out"


def test_only_recorded_files_that_nothing_uses_are_removed():
    with tempfile.TemporaryDirectory() as d:
        models = Path(d, "models")
        for rel in ("old/only.bin", "shared/both.bin", "old/only.bin.speakrail-patched", "old/only.bin.part", "mine/notes.txt"):
            (models / rel).parent.mkdir(parents=True, exist_ok=True); (models / rel).write_text("x")
        outside = Path(d, "precious.txt"); outside.write_text("keep")
        mk = lambda files, images=(): {"models": str(models), "images": list(images), "files": [{"dest": f} for f in files]}
        dropped = mk(["old/only.bin", "shared/both.bin", "../precious.txt"], ["img:old", "img:shared"])
        files, images = lifecycle.unused(dropped, [mk(["shared/both.bin"], ["img:shared"]), None])
        assert files == [models / "old/only.bin"], files
        assert images == ["img:old"]
        lifecycle.remove_files(files, models)
        assert not (models / "old").exists(), "the file, its patch marker and download leftovers, and the folder left empty"
        assert (models / "shared/both.bin").exists() and (models / "mine/notes.txt").exists() and outside.exists()


def test_a_file_from_another_commit_is_fetched_again():
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d, "m.bin"); dest.write_bytes(b"1234")
        f = File("org/repo", "new", "m.bin", dest, 4, None)
        assert missing([f]) == [], "the size alone says it's there"
        assert missing([f], {str(dest): {"repo": "org/repo", "revision": "old", "path": "m.bin"}}) == [f]
        assert missing([f], {str(dest): {"repo": "org/repo", "revision": "new", "path": "m.bin"}}) == []


def test_a_picked_config_is_kept_for_start():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        assert lifecycle.installed_pin(home) is None
        lifecycle.save_state(home, {"current": {"config": "nvidia-12b-24gb", "pinned": False}, "previous": None})
        assert lifecycle.installed_pin(home) is None, "the recommended one: auto mode chooses again each start"
        lifecycle.save_state(home, {"current": {"config": "nvidia-12b-24gb-highctx", "pinned": True, "force": True}, "previous": None})
        assert lifecycle.installed_pin(home) == ("nvidia-12b-24gb-highctx", True)


def test_changes_name_the_commits():
    old = {"files": [{"dest": "a", "repo": "o/r", "revision": "1" * 40}, {"dest": "gone", "repo": "o/r", "revision": "1" * 40}]}
    new = {"files": [{"dest": "a", "repo": "o/r", "revision": "2" * 40}, {"dest": "b", "repo": "o/s", "revision": "3" * 40}]}
    assert lifecycle.changes(old, new) == ["  a: o/r@11111111 -> o/r@22222222", "  b: new, o/s@33333333", "  gone: no longer used"]


# ---------------------------------------------------------------------- update and rollback
def test_update_then_rollback():
    with tempfile.TemporaryDirectory() as d:
        url, first, second = origin_with_two_commits(Path(d))
        home = installed_home(Path(d), url, first)
        use_home(home)
        run, calls = fake_installer(lambda c: [("m/a.bin", "r2" if c == second else "r1")])
        assert lifecycle.update_main(args(check=True), run=run) == 0 and not calls, "--check only looks"
        assert lifecycle.update_main(args(), run=run) == 0
        assert calls == [(second, None, False)], "auto mode chooses again, as it did at install"
        st = lifecycle.load_state(home)
        assert (st["current"]["app"]["commit"], st["previous"]["app"]["commit"]) == (second, first)
        assert lifecycle.update_main(args(), run=run) == 0 and len(calls) == 1, "up to date: nothing to do"
        assert lifecycle.rollback_main(args(), run=run) == 0
        assert calls[-1] == (first, "c", True), "back to the earlier version's config, still marked as auto mode's choice"
        st = lifecycle.load_state(home)
        assert (st["current"]["app"]["commit"], st["previous"]["app"]["commit"]) == (first, second), "and a rollback can be undone"


def test_an_update_that_did_not_finish():
    with tempfile.TemporaryDirectory() as d:
        url, first, second = origin_with_two_commits(Path(d))
        home = installed_home(Path(d), url, first)
        use_home(home)

        def breaks(home_, app, commit, *_, **__):
            sh("git", "checkout", "-q", commit, cwd=app); return 1
        assert lifecycle.update_main(args(), run=breaks) == 1
        assert lifecycle.app_commit(home / "app") == second and lifecycle.load_state(home)["current"]["app"]["commit"] == first
        run, calls = fake_installer(lambda c: [])
        assert lifecycle.update_main(args(), run=run) == 0 and calls[0][0] == second, "running it again carries on"
        sh("git", "checkout", "-q", second, cwd=home / "app")
        lifecycle.save_state(home, {"current": lifecycle.load_state(home)["previous"], "previous": None})    # as if it broke again
        run, calls = fake_installer(lambda c: [])
        assert lifecycle.rollback_main(args(), run=run) == 0 and calls[0][0] == first, "rollback goes back to what ran before it"


def test_a_folder_install_cannot_update():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d, "home"); (home / "app").mkdir(parents=True)
        use_home(home)
        assert lifecycle.update_main(args(), run=None) == 1 and lifecycle.rollback_main(args(), run=None) == 1
        assert lifecycle.update_notice(home) == ""


# ---------------------------------------------------------------------- the notice
def test_the_notice_asks_once_a_day_and_can_be_turned_off():
    with tempfile.TemporaryDirectory() as d:
        url, first, second = origin_with_two_commits(Path(d))
        home = installed_home(Path(d), url, first)
        use_home(home)
        assert "speakrail update" in lifecycle.update_notice(home)
        sh("git", "remote", "set-url", "origin", "file:///nowhere", cwd=home / "app")
        assert "speakrail update" in lifecycle.update_notice(home), "within a day it answers from what it saw"
        assert lifecycle.update_notice(home, now=lambda: time.time() + 2 * lifecycle.CHECK_EVERY_S) == "", "offline: quiet"
        sh("git", "remote", "set-url", "origin", url, cwd=home / "app")
        sh("git", "checkout", "-q", second, cwd=home / "app")
        assert lifecycle.update_notice(home) == "", "up to date"
        use_home(home, update_check=False)
        sh("git", "checkout", "-q", first, cwd=home / "app")
        assert lifecycle.update_notice(home) == ""


# ---------------------------------------------------------------------- uninstall
def test_uninstall_takes_away_only_what_it_put_there():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        home, models, user = d / "home", d / "models", d / "user"
        (home / "bin").mkdir(parents=True); (home / "bin" / "speakrail").write_text("")
        (home / "venv").mkdir(); (home / "venv" / "x").write_text("")
        for rel in ("voxtral/v.gguf", "voxtral/v.gguf.speakrail-patched"):
            (models / rel).parent.mkdir(parents=True, exist_ok=True); (models / rel).write_text("x")
        (models / "mine.bin").write_text("someone else's")
        lifecycle.save_state(home, {"current": {"models": str(models), "images": ["img:a"], "files": [{"dest": "voxtral/v.gguf"}, {"dest": "../outside"}]}, "previous": None})
        user.mkdir(); rc = user / ".bashrc"
        rc.write_text(f"alias ll=ls\nexport PATH=/x:\"$PATH\"  {RC_MARK}\nexport EDITOR=vi\n")
        use_home(home)
        assert lifecycle.uninstall_main(args(yes=False), user_home=user, use_docker=False) == 0 and home.exists(), "no terminal to answer: nothing happens"
        assert lifecycle.uninstall_main(args(), user_home=user, use_docker=False) == 0
        assert not home.exists() and not (models / "voxtral").exists()
        assert (models / "mine.bin").exists(), "a file it didn't download stays"
        assert rc.read_text() == "alias ll=ls\nexport EDITOR=vi\n"


def test_uninstall_keeps_the_models_when_asked():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d, "home"); (home / "bin").mkdir(parents=True); (home / "bin" / "speakrail").write_text("")
        (home / "models" / "m").mkdir(parents=True); (home / "models" / "m" / "w.bin").write_text("x")
        (home / "logs").mkdir(); (home / "logs" / "a.log").write_text("x")
        use_home(home, models_dir="")
        assert lifecycle.uninstall_main(args(keep_models=True), user_home=Path(d), use_docker=False) == 0
        assert (home / "models" / "m" / "w.bin").exists() and not (home / "logs").exists() and not (home / "bin").exists()


def test_uninstall_refuses_a_folder_it_did_not_make():
    with tempfile.TemporaryDirectory() as d:
        stranger = Path(d, "documents"); stranger.mkdir(); (stranger / "thesis.txt").write_text("x")
        use_home(stranger)
        assert lifecycle.uninstall_main(args(), user_home=Path(d), use_docker=False) == 1 and (stranger / "thesis.txt").exists()
        assert not lifecycle.looks_like_home(Path.home()) and not lifecycle.looks_like_home(Path("/"))


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
