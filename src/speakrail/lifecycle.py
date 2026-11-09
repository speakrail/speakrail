"""speakrail update | rollback | uninstall, and the new-version notice in `speakrail status`.

<home>/state.json says what is installed: the app's commit, the config, and every model file with the repository and commit it came
from. The installer writes it at the end of every install. It keeps two entries: what runs now ("current") and what ran before it
("previous"). `update` installs the newest commit of the app's branch, `rollback` goes back to "previous", and the files and images
that neither of the two uses are removed. Only what state.json recorded is ever removed, never other files in the models folder.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .config import CFG
from .ui import error, say

STATE = "state.json"
CHECK_EVERY_S = 24 * 3600
PATCH_MARK = ".speakrail-patched"


# ---------------------------------------------------------------------- state
def state_file(home: Path) -> Path:
    return Path(home) / STATE


def load_state(home: Path) -> dict:
    try:
        raw = json.loads(state_file(home).read_text())
    except (OSError, ValueError):
        raw = {}
    return {"current": raw.get("current"), "previous": raw.get("previous")}


def save_state(home: Path, state: dict) -> None:
    f = state_file(home)
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    os.replace(tmp, f)


def git(app: Path, *args, timeout: float = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(app), *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", "timed out")


def app_commit(app: Path) -> str | None:
    if not (Path(app) / ".git").exists():
        return None
    r = git(app, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def version_of(app: Path) -> str:
    """the version the code says it is, read without importing it (it may be another version than the one running)"""
    try:
        for line in (Path(app) / "src/speakrail/__init__.py").read_text().splitlines():
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return "?"


def snapshot(app: Path, branch: str, config: str, pinned: bool, models: Path, files, patches, images, force: bool = False) -> dict:
    """what an install that just finished put on this machine. `pinned`: the person picked this config rather than the recommended one,
    so `speakrail start` keeps it; `force`: it needs more memory than this machine has, and they wanted it anyway."""
    models = Path(models)
    return {"app": {"commit": app_commit(app), "branch": branch, "version": version_of(app)},
            "config": config, "pinned": pinned, "force": force, "models": str(models),
            "files": [{"repo": f.repo, "revision": f.revision, "path": f.path, "dest": str(Path(f.dest).relative_to(models)), "size": f.size, "sha256": f.sha256} for f in files],
            "patches": [p["weights"] for p in patches], "images": sorted(images), "installed_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def installed_pin(home: Path) -> tuple[str, bool] | None:
    """(config, force) when the config installed here was picked by name, else None: auto mode chooses again each start"""
    current = load_state(home)["current"]
    if current and current.get("pinned") and current.get("config"):
        return current["config"], bool(current.get("force"))
    return None


def recorded_files(entry: dict | None, models: Path) -> dict[str, dict]:
    """absolute path -> what was downloaded there, for an entry that used this models folder"""
    if not entry or Path(entry["models"]) != Path(models):
        return {}
    return {str(Path(models) / f["dest"]): f for f in entry["files"]}


def record(home: Path, new: dict) -> dict | None:
    """an install finished: `new` is what runs now. The entry before it becomes "previous" when the version or the config changed
    (installing the same thing again changes nothing). -> the entry that dropped out, whose leftovers can go."""
    state = load_state(home)
    current, previous = state["current"], state["previous"]
    dropped = None
    if current and (current["app"]["commit"] != new["app"]["commit"] or current["config"] != new["config"]):
        dropped, previous = previous, current
    save_state(home, {"current": new, "previous": previous})
    return dropped


# ---------------------------------------------------------------------- what is no longer used
def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def unused(dropped: dict | None, keep: list) -> tuple[list[Path], list[str]]:
    """the files and images only `dropped` used. A path state.json points outside its models folder is never listed."""
    if not dropped:
        return [], []
    kept = {str(Path(e["models"]) / f["dest"]) for e in keep if e for f in e["files"]}
    root = Path(dropped["models"])
    files = []
    for f in dropped["files"]:
        p = root / f["dest"]
        if str(p) not in kept and _inside(p, root):
            files.append(p)
    images = sorted(set(dropped["images"]) - {i for e in keep if e for i in e["images"]})
    return files, images


def remove_files(files: list[Path], root: Path) -> int:
    """-> bytes freed. A file's download leftovers and patch marker go with it, and folders left empty under `root`."""
    freed = 0
    for p in files:
        for q in (p, p.with_name(p.name + ".part"), p.with_name(p.name + ".part.done"), p.with_name(p.name + PATCH_MARK)):
            if q.is_file() and _inside(q, root):
                freed += q.stat().st_size
                q.unlink()
        parent = p.parent
        while _inside(parent, root) and Path(os.path.realpath(parent)) != Path(os.path.realpath(root)):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return freed


def remove_images(images: list[str]) -> list[str]:
    """-> the ones removed. One a container still uses, or that isn't there, is left alone."""
    if not images or not shutil.which("docker"):
        return []
    return [i for i in images if subprocess.run(["docker", "rmi", i], capture_output=True).returncode == 0]


def prune(dropped: dict | None, keep: list) -> None:
    files, images = unused(dropped, keep)
    if not files and not images:
        return
    freed = remove_files(files, Path(dropped["models"])) if files else 0
    gone = remove_images(images)
    parts = []
    if files:
        parts.append(f"{len(files)} model file{'s' if len(files) != 1 else ''} ({freed / 1e9:.1f} GB)")
    if gone:
        parts.append(f"the image{'s' if len(gone) != 1 else ''} {', '.join(gone)}")
    if parts:
        say(f"Removed what neither this version nor the one before it uses: {' and '.join(parts)}.")


def changes(old: dict | None, new: dict | None) -> list[str]:
    """what an update or rollback changed in the models, one line each"""
    if not old or not new:
        return []
    a = {f["dest"]: f for f in old["files"]}
    b = {f["dest"]: f for f in new["files"]}
    out = []
    for dest in sorted(set(a) | set(b)):
        x, y = a.get(dest), b.get(dest)
        if x and y and (x["repo"], x["revision"]) != (y["repo"], y["revision"]):
            out.append(f"  {dest}: {x['repo']}@{x['revision'][:8]} -> {y['repo']}@{y['revision'][:8]}")
        elif y and not x:
            out.append(f"  {dest}: new, {y['repo']}@{y['revision'][:8]}")
        elif x and not y:
            out.append(f"  {dest}: no longer used")
    return out


# ---------------------------------------------------------------------- update and rollback
def _ask(prompt: str, default: bool) -> bool:
    try:
        with open("/dev/tty") as tty:
            print(prompt, end="", flush=True)
            answer = tty.readline().strip().lower()
    except OSError:
        return False
    return default if not answer else answer.startswith("y")


def _uv_env(home: Path) -> dict:
    """the same private uv as install.sh: its cache and Python stay in the home folder"""
    return {**os.environ, "UV_CACHE_DIR": str(home / "cache" / "uv"), "UV_PYTHON_INSTALL_DIR": str(home / "python"), "UV_NO_MODIFY_PATH": "1", "UV_PYTHON_PREFERENCE": "only-managed"}


def install_version(home: Path, app: Path, commit: str, branch: str, config: str | None, chosen: bool, models: str | None, force: bool = False) -> int:
    """put the app at `commit` and run that version's own installer, which fetches what it needs and records it.
    -> the installer's exit code. Speakrail is stopped first: its engines may change under it."""
    from . import service
    if service.find():
        say("Stopping Speakrail...")
        service.stop_main(None)
    r = git(app, "checkout", "-q", commit)
    if r.returncode != 0:
        error(f"Couldn't switch the code to {commit[:8]}: {r.stderr.strip()[:300]}")
        return 1
    uv, python = home / "bin" / "uv", home / "venv" / "bin" / "python"
    if not uv.exists() or not python.exists():
        error(f"This installation has no {uv if not uv.exists() else python}; run install.sh again.")
        return 1
    say("Installing its Python libraries...")
    if subprocess.run([str(uv), "pip", "install", "-q", "--python", str(python), "-e", f"{app}[install]"], env=_uv_env(home)).returncode != 0:
        error("Installing the Python libraries failed; nothing else was changed. `speakrail rollback` goes back.")
        return 1
    cmd = [str(python), "-m", "speakrail", "install", "--yes", "--no-modify-path"]
    if config:
        cmd += ["--config", config] + (["--chosen"] if chosen else []) + (["--force"] if force else [])
    if models:
        cmd += ["--models", models]
    env = {**_uv_env(home), "SPEAKRAIL_HOME": str(home), "SPEAKRAIL_BRANCH": branch}
    return subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL).returncode


def _report(home: Path, before: dict | None, rc: int, what: str) -> int:
    after = load_state(home)["current"]
    if rc != 0:
        error(f"\nThe {what} didn't finish. Run `speakrail {what}` again to carry on, or `speakrail rollback` to go back.")
        return rc
    lines = changes(before, after)
    if lines:
        say("Models that changed:\n" + "\n".join(lines))
    return 0


def update_main(a, run=install_version) -> int:
    home = Path(CFG.home)
    app = home / "app"
    if not (app / ".git").exists():
        error("This installation came from a folder (install.sh --source), so it can't update itself. Run install.sh --source again with the new code.")
        return 1
    current = load_state(home)["current"] or {}
    branch = (current.get("app") or {}).get("branch") or "main"
    r = git(app, "fetch", "-q", "--depth", "1", "origin", branch)
    if r.returncode != 0:
        error(f"Couldn't reach the repository: {r.stderr.strip()[:300]}")
        return 1
    new, head = git(app, "rev-parse", "FETCH_HEAD").stdout.strip(), app_commit(app)
    unfinished = bool(current) and (current.get("app") or {}).get("commit") != head
    _remember_check(home, new, head)
    if new == head and not unfinished:
        say(f"Speakrail is up to date: {version_of(app)} ({head[:8]}).")
        return 0
    if a.check:
        say(f"A new version is out: {new[:8]} (this is {head[:8]}). Get it with: speakrail update" if new != head else
            "The last update didn't finish. Run speakrail update to carry on.")
        return 0
    say(f"Updating Speakrail from {head[:8]} to {new[:8]}." if new != head else f"Finishing the update to {new[:8]}.")
    if not a.yes and not _ask("It stops Speakrail if it's running. Go ahead? [Y/n] ", True):
        say("Cancelled.")
        return 0
    pinned = current.get("pinned")
    rc = run(home, app, new, branch, current.get("config") if pinned else None, False, current.get("models"), force=bool(pinned and current.get("force")))
    return _report(home, current or None, rc, "update")


def rollback_main(a, run=install_version) -> int:
    home = Path(CFG.home)
    app = home / "app"
    if not (app / ".git").exists():
        error("This installation came from a folder (install.sh --source), so there is no earlier version to go back to.")
        return 1
    state, head = load_state(home), app_commit(app)
    current, previous = state["current"], state["previous"]
    target = current if current and current["app"]["commit"] != head else previous   # an update that didn't finish goes back to what ran before it
    if not target or not target["app"]["commit"]:
        error("There is no earlier version to go back to.")
        return 1
    commit = target["app"]["commit"]
    say(f"Going back to {target['app']['version']} ({commit[:8]}, installed {target['installed_at']}) with {target['config']}.")
    if not a.yes and not _ask("It stops Speakrail if it's running. Go ahead? [Y/n] ", True):
        say("Cancelled.")
        return 0
    if git(app, "cat-file", "-e", commit + "^{commit}").returncode != 0:
        r = git(app, "fetch", "-q", "--depth", "1", "origin", commit)
        if r.returncode != 0:
            error(f"Couldn't fetch {commit[:8]}: {r.stderr.strip()[:300]}")
            return 1
    rc = run(home, app, commit, target["app"]["branch"] or "main", target["config"], not target.get("pinned"), target["models"], force=bool(target.get("force")))
    return _report(home, current, rc, "rollback")


# ---------------------------------------------------------------------- the notice in `speakrail status`
def _remember_check(home: Path, remote: str, local: str | None) -> None:
    f = Path(home) / "run" / "update-check.json"
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"checked_at": time.time(), "remote": remote, "local": local}))
    except OSError:
        pass


def update_notice(home: Path, now=time.time) -> str:
    """a line for `speakrail status` when the app's branch has a newer commit, or "". Asks the repository at most once a day,
    and stays quiet when it can't (offline, or installed from a folder). SPEAKRAIL_UPDATE_CHECK=0 turns it off."""
    if not CFG.update_check:
        return ""
    home = Path(home)
    app = home / "app"
    head = app_commit(app)
    if not head:
        return ""
    f = home / "run" / "update-check.json"
    try:
        cached = json.loads(f.read_text())
    except (OSError, ValueError):
        cached = {}
    if now() - cached.get("checked_at", 0) > CHECK_EVERY_S or cached.get("local") != head:
        branch = ((load_state(home)["current"] or {}).get("app") or {}).get("branch") or "main"
        r = git(app, "ls-remote", "origin", f"refs/heads/{branch}", timeout=5)
        if r.returncode != 0 or not r.stdout.strip():
            return ""
        cached = {"remote": r.stdout.split()[0], "local": head}
        _remember_check(home, cached["remote"], head)
    return "A new version of Speakrail is out. Get it with: speakrail update" if cached.get("remote") != head else ""


# ---------------------------------------------------------------------- uninstall
def looks_like_home(home: Path) -> bool:
    """only a folder the installer made: it has the command or state.json, and it isn't the user's home or a top-level folder"""
    home = Path(os.path.realpath(home))
    if home in (Path("/"), Path(os.path.realpath(Path.home()))) or len(home.parts) < 3:
        return False
    return (home / "bin" / "speakrail").exists() or state_file(home).exists()


def rc_files(user_home: Path) -> list[Path]:
    from .installer import RC_MARK
    return [rc for rc in (user_home / ".zshrc", user_home / ".bashrc", user_home / ".bash_profile") if rc.is_file() and RC_MARK in rc.read_text()]


def remove_path_line(rc: Path) -> None:
    from .installer import RC_MARK
    rc.write_text("".join(line for line in rc.read_text().splitlines(keepends=True) if RC_MARK not in line))


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) if path.is_dir() else 0


def _containers() -> list[str]:
    if not shutil.which("docker"):
        return []
    r = subprocess.run(["docker", "ps", "-a", "--filter", "name=speakrail-", "--format", "{{.Names}}"], capture_output=True, text=True)
    return [n for n in r.stdout.split() if n.startswith("speakrail-")]


def uninstall_main(a, user_home: Path | None = None, use_docker: bool = True) -> int:
    from . import containers, service
    home = Path(os.path.realpath(CFG.home))
    user_home = Path(user_home or Path.home())
    if not looks_like_home(home):
        error(f"{home} doesn't look like a Speakrail installation (it has no bin/speakrail or state.json), so nothing was removed.")
        return 1
    state = load_state(home)
    entries = [e for e in (state["current"], state["previous"]) if e]
    models = Path(entries[0]["models"]) if entries else Path(CFG.models_dir or home / "models")
    models_inside = _inside(models, home)
    model_files = sorted({Path(e["models"]) / f["dest"] for e in entries for f in e["files"] if _inside(Path(e["models"]) / f["dest"], Path(e["models"]))})
    images = sorted({i for e in entries for i in e["images"]}) if use_docker else []
    rcs = rc_files(user_home)

    plan = ["Stops Speakrail" + (" and removes its containers" if use_docker else "")]
    if a.keep_models and models_inside:
        plan.append(f"Removes {home}, except the models in {models}")
    else:
        plan.append(f"Removes {home} ({_size(home) / 1e9:.1f} GB{', with the models in it' if models_inside else ''})")
    if not models_inside:
        if a.keep_models:
            plan.append(f"Keeps the models in {models}")
        elif model_files:
            n = len(model_files)
            plan.append(f"Removes the {n} model file{'s' if n != 1 else ''} it downloaded into {models} ({sum(_size(p) for p in model_files) / 1e9:.1f} GB); other files there stay")
        else:
            plan.append(f"Leaves {models} alone: there's no record of what it downloaded there")
    if images:
        plan.append(f"{'Keeps' if a.keep_images else 'Removes'} the Docker images {', '.join(images)}")
    for rc in rcs:
        plan.append(f"Takes its line out of {rc}")
    if CFG.sessions_dir:
        plan.append(f"Leaves the saved sessions in {CFG.sessions_dir}")
    say("This uninstalls Speakrail:\n" + "\n".join(f"  - {p}" for p in plan))
    if not a.yes and not _ask("Uninstall? [y/N] ", False):
        say("Cancelled.")
        return 0

    if service.find():
        service.stop_main(None)
    for name in _containers() if use_docker else []:
        containers.remove(name)
    if images and not a.keep_images:
        remove_images(images)
    if not models_inside and not a.keep_models and model_files:
        remove_files(model_files, models)
    for rc in rcs:
        remove_path_line(rc)
    if a.keep_models and models_inside:
        for child in home.iterdir():
            if not _inside(models, child) and not _inside(child, models):
                shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    else:
        shutil.rmtree(home)
    say("Speakrail is uninstalled." + (" Open a new terminal: this one still has it on its PATH." if rcs else ""))
    return 0
