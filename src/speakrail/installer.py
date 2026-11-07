"""`speakrail install`: what install.sh hands over to once it has Python. Asks what to do, then builds audio.cpp, fetches the models and adds the command.

Running it again carries on: finished steps are skipped and a partial download resumes. Everything goes in the home folder, except one
line in the shell's startup file, which the wizard asks about.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import CFG, REPO_ROOT
from .hardware import detect, platform_module
from .ui import say

LICENSES = """Licenses
  Speakrail's code is Apache 2.0. The models keep their own licenses; NOTICE has the details.
  Voxtral and Gemma: Apache 2.0. Pocket TTS: CC-BY-4.0, by Kyutai.
  Pocket's terms prohibit illegal, harmful or deceptive use, including cloning a voice without explicit and lawful consent."""
BREEZE_LICENSE = "  Breeze TTS 2: its weights and what they generate are for research and non-commercial use only."
RC_MARK = "# added by the Speakrail installer"
HUB = "https://huggingface.co"
CHUNK = 64 << 20
STREAMS = 8
RETRIES = 5


class InstallError(Exception):
    pass


# ---------------------------------------------------------------------- the wizard
def tty_ask(prompt: str) -> str:
    with open("/dev/tty") as tty:
        print(prompt, end="", flush=True)
        return tty.readline().strip()


@dataclass
class Choices:
    models: Path
    modify_path: bool


def licenses(config) -> str:
    return LICENSES + ("\n" + BREEZE_LICENSE if "tts/breeze-docker" in config.engines.values() else "")


def memory_line(hw, config) -> str:
    need = config.memory_peak_gib + hw.reserve_gib
    kind = "GPU memory" if hw.memory_kind == "dedicated" else "memory"
    fit = "" if need <= hw.memory_gib else f": more than the {hw.memory_gib:.0f} GB here, so it may not start"
    return f"needs {need:.1f} GB of {kind} ({config.memory_peak_gib:.1f} at its peak, {hw.reserve_gib:.1f} kept free){fit}"


def pick_config(hw, opts, ask=tty_ask):
    """-> the Option a person picks. Enter takes the recommended one, and any of them can be picked: one that needs more memory than
    this machine has is confirmed first."""
    default = next((i for i, o in enumerate(opts, 1) if o.recommended), None)
    lines = [f"Setups for this machine ({hw.describe()}):"]
    for i, o in enumerate(opts, 1):
        tags = (["recommended"] if o.recommended else []) + ([] if o.config.status == "tested" else ["experimental"])
        lines.append(f"\n  {i}) {o.config.name}{'  (' + ', '.join(tags) + ')' if tags else ''}\n     {o.config.description}\n     {memory_line(hw, o.config)}")
    say("\n".join(lines) + "\n")
    while True:
        answer = ask(f"Which one? [{default}]: " if default else "Which one? ")
        if not answer and default:
            o = opts[default - 1]
        elif answer.isdigit() and 1 <= int(answer) <= len(opts):
            o = opts[int(answer) - 1]
        else:
            o = next((x for x in opts if x.config.name == answer), None)
            if o is None:
                say(f"Pick a number from 1 to {len(opts)}, not {answer!r}.")
                continue
        if not o.fits and not ask(f"{o.config.name} needs {o.need_gib:.1f} GB and this machine has {hw.memory_gib:.0f}. Install it anyway? [y/N]: ").lower().startswith("y"):
            continue
        return o


def summary(hw, config, home: Path, c: Choices, rc: Path | None, size_gb: float, builds: bool = True, images: list[str] = ()) -> str:
    lines = [f"Machine   {hw.describe()}", f"Config    {config.name} ({config.status}): {config.description}", f"Memory    {memory_line(hw, config)}",
             f"Installs  to {home}", f"Models    to {c.models}: about {size_gb:.1f} GB to download"]
    if images:
        lines.append(f"Images    Docker pulls {', '.join(images)}, unless it has them")
    if builds:
        lines.append("Build     audio.cpp from source, about a minute")
    lines.append(f"Path      {'adds one line to ' + str(rc) if c.modify_path and rc else 'leaves your shell alone; you run ' + str(home / 'bin/speakrail') + ' by its full path'}")
    return "\n".join(lines)


def wizard(hw, config, home: Path, c: Choices, rc: Path | None, size_gb: float, builds: bool = True, ask=tty_ask, images: list[str] = ()) -> bool:
    """-> go ahead? Changes `c` if asked to."""
    while True:
        say("\n" + summary(hw, config, home, c, rc, size_gb, builds, images) + "\n\n" + licenses(config) + "\n\n  1) Proceed\n  2) Customize\n  3) Cancel\n")
        answer = ask("> ")
        if answer in ("", "1"):
            return True
        if answer == "3":
            return False
        if answer == "2":
            entered = ask(f"Models folder [{c.models}]: ")
            if entered:
                c.models = Path(entered).expanduser().resolve()
            if rc:
                entered = ask(f"Add speakrail to your PATH, in {rc}? [{'Y/n' if c.modify_path else 'y/N'}]: ").lower()
                if entered:
                    c.modify_path = entered.startswith("y")
        else:
            say(f"Pick 1, 2 or 3, not {answer!r}.")


# ---------------------------------------------------------------------- downloads
@dataclass
class File:
    repo: str
    revision: str
    path: str                 # in the repo
    dest: Path
    size: int
    sha256: str | None


def select(listing, entry: dict, models: Path) -> list[File]:
    """one [[download]] -> its files. `listing` is what the repo holds under entry["from"]: (path, size, sha256)."""
    base = entry.get("from", "").strip("/")
    out = []
    for path, size, sha in listing:
        rel = path[len(base) + 1:] if base else path
        if any(fnmatch.fnmatch(rel, pattern) for pattern in entry["include"]) and not rel.startswith("."):
            out.append(File(entry["repo"], entry["revision"], path, models / entry["to"] / rel, size, sha))
    if not out:
        where = f"{entry['repo']}/{base}" if base else entry["repo"]
        raise InstallError(f"Nothing to download at {where}, revision {entry['revision']}: expected {', '.join(entry['include'])}. The repo may not have these files yet.")
    return out


def repo_listing(repo: str, revision: str, folder: str) -> list[tuple]:
    """(path, size, sha256) of every file under a folder, from Hugging Face's API. A folder that isn't there lists nothing."""
    url = f"{HUB}/api/models/{repo}/tree/{revision}" + (f"/{folder}" if folder else "") + "?recursive=true"
    out = []
    while url:
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException as e:
            raise InstallError(f"Can't reach Hugging Face: {type(e).__name__}") from e
        if r.status_code == 404:
            return []
        if r.status_code != 200:
            raise InstallError(f"Can't read {repo} on Hugging Face: HTTP {r.status_code}")
        out += [(x["path"], x["size"], (x.get("lfs") or {}).get("oid")) for x in r.json() if x["type"] == "file"]
        url = r.links.get("next", {}).get("url")
    return out


def plan(downloads, models: Path) -> list[File]:
    files = []
    for entry in downloads:
        files += select(repo_listing(entry["repo"], entry["revision"], entry.get("from", "").strip("/")), entry, models)
    return files


def missing(files: list[File], recorded: dict | None = None) -> list[File]:
    """not there, or not the right size, or there from another repo or commit than this config wants (`recorded`: what state.json
    says each path holds), which the size alone can't tell"""
    def other(f):
        r = (recorded or {}).get(str(f.dest))
        return r is not None and (r["repo"], r["revision"], r["path"]) != (f.repo, f.revision, f.path)
    return [f for f in files if other(f) or not (f.dest.exists() and f.dest.stat().st_size == f.size)]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, sha256: str | None, retry_wait: float = 2.0) -> None:
    """several connections at once, each fetching one chunk. <dest>.part holds the data and <dest>.part.done the chunks that finished,
    so a run that stops loses only what was in flight, and the next one fetches the rest. The file gets its name once its checksum is right."""
    part, log = dest.with_name(dest.name + ".part"), dest.with_name(dest.name + ".part.done")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not part.exists():
        log.unlink(missing_ok=True)
    done = {int(x) for x in log.read_text().split()} if log.exists() else set()
    chunks = max(1, -(-size // CHUNK))
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    lock, t0, got = threading.Lock(), time.monotonic(), [0]
    show = sys.stdout.isatty() and size >= CHUNK

    def one(i: int) -> None:
        start, end, error = i * CHUNK, min(size, (i + 1) * CHUNK) - 1, None
        for attempt in range(RETRIES):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(15, 60)) as r:
                    if r.status_code != 206:
                        raise OSError(f"HTTP {r.status_code}")
                    pos = start
                    for block in r.iter_content(1 << 20):
                        os.pwrite(fd, block, pos)
                        pos += len(block)
                    if pos != end + 1:
                        raise OSError("the connection ended early")
                with lock:
                    with open(log, "a") as f:
                        f.write(f"{i}\n")
                    got[0] += end + 1 - start
                    if show:
                        have = len(done) * CHUNK + got[0]
                        print(f"\r  {min(1.0, have / size) * 100:3.0f}%  {have / 1e9:.2f} of {size / 1e9:.2f} GB  {got[0] / 1e6 / max(time.monotonic() - t0, 0.1):.0f} MB/s ", end="", flush=True)
                return
            except (requests.RequestException, OSError) as e:
                error = e
                time.sleep(retry_wait * 2 ** attempt)
        raise InstallError(f"Couldn't download {dest.name}: {error}")

    try:
        with ThreadPoolExecutor(STREAMS) as pool:
            list(pool.map(one, [i for i in range(chunks) if i not in done]))
    finally:
        os.close(fd)
        if show:
            print(flush=True)
    if sha256 and sha256_of(part) != sha256:
        part.unlink()
        log.unlink(missing_ok=True)
        raise InstallError(f"{dest.name} came down damaged (its checksum doesn't match). Run the installer again to fetch it again.")
    os.replace(part, dest)
    log.unlink(missing_ok=True)


def fetch(files: list[File], recorded: dict | None = None) -> None:
    todo = missing(files, recorded)
    for i, f in enumerate(todo, 1):
        say(f"[{i}/{len(todo)}] {f.dest.name}, {f.size / 1e9:.2f} GB" if f.size >= 1e7 else f"[{i}/{len(todo)}] {f.dest.name}")
        f.dest.with_name(f.dest.name + ".speakrail-patched").unlink(missing_ok=True)     # a fresh copy has no trained rows in it yet
        download(f"{HUB}/{f.repo}/resolve/{f.revision}/{f.path}", f.dest, f.size, f.sha256)


# ---------------------------------------------------------------------- trained rows
ROW_TARGETS = {"embed_tokens": ("model.language_model.embed_tokens.weight", "lm_head.weight"), "embed_tokens_per_layer": ("model.language_model.embed_tokens_per_layer.weight",)}


def _safetensors_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def write_rows(weights: Path, rows: Path) -> None:
    """the trained rows of the control tokens into a Google checkpoint we don't re-host, in place, so the result is the benchmarked model byte for byte.
    The file says which token ids it holds; a table the model doesn't have is an error, one it has twice (the tied head) gets both."""
    base_r, hr = _safetensors_header(rows)
    meta = hr.get("__metadata__") or {}
    first, _, last = meta.get("ids", "6-27").partition("-")
    base_w, hw = _safetensors_header(weights)
    with open(rows, "rb") as fr, open(weights, "r+b") as fw:
        for key, t in hr.items():
            if key == "__metadata__":
                continue
            if key not in ROW_TARGETS or t["dtype"] != "BF16" or t["shape"][0] != int(last) - int(first) + 1:
                raise InstallError(f"Unexpected token rows in {rows.name}: {key} {t}")
            fr.seek(base_r + t["data_offsets"][0])
            data = fr.read(t["data_offsets"][1] - t["data_offsets"][0])
            found = [hw[n] for n in ROW_TARGETS[key] if n in hw]
            if not found or any(m["dtype"] != "BF16" or m["shape"][1] != t["shape"][1] for m in found):
                raise InstallError(f"{weights.name} doesn't match the rows for {key}")
            for m in found:
                fw.seek(base_w + m["data_offsets"][0] + int(first) * m["shape"][1] * 2)
                fw.write(data)


def apply_patch(entry: dict, models: Path) -> None:
    """one [[patch]]. A marker records it's done, so it runs once; a wrong result is removed, and the next install fetches the checkpoint again."""
    weights, rows = models / entry["weights"], models / entry["rows"]
    marker = weights.with_name(weights.name + ".speakrail-patched")
    if marker.exists() and marker.read_text().strip() == entry["sha256"]:
        return
    say(f"Writing the trained token rows into {weights.parent.name}/{weights.name}...")
    write_rows(weights, rows)
    if sha256_of(weights) != entry["sha256"]:
        weights.unlink()
        raise InstallError(f"{weights.name} doesn't match the benchmarked model after patching, so it was removed. Run the installer again.")
    marker.write_text(entry["sha256"] + "\n")


# ---------------------------------------------------------------------- audio.cpp
def build_audiocpp(home: Path, platform: str) -> None:
    pin = tomllib.loads((REPO_ROOT / "engines/audiocpp.toml").read_text())
    build = pin["build"][sys.platform]
    arch = platform_module(platform).cpu_arch()
    stamp, out = f"{pin['source']['commit']} {arch}\n", home / "bin" / "audiocpp"
    if (out / ".built").exists() and (out / ".built").read_text() == stamp and (out / build["target"]).exists():
        say("audio.cpp is already built")
        return
    say(f"Building audio.cpp {pin['source']['commit'][:8]} for {arch}, about a minute...")
    src, tools = home / "cache" / "audio.cpp", Path(sys.executable).parent
    log_path = home / "logs" / "build-audiocpp.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    steps = [["git", "init", "-q"], ["git", "fetch", "-q", "--depth", "1", pin["source"]["repo"], pin["source"]["commit"]], ["git", "checkout", "-q", "FETCH_HEAD"],
             [str(tools / "cmake"), "-S", ".", "-B", "build", "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={tools / 'ninja'}", *build["args"], f"-D{build['arch_flag']}={arch}"],
             [str(tools / "cmake"), "--build", "build", "--target", build["target"]]]
    with open(log_path, "wb") as log:
        for step in steps:
            log.write(f"\n=== {shlex.join(step)}\n".encode())
            log.flush()
            if subprocess.run(step, cwd=src, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
                tail = log_path.read_text(errors="replace").splitlines()[-12:]
                raise InstallError("Building audio.cpp failed:\n" + "\n".join("  " + x for x in tail) + f"\nThe full log is {log_path}")
    binary = next(p for p in (src / "build").rglob(build["target"]) if p.is_file() and os.access(p, os.X_OK))
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy2(binary, out / build["target"])
    shutil.copytree(src / "model_specs", out / "model_specs")
    (out / ".built").write_text(stamp)
    shutil.rmtree(src)


# ---------------------------------------------------------------------- the command
def write_command(home: Path, models: Path) -> Path:
    path = home / "bin" / "speakrail"
    path.parent.mkdir(parents=True, exist_ok=True)
    env = f"export SPEAKRAIL_HOME={shlex.quote(str(home))}\n"
    if models != home / "models":
        env += f"export MODELS_DIR={shlex.quote(str(models))}\n"
    path.write_text(f"#!/bin/sh\n{env}exec {shlex.quote(sys.executable)} -m speakrail \"$@\"\n")
    path.chmod(0o755)
    return path


def rc_file(shell: str, user_home: Path) -> Path | None:
    name = Path(shell or "").name
    if name == "zsh":
        return user_home / ".zshrc"
    if name == "bash":
        return user_home / (".bash_profile" if sys.platform == "darwin" else ".bashrc")
    return None


def add_to_path(rc: Path, bin_dir: Path) -> bool:
    """-> whether it changed anything. It's one marked line, so taking Speakrail out again is deleting it."""
    current = rc.read_text() if rc.exists() else ""
    if RC_MARK in current:
        return False
    with open(rc, "a") as f:
        f.write(f"{'' if not current or current.endswith(chr(10)) else chr(10)}export PATH={shlex.quote(str(bin_dir))}:\"$PATH\"  {RC_MARK}\n")
    return True


# ---------------------------------------------------------------------- the whole thing
def main(a, ask=tty_ask) -> int:
    from . import lifecycle
    from .manifest import discover
    from .resolver import ResolveError, discover_configs, options
    from .run import setup
    from .supervisor import EngineError, images as engine_images, install as install_env
    home = Path(CFG.home)
    hw = detect()
    name, force, picked = a.config, getattr(a, "force", False), None
    try:
        if not a.config and not a.yes and hw is not None:
            opts = options(discover_configs(home=home), hw, discover())
            if opts:
                picked = pick_config(hw, opts, ask)
                name, force = picked.config.name, not picked.fits
        resolved, _, _ = setup(argparse.Namespace(config=name, set=None, param=None, force=force, use_installed=picked is None))
    except (ResolveError, ValueError) as e:
        say(str(e))
        return _cancel(home, a, 2)
    config = resolved.config
    builds = any("audiocpp_dir" in e.params for e in resolved.engines.values())
    images = sorted({i for e in resolved.engines.values() for i in engine_images(e, home, resolved.params.get(e.id))})
    choices = Choices(models=Path(a.models).expanduser().resolve() if a.models else Path(CFG.models_dir or home / "models"), modify_path=not a.no_modify_path)
    rc = rc_file(os.environ.get("SHELL", ""), Path.home())
    try:
        files = plan(config.downloads, choices.models)
        size_gb = sum(f.size for f in missing(files)) / 1e9
        if not a.yes and not wizard(hw, config, home, choices, rc, size_gb, builds, ask=ask, images=images):
            say("Cancelled.")
            return _cancel(home, a, 0)
        if a.yes:
            say("\n" + summary(hw, config, home, choices, rc, size_gb, builds, images) + "\n\n" + licenses(config) + "\n")
        files = plan(config.downloads, choices.models)     # the models folder may have changed
        state = lifecycle.load_state(home)
        recorded = lifecycle.recorded_files(state["current"], choices.models)
        need = sum(f.size for f in missing(files, recorded)) / 2**30
        choices.models.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(choices.models).free / 2**30
        if need > free:
            raise InstallError(f"The models need {need:.1f} GB and {choices.models} has {free:.1f} GB free.")
        for e in resolved.engines.values():
            if e.python:
                say(f"\nSetting up {e.id}, which has its own Python libraries...")
                install_env(e, home, uv=str(home / "bin" / "uv"))
            elif e.in_containers:
                say(f"\nGetting the container image for {e.id}...")
                install_env(e, home, params=resolved.params.get(e.id))
        if builds:
            build_audiocpp(home, hw.platform)
        if missing(files, recorded):
            say("Fetching the models. If this stops, run the installer again and it carries on.")
            fetch(files, recorded)
        for entry in config.patches:
            apply_patch(entry, choices.models)
        branch = os.environ.get("SPEAKRAIL_BRANCH") or ((state["current"] or {}).get("app") or {}).get("branch") or "main"
        if picked:                                   # picked from the list: kept for `speakrail start` unless it was the recommended one
            pinned, forced = not picked.recommended, not picked.fits
        elif a.config:
            pinned, forced = not getattr(a, "chosen", False), force
        else:                                        # --yes: the earlier pick, if there was one and it was used
            earlier = lifecycle.installed_pin(home)
            pinned, forced = (True, earlier[1]) if earlier and earlier[0] == config.name else (False, False)
        now = lifecycle.snapshot(REPO_ROOT, branch, config.name, pinned, choices.models, files, config.patches, images, force=forced)
        dropped = lifecycle.record(home, now)
        lifecycle.prune(dropped, [now, lifecycle.load_state(home)["previous"]])
    except (InstallError, EngineError, subprocess.CalledProcessError, OSError) as e:
        say(f"{e}\nRun the installer again to carry on from here.")
        return 1
    except KeyboardInterrupt:
        say("\nStopped. Run the installer again to carry on from here.")
        os._exit(130)                      # not waiting for the connections in flight
    command = write_command(home, choices.models)
    added = choices.modify_path and rc is not None and add_to_path(rc, command.parent)
    shutil.rmtree(home / "cache", ignore_errors=True)
    say(f"\nSpeakrail is installed.\n")
    on_path = choices.modify_path and rc is not None
    if on_path:
        say(f"Open a new terminal, or run this one first:\n  export PATH={shlex.quote(str(command.parent))}:\"$PATH\"\n")
    say(f"Then start it: {'speakrail' if on_path else command} start\nThe first start measures this machine, a minute or two. Then open the page it shows you.")
    return 0


def _cancel(home: Path, a, code: int) -> int:
    """nothing is left behind if this run is what made the folder"""
    if a.fresh:
        shutil.rmtree(home, ignore_errors=True)
    return code
