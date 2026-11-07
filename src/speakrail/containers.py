"""Engines that run in Docker. A process in a manifest with `image` runs as one container on the host network, so its ports and health
checks work the same as for a local process. GPU access goes through the NVIDIA Container Toolkit (`--gpus all`).

Every container has a fixed name. That is what makes stopping reliable: a killed `docker run` client leaves its container running and
holding GPU memory, so stopping goes by name, and starting first clears anything an earlier run left behind.
"""
from __future__ import annotations

import re
import shutil
import subprocess

STOP_GRACE_S = 8


def name(engine_slug: str, process: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", f"speakrail-{engine_slug}-{process}")


def run_command(container: str, image: str, args: list[str], entrypoint: str = "", gpu: bool = False, mounts: list[str] | None = None, env: dict | None = None, extra: list[str] | None = None) -> list[str]:
    """Health is the supervisor's job, so an image's own HEALTHCHECK stays off. The same image can serve different engines."""
    cmd = ["docker", "run", "--rm", "--name", container, "--network", "host", "--no-healthcheck"]
    if gpu:
        cmd += ["--gpus", "all"]
    for m in mounts or []:
        cmd += ["-v", m]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    if entrypoint:
        cmd += ["--entrypoint", entrypoint]
    return cmd + list(extra or []) + [image] + list(args)


def _docker(*args, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def remove(container: str) -> None:
    if shutil.which("docker"):
        _docker("rm", "-f", container)


def stop(container: str) -> None:
    """a clean stop, then gone whatever happened. Safe when it isn't running."""
    if not shutil.which("docker"):
        return
    try:
        _docker("stop", "-t", str(STOP_GRACE_S), container, timeout=STOP_GRACE_S + 20)
    except subprocess.TimeoutExpired:
        pass
    remove(container)


def image_present(image: str) -> bool:
    return _docker("image", "inspect", image).returncode == 0


def pull(image: str) -> None:
    subprocess.run(["docker", "pull", image], check=True)


def problem(gpu: bool = True) -> str:
    """why containers can't run here, with what to do about it, or ""."""
    if not shutil.which("docker"):
        return "Docker isn't installed. Install it: https://docs.docker.com/engine/install/"
    info = _docker("info", "--format", "{{json .Runtimes}}")
    if info.returncode != 0:
        return f"Docker isn't running, or this user can't reach it: {info.stderr.strip()[:160]}"
    if gpu and "nvidia" not in info.stdout and not _wsl():     # Docker Desktop on WSL2 passes GPUs with --gpus alone, without a named runtime
        return ("Docker doesn't have the NVIDIA runtime. Install the NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), "
                "then run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker")
    return ""


def _wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False
