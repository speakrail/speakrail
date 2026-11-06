"""Apple Silicon: the chip's name and unified memory, from sysctl and vm_stat."""
import platform
import re
import subprocess
import sys


def _sysctl(name):
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip()


def parse_vm_stat(text):
    """free bytes: pages that macOS can hand out straight away"""
    size = int((re.search(r"page size of (\d+) bytes", text) or [0, 16384])[1])
    pages = dict((k.strip().lower(), int(v)) for k, v in re.findall(r"^Pages ([^:]+):\s+(\d+)\.", text, flags=re.M))
    return sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable")) * size


def detect():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return None
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    return {"name": _sysctl("machdep.cpu.brand_string") or "Apple Silicon", "memory_gib": int(_sysctl("hw.memsize")) / 2**30, "free_memory_gib": parse_vm_stat(vm) / 2**30}
