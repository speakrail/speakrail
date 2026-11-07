"""NVIDIA: the largest GPU that nvidia-smi reports. Also on WSL2, where the GPU comes through from the Windows driver."""
import shutil
import subprocess
from pathlib import Path


def parse_nvidia_smi(text):
    """lines of `name, memory.total, memory.free`, in MiB -> the GPU with the most memory"""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            gpus.append({"name": parts[0], "memory_gib": int(parts[1]) / 1024, "free_memory_gib": int(parts[2]) / 1024})
    return max(gpus, key=lambda g: g["memory_gib"]) if gpus else None


def is_wsl(version_text=None):
    if version_text is None:
        p = Path("/proc/version")
        version_text = p.read_text() if p.exists() else ""
    return "microsoft" in version_text.lower()


WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"     # where WSL2 puts it, not always on PATH (NVIDIA's WSL guide)


def detect():
    smi = shutil.which("nvidia-smi") or (WSL_NVIDIA_SMI if Path(WSL_NVIDIA_SMI).exists() else None)
    if not smi:
        return None
    out = subprocess.run([smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        return None
    gpu = parse_nvidia_smi(out.stdout)
    if gpu and is_wsl():
        gpu["wsl"] = True
    return gpu
