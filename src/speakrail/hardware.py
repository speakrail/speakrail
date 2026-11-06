"""What machine this is. Each folder in platforms/ knows how to recognise its own hardware."""
from __future__ import annotations

import importlib.util
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import CFG, REPO_ROOT


@dataclass(frozen=True)
class Hardware:
    platform: str
    name: str
    memory_gib: float
    free_memory_gib: float
    memory_kind: str
    reserve_gib: float
    disk_free_gib: float = 0.0
    wsl: bool = False

    def describe(self) -> str:
        wsl = ", under WSL2" if self.wsl else ""
        return f"{self.name}, {self.memory_gib:.0f} GB {self.memory_kind} memory{wsl}"


def platform_dirs(paths=None) -> list[Path]:
    roots = paths if paths is not None else [REPO_ROOT / "platforms", Path(CFG.home) / "platforms"]
    return [d for root in roots if Path(root).is_dir() for d in sorted(Path(root).iterdir()) if (d / "platform.toml").exists()]


def _load_detect(folder: Path):
    spec = importlib.util.spec_from_file_location(f"speakrail_platform_{folder.name}", folder / "detect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect(paths=None) -> Hardware | None:
    """the first platform that recognises this machine, or None"""
    for folder in platform_dirs(paths):
        info = tomllib.loads((folder / "platform.toml").read_text())
        found = _load_detect(folder).detect()
        if found:
            home = Path(CFG.home)
            probe = next((p for p in (home, *home.parents) if p.exists()), Path("/"))
            return Hardware(platform=info["name"], name=found["name"], memory_gib=found["memory_gib"], free_memory_gib=found["free_memory_gib"],
                            memory_kind=info["memory_kind"], reserve_gib=float(info["reserve_gib"]), disk_free_gib=shutil.disk_usage(probe).free / 2**30,
                            wsl=bool(found.get("wsl")))
    return None
