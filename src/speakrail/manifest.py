"""Engine manifests: engines/<role>/<name>/engine.toml, also found in <home>/engines and in $SPEAKRAIL_ENGINES_PATH.

    contract = 1
    role = "llm"
    name = "mlx"
    description = "..."
    capabilities = ["chat", ...]
    platforms = ["darwin"]            # sys.platform values; a platform not listed here cannot run the engine
    url = "http://127.0.0.1:${port_main}/v1/completions"
    startup_timeout_s = 180

    [ports]                           # `main` is required; each becomes ${port_<name>}
    main = 8000

    [python]                          # optional: an environment of its own at <home>/envs/<role>-<name>, ${python} runs in it
    version = "3.12"
    requirements = "requirements.txt"

    [params]                          # defaults; each becomes ${<name>} and can be overridden per run
    model = "${models}/some-model"

    [harness]                         # optional: how the harness talks to it
    worker = "pocket"

    needs = ["${model}/config.json"]  # optional, top level: files that must exist before starting, checked by `doctor` too

    [[file]]                          # optional: a template rendered to <run>/<name> before starting
    name = "server.json"
    template = "server.json.in"

    [[process]]                       # one or more, started in order; any one of them dying fails the engine
    name = "server"
    cmd = ["${python}", "-u", "${engine_dir}/server.py", "--port", "${port_main}"]
    cwd = "${engine_dir}"             # optional
    env = { KEY = "value" }           # optional
    health = "http://127.0.0.1:${port_main}/health"       # http://... (200) or tcp://host:port
"""
from __future__ import annotations

import os
import string
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .config import CFG, REPO_ROOT
from .contracts import CAPABILITIES, CONTRACT_VERSION, ROLES


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Process:
    name: str
    cmd: tuple[str, ...]
    cwd: str = ""
    env: dict = field(default_factory=dict)
    health: str = ""


@dataclass(frozen=True)
class Engine:
    role: str
    name: str
    dir: Path
    description: str
    capabilities: frozenset[str]
    platforms: tuple[str, ...]
    url: str
    ports: dict
    params: dict
    processes: tuple[Process, ...]
    files: tuple[dict, ...] = ()
    python: dict | None = None
    harness: dict = field(default_factory=dict)
    startup_timeout_s: float = 120.0
    needs: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return f"{self.role}/{self.name}"

    @property
    def slug(self) -> str:
        return f"{self.role}-{self.name}"

    def env_dir(self, home: Path) -> Path:
        return Path(home) / "envs" / self.slug

    def runs_on(self, platform: str = sys.platform) -> bool:
        return platform in self.platforms


def _need(raw, key, kind, where):
    if key not in raw:
        raise ManifestError(f"{where}: `{key}` is required")
    if not isinstance(raw[key], kind):
        raise ManifestError(f"{where}: `{key}` has the wrong type")
    return raw[key]


def load_manifest(path: Path) -> Engine:
    where = str(path)
    try:
        raw = tomllib.loads(Path(path).read_text())
    except tomllib.TOMLDecodeError as e:
        raise ManifestError(f"{where}: {e}") from e
    known = {"contract", "role", "name", "description", "capabilities", "platforms", "url", "startup_timeout_s", "needs", "ports", "python", "params", "harness", "file", "process"}
    if extra := set(raw) - known:
        raise ManifestError(f"{where}: unknown keys {sorted(extra)}")
    if (v := _need(raw, "contract", int, where)) != CONTRACT_VERSION:
        raise ManifestError(f"{where}: written for contract {v}, this Speakrail speaks {CONTRACT_VERSION}")
    role = _need(raw, "role", str, where)
    if role not in ROLES:
        raise ManifestError(f"{where}: role must be one of {ROLES}")
    caps = _need(raw, "capabilities", list, where)
    if extra := set(caps) - set(CAPABILITIES[role]):
        raise ManifestError(f"{where}: unknown {role} capabilities {sorted(extra)}; known: {sorted(CAPABILITIES[role])}")
    ports = _need(raw, "ports", dict, where)
    if "main" not in ports:
        raise ManifestError(f"{where}: [ports] needs `main`")
    procs = []
    for i, p in enumerate(_need(raw, "process", list, where)):
        w = f"{where}: process {i}"
        cmd = _need(p, "cmd", list, w)
        if not cmd:
            raise ManifestError(f"{w}: empty cmd")
        procs.append(Process(_need(p, "name", str, w), tuple(str(c) for c in cmd), p.get("cwd", ""), dict(p.get("env", {})), p.get("health", "")))
    if not procs:
        raise ManifestError(f"{where}: needs at least one [[process]]")
    if not any(p.health for p in procs):
        raise ManifestError(f"{where}: at least one process needs `health`")
    py = raw.get("python")
    if py is not None and "requirements" not in py:
        raise ManifestError(f"{where}: [python] needs `requirements`")
    return Engine(role=role, name=_need(raw, "name", str, where), dir=Path(path).resolve().parent, description=raw.get("description", ""),
                  capabilities=frozenset(caps), platforms=tuple(_need(raw, "platforms", list, where)), url=_need(raw, "url", str, where),
                  ports=dict(ports), params=dict(raw.get("params", {})), processes=tuple(procs), files=tuple(raw.get("file", [])), python=py,
                  harness=dict(raw.get("harness", {})), startup_timeout_s=float(raw.get("startup_timeout_s", 120)), needs=tuple(raw.get("needs", ())))


def search_path() -> list[Path]:
    extra = [Path(p) for p in CFG.engines_path.split(os.pathsep) if p]
    return [REPO_ROOT / "engines", Path(CFG.home) / "engines", *extra]


def discover(paths: list[Path] | None = None) -> dict[str, Engine]:
    """engine id -> Engine. Later paths override earlier ones, so a local engine can replace a shipped one."""
    found = {}
    for root in paths if paths is not None else search_path():
        if not Path(root).is_dir():
            continue
        for f in sorted(Path(root).glob("*/*/engine.toml")):
            e = load_manifest(f)
            if f"{f.parent.parent.name}/{f.parent.name}" != e.id:
                raise ManifestError(f"{f}: the folder says {f.parent.parent.name}/{f.parent.name}, the manifest says {e.id}")
            found[e.id] = e
    return found


def render(text: str, values: dict, where: str = "") -> str:
    try:
        return string.Template(text).substitute(values)
    except KeyError as e:
        raise ManifestError(f"{where}: unknown variable {e} in {text!r}") from None
    except ValueError as e:
        raise ManifestError(f"{where}: bad template {text!r}: {e}") from None


def values_for(engine: Engine, home: Path, models: Path, run_dir: Path, overrides: dict | None = None, python: str | None = None) -> dict:
    """everything a manifest can refer to. Params may refer to the built-in variables and to each other (in file order)."""
    if overrides and (unknown := set(overrides) - set(engine.params)):
        raise ManifestError(f"{engine.id}: unknown params {sorted(unknown)}; known: {sorted(engine.params)}")
    if engine.python:
        py = python or str(engine.env_dir(home) / "bin" / "python")
    else:
        py = python or sys.executable
    v = {"home": str(home), "models": str(models), "run": str(run_dir), "engine_dir": str(engine.dir), "python": py}
    for k, port in engine.ports.items():
        v[f"port_{k}"] = str(port)
    for k, default in engine.params.items():
        v[k] = render(str((overrides or {}).get(k, default)), v, f"{engine.id} param {k}")
    return v
