"""From a machine to a setup: which config, which engines and which features, each with a reason someone can read.

A config says what it wants. The resolver turns off what the chosen engines can't do, then what the benchmark says this
machine can't afford. A setting forced with --set wins over the benchmark, but never over what an engine can't do.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .config import Config, REPO_ROOT, TUNED
from .contracts import REQUIRED
from .hardware import Hardware, idle_minutes
from .manifest import Engine, render

# measured on an M4 Pro, where everything is on: 31 ms per frame, peeks of about 280 ms, about 44 tok/s
MAX_FRAME_MS_SPECULATION = 40.0
MAX_PEEK_MS_SPECULATION = 400.0
REALTIME_FRAME_MS = 72.0                 # a frame is 80 ms of audio: above this the recognizer falls behind under any load
NOTES_MIN_TOK_S = 75.0                   # what notes were budgeted for; 28 tok/s made them arrive late
HARNESS_FIELDS = {f.name: f.type for f in fields(Config)}
ENGINE_SET_FIELDS = {"asr_url", "llm_url", "tts_url", "tts_engine", "tts_model", "home", "models_dir", "engines_path"}


class ConfigError(ValueError):
    pass


class ResolveError(RuntimeError):
    def __init__(self, message: str, rejected=None):
        super().__init__(message)
        self.rejected = rejected or []


@dataclass(frozen=True)
class SetupConfig:
    name: str
    status: str
    description: str
    platform: str
    priority: int
    memory_peak_gib: float
    disk_gib: float
    engines: dict
    harness: dict
    params: dict
    phrase_cache: bool
    evidence: dict
    path: Path | None = None
    downloads: tuple = ()


@dataclass
class Decision:
    setting: str
    value: object
    why: str


@dataclass
class Resolved:
    config: SetupConfig
    hardware: Hardware
    engines: dict                         # role -> Engine
    harness: dict
    params: dict
    why_config: str
    decisions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    benchmark: dict | None = None


def load_config(path: Path) -> SetupConfig:
    where = str(path)
    try:
        raw = tomllib.loads(Path(path).read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{where}: {e}") from e
    known = {"name", "status", "description", "platform", "priority", "memory_peak_gib", "disk_gib", "engines", "harness", "params", "run", "evidence", "download"}
    if extra := set(raw) - known:
        raise ConfigError(f"{where}: unknown keys {sorted(extra)}")
    for key in ("name", "status", "platform", "memory_peak_gib", "engines"):
        if key not in raw:
            raise ConfigError(f"{where}: `{key}` is required")
    if raw["status"] not in ("tested", "experimental"):
        raise ConfigError(f"{where}: status must be tested or experimental")
    if raw["status"] == "tested" and not raw.get("evidence"):
        raise ConfigError(f"{where}: a tested config needs [evidence] saying what it was tested on")
    if missing := {"asr", "tts", "llm"} - set(raw["engines"]):
        raise ConfigError(f"{where}: [engines] is missing {sorted(missing)}")
    for d in raw.get("download", []):
        if missing := {"repo", "revision", "include", "to"} - set(d):
            raise ConfigError(f"{where}: a [[download]] is missing {sorted(missing)}")
        if extra := set(d) - {"repo", "revision", "from", "include", "to"}:
            raise ConfigError(f"{where}: a [[download]] has unknown keys {sorted(extra)}")
    harness = raw.get("harness", {})
    for key, value in harness.items():
        if key not in HARNESS_FIELDS:
            raise ConfigError(f"{where}: unknown harness setting {key!r}")
        if key in ENGINE_SET_FIELDS:
            raise ConfigError(f"{where}: {key} comes from the engines and the install, not from a config")
        _check_type(key, value, where)
    return SetupConfig(name=raw["name"], status=raw["status"], description=raw.get("description", ""), platform=raw["platform"], priority=int(raw.get("priority", 0)),
                       memory_peak_gib=float(raw["memory_peak_gib"]), disk_gib=float(raw.get("disk_gib", 0)), engines=dict(raw["engines"]), harness=dict(harness),
                       params={k: dict(v) for k, v in raw.get("params", {}).items()}, phrase_cache=bool(raw.get("run", {}).get("phrase_cache", True)),
                       evidence=dict(raw.get("evidence", {})), path=Path(path), downloads=tuple(raw.get("download", [])))


def _check_type(key, value, where):
    kind = HARNESS_FIELDS[key]
    ok = {"str": isinstance(value, str), "bool": isinstance(value, bool), "int": isinstance(value, int) and not isinstance(value, bool),
          "float": isinstance(value, (int, float)) and not isinstance(value, bool)}[kind]
    if not ok:
        raise ConfigError(f"{where}: {key} must be {kind}, got {value!r}")


def discover_configs(paths=None, home=None) -> dict[str, SetupConfig]:
    roots = paths if paths is not None else [REPO_ROOT / "configs", *([Path(home) / "configs"] if home else [])]
    out = {}
    for root in roots:
        if Path(root).is_dir():
            for f in sorted(Path(root).glob("*.toml")):
                c = load_config(f)
                out[c.name] = c
    return out


def parse_set(items) -> dict:
    """--set key=value, typed like the setting it sets"""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set wants key=value, got {item!r}")
        key, text = item.split("=", 1)
        if key in TUNED:
            raise ConfigError(f"{key} is tuned together with the turn head or the checkpoint and can't be changed here. It belongs to a config.")
        if key not in HARNESS_FIELDS:
            raise ConfigError(f"unknown setting {key!r}")
        kind = HARNESS_FIELDS[key]
        try:
            if kind == "bool":
                if text.lower() not in ("true", "false", "1", "0", "on", "off"):
                    raise ValueError
                out[key] = text.lower() in ("true", "1", "on")
            else:
                out[key] = {"str": str, "int": int, "float": float}[kind](text)
        except ValueError:
            raise ConfigError(f"{key} must be {kind}, got {text!r}") from None
    return out


def _unusable(c: SetupConfig, hw: Hardware, engines: dict) -> str | None:
    """why this config can't run here at all, or None"""
    if c.platform not in ("any", hw.platform):
        return f"needs platform {c.platform}, this is {hw.platform}"
    for role, eid in c.engines.items():
        e = engines.get(eid)
        if e is None:
            return f"its engine {eid} isn't available"
        if e.role != role:
            return f"{eid} is a {e.role} engine, not {role}"
        for cap in REQUIRED.get(role, ()):
            if cap not in e.capabilities:
                return f"{eid} doesn't provide {cap}, which the harness can't run without"
    return None


def _too_big(c: SetupConfig, hw: Hardware) -> str | None:
    need = c.memory_peak_gib + hw.reserve_gib
    if need > hw.memory_gib:
        return f"needs {need:.1f} GB ({c.memory_peak_gib:.1f} peak + {hw.reserve_gib:.1f} kept free), this machine has {hw.memory_gib:.0f}"
    return None


def choose(configs: dict, hw: Hardware, engines: dict, pinned: str | None = None, force: bool = False) -> tuple[SetupConfig, str, list]:
    rejected = []
    if pinned:
        if pinned not in configs:
            raise ResolveError(f"no config named {pinned!r}; known: {', '.join(sorted(configs))}")
        c = configs[pinned]
        if why := _unusable(c, hw, engines):
            raise ResolveError(f"{pinned} can't run here: {why}")
        if (why := _too_big(c, hw)) and not force:
            raise ResolveError(f"{pinned} doesn't fit: {why}. Use --force to try anyway")
        return c, f"asked for by name ({c.status})", rejected
    candidates = []
    for c in sorted(configs.values(), key=lambda x: (-x.priority, x.name)):
        why = _unusable(c, hw, engines) or _too_big(c, hw)
        if not why and c.status != "tested":
            why = "experimental: only runs when asked for by name, with --config"
        if why:
            rejected.append((c.name, why))
        else:
            candidates.append(c)
    if not candidates:
        lines = "\n".join(f"  {n}: {w}" for n, w in rejected) or "  (no configs found)"
        raise ResolveError(f"No tested config fits {hw.describe()}:\n{lines}", rejected)
    c = candidates[0]
    rejected += [(o.name, f"a lower priority than {c.name}") for o in candidates[1:]]
    return c, "the best tested config that fits this machine", rejected


def resolve(configs: dict, engines: dict, hw: Hardware, *, pinned=None, sets=None, params=None, benchmark=None, force=False, models="", home="") -> Resolved:
    c, why, rejected = choose(configs, hw, engines, pinned, force)
    role_engines = {role: engines[eid] for role, eid in c.engines.items()}
    h = dict(c.harness)
    r = Resolved(config=c, hardware=hw, engines=role_engines, harness=h, params={k: dict(v) for k, v in c.params.items()}, why_config=why, rejected=rejected, benchmark=benchmark)
    for eid, values in (params or {}).items():
        if eid not in c.engines.values():
            raise ConfigError(f"--param names {eid}, which this config doesn't use")
        r.params.setdefault(eid, {}).update(values)
    sets = dict(sets or {})
    for key in sets:
        if key in ENGINE_SET_FIELDS:
            raise ConfigError(f"{key} comes from the engines and the install; it can't be set here")
    defaults = Config()

    def value(key):
        return h.get(key, getattr(defaults, key))

    def off(key, off_value, why_off):
        """an engine can't do it: this beats the config and --set alike"""
        was_forced = key in sets and sets[key] != off_value
        if value(key) != off_value or was_forced:
            r.decisions.append(Decision(key, off_value, why_off))
            if was_forced:
                r.warnings.append(f"--set {key} ignored: {why_off}")
        h[key] = off_value
        sets.pop(key, None)

    asr, llm = role_engines["asr"].capabilities, role_engines["llm"].capabilities
    if "peek" not in asr:
        off("peek", False, f"{c.engines['asr']} can't peek")
        off("spec", False, "speculation needs peek")
    if "anticipation" not in asr:
        off("ant4", "", f"{c.engines['asr']} has no anticipation outputs")
    if "parallel" not in llm:
        off("spec", False, f"{c.engines['llm']} can't run requests in parallel")
    for cap in ("chat", "parallel", "per_request_adapter"):
        if cap not in llm:
            off("notes", False, f"notes need {cap}, which {c.engines['llm']} doesn't have")
    if "sampling" not in llm:
        off("reply_temp", 0.0, f"{c.engines['llm']} can only reply greedily")

    if benchmark:
        frame, peek, tok_s = benchmark.get("asr_frame_ms"), benchmark.get("asr_peek_ms"), benchmark.get("llm_decode_tok_s")
        if frame is not None and frame > REALTIME_FRAME_MS:
            r.warnings.append(f"the recognizer takes {frame:.0f} ms per 80 ms frame here, so it may fall behind")
        if value("spec") and "spec" not in sets:
            if frame is not None and frame > MAX_FRAME_MS_SPECULATION:
                r.decisions.append(Decision("spec", False, f"frames take {frame:.0f} ms here (limit {MAX_FRAME_MS_SPECULATION:.0f}): no room for speculation"))
                h["spec"] = False
            elif peek is not None and peek > MAX_PEEK_MS_SPECULATION:
                r.decisions.append(Decision("spec", False, f"peeks take {peek:.0f} ms here (limit {MAX_PEEK_MS_SPECULATION:.0f}), and each one stalls the recognizer"))
                h["spec"] = False
        if sets.get("notes", value("notes")) and tok_s is not None and tok_s < NOTES_MIN_TOK_S:
            r.warnings.append(f"notes are on, but the LLM makes {tok_s:.0f} tok/s here and they were budgeted for {NOTES_MIN_TOK_S:.0f}, so they'll often be late")
    for key, val in sets.items():
        r.decisions.append(Decision(key, val, "set on the command line"))
        h[key] = val

    # what the engines and the install decide
    tts = role_engines["tts"]
    h["tts_engine"] = tts.harness.get("worker", "pocket")
    h["tts_model"] = tts.harness.get("model", tts.name)
    subs = {"models": str(models), "home": str(home)}
    for key, val in list(h.items()):
        if isinstance(val, str) and "$" in val:
            h[key] = render(val, subs, f"{c.name} harness {key}")
    if c.platform != "any":
        h.setdefault("phrase_cache_dir", str(Path(home) / "phrase_cache"))
    if (voice := h.get("voice")) and "/" not in voice and voice.endswith(".wav"):
        h["voice"] = str(Path(__file__).resolve().parent / "harness" / "voices" / voice)
    return r


FEATURES = (("spec", "speculation"), ("peek", "peek"), ("ant4", "ANT4 early trigger"), ("notes", "listening notes"), ("interject", "interjections"),
            ("reply_temp", "reply temperature"), ("decision_bias", "decision bias"))


def explain(r: Resolved) -> str:
    c, hw = r.config, r.hardware
    out = [f"Machine   {hw.describe()} (platform {hw.platform}); {hw.free_memory_gib:.0f} GB free now, {hw.disk_free_gib:.0f} GB disk free",
           f"Config    {c.name} ({c.status}): {c.description}", f"          Why: {r.why_config}"]
    if c.evidence:
        ev = c.evidence
        out.append(f"          Tested on {ev.get('machine', '?')}, {ev.get('date', '?')}" + (f": last word to ear {ev['last_word_to_ear_ms']} ms" if ev.get("last_word_to_ear_ms") else ""))
    fit = c.memory_peak_gib + hw.reserve_gib
    out.append(f"Memory    {c.memory_peak_gib:.1f} GB peak + {hw.reserve_gib:.1f} GB kept free = {fit:.1f} of {hw.memory_gib:.0f} GB")
    out.append("Engines   " + ", ".join(f"{e.id}" for e in r.engines.values()))
    idle = idle_minutes(hw)
    out.append(f"Idle      {f'engines unload after {idle:g} min with no session, the next session loads them again' if idle else 'engines stay loaded'}")
    if r.benchmark:
        b = r.benchmark
        out.append(f"Benchmark ASR {b.get('asr_frame_ms', 0):.0f} ms per frame, peek {b.get('asr_peek_ms', 0):.0f} ms, LLM {b.get('llm_decode_tok_s', 0):.0f} tok/s, decision {b.get('llm_decision_ms', 0):.0f} ms, first audio {b.get('tts_ttfa_ms', 0):.0f} ms")
    else:
        out.append("Benchmark none yet: it runs at first launch")
    out.append("Features")
    decided = {d.setting: d.why for d in r.decisions}
    defaults = Config()
    for key, label in FEATURES:
        val = r.harness.get(key, getattr(defaults, key))
        shown = {True: "on", False: "off", "": "off"}.get(val, val) if not isinstance(val, (int, float)) or isinstance(val, bool) else val
        why = decided.get(key) or ("from the config" if key in c.harness else "default")
        out.append(f"  {label:20} {str(shown):8} {why}")
    if r.warnings:
        out.append("Warnings")
        out += [f"  {w}" for w in r.warnings]
    if r.rejected:
        out.append("Not chosen")
        out += [f"  {n}: {w}" for n, w in r.rejected]
    return "\n".join(out)


def to_file(r: Resolved, urls: dict, path: Path) -> None:
    """the one file the harness reads"""
    values = dict(r.harness)
    values.update(asr_url=urls["asr"], llm_url=urls["llm"], tts_url=urls["tts"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(values, indent=1) + "\n")
