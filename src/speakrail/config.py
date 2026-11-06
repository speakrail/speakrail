"""The one place where Speakrail's settings are read.

Precedence: defaults < config file (SPEAKRAIL_CONFIG, JSON) < environment variables < command-line flags (harness/server.py).
Every other module reads CFG. API keys are the exception: they only come from the environment, never from a
file, and secret() reads them when they are needed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # engines
    asr_url: str = "ws://127.0.0.1:8765/v1/transcribe"
    llm_url: str = "http://127.0.0.1:8001/v1/completions"
    llm_lora: str = "speakrail"
    llm_base_model: str = "speakrail-base"
    tts_engine: str = "breeze"
    tts_url: str = ""                # "" = the engine's own default port
    tts_model: str = "pocket-tts"
    breeze_int8: str = "backbone,text_encoder,depth"
    # paths
    home: str = str(Path.home() / ".speakrail")       # engine environments, logs and runtime files
    models_dir: str = ""             # "" = <home>/models
    engines_path: str = ""           # more folders to find engines in, separated by the OS path separator
    tokenizer_dir: str = str(REPO_ROOT / "models" / "speakrail-gemma" / "tokenizer")
    phrase_cache_dir: str = str(REPO_ROOT / "run" / "phrase_cache")
    sessions_dir: str = ""           # "" = sessions are not saved
    # web UI
    ui_host: str = "127.0.0.1"
    ui_port: int = 8080
    # assistant
    persona: str = "nova"
    voice: str = ""
    home_city: str = "London"
    home_timezone: str = "Europe/London"
    # turn taking
    silence_ms: int = 1000
    turn_tau: float = 0.9
    turn_k: int = 3
    spec: bool = True
    peek_frames: int = 0             # 0 = delay/80 + 2, which is 8
    ant4: str = ""                   # "<p index>:<tau>[,...]", "" = off
    ant4_age_s: float = 0.5
    ant4_confirm_skip: bool = True
    # features
    notes: bool = True
    interject: bool = True
    tool_hold_ms: int = 600
    search: str = "off"
    searxng_url: str = "http://127.0.0.1:8888/search"

    @property
    def ant4_triggers(self) -> list[tuple[int, float]]:
        return [(int(a), float(b)) for a, b in (x.split(":") for x in self.ant4.split(","))] if self.ant4 else []


def _on_if_1(v: str) -> bool:
    return v == "1"


def _off_if_0(v: str) -> bool:
    return v != "0"


# field -> (environment variable, parser). Names and meanings are the ones the project has always used.
ENV = {
    "asr_url": ("ASR_URL", str), "llm_url": ("LLM_URL", str), "llm_lora": ("LLM_LORA", str), "llm_base_model": ("LLM_BASE_MODEL", str),
    "tts_engine": ("TTS_ENGINE", str), "tts_url": ("TTS_URL", str), "tts_model": ("TTS_MODEL", str), "breeze_int8": ("BREEZE_INT8", str),
    "home": ("SPEAKRAIL_HOME", str), "models_dir": ("MODELS_DIR", str), "engines_path": ("SPEAKRAIL_ENGINES_PATH", str),
    "tokenizer_dir": ("SPEAKRAIL_TOKENIZER_DIR", str), "phrase_cache_dir": ("PHRASE_CACHE_DIR", str), "sessions_dir": ("SESSIONS_DIR", str),
    "ui_host": ("UI_HOST", str), "ui_port": ("UI_PORT", int),
    "persona": ("PERSONA", str), "voice": ("VOICE", str), "home_city": ("HOME_CITY", str), "home_timezone": ("HOME_TIMEZONE", str),
    "silence_ms": ("SILENCE_MS", int), "turn_tau": ("TURN_TAU", float), "turn_k": ("TURN_K", int), "spec": ("SPEC", _on_if_1),
    "peek_frames": ("PEEK_FRAMES", int), "ant4": ("ANT4", str), "ant4_age_s": ("ANT4_AGE_S", float), "ant4_confirm_skip": ("ANT4_CONFIRM_SKIP", _on_if_1),
    "notes": ("NOTES", _on_if_1), "interject": ("INTERJECT", _off_if_0), "tool_hold_ms": ("TOOL_HOLD_MS", int), "search": ("SEARCH", str),
    "searxng_url": ("SEARXNG_URL", str),
}


def _from_file(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object")
    known = {f.name: f for f in fields(Config)}
    out = {}
    for key, value in raw.items():
        if key not in known:
            raise ValueError(f"{path}: unknown setting {key!r}")
        kind = known[key].type
        ok = {"str": isinstance(value, str), "bool": isinstance(value, bool),
              "int": isinstance(value, int) and not isinstance(value, bool),
              "float": isinstance(value, (int, float)) and not isinstance(value, bool)}[kind]
        if not ok:
            raise ValueError(f"{path}: {key} must be {kind}, got {value!r}")
        out[key] = float(value) if kind == "float" else value
    return out


def load(env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    values = {}
    if env.get("SPEAKRAIL_CONFIG"):
        values.update(_from_file(Path(env["SPEAKRAIL_CONFIG"])))
    for name, (var, parse) in ENV.items():
        if var in env:
            values[name] = parse(env[var])
    return replace(Config(), **values)


def secret(name: str) -> str:
    """an API key from the environment, read when it is needed"""
    return os.environ.get(name, "")


CFG = load()
