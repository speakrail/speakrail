#!/usr/bin/env python3
"""The one config: defaults, environment variables, the config file, and that nothing else reads the environment.

    python tests/test_config.py
"""
import json, os, re, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]
from speakrail.config import Config, ENV, load, REPO_ROOT


def test_defaults_match_the_old_code():
    c = load({})
    expected = dict(asr_url="ws://127.0.0.1:8765/v1/transcribe", llm_url="http://127.0.0.1:8001/v1/completions", llm_lora="speakrail",
                    llm_base_model="speakrail-base", tts_engine="breeze", tts_url="", tts_model="pocket-tts", breeze_int8="backbone,text_encoder,depth",
                    tokenizer_dir=str(ROOT / "models/speakrail-gemma/tokenizer"), phrase_cache_dir=str(ROOT / "run/phrase_cache"), sessions_dir="",
                    ui_host="127.0.0.1", ui_port=8080, persona="nova", voice="", home_city="London", home_timezone="Europe/London",
                    silence_ms=1000, turn_tau=0.9, turn_k=3, spec=True, peek_frames=0, ant4="", ant4_age_s=0.5, ant4_confirm_skip=True,
                    notes=True, interject=True, tool_hold_ms=600, search="off", searxng_url="http://127.0.0.1:8888/search")
    for k, v in expected.items():
        assert getattr(c, k) == v, (k, getattr(c, k), v)
    assert set(expected) == {f for f in c.__dataclass_fields__}, "a setting has no expected default here"
    assert c.ant4_triggers == []


def test_every_setting_has_an_env_var():
    assert set(ENV) == set(Config.__dataclass_fields__)


def test_environment_variables():
    c = load({"TURN_K": "1", "TURN_TAU": "0.8", "ANT4": "6:0.7,5:0.9", "UI_PORT": "9000", "SPEAKRAIL_TOKENIZER_DIR": "/x"})
    assert (c.turn_k, c.turn_tau, c.ant4_triggers, c.ui_port, c.tokenizer_dir) == (1, 0.8, [(6, 0.7), (5, 0.9)], 9000, "/x")


def test_boolean_meanings_are_unchanged():
    # NOTES, SPEC, ANT4_CONFIRM_SKIP are on only for "1"; INTERJECT is off only for "0"
    for var, field in (("NOTES", "notes"), ("SPEC", "spec"), ("ANT4_CONFIRM_SKIP", "ant4_confirm_skip")):
        assert getattr(load({var: "1"}), field) is True and getattr(load({var: "0"}), field) is False and getattr(load({var: "true"}), field) is False
    assert load({"INTERJECT": "0"}).interject is False and load({"INTERJECT": "1"}).interject is True and load({"INTERJECT": "x"}).interject is True


def test_empty_env_var_overrides_a_file_or_default():
    assert load({"ANT4": ""}).ant4_triggers == [] and load({"VOICE": ""}).voice == ""


def _file(obj):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, f); f.close()
    return f.name


def test_file_then_env_precedence():
    path = _file({"turn_k": 1, "notes": False, "ant4": "6:0.7", "turn_tau": 1})
    try:
        c = load({"SPEAKRAIL_CONFIG": path})
        assert (c.turn_k, c.notes, c.ant4_triggers, c.turn_tau) == (1, False, [(6, 0.7)], 1.0)
        assert load({"SPEAKRAIL_CONFIG": path, "TURN_K": "5"}).turn_k == 5
    finally:
        os.unlink(path)


def test_file_mistakes_are_errors():
    for bad in ({"no_such_setting": 1}, {"turn_k": "1"}, {"notes": 1}, {"ui_port": True}, {"voice": 3}):
        path = _file(bad)
        try:
            load({"SPEAKRAIL_CONFIG": path})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
        finally:
            os.unlink(path)


def test_only_config_reads_the_environment():
    allowed = {"config.py", "claude_tool.py"}        # claude_tool copies the whole environment for its child process, minus secrets
    found = []
    for p in (ROOT / "src").rglob("*.py"):
        if p.name not in allowed and re.search(r"os\.environ|os\.getenv|getenv\(", p.read_text()):
            found.append(str(p))
    assert not found, found
    claude = (ROOT / "src/speakrail/harness/claude_tool.py").read_text()
    assert len(re.findall(r"os\.environ", re.sub(r'""".*?"""', "", claude, flags=re.S))) == 1, "claude_tool reads the environment beyond its child's copy"


def test_repo_root():
    assert REPO_ROOT == ROOT


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} passed")
