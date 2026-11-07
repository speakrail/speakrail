#!/usr/bin/env python3
"""The resolver: fake machines in, expected setups out. No GPU and no models.

    python tests/test_resolver.py
"""
import importlib.util, os, sys, tempfile, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]
TMP = Path(tempfile.mkdtemp(prefix="speakrail-resolver-"))
os.environ["SPEAKRAIL_HOME"] = str(TMP)
from speakrail.hardware import Hardware
from speakrail.manifest import discover, load_manifest
from speakrail.resolver import ConfigError, ResolveError, discover_configs, explain, load_config, parse_set, resolve

ENGINES = discover([ROOT / "engines"])
CONFIGS = discover_configs([ROOT / "configs"])


def mac(mem=24, name="Apple M4 Pro"):
    return Hardware("apple", name, mem, mem * 0.7, "unified", 4.0, 200)


def nvidia(mem=24, name="NVIDIA GeForce RTX 4090"):
    return Hardware("cuda", name, mem, mem, "dedicated", 1.0, 200)


def run(hw, **kw):
    return resolve(CONFIGS, ENGINES, hw, models="/m", home="/h", **kw)


# name, machine, config expected (None = refused), text the refusal must contain
TABLE = [
    ("M4 Pro 24 GB", mac(24), "mac-e4b", None),
    ("M5 Max 128 GB", mac(128, "Apple M5 Max"), "mac-e4b", None),
    ("M4 Pro 18 GB", mac(18), "mac-e4b", None),
    ("M4 16 GB", mac(16, "Apple M4"), "mac-e4b", None),
    ("M2 8 GB", mac(8, "Apple M2"), None, "more than the 8 GB here"),
    ("RTX 4090, as nvidia-smi reports it", nvidia(23.99), "nvidia-12b-24gb", None),
    ("RTX 5090", nvidia(32, "NVIDIA GeForce RTX 5090"), "nvidia-12b-24gb", None),
    ("RTX 4080, as nvidia-smi reports it", nvidia(15.99, "NVIDIA GeForce RTX 4080"), "nvidia-e4b-16gb", None),
    ("RTX 3060 12 GB", nvidia(12, "NVIDIA GeForce RTX 3060"), None, "more than the 12 GB here"),
    ("unknown vendor", Hardware("rocm", "Radeon", 24, 24, "dedicated", 1.0, 200), None, "needs platform apple"),
]


def test_table():
    for name, hw, expected, text in TABLE:
        try:
            r = run(hw)
        except ResolveError as e:
            assert expected is None, (name, str(e))
            assert text in str(e), (name, str(e))
            continue
        assert r.config.name == expected, (name, r.config.name)
        assert expected is not None, (name, "should have been refused")


def test_nvidia_configs_are_what_was_measured():
    h = run(nvidia(23.99)).harness
    assert (h["turn_k"], h["spec"], h["ant4"], h["peek_frames"], h["notes"], h["tts_engine"]) == (1, True, "", 10, True, "breeze"), h
    r = run(nvidia(15.99))
    assert (r.harness["turn_k"], r.harness["ant4"], r.harness["peek_frames"], r.harness["tts_engine"], r.harness["voice"]) == (1, "", 10, "pocket", "jane"), r.harness
    assert r.params["asr/audiocpp-docker"]["gguf_file"].endswith("q8_0.gguf")
    assert r.params["llm/vllm-docker"]["cpu_offload_gb"] == "6", r.params["llm/vllm-docker"]
    big = run(nvidia(23.99))
    assert "cpu_offload_gb" not in big.params["llm/vllm-docker"] and big.engines["llm"].params["cpu_offload_gb"] == "0", "the 24 GB config keeps everything on the GPU"
    assert "image" not in big.params["llm/vllm-docker"] and "image" not in r.params["llm/vllm-docker"], "both run the engine's one vLLM image"


def test_mac_config_is_what_was_measured():
    h = run(mac()).harness
    assert (h["turn_k"], h["spec"], h["ant4"], h["peek_frames"], h["notes"], h["tool_hold_ms"]) == (1, True, "", 8, False, 300), h
    assert h["tts_engine"] == "pocket" and h["tts_model"] == "pocket-tts"
    assert h["tokenizer_dir"] == "/m/speakrail-e4b-adapter/tokenizer" and h["voice"] == "jane", "a built-in voice is a name, not a file of ours"


def test_experimental_only_by_name():
    r = run(mac())
    assert any(n == "stub" and "experimental" in w for n, w in r.rejected), r.rejected
    r = run(mac(), pinned="stub")
    assert r.config.name == "stub" and r.engines["llm"].id == "llm/stub"
    assert run(nvidia(), pinned="stub").config.name == "stub"          # platform "any"


def test_memory_is_said_as_used_and_left():
    from speakrail.resolver import memory_use
    c = CONFIGS["mac-e4b"]
    assert memory_use(c, mac(24)) == "needs up to 10.7 GB of your 24 GB of memory, leaving 13.3 GB free"
    assert "leaving 3.3 GB free, too little" in memory_use(c, mac(14)), "fits in memory, but not with macOS beside it"
    assert memory_use(c, mac(8)) == "needs up to 10.7 GB of memory, more than the 8 GB here"
    assert "GPU memory" in memory_use(CONFIGS["nvidia-12b-24gb"], nvidia(23.99))


def test_fidelity_table_is_checked():
    import tempfile
    base = 'name="x"\nstatus="experimental"\nplatform="cuda"\nmemory_peak_gib=1\n[engines]\nasr="a"\ntts="t"\nllm="l"\n'
    with tempfile.TemporaryDirectory() as d:
        f = Path(d, "x.toml")
        f.write_text(base + '[fidelity]\nreference="nvidia-12b-24gb"\nagree=0.9\n')
        assert load_config(f).fidelity == {"reference": "nvidia-12b-24gb", "agree": 0.9}
        for bad in ('[fidelity]\nreference=12\n', '[fidelity]\nagree=1.5\n', '[fidelity]\nthreshold=0.9\n'):
            f.write_text(base + bad)
            try:
                load_config(f); raise AssertionError("accepted " + bad)
            except ConfigError:
                pass


def test_highctx_only_by_name():
    assert run(nvidia(23.99)).config.name == "nvidia-12b-24gb", "auto mode keeps Breeze on 24 GB"
    r = run(nvidia(23.99), pinned="nvidia-12b-24gb-highctx")
    assert r.engines["tts"].id == "tts/pocket-docker" and r.harness["voice"] == "jane", r.harness
    assert r.params["llm/vllm-docker"]["max_model_len"] == "131072", r.params["llm/vllm-docker"]


def test_the_list_to_pick_from():
    from speakrail.resolver import options
    names = lambda hw: [(o.config.name, o.fits, o.recommended) for o in options(CONFIGS, hw, ENGINES)]
    assert names(nvidia(23.99)) == [("nvidia-12b-24gb", True, True), ("nvidia-e4b-16gb", True, False), ("nvidia-12b-24gb-highctx", True, False)], names(nvidia(23.99))
    assert names(nvidia(15.99)) == [("nvidia-e4b-16gb", True, True), ("nvidia-12b-24gb", False, False), ("nvidia-12b-24gb-highctx", False, False)], "the ones that don't fit are still offered, last"
    assert not any(r for _, _, r in names(nvidia(12))), "nothing to recommend on 12 GB, and still a list"
    assert [n for n, _, _ in names(mac())] == ["mac-e4b"], "the stub, for any platform, is for tests"


def test_pinned_config_that_does_not_fit():
    try:
        run(mac(8), pinned="mac-e4b")
    except ResolveError as e:
        assert "--force" in str(e)
    else:
        raise AssertionError("ran on 8 GB")
    assert run(mac(8), pinned="mac-e4b", force=True).config.name == "mac-e4b"
    try:
        run(nvidia(), pinned="mac-e4b", force=True)                    # --force is about memory, not about the wrong hardware
    except ResolveError as e:
        assert "platform" in str(e)
    else:
        raise AssertionError("ran an Apple config on NVIDIA")


def test_unknown_config():
    try:
        run(mac(), pinned="nope")
    except ResolveError as e:
        assert "mac-e4b" in str(e)
    else:
        raise AssertionError


GOOD_BENCH = {"asr_frame_ms": 31.0, "asr_peek_ms": 280.0, "llm_decode_tok_s": 44.0, "llm_decision_ms": 37.0, "tts_ttfa_ms": 34.0}


def test_benchmark_gates():
    assert run(mac(), benchmark=GOOD_BENCH).harness["spec"] is True
    for slow, text in (({"asr_frame_ms": 45.0}, "frames take 45"), ({"asr_peek_ms": 500.0}, "peeks take 500")):
        r = run(mac(16, "Apple M4"), benchmark={**GOOD_BENCH, **slow})
        assert r.harness["spec"] is False and any(text in d.why for d in r.decisions), r.decisions
        assert text in explain(r)
    r = run(mac(16), benchmark={**GOOD_BENCH, "asr_frame_ms": 75.0})
    assert any("fall behind" in w for w in r.warnings), r.warnings


def test_set_beats_the_benchmark_and_warns():
    r = run(mac(16), benchmark={**GOOD_BENCH, "asr_frame_ms": 45.0}, sets={"spec": True})
    assert r.harness["spec"] is True
    r = run(mac(), benchmark=GOOD_BENCH, sets={"notes": True})
    assert r.harness["notes"] is True and any("late" in w for w in r.warnings), r.warnings
    r = run(mac(), benchmark={**GOOD_BENCH, "llm_decode_tok_s": 90.0}, sets={"notes": True})
    assert not any("late" in w for w in r.warnings)


def engines_without(role, *caps):
    base = ENGINES[f"{role}/stub"]
    import dataclasses
    e = dataclasses.replace(base, capabilities=frozenset(base.capabilities - set(caps)))
    return {**ENGINES, e.id: e}


def stub_run(engines, **kw):
    return resolve(CONFIGS, engines, mac(), pinned="stub", models="/m", home="/h", **kw)


def test_capabilities_turn_features_off():
    r = stub_run(engines_without("asr", "peek"), sets={"spec": True})
    assert r.harness["peek"] is False and r.harness["spec"] is False
    assert any("--set spec ignored" in w for w in r.warnings), r.warnings                # an engine limit beats --set
    r = stub_run(engines_without("asr", "anticipation"), sets={"ant4": "6:0.7"})
    assert r.harness["ant4"] == ""
    assert any("--set ant4 ignored" in w for w in r.warnings)
    r = stub_run(engines_without("llm", "parallel"))
    assert r.harness["spec"] is False and r.harness["notes"] is False
    r = stub_run(engines_without("llm", "chat"), sets={"notes": True})
    assert r.harness["notes"] is False and "chat" in " ".join(d.why for d in r.decisions)


def test_sampling_and_bias_come_from_the_config():
    import dataclasses
    wants = dataclasses.replace(CONFIGS["stub"], harness={**CONFIGS["stub"].harness, "reply_temp": 0.7, "reply_top_k": 64, "decision_bias": "106:1.0"})
    configs = {**CONFIGS, "stub": wants}
    r = resolve(configs, ENGINES, mac(), pinned="stub", models="/m", home="/h")
    assert (r.harness["reply_temp"], r.harness["reply_top_k"], r.harness["decision_bias"]) == (0.7, 64, "106:1.0")
    r = resolve(configs, engines_without("llm", "sampling"), mac(), pinned="stub", models="/m", home="/h")      # the mlx server answers 400 to any other temperature
    assert r.harness["reply_temp"] == 0.0 and "greedily" in " ".join(d.why for d in r.decisions), r.decisions
    assert r.harness["decision_bias"] == "106:1.0"                          # every engine has logit_bias
    assert "sampling" not in ENGINES["llm/mlx"].capabilities


def test_asr_without_turn_frames_is_refused():
    try:
        stub_run(engines_without("asr", "turn_head"))
    except ResolveError as e:
        assert "turn_head" in str(e)
    else:
        raise AssertionError("the harness can't end a turn without turn frames")


def write(name, text):
    p = TMP / f"{name}.toml"
    p.write_text(textwrap.dedent(text))
    return p


BASE = """
    name = "x"
    status = "experimental"
    platform = "any"
    memory_peak_gib = 1.0
    [engines]
    asr = "asr/stub"
    tts = "tts/stub"
    llm = "llm/stub"
"""


def test_config_mistakes_are_errors():
    cases = {"tested without evidence": BASE.replace('"experimental"', '"tested"'), "bad status": BASE.replace('"experimental"', '"maybe"'),
             "unknown key": '    colour = "red"\n' + BASE, "unknown setting": BASE + "\n    [harness]\n    volume = 3\n", "wrong type": BASE + '\n    [harness]\n    notes = "yes"\n',
             "engine setting": BASE + '\n    [harness]\n    llm_url = "http://x"\n', "missing role": BASE.replace('llm = "llm/stub"', "")}
    for name, text in cases.items():
        try:
            load_config(write("bad", text))
        except ConfigError:
            continue
        raise AssertionError(f"accepted a config: {name}")


def test_priority_picks_between_tested_configs():
    good = BASE.replace('"experimental"', '"tested"') + '\n    [evidence]\n    machine = "here"\n'
    a = load_config(write("a", good.replace('name = "x"', 'name = "low"\n    priority = 1')))
    b = load_config(write("b", good.replace('name = "x"', 'name = "high"\n    priority = 5')))
    r = resolve({"low": a, "high": b}, ENGINES, mac(), models="/m", home="/h")
    assert r.config.name == "high" and any(n == "low" for n, _ in r.rejected)


def test_a_config_that_needs_more_memory_loses_to_one_that_fits():
    good = BASE.replace('"experimental"', '"tested"') + '\n    [evidence]\n    machine = "here"\n'
    big = load_config(write("big", good.replace('name = "x"', 'name = "big"\n    priority = 9').replace("memory_peak_gib = 1.0", "memory_peak_gib = 100.0")))
    small = load_config(write("small", good.replace('name = "x"', 'name = "small"\n    priority = 1')))
    assert resolve({"big": big, "small": small}, ENGINES, mac(), models="/m", home="/h").config.name == "small"


def test_set_parsing():
    for key in ("turn_k", "peek_frames", "tool_hold_ms", "decision_bias", "reply_temp", "reply_top_p", "reply_top_k"):
        try:
            parse_set([f"{key}=2"])
            raise AssertionError(f"{key} can be set")
        except ConfigError as e:
            assert "tuned" in str(e)
    assert parse_set(["notes=true", "turn_tau=0.8", "voice=x.wav"]) == {"notes": True, "turn_tau": 0.8, "voice": "x.wav"}
    for bad in (["notes"], ["nope=1"], ["turn_tau=abc"], ["notes=maybe"]):
        try:
            parse_set(bad)
        except ConfigError:
            continue
        raise AssertionError(bad)
    try:
        run(mac(), sets={"llm_url": "http://x"})
    except ConfigError:
        return
    raise AssertionError("set an engine URL")


def test_explain_says_why():
    text = explain(run(mac()))
    for part in ("mac-e4b", "best tested config", "Apple M4 Pro", "Tested on", "needs up to 10.7 GB of your 24 GB", "asr/audiocpp", "speculation", "from the config", "stub: experimental"):
        assert part in text, (part, text)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_platform_parsers():
    cuda = _module(ROOT / "platforms/cuda/detect.py", "t_cuda")
    g = cuda.parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288, 12000\nNVIDIA GeForce RTX 4090, 24564, 23000\n")
    assert g["name"].endswith("4090") and round(g["memory_gib"]) == 24 and round(g["free_memory_gib"]) == 22
    assert cuda.parse_nvidia_smi("") is None and cuda.parse_nvidia_smi("garbage\n") is None
    assert cuda.is_wsl("Linux version 5.15.167.4-microsoft-standard-WSL2") and not cuda.is_wsl("Linux version 6.8.0-generic")
    apple = _module(ROOT / "platforms/apple/detect.py", "t_apple")
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:     100.\nPages active:   999.\nPages inactive:  50.\nPages speculative: 10.\nPages purgeable: 40.\n"
    assert apple.parse_vm_stat(text) == 200 * 16384


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {str(e)[:500]}")
    print(f"{len(tests) - bad} of {len(tests)} passed")
    raise SystemExit(1 if bad else 0)
