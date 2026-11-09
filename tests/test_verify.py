#!/usr/bin/env python3
"""What `speakrail verify` makes of a conversation, on made-up event streams: nothing is started.

    python tests/test_verify.py
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src")]
from speakrail import verify
from speakrail.verify import judge, replies_of

TAPE = {"tape": "t", "seconds": 30.0,
        "speech": [[1.0, 3.0, "What is seventeen times twenty-three?"], [8.0, 10.0, "How many kilometers is twenty-six miles?"],
                   [14.0, 14.5, "Mm-hmm."], [18.0, 20.0, "Wait, add bread."]],
        "labels": [{"kind": "reply", "at": 3.0, "within": 4.0, "text": "q1", "contains_any": ["391"]},
                   {"kind": "hold", "from": 8.6, "to": 9.4},
                   {"kind": "reply", "at": 10.0, "within": 4.0, "text": "q2", "contains_any": ["forty-one"]},
                   {"kind": "backchannel", "at": 14.0},
                   {"kind": "interrupt", "at": 18.0},
                   {"kind": "reply", "at": 20.0, "within": 4.0, "text": "q3", "tools_any": ["list_add"]}]}


def rec(replies, stops=(), tools=(), errors=0):
    """replies: (utt, turn, start, seconds of audio, text)"""
    ev = []
    for utt, turn, start, secs, text in replies:
        ev.append({"type": "reply_start", "utt": utt, "turn": turn, "_t": start - 0.2})
        ev += [{"type": "reply_delta", "turn": turn, "text": w + " ", "_t": start} for w in text.split()]
    ev += [{"type": "stop_audio", "turn": turn, "reason": reason, "_t": t} for t, turn, reason in stops]
    ev += [{"type": "tool", "name": n, "_t": t} for t, n in tools]
    ev += [{"type": "error", "message": "boom", "_t": 1.0}] * errors
    audio = {utt: {"first": start, "last": start + 0.1, "samples": int(secs * 24000)} for utt, _, start, secs, _ in replies}
    return {"events": ev, "audio": audio}


def results(r):
    return {(c["check"], c["at"]): c["ok"] for c in judge(TAPE, r)}


def test_a_good_conversation_passes():
    r = rec([(1, 0, 3.7, 2.0, "Seventeen times twenty-three is 391."),
             (2, 1, 10.8, 6.0, "That's about forty one point eight kilometers."),   # still speaking at 14 s: the backchannel counts
             (3, 2, 17.0, 5.0, "Here is a long story about Rome"),                  # cut by the interruption
             (4, 3, 20.7, 1.0, "Added bread.")],
            stops=[(18.9, 2, "yield")], tools=[(20.5, "list_add")])
    got = results(r)
    assert all(v is not False for v in got.values()), got
    assert got[("backchannel", 14.0)] is True and got[("interrupt", 18.0)] is True
    assert got[("answer", 10.0)] is True, "forty one = forty-one"


def test_each_kind_of_failure_is_caught():
    r = rec([(1, 0, 3.7, 2.0, "It's 400."),                       # wrong answer
             (2, 1, 9.0, 6.0, "Sure, what?"),                      # cut in during the pause, and no answer after it
             (3, 2, 12.0, 6.0, "A long answer")],                  # stopped by the backchannel, and not by the interruption
            stops=[(15.0, 2, "yield")], tools=[], errors=1)
    got = results(r)
    assert got[("answer", 3.0)] is False and got[("hold", 8.6)] is False and got[("backchannel", 14.0)] is False
    assert got[("reply", 20.0)] is False and got[("tool", 20.0)] is False and got[("errors", None)] is False


def test_nothing_to_judge_is_not_a_failure():
    r = rec([(1, 0, 3.7, 1.0, "391"), (2, 1, 10.6, 1.0, "forty-one"), (4, 3, 20.6, 1.0, "ok")], tools=[(20.4, "list_add")])
    got = results(r)
    assert got[("backchannel", 14.0)] is None and got[("interrupt", 18.0)] is None, "it wasn't speaking: n/a"


def test_a_reply_plays_for_its_length_or_until_stopped():
    r = rec([(1, 0, 2.0, 3.0, "a"), (2, 1, 8.0, 10.0, "b")], stops=[(9.0, 1, "yield")])
    rs = {x["utt"]: x for x in replies_of(r)}
    assert rs[1]["end"] == 5.0, "audio arrives faster than it plays: start + length"
    assert rs[2]["end"] == 9.0, "a stop ends it"


def test_a_reference_belongs_to_the_model_files():
    import tempfile, types
    with tempfile.TemporaryDirectory() as d:
        ref = Path(d, "big.json")
        ref.write_text(json.dumps({"meta": {"models": ["google/gemma@1", "speakrail/gemma@2"]}, "items": []}))
        old, verify.REFERENCES = verify.REFERENCES, Path(d)
        try:
            cfg = lambda *pins: types.SimpleNamespace(downloads=[{"repo": r, "revision": v} for r, v in pins])
            assert verify.reference_for(cfg(("google/gemma", "1"), ("speakrail/gemma", "2"), ("audio/x", "9"))) == ref
            assert verify.reference_for(cfg(("google/gemma", "1"), ("speakrail/gemma", "3"))) is None, "another LoRA commit decides differently"
            port = cfg(("someone/gemma-12b-gguf", "7"))
            assert verify.reference_for(port) is None, "a port to other files isn't found by its files"
            port.fidelity = {"reference": "big", "agree": 0.9}
            assert verify.reference_for(port) == ref, "it names the model it ported"
        finally:
            verify.REFERENCES = old


def test_the_shipped_references_are_whole():
    for f in sorted((ROOT / "verify/fidelity").glob("*.json")):
        d = json.loads(f.read_text())
        assert d["meta"]["models"] and len(d["items"]) >= 50, f.name
        assert all({"model", "prompt", "allowed", "token", "top"} <= set(it) for it in d["items"]), f.name
        assert len({it["token"] for it in d["items"]}) >= 2, f"{f.name}: only one kind of decision"


def test_the_shipped_tapes_are_whole():
    tapes = sorted((ROOT / "verify/tapes").glob("*.json"))
    assert len(tapes) == 5, tapes
    import wave
    for t in tapes:
        d = json.loads(t.read_text())
        with wave.open(str(t.with_suffix(".wav"))) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), t.name
            assert abs(w.getnframes() / 16000 - d["seconds"]) < 0.1, t.name
        assert d["labels"] and all(lb["kind"] in ("reply", "hold", "backchannel", "interrupt") for lb in d["labels"]), t.name


def test_params_reach_the_stack_it_starts():
    """--param has to reach the `speakrail run` verify starts, not only the report: a 12 GB check on a bigger card depends on it"""
    import argparse, subprocess, tempfile
    seen = {}

    class Stop(Exception):
        pass

    def fake(cmd, **kw):
        seen["cmd"] = cmd
        raise Stop
    real, subprocess.Popen = subprocess.Popen, fake
    try:
        a = argparse.Namespace(config="c", force=False, param=["llm/vllm-docker:gpu_share=0.27", "asr/x:threads=2"])
        try:
            verify.start_stack(a, Path(tempfile.mkdtemp()), 1)
        except Stop:
            pass
    finally:
        subprocess.Popen = real
    cmd = seen["cmd"]
    assert cmd[cmd.index("run"):] == ["run", "--config", "c", "--param", "llm/vllm-docker:gpu_share=0.27", "--param", "asr/x:threads=2"], cmd


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    bad = 0
    for t in tests:
        try:
            t0 = time.monotonic(); t()
            print(f"PASS {t.__name__} ({time.monotonic() - t0:.0f} s)", flush=True)
        except Exception as e:
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {str(e)[:700]}", flush=True)
    print(f"{len(tests) - bad} of {len(tests)} passed")
    raise SystemExit(1 if bad else 0)
