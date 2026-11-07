#!/usr/bin/env python3
"""Make the synthetic test tapes that `speakrail verify` plays: the user's side of five short conversations, spoken by Pocket TTS
voices, with labels for what the assistant should do and when. They are synthetic on purpose: no real person's recording ships.

    speakrail engine run tts/pocket-docker        # or tts/pocket on a Mac; the voices below must be in its voices folder
    python verify/make_tapes.py http://127.0.0.1:8091/v1/audio/speech [TAPE ...]     # all of them, or only these

Each tape is verify/tapes/<name>.wav (16 kHz mono, a quiet room's noise between the words) and <name>.json (the labels).
Clean synthetic speech is easier to hear than a real microphone, so the tapes check that a setup works and answers sensibly,
not how good it is: the published numbers come from real benchmarks.

`speech` lists every utterance as [start, end, text]. Labels, all times in seconds on the tape:
    reply        the user's turn ends at `at`: a reply should start within `within`; `contains_any`: its text should hold one of these;
                 `tools_any`: one of these tools should be called for it
    hold         the user pauses mid-sentence from `from` to `to`: no reply may start then
    backchannel  "mm-hmm" at `at` while the assistant talks: it should keep talking
    interrupt    the user cuts in at `at` while the assistant talks: it should stop or turn its voice down
"""
import json
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

OUT = Path(__file__).resolve().parent / "tapes"
SR_TTS, SR = 24000, 16000
LEAD_S = 1.5                       # quiet room before the first word

# name, voice, turns. A turn: "say" (text, or [text, pause seconds, text, ...] for a pause mid-sentence), "after" (seconds of quiet
# after it), "kind" (turn / backchannel / interrupt), "reply" (what to expect of the answer; None = no expectation)
TAPES = [
    ("everyday", "alba", [
        {"say": "Hi! How are you doing today?", "after": 6.0, "reply": {"within": 4.0}},
        {"say": "What is seventeen times twenty-three?", "after": 8.0, "reply": {"within": 5.0, "contains_any": ["391", "three hundred ninety-one", "three hundred and ninety-one"]}},
        {"say": "Perfect, thanks!", "after": 5.0, "reply": None},
    ]),
    ("pauses", "george", [
        {"say": "Hey, quick question.", "after": 5.0, "reply": {"within": 4.0}},
        {"say": ["How many kilometers is", 0.8, "twenty-six miles?"], "after": 8.0, "reply": {"within": 5.0, "contains_any": ["41", "42", "forty-one", "forty-two"]}},
        {"say": ["And what is", 0.8, "a hundred degrees Fahrenheit in Celsius?"], "after": 8.0, "reply": {"within": 5.0, "contains_any": ["37", "38", "thirty-seven", "thirty-eight"]}},
    ]),
    ("backchannels", "eve", [        # words a recognizer writes down as they are: Pocket's "mm-hmm" came out as "Mum"
        {"say": "Can you give me five tips for sleeping better?", "after": 2.2, "reply": {"within": 2.0}},
        {"say": "Right.", "kind": "backchannel", "after": 2.0, "reply": None},
        {"say": "Okay.", "kind": "backchannel", "after": 10.0, "reply": None},
        {"say": "That's helpful, thanks.", "after": 5.0, "reply": None},
    ]),
    ("interruption", "michael", [
        {"say": "Tell me about the history of the Roman Empire.", "after": 3.0, "reply": {"within": 3.0}},
        {"say": "Sorry, wait. Actually, how many legs does a spider have?", "kind": "interrupt", "after": 7.0,
         "reply": {"within": 5.0, "contains_any": ["eight", "8"]}},
        {"say": "Got it, thanks.", "after": 5.0, "reply": None},
    ]),
    ("tools", "marius", [
        {"say": "Please add milk and eggs to my shopping list.", "after": 7.0, "reply": {"within": 5.0, "tools_any": ["list_add"]}},
        {"say": "Make a note that the team meeting moved to Friday.", "after": 7.0, "reply": {"within": 5.0, "tools_any": ["record_note"]}},
        {"say": "Oh, and add bread too.", "after": 7.0, "reply": {"within": 5.0, "tools_any": ["list_add"]}},
    ]),
]


def speak(url: str, voice: str, text: str) -> np.ndarray:
    """one utterance from the TTS engine, at 16 kHz, with its leading and trailing silence cut"""
    body = json.dumps({"model": "pocket-tts", "input": text, "response_format": "pcm", "stream_format": "audio", "voice": voice}).encode()
    with urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=120) as r:
        pcm = np.frombuffer(r.read(), "<i2").astype(np.float32)
    if pcm.size == 0 or np.abs(pcm).max() < 300:
        raise SystemExit(f"{voice}: no speech for {text!r}. Is the voice in the engine's voices folder?")
    loud = np.flatnonzero(np.abs(pcm) > 300)
    margin = int(0.03 * SR_TTS)
    pcm = pcm[max(0, loud[0] - margin):loud[-1] + margin]
    return resample_poly(pcm, SR, SR_TTS)


def build(url: str, name: str, voice: str, turns: list) -> None:
    rng = np.random.default_rng(len(name))
    parts, t, labels, speech = [np.zeros(int(LEAD_S * SR), np.float32)], LEAD_S, [], []

    def add(x):
        nonlocal t
        parts.append(x.astype(np.float32)); t += len(x) / SR

    for turn in turns:
        say = turn["say"] if isinstance(turn["say"], list) else [turn["say"]]
        kind, start = turn.get("kind", "turn"), t
        for piece in say:
            if isinstance(piece, (int, float)):
                labels.append({"kind": "hold", "from": round(t, 2), "to": round(t + piece, 2)})
                add(np.zeros(int(piece * SR)))
            else:
                add(speak(url, voice, piece))
        end = t
        speech.append([round(start, 2), round(end, 2), " ".join(p for p in say if isinstance(p, str))])
        if kind in ("backchannel", "interrupt"):
            labels.append({"kind": kind, "at": round(start, 2)})
        if turn["reply"] is not None:
            labels.append({"kind": "reply", "at": round(end, 2), "text": " ".join(p for p in say if isinstance(p, str)), **turn["reply"]})
        add(np.zeros(int(turn["after"] * SR)))
    audio = np.concatenate(parts)
    audio = audio + rng.normal(0, 3, audio.size)              # a quiet room, not digital zeros: a recognizer treats those differently
    OUT.mkdir(exist_ok=True)
    pcm = np.clip(audio, -32768, 32767).astype("<i2")
    with wave.open(str(OUT / f"{name}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    (OUT / f"{name}.json").write_text(json.dumps({"tape": name, "voice": voice, "seconds": round(t, 2), "speech": speech, "labels": labels}, indent=1) + "\n")
    print(f"{name}: {voice}, {t:.0f} s, {len(labels)} labels")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for name, voice, turns in TAPES:
        if len(sys.argv) == 2 or name in sys.argv[2:]:
            build(sys.argv[1], name, voice, turns)
