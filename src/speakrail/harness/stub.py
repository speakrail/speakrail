"""StubSession: the browser protocol without any models, for trying the UI (server.py --stub).

It speaks the same websocket protocol as session.MicroSession: int16 16 kHz mic audio in, JSON events and 24 kHz PCM
frames out. Instead of the ASR, the turn head, the LLM and the TTS it uses the mic level: P(speaking) follows the
loudness, a turn ends after STUB_END_MS of quiet, and the reply is a canned sentence voiced by a soft synthetic hum.
Replies rotate through a plain answer, a web search card and a tool card, so every part of the UI shows up.
"""
from __future__ import annotations

import asyncio
import math
import struct
import time

import numpy as np

FRAME = 1280                  # 80 ms of 16 kHz mic audio
TTS_RATE = 24000
CHUNK = 1920                  # 80 ms of 24 kHz reply audio
SPEECH_DBFS = -42.0           # louder than this counts as speech
STUB_END_MS = 640             # quiet after speech that ends the turn
MIN_SPEECH_MS = 320           # shorter bursts are ignored (clicks, coughs)

REPLIES = [
    {"text": "This is a stub reply. The models aren't running, so I can't understand you yet, but the interface "
             "works: talk again and I'll answer with a search card."},
    {"text": "One sec, looking that up. In the real system the answer would come from the search results shown here.",
     "search": {"query": "current weather in Berlin", "ms": 412, "results": [
         {"title": "Berlin weather forecast", "snippet": "Partly cloudy, around eighteen degrees, light breeze.",
          "url": "https://www.example.com/weather/berlin"},
         {"title": "Berlin, Germany: hourly forecast", "snippet": "Dry through the evening, cooler overnight.",
          "url": "https://www.example.org/forecast/berlin"}]}},
    {"text": "Sure, I've started a stopwatch for you. That's a tool card, the last of the stub replies.",
     "tool": {"name": "stopwatch", "args": {"action": "start"}, "result": {"elapsed_s": 0, "laps": []}, "ms": 3}},
]


class StubSession:
    def __init__(self, cfg, send, mcfg=None):
        self.cfg, self.mcfg, self._send = cfg, mcfg, send
        self.log_dir = None
        self.counters = {"barge_ins": 0}
        self.t0 = time.monotonic()
        self.buf = np.zeros(0, dtype=np.int16)
        self.frame = 0                      # 80 ms mic frames seen
        self.speech_frames = 0              # in the current user segment
        self.quiet_frames = 0
        self.turn = 0
        self.utt = 0
        self.n_reply = 0
        self.reply_task: asyncio.Task | None = None
        self.replying = False
        self.asr_reconnects = 0

    # ------------------------------------------------------------------ the interface server.py uses
    async def start(self):
        c = self.cfg
        self._send({"type": "ready", "stub": True,
                    "cfg": {"delay_ms": c.delay_ms, "barge_in": c.barge_in, "silence_ms": STUB_END_MS}})

    def feed_audio(self, data: bytes):
        self.buf = np.concatenate([self.buf, np.frombuffer(data, dtype=np.int16)])
        while len(self.buf) >= FRAME:
            fr, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            self._on_frame(fr)

    def note_play_start(self, utt: int, delay_ms: float):
        pass

    def _cancel_reply(self, reason: str):
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        if self.replying:
            self.replying = False
            self._send({"type": "stop_audio", "reason": reason, "turn": self.turn})

    async def reconnect_asr(self):
        self.asr_reconnects += 1
        self._send({"type": "asr_reconnect", "n": self.asr_reconnects, "at_ms": self.frame * 80})

    async def close(self):
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()

    def summary(self) -> dict:
        return {"turns": self.turn, "word_end_to_ear": {"p50": None}}

    # ------------------------------------------------------------------ fake ASR + turn head
    def _rel(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def _on_frame(self, fr: np.ndarray):
        self.frame += 1
        rms = float(np.sqrt(np.mean((fr.astype(np.float32) / 32768.0) ** 2))) + 1e-9
        dbfs = 20 * math.log10(rms)
        p_speak = min(1.0, max(0.0, (dbfs - SPEECH_DBFS + 6) / 12))
        speaking = dbfs > SPEECH_DBFS
        if speaking:
            if self.speech_frames == 0 and self.quiet_frames > 0:
                self.quiet_frames = 0
            self.speech_frames += 1
            if self.speech_frames * 80 == MIN_SPEECH_MS:
                if self.replying:
                    self.counters["barge_ins"] += 1
                    self._cancel_reply("user spoke over the reply (stub)")
                self._send({"type": "word", "raw": "(stub: no speech recognition) ", "text": "(stub)",
                            "t": self._rel(), "start": self.frame * 80, "end": self.frame * 80, "lag_ms": None})
        elif self.speech_frames:
            self.quiet_frames += 1
        # complete rises with the quiet after real speech, the way the head's does
        enough = self.speech_frames * 80 >= MIN_SPEECH_MS
        p_complete = min(0.97, self.quiet_frames * 80 / STUB_END_MS) if enough and not speaking else 0.0
        rest = max(0.0, 1.0 - p_speak - p_complete)
        self._send({"type": "head", "t": self._rel(), "frame_ms": self.frame * 80,
                    "p": [round(p_speak, 3), round(p_complete, 3), round(rest, 3), 0.0, 0.0],
                    "armed": p_complete >= 0.9, "spec": False, "agent": self.replying, "turn": self.turn})
        if enough and self.quiet_frames * 80 >= STUB_END_MS:
            dur = self.speech_frames * 0.08
            self.speech_frames = self.quiet_frames = 0
            self.turn += 1
            self._send({"type": "endpoint", "turn": self.turn, "t": self._rel(), "reason": "stub",
                        "text": f"(stub: about {dur:.1f} s of speech)"})
            self.reply_task = asyncio.ensure_future(self._reply())
        elif not enough and self.quiet_frames * 80 >= STUB_END_MS:
            self.speech_frames = self.quiet_frames = 0

    # ------------------------------------------------------------------ fake LLM + TTS
    async def _reply(self):
        r = REPLIES[self.n_reply % len(REPLIES)]
        self.n_reply += 1
        self.utt += 1
        turn, utt = self.turn, self.utt
        await asyncio.sleep(0.25)
        self.replying = True
        self._send({"type": "reply_start", "turn": turn, "utt": utt, "kind": "speak"})
        if "search" in r:
            s = r["search"]
            self._send({"type": "search", "turn": turn, "query": s["query"], "t": self._rel()})
            await asyncio.sleep(s["ms"] / 1000)
            self._send({"type": "search_done", "turn": turn, "n": len(s["results"]), "ms": s["ms"], "results": s["results"]})
        if "tool" in r:
            self._send({"type": "tool", "turn": turn, "t": self._rel(), **r["tool"]})
        words = r["text"].split(" ")
        pcm = _hum(len(words))
        per_word = len(pcm) // len(words)
        sent_words = 0
        for i in range(0, len(pcm), CHUNK):
            chunk = pcm[i:i + CHUNK]
            self._send(struct.pack("<HH", utt, 0) + chunk.tobytes())
            while sent_words < len(words) and (sent_words + 1) * per_word <= i + CHUNK:
                self._send({"type": "reply_delta", "turn": turn, "text": (" " if sent_words else "") + words[sent_words]})
                sent_words += 1
            await asyncio.sleep(CHUNK / TTS_RATE * 0.95)
        if sent_words < len(words):
            self._send({"type": "reply_delta", "turn": turn, "text": " " + " ".join(words[sent_words:])})
        await asyncio.sleep(0.3)
        self.replying = False
        self._send({"type": "reply_end", "turn": turn})
        self._send({"type": "turn", "turn": turn, "reply": r["text"], "word_end_to_ear": None, "llm_ttft": None, "tts_ttfa": None})


def _hum(n_words: int) -> np.ndarray:
    """a quiet, voice-like hum: one soft syllable per word, so the playback and the UI's level meters move"""
    syl = int(0.28 * TTS_RATE)
    t = np.arange(syl) / TTS_RATE
    out = []
    rng = np.random.default_rng(0)
    for _ in range(n_words):
        f0 = 150 + 25 * rng.standard_normal()
        env = np.sin(np.pi * t / t[-1]) ** 2
        x = (np.sin(2 * np.pi * f0 * t) + 0.35 * np.sin(4 * np.pi * f0 * t) + 0.15 * np.sin(6 * np.pi * f0 * t)) * env
        out.append(x)
        out.append(np.zeros(int(0.04 * TTS_RATE)))
    y = np.concatenate(out) * 0.06
    return (y * 32767).astype(np.int16)
