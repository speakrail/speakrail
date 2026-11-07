"""TTS workers: text segments in, 24 kHz PCM chunks out, killable per utterance. One class per engine (ENGINES)."""
from __future__ import annotations

import asyncio
import base64
import queue
import re
import threading
import time

from ..config import CFG


class Segment:
    __slots__ = ("utt", "seg", "text", "samples", "done", "t_queued", "t_first")

    def __init__(self, utt: int, seg: int, text: str):
        self.utt, self.seg, self.text = utt, seg, text
        self.samples = 0
        self.done = False
        self.t_queued = time.monotonic()
        self.t_first = None


TAG_RE = re.compile(r"<\|[a-z_]+:[a-z_]+\|>")
VOCAL_RE = re.compile(r"\((?:laugh|sigh|cough|clears throat)\)")   # Breeze's inline vocal events


def strip_tags(text: str) -> str:
    """Remove <|category:value|> control tags (never spoken, never displayed)."""
    return " ".join(TAG_RE.sub("", text).split())


class TTSWorker(threading.Thread):
    """The interface the harness uses: say(seg) queues a segment, kill(utt) drops an utterance (queued segments and the
    one being spoken), stop() ends the thread; `ready` is set after the warm-up, `q` / `current` tell whether it is idle.
    Audio goes out through on_audio(seg, pcm) on the event loop (int16 little endian at `sample_rate`, even-length
    chunks), then on_segment_done(seg). An engine implements _stream(seg): yield PCM bytes for one segment, keep the open
    response in self.resp (kill closes it, which makes the server drop the request) and stop once seg.utt is killed."""

    engine = "?"
    sample_rate = 24000
    device = "?"

    def __init__(self, loop: asyncio.AbstractEventLoop, on_audio, on_segment_done):
        super().__init__(daemon=True, name=f"{self.engine}-tts")
        self.loop, self.on_audio, self.on_segment_done = loop, on_audio, on_segment_done
        self.q: queue.Queue[Segment | None] = queue.Queue()
        self.killed_utts: set[int] = set()
        self.current: Segment | None = None
        self.resp = None
        self.ready = threading.Event()
        self.stats = {"segments": 0, "ttfa_ms": [], "rtf": [], "silent_lead": []}

    def describe(self) -> str:
        return ""

    def say(self, seg: Segment):
        self.q.put(seg)

    def stop(self):
        self.q.put(None)

    def kill(self, utt: int):
        self.killed_utts.add(utt)
        keep = []
        try:
            while True:
                s = self.q.get_nowait()
                if s is not None and s.utt != utt:
                    keep.append(s)
        except queue.Empty:
            pass
        for s in keep:
            self.q.put(s)
        cur, resp = self.current, self.resp
        if cur is not None and cur.utt == utt and resp is not None:
            try:
                resp.close()                          # drops the stream -> the TTS server aborts the request
            except Exception:
                pass

    def _stream(self, seg: Segment):
        raise NotImplementedError

    def run(self):
        t0 = time.time()
        try:                                              # warm: graphs, caches, the voice on the server
            for _ in self._stream(Segment(-1, 0, "Hello there, warming up.")):
                pass
            print(f"[tts] {self.engine} ready ({self.describe()}) in {time.time()-t0:.1f}s, sr={self.sample_rate}", flush=True)
        except Exception as e:
            print(f"[tts] {self.engine} warmup failed: {e!r}", flush=True)
        self.ready.set()
        while True:
            seg = self.q.get()
            if seg is None:
                return
            if seg.utt in self.killed_utts:
                continue
            self.current = seg
            t_start = time.monotonic()
            try:
                for pcm in self._stream(seg):
                    if seg.t_first is None:
                        seg.t_first = time.monotonic()
                        self.stats["ttfa_ms"].append((seg.t_first - t_start) * 1000)
                        self.stats["silent_lead"].append(0)
                    seg.samples += len(pcm) // 2
                    self.loop.call_soon_threadsafe(self.on_audio, seg, pcm)
            except Exception as e:
                if seg.utt not in self.killed_utts:
                    print(f"[tts] {self.engine} error on {seg.text!r}: {e!r}", flush=True)
            self.resp = None
            seg.done = True
            dur = seg.samples / self.sample_rate
            if dur > 0:
                self.stats["rtf"].append((time.monotonic() - t_start) / dur)
            self.stats["segments"] += 1
            self.current = None
            self.loop.call_soon_threadsafe(self.on_segment_done, seg)


class BreezeWorker(TTSWorker):
    """Breeze TTS 2 over its streaming API (:7860): one multipart request per segment,
    raw 24 kHz int16 PCM streamed back in 80 ms chunks. `ref` is a wav whose
    transcript sits next to it as .txt (voice clone; the server caches the reference codes by content hash).
    Vocal events go inline in the text: (laugh) (sigh) (cough) (clears throat)."""

    engine = "breeze"
    device = "cuda"

    def __init__(self, ref: str, loop: asyncio.AbstractEventLoop, on_audio, on_segment_done,
                 url: str = CFG.tts_url or "http://127.0.0.1:7860/v1/audio/speech", seed: int = 42, instruction: str | None = None,
                 cfg_scale: float = 1.0):
        super().__init__(loop, on_audio, on_segment_done)
        self.ref = ref
        self.url, self.seed, self.instruction, self.cfg_scale = url, seed, instruction, cfg_scale
        self.ref_bytes = open(ref, "rb").read()
        self.ref_text = open(ref.rsplit(".", 1)[0] + ".txt").read().strip()

    def describe(self):
        return self.ref.rsplit("/", 1)[-1]

    def _stream(self, seg: Segment):
        import requests
        data = {"text": seg.text, "ref_text": self.ref_text, "seed": str(self.seed), "cfg_scale": str(self.cfg_scale)}
        if self.instruction:
            data["instruction"] = self.instruction
        files = {"ref_audio": ("ref.wav", self.ref_bytes, "audio/wav")}
        for _ in range(75):                               # 409 = the request we just killed is still releasing (~20-50 ms)
            r = requests.post(self.url, data=data, files=files, stream=True, timeout=(5, 60))
            if r.status_code != 409 or seg.utt in self.killed_utts:
                break
            r.close(); time.sleep(0.02)
        with r:
            self.resp = r
            r.raise_for_status()
            for pcm in r.iter_content(chunk_size=None):
                if seg.utt in self.killed_utts:
                    return
                if pcm:
                    yield pcm


def pocket_voice_args(voice: str) -> dict:
    """the voice part of a Pocket request: a wav is cloned (sent inline), anything else is a voice name ("" = default)"""
    if voice.endswith(".wav"):
        return {"voice_ref": {"type": "base64", "data": base64.b64encode(open(voice, "rb").read()).decode()}}
    return {"voice": voice} if voice else {}


def pocket_body(model: str, text: str, voice_args: dict) -> dict | None:
    """the /v1/audio/speech request for one segment (Breeze's vocal events dropped); None if nothing is left to say"""
    text = " ".join(VOCAL_RE.sub("", text).split())
    return {"model": model, "input": text, "response_format": "pcm", "stream_format": "audio", **voice_args} if text else None


class PocketWorker(TTSWorker):
    """Kyutai Pocket TTS on audiocpp_server (github.com/0xShug0/audio.cpp, a `pocket_tts` entry with mode "streaming",
    best on the CPU backend): one OpenAI-style JSON request per segment to /v1/audio/speech with stream_format "audio",
    raw 24 kHz int16 PCM back over chunked transfer. `voice` is a voice name the server knows (alba, ... or one of its
    voice_presets; "" = the entry's default_voice_preset) or a wav to clone, sent inline. Pocket has no vocal events:
    Breeze's (laugh) etc. are dropped from the text. TTS_MODEL is the entry's id in the server config."""

    engine = "pocket"
    device = "audio.cpp"

    def __init__(self, voice: str, loop: asyncio.AbstractEventLoop, on_audio, on_segment_done,
                 url: str = CFG.tts_url or "http://127.0.0.1:8091/v1/audio/speech",
                 model: str = CFG.tts_model):
        super().__init__(loop, on_audio, on_segment_done)
        self.url, self.model, self.voice = url, model, voice
        self.voice_args = pocket_voice_args(voice)        # a clone: the server caches the prepared voice state

    def describe(self):
        return f"{self.model}, voice {self.voice.rsplit('/', 1)[-1] or 'default'}"

    def _stream(self, seg: Segment):
        import requests
        body = pocket_body(self.model, seg.text, self.voice_args)
        if body is None:
            return
        for _ in range(75):                               # 503 = the model is still busy (a killed request releasing)
            r = requests.post(self.url, json=body, stream=True, timeout=(5, 60))
            if r.status_code != 503 or seg.utt in self.killed_utts:
                break
            r.close(); time.sleep(0.02)
        with r:
            self.resp = r
            r.raise_for_status()
            odd = b""                                     # chunk boundaries are arbitrary: keep whole int16 samples
            for pcm in r.iter_content(chunk_size=None):
                if seg.utt in self.killed_utts:
                    return
                pcm = odd + pcm
                odd = pcm[len(pcm) & ~1:]
                if len(pcm) > 1:
                    yield pcm[:len(pcm) & ~1]


ENGINES = {"breeze": BreezeWorker, "pocket": PocketWorker}


def make_worker(engine: str, voice: str, loop: asyncio.AbstractEventLoop, on_audio, on_segment_done) -> TTSWorker:
    if engine not in ENGINES:
        raise ValueError(f"unknown TTS engine {engine!r} (one of {', '.join(ENGINES)})")
    return ENGINES[engine](voice, loop, on_audio, on_segment_done)
