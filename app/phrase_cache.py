#!/usr/bin/env python3
"""Pre-rendered first segments (2026-10-03): the TTS audio of the phrases replies most often START with ("Okay,", "Sure.",
"Checking.", "One sec," ...), rendered once per voice, so the first clause of a reply plays without waiting for TTS.

Phrases: data/phrases.json, built from the training replies (the first TTS segment exactly as session._cut
cuts it live: up to the first . ? ! ; : or comma after 3 characters, at most 6 words; the most frequent N) plus the
diversifier's lead-in pools (generator/diversify_v3.py). Rendering: the live TTS request exactly (Breeze API, the voice's
reference wav + its transcript, seed, cfg scale, the same int8 build). Breeze is not bit-reproducible across runs (takes of
one phrase differ, measured 10-03), so a clip is one take of what the live TTS would say, not the very same audio; leading silence
and trailing silence over MAX_TAIL_S are trimmed. Cache: phrase_cache/<voice key>/ (one .pcm per phrase + index.json); the voice key hashes the reference audio,
its transcript and every render setting, so another voice (a new clone) renders into its own directory and never reuses
another voice's clips.

    python3 phrase_cache.py render --voice <ref.wav>    renders what is missing for that voice (idempotent)
    python3 phrase_cache.py status --voice <ref.wav>
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PHRASE_CACHE_DIR", HERE.parent / "run" / "phrase_cache"))
SR = 24000
BREEZE_URL = os.environ.get("TTS_URL", "http://127.0.0.1:7860/v1/audio/speech")
FIRST_CLAUSE_CHARS = 3                       # session.MicroCfg.first_clause_chars
LEAD_KEEP_S = 0.03                           # leading silence kept before the first voiced sample
SILENCE = 300                                # |int16| below this is silence (about -40 dBFS)
MAX_TAIL_S = 0.15                            # trailing silence kept (a quarter of the takes end in 0.4-1 s of it: a gap before the
                                             # next clause, which plays right after the cached one)


def first_segment(text: str, need: int = FIRST_CLAUSE_CHARS):
    """the first TTS segment of a reply as session._cut cuts it (first=True: a comma ends it too); None if the
    text has no boundary"""
    seen = 0
    for i, ch in enumerate(text):
        seen += 1
        if seen >= need and (ch in ".?!;:\n" or ch == ","):
            return text[:i + 1].strip() or None
    return None


def norm(s: str) -> str:
    """lookup key: whitespace collapsed (the live segment passes through strip_tags + strip)"""
    return " ".join(s.split())


# ------------------------------------------------------------------ the phrase list
class PhraseCache:
    def __init__(self, ref: str, seed: int = 42, cfg_scale: float = 1.0, instruction: str | None = None,
                 engine: str = "breeze2-slim-int8:" + os.environ.get("BREEZE_INT8", "backbone,text_encoder,depth")):
        # engine: the TTS build the clips come from (the server runs Breeze 2 slim, int8 backbone,text_encoder,depth):
        # part of the voice key, so clips from another build are never served
        self.ref, self.seed, self.cfg_scale, self.instruction, self.engine = ref, seed, cfg_scale, instruction, engine
        self.ref_bytes = open(ref, "rb").read()
        self.ref_text = open(ref.rsplit(".", 1)[0] + ".txt").read().strip()
        key = json.dumps({"engine": engine, "ref_sha": hashlib.sha256(self.ref_bytes).hexdigest(), "ref_text": self.ref_text,
                          "seed": seed, "cfg": cfg_scale, "instruction": instruction, "trim": [LEAD_KEEP_S, SILENCE, MAX_TAIL_S]}, sort_keys=True)
        self.key = hashlib.sha256(key.encode()).hexdigest()[:16]
        self.dir = ROOT / f"{Path(ref).stem}_{self.key}"
        self.index = json.load(open(self.dir / "index.json")) if (self.dir / "index.json").exists() else {}
        self.pcm: dict[str, bytes] = {}

    def load(self):
        """phrase -> int16 PCM bytes, for every rendered phrase"""
        for p, f in self.index.items():
            fp = self.dir / f
            if fp.exists(): self.pcm[p] = fp.read_bytes()
        return self

    def get(self, segment: str) -> bytes | None:
        return self.pcm.get(norm(segment))

    def _render_one(self, text: str, url: str = BREEZE_URL) -> bytes:
        import requests
        data = {"text": text, "ref_text": self.ref_text, "seed": str(self.seed), "cfg_scale": str(self.cfg_scale)}
        if self.instruction: data["instruction"] = self.instruction
        for _ in range(500):                                         # 409: the server is busy with another request
            r = requests.post(url, data=data, files={"ref_audio": ("ref.wav", self.ref_bytes, "audio/wav")}, stream=True, timeout=(5, 60))
            if r.status_code != 409: break
            r.close(); time.sleep(0.05)
        r.raise_for_status()
        return b"".join(c for c in r.iter_content(chunk_size=None) if c)

    @staticmethod
    def trim_lead(pcm: bytes) -> bytes:
        import numpy as np
        a = np.frombuffer(pcm, dtype=np.int16)
        voiced = np.nonzero(np.abs(a) > SILENCE)[0]
        if not len(voiced): return pcm
        start = max(0, int(voiced[0]) - int(LEAD_KEEP_S * SR))
        end = min(len(a), int(voiced[-1]) + 1 + int(MAX_TAIL_S * SR))
        return a[start:end].tobytes()

    def render(self, phrases: list[str], log=print, urls: list[str] | None = None):
        """renders what is missing; several identical Breeze servers (urls) render in parallel (one request each: the
        server allows one at a time)"""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        urls = urls or [BREEZE_URL]
        self.dir.mkdir(parents=True, exist_ok=True)
        json.dump({"ref": self.ref, "ref_text": self.ref_text, "seed": self.seed, "cfg_scale": self.cfg_scale,
                   "instruction": self.instruction, "engine": self.engine}, open(self.dir / "voice.json", "w"), indent=1)
        todo = [p for p in phrases if p not in self.index]
        t0 = time.time(); st = {"k": 0, "trim": 0.0}; lock = threading.Lock()
        def worker(w):
            for p in todo[w::len(urls)]:
                raw = self._render_one(p, urls[w]); pcm = self.trim_lead(raw)
                f = hashlib.sha256(p.encode()).hexdigest()[:16] + ".pcm"
                (self.dir / f).write_bytes(pcm)
                with lock:
                    self.index[p] = f; st["k"] += 1; st["trim"] += (len(raw) - len(pcm)) / 2 / SR; k = st["k"]
                    if k % 50 == 0 or k == len(todo):
                        json.dump(self.index, open(self.dir / "index.json", "w"), indent=0, ensure_ascii=False)
                        log(f"[phrase cache] {k}/{len(todo)} rendered on {len(urls)} server(s) ({time.time() - t0:.0f} s, "
                            f"silence trimmed {st['trim'] / k * 1000:.0f} ms avg)")
        with ThreadPoolExecutor(len(urls)) as ex: list(ex.map(worker, range(len(urls))))
        json.dump(self.index, open(self.dir / "index.json", "w"), indent=0, ensure_ascii=False)
        return len(todo)


class PocketPhraseCache(PhraseCache):
    """The same clips rendered by Pocket TTS on audiocpp_server, with the live PocketWorker's request (tts.pocket_body).
    The key covers the engine, the server's model id and the voice (a wav's bytes, or a voice name), so Pocket clips
    never mix with Breeze clips or with another Pocket voice. Pocket samples too: a clip is one take, as with Breeze."""

    def __init__(self, voice: str, model: str = os.environ.get("TTS_MODEL", "pocket-tts")):
        from tts import pocket_voice_args
        self.voice, self.model, self.engine = voice, model, f"pocket:{model}"
        self.voice_args = pocket_voice_args(voice)
        ident = hashlib.sha256(open(voice, "rb").read()).hexdigest() if voice.endswith(".wav") else voice
        key = json.dumps({"engine": self.engine, "voice": ident, "trim": [LEAD_KEEP_S, SILENCE, MAX_TAIL_S]}, sort_keys=True)
        self.key = hashlib.sha256(key.encode()).hexdigest()[:16]
        self.dir = ROOT / f"pocket_{Path(voice).stem or 'default'}_{self.key}"
        self.index = json.load(open(self.dir / "index.json")) if (self.dir / "index.json").exists() else {}
        self.pcm: dict[str, bytes] = {}
        self.ref = voice; self.ref_text = self.seed = self.cfg_scale = self.instruction = None   # voice.json fields

    def _render_one(self, text: str, url: str = BREEZE_URL) -> bytes:
        import requests
        from tts import pocket_body
        for _ in range(500):                                         # 503: the model is busy with another request
            r = requests.post(url, json=pocket_body(self.model, text, self.voice_args), stream=True, timeout=(5, 60))
            if r.status_code != 503: break
            r.close(); time.sleep(0.05)
        r.raise_for_status()
        pcm = b"".join(c for c in r.iter_content(chunk_size=None) if c)
        return pcm[:len(pcm) & ~1]


def phrases_list():
    p = HERE / "data" / "phrases.json"
    return json.load(open(p))["phrases"] if p.exists() else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "status"])
    ap.add_argument("--engine", default=os.environ.get("TTS_ENGINE", "breeze"), choices=["breeze", "pocket"])
    ap.add_argument("--voice", default="", help="breeze: reference wav (its transcript next to it as .txt); pocket: a voice name or a wav")
    ap.add_argument("--urls", default=BREEZE_URL, help="comma-separated TTS servers to render on in parallel")
    a = ap.parse_args()
    if a.engine == "breeze" and not a.voice:
        ap.error("--voice is required for Breeze")
    pc = PhraseCache(a.voice) if a.engine == "breeze" else PocketPhraseCache(a.voice); ph = phrases_list()
    if a.cmd == "status":
        print(f"{pc.dir}: {sum(1 for p in ph if p in pc.index)}/{len(ph)} phrases rendered")
    else:
        n = pc.render(ph, urls=a.urls.split(",")); print(f"{pc.dir}: {n} rendered now, {len(pc.index)} total")
