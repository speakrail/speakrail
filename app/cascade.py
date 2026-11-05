"""The base voice session: transport, ASR, turn head, peek, TTS streaming and the playback clock.

    mic 16 kHz --> Voxtral realtime ASR (audio.cpp, 480 ms delay) + turn head --> words + turn probabilities
               --> brain (session.MicroSession: the turn-taking LLM) --> clause splitter
               --> Breeze TTS 2 --> 24 kHz PCM --> speaker

Latency anchor: the END OF THE LAST WORD on the audio clock (Voxtral's `end`,
80 ms resolution), mapped to wall time through the moment that audio chunk
reached the harness.

The Session is transport-agnostic: `feed_audio(bytes)` in, `send(obj)` out
(dict -> JSON event, bytes -> PCM frame with a 4-byte utterance header).
`server.py` puts a browser on it.
"""
from __future__ import annotations

import asyncio
import bisect
import json
import os
import re
import time
import wave
from dataclasses import dataclass, field

import aiohttp
import numpy as np
import websockets
from scipy.signal import resample_poly

from backchannel import normalize
from tts import make_worker

SAMPLE_RATE = 16000


@dataclass
class Cfg:
    asr_url: str = "ws://127.0.0.1:8765/v1/transcribe"
    delay_ms: int = 480              # Voxtral transcription delay; the turn head is trained at 480
    barge_in: str = "duck"           # the user speaks over a reply: off | duck (lower our volume, the model decides) |
                                     # word (cut on the first new user word) | head (also cut on the head's speaking class)
    mic_rms: float = 0.003           # measurement only: "mic had energy" (-50 dBFS)
    voice: str = ""                  # Breeze reference wav; its transcript sits next to it as .txt
    search: str = "off"              # off | searxng (:8888) | serper (SERPER_API_KEY) | brave (BRAVE_API_KEY): the web_search tool
    search_url: str = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888/search")
    search_engines: str = "google,brave,duckduckgo,google cse"   # SearXNG upstreams per query. Each of them rate-limits a
                                     # server IP after a burst and comes back hours later, so ask all of them: suspended ones
                                     # are skipped instantly. "google cse" = the Programmable Search endpoint (separate quota).
    search_timeout: float = 2.5      # s; on timeout the model answers without results
    search_results: int = 5          # snippets handed back to the model (~60 tokens each)
    silence_ms: int = 0              # fallback: this much silence after words with no head fire ends the turn; 0 = off
    commit_extra_frames: int = 2     # after arming, wait delay + this many frames so the last word's text has arrived
    peek: bool = True                # on an endpoint, ask the ASR to drain its delay line on a forked decoder
    peek_frames: int = 0             # frames of silence to feed the fork (0 = delay/80 + 2)
    # speculative reply: on the first silent frame with P(fire) >= spec_tau, peek and start the reply with the audio
    # held server-side; the endpoint rule then only releases it. Speech (a head "speaking" frame, or a live word the peek
    # didn't have) discards it.
    spec: bool = True
    spec_tau: float = 0.5


class AudioClock:
    """audio-ms -> wall time, from when each chunk reached the harness."""
    def __init__(self):
        self.ends: list[float] = []      # cumulative audio ms at the end of each chunk
        self.walls: list[float] = []
        self.samples = 0

    def push(self, n_samples: int, t: float):
        self.samples += n_samples
        self.ends.append(self.samples * 1000.0 / SAMPLE_RATE)
        self.walls.append(t)
        if len(self.ends) > 6000:
            del self.ends[:3000]; del self.walls[:3000]

    @property
    def audio_ms(self) -> float:
        return self.samples * 1000.0 / SAMPLE_RATE

    def wall_of(self, audio_ms: float) -> float | None:
        if not self.ends:
            return None
        i = bisect.bisect_left(self.ends, audio_ms)
        return self.walls[min(i, len(self.walls) - 1)]

    def ms_at(self, t: float) -> float:
        """Inverse: the mic position (audio ms) at wall time t, extrapolated from the last chunk before t."""
        if not self.walls:
            return 0.0
        i = bisect.bisect_right(self.walls, t) - 1
        if i < 0:
            return max(0.0, self.ends[0] - (self.walls[0] - t) * 1000)
        return self.ends[i] + (t - self.walls[i]) * 1000


class Tape:
    """Stereo session recording on the mic clock: left = user (mic as fed), right = agent (TTS as the
    browser played it). Agent chunks are placed the way app.js schedules them: contiguous from the
    reported play_start, re-anchored to arrival when the buffer underruns; a stop_audio cuts the
    utterance where it was heard. Chunks of an utterance that never reached play_start were never
    heard and are dropped. Positions are frozen at placement time because the clock forgets."""
    TTS_RATE = 24000

    def __init__(self, clock: AudioClock):
        self.clock = clock
        self.mic: list[bytes] = []
        self.utts: dict[int, dict] = {}

    def mic_chunk(self, pcm: bytes):
        self.mic.append(pcm)

    def _utt(self, utt: int) -> dict:
        return self.utts.setdefault(utt, {"t_play": None, "lat": 0.0, "cursor": 0.0,
                                          "pending": [], "chunks": [], "cut_ms": None})

    def tts_chunk(self, utt: int, pcm: bytes, t_send: float):
        u = self._utt(utt)
        if u["t_play"] is None:
            u["pending"].append((t_send, pcm))
        else:
            self._place(u, t_send, pcm)

    def play_start(self, utt: int, t_play: float):
        u = self._utt(utt)
        if u["t_play"] is not None:
            return
        u["t_play"] = u["cursor"] = t_play
        if u["pending"]:
            u["lat"] = t_play - u["pending"][0][0]          # send -> speaker for this utterance's first chunk
        for t_send, pcm in u["pending"]:
            self._place(u, t_send, pcm)
        u["pending"] = []

    def _place(self, u: dict, t_send: float, pcm: bytes):
        start = max(u["cursor"], t_send + u["lat"])          # app.js: playCursor = max(playCursor, now)
        u["cursor"] = start + len(pcm) / 2 / self.TTS_RATE
        u["chunks"].append((self.clock.ms_at(start), np.frombuffer(pcm, dtype=np.int16)))

    def stop(self, utt: int, t: float):
        u = self.utts.get(utt)
        if u is not None and u["cut_ms"] is None:
            u["cut_ms"] = self.clock.ms_at(t)

    def write(self, path: str):
        mic = np.frombuffer(b"".join(self.mic), dtype=np.int16)
        if not mic.size:
            return None
        r = self.TTS_RATE
        left = resample_poly(mic.astype(np.float32), r // 8000, SAMPLE_RATE // 8000)   # 16k -> 24k
        right = np.zeros_like(left)
        n = left.size
        for u in self.utts.values():
            for ms, pcm in u["chunks"]:
                a = int(round(ms / 1000 * r))
                if u["cut_ms"] is not None:
                    pcm = pcm[:max(0, int(round((u["cut_ms"] - ms) / 1000 * r)))]
                b = min(a + pcm.size, n)
                if a < b:
                    right[a:b] += pcm[:b - a].astype(np.float32)
        tape = np.stack([left, right], axis=1)
        tape = np.clip(tape, -32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(r)
            w.writeframes(tape.tobytes())
        return {"path": path, "seconds": round(n / r, 2), "utterances": sum(1 for u in self.utts.values() if u["chunks"])}


@dataclass
class Turn:
    id: int
    words: list[dict] = field(default_factory=list)
    t_last_word_ev: float | None = None      # wall: transcript event carrying the last word
    last_word_end_ms: float | None = None    # audio clock
    t_word_end: float | None = None          # wall of that audio position, frozen at arrival (the clock forgets)
    t_mic_offset: float | None = None        # wall: last chunk with mic energy before the endpoint
    t_endpoint: float | None = None
    ep_reason: str | None = None             # punct | head | silence (+peek)
    t_head_fire: float | None = None         # wall: the head armed the endpoint (before the text commit wait)
    t_peek: float | None = None              # wall: the peek transcript arrived
    t_spec: float | None = None              # wall: the speculative reply was triggered (its peek sent), if it was promoted
    peek_ms: float | None = None             # GPU cost of the peek
    peek_words: int = 0                      # words that came from the peek rather than the live stream
    t_llm_sent: float | None = None
    t_first_token: float | None = None
    t_llm_done: float | None = None
    t_tts_in: float | None = None
    t_first_pcm: float | None = None
    t_play: float | None = None
    reply: str = ""
    n_tokens: int = 0
    search_query: str | None = None          # the web_search call, if the model made one
    t_search_sent: float | None = None       # wall: tool call complete -> search fired
    t_search_done: float | None = None       # wall: results (or timeout/error) handed back to the LLM
    search_n: int | None = None              # results returned; -1 = failed/timed out
    search_filler: bool = False              # the model called without a lead-in, so the harness spoke one
    search_results: list[dict] = field(default_factory=list)   # title/snippet/url of what was handed to the model (UI card)
    cut: str = ""                            # why the reply was cut, if it was
    premature: bool | None = None            # the user kept talking right after the endpoint
    next_gap_ms: float | None = None
    logged: bool = False

    def text(self) -> str:
        return "".join(w["raw"] for w in self.words).strip()

    def chain(self, clock: AudioClock) -> dict:
        def d(a, b):
            return None if a is None or b is None else round((b - a) * 1000)
        w_end = self.t_word_end
        return {
            "turn": self.id, "cut": self.cut, "premature": self.premature, "next_gap_ms": self.next_gap_ms,
            "ep_reason": self.ep_reason,
            "word_end_to_head": d(w_end, self.t_head_fire),            # last word's audio end -> head armed
            "head_to_peek": d(self.t_head_fire, self.t_peek), "peek_ms": self.peek_ms, "peek_words": self.peek_words,
            "spec": self.t_spec is not None, "word_end_to_spec": d(w_end, self.t_spec),
            "spec_lead": d(self.t_spec, self.t_endpoint),            # speculation head start over the endpoint
            "word_end_to_endpoint": d(w_end, self.t_endpoint),         # -> LLM fired (after the text commit wait)
            "mic_offset_to_endpoint": d(self.t_mic_offset, self.t_endpoint),
            "word_end_to_asr": d(w_end, self.t_last_word_ev),
            "mic_offset_to_asr": d(self.t_mic_offset, self.t_last_word_ev),
            "llm_ttft": d(self.t_llm_sent, self.t_first_token),
            "first_token_to_tts_in": d(self.t_first_token, self.t_tts_in),
            "tts_ttfa": d(self.t_tts_in, self.t_first_pcm),
            "pcm_to_play": d(self.t_first_pcm, self.t_play),
            "word_end_to_first_pcm": d(w_end, self.t_first_pcm),
            "word_end_to_ear": d(w_end, self.t_play),
            "mic_offset_to_ear": d(self.t_mic_offset, self.t_play),
            "llm_total": d(self.t_llm_sent, self.t_llm_done), "n_tokens": self.n_tokens,
            "search": self.search_query, "search_n": self.search_n, "search_filler": self.search_filler or None,
            "tool_call_ms": d(self.t_llm_sent, self.t_search_sent),   # decide + emit the call (incl. the spoken lead-in)
            "search_ms": d(self.t_search_sent, self.t_search_done),
            "user": self.text()[:160], "reply": self.reply[:160],
        }


class Spec:
    """A speculative reply for the turn being collected: triggered at a lower head threshold, its audio and text
    deltas are held until the head's normal rule commits the turn (promote) or the user speaks again (discard)."""
    def __init__(self, t: float, frame_ms: float):
        self.t, self.frame_ms = t, frame_ms
        self.peek_id = f"spec:{frame_ms}"
        self.peek_ev: dict | None = None     # the peek transcript (None until it arrives)
        self.t_peek: float | None = None
        self.norm: list[str] = []            # normalized words the reply was generated for
        self.user_text = ""
        self.started = False                 # the LLM task is running
        self.held = True                     # audio/deltas buffered; False once promoted
        self.dead = False                    # discarded: its task must not touch the history
        self.promote: tuple | None = None    # (t, reason) if the head committed before the peek came back
        self.prev_turn: "Turn | None" = None # reply_turn before the speculation (may still be playing in the browser)
        self.ev_save = (0, 0)                # Breeze event budget to restore on discard
        self.pcm: list[tuple] = []           # (utt, pcm, t_generated)
        self.deltas: list[str] = []
        self.hist: list[dict] | None = None  # history entries of a reply that finished while held


class TTSHub:
    """One TTS worker per process; utterance ids are global, routed to sessions.
    engine: a tts.ENGINES key (breeze: voice = reference wav with its transcript in a sibling .txt; pocket: a voice
    name or a wav to clone)."""
    def __init__(self, loop, voice, engine="breeze"):
        self.owners: dict[int, "Session"] = {}
        self.next_utt = 1
        self.engine = engine
        self.worker = make_worker(engine, voice, loop, self._on_audio, self._on_done)
        self.worker.start()

    async def wait_ready(self):
        await asyncio.get_running_loop().run_in_executor(None, self.worker.ready.wait)

    def new_utt(self, s: "Session") -> int:
        u = self.next_utt; self.next_utt += 1
        self.owners[u] = s
        return u

    def idle(self) -> bool:
        return self.worker.q.empty() and self.worker.current is None

    def _on_audio(self, seg, pcm):
        s = self.owners.get(seg.utt)
        if s: s.on_tts_audio(seg, pcm)

    def _on_done(self, seg):
        s = self.owners.get(seg.utt)
        if s: s.on_tts_done(seg)


class Session:
    def __init__(self, cfg: Cfg, hub: TTSHub, send, log_dir: str | None = None):
        self.cfg, self.hub, self._send_out = cfg, hub, send
        # events.jsonl: every event the client gets (head probs per frame, spec/arm/peek/endpoint, reply deltas...) plus
        # log-only ones (llm_request, tts_seg, play_start), streamed line by line so a crashed session keeps its log.
        # Each line: the event + "rel" (s since session start), "wall" (epoch s), "audio_ms" (mic audio fed so far).
        self.evlog = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.evlog = open(f"{log_dir}/events.jsonl", "a", buffering=1)
        self.clock = AudioClock()
        self.tape = Tape(self.clock) if log_dir else None
        self.asr_ws = None
        self.asr_gen = 0
        self.asr_offset_ms = 0.0             # audio ms at which the current ASR session's audio starts
        self.asr_readers: list[asyncio.Task] = []
        self.asr_out: asyncio.Queue = asyncio.Queue()
        self.asr_sender = None
        self.reconnects = 0
        self.http: aiohttp.ClientSession | None = None
        self.history: list[dict] = []
        self.turn: Turn | None = None          # the user turn being collected
        self.reply_turn: Turn | None = None    # the turn whose reply is being produced/spoken
        self.last_ended: Turn | None = None
        self.turn_objs: list[Turn] = []
        self.n_turns = 0
        self.llm_task: asyncio.Task | None = None
        self.utt: int | None = None
        self.out_buf = ""
        self.n_seg = 0
        self.seg_restart = -1                # n_seg at which a reply resumed after a search (first-clause cut rule again)
        self.speaking = False
        self.spec: Spec | None = None
        self.mic_last_hot: float | None = None
        self.words_all: list[dict] = []
        self.word_lags: list[float] = []
        self.counters = {"endpoints": 0, "premature": 0, "replaced": 0, "barge_ins": 0, "asr_reconnects": 0,
                         "head_fires": 0, "silence_fires": 0, "backchannels": 0, "cancelled_arms": 0, "events_stripped": 0,
                         "searches": 0, "search_failed": 0, "search_fillers": 0,
                         "spec_fired": 0, "spec_promoted": 0, "spec_discarded": 0, "spec_mismatch": 0}
        # turn head state
        self.head_p: list[float] | None = None
        self.head_hits = 0                   # consecutive silent frames with P(fire) >= tau
        self.sil_frames = 0                  # consecutive frames the head calls non-speaking
        self.spk_frames = 0                  # consecutive frames the head calls speaking
        self.seg_bc = 0.0                    # max P(backchannel) seen in the current speech segment (+ its silence)
        self.seg_bc_logged = False
        self.pending_ep: dict | None = None  # armed endpoint waiting for the last word's text
        self.last_head_frame = -1
        self.consumed_until_ms = -1.0        # live words ending before this were already delivered by a peek
        self.events_in_reply = 0             # Breeze vocal events emitted in the current reply
        self.last_event_turn = -10**6        # reply turn id of the last emitted vocal event
        self.peek_tail: list[str] = []       # normalized texts of the last peek's words (the fork dates them up to a frame early)
        self.closed = False
        self.t0 = time.monotonic()
        self.log_dir = log_dir

    # ------------------------------------------------------------- event log
    def send(self, obj):
        if self.evlog is not None and isinstance(obj, dict):
            self._log(obj)
        self._send_out(obj)

    def _log(self, obj: dict):
        """Write one event to events.jsonl without sending it to the client."""
        if self.evlog is None:
            return
        try:
            rec = {**obj, "rel": round(time.monotonic() - self.t0, 4), "wall": round(time.time(), 4),
                   "audio_ms": round(self.clock.audio_ms, 1)}
            self.evlog.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------- lifecycle
    def _asr_url(self):
        return (f"{self.cfg.asr_url}?encoding=pcm_s16le&sample_rate=16000&channels=1"
                f"&delay_ms={self.cfg.delay_ms}&turn_detection=none&client_id=speakrail&turn_probs=true")

    async def start(self):
        self.http = aiohttp.ClientSession()
        self.asr_ws = await websockets.connect(self._asr_url(), max_size=None)
        created = json.loads(await self.asr_ws.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"asr: {created}")
        self.asr_readers.append(asyncio.create_task(self._asr_reader(self.asr_ws, self.asr_gen, 0.0)))
        self.asr_sender = asyncio.create_task(self._asr_send_loop())
        self.send({"type": "ready", "asr": created, "cfg": self.cfg.__dict__})

    async def reconnect_asr(self):
        """Manual: open a fresh ASR session (for when the decoder has gone mute)."""
        old = self.asr_ws
        ws = await websockets.connect(self._asr_url(), max_size=None)
        created = json.loads(await ws.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"asr: {created}")
        self.asr_gen += 1
        self.asr_offset_ms = self.clock.audio_ms
        self.asr_ws = ws
        self.asr_readers.append(asyncio.create_task(self._asr_reader(ws, self.asr_gen, self.asr_offset_ms)))
        self.counters["asr_reconnects"] += 1
        self.send({"type": "asr_reconnect", "at_ms": round(self.asr_offset_ms), "n": self.counters["asr_reconnects"]})
        try:
            await old.send(json.dumps({"type": "input_audio.end"}))
        except Exception:
            pass

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self._cancel_reply("session closed")
        try:
            await self.asr_out.put(None)
            await asyncio.wait_for(self.asr_sender, timeout=5)
            await asyncio.wait_for(asyncio.gather(*self.asr_readers, return_exceptions=True), timeout=8)
        except Exception:
            pass
        try:
            if self.asr_ws: await self.asr_ws.close()
        except Exception:
            pass
        if self.http:
            await self.http.close()
        for tr in self.turn_objs:
            if tr.t_endpoint is not None and not tr.logged:
                self._log_turn(tr)
        self.send({"type": "summary", **self.summary()})
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            json.dump(self.chains(), open(f"{self.log_dir}/turns.json", "w"), indent=1)
            json.dump(self.summary(), open(f"{self.log_dir}/summary.json", "w"), indent=1)
            with open(f"{self.log_dir}/words.jsonl", "w") as f:
                for w in self.words_all: f.write(json.dumps(w) + "\n")
            if self.tape:
                info = self.tape.write(f"{self.log_dir}/tape.wav")
                if info: print(f"[tape] {info['seconds']} s, {info['utterances']} agent utterances -> {info['path']}", flush=True)
        if self.evlog is not None:
            self.evlog.close(); self.evlog = None

    # ------------------------------------------------------------- audio in
    def feed_audio(self, pcm: bytes):
        """int16 mono 16 kHz from the client. Forward to the ASR, keep the clock."""
        if self.closed or not self.asr_ws:
            return
        t = time.monotonic()
        self.clock.push(len(pcm) // 2, t)
        if self.tape: self.tape.mic_chunk(pcm)
        a = np.frombuffer(pcm, dtype=np.int16)
        if a.size and float(np.sqrt(np.mean((a.astype(np.float32) / 32768) ** 2))) > self.cfg.mic_rms:
            self.mic_last_hot = t
        self.asr_out.put_nowait(pcm)

    async def _asr_send_loop(self):
        try:
            while True:
                pcm = await self.asr_out.get()
                if pcm is None:
                    await self.asr_ws.send(json.dumps({"type": "input_audio.end"}))
                    return
                await self.asr_ws.send(pcm)
        except Exception as e:
            self.send({"type": "error", "where": "asr send", "detail": repr(e)})

    # ------------------------------------------------------------- ASR events
    async def _asr_reader(self, ws, gen, offset_ms):
        try:
            async for msg in ws:
                if isinstance(msg, bytes) or gen != self.asr_gen:
                    continue
                ev = json.loads(msg)
                t = time.monotonic()
                k = ev.get("type")
                if k == "transcript":
                    for w in ev.get("words", []):
                        w["start"] += offset_ms; w["end"] += offset_ms
                    self._on_words(ev, t)
                elif k == "turn":
                    self._on_head(ev, t, offset_ms)
                elif k == "transcript.peek":
                    for w in ev.get("words", []):
                        w["start"] += offset_ms; w["end"] += offset_ms
                    self._on_peek(ev, t)
                elif k == "error":
                    self.send({"type": "error", "where": "asr", "detail": ev})
                elif k == "session.ended":
                    break
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.send({"type": "error", "where": "asr reader", "detail": repr(e)})

    def _peek_duplicate(self, w) -> bool:
        """The live stream re-emits the words a peek already delivered, dated up to ~1 frame later than the fork
        dated them (seen: peek 'access?' end 11200, live 'access?' end 11280 -> a duplicate turn and a cut reply)."""
        if w["end"] <= self.consumed_until_ms:
            return True
        if w["end"] > self.consumed_until_ms + 240:
            return False
        text = " ".join(normalize(w["text"]))
        if text in self.peek_tail:
            return True
        # the same word, re-transcribed: the live stream spelled the peek's last word differently (live: peek 'FDBV3'
        # 19040-19360, live 'FDBB3?' 19040-19440 -> taken as new speech -> the early start undone). Same start, close spelling.
        pl = getattr(self, "peek_last", None)
        if pl is not None and abs(w["start"] - pl[0]) <= 80 and text and pl[1]:
            from difflib import SequenceMatcher
            if text.startswith(pl[1]) or pl[1].startswith(text) or SequenceMatcher(None, text, pl[1]).ratio() >= 0.6:
                return True
        return False

    async def _web_search(self, tr: Turn, query: str) -> str:
        """Search backend -> a numbered list of title: snippet (SearXNG infobox first if there is one). Never raises:
        a timeout or error becomes a short note so the model answers without results instead of stalling.
        "searxng": Cfg.search_url JSON API. "brave": Brave Search API (BRAVE_API_KEY; ~500 ms, $5/1k, own index) for
        when SearXNG's upstreams are captcha'd -- written from the API docs, not yet exercised here."""
        try:
            timeout = aiohttp.ClientTimeout(total=self.cfg.search_timeout)
            if self.cfg.search == "serper":
                # Serper (Google results, SERPER_API_KEY)
                async with self.http.post("https://google.serper.dev/search", json={"q": query, "num": self.cfg.search_results},
                                          headers={"X-API-KEY": os.environ.get("SERPER_API_KEY", "")}, timeout=timeout) as r:
                    if r.status != 200:
                        raise RuntimeError(f"serper {r.status}")
                    j = await r.json()
                box = j.get("answerBox") or {}; kg = j.get("knowledgeGraph") or {}
                info = box.get("answer") or box.get("snippet") or kg.get("description") or ""
                d = {"infoboxes": [{"content": info}] if info else [],
                     "results": [{"title": x.get("title", ""), "content": x.get("snippet", ""), "url": x.get("link", "")}
                                 for x in j.get("organic") or []]}
            elif self.cfg.search == "brave":
                async with self.http.get("https://api.search.brave.com/res/v1/web/search",
                                         params={"q": query, "count": self.cfg.search_results, "text_decorations": "false"},
                                         headers={"X-Subscription-Token": os.environ.get("BRAVE_API_KEY", ""),
                                                  "Accept": "application/json"}, timeout=timeout) as r:
                    if r.status != 200:
                        raise RuntimeError(f"brave {r.status}")
                    d = await r.json()
                d = {"results": [{"title": x.get("title", ""), "content": x.get("description", "")}
                                 for x in (d.get("web") or {}).get("results") or []]}
            else:
                params = {"q": query, "format": "json", "engines": self.cfg.search_engines, "language": "en"}
                async with self.http.get(self.cfg.search_url, params=params, timeout=timeout) as r:
                    if r.status != 200:
                        raise RuntimeError(f"searxng {r.status}")
                    d = await r.json()
        except Exception as e:                       # noqa: BLE001 -- the reply must go on
            tr.search_n = -1; self.counters["search_failed"] += 1
            return f"Search failed ({type(e).__name__}); answer from what you know and say you could not check."
        lines = []
        for box in d.get("infoboxes") or []:
            if box.get("content"):
                lines.append(f"Summary: {box['content'][:400]}")
                break
        res = (d.get("results") or [])[: self.cfg.search_results]
        tr.search_n = len(res)
        for i, x in enumerate(res, 1):
            snippet = re.sub(r"\s+", " ", x.get("content") or "").strip()
            lines.append(f"{i}. {x.get('title', '').strip()}: {snippet[:300]}")
            tr.search_results.append({"title": x.get("title", "").strip()[:120], "snippet": snippet[:200], "url": x.get("url", "")})
        if not lines:
            return "No results; say you could not find anything on that."
        return "\n".join(lines)

    def _send_pcm(self, utt: int, pcm: bytes):
        if self.tape: self.tape.tts_chunk(utt, pcm, time.monotonic())
        self.send(utt.to_bytes(2, "little") + b"\x00\x00" + pcm)

    def _log_turn(self, tr: Turn, again: bool = False):
        if tr.logged and not again:
            return
        tr.logged = True
        self.send({"type": "turn", **tr.chain(self.clock)})

    # ------------------------------------------------------------- summary
    def chains(self) -> list[dict]:
        return [tr.chain(self.clock) for tr in self.turn_objs if tr.t_endpoint is not None]

    def summary(self) -> dict:
        def pct(a, q):
            if not a: return None
            s = sorted(a); return s[min(len(s) - 1, int(len(s) * q))]
        turns = self.chains()
        keys = ["word_end_to_spec", "spec_lead", "word_end_to_head", "head_to_peek", "word_end_to_endpoint", "mic_offset_to_endpoint",
                "word_end_to_asr", "mic_offset_to_asr", "llm_ttft", "first_token_to_tts_in", "tts_ttfa",
                "pcm_to_play", "word_end_to_first_pcm", "word_end_to_ear", "mic_offset_to_ear", "llm_total",
                "tool_call_ms", "search_ms"]
        out = {"seconds": round(self.clock.audio_ms / 1000, 1), "turns": len(turns),
               "spoken_turns": sum(1 for t in turns if t["word_end_to_ear"] is not None),
               "words": len(self.words_all), "delay_ms": self.cfg.delay_ms,
               "word_lag_ms": {"n": len(self.word_lags), "p50": pct(self.word_lags, .5),
                               "p90": pct(self.word_lags, .9), "max": max(self.word_lags) if self.word_lags else None},
               "counters": dict(self.counters), "history_turns": len(self.history) // 2}
        for k in keys:
            v = [t[k] for t in turns if t.get(k) is not None]
            out[k] = {"n": len(v), "p50": pct(v, .5), "p90": pct(v, .9), "max": max(v) if v else None}
        w = self.hub.worker.stats
        out["tts"] = {"engine": self.hub.engine, "device": self.hub.worker.device, "segments": w["segments"], "ttfa_p50": pct(w["ttfa_ms"], .5),
                      "rtf_p50": pct(w["rtf"], .5), "silent_lead_p50": pct(w["silent_lead"], .5)}
        return out
