"""Audio for the gateway: a streaming 24 kHz -> 16 kHz resampler, and real-time pacing of what goes out."""
from __future__ import annotations

import asyncio
import time

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi


class Downsampler:
    """24 kHz -> 16 kHz int16 in chunks of any size: the same samples come out however the input is cut up.
    The filter is scipy.signal.resample_poly's, the one every benchmark used, and its 10 samples (0.6 ms) of delay are kept: dropping
    them would make each 80 ms of input come out a little short, and the harness times words by how much audio has arrived.
    Recognition doesn't depend on this filter's details: shifting identical audio by a few samples changes 5-9 words in 280 either way."""
    UP, DOWN = 2, 3
    HALF_LEN = 10 * max(UP, DOWN)

    def __init__(self):
        self.h = firwin(2 * self.HALF_LEN + 1, 1.0 / max(self.UP, self.DOWN), window=("kaiser", 5.0)) * self.UP
        self.zi = lfilter_zi(self.h, 1.0) * 0.0
        self.phase = 0                                         # position in the 48 kHz stream, modulo DOWN
        self.odd = b""

    def process(self, pcm: bytes) -> bytes:
        pcm = self.odd + pcm
        cut = len(pcm) & ~1
        self.odd = pcm[cut:]
        x = np.frombuffer(pcm[:cut], dtype="<i2").astype(np.float64)
        if not x.size:
            return b""
        up = np.zeros(x.size * self.UP)
        up[::self.UP] = x
        y, self.zi = lfilter(self.h, 1.0, up, zi=self.zi)
        start = (-self.phase) % self.DOWN
        out = y[start::self.DOWN]
        self.phase = (self.phase + up.size) % self.DOWN
        return np.clip(np.round(out), -32768, 32767).astype("<i2").tobytes()


class Pacer:
    """sends audio about as fast as it plays, ahead by `lead_s`. Audio not yet sent can still be changed: an interruption drops it,
    and ducking lowers it, with no help from the client. So a client that only plays what it receives still behaves."""
    CHUNK_S = 0.04

    def __init__(self, send, rate: int, lead_s: float = 0.2, on_first=None):
        self.send, self.rate, self.lead_s, self.on_first = send, rate, lead_s, on_first
        self.queue: list[tuple[int, bytes]] = []
        self.gain = 1.0
        self.wake = asyncio.Event()
        self.play_end = 0.0                                    # when what was already sent finishes playing
        self.seen: set[int] = set()
        self.resamplers: dict[int, Downsampler] = {}
        self.generation = 0

    def push(self, utt: int, pcm: bytes) -> None:
        if self.rate == 16000:
            pcm = self.resamplers.setdefault(utt, Downsampler()).process(pcm)
        self.queue.append((utt, pcm))
        self.wake.set()

    def clear(self) -> None:
        self.queue.clear()
        self.generation += 1

    def set_gain(self, gain: float) -> None:
        self.gain = gain

    async def run(self) -> None:
        size = int(self.rate * self.CHUNK_S) * 2
        while True:
            if not self.queue:
                self.wake.clear()
                await self.wake.wait()
                continue
            utt, pcm = self.queue[0]
            if len(pcm) <= size:
                self.queue.pop(0)
                chunk = pcm
            else:
                chunk, self.queue[0] = pcm[:size], (utt, pcm[size:])
            if not chunk:
                continue
            ahead = self.play_end - time.monotonic() - self.lead_s
            if ahead > 0:
                generation = self.generation
                await asyncio.sleep(ahead)
                if generation != self.generation:              # an interruption came while this chunk waited: it never goes out
                    continue
            now = time.monotonic()
            if utt not in self.seen:
                self.seen.add(utt)
                if self.on_first:
                    self.on_first(utt)
            self.play_end = max(self.play_end, now) + len(chunk) / 2 / self.rate
            if self.gain != 1.0:
                a = np.frombuffer(chunk, dtype="<i2").astype(np.float32) * self.gain
                chunk = np.clip(a, -32768, 32767).astype("<i2").tobytes()
            await self.send(utt, chunk)
