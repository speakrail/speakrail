#!/usr/bin/env python3
"""Speakrail ASR bridge: a WebSocket front for audio.cpp's Voxtral realtime with the turn head and peek.

The harness (app/cascade.py) speaks a small streaming protocol on /v1/transcribe: raw 16 kHz PCM in, `transcript`
events with timed words and `turn` events with the turn head's class probabilities every 80 ms out. This bridge runs
each session as one `POST /v1/audio/transcriptions/live` request against audiocpp_server (chunked PCM up, SSE down),
reads the server's `transcript.steps` events (one per decoder step: frame, token, text, turn probabilities) and
groups tokens into words.

    python bridge.py --port 8765 --acpp 127.0.0.1:8090 --model voxtral-rt --head /models/turn_head.vxth.json

Time base: the prefill's token is audio frame 0 and every decoder step adds one 80 ms frame; a word ends at
(last frame + 1) * 80 ms; a `turn` event is stamped with the audio frame the head saw (token frame + delay).
audio.cpp fixes the emission delay at 480 ms (6 frames), so only delay_ms=480 is accepted.
Mute restart (query `mute_restart=false` to disable): Voxtral's decoder can fall into emitting only [STREAMING_PAD]
on some recordings. The bridge then restarts the decoder (a new live request seeded with the last frames) when no text
came out for 2.4 s of speech, judged by the head's P(speaking) or, without a head, PCM energy. Timestamps stay on the
session clock; a `{"type":"asr_restart"}` event marks it.
Peek: `{"type":"peek","frames":N,"id":..}` -> `transcript.peek`: the live request carries a `live_id`, and
POST /v1/audio/transcriptions/live/peek runs N steps of silence on a snapshot of the decoder (on the stream's own
thread, restored afterwards); the steps are assembled into words on a copy of this session's WordAssembler.
Not supported: diarization, multichannel, sample rates other than 16 kHz, encodings other than pcm_s16le.
"""
import argparse
import asyncio
import copy
import json
import logging
import os
import time
from urllib.parse import parse_qs, urlparse

from websockets.asyncio.server import serve

log = logging.getLogger("asr_bridge")
FRAME_MS = 80
ACPP_DELAY_MS = 480          # audio.cpp tokenizer_text.cpp: 0.480 s of delay pads, not configurable
ARGS = None
HEAD_META = {}
ACTIVE = set()               # one live session at a time (audiocpp_server holds the model lock per request)


class WordAssembler:
    """Groups sub-word tokens into words on leading-space boundaries.
    A token that starts with whitespace opens a new word; a control token (decodes to "") closes the current one.
    `sf` is the first token's frame, `ef` one past the last."""

    def __init__(self):
        self._raw = ""
        self._start = None
        self._end = None

    def push(self, text, frame):
        out = []
        if text == "":
            out.extend(self._close())
            return out
        if text[0].isspace() and self._raw:
            out.extend(self._close())
        if not self._raw:
            self._start = frame
        self._raw += text
        self._end = frame
        return out

    def flush(self):
        return self._close()

    def _close(self):
        if not self._raw:
            return []
        word = None
        if self._raw.strip():
            word = {"raw": self._raw, "text": self._raw.strip(), "sf": max(self._start, 0), "ef": max(self._end, 0) + 1}
        self._raw = ""
        self._start = self._end = None
        return [word] if word else []


def _bad(ws_send, code, msg):
    return ws_send({"type": "error", "code": code, "message": msg})


async def http_post_json(host, port, path, timeout=2.0):
    """a bodyless POST to audiocpp_server -> parsed JSON body (its JSON replies carry Content-Length)"""
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    try:
        writer.write(f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout)
    finally:
        writer.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    if b"transfer-encoding: chunked" in head.lower():
        out, rest = b"", body
        while rest:
            n, _, rest = rest.partition(b"\r\n")
            size = int(n.strip() or b"0", 16)
            if size == 0: break
            out += rest[:size]; rest = rest[size + 2:]
        body = out
    return json.loads(body.decode(errors="replace") or "{}")


class LiveRequest:
    """One chunked HTTP request to audiocpp_server's live route; SSE events come back on the same socket."""

    def __init__(self, host, port, model, sample_rate, live_id=None):
        self.host, self.port, self.model, self.sample_rate, self.live_id = host, port, model, sample_rate, live_id
        self.reader = None
        self.writer = None
        self.status = None

    async def open(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        q = f"model={self.model}&sample_rate={self.sample_rate}&channels=1&sample_format=s16le"
        if self.live_id:
            q += f"&live_id={self.live_id}"                     # addressable by the peek route
        head = (f"POST /v1/audio/transcriptions/live?{q} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
                "Transfer-Encoding: chunked\r\nContent-Type: application/octet-stream\r\nAccept: text/event-stream\r\n"
                "Connection: close\r\n\r\n")
        self.writer.write(head.encode())
        await self.writer.drain()

    async def send_pcm(self, pcm: bytes):
        if not pcm or self.writer is None or self.writer.is_closing():
            return
        self.writer.write(f"{len(pcm):x}\r\n".encode() + pcm + b"\r\n")
        await self.writer.drain()

    async def end_body(self):
        if self.writer is None or self.writer.is_closing():
            return
        try:
            self.writer.write(b"0\r\n\r\n")
            await self.writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass

    async def read_headers(self):
        line = await self.reader.readline()
        if not line:
            raise RuntimeError("audio.cpp closed the connection before answering")
        parts = line.decode(errors="replace").split()
        self.status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        headers = {}
        while True:
            line = await self.reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode(errors="replace").partition(":")
            headers[k.strip().lower()] = v.strip()
        return headers

    async def events(self, headers):
        """Yield parsed SSE `data:` JSON objects (the body may itself be chunk-encoded)."""
        chunked = headers.get("transfer-encoding", "").lower() == "chunked"
        buf = b""
        while True:
            if chunked:
                size_line = await self.reader.readline()
                if not size_line:
                    return
                try:
                    n = int(size_line.strip().split(b";")[0], 16)
                except ValueError:
                    return
                if n == 0:
                    return
                data = await self.reader.readexactly(n)
                await self.reader.readline()          # CRLF after the chunk
            else:
                data = await self.reader.read(65536)
                if not data:
                    return
            buf += data
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                for line in block.split(b"\n"):
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        log.warning("bad SSE payload: %r", payload[:200])

    async def close(self):
        if self.writer is not None:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass


MUTE_FRAMES = int(os.environ.get("ACPP_MUTE_FRAMES", "30"))        # no text for this many token frames (2.4 s)...
MUTE_SPEECH = float(os.environ.get("ACPP_MUTE_SPEECH", "0.5"))      # ...while speech was present in >= this share of them
MUTE_COOLDOWN = int(os.environ.get("ACPP_MUTE_COOLDOWN", "40"))     # audio frames between restarts (3.2 s: a restart that mutes again is retried after the next window)
MUTE_MAX_RESTARTS = int(os.environ.get("ACPP_MUTE_MAX_RESTARTS", "8"))
MUTE_RMS = float(os.environ.get("ACPP_MUTE_RMS", "0.01"))           # -40 dBFS: "speech" for the energy detector (no head)
TRACE = os.environ.get("ACPP_BRIDGE_TRACE") == "1"
HISTORY_FRAMES = int(os.environ.get("ACPP_RESTART_SEED_FRAMES", "12"))  # frames re-fed on a restart


class Segment:
    """One live request = one decoder life. `base` maps its frame 0 to the session's global audio frame."""

    def __init__(self, gen, base, live):
        self.gen, self.base, self.live = gen, base, live
        self.task = None
        self.done = asyncio.Event()


async def handler(ws):
    path = ws.request.path
    u = urlparse(path)
    if u.path != "/v1/transcribe":
        await ws.close(4004, "not found")
        return
    q = {k: v[-1] for k, v in parse_qs(u.query).items()}
    outq: asyncio.Queue = asyncio.Queue()

    def send(obj):
        outq.put_nowait(obj)

    async def writer():
        try:
            while True:
                obj = await outq.get()
                if obj is None:
                    return
                await ws.send(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass

    wtask = asyncio.create_task(writer())

    async def fail(code, msg, ws_code=4000):
        send({"type": "error", "code": code, "message": msg})
        send(None)
        await wtask
        await ws.close(ws_code, code)

    enc = q.get("encoding", "pcm_s16le")
    sr = int(q.get("sample_rate", "16000"))
    ch = int(q.get("channels", "1"))
    delay = int(q.get("delay_ms", str(ACPP_DELAY_MS)))
    turn_probs = q.get("turn_probs", "false").lower() in ("1", "true", "yes", "on")
    mute_detect = q.get("mute_restart", "true").lower() not in ("0", "false", "no", "off")
    if enc != "pcm_s16le" or ch != 1 or sr != 16000 or q.get("multichannel", "false").lower() in ("1", "true"):
        return await fail("invalid_config", "the audio.cpp bridge supports pcm_s16le mono 16 kHz only")
    if q.get("diarize", "false").lower() in ("1", "true") or q.get("turn_detection", "none") != "none":
        return await fail("invalid_config", "diarize / turn_detection are not available on the audio.cpp bridge")
    if delay != ACPP_DELAY_MS:
        return await fail("invalid_config", f"audio.cpp Voxtral realtime runs at a fixed delay of {ACPP_DELAY_MS} ms "
                                            f"(got delay_ms={delay})")
    if turn_probs and not HEAD_META:
        return await fail("invalid_config", "turn_probs requested but no turn head is configured (--head)")
    if ACTIVE:
        return await fail("session_full", "the audio.cpp model runs one live session at a time", 4029)
    n_delay = ACPP_DELAY_MS // FRAME_MS
    frame_bytes = 2 * (sr * FRAME_MS // 1000)
    ACTIVE.add(1)

    sid = f"acpp_{int(time.time() * 1000) % 10_000_000:07d}"
    t_start = time.monotonic()
    asm = WordAssembler()
    stats = {"steps": 0, "words": 0, "bytes_in": 0, "first_step_t": None, "restarts": 0}
    # session-global state (audio frames = 80 ms of PCM fed, token frames = the frame a token describes)
    st = {"audio_frames": 0, "last_word_end": -1, "last_turn_af": -1, "last_text_frame": 0, "cur_token_frame": -1,
          "last_restart_audio_frame": -10 ** 9, "hist": bytearray(), "framer": bytearray(), "t_last_write": 0.0}
    speech = []                        # per audio frame: 1.0 / 0.0 from the head (af) or PCM energy
    segs = []                          # Segment history; segs[-1] is live
    closed = asyncio.Event()

    def speech_share(n):
        w = speech[-n:]
        return sum(w) / len(w) if w else 0.0

    async def open_segment(base, seed: bytes):
        live = LiveRequest(ARGS.acpp_host, ARGS.acpp_port, ARGS.model, sr, live_id=f"{sid}_{len(segs)}")
        await live.open()
        seg = Segment(len(segs), base, live)
        segs.append(seg)
        seg.task = asyncio.create_task(read_segment(seg))
        if seed:
            await live.send_pcm(seed)
        return seg

    async def read_segment(seg):
        live = seg.live
        try:
            headers = await live.read_headers()
            if live.status != 200:
                body = b""
                try:
                    body = await asyncio.wait_for(live.reader.read(4096), 2)
                except Exception:
                    pass
                send({"type": "error", "code": "backend_error", "status": live.status,
                      "message": body.decode(errors="replace")[:500]})
                return
            async for ev in live.events(headers):
                if seg is not segs[-1]:
                    continue                                   # a restart superseded this decoder
                k = ev.get("type")
                if k == "transcript.steps":
                    words = []
                    for stp in ev["steps"]:
                        stats["steps"] += 1
                        if stats["first_step_t"] is None:
                            stats["first_step_t"] = time.monotonic() - t_start
                        fr = seg.base + stp["frame"]
                        st["cur_token_frame"] = fr
                        if TRACE:
                            # audio fed (ms) beyond what this token needs (frame fr+6 end + 40 ms block) and the age of the last write
                            need = (fr + 7) * FRAME_MS + 40
                            log.info("trace step frame=%d fed=%.0f over=%.0f last_write_age=%.1f ms mono=%.4f", fr, stats["bytes_in"] / (2 * sr) * 1000,
                                     stats["bytes_in"] / (2 * sr) * 1000 - need, (time.monotonic() - st["t_last_write"]) * 1000, time.monotonic())
                        if stp.get("text"):
                            st["last_text_frame"] = fr
                        if stp.get("turn"):
                            af = fr + n_delay
                            if turn_probs and af > st["last_turn_af"]:
                                st["last_turn_af"] = af
                                send({"type": "turn", "frame": af, "t": af * FRAME_MS, "token_frame": fr,
                                      "p": [round(x, 4) for x in stp["turn"]]})
                            if HEAD_META and af >= 0:
                                while len(speech) <= af:
                                    speech.append(0.0)
                                speech[af] = 1.0 if stp["turn"][0] >= 0.5 else 0.0
                        words.extend(asm.push(stp.get("text", ""), fr))
                    words = [w for w in words if w["ef"] > st["last_word_end"]]
                    if words:
                        st["last_word_end"] = words[-1]["ef"]
                        stats["words"] += len(words)
                        send({"type": "transcript", "text": "".join(w["raw"] for w in words),
                              "words": [{"text": w["text"], "start": w["sf"] * FRAME_MS, "end": w["ef"] * FRAME_MS}
                                        for w in words]})
                    if mute_detect:
                        await maybe_restart()
                elif k == "transcript.text.done":
                    words = [w for w in asm.flush() if w["ef"] > st["last_word_end"]]
                    if words:
                        send({"type": "transcript", "text": "".join(w["raw"] for w in words),
                              "words": [{"text": w["text"], "start": w["sf"] * FRAME_MS, "end": w["ef"] * FRAME_MS}
                                        for w in words]})
                    if closed.is_set():
                        send({"type": "session.ended", "session_id": sid, "restarts": stats["restarts"],
                              "timing": ev.get("timing")})
                elif k == "error":
                    send({"type": "error", "code": "backend_error", "message": ev.get("message") or json.dumps(ev)})
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            send({"type": "error", "code": "bridge", "message": repr(e)})
        finally:
            seg.done.set()

    async def maybe_restart():
        """The mute rule: the decoder produced no text for MUTE_FRAMES token
        frames although speech was present (head P(speaking), else PCM energy) in most of them -> fresh decoder on
        the last HISTORY_FRAMES frames of audio. The words still in the delay line are re-decoded from that seed."""
        cur = st["cur_token_frame"]
        if cur - st["last_text_frame"] < MUTE_FRAMES or stats["restarts"] >= MUTE_MAX_RESTARTS:
            return
        if st["audio_frames"] - st["last_restart_audio_frame"] < MUTE_COOLDOWN:
            return
        if speech_share(MUTE_FRAMES) < MUTE_SPEECH:
            return
        A = st["audio_frames"]
        # the last HISTORY_FRAMES complete frames plus the partial one, so the new decoder's frame grid is the session's
        seed = bytes(st["hist"][-(HISTORY_FRAMES * frame_bytes + len(st["framer"])):])
        h = min(HISTORY_FRAMES, (len(seed) - len(st["framer"])) // frame_bytes)
        old = segs[-1]
        asm.flush()                                            # anything half-assembled belongs to the dead decoder
        stats["restarts"] += 1
        st["last_restart_audio_frame"] = A
        st["last_text_frame"] = A                              # give the new decoder a full window
        t0 = time.monotonic()
        # the model lock is per request: end the old body first so the new request takes the lock at once
        # (audio arriving meanwhile queues in the new request's socket; the seed covers the delay line)
        await old.live.end_body()
        try:
            await open_segment(A - h, seed)
        except OSError as e:
            send({"type": "error", "code": "backend_unavailable", "message": f"restart failed: {e}"})
            return
        send({"type": "asr_restart", "n": stats["restarts"], "at_ms": A * FRAME_MS, "seed_frames": h,
              "silent_frames": cur - (A - MUTE_FRAMES), "speech_share": round(speech_share(MUTE_FRAMES), 2),
              "ms": round((time.monotonic() - t0) * 1000)})
        log.info("session %s mute restart #%d at %.1f s (no text since token frame %d, speech share %.2f)", sid,
                 stats["restarts"], A * FRAME_MS / 1000, cur, speech_share(MUTE_FRAMES))
        asyncio.get_running_loop().call_later(5, old.task.cancel)

    async def do_peek(m):
        """Peek: drain the delay line on a snapshot of the live decoder"""
        rid = m.get("id"); n = max(1, min(40, int(m.get("frames", n_delay + 2)))); seg = segs[-1]
        t0 = time.monotonic()
        ev = {"type": "transcript.peek", "text": "", "words": [], "n_frames": n, "frames_acked": st["audio_frames"]}
        if rid is not None: ev["id"] = rid
        try:
            res = await http_post_json(ARGS.acpp_host, ARGS.acpp_port,
                                       f"/v1/audio/transcriptions/live/peek?live_id={seg.live.live_id}&frames={n}")
        except Exception as e:
            ev["error"] = repr(e)[:200]; ev["ms"] = round((time.monotonic() - t0) * 1000, 1); send(ev); return
        if isinstance(res.get("error"), (str, dict)):
            ev["error"] = res["error"] if isinstance(res["error"], str) else res["error"].get("message", "error")
            ev["ms"] = round((time.monotonic() - t0) * 1000, 1); send(ev); return
        a = copy.deepcopy(asm); words = []; last_turn = None
        for stp in res.get("steps", []):
            words.extend(a.push(stp.get("text", ""), seg.base + stp["frame"]))
            if stp.get("turn"): last_turn = stp["turn"]
        words.extend(a.flush())
        words = [w for w in words if w["ef"] > st["last_word_end"]]
        ev.update(text="".join(w["raw"] for w in words),
                  words=[{"text": w["text"], "start": w["sf"] * FRAME_MS, "end": w["ef"] * FRAME_MS} for w in words],
                  ms=round(res.get("ms", (time.monotonic() - t0) * 1000), 1), rtt_ms=round((time.monotonic() - t0) * 1000, 1))
        if last_turn is not None: ev["turn"] = [round(x, 4) for x in last_turn]
        send(ev)

    def note_pcm(pcm: bytes):
        st["hist"] += pcm
        if len(st["hist"]) > 4 * (HISTORY_FRAMES + 1) * frame_bytes:
            del st["hist"][:-2 * (HISTORY_FRAMES + 1) * frame_bytes]
        st["framer"] += pcm
        while len(st["framer"]) >= frame_bytes:
            fr = bytes(st["framer"][:frame_bytes]); del st["framer"][:frame_bytes]
            af = st["audio_frames"]; st["audio_frames"] += 1
            if not HEAD_META:                                  # energy VAD stands in for the head
                import array
                a = array.array("h", fr)
                rms = (sum(x * x for x in a) / len(a)) ** 0.5 / 32768.0
                while len(speech) <= af:
                    speech.append(0.0)
                speech[af] = 1.0 if rms >= MUTE_RMS else 0.0

    try:
        await open_segment(0, b"")
    except OSError as e:
        return await fail("backend_unavailable", f"audio.cpp server {ARGS.acpp_host}:{ARGS.acpp_port}: {e}", 4003)
    created = {"type": "session.created", "session_id": sid, "model": "voxtral-realtime-audio.cpp", "delay_ms": delay,
               "audio": {"encoding": enc, "sample_rate": sr, "channels": 1, "multichannel": False},
               "turn_detection": "none", "backend": "audio.cpp", "acpp_model": ARGS.model, "mute_restart": mute_detect}
    if turn_probs:
        created["turn_head"] = {"classes": HEAD_META.get("classes"), "tap": HEAD_META.get("tap"), "file": ARGS.head}
    if q.get("client_id"):
        created["client_id"] = q["client_id"]
    send(created)
    log.info("session %s open (turn_probs=%s)", sid, turn_probs)
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                stats["bytes_in"] += len(msg)
                note_pcm(msg)
                await segs[-1].live.send_pcm(msg)
                st["t_last_write"] = time.monotonic()
            else:
                try:
                    m = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                t = m.get("type")
                if t == "input_audio.end":
                    closed.set()
                    # drain the delay line: the token for frame k only comes out after frame
                    # k + n_delay was consumed, so the last n_delay frames need silence behind them
                    await segs[-1].live.send_pcm(bytes(frame_bytes * (n_delay + 2)))
                    await segs[-1].live.end_body()
                    try:
                        await asyncio.wait_for(segs[-1].done.wait(), 15)
                    except asyncio.TimeoutError:
                        send({"type": "error", "code": "bridge", "message": "audio.cpp did not finish the stream"})
                    break
                elif t == "peek":
                    asyncio.create_task(do_peek(m))
                elif t == "session.update":
                    send({"type": "error", "code": "invalid_config", "message": "session.update is not supported here"})
                elif t == "keepalive":
                    pass
    except Exception as e:
        log.info("session %s ws error: %r", sid, e)
    finally:
        ACTIVE.clear()
        closed.set()
        for sg in segs:
            await sg.live.end_body()
        try:
            await asyncio.wait_for(segs[-1].done.wait(), 5)
        except asyncio.TimeoutError:
            pass
        for sg in segs:
            if sg.task:
                sg.task.cancel()
            await sg.live.close()
        send(None)
        await wtask
        log.info("session %s closed: %.1f s audio in, %d steps, %d words, %d restarts, first step after %s s", sid,
                 stats["bytes_in"] / (2 * sr), stats["steps"], stats["words"], stats["restarts"],
                 None if stats["first_step_t"] is None else round(stats["first_step_t"], 2))
        try:
            await ws.close()
        except Exception:
            pass


async def main():
    global ARGS, HEAD_META
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--acpp", default="127.0.0.1:8090", help="audiocpp_server host:port")
    ap.add_argument("--model", default="voxtral-rt", help="model id in the audiocpp_server config")
    ap.add_argument("--head", default=None, help="the exported head's .json sidecar (class names); enables turn_probs")
    ARGS = ap.parse_args()
    ARGS.acpp_host, _, p = ARGS.acpp.partition(":")
    ARGS.acpp_port = int(p or 8090)
    if ARGS.head:
        HEAD_META = json.load(open(ARGS.head))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    async with serve(handler, ARGS.host, ARGS.port, max_size=None, ping_interval=20):
        log.info("asr_bridge listening on ws://%s:%d/v1/transcribe -> audio.cpp %s model=%s head=%s (server_ready)",
                 ARGS.host, ARGS.port, ARGS.acpp, ARGS.model, ARGS.head)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
