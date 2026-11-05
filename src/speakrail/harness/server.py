#!/usr/bin/env python3
"""Speakrail web server (aiohttp).

    /                the voice UI (web/live)
    /debug/          the measurement UI (web): head meters, decisions, per-turn latencies
    /ws, /debug/ws   one voice session per websocket
                     binary in  = int16 mono 16 kHz mic audio
                     binary out = 2-byte utterance id, 2 pad bytes, int16 24 kHz PCM
                     JSON in:  {cmd: play_start, utt, delay_ms} | {cmd: interrupt} | {cmd: reconnect_asr} | {cmd: stop}
                     JSON out: the session's events (word, endpoint, reply_delta, turn, ...)

`--stub` serves the same UI and protocol with no models at all (stub.py), to try the interface.
"""
import argparse
import asyncio
import json
import os
import time

from aiohttp import web, WSMsgType

from ..config import CFG, secret

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = None
ARGS = None


def page(path):
    async def handler(_):
        return web.FileResponse(os.path.join(HERE, "web", path))
    return handler


def session_cfg(q):
    from .cascade import Cfg
    barge = {"0": "off", "1": "word"}.get(q.get("barge", ARGS.barge), q.get("barge", ARGS.barge))
    return Cfg(asr_url=CFG.asr_url, barge_in=barge,
               voice=ARGS.voice, silence_ms=int(q.get("silence_ms", ARGS.silence_ms)),
               search=search_for(q), spec=CFG.spec, peek_frames=CFG.peek_frames)   # SPEC=0: no speculation (one peek at the endpoint)


def search_for(q):
    """the server's SEARCH decides which backend exists; the page can only turn it off for its session"""
    if ARGS.search == "off" or q.get("search") == "off":
        return "off"
    return ARGS.search


def micro_cfg(q, cfg):
    from .claude_tool import check_key
    from .session import MicroCfg
    key_ok = check_key(q.get("ck"))
    # end-of-turn rule: P(fire) >= TURN_TAU on TURN_K frames in a row (0.9 x 3; the all_ant80 head was benchmarked with K=1)
    return MicroCfg(url=ARGS.mt_url, lora=CFG.llm_lora,
                    tau=CFG.turn_tau, k=CFG.turn_k,
                    think_model=CFG.llm_base_model, undo_words=ARGS.undo_words, tool_hold_ms=ARGS.tool_hold_ms,
                    persona=ARGS.persona,
                    allow_interject=not ARGS.no_interject, silence_fallback_ms=cfg.silence_ms,
                    notes=ARGS.notes, phrase_cache=not ARGS.no_phrase_cache, safety_yield_s=ARGS.safety_yield_s,
                    claude=bool(ARGS.claude and key_ok), claude_model=ARGS.claude_model or None,
                    claude_mode=ARGS.claude_mode, claude_cwd=os.path.expanduser(ARGS.claude_cwd),
                    frontier=ARGS.frontier)               # paid, on the user's own Fireworks key; no shell access


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=0, heartbeat=20)
    await ws.prepare(request)
    q = request.query
    out: asyncio.Queue = asyncio.Queue()

    def send(obj):
        out.put_nowait(obj)

    async def writer():
        try:
            while True:
                obj = await out.get()
                if obj is None:
                    return
                if isinstance(obj, bytes):
                    await ws.send_bytes(obj)
                else:
                    await ws.send_str(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass

    wtask = asyncio.create_task(writer())
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if ARGS.stub:
        from types import SimpleNamespace
        from .stub import StubSession
        sess = StubSession(SimpleNamespace(delay_ms=480, barge_in=q.get("barge", ARGS.barge)), send)
    else:
        from .session import MicroSession
        cfg = session_cfg(q)
        log_dir = os.path.join(ARGS.sessions_dir, f"session-{stamp}") if ARGS.sessions_dir else None
        sess = MicroSession(cfg, HUB, send, log_dir=log_dir, mcfg=micro_cfg(q, cfg))
    try:
        await sess.start()
    except Exception as e:
        send({"type": "error", "where": "asr connect", "detail": repr(e)})
        send(None)
        await wtask
        await ws.close()
        return ws
    print(f"[ws] session {stamp} started{' (stub)' if ARGS.stub else ''}", flush=True)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                sess.feed_audio(msg.data)
            elif msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                c = m.get("cmd")
                if c == "play_start":
                    sess.note_play_start(int(m["utt"]), float(m.get("delay_ms", 0)))
                elif c == "interrupt":
                    sess.counters["barge_ins"] += 1
                    sess._cancel_reply("interrupt button")
                elif c == "reconnect_asr":
                    try:
                        await sess.reconnect_asr()
                    except Exception as e:
                        send({"type": "error", "where": "asr reconnect", "detail": repr(e)})
                elif c == "stop":
                    break
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
    finally:
        await sess.close()
        send(None)
        await wtask
        summ = sess.summary()
        print(f"[ws] session {stamp} closed: {summ['turns']} turns, word_end_to_ear p50 {summ['word_end_to_ear']['p50']} ms", flush=True)
        if not ws.closed:
            await ws.close()
    return ws


async def on_startup(app):
    global HUB
    if not ARGS.stub:
        from .cascade import TTSHub
        HUB = TTSHub(asyncio.get_running_loop(), ARGS.voice, ARGS.tts_engine)
        await HUB.wait_ready()
    print(f"speakrail{' (stub mode, no models)' if ARGS.stub else ''}: open http://{ARGS.host}:{ARGS.port}/", flush=True)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="no models: fake ASR / turn head / LLM / TTS, to try the UI")
    ap.add_argument("--port", type=int, default=CFG.ui_port)
    ap.add_argument("--host", default=CFG.ui_host)
    ap.add_argument("--tts-engine", default=CFG.tts_engine, choices=["breeze", "pocket"],
                    help="TTS backend at TTS_URL: breeze (Breeze TTS 2 API) or pocket (Pocket TTS on audiocpp_server, model TTS_MODEL)")
    ap.add_argument("--voice", default=CFG.voice,
                    help="breeze: reference wav, its transcript next to it as .txt; pocket: a voice name (alba, ...) or a wav to clone")
    ap.add_argument("--persona", default=CFG.persona, choices=["nova"])
    ap.add_argument("--sessions-dir", default=CFG.sessions_dir,
                    help="save each session's event log, context and audio recording here ('' = off, the default)")
    # turn taking
    ap.add_argument("--silence-ms", type=int, default=CFG.silence_ms,
                    help="silence fallback in ms after words with no head fire (0 = only the turn head ends a turn)")
    ap.add_argument("--barge", default="duck", choices=["off", "duck", "word", "head"],
                    help="user speaks over a reply: duck = lower our volume, the model decides")
    ap.add_argument("--safety-yield-s", type=float, default=1.5, help="yield if the model keeps listening through this much speech (0 = off)")
    ap.add_argument("--undo-words", type=int, default=2, help="undo a reply if the user goes on while <= this many of its words played")
    # the model
    ap.add_argument("--mt-url", default=CFG.llm_url)
    ap.add_argument("--notes", action="store_true", default=CFG.notes,
                    help="listening notes: the base model takes notes while the user speaks")
    ap.add_argument("--no-notes", dest="notes", action="store_false")
    ap.add_argument("--no-interject", action="store_true", default=not CFG.interject,
                    help="never offer <interject> (short remarks while the user keeps talking)")
    ap.add_argument("--no-phrase-cache", action="store_true", help="never play a first clause from pre-rendered clips")
    # tools
    ap.add_argument("--tool-hold-ms", type=int, default=CFG.tool_hold_ms, help="run tool calls only after this much quiet (0 = off)")
    ap.add_argument("--search", default=CFG.search, choices=["searxng", "serper", "brave", "off"],
                    help="web search backend for the web_search tool (off by default)")
    ap.add_argument("--frontier", action="store_true", default=bool(secret("FIREWORKS_API_KEY")),
                    help="offer ask_frontier (a larger model on Fireworks, paid; on when FIREWORKS_API_KEY is set)")
    ap.add_argument("--claude", action="store_true", help="offer claude_code (Claude Code on this machine: a shell!) to sessions "
                    "whose page URL carries ?ck=<SPEAKRAIL_ACCESS_KEY>")
    ap.add_argument("--claude-model", default="", help="claude -p --model (empty: the CLI default)")
    ap.add_argument("--claude-mode", default="auto", help="claude -p --permission-mode")
    ap.add_argument("--claude-cwd", default="~")
    ARGS = ap.parse_args()
    if not ARGS.stub and not ARGS.voice and ARGS.tts_engine == "breeze":
        ap.error("--voice (or VOICE) is required for Breeze unless --stub")
    app = web.Application()
    app.on_startup.append(on_startup)
    app.router.add_get("/", page("live/index.html"))
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", os.path.join(HERE, "web", "live"))
    # the same UI under /live/, for reverse proxies that mount it there
    app.router.add_get("/live/", page("live/index.html"))
    app.router.add_get("/live/ws", ws_handler)
    app.router.add_static("/live/static", os.path.join(HERE, "web", "live"))
    app.router.add_get("/debug", lambda r: web.HTTPFound("/debug/"))
    app.router.add_get("/debug/", page("index.html"))
    app.router.add_get("/debug/ws", ws_handler)
    app.router.add_static("/debug/static", os.path.join(HERE, "web"))
    web.run_app(app, host=ARGS.host, port=ARGS.port, print=None)


if __name__ == "__main__":
    main()
