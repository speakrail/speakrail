#!/usr/bin/env python3
"""The reference LLM engine. It has no model: it answers by rule, but speaks the whole LLM contract.

  POST /v1/completions       token-id prompts. allowed_token_ids + logprobs for decisions, SSE streams for replies.
  POST /v1/chat/completions  non-streamed chat (the listening notes)
  GET  /health

--flags breaks one rule on purpose, which is how the tests prove that the conformance checks catch it.
"""
import argparse
import asyncio
import json

from aiohttp import web

SPEAK, COMPLETE, LISTEN, YIELD = 106, 6, 16, 20
REPLY = "Sure, this is a reply from the stub engine".split()
TOKENS = [1000 + i for i in range(len(REPLY))]
MODELS = ("speakrail", "speakrail-base")
MAX_CONTEXT = 16384
FLAGS = set()
LAST_PROMPT = {}


def bad(message):
    return web.json_response({"error": {"message": message}}, status=400)


def decide(prompt, allowed):
    if SPEAK in allowed and COMPLETE in prompt[-8:]:
        return SPEAK
    if YIELD in allowed:
        return YIELD
    return LISTEN if LISTEN in allowed else allowed[0]


def logprobs(chosen, model, candidates):
    spread = 6.0 if model == "speakrail" else 5.0     # the adapter changes the numbers, as a real LoRA does
    top = {f"token_id:{t}": (-0.01 if t == chosen else -spread) for t in sorted(set(candidates) | {chosen})}
    reported = "None" if "none_on_eos" in FLAGS and chosen == SPEAK else chosen
    return {"tokens": [f"token_id:{reported}"], "token_logprobs": [top[f"token_id:{chosen}"]], "top_logprobs": [top]}


async def prefill(model, prompt):
    """a request costs time for what isn't already in the prefix cache"""
    last, n = LAST_PROMPT.get(model, []), 0
    while n < min(len(last), len(prompt)) and last[n] == prompt[n]:
        n += 1
    if "no_prefix_cache" in FLAGS:
        n = 0
    LAST_PROMPT[model] = prompt
    await asyncio.sleep(0.001 * (len(prompt) - n))


async def completions(request):
    b = await request.json()
    prompt, model = b.get("prompt"), b.get("model")
    if not (isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt)):
        return bad("prompt must be a list of token ids")
    if model not in MODELS:
        return bad(f"unknown model {model!r}")
    max_tokens = int(b.get("max_tokens", 16))
    if len(prompt) + max_tokens > MAX_CONTEXT:
        return bad("context too long")
    await prefill(model, prompt)
    allowed, stops = b.get("allowed_token_ids") or [], set(b.get("stop_token_ids") or [])
    banned = {int(k) for k, v in (b.get("logit_bias") or {}).items() if v <= -100}
    if allowed:
        choice = decide(prompt, allowed)
        if "ignore_allowed" in FLAGS:
            choice = 999
        out = {"text": "", "index": 0, "finish_reason": "stop" if choice in stops else "length", "stop_reason": None}
        if b.get("logprobs"):
            out["logprobs"] = logprobs(choice, model, allowed)
        return web.json_response({"choices": [out], "usage": {"prompt_tokens": len(prompt), "completion_tokens": 1}})
    tokens = [t for t in TOKENS if t not in banned or "ignore_bias" in FLAGS]
    if b.get("stream"):
        return await stream(request, b, prompt, tokens, stops, max_tokens)
    produced, reason, stop_reason = [], "length", None
    for t in tokens[:max_tokens]:
        if t in stops:
            reason, stop_reason = "stop", t
            break
        produced.append(t)
    else:
        if len(tokens) < max_tokens:
            reason, stop_reason = "stop", SPEAK if SPEAK in stops else None
    out = {"text": " ".join(REPLY[TOKENS.index(t)] for t in produced), "index": 0, "finish_reason": reason, "stop_reason": stop_reason}
    if b.get("logprobs") and tokens:
        out["logprobs"] = logprobs(tokens[0], model, tokens[:3])
    return web.json_response({"choices": [out], "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(produced)}})


async def stream(request, b, prompt, tokens, stops, max_tokens):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    send = lambda o: resp.write(f"data: {json.dumps(o)}\n\n".encode())
    n, reason, stop_reason = 0, "length", None
    for i, t in enumerate(tokens[:max_tokens]):
        if t in stops:
            reason, stop_reason = "stop", t
            break
        await asyncio.sleep(0.02)
        await send({"choices": [{"text": ("" if n == 0 else " ") + REPLY[TOKENS.index(t)], "index": 0, "finish_reason": None}]})
        n += 1
    else:
        if len(tokens) < max_tokens:
            reason, stop_reason = "stop", SPEAK if SPEAK in stops else None
    await send({"choices": [{"text": "", "index": 0, "finish_reason": reason, "stop_reason": stop_reason}]})
    if (b.get("stream_options") or {}).get("include_usage"):
        await send({"choices": [], "usage": {"prompt_tokens": len(prompt), "completion_tokens": n}})
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def chat(request):
    b = await request.json()
    if b.get("model") not in MODELS:
        return bad(f"unknown model {b.get('model')!r}")
    await asyncio.sleep(0.1)
    note = "STATE: the user is talking to the stub engine."
    return web.json_response({"choices": [{"index": 0, "message": {"role": "assistant", "content": note}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 10, "completion_tokens": 10}})


async def health(_):
    return web.json_response({"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--flags", default="")
    a = ap.parse_args()
    FLAGS.update(f for f in a.flags.split(",") if f)
    app = web.Application(client_max_size=1 << 26)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/health", health)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
