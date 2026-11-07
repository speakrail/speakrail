#!/usr/bin/env python3
"""Speakrail LLM server on native MLX: the HTTP surface the harness uses from vLLM, one process, one inference thread.

  POST /v1/completions       token-id prompts; allowed_token_ids + raw top-20 logprobs (decisions); SSE streams (replies)
  POST /v1/chat/completions  non-streamed chat for the listening notes (base model)
  GET  /health /v1/models /metrics

Design
  * One worker thread owns MLX. Work is cut into atomic steps (a prefill slice or one decoded token), and a scheduler picks
    the next step: decisions first, then replies, then notes. A decision therefore waits for at most one step.
  * The adapter is a flag read at graph-build time, so base and LoRA steps interleave freely on one set of weights.
  * KV: every layer keeps its full history (sliding layers hand attention only their last window), so a cache can be rewound
    to any point in O(1); mlx_lm's rotating cache cannot rewind past its window. Entries are separate per adapter and cache_salt.
    A request takes the entry with the longest common prefix: extended in place when it is a pure extension or only a short
    tail would be lost, cloned otherwise (a busy entry, or a long tail such as a speculative reply that may still be adopted).
  * Fast prefill (--fast-prefill, on by default): the KV-shared layers own no KV, so a token only needs them when its logits
    are read. They run on the last token of the prompt only. Same values, fewer layers.
  * The per-layer embedding table (5.25 GiB for E4B) stays out of MLX: rows are gathered from a read-only memory map of
    the checkpoint on the CPU (--ple-mmap, on by default). It saved 5 GiB and made nothing slower.
"""
import argparse
import asyncio
import json
import logging
import os
import struct
import threading
import time
import traceback
from collections import defaultdict, deque

import numpy as np

log = logging.getLogger("mlx-llm")
DECIDE, REPLY, WARM, NOTES = 0, 1, 2, 3
KIND = {DECIDE: "decide", REPLY: "reply", WARM: "warm", NOTES: "notes"}


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--base-name", default="speakrail-base")
    p.add_argument("--lora-name", default="speakrail")
    p.add_argument("--max-ctx", type=int, default=16384)
    p.add_argument("--slice", type=int, default=64, help="prefill tokens per step while other work is waiting")
    p.add_argument("--slice-alone", type=int, default=512, help="prefill tokens per step when nothing else is waiting")
    p.add_argument("--max-entries", type=int, default=4, help="KV entries kept per adapter")
    p.add_argument("--max-kv-gb", type=float, default=2.5, help="KV bytes kept over all entries")
    p.add_argument("--clone-tail", type=int, default=16, help="clone instead of rewinding when more than this many tokens would be lost")
    p.add_argument("--fast-prefill", type=int, default=1)
    p.add_argument("--ple-mmap", type=int, default=1, help="serve the per-layer embedding table from a memory map")
    p.add_argument("--memory-limit-gb", type=float, default=13.0)
    p.add_argument("--wired-limit-gb", type=float, default=11.0)
    p.add_argument("--cache-limit-gb", type=float, default=1.0)
    p.add_argument("--notes-share", type=int, default=3, help="while replies are decoding, every Nth step goes to the notes")
    p.add_argument("--request-log", default="")
    return p.parse_args()


ARGS = build_args()
import mlx.core as mx                                        # noqa: E402
import mlx.nn as nn                                          # noqa: E402
from mlx_lm import load                                      # noqa: E402
from mlx_lm.models.cache import KVCache                      # noqa: E402
from mlx_lm.models.base import create_causal_mask            # noqa: E402
from mlx_lm.models import gemma4_text                        # noqa: E402
from mlx_lm.models.gemma4_text import logit_softcap          # noqa: E402
from mlx_lm.tuner.lora import LoRALinear                     # noqa: E402
from aiohttp import web                                      # noqa: E402


class MappedEmbedding:
    """stands in for nn.Embedding(vocab, dim): gathers the rows of the current tokens from a memory map of the bf16 table"""
    KEY = "embed_tokens_per_layer.weight"

    def __init__(self, model_dir):
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        name = next(k for k in index if k.endswith(self.KEY))
        path = os.path.join(model_dir, index[name])
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            info = json.loads(f.read(n))[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, found {info['dtype']}")
        self.table = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + info["data_offsets"][0], shape=tuple(info["shape"]))

    def __call__(self, ids):
        idx = np.asarray(ids, dtype=np.int64)
        rows = self.table[idx.reshape(-1)]
        return mx.array(rows).view(mx.bfloat16).reshape(*idx.shape, rows.shape[-1])


def map_per_layer_embeddings(model_dir):
    """keep the table out of MLX: the parameter is never created, and its weight is never read from disk"""
    table = MappedEmbedding(model_dir)
    TextModel, Model = gemma4_text.Gemma4TextModel, gemma4_text.Model
    init, sanitize = TextModel.__init__, Model.sanitize

    def new_init(self, config):
        init(self, config)
        if self.hidden_size_per_layer_input:
            self.pop("embed_tokens_per_layer")
            object.__setattr__(self, "embed_tokens_per_layer", table)

    def new_sanitize(self, weights):
        return {k: v for k, v in sanitize(self, weights).items() if not k.endswith(MappedEmbedding.KEY)}

    TextModel.__init__, Model.sanitize = new_init, new_sanitize


class ToggleLoRA(LoRALinear):
    """the adapter is applied only while ToggleLoRA.on; read when the graph is built, so steps can alternate freely"""
    on = True

    def __call__(self, x):
        y = self.linear(x)
        if not ToggleLoRA.on:
            return y
        return y + (self.scale * ((x @ self.lora_a) @ self.lora_b)).astype(x.dtype)


class WindowKV(KVCache):
    """a sliding-attention layer's KV: the whole history is kept (rewindable), attention sees the last `window` per token"""

    def __init__(self, window):
        super().__init__()
        self.window = window

    def update_and_fetch(self, keys, values):
        n, prev = keys.shape[2], self.offset
        k, v = super().update_and_fetch(keys, values)
        start = max(0, prev - (self.window - 1))
        return (k[..., start:, :], v[..., start:, :]) if start else (k, v)

    def make_mask(self, N, window_size=None, return_array=False):
        if N == 1:
            return None
        return create_causal_mask(N, min(self.window - 1, self.offset), window_size=self.window)


def make_cache(model):
    a = model.args
    first_shared = a.num_hidden_layers - a.num_kv_shared_layers
    return [KVCache() if a.layer_types[i] == "full_attention" else WindowKV(a.sliding_window) for i in range(first_shared)]


def clone_cache(cache):
    out = []
    for c in cache:
        d = WindowKV(c.window) if isinstance(c, WindowKV) else KVCache()
        d.offset = c.offset
        if c.keys is not None:
            d.keys, d.values = mx.array(c.keys), mx.array(c.values)
        out.append(d)
    return out


def rewind(cache, n):
    for c in cache:
        c.trim(n)


def cache_arrays(cache):
    return [a for c in cache if c.keys is not None for a in (c.keys, c.values)]


def forward(model, ids, cache, need_logits, fast):
    """-> logits of the last token (lazy) or None. fast: the KV-shared layers run on the last token only."""
    inputs = mx.array([ids])
    if not fast:
        return model(inputs, cache=cache)[0, -1] if need_logits else _kv_only(model, inputs, cache)
    return _fast(model, inputs, cache, need_logits)


def _run_kv_layers(m, inputs, cache):
    """embed and run the layers that own KV. -> (h, per_layer_inputs, intermediates)"""
    emb = m.embed_tokens(inputs)
    h = emb * m.embed_scale
    per = None
    if m.hidden_size_per_layer_input:
        per = m._project_per_layer_inputs(h, m._get_per_layer_inputs(inputs, emb))
    full_cache = cache + [None] * (len(m.layers) - len(cache))
    masks = m._make_masks(h, full_cache)
    inter = [(None, None)] * len(m.layers)
    for i in range(len(cache)):
        kvs, offset = inter[m.previous_kvs[i]]
        h, kvs, offset = m.layers[i](h, masks[i], cache[i], per_layer_input=per[:, :, i, :] if per is not None else None,
                                     shared_kv=kvs, offset=offset)
        inter[i] = (kvs, offset)
    return h, per, inter


def _kv_only(model, inputs, cache):
    _run_kv_layers(model.model, inputs, cache)
    return None


def _fast(model, inputs, cache, need_logits):
    m = model.model
    h, per, inter = _run_kv_layers(m, inputs, cache)
    if not need_logits:
        return None
    n = inputs.shape[1]
    h = h[:, -1:, :]
    for i in range(len(cache), len(m.layers)):
        src = m.previous_kvs[i]
        kvs, offset = inter[src]
        layer = m.layers[i]
        if src < len(cache):                                 # the last token's view of the KV-owning layer it shares
            keys, values = kvs
            if layer.self_attn.is_sliding and keys.shape[2] > m.window_size:
                keys, values = keys[..., -m.window_size:, :], values[..., -m.window_size:, :]
            kvs, offset = (keys, values), offset + (n - 1)
        h, kvs, offset = layer(h, None, None, per_layer_input=per[:, -1:, i, :] if per is not None else None,
                               shared_kv=kvs, offset=offset)
        inter[i] = (kvs, offset)
    out = m.embed_tokens.as_linear(m.norm(h)) if model.tie_word_embeddings else model.lm_head(m.norm(h))
    if model.final_logit_softcapping is not None:
        out = logit_softcap(model.final_logit_softcapping, out)
    return out[0, -1]


class Entry:
    __slots__ = ("lora", "salt", "tokens", "cache", "job", "used")

    def __init__(self, lora, salt, cache):
        self.lora, self.salt, self.tokens, self.cache, self.job, self.used = lora, salt, [], cache, None, time.monotonic()

    def nbytes(self):
        return sum(a.nbytes for a in cache_arrays(self.cache))


def lcp(a, b):
    n = min(len(a), len(b))
    if not n:
        return 0
    x, y = np.asarray(a[:n], dtype=np.int64), np.asarray(b[:n], dtype=np.int64)
    d = np.nonzero(x != y)[0]
    return int(d[0]) if d.size else n


class Job:
    def __init__(self, kind, lora, prompt, max_tokens, stops, allowed=None, bias=None, salt=None, ignore_eos=False,
                 want_logprobs=False, stream=False, skip_special=False, loop=None, out=None):
        self.kind, self.lora, self.prompt, self.max_tokens, self.stops = kind, lora, prompt, max_tokens, set(stops)
        self.allowed, self.bias, self.salt, self.ignore_eos = allowed, bias, salt, ignore_eos
        self.want_logprobs, self.stream, self.skip_special = want_logprobs, stream, skip_special
        self.loop, self.out = loop, out
        self.entry = None; self.pos = 0; self.reused = 0; self.cloned = False
        self.gen = []; self.logits = None; self.emitted = 0; self.prefix = 0; self.done = False
        self.cancelled = False; self.t_submit = time.monotonic(); self.t_start = None; self.t_first = None; self.steps = 0

    def post(self, item):
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)


class Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="mlx-worker")
        self.cv = threading.Condition()
        self.queues = {k: deque() for k in KIND}
        self.entries = []
        self.counter = 0
        self.stats = defaultdict(list)
        self.ready = threading.Event()
        self.req_log = open(ARGS.request_log, "a", buffering=1) if ARGS.request_log else None
        self.peak_active = 0
        self.fatal = None

    # ------------------------------------------------------------------ set-up (runs on the worker thread)
    def load_model(self):
        t0 = time.perf_counter()
        info = mx.device_info()
        mx.set_memory_limit(int(ARGS.memory_limit_gb * 2**30))
        mx.set_wired_limit(min(int(ARGS.wired_limit_gb * 2**30), info["max_recommended_working_set_size"]))
        mx.set_cache_limit(int(ARGS.cache_limit_gb * 2**30))
        if ARGS.ple_mmap:
            map_per_layer_embeddings(ARGS.model)
        outer, self.tok = load(ARGS.model)
        cfg = json.load(open(os.path.join(ARGS.adapter, "adapter_config.json")))
        lora = mx.load(os.path.join(ARGS.adapter, "adapter_model.safetensors"))
        mods = dict(outer.named_modules())
        attached, skipped = 0, 0
        for key in [k for k in lora if k.endswith("lora_A.weight")]:
            path = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
            parent, name = path.rsplit(".", 1)
            base = getattr(mods.get(parent), name, None) if parent in mods else None
            if base is None:
                skipped += 1
                continue
            out_dims, in_dims = base.weight.shape
            if isinstance(base, nn.QuantizedLinear):
                in_dims = in_dims * 32 // base.bits
            layer = ToggleLoRA(input_dims=in_dims, output_dims=out_dims, r=cfg["r"], scale=cfg["lora_alpha"] / cfg["r"])
            layer.linear = base
            layer.lora_a = lora[key].T.astype(mx.bfloat16)
            layer.lora_b = lora[key.replace("lora_A", "lora_B")].T.astype(mx.bfloat16)
            setattr(mods[parent], name, layer)
            attached += 1
        outer.eval()
        mx.eval(outer.parameters())
        self.model = outer.language_model                    # gemma4_text.Model: forward() drives its layers directly
        self.load_s = time.perf_counter() - t0
        self.eos = {1, 106, 50}                              # config.json eos_token_id
        print(f"[mlx-llm] loaded in {self.load_s:.1f} s; LoRA {attached} modules attached, {skipped} skipped (KV-shared layers); "
              f"active {mx.get_active_memory() / 2**30:.2f} GiB; fast_prefill={ARGS.fast_prefill}", flush=True)
        # warm-up: compile the kernels on a throwaway cache, with and without the adapter
        for on in (True, False):
            ToggleLoRA.on = on
            c = make_cache(self.model)
            mx.eval(forward(self.model, list(range(1000, 1064)), c, True, bool(ARGS.fast_prefill)), cache_arrays(c))
            l = forward(self.model, [1064], c, True, bool(ARGS.fast_prefill)); mx.eval(l, cache_arrays(c))
        mx.clear_cache()
        self.ready.set()

    # ------------------------------------------------------------------ scheduler
    def submit(self, job):
        with self.cv:
            self.queues[job.kind].append(job)
            self.cv.notify()

    def pick(self):
        """the next job to step: decisions first; then replies and warm-ups, with every Nth step for the notes"""
        self.counter += 1
        for q in self.queues.values():
            while q and q[0].cancelled:
                self.finish(q.popleft(), cancelled=True)
        d = self.queues[DECIDE]
        if d:
            return d
        r, w, n = self.queues[REPLY], self.queues[WARM], self.queues[NOTES]
        if n and (not r or self.counter % ARGS.notes_share == 0):
            return n
        if r:
            return r
        if w:
            return w
        return n if n else None

    def others_waiting(self, q):
        return any(x and x is not q for x in self.queues.values()) or len(q) > 1

    def run(self):
        try:
            self.load_model()
        except Exception:
            self.fatal = traceback.format_exc(); print("[mlx-llm] FATAL\n" + self.fatal, flush=True); os._exit(3)
        while True:
            with self.cv:
                q = self.pick()
                while q is None:
                    self.cv.wait()
                    q = self.pick()
            job = q[0]
            q.rotate(-1) if job.kind in (REPLY, NOTES) and len(q) > 1 else None   # round robin inside a class
            try:
                others = self.others_waiting(q)
                self.step(job, others)
            except Exception:
                tb = traceback.format_exc()
                print(f"[mlx-llm] ERROR in {KIND[job.kind]} job\n{tb}", flush=True)
                job.error = tb
                self.finish(job, error=tb)
            if job.done:
                with self.cv:
                    if job in q:
                        q.remove(job)

    # ------------------------------------------------------------------ KV entries
    def acquire(self, job):
        """find or make the KV entry for job.prompt; rewinds or clones. Leaves entry.tokens = what the KV holds."""
        limit = len(job.prompt) - 1                          # at least the last token runs: its logits are not stored
        best, best_n = None, -1
        for e in self.entries:
            if e.lora == job.lora and e.salt == job.salt:
                n = lcp(e.tokens, job.prompt)
                # prefer an idle entry on a tie
                if n > best_n or (n == best_n and best is not None and best.job is not None and e.job is None):
                    best, best_n = e, n
        best_n = min(best_n, limit)
        if best is None or best_n < 16:
            e = Entry(job.lora, job.salt, make_cache(self.model))
            self.entries.append(e)
            best_n = 0
        else:
            tail = len(best.tokens) - best_n
            if best.job is None and tail <= ARGS.clone_tail:
                e = best
                if tail:
                    rewind(e.cache, tail); del e.tokens[best_n:]
            else:
                c = clone_cache(best.cache)
                rewind(c, len(best.tokens) - best_n)
                mx.eval(cache_arrays(c))
                e = Entry(job.lora, job.salt, c)
                e.tokens = best.tokens[:best_n]
                self.entries.append(e)
                job.cloned = True
        e.job = job
        job.entry = e; job.pos = best_n; job.reused = best_n
        self.evict(keep=e)

    def evict(self, keep):
        idle = sorted((e for e in self.entries if e.job is None and e is not keep), key=lambda e: e.used)
        for lora in (True, False):
            mine = [e for e in idle if e.lora == lora]
            total = sum(1 for e in self.entries if e.lora == lora)
            while total > ARGS.max_entries and mine:
                self.entries.remove(mine.pop(0)); idle = [e for e in idle if e in self.entries]; total -= 1
        limit = ARGS.max_kv_gb * 2**30
        while sum(e.nbytes() for e in self.entries) > limit and idle:
            self.entries.remove(idle.pop(0))

    # ------------------------------------------------------------------ one step
    def step(self, job, others):
        job.steps += 1
        if job.cancelled:
            return self.finish(job, cancelled=True)
        if job.entry is None:
            job.t_start = time.monotonic()
            ToggleLoRA.on = job.lora
            self.acquire(job)
        e = job.entry
        ToggleLoRA.on = job.lora
        fast = bool(ARGS.fast_prefill)
        if job.pos < len(job.prompt):                        # prefill slice
            size = ARGS.slice if others else ARGS.slice_alone
            chunk = job.prompt[job.pos:job.pos + size]
            last = job.pos + len(chunk) == len(job.prompt)
            logits = forward(self.model, chunk, e.cache, last, fast)
            mx.eval(*([logits] if logits is not None else []), cache_arrays(e.cache))
            e.tokens.extend(chunk); job.pos += len(chunk); e.used = time.monotonic()
            if not last:
                return
            job.logits = logits
            if self.peak_active < mx.get_active_memory():
                self.peak_active = mx.get_active_memory()
            return self.emit(job)
        tok = job.gen[-1]                                    # one decoded token
        logits = forward(self.model, [tok], e.cache, True, fast)
        mx.eval(logits, cache_arrays(e.cache))
        e.tokens.append(tok); e.used = time.monotonic()
        job.logits = logits
        self.emit(job)

    def emit(self, job):
        """pick the next token from job.logits, send it, or end the job"""
        logits = job.logits.astype(mx.float32)
        raw = None
        if job.want_logprobs:
            raw = logits - mx.logsumexp(logits)
            top = mx.argpartition(-raw, 19)[:20]
        if job.bias or job.allowed:
            biased = logits
            if job.bias:
                idx, val = mx.array(list(job.bias.keys())), mx.array(list(job.bias.values()), dtype=mx.float32)
                biased = biased.at[idx].add(val)
            if job.allowed:
                a = mx.array(job.allowed)
                tok = int(job.allowed[int(mx.argmax(biased[a]).item())])
            else:
                tok = int(mx.argmax(biased).item())
        else:
            tok = int(mx.argmax(logits).item())
        if job.t_first is None:
            job.t_first = time.monotonic()
            job.first_tok = tok
        if job.want_logprobs:
            ids = np.array(top.tolist()); vals = np.array(raw[top].tolist(), dtype=np.float64)
            order = np.argsort(-vals)
            job.top = {f"token_id:{int(ids[i])}": float(vals[i]) for i in order}
            job.tok_lp = float(raw[tok].item())
        stop = (tok in job.stops or (not job.ignore_eos and tok in self.eos))
        if stop:
            job.stop_reason = tok if tok in job.stops else None
            job.finish_reason = "stop"
            if job.allowed:                                  # a decision's token is its answer, also when it is an eos id
                job.gen.append(tok)
            return self.finish(job)
        job.gen.append(tok)
        if job.stream:
            text = self.incremental(job)
            if text:
                job.post(("delta", text))
        if len(job.gen) >= job.max_tokens:
            job.stop_reason = None
            job.finish_reason = "length"
            self.finish(job)

    def incremental(self, job):
        s = self.tok.decode(job.gen[job.prefix:], skip_special_tokens=job.skip_special, clean_up_tokenization_spaces=False)
        if s.endswith("�"):                             # a character split over tokens: wait for the rest
            return ""
        new = s[job.emitted:]
        job.emitted = len(s)
        if len(job.gen) - job.prefix > 32:                   # keep decoding cost bounded
            job.prefix, job.emitted = len(job.gen), 0
        return new

    def finish(self, job, cancelled=False, error=None):
        if job.done:
            return
        job.done = True
        if job.entry is not None:
            job.entry.job = None
        now = time.monotonic()
        rec = {"t": time.time(), "kind": KIND[job.kind], "model": "lora" if job.lora else "base", "n_prompt": len(job.prompt),
               "reused": job.reused, "new": len(job.prompt) - job.reused, "n_gen": len(job.gen), "cloned": job.cloned,
               "wait_ms": round(((job.t_start or now) - job.t_submit) * 1000, 2),
               "ttft_ms": round((job.t_first - job.t_submit) * 1000, 2) if job.t_first else None,
               "total_ms": round((now - job.t_submit) * 1000, 2), "steps": job.steps, "cancelled": cancelled, "error": bool(error)}
        self.stats[rec["kind"]].append(rec["ttft_ms"] or 0)
        if self.req_log:
            self.req_log.write(json.dumps(rec) + "\n")
        if error:
            job.post(("error", error[-400:]))
        elif not cancelled:
            text = None
            if not job.stream:
                text = self.tok.decode(job.gen, skip_special_tokens=job.skip_special, clean_up_tokenization_spaces=False)
            job.post(("end", text))
        else:
            job.post(("cancelled", None))


W = Worker()


# ---------------------------------------------------------------------- HTTP
def model_is_lora(name):
    if name == ARGS.lora_name:
        return True
    if name == ARGS.base_name or name is None:
        return False
    raise web.HTTPBadRequest(text=json.dumps({"error": {"message": f"unknown model {name!r}"}}), content_type="application/json")


def bad(msg):
    raise web.HTTPBadRequest(text=json.dumps({"error": {"message": msg}}), content_type="application/json")


async def completions(request):
    if not W.ready.is_set():
        raise web.HTTPServiceUnavailable(text="loading")
    b = await request.json()
    prompt = b.get("prompt")
    if not (isinstance(prompt, list) and prompt and all(isinstance(x, int) for x in prompt)):
        bad("prompt must be a list of token ids")
    if len(prompt) + int(b.get("max_tokens", 16)) > ARGS.max_ctx:
        bad(f"context length {len(prompt)} + {b.get('max_tokens')} exceeds {ARGS.max_ctx}")
    if float(b.get("temperature", 0)) != 0:
        bad("only temperature 0 is implemented")
    lora = model_is_lora(b.get("model"))
    allowed = b.get("allowed_token_ids") or None
    bias = {int(k): float(v) for k, v in (b.get("logit_bias") or {}).items()} or None
    stream = bool(b.get("stream"))
    max_tokens = int(b.get("max_tokens", 16))
    kind = DECIDE if (allowed and max_tokens == 1) else (WARM if max_tokens == 1 else REPLY)
    loop = asyncio.get_running_loop()
    job = Job(kind, lora, prompt, max_tokens, b.get("stop_token_ids") or [], allowed=allowed, bias=bias, salt=b.get("cache_salt"),
              ignore_eos=bool(b.get("ignore_eos")), want_logprobs=bool(b.get("logprobs")), stream=stream,
              skip_special=bool(b.get("skip_special_tokens", True)), loop=loop, out=asyncio.Queue())
    W.submit(job)
    cid = f"cmpl-{id(job):x}"
    if not stream:
        try:
            kind_, payload = await job.out.get()
        except asyncio.CancelledError:
            job.cancelled = True
            with W.cv:
                W.cv.notify()
            raise
        if kind_ == "error":
            return web.json_response({"error": {"message": payload}}, status=500)
        choice = {"index": 0, "text": payload, "finish_reason": job.finish_reason, "stop_reason": getattr(job, "stop_reason", None)}
        if job.want_logprobs:
            choice["logprobs"] = {"tokens": [f"token_id:{job.first_tok}"], "token_logprobs": [job.tok_lp], "top_logprobs": [job.top]}
        return web.json_response({"id": cid, "object": "text_completion", "model": b.get("model"), "choices": [choice],
                                  "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(job.gen)}})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    sse = lambda o: ("data: " + json.dumps(o) + "\n\n").encode()
    try:
        while True:
            kind_, payload = await job.out.get()
            if kind_ == "delta":
                await resp.write(sse({"id": cid, "object": "text_completion", "choices": [{"index": 0, "text": payload, "finish_reason": None}]}))
            elif kind_ == "end":
                await resp.write(sse({"id": cid, "object": "text_completion", "choices": [{"index": 0, "text": "", "finish_reason": job.finish_reason,
                                                                                         "stop_reason": job.stop_reason}]}))
                if (b.get("stream_options") or {}).get("include_usage"):
                    await resp.write(sse({"id": cid, "object": "text_completion", "choices": [], "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(job.gen)}}))
                await resp.write(b"data: [DONE]\n\n")
                break
            elif kind_ == "error":
                await resp.write(sse({"error": {"message": payload}}))
                break
            else:
                break
    except ConnectionResetError:                             # the harness dropped the stream: a discarded reply
        job.cancelled = True
        with W.cv:
            W.cv.notify()
        return resp
    except asyncio.CancelledError:
        job.cancelled = True
        with W.cv:
            W.cv.notify()
        raise
    await resp.write_eof()
    return resp


async def chat(request):
    if not W.ready.is_set():
        raise web.HTTPServiceUnavailable(text="loading")
    b = await request.json()
    if float(b.get("temperature", 0)) != 0:
        bad("only temperature 0 is implemented")
    kw = b.get("chat_template_kwargs") or {}
    # content as text parts: the form vLLM hands the template, which keeps the training prompt byte for byte
    msgs = [{**m, "content": [{"type": "text", "text": m["content"]}]} if isinstance(m.get("content"), str) else m for m in b["messages"]]
    ids = W.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    lora = model_is_lora(b.get("model"))
    job = Job(NOTES, lora, list(ids), int(b.get("max_tokens", 256)), b.get("stop_token_ids") or [], salt=b.get("cache_salt"),
              skip_special=True, loop=asyncio.get_running_loop(), out=asyncio.Queue())
    W.submit(job)
    try:
        kind_, payload = await job.out.get()
    except asyncio.CancelledError:
        job.cancelled = True
        with W.cv:
            W.cv.notify()
        raise
    if kind_ == "error":
        return web.json_response({"error": {"message": payload}}, status=500)
    return web.json_response({"id": f"chat-{id(job):x}", "object": "chat.completion", "model": b.get("model"),
                              "choices": [{"index": 0, "message": {"role": "assistant", "content": payload},
                                           "finish_reason": job.finish_reason, "stop_reason": job.stop_reason}],
                              "usage": {"prompt_tokens": len(ids), "completion_tokens": len(job.gen)}})


async def health(request):
    return web.json_response({"ok": True, "ready": W.ready.is_set()}, status=200 if W.ready.is_set() else 503)


async def models(request):
    return web.json_response({"object": "list", "data": [{"id": ARGS.base_name, "object": "model"}, {"id": ARGS.lora_name, "object": "model"}]})


async def metrics(request):
    return web.json_response({"active_gib": mx.get_active_memory() / 2**30, "peak_gib": mx.get_peak_memory() / 2**30,
                              "cache_gib": mx.get_cache_memory() / 2**30, "peak_active_gib": W.peak_active / 2**30,
                              "entries": [{"lora": e.lora, "tokens": len(e.tokens), "mib": e.nbytes() / 2**20, "busy": e.job is not None} for e in W.entries],
                              "load_s": getattr(W, "load_s", None), "requests": {k: len(v) for k, v in W.stats.items()},
                              "fast_prefill": ARGS.fast_prefill})


def main():
    W.start()
    app = web.Application(client_max_size=1 << 28)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics)
    web.run_app(app, host=ARGS.host, port=ARGS.port, print=None, access_log=None, handle_signals=True, handler_cancellation=True)


if __name__ == "__main__":
    main()
