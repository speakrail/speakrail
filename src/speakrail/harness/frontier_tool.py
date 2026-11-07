"""ask_frontier: a tool from the training data (catalog declaration used verbatim), backed by GLM-5.3 Flash on
Fireworks. Same async shape as in training: {task_id} at once ("f1", "f2", ...), later {answer, confidence, task_id} (or
{error, task_id}) inserted into the conversation; task_status / cancel_task cover these tasks too. The model sees the
recent conversation, as the catalog promises ("a larger frontier model with the full conversation context").

Paid: Fireworks serverless, key from FIREWORKS_API_KEY (never logged). Prices checked 2026-09-26: $0.15 / M input, $0.50 / M output; every task's tokens and cost go to
frontier_tasks.jsonl in the session folder."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

import aiohttp

from ..config import secret

FRONTIER_TOOL = "ask_frontier"
ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL = "accounts/fireworks/models/glm-5p3-flash"
PRICE_IN, PRICE_OUT = 0.15, 0.50                      # $ per million tokens (2026-09-26)
EFFORT = {"quick": "low", "standard": "medium", "deep": "high"}
MAX_TOKENS = {"quick": 4000, "standard": 8000, "deep": 16000}
TIMEOUT_S = {"quick": 90, "standard": 240, "deep": 600}
SYS = ("You are the expert behind a voice assistant. It hands you hard questions from a live spoken conversation (speech "
       "recognition, so words can be wrong) and reads a short summary of your answer aloud. Think the question through, "
       "then answer in plain prose for the ear: the conclusion first, then the key reasons or numbers, no markdown, no "
       "lists, at most 150 words. Say what the answer depends on when it matters. End with one line: "
       "'Confidence: high', 'Confidence: medium' or 'Confidence: low'.")
MAX_ANSWER_CHARS = 1500


def _key():
    return secret("FIREWORKS_API_KEY")


class FrontierTasks:
    """background ask_frontier calls of one live session; on_done(task_id, result) is called on the event loop"""

    def __init__(self, on_done, log_dir=None, model=MODEL):
        self.on_done, self.log_dir, self.model = on_done, log_dir, model
        self.tasks: dict[str, dict] = {}
        self.n = 0

    def start(self, question: str, context: str = "", depth: str = "standard", conversation: str = "") -> str:
        self.n += 1; tid = f"f{self.n}"
        depth = depth if depth in EFFORT else "standard"
        user = "Question: " + question.strip()
        if context: user += "\n\nContext: " + context.strip()
        if conversation: user = "The conversation so far:\n" + conversation + "\n\n" + user
        self.tasks[tid] = {"task_id": tid, "kind": FRONTIER_TOOL, "status": "running", "t0": time.monotonic(),
                           "question": question, "depth": depth, "aio": None}
        self.tasks[tid]["aio"] = asyncio.ensure_future(self._run(tid, user, depth))
        return tid

    async def _run(self, tid, user, depth):
        t = self.tasks[tid]
        body = {"model": self.model, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                "max_tokens": MAX_TOKENS[depth], "temperature": 0.6, "top_p": 0.95, "reasoning_effort": EFFORT[depth]}
        rec = {"task_id": tid, "question": t["question"], "depth": depth, "model": self.model, "t_start": time.time()}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT_S[depth])) as http:
                for attempt in range(3):
                    async with http.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {_key()}"}) as r:
                        txt = await r.text()
                        if r.status == 200: break
                        if r.status == 400 and "reasoning" in txt.lower() and "reasoning_effort" in body:
                            body.pop("reasoning_effort"); continue            # a model without the knob
                        if r.status not in (429, 500, 502, 503, 504) or attempt == 2:
                            raise RuntimeError(f"fireworks {r.status}: {txt[:200]}")
                    await asyncio.sleep(3 * (attempt + 1))
            d = json.loads(txt); msg = d["choices"][0]["message"]; u = d.get("usage") or {}
            text = (msg.get("content") or "").strip()
            m = re.search(r"\n?\s*confidence:\s*(high|medium|low)\W*$", text, re.I)
            conf = m.group(1).lower() if m else None
            if m: text = text[:m.start()].strip()
            rec.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                       usd=round(((u.get("prompt_tokens") or 0) * PRICE_IN + (u.get("completion_tokens") or 0) * PRICE_OUT) / 1e6, 5),
                       finish=d["choices"][0].get("finish_reason"))
            result = ({"task_id": tid, "answer": text[:MAX_ANSWER_CHARS], **({"confidence": conf} if conf else {})} if text
                      else {"task_id": tid, "error": "no answer (" + str(d["choices"][0].get("finish_reason")) + ")"})
        except asyncio.CancelledError:
            t["status"] = "cancelled"; rec.update(cancelled=True); self._log(rec); raise
        except asyncio.TimeoutError:
            result = {"task_id": tid, "error": "timeout"}
        except Exception as e:                                      # noqa: BLE001 -- the conversation goes on
            result = {"task_id": tid, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        t["status"] = "failed" if "error" in result else "done"; t["elapsed_s"] = round(time.monotonic() - t["t0"], 1)
        rec.update(result=result, elapsed_s=t["elapsed_s"]); self._log(rec)
        self.on_done(tid, result)

    def _log(self, rec):
        if not self.log_dir: return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "frontier_tasks.jsonl"), "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def status_list(self):
        now = time.monotonic()
        return [{"task_id": t["task_id"], "kind": t["kind"], "status": t["status"],
                 "elapsed_s": t.get("elapsed_s", round(now - t["t0"]))} for t in self.tasks.values()]

    def cancel(self, task_id: str) -> dict:
        t = self.tasks.get(str(task_id))
        if t is None: return {"error": "not found"}
        if t["status"] != "running": return {"error": "already done"}
        t["aio"].cancel(); t["status"] = "cancelled"
        return {"ok": True}

    def close(self):
        for t in self.tasks.values():
            if t["status"] == "running" and t["aio"]: t["aio"].cancel()
