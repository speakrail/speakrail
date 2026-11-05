"""claude_code: the live harness hands a spoken task to Claude Code on this machine (headless `claude -p`) and gets
the result back as an async tool result, in the shape the model was trained on for async tools: {task_id} at once, later
{answer, task_id} (or {error, task_id}) inserted into the conversation. task_status / cancel_task (training catalog) work on
these tasks. All tasks of one live session continue the same Claude Code session (--resume) until reset_chat.

Claude Code runs as the logged-in claude.ai account (the subscription): the API key and the parent Claude Code session's
variables are removed from its environment. Its final message is written to be read aloud (VOICE_SYS); the talker
summarises it again in one to three sentences.

Security: this is a shell on the machine. server.py declares the tool only when started with --claude, and only for a
websocket that presents the access key from SPEAKRAIL_ACCESS_KEY (?ck=...)."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import time

from ..config import secret

CLAUDE_TOOL = "claude_code"
CLAUDE_TOOL_SPEC = {
    "domain": "background_ai", "mode": "async", "args": {"task": "string", "context?": "string"},
    "arg_descriptions": {"task": "the question or what to do, in full: the user's request with every detail they gave (names, paths, numbers)",
                         "context": "anything relevant from the conversation that the request itself doesn't say"},
    "description": "Ask Claude a question or hand it a task. Claude is a powerful AI agent working on the user's computer: it "
                   "can read and change files, run commands and programs, check the user's projects, experiments and servers, "
                   "and search the web, and it remembers the earlier questions and tasks of this conversation. Use it whenever "
                   "the user asks you to do, check or find out something on the computer or in their files, or asks for Claude; "
                   "follow-up questions about its last answer go to it too. Async: returns a task_id at once; the result "
                   "arrives later, possibly while someone is talking. Report it in one to three sentences, say it came from "
                   "Claude, never read it verbatim, and don't add details it didn't give."}
VOICE_SYS = ("You were called by a voice assistant on behalf of its user, who is talking to it out loud; the user will hear a "
             "short summary of your final message. Do the task. Make your FINAL message 1-4 short plain sentences with the "
             "outcome: what you did, what you found, anything the user must decide. No markdown, no code blocks, no lists, "
             "no long paths or ids unless they are the answer. Don't ask questions back unless you cannot go on; if you need "
             "a decision, say so in the final message. Never delete or overwrite the user's data, push code, spend money, "
             "rent machines or stop running services unless the task explicitly asks for it.")
MAX_ANSWER_CHARS = 1500


def check_key(given: str | None) -> bool:
    key = secret("SPEAKRAIL_ACCESS_KEY").strip()
    return bool(key) and bool(given) and given.strip() == key


def _child_env():
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CLAUDE") or k in ("ANTHROPIC_API_KEY", "AI_AGENT"))}   # the subscription, not the API key
    return env


class ClaudeTasks:
    """background Claude Code runs of one live session; on_done(task_id, result) is called on the event loop"""

    def __init__(self, on_done, log_dir=None, model=None, mode="auto", cwd="~", timeout_s=900.0):
        self.on_done, self.log_dir, self.model, self.mode, self.cwd, self.timeout_s = on_done, log_dir, model, mode, os.path.expanduser(cwd), timeout_s
        self.bin = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
        self.tasks: dict[str, dict] = {}
        self.n = 0
        self.session_id: str | None = None          # the Claude Code session this live session continues

    def start(self, task: str, context: str = "", conversation: str = "") -> str:
        self.n += 1; tid = f"c{self.n}"
        prompt = task.strip()
        if context: prompt += "\n\nContext: " + context.strip()
        if conversation: prompt += "\n\nThe conversation so far (speech recognition, may contain errors):\n" + conversation
        self.tasks[tid] = {"task_id": tid, "kind": CLAUDE_TOOL, "status": "running", "t0": time.monotonic(), "task": task,
                           "proc": None, "aio": None}
        self.tasks[tid]["aio"] = asyncio.ensure_future(self._run(tid, prompt))
        return tid

    async def _run(self, tid, prompt):
        t = self.tasks[tid]
        args = [self.bin, "-p", prompt, "--output-format", "json", "--permission-mode", self.mode,
                "--append-system-prompt", VOICE_SYS]
        if self.model: args += ["--model", self.model]
        if self.session_id: args += ["--resume", self.session_id]
        result, rec = None, {"task_id": tid, "task": t["task"], "resume": self.session_id, "t_start": time.time()}
        try:
            proc = t["proc"] = await asyncio.create_subprocess_exec(
                *args, cwd=self.cwd, env=_child_env(), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=True)
            out, err = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            try:
                d = json.loads(out.decode("utf-8", "replace"))
            except Exception:
                d = {}
            if isinstance(d, list):                               # this CLI prints every message; the last "result" one counts
                d = next((x for x in reversed(d) if isinstance(x, dict) and x.get("type") == "result"), {})
            if not isinstance(d, dict): d = {}
            rec.update(rc=proc.returncode, cost_usd=d.get("total_cost_usd"), turns=d.get("num_turns"),
                       session_id=d.get("session_id"), stderr=err.decode("utf-8", "replace")[-500:])
            if d.get("session_id"): self.session_id = d["session_id"]
            text = str(d.get("result") or "").strip()
            if d and not d.get("is_error") and text:
                result = {"task_id": tid, "answer": text[:MAX_ANSWER_CHARS]}
            else:
                why = text or err.decode("utf-8", "replace").strip()[-300:] or f"exit code {proc.returncode}"
                result = {"task_id": tid, "error": why[:400]}
        except asyncio.TimeoutError:
            self._kill(t); result = {"task_id": tid, "error": "timed out"}
        except asyncio.CancelledError:
            self._kill(t); t["status"] = "cancelled"; rec.update(cancelled=True); self._log(rec); raise
        except Exception as e:                                      # noqa: BLE001 -- the conversation goes on
            result = {"task_id": tid, "error": f"{type(e).__name__}: {e}"}
        t["status"] = "failed" if "error" in result else "done"; t["elapsed_s"] = round(time.monotonic() - t["t0"], 1)
        rec.update(result=result, elapsed_s=t["elapsed_s"]); self._log(rec)
        self.on_done(tid, result)

    def _kill(self, t):
        p = t.get("proc")
        if p is not None and p.returncode is None:
            try: os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError): pass

    def _log(self, rec):
        if not self.log_dir: return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "claude_tasks.jsonl"), "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ---- the catalog's task_status / cancel_task
    def status(self, task_id: str | None = None) -> dict:
        ts = [t for t in self.tasks.values() if not task_id or t["task_id"] == task_id]
        if task_id and not ts: return {"error": "not found"}
        now = time.monotonic()
        return {"tasks": [{"task_id": t["task_id"], "kind": t["kind"], "status": t["status"],
                           "elapsed_s": t.get("elapsed_s", round(now - t["t0"]))} for t in ts]}

    def cancel(self, task_id: str) -> dict:
        t = self.tasks.get(str(task_id))
        if t is None: return {"error": "not found"}
        if t["status"] != "running": return {"error": "already done"}
        t["aio"].cancel(); t["status"] = "cancelled"
        return {"ok": True}

    def reset(self):
        """reset_chat: the next task starts a new Claude Code session (running tasks finish and still report)"""
        self.session_id = None

    def close(self):
        for t in self.tasks.values():
            if t["status"] == "running" and t["aio"]: t["aio"].cancel()
