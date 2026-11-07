"""The harness's side of the host: a session asks for the engines here, and /v1/control/* is forwarded here.

GET /v1/control/status[?memory=1], POST /v1/control/load {"force": bool}, POST /v1/control/offload {"force": bool}.
A client sends the same Bearer token as the Live API. The web page, served from here, has no token and is let in as this
server's own page instead.
"""
from __future__ import annotations

import asyncio
import json

import aiohttp
from aiohttp import web

from ..config import CFG
from .live import Rejected, origin_allowed, token_ok

LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


class Gate:
    def __init__(self, url: str, token: str, warm=None):
        self.url, self.token, self.warm = url, token, warm
        self.http: aiohttp.ClientSession | None = None
        self.open: set = set()               # the websockets of the sessions that are in
        self.warm_generation = 1             # the engines this process started with were warmed when it did

    async def call(self, method: str, path: str, body: dict | None = None, timeout: float | None = 15.0) -> tuple[int, dict]:
        if self.http is None:
            self.http = aiohttp.ClientSession(headers={"Authorization": f"Bearer {self.token}"})
        try:
            async with self.http.request(method, self.url + path, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return r.status, await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            return 503, {"error": {"code": "host_unavailable", "message": f"can't reach the host: {type(e).__name__}"}}

    async def report(self) -> None:
        await self.call("POST", "/sessions", {"open": len(self.open)})

    async def enter(self, ws, say_loading) -> None:
        """counted first, so nothing unloads while this waits. Raises Rejected if the engines won't come up for it."""
        self.open.add(ws)
        await self.report()
        code, status = await self.call("GET", "/status")
        if code == 200 and (status["state"] in ("loading", "unloading") or status["wakes_on_session"]):
            say_loading("Loading the engines...")
        code, r = await self.call("POST", "/ensure_ready", timeout=None)      # as long as the engines take; their own timeouts apply
        if code != 200:
            await self.leave(ws)
            raise Rejected(r["error"]["code"], r["error"]["message"])
        if r["generation"] != self.warm_generation:
            self.warm_generation = r["generation"]
            if self.warm:
                await self.warm()

    async def leave(self, ws) -> None:
        if ws in self.open:
            self.open.discard(ws)
            await self.report()

    async def end_sessions(self) -> None:
        for ws in list(self.open):
            await ws.close(code=1001, message=b"the engines are being offloaded")

    async def close(self) -> None:
        if self.http:
            await self.http.close()


def our_page(request) -> bool:
    """a browser on the page this server serves. Checking the host too keeps out other sites' pages through DNS rebinding."""
    host = request.host.rsplit(":", 1)[0] if not request.host.endswith("]") else request.host
    if host not in (*LOCAL_HOSTS, CFG.ui_host):
        return False
    if request.method == "GET":
        return request.headers.get("Sec-Fetch-Site") == "same-origin"
    return bool(request.headers.get("Origin")) and origin_allowed(request)


def add_routes(app, gate: Gate | None, prefix: str) -> None:
    def fail(status: int, code: str, message: str):
        return web.json_response({"error": {"code": code, "message": message}}, status=status)

    def refused(request):
        if not origin_allowed(request):
            return fail(403, "forbidden", "this origin may not use this server")
        if gate is None:
            return fail(503, "no_host", "this server has no engines to control")
        if request.headers.get("Authorization"):
            return None if token_ok(request) else fail(401, "unauthorized", "a valid Authorization: Bearer token is required")
        return None if our_page(request) else fail(401, "unauthorized", "a valid Authorization: Bearer token is required")

    async def status(request):
        if (no := refused(request)) is not None:
            return no
        code, body = await gate.call("GET", "/status" + ("?memory=1" if request.query.get("memory") == "1" else ""), timeout=30)
        return web.json_response(body, status=code)

    def forward(path: str, ends_sessions: bool):
        async def post(request):
            if (no := refused(request)) is not None:
                return no
            try:
                body = await request.json() if request.can_read_body else {}
            except json.JSONDecodeError:
                return fail(400, "invalid_json", "the body isn't JSON")
            if not isinstance(body, dict):
                return fail(400, "invalid_request", "the body must be a JSON object")
            force = body.get("force") is True
            if ends_sessions and force:
                await gate.end_sessions()
            code, reply = await gate.call("POST", path, {"force": force}, timeout=120)
            return web.json_response(reply, status=code)
        return post

    app.router.add_get(f"{prefix}/v1/control/status", status)
    app.router.add_post(f"{prefix}/v1/control/load", forward("/load", False))
    app.router.add_post(f"{prefix}/v1/control/offload", forward("/offload", True))
