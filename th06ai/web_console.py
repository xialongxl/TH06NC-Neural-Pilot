"""Loopback REST/SSE server and static frontend; owns no CUDA/game handles."""
from __future__ import annotations
from .i18n import msg

from contextlib import asynccontextmanager
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import threading
import time
from typing import Literal
from urllib.request import urlopen
import webbrowser

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Config, DEFAULT_CONSOLE_PORT, DEFAULT_PREVIEW_PORT, ROOT
from .battle_view import BattleViewStream
from .console_performance import CpuMonitor
from .console_service import ConsoleError, ConsoleService
from .console_stream import SnapshotBroker


class GameSetupRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    mode: Literal['story','practice'] = 'story'
    difficulty: Literal['easy','normal','hard','lunatic','extra'] = 'normal'
    character: Literal['reimu_a','reimu_b','marisa_a','marisa_b'] = 'reimu_a'
    stage: int = Field(default=1, ge=1, le=6)


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["resume", "train", "play", "evaluate", "check", "monitor", "calibrate"]
    minutes: float = Field(gt=0, le=60)
    model_id: str | None = None
    goal: Literal['survival','nmnb'] | None = None
    game_setup: GameSetupRequest | None = None


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=80)


class PauseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    paused: bool

class BattleViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool

class AccelerationRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    collision:Literal['python','rust','auto']
    input:Literal['python','rust','auto']


def create_app(config: Config, *, root=ROOT, preview=False, service=None, frontend=None,
               shutdown=None, shutdown_timeout=45):
    root = Path(root).resolve()
    service = service or ConsoleService(config, root=root, preview=preview)
    frontend = Path(frontend or root / "frontend/dist").resolve()
    battle_view = BattleViewStream(service.battle_view_session)
    broker = SnapshotBroker(service, lambda: battle_view.enabled)
    cpu_monitor = CpuMonitor()
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            task = app.state.shutdown_task
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await battle_view.close()
            await broker.close()
    app = FastAPI(title="TH06NC Console", version="1.0", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.console = service
    app.state.stream = broker
    app.state.shutdown_task = None
    token = secrets.token_urlsafe(32)
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_controls(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if (origin is not None and origin != expected) or not secrets.compare_digest(
                    request.headers.get("x-console-token", ""), token):
                return JSONResponse({"detail": msg('backend.control_request_unauthorized_reopen_the_local_console')}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'")
        if request.url.path.startswith("/api/") or request.url.path in {"/", "/index.html", "/battle-view"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ConsoleError)
    async def console_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status_code)

    @app.get("/api/health")
    def health():
        return {"service": "th06nc-console", "project": identity, "preview": service.preview, "pid": os.getpid()}

    @app.get("/api/bootstrap")
    def bootstrap(state: bool = True):
        result = {"token": token, "protocol": 2}
        if state:
            result["state"] = {**service.snapshot(), 'battle_view_enabled': battle_view.enabled}
        return result

    @app.get("/api/state")
    def state():
        return {**service.snapshot(), 'battle_view_enabled': battle_view.enabled}

    @app.get('/api/performance')
    def performance():
        return {'cpu': cpu_monitor.snapshot()}

    @app.get("/api/events")
    async def events(request: Request, protocol: Literal["1", "2"] = "1", hidden: bool = False):
        return StreamingResponse(broker.events(request, protocol=int(protocol), hidden=hidden),
                                 media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.get('/api/battle-view/events')
    async def battle_view_events(request: Request):
        return StreamingResponse(battle_view.events(request), media_type='text/event-stream',
                                 headers={'X-Accel-Buffering': 'no'})

    @app.post('/api/battle-view')
    async def configure_battle_view(body: BattleViewRequest):
        if service.preview:
            raise ConsoleError(msg('backend.battle_view_unavailable_in_preview'), 403)
        battle_view.set_enabled(body.enabled)
        return {'enabled': battle_view.enabled, 'message': msg(
            'backend.battle_view_enabled' if body.enabled else 'backend.battle_view_disabled')}

    @app.get("/api/logs")
    def logs(request: Request):
        data = service.logs()
        etag = '"'+hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()+'"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return JSONResponse(data, headers={"ETag": etag})

    @app.post("/api/models/select")
    def select_model(body: ModelRequest):
        return {"selected_model_id": service.select_model(body.model_id)}

    @app.post("/api/sessions", status_code=202)
    def start(body: StartRequest):
        options = {}
        if body.goal is not None:
            options['goal'] = body.goal
        if body.game_setup is not None:
            options['game_setup'] = body.game_setup.model_dump()
        return service.start(body.mode, body.minutes, body.model_id, **options)

    @app.post("/api/session/pause")
    def pause(body: PauseRequest):
        return service.control("pause", paused=body.paused)

    @app.post("/api/session/finish")
    def finish():
        return service.control("finish")

    @app.post("/api/session/stop")
    def stop():
        return service.control("stop")

    @app.post('/api/session/focus')
    def focus():
        return service.control('focus')

    @app.post('/api/game/focus')
    def focus_game():
        return service.focus_game()

    @app.post('/api/game/probe')
    def probe_game():
        return service.probe_game()

    @app.post('/api/settings/acceleration')
    def acceleration(body:AccelerationRequest):
        return service.set_acceleration(body.collision,body.input)

    async def finish_shutdown():
        try:
            deadline = time.monotonic() + shutdown_timeout
            while not await asyncio.to_thread(service.shutdown_ready):
                if time.monotonic() >= deadline:
                    raise RuntimeError(msg('backend.the_session_is_still_running_check_save'))
                await asyncio.sleep(.1)
            service.shutdown_status('stopping', msg('backend.session_ended_shutting_down_the_service'))
            # Let the POST acknowledgement reach the browser before closing SSE.
            await asyncio.sleep(.2)
            shutdown()
            await battle_view.close()
            await broker.close(shutdown=True)
        except Exception as error:
            service.shutdown_status('error', str(error))

    @app.post('/api/server/shutdown', status_code=202)
    async def close_server(request: Request):
        if shutdown is None:
            raise ConsoleError(msg('backend.this_service_does_not_support_browser_shutdown'), 409)
        result = await asyncio.to_thread(service.request_shutdown, port=request.url.port or DEFAULT_CONSOLE_PORT)
        if result['state'] != 'stopping' and (app.state.shutdown_task is None or app.state.shutdown_task.done()):
            app.state.shutdown_task = asyncio.create_task(finish_shutdown())
        return {**result, 'message': result['message']}

    if (frontend / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/{path:path}")
    def page(path: str):
        if path.startswith("api/"):
            return JSONResponse({"detail": msg('backend.api_endpoint_not_found')}, status_code=404)
        if path in {"favicon.svg", "icon.svg"} and (frontend / path).is_file():
            return FileResponse(frontend / path)
        if not (frontend / "index.html").is_file():
            return JSONResponse({"detail": msg('backend.build_the_frontend_by_running_npm_ci')}, status_code=503)
        return FileResponse(frontend / "index.html")

    return app


def launch(config: Config, *, preview=False, port=DEFAULT_CONSOLE_PORT, open_browser=True):
    import uvicorn
    if not 1024 <= port <= 65535:
        raise ValueError(msg('backend.the_console_port_must_be_between_1024'))
    if preview and port == DEFAULT_CONSOLE_PORT:
        port = DEFAULT_PREVIEW_PORT
    url = f"http://127.0.0.1:{port}"
    identity = hashlib.sha256(str(ROOT.resolve()).encode()).hexdigest()[:16]
    try:
        with urlopen(url+"/api/health", timeout=.6) as response:
            existing = json.load(response)
        if (existing.get("service"), existing.get("project"), existing.get("preview")) == ("th06nc-console", identity, preview):
            if open_browser:
                webbrowser.open(url)
            return
    except (OSError, ValueError):
        pass
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        raise RuntimeError(msg('backend.port_0_is_in_use_select_another', port)) from None
    (ROOT / "runs").mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / "runs/web-console.log", level=logging.INFO, encoding="utf-8")
    def shutdown():
        server.should_exit = True
    server = uvicorn.Server(uvicorn.Config(create_app(config, preview=preview, shutdown=shutdown),
                                         host="127.0.0.1", port=port, timeout_graceful_shutdown=7,
                                         log_config=None, access_log=False))
    if open_browser:
        def show_when_ready():
            for _ in range(100):
                if server.started:
                    webbrowser.open(url)
                    return
                time.sleep(.1)
        threading.Thread(target=show_when_ready, name="console-browser", daemon=True).start()
    try:
        server.run(sockets=[sock])
    finally:
        sock.close()
