"""FastAPI application factory for SerenityBoard."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from serenityboard.server.live_updates import LiveUpdateManager, SubscriptionFilter
from serenityboard.server.routes import (
    artifacts_router,
    audio_router,
    custom_scalars_router,
    distributions_router,
    embeddings_router,
    eval_router,
    graphs_router,
    hparams_router,
    histograms_router,
    images_router,
    meshes_router,
    metrics_router,
    notes_router,
    plugins_router,
    pr_curves_router,
    scalars_router,
    set_watcher,
    text_router,
    traces_router,
)
from serenityboard.server.routes.automations import (
    router as automations_router,
    set_automations_service,
)
from serenityboard.server.routes.registry import (
    router as registry_router,
    set_registry_service,
)
from serenityboard.server.routes.sweeps import (
    router as sweeps_router,
    set_sweeps_service,
)
from serenityboard.server.routes.tables import (
    router as tables_router,
    set_tables_service,
)
from serenityboard.server.run_manager import RunWatcher

__all__ = ["create_app"]

logger = logging.getLogger(__name__)


def create_app(logdir: str) -> FastAPI:
    """Build and return the SerenityBoard FastAPI application.

    Parameters
    ----------
    logdir:
        Root directory containing run sub-directories with ``board.db`` files.
    """
    watcher = RunWatcher(logdir)
    watcher.scan_once()
    set_watcher(watcher)

    live_manager = LiveUpdateManager()
    live_manager.set_watcher(watcher)

    # ── Ops modules (opt-in, fail gracefully) ───────────────────────
    try:
        from serenityboard.ops.db import OpsDB
        from serenityboard.ops.automations import AutomationsService
        from serenityboard.ops.registry import RegistryService
        from serenityboard.ops.sweeps import SweepsService
        from serenityboard.ops.tables import TablesService

        ops_db = OpsDB(logdir)
        set_tables_service(TablesService(ops_db))
        set_registry_service(RegistryService(ops_db))
        set_sweeps_service(SweepsService(ops_db))
        set_automations_service(AutomationsService(ops_db))
    except Exception:
        logger.info("Ops modules not available, running in core-only mode")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def _poll_loop():
            while True:
                try:
                    await live_manager.poll_and_push()
                except Exception:
                    logger.warning("poll_and_push error", exc_info=True)
                await asyncio.sleep(1.0)

        poll_task = asyncio.create_task(_poll_loop())
        scan_task = asyncio.create_task(watcher.scan_loop())
        yield
        poll_task.cancel()
        scan_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        try:
            await scan_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="SerenityBoard", lifespan=lifespan)

    @app.middleware("http")
    async def frontend_no_cache_middleware(request: Request, call_next):
        """Prevent stale frontend assets from being served out of browser cache."""
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".js", ".css", ".html")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # ── spec-compliant error model (§9) ───────────────────────────
    _ERROR_CODES = {
        400: "invalid_request",
        404: "not_found",
        409: "conflict",
        429: "rate_limited",
        503: "service_unavailable",
    }

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        code = _ERROR_CODES.get(exc.status_code, "error")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": str(exc.detail),
                    "details": {},
                }
            },
        )

    # ── sub-routers ────────────────────────────────────────────────
    app.include_router(scalars_router)
    app.include_router(images_router)
    app.include_router(text_router)
    app.include_router(hparams_router)
    app.include_router(histograms_router)
    app.include_router(distributions_router)
    app.include_router(plugins_router)
    app.include_router(traces_router)
    app.include_router(eval_router)
    app.include_router(artifacts_router)
    app.include_router(metrics_router)
    app.include_router(custom_scalars_router)
    app.include_router(pr_curves_router)
    app.include_router(audio_router)
    app.include_router(graphs_router)
    app.include_router(meshes_router)
    app.include_router(embeddings_router)
    app.include_router(notes_router)
    app.include_router(tables_router)
    app.include_router(registry_router)
    app.include_router(sweeps_router)
    app.include_router(automations_router)

    # ── top-level endpoints ────────────────────────────────────────

    @app.get("/api/runs")
    async def list_runs():
        watcher.scan_once()
        return watcher.get_runs()

    @app.get("/api/runs/{run}/tags")
    async def get_tags(run: str):
        provider = watcher.get_provider(run)
        if not provider:
            raise HTTPException(404, "Run not found")
        return provider.get_tags()

    @app.delete("/api/runs/{run}")
    async def delete_run(run: str):
        if watcher.delete_run(run):
            return {"deleted": run}
        raise HTTPException(404, "Run not found")

    @app.get("/api/compare/scalars")
    async def compare_scalars(
        tag: str, runs: str, downsample: int = 5000, x_axis: str = "step"
    ):
        run_names = [r.strip() for r in runs.split(",") if r.strip()]
        result = {}
        for name in run_names:
            provider = watcher.get_provider(name)
            if provider:
                rows = provider.read_scalars_downsampled(tag, downsample)
                if x_axis == "wall_time":
                    result[name] = [[r[1], r[1], r[2]] for r in rows]
                elif x_axis == "relative":
                    t0 = rows[0][1] if rows else 0
                    result[name] = [[r[1] - t0, r[1], r[2]] for r in rows]
                else:
                    result[name] = [[r[0], r[1], r[2]] for r in rows]
        return result

    @app.get("/api/compare/eval")
    async def compare_eval(
        suite: str, runs: str, score: str | None = None
    ):
        run_names = [r.strip() for r in runs.split(",") if r.strip()]
        result = {}
        for name in run_names:
            provider = watcher.get_provider(name)
            if provider:
                evals = provider.read_eval_results(suite_name=suite)
                if score:
                    evals = [e for e in evals if e["score_name"] == score]
                result[name] = evals
        return result

    # ── WebSocket live updates ────────────────────────────────────

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    continue
                if "subscribe" in msg:
                    sub = msg["subscribe"]
                    tag_patterns = set(sub.get("tags") or ["*"])
                    kinds = set(sub.get("kinds") or ["scalar"])
                    filt = SubscriptionFilter(
                        runs=sub.get("runs", []),
                        tag_patterns=tag_patterns,
                        kinds=kinds,
                    )
                    live_manager.subscribe(ws, filt)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("WebSocket error", exc_info=True)
        finally:
            live_manager.unsubscribe(ws)

    # ── static frontend (if built) ─────────────────────────────────

    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.isdir(frontend_dir):
        from fastapi.staticfiles import StaticFiles

        app.mount(
            "/", StaticFiles(directory=frontend_dir, html=True), name="frontend"
        )

    return app
