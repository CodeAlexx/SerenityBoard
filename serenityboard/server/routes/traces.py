"""Trace events API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/traces")
async def get_traces(
    run: str,
    step_from: int | None = None,
    step_to: int | None = None,
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    return provider.read_trace_events(step_from=step_from, step_to=step_to)
