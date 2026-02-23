"""Unified metrics API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/metrics")
async def get_metrics(
    run: str,
    watcher=Depends(get_watcher),
):
    """Return unified tag index with counts across all data types."""
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    return provider.get_all_metric_tags()


@router.post("/api/runs/{run}/metrics/timeseries")
async def get_metrics_timeseries(
    run: str,
    request: Request,
    watcher=Depends(get_watcher),
):
    """Batch fetch multiple tags across types in one request."""
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    body = await request.json()
    requests = body.get("requests", [])
    if not requests:
        return []
    return provider.read_metric_timeseries(requests)
