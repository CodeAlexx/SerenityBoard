"""PR Curves API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/pr-curves")
async def get_pr_curves(
    run: str,
    tag: str,
    downsample: int = 50,
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    return provider.read_pr_curves(tag, downsample=downsample)
