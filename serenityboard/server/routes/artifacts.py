"""Artifact API routes (canonical endpoint replacing images)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/artifacts")
async def get_artifacts(
    run: str,
    tag: str,
    downsample: int = 100,
    kind: str | None = None,
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    return provider.read_artifacts(tag, downsample=downsample, kind=kind)
