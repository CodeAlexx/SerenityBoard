"""Embeddings API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/embeddings")
async def list_embeddings(
    run: str,
    tag: str | None = None,
    step: int | None = None,
    downsample: int = 20,
    watcher=Depends(get_watcher),
):
    """List embeddings, optionally filtered by tag and step."""
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    result = provider.read_embeddings(tag=tag, step=step, downsample=downsample)
    if tag and step is not None and not result:
        raise HTTPException(404, "Embedding not found")
    if tag and step is not None and result:
        return result[0]
    return result
