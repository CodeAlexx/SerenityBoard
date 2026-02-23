"""Mesh API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/meshes")
async def list_mesh_tags(
    run: str,
    tag: str | None = None,
    step: int | None = None,
    downsample: int = 50,
    watcher=Depends(get_watcher),
):
    """List meshes, optionally filtered by tag and step."""
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    results = provider.read_meshes(tag=tag, step=step, downsample=downsample)
    if tag and step is not None and not results:
        raise HTTPException(404, "Mesh not found for this tag and step")
    if tag and step is not None and results:
        return results[0]
    return results
