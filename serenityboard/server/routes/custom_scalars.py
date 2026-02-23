"""Custom scalars layout API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/custom-scalars/layout")
async def get_custom_scalars_layout(
    run: str,
    watcher=Depends(get_watcher),
):
    """Return custom scalars layout config."""
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    layout = provider.get_custom_scalar_layout()
    if layout is None:
        return {"categories": []}
    return layout


@router.get("/api/runs/{run}/custom-scalars/data")
async def get_custom_scalars_data(
    run: str,
    tags: str = Query(..., description="Comma-separated regex patterns"),
    downsample: int = 5000,
    watcher=Depends(get_watcher),
):
    """Return scalar data for tags matching regex patterns."""
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    tag_regexes = [t.strip() for t in tags.split(",") if t.strip()]
    if not tag_regexes:
        return {}
    return provider.read_custom_scalars(tag_regexes, downsample)
