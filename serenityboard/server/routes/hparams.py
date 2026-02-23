"""Hyperparameter and comparison API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/hparams")
async def get_hparams(
    run: str,
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    return provider.get_hparams()


@router.get("/api/compare/hparams")
async def compare_hparams(
    runs: str,
    watcher=Depends(get_watcher),
):
    run_names = [r.strip() for r in runs.split(",") if r.strip()]
    result = []
    for name in run_names:
        provider = watcher.get_provider(name)
        if provider:
            hp = provider.get_hparams()
            result.append({"run": name, **hp})
    return result
