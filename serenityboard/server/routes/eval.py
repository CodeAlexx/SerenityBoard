"""Eval results API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/eval")
async def get_eval(
    run: str,
    suite: str,
    step: int | None = None,
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    return provider.read_eval_results(suite_name=suite, step=step)
