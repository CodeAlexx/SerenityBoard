"""Run notes API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


class NoteBody(BaseModel):
    note: str


@router.get("/api/runs/{run}/notes")
async def get_note(
    run: str,
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    return provider.get_note()


@router.put("/api/runs/{run}/notes")
async def put_note(
    run: str,
    body: NoteBody,
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    provider.set_note(body.note)
    return {"ok": True}
