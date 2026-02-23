"""Image and blob API routes."""
from __future__ import annotations

import os
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()

_BLOB_KEY_RE = re.compile(r"^[a-f0-9]{16}\.[a-z0-9]+$")


@router.get("/api/runs/{run}/images")
async def get_images(
    run: str,
    tag: str,
    downsample: int = 100,
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    return provider.read_images(tag, downsample)


@router.get("/api/runs/{run}/blob/{blob_key}")
async def get_blob(
    run: str,
    blob_key: str,
    watcher=Depends(get_watcher),
):
    # Security: validate blob_key format to prevent path traversal
    if not _BLOB_KEY_RE.match(blob_key):
        raise HTTPException(400, "Invalid blob key format")

    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")

    info = provider.get_blob_info(blob_key)
    if info is None:
        raise HTTPException(404, "Blob not found")

    blob_path = os.path.join(provider.run_dir, "blobs", blob_key)
    if not os.path.isfile(blob_path):
        raise HTTPException(404, "Blob file missing")

    return FileResponse(blob_path, media_type=info["mime_type"])
