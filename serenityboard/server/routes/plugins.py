"""Plugin discovery API routes."""
from __future__ import annotations

from fastapi import APIRouter

__all__ = ["router"]

router = APIRouter()


@router.get("/api/plugins")
async def list_plugins():
    return [
        {"name": "scalars", "display_name": "Scalars", "active": True},
        {"name": "images", "display_name": "Images", "active": True},
        {"name": "text", "display_name": "Text", "active": True},
        {"name": "histograms", "display_name": "Histograms", "active": False},
        {"name": "hparams", "display_name": "HParams", "active": True},
        {"name": "traces", "display_name": "Traces", "active": True},
        {"name": "eval", "display_name": "Eval", "active": True},
        {"name": "artifacts", "display_name": "Artifacts", "active": True},
        {"name": "graphs", "display_name": "Graphs", "active": True},
        {"name": "embeddings", "display_name": "Embeddings", "active": True},
        {"name": "meshes", "display_name": "Meshes", "active": True},
    ]
