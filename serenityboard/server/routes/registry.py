"""Artifact Registry API routes."""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/registry", tags=["registry"])

_registry_service = None


def set_registry_service(svc):
    global _registry_service
    _registry_service = svc


def _svc():
    if _registry_service is None:
        raise HTTPException(503, "Registry service not initialized")
    return _registry_service


# ── Collections ─────────────────────────────────────────────────────


@router.post("/collections")
async def create_collection(request: Request):
    body = await request.json()
    name = body.get("name")
    if not name:
        raise HTTPException(400, "name is required")
    kind = body.get("kind")
    if not kind:
        raise HTTPException(400, "kind is required")
    try:
        return _svc().create_collection(
            name=name, kind=kind, description=body.get("description", "")
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"Collection {name!r} already exists")


@router.get("/collections")
async def list_collections():
    return _svc().list_collections()


@router.get("/collections/{collection_id}")
async def get_collection(collection_id: str):
    coll = _svc().get_collection(collection_id)
    if not coll:
        raise HTTPException(404, f"Collection {collection_id} not found")
    return coll


@router.patch("/collections/{collection_id}")
async def update_collection(collection_id: str, request: Request):
    body = await request.json()
    try:
        result = _svc().update_collection(collection_id, **body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not result:
        raise HTTPException(404, f"Collection {collection_id} not found")
    return result


# ── Versions ────────────────────────────────────────────────────────


@router.post("/collections/{collection_id}/versions")
async def create_version(collection_id: str, request: Request):
    coll = _svc().get_collection(collection_id)
    if not coll:
        raise HTTPException(404, f"Collection {collection_id} not found")
    body = await request.json()
    try:
        return _svc().create_version(
            collection_id=collection_id,
            digest=body["digest"],
            size_bytes=body["size_bytes"],
            storage_uri=body["storage_uri"],
            metadata=body.get("metadata"),
            created_by_run=body.get("created_by_run"),
        )
    except KeyError as e:
        raise HTTPException(400, f"Missing required field: {e}")
    except sqlite3.IntegrityError as e:
        raise HTTPException(409, str(e))


@router.get("/collections/{collection_id}/versions")
async def list_versions(
    collection_id: str, limit: int = 100, offset: int = 0
):
    return _svc().list_versions(collection_id, limit, offset)


@router.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str):
    art = _svc().get_artifact(artifact_id)
    if not art:
        raise HTTPException(404, f"Artifact {artifact_id} not found")
    return art


# ── Aliases ─────────────────────────────────────────────────────────


@router.put("/collections/{collection_id}/aliases/{alias}")
async def set_alias(collection_id: str, alias: str, request: Request):
    body = await request.json()
    artifact_id = body.get("artifact_id")
    if not artifact_id:
        raise HTTPException(400, "artifact_id is required")
    return _svc().set_alias(collection_id, alias, artifact_id)


@router.get("/collections/{collection_id}/aliases")
async def list_aliases(collection_id: str):
    return _svc().list_aliases(collection_id)


@router.get("/resolve/{collection_id}:{alias}")
async def resolve_alias(collection_id: str, alias: str):
    result = _svc().resolve_alias(collection_id, alias)
    if not result:
        raise HTTPException(404, f"Alias {alias!r} not found in {collection_id}")
    return result


# ── Lineage ─────────────────────────────────────────────────────────


@router.post("/lineage")
async def add_lineage(request: Request):
    body = await request.json()
    try:
        return _svc().add_lineage(
            parent_artifact_id=body["parent_artifact_id"],
            child_artifact_id=body["child_artifact_id"],
            relation=body["relation"],
        )
    except KeyError as e:
        raise HTTPException(400, f"Missing required field: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Lineage edge already exists")


@router.get("/artifacts/{artifact_id}/lineage")
async def get_lineage(
    artifact_id: str, direction: str = "down", depth: int = 3
):
    return _svc().get_lineage(artifact_id, direction, depth)
