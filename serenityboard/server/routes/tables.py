"""Tables plugin API routes."""
from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/tables", tags=["tables"])

# Set by app factory
_tables_service = None


def set_tables_service(svc):
    global _tables_service
    _tables_service = svc


def _svc():
    if _tables_service is None:
        raise HTTPException(503, "Tables service not initialized")
    return _tables_service


# ── Table CRUD ──────────────────────────────────────────────────────


@router.post("")
async def create_table(request: Request):
    body = await request.json()
    name = body.get("name")
    if not name:
        raise HTTPException(400, "name is required")
    schema = body.get("schema")
    if not schema:
        raise HTTPException(400, "schema is required")
    try:
        result = _svc().create_table(
            name=name,
            schema=schema,
            description=body.get("description", ""),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"Table name {name!r} already exists")
    return result


@router.get("")
async def list_tables():
    return _svc().list_tables()


@router.get("/{table_id}")
async def get_table(table_id: str):
    tbl = _svc().get_table(table_id)
    if not tbl:
        raise HTTPException(404, f"Table {table_id} not found")
    return tbl


@router.patch("/{table_id}")
async def update_table(table_id: str, request: Request):
    body = await request.json()
    try:
        result = _svc().update_table(table_id, **body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not result:
        raise HTTPException(404, f"Table {table_id} not found")
    return result


@router.delete("/{table_id}")
async def delete_table(table_id: str):
    if not _svc().delete_table(table_id):
        raise HTTPException(404, f"Table {table_id} not found")
    return {"deleted": True}


# ── Row CRUD ────────────────────────────────────────────────────────


@router.post("/{table_id}/rows")
async def insert_row(table_id: str, request: Request):
    tbl = _svc().get_table(table_id)
    if not tbl:
        raise HTTPException(404, f"Table {table_id} not found")
    body = await request.json()
    values = body.get("values", {})
    return _svc().insert_row(
        table_id=table_id,
        values=values,
        run_name=body.get("run_name"),
        step=body.get("step"),
        tags=body.get("tags"),
    )


@router.post("/{table_id}/rows:bulk")
async def bulk_insert_rows(table_id: str, request: Request):
    tbl = _svc().get_table(table_id)
    if not tbl:
        raise HTTPException(404, f"Table {table_id} not found")
    body = await request.json()
    rows = body.get("rows", [])
    if not rows:
        raise HTTPException(400, "rows array is required and must be non-empty")
    try:
        return _svc().bulk_insert_rows(table_id, rows)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/{table_id}/rows")
async def query_rows(
    table_id: str,
    filter: str | None = None,
    sort: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    tbl = _svc().get_table(table_id)
    if not tbl:
        raise HTTPException(404, f"Table {table_id} not found")
    filter_expr = None
    if filter:
        try:
            filter_expr = json.loads(filter)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid filter JSON: {e}")
    try:
        return _svc().query_rows(
            table_id=table_id,
            filter_expr=filter_expr,
            sort=sort,
            limit=limit,
            offset=offset,
            schema=tbl["schema"],
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.patch("/{table_id}/rows/{row_id}")
async def update_row(table_id: str, row_id: str, request: Request):
    body = await request.json()
    result = _svc().update_row(table_id, row_id, **body)
    if not result:
        raise HTTPException(404, "Row not found")
    return result


@router.delete("/{table_id}/rows/{row_id}")
async def delete_row(table_id: str, row_id: str):
    if not _svc().delete_row(table_id, row_id):
        raise HTTPException(404, "Row not found")
    return {"deleted": True}
