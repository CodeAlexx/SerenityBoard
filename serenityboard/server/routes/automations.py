"""Automations engine API routes."""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/automations", tags=["automations"])

_automations_service = None


def set_automations_service(svc):
    global _automations_service
    _automations_service = svc


def _svc():
    if _automations_service is None:
        raise HTTPException(503, "Automations service not initialized")
    return _automations_service


# ── Rule CRUD ───────────────────────────────────────────────────────


@router.post("/rules")
async def create_rule(request: Request):
    body = await request.json()
    try:
        return _svc().create_rule(
            name=body["name"],
            trigger_type=body["trigger_type"],
            trigger=body["trigger"],
            action_type=body["action_type"],
            action=body["action"],
            enabled=body.get("enabled", True),
            cooldown_seconds=body.get("cooldown_seconds", 0),
            dedupe_window_sec=body.get("dedupe_window_sec", 300),
        )
    except KeyError as e:
        raise HTTPException(400, f"Missing required field: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"Rule name {body.get('name')!r} already exists")


@router.get("/rules")
async def list_rules():
    return _svc().list_rules()


@router.patch("/rules/{rule_id}")
async def update_rule(rule_id: str, request: Request):
    body = await request.json()
    result = _svc().update_rule(rule_id, **body)
    if not result:
        raise HTTPException(404, f"Rule {rule_id} not found")
    return result


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    if not _svc().delete_rule(rule_id):
        raise HTTPException(404, f"Rule {rule_id} not found")
    return {"deleted": True}


# ── Event Ingest ────────────────────────────────────────────────────


@router.post("/events")
async def ingest_event(request: Request):
    body = await request.json()
    try:
        return _svc().ingest_event(
            event_type=body["event_type"],
            event_key=body["event_key"],
            payload=body.get("payload", {}),
        )
    except KeyError as e:
        raise HTTPException(400, f"Missing required field: {e}")


# ── Dry-Run Test ────────────────────────────────────────────────────


@router.post("/test")
async def test_rule(request: Request):
    """Dry-run: evaluate event against rules without dispatching."""
    body = await request.json()
    event = {
        "event_id": "dry_run",
        "event_type": body.get("event_type", ""),
        "event_key": body.get("event_key", "test"),
        "payload": body.get("payload", {}),
        "created_at": 0,
    }
    rules = _svc().list_rules()
    matches = []
    for rule in rules:
        if _svc()._matches_trigger(rule, event):
            matches.append({"rule_id": rule["rule_id"], "name": rule["name"]})
    return {"matches": matches, "event": event}


# ── Dispatch ────────────────────────────────────────────────────────


@router.get("/dispatch")
async def list_dispatch(status: str | None = None, limit: int = 100):
    return _svc().list_dispatch(status, limit)
