"""Sweeps service API routes."""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/sweeps", tags=["sweeps"])

_sweeps_service = None


def set_sweeps_service(svc):
    global _sweeps_service
    _sweeps_service = svc


def _svc():
    if _sweeps_service is None:
        raise HTTPException(503, "Sweeps service not initialized")
    return _sweeps_service


# ── Sweep Lifecycle ─────────────────────────────────────────────────


@router.post("")
async def create_sweep(request: Request):
    body = await request.json()
    try:
        objective = body.get("objective", {})
        return _svc().create_sweep(
            name=body["name"],
            method=body["method"],
            objective_metric=objective.get("metric", body.get("objective_metric", "")),
            objective_mode=objective.get("mode", body.get("objective_mode", "")),
            parameter_space=body["parameters"],
            max_trials=body["max_trials"],
            parallelism=body.get("parallelism", 1),
        )
    except KeyError as e:
        raise HTTPException(400, f"Missing required field: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"Sweep name {body.get('name')!r} already exists")


@router.get("")
async def list_sweeps():
    return _svc().list_sweeps()


@router.get("/{sweep_id}")
async def get_sweep(sweep_id: str):
    sweep = _svc().get_sweep(sweep_id)
    if not sweep:
        raise HTTPException(404, f"Sweep {sweep_id} not found")
    return sweep


@router.post("/{sweep_id}/pause")
async def pause_sweep(sweep_id: str):
    result = _svc().pause_sweep(sweep_id)
    if not result:
        raise HTTPException(404, f"Sweep {sweep_id} not found")
    return result


@router.post("/{sweep_id}/resume")
async def resume_sweep(sweep_id: str):
    result = _svc().resume_sweep(sweep_id)
    if not result:
        raise HTTPException(404, f"Sweep {sweep_id} not found")
    return result


@router.post("/{sweep_id}/cancel")
async def cancel_sweep(sweep_id: str):
    result = _svc().cancel_sweep(sweep_id)
    if not result:
        raise HTTPException(404, f"Sweep {sweep_id} not found")
    return result


# ── Trials ──────────────────────────────────────────────────────────


@router.get("/{sweep_id}/trials")
async def list_trials(
    sweep_id: str,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    return _svc().list_trials(sweep_id, status, limit, offset)


@router.get("/{sweep_id}/best")
async def best_trials(sweep_id: str, top_k: int = 1):
    return _svc().get_best_trials(sweep_id, top_k)


# ── Agent Protocol ──────────────────────────────────────────────────


@router.post("/agents/register")
async def register_agent(request: Request):
    body = await request.json()
    return _svc().register_agent(labels=body.get("labels"))


@router.post("/agents/{agent_id}/heartbeat")
async def agent_heartbeat(agent_id: str):
    return _svc().heartbeat(agent_id)


@router.post("/agents/{agent_id}/next_trial")
async def next_trial(agent_id: str):
    result = _svc().next_trial(agent_id)
    if not result:
        return {"trial_id": None, "message": "no trials available"}
    return result


@router.post("/trials/{trial_id}/start")
async def start_trial(trial_id: str):
    result = _svc().start_trial(trial_id)
    if not result:
        raise HTTPException(404, f"Trial {trial_id} not found")
    return result


@router.post("/trials/{trial_id}/report")
async def report_trial(trial_id: str, request: Request):
    body = await request.json()
    result = _svc().report_trial(trial_id, body["objective_value"])
    if not result:
        raise HTTPException(404, f"Trial {trial_id} not found")
    return result


@router.post("/trials/{trial_id}/finish")
async def finish_trial(trial_id: str):
    result = _svc().finish_trial(trial_id)
    if not result:
        raise HTTPException(404, f"Trial {trial_id} not found")
    return result


@router.post("/trials/{trial_id}/fail")
async def fail_trial(trial_id: str, request: Request):
    body = await request.json()
    result = _svc().fail_trial(trial_id, body.get("error_text", ""))
    if not result:
        raise HTTPException(404, f"Trial {trial_id} not found")
    return result
