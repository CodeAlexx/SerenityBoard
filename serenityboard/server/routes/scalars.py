"""Scalar data API routes."""
from __future__ import annotations

import csv
import io
import math
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from serenityboard.server.routes import get_watcher

__all__ = ["router"]

router = APIRouter()


@router.get("/api/runs/{run}/scalars")
async def get_scalars(
    run: str,
    tag: str,
    downsample: int = 5000,
    x_axis: str = "step",
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    rows = provider.read_scalars_downsampled(tag, downsample)
    # rows are [step, wall_time, value]
    if x_axis == "wall_time":
        return [[r[1], r[1], r[2]] for r in rows]
    elif x_axis == "relative":
        if rows:
            t0 = rows[0][1]
            return [[r[1] - t0, r[1], r[2]] for r in rows]
        return []
    # Default: x_axis="step"
    return [[r[0], r[1], r[2]] for r in rows]


@router.get("/api/runs/{run}/scalars/last")
async def get_scalars_last(
    run: str,
    tags: str,
    watcher=Depends(get_watcher),
):
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    return provider.read_scalars_last(tag_list)


@router.get("/api/runs/{run}/export")
async def export_scalars(
    run: str,
    format: str = "csv",
    tags: str | None = None,
    x_axis: str = "step",
    watcher=Depends(get_watcher),
):
    """Export scalar data as CSV or JSON for a single run."""
    provider = watcher.get_provider(run)
    if not provider:
        raise HTTPException(404, "Run not found")

    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        all_tags = provider.get_tags()
        tag_list = sorted(all_tags.get("scalars", []))

    if not tag_list:
        raise HTTPException(404, "No scalar tags found")

    tag_data: dict[str, list[tuple]] = {}
    for tag in tag_list:
        rows = provider.read_scalars_downsampled(tag, 0)
        tag_data[tag] = rows

    if format == "json":
        return _build_json_response(tag_data, x_axis, run, tag_list)
    return _build_csv_response(tag_data, x_axis, run, tag_list)


def _build_csv_response(
    tag_data: dict[str, list[tuple]],
    x_axis: str,
    run: str,
    tag_list: list[str],
) -> StreamingResponse:
    """Build a pivoted CSV: step, wall_time, tag1, tag2, ..."""
    t0: float | None = None
    if x_axis == "relative":
        for rows in tag_data.values():
            if rows:
                candidate = rows[0][1]
                if t0 is None or candidate < t0:
                    t0 = candidate

    step_index: dict[int, dict] = {}
    for tag, rows in tag_data.items():
        for r in rows:
            step = r[0]
            if step not in step_index:
                step_index[step] = {"wall_time": r[1]}
            step_index[step][tag] = r[2]

    sorted_steps = sorted(step_index.keys())

    buf = io.StringIO()
    writer = csv.writer(buf)

    x_label = "step"
    if x_axis == "time":
        x_label = "wall_time"
    elif x_axis == "relative":
        x_label = "relative_time"
    writer.writerow([x_label, "wall_time"] + tag_list)

    for step in sorted_steps:
        entry = step_index[step]
        wt = entry["wall_time"]
        x = step if x_axis == "step" else (wt if x_axis == "time" else wt - (t0 or 0.0))
        row = [x, wt]
        for tag in tag_list:
            val = entry.get(tag)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                row.append("")
            else:
                row.append(val)
        writer.writerow(row)

    safe_run = re.sub(r"[^a-zA-Z0-9_\-]", "_", run)
    safe_tags = "_".join(re.sub(r"[^a-zA-Z0-9_\-]", "_", t) for t in tag_list[:3])
    if len(tag_list) > 3:
        safe_tags += f"_+{len(tag_list) - 3}"
    filename = f"{safe_run}_{safe_tags}.csv"

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _build_json_response(
    tag_data: dict[str, list[tuple]],
    x_axis: str,
    run: str,
    tag_list: list[str],
) -> JSONResponse:
    """Build JSON array: [{step, wall_time, values: {tag: value}}]."""
    t0: float | None = None
    if x_axis == "relative":
        for rows in tag_data.values():
            if rows:
                candidate = rows[0][1]
                if t0 is None or candidate < t0:
                    t0 = candidate

    step_index: dict[int, dict] = {}
    for tag, rows in tag_data.items():
        for r in rows:
            step = r[0]
            if step not in step_index:
                step_index[step] = {"wall_time": r[1], "values": {}}
            step_index[step]["values"][tag] = r[2]

    result = []
    for step in sorted(step_index.keys()):
        entry = step_index[step]
        wt = entry["wall_time"]
        x = step if x_axis == "step" else (wt if x_axis == "time" else wt - (t0 or 0.0))
        result.append({
            "step": step,
            "wall_time": wt,
            "x": x,
            "values": entry["values"],
        })

    return JSONResponse(content=result)
