"""LoRA weight analysis API routes."""
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

__all__ = ["router"]

router = APIRouter()


class AnalyzeRequest(BaseModel):
    path: str


class CompareRequest(BaseModel):
    path_a: str
    path_b: str


def _get_analytics():
    """Import the analytics module (standalone, no Serenity dependency)."""
    try:
        from serenityboard.lora_analytics import (
            analyze_lora_file,
            compare_lora_files,
            diagnose,
            summary_stats,
        )
        return analyze_lora_file, compare_lora_files, diagnose, summary_stats
    except ImportError:
        raise HTTPException(
            503, "LoRA analytics not available (torch or safetensors missing)"
        )


# ---------------------------------------------------------------------------
# Path-based endpoints (server has filesystem access)
# ---------------------------------------------------------------------------

@router.post("/api/lora/analyze")
async def analyze_lora(req: AnalyzeRequest):
    """Analyze a LoRA safetensors file by path."""
    analyze_lora_file, _, diagnose, summary_stats = _get_analytics()

    if not os.path.isfile(req.path):
        raise HTTPException(404, f"File not found: {req.path}")
    if not req.path.endswith(".safetensors"):
        raise HTTPException(400, "Only .safetensors files are supported")

    metrics = analyze_lora_file(req.path)
    return {
        "layers": metrics,
        "summary": summary_stats(metrics),
        "diagnostics": diagnose(metrics),
        "num_layers": len(metrics),
    }


@router.post("/api/lora/compare")
async def compare_lora(req: CompareRequest):
    """Compare two LoRA safetensors files by path."""
    _, compare_lora_files, diagnose, summary_stats = _get_analytics()
    analyze_lora_file = _get_analytics()[0]

    for p in (req.path_a, req.path_b):
        if not os.path.isfile(p):
            raise HTTPException(404, f"File not found: {p}")
        if not p.endswith(".safetensors"):
            raise HTTPException(400, "Only .safetensors files are supported")

    comparison = compare_lora_files(req.path_a, req.path_b)

    # Compute diagnostics for each file separately
    m1 = analyze_lora_file(req.path_a)
    m2 = analyze_lora_file(req.path_b)

    return {
        "layers": comparison,
        "summary_a": summary_stats(m1),
        "summary_b": summary_stats(m2),
        "diagnostics": diagnose(m1) + diagnose(m2),
        "num_layers": len(comparison),
    }


# ---------------------------------------------------------------------------
# Upload-based endpoints (no filesystem access needed)
# ---------------------------------------------------------------------------

@router.post("/api/lora/analyze-upload")
async def analyze_lora_upload(file: UploadFile = File(...)):
    """Analyze an uploaded LoRA safetensors file."""
    analyze_lora_file, _, diagnose, summary_stats = _get_analytics()

    if not file.filename or not file.filename.endswith(".safetensors"):
        raise HTTPException(400, "Only .safetensors files are supported")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        metrics = analyze_lora_file(tmp.name)

    return {
        "layers": metrics,
        "summary": summary_stats(metrics),
        "diagnostics": diagnose(metrics),
        "num_layers": len(metrics),
        "filename": file.filename,
    }


@router.post("/api/lora/compare-upload")
async def compare_lora_upload(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    """Compare two uploaded LoRA safetensors files."""
    (analyze_lora_file, compare_lora_files,
     diagnose, summary_stats) = _get_analytics()

    for f in (file_a, file_b):
        if not f.filename or not f.filename.endswith(".safetensors"):
            raise HTTPException(400, "Only .safetensors files are supported")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmp_a, \
         tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmp_b:
        tmp_a.write(await file_a.read())
        tmp_a.flush()
        tmp_b.write(await file_b.read())
        tmp_b.flush()

        comparison = compare_lora_files(tmp_a.name, tmp_b.name)
        m1 = analyze_lora_file(tmp_a.name)
        m2 = analyze_lora_file(tmp_b.name)

    return {
        "layers": comparison,
        "summary_a": summary_stats(m1),
        "summary_b": summary_stats(m2),
        "diagnostics": diagnose(m1) + diagnose(m2),
        "num_layers": len(comparison),
        "filename_a": file_a.filename,
        "filename_b": file_b.filename,
    }
