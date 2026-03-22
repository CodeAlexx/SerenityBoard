"""End-to-end integration tests: SummaryWriter -> board.db -> FastAPI -> API responses.

Each test simulates a real training run by writing metrics via SummaryWriter,
then verifying the full round-trip through the FastAPI application endpoints.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile

import numpy as np
import pytest
from starlette.testclient import TestClient
from unittest.mock import patch

from serenityboard.server.app import create_app
from serenityboard.server.data_provider import RunDataProvider
from serenityboard.writer.summary_writer import SummaryWriter


# ---------------------------------------------------------------------------
# Thread-safety patch for TestClient
# ---------------------------------------------------------------------------
# Starlette's TestClient runs the ASGI app in a background thread, but
# RunDataProvider opens sqlite3 connections with the default
# check_same_thread=True.  We patch __init__ for tests only.

def _thread_safe_rdp_init(self, db_path: str) -> None:
    """Same as RunDataProvider.__init__ but with check_same_thread=False."""
    self._db_path = db_path
    self.run_dir = os.path.dirname(db_path)
    self._conn = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
    )
    self._conn.execute("PRAGMA query_only = ON")
    self._last_seen: dict[str, int] = {}
    self._known_session_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_writer(logdir: str, run_name: str, **kwargs) -> SummaryWriter:
    """Create a SummaryWriter in the given logdir with the specified run_name."""
    return SummaryWriter(logdir, run_name=run_name, **kwargs)


def _make_app_client(logdir: str) -> TestClient:
    """Build a TestClient from a logdir, with thread-safety patch applied."""
    app = create_app(logdir)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullTrainingRun:
    """End-to-end: write metrics via SummaryWriter, start app, query API, verify."""

    def test_full_training_run(self):
        logdir = tempfile.mkdtemp(prefix="sb_integration_")
        try:
            # 1. Create SummaryWriter with hparams
            w = _make_writer(
                logdir,
                "integration_run",
                hparams={"lr": 1e-4, "batch_size": 32, "model": "resnet50"},
            )

            # 2. Write 100 scalar points for loss, lr, grad_norm
            for step in range(100):
                w.add_scalar("loss/train", 1.0 - step * 0.008, step)
                w.add_scalar("lr", 1e-4 * (1 - step / 100), step)
                w.add_scalar("grad_norm", np.random.default_rng(step).random() * 2, step)

            # 3. Write 3 images at steps 25, 50, 75
            rng = np.random.default_rng(42)
            for step in [25, 50, 75]:
                img = rng.random((3, 64, 64), dtype=np.float32)
                w.add_image("samples", img, step)

            # 4. Write 1 histogram at step 50
            hist_values = rng.standard_normal(1000)
            w.add_histogram("weights/layer1", hist_values, step=50)

            # 5. Flush and close
            w.flush()
            w.close()

            # 6. Start FastAPI app
            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # 7. GET /api/runs - verify 1 run with status "complete"
                resp = client.get("/api/runs")
                assert resp.status_code == 200
                runs = resp.json()
                assert len(runs) == 1
                assert runs[0]["name"] == "integration_run"
                assert runs[0]["status"] == "complete"

                run_name = runs[0]["name"]

                # 8. GET /api/runs/{run}/tags - verify all expected tags
                resp = client.get(f"/api/runs/{run_name}/tags")
                assert resp.status_code == 200
                tags = resp.json()
                assert "loss/train" in tags["scalars"]
                assert "lr" in tags["scalars"]
                assert "grad_norm" in tags["scalars"]

                # 9. GET scalars for loss tag - verify 100 points (downsample=0 for full res)
                resp = client.get(
                    f"/api/runs/{run_name}/scalars",
                    params={"tag": "loss/train", "downsample": 0},
                )
                assert resp.status_code == 200
                scalars = resp.json()
                assert len(scalars) == 100
                # First point: step=0, value=1.0
                assert scalars[0][0] == 0
                assert scalars[0][2] == pytest.approx(1.0)
                # Last point: step=99, value=1.0 - 99*0.008 = 0.208
                assert scalars[-1][0] == 99
                assert scalars[-1][2] == pytest.approx(0.208, abs=1e-6)

                # 10. GET artifacts for samples tag - verify 3 images
                resp = client.get(
                    f"/api/runs/{run_name}/artifacts", params={"tag": "samples"}
                )
                assert resp.status_code == 200
                artifacts = resp.json()
                assert len(artifacts) == 3
                steps_returned = [a["step"] for a in artifacts]
                assert steps_returned == [25, 50, 75]
                # Each artifact should have a blob_key and image dimensions
                for a in artifacts:
                    assert "blob_key" in a
                    assert a["width"] == 64
                    assert a["height"] == 64
                    assert a["kind"] == "image"

                # 11. Downsampling: request scalars with downsample=10
                resp = client.get(
                    f"/api/runs/{run_name}/scalars",
                    params={"tag": "loss/train", "downsample": 10},
                )
                assert resp.status_code == 200
                sampled = resp.json()
                assert len(sampled) <= 10
                # Must include first and last steps
                sampled_steps = [s[0] for s in sampled]
                assert 0 in sampled_steps
                assert 99 in sampled_steps

                # 12. DELETE the run
                resp = client.delete(f"/api/runs/{run_name}")
                assert resp.status_code == 200
                assert resp.json()["deleted"] == run_name

                # Verify it's gone
                resp = client.get("/api/runs")
                assert resp.status_code == 200
                assert len(resp.json()) == 0

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestHparamsRoundtrip:
    """Write hparams via SummaryWriter, read back via /api/runs/{run}/hparams."""

    def test_hparams_roundtrip(self):
        logdir = tempfile.mkdtemp(prefix="sb_hparams_")
        try:
            hparams = {
                "lr": 3e-5,
                "batch_size": 16,
                "optimizer": "adamw",
                "warmup_steps": 500,
            }
            w = _make_writer(logdir, "hparams_run", hparams=hparams)
            # Write a dummy scalar so the run has data
            w.add_scalar("loss", 0.5, 0)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                resp = client.get("/api/runs/hparams_run/hparams")
                assert resp.status_code == 200
                data = resp.json()
                assert "hparams" in data
                hp = data["hparams"]
                assert hp["lr"] == pytest.approx(3e-5)
                assert hp["batch_size"] == 16
                assert hp["optimizer"] == "adamw"
                assert hp["warmup_steps"] == 500

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestHistogramsRoundtrip:
    """Write histogram data via SummaryWriter, read via /api/runs/{run}/histograms."""

    def test_histograms_roundtrip(self):
        logdir = tempfile.mkdtemp(prefix="sb_hist_")
        try:
            w = _make_writer(logdir, "hist_run")
            rng = np.random.default_rng(99)
            # Write histograms at 3 steps
            for step in [10, 20, 30]:
                values = rng.standard_normal(500)
                w.add_histogram("weights/fc1", values, step=step)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Verify tag discovery
                resp = client.get("/api/runs/hist_run/tags")
                assert resp.status_code == 200
                tags = resp.json()
                assert "weights/fc1" in tags["tensors"]

                # Fetch histogram data
                resp = client.get(
                    "/api/runs/hist_run/histograms",
                    params={"tag": "weights/fc1"},
                )
                assert resp.status_code == 200
                histograms = resp.json()
                assert len(histograms) == 3
                steps = [h["step"] for h in histograms]
                assert steps == [10, 20, 30]
                # Each histogram should have bins (list of [left, right, count])
                for h in histograms:
                    assert "bins" in h
                    assert "wall_time" in h
                    bins = h["bins"]
                    assert len(bins) == 64  # default bin count
                    # Each bin is [left_edge, right_edge, count]
                    assert len(bins[0]) == 3
                    # Counts should be non-negative
                    for b in bins:
                        assert b[2] >= 0

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestTextRoundtrip:
    """Write text events via SummaryWriter, read via /api/runs/{run}/text."""

    def test_text_roundtrip(self):
        logdir = tempfile.mkdtemp(prefix="sb_text_")
        try:
            w = _make_writer(logdir, "text_run")
            w.add_text("config", '{"lr": 1e-4}', step=0)
            w.add_text("config", '{"lr": 5e-5}', step=10)
            w.add_text("notes", "Training started", step=0)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Verify tag discovery
                resp = client.get("/api/runs/text_run/tags")
                assert resp.status_code == 200
                tags = resp.json()
                assert "config" in tags["text_events"]
                assert "notes" in tags["text_events"]

                # Fetch text data for config tag
                resp = client.get(
                    "/api/runs/text_run/text", params={"tag": "config"}
                )
                assert resp.status_code == 200
                texts = resp.json()
                assert len(texts) == 2
                assert texts[0]["step"] == 0
                assert texts[0]["value"] == '{"lr": 1e-4}'
                assert texts[1]["step"] == 10
                assert texts[1]["value"] == '{"lr": 5e-5}'

                # Fetch with limit=1
                resp = client.get(
                    "/api/runs/text_run/text", params={"tag": "config", "limit": 1}
                )
                assert resp.status_code == 200
                assert len(resp.json()) == 1

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestTracesRoundtrip:
    """Write trace events via SummaryWriter, read via /api/runs/{run}/traces."""

    def test_traces_roundtrip(self):
        logdir = tempfile.mkdtemp(prefix="sb_traces_")
        try:
            w = _make_writer(logdir, "trace_run")
            w.add_trace(step=0, phase="forward", duration_ms=12.5)
            w.add_trace(
                step=0,
                phase="backward",
                duration_ms=25.0,
                details={"grad_norm": 1.2},
            )
            w.add_trace(step=1, phase="forward", duration_ms=11.8)
            w.add_trace(step=1, phase="backward", duration_ms=24.3)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Verify tag discovery (traces use phases)
                resp = client.get("/api/runs/trace_run/tags")
                assert resp.status_code == 200
                tags = resp.json()
                assert "forward" in tags["trace_events"]
                assert "backward" in tags["trace_events"]

                # Fetch all traces
                resp = client.get("/api/runs/trace_run/traces")
                assert resp.status_code == 200
                traces = resp.json()
                assert len(traces) == 4

                # Each trace should have step, phase, duration_ms
                for t in traces:
                    assert "step" in t
                    assert "phase" in t
                    assert "duration_ms" in t

                # Verify details round-trip
                backward_step0 = [
                    t for t in traces if t["step"] == 0 and t["phase"] == "backward"
                ]
                assert len(backward_step0) == 1
                assert backward_step0[0]["details"]["grad_norm"] == pytest.approx(1.2)

                # Fetch with step range filter
                resp = client.get(
                    "/api/runs/trace_run/traces",
                    params={"step_from": 1, "step_to": 1},
                )
                assert resp.status_code == 200
                filtered = resp.json()
                assert len(filtered) == 2
                assert all(t["step"] == 1 for t in filtered)

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestEvalRoundtrip:
    """Write eval results via SummaryWriter, read via /api/runs/{run}/eval."""

    def test_eval_roundtrip(self):
        logdir = tempfile.mkdtemp(prefix="sb_eval_")
        try:
            w = _make_writer(logdir, "eval_run")
            # Write FID scores at different steps
            w.add_eval("fid", "case_0", step=100, score_name="fid_score", score_value=45.2)
            w.add_eval("fid", "case_0", step=200, score_name="fid_score", score_value=32.1)
            w.add_eval(
                "fid",
                "case_1",
                step=100,
                score_name="fid_score",
                score_value=50.0,
                details={"num_samples": 1000},
            )
            # Write CLIP score
            w.add_eval("clip", "case_0", step=100, score_name="clip_score", score_value=0.85)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Verify tag discovery (eval uses suite names)
                resp = client.get("/api/runs/eval_run/tags")
                assert resp.status_code == 200
                tags = resp.json()
                assert "fid" in tags["eval_suites"]
                assert "clip" in tags["eval_suites"]

                # Fetch all FID evals
                resp = client.get(
                    "/api/runs/eval_run/eval", params={"suite": "fid"}
                )
                assert resp.status_code == 200
                evals = resp.json()
                assert len(evals) == 3
                # Verify structure
                for e in evals:
                    assert "suite_name" in e
                    assert "case_id" in e
                    assert "score_name" in e
                    assert "score_value" in e
                    assert e["suite_name"] == "fid"

                # Filter by step
                resp = client.get(
                    "/api/runs/eval_run/eval",
                    params={"suite": "fid", "step": 100},
                )
                assert resp.status_code == 200
                step100 = resp.json()
                assert len(step100) == 2  # case_0 and case_1 at step 100

                # Verify details round-trip
                case1 = [e for e in step100 if e["case_id"] == "case_1"]
                assert len(case1) == 1
                assert case1[0]["details"]["num_samples"] == 1000

                # Fetch CLIP eval
                resp = client.get(
                    "/api/runs/eval_run/eval", params={"suite": "clip"}
                )
                assert resp.status_code == 200
                clip_evals = resp.json()
                assert len(clip_evals) == 1
                assert clip_evals[0]["score_value"] == pytest.approx(0.85)

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestCompareScalars:
    """Write same tag in 2 runs, use /api/compare/scalars to compare them."""

    def test_compare_scalars(self):
        logdir = tempfile.mkdtemp(prefix="sb_compare_")
        try:
            # Run A: lower loss curve
            w1 = _make_writer(logdir, "run_a")
            for step in range(50):
                w1.add_scalar("loss/train", 1.0 - step * 0.015, step)
            w1.flush()
            w1.close()

            # Run B: higher loss curve
            w2 = _make_writer(logdir, "run_b")
            for step in range(50):
                w2.add_scalar("loss/train", 1.5 - step * 0.020, step)
            w2.flush()
            w2.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Verify both runs exist
                resp = client.get("/api/runs")
                assert resp.status_code == 200
                runs = resp.json()
                names = {r["name"] for r in runs}
                assert names == {"run_a", "run_b"}

                # Compare scalars
                resp = client.get(
                    "/api/compare/scalars",
                    params={"tag": "loss/train", "runs": "run_a,run_b"},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert "run_a" in data
                assert "run_b" in data
                assert len(data["run_a"]) == 50
                assert len(data["run_b"]) == 50

                # Verify values differ between runs
                # run_a step 0: 1.0, run_b step 0: 1.5
                assert data["run_a"][0][2] == pytest.approx(1.0)
                assert data["run_b"][0][2] == pytest.approx(1.5)

                # Compare with wall_time x-axis
                resp = client.get(
                    "/api/compare/scalars",
                    params={
                        "tag": "loss/train",
                        "runs": "run_a,run_b",
                        "x_axis": "wall_time",
                    },
                )
                assert resp.status_code == 200
                data_wt = resp.json()
                # x should equal wall_time
                assert data_wt["run_a"][0][0] == data_wt["run_a"][0][1]

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestNonexistentRun404:
    """Verify 404 responses for nonexistent runs across all endpoints."""

    def test_nonexistent_run_404(self):
        logdir = tempfile.mkdtemp(prefix="sb_404_")
        try:
            # Create one valid run so the app starts properly
            w = _make_writer(logdir, "valid_run")
            w.add_scalar("loss", 0.5, 0)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)
                fake = "nonexistent_run"

                # Tags endpoint
                resp = client.get(f"/api/runs/{fake}/tags")
                assert resp.status_code == 404

                # Scalars endpoint
                resp = client.get(
                    f"/api/runs/{fake}/scalars", params={"tag": "loss"}
                )
                assert resp.status_code == 404

                # Artifacts endpoint
                resp = client.get(
                    f"/api/runs/{fake}/artifacts", params={"tag": "samples"}
                )
                assert resp.status_code == 404

                # Histograms endpoint
                resp = client.get(
                    f"/api/runs/{fake}/histograms", params={"tag": "w"}
                )
                assert resp.status_code == 404

                # Text endpoint
                resp = client.get(
                    f"/api/runs/{fake}/text", params={"tag": "t"}
                )
                assert resp.status_code == 404

                # Traces endpoint
                resp = client.get(f"/api/runs/{fake}/traces")
                assert resp.status_code == 404

                # Eval endpoint
                resp = client.get(
                    f"/api/runs/{fake}/eval", params={"suite": "fid"}
                )
                assert resp.status_code == 404

                # Hparams endpoint
                resp = client.get(f"/api/runs/{fake}/hparams")
                assert resp.status_code == 404

                # Delete endpoint
                resp = client.delete(f"/api/runs/{fake}")
                assert resp.status_code == 404

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestDownsamplingBehavior:
    """Write 1000 scalar points, verify downsample=50 returns <=50 points."""

    def test_downsampling_behavior(self):
        logdir = tempfile.mkdtemp(prefix="sb_downsample_")
        try:
            w = _make_writer(logdir, "ds_run")
            for step in range(1000):
                w.add_scalar("loss/train", 1.0 / (step + 1), step)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Full resolution
                resp = client.get(
                    "/api/runs/ds_run/scalars",
                    params={"tag": "loss/train", "downsample": 0},
                )
                assert resp.status_code == 200
                full = resp.json()
                assert len(full) == 1000

                # Downsample to 50
                resp = client.get(
                    "/api/runs/ds_run/scalars",
                    params={"tag": "loss/train", "downsample": 50},
                )
                assert resp.status_code == 200
                sampled = resp.json()
                assert len(sampled) <= 50
                # Must include first and last
                steps = [s[0] for s in sampled]
                assert steps[0] == 0
                assert steps[-1] == 999

                # Downsample to 10
                resp = client.get(
                    "/api/runs/ds_run/scalars",
                    params={"tag": "loss/train", "downsample": 10},
                )
                assert resp.status_code == 200
                sampled10 = resp.json()
                assert len(sampled10) <= 10
                assert len(sampled10) >= 3  # at least first, middle, last

                # Downsample to 5000 (more than data) returns all
                resp = client.get(
                    "/api/runs/ds_run/scalars",
                    params={"tag": "loss/train", "downsample": 5000},
                )
                assert resp.status_code == 200
                all_data = resp.json()
                assert len(all_data) == 1000

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestMultipleRuns:
    """Create 3 runs, verify /api/runs lists all 3 with correct metadata."""

    def test_multiple_runs(self):
        logdir = tempfile.mkdtemp(prefix="sb_multi_")
        try:
            run_configs = [
                ("exp_lr1e3", {"lr": 1e-3, "model": "vit_base"}),
                ("exp_lr1e4", {"lr": 1e-4, "model": "vit_base"}),
                ("exp_lr1e5", {"lr": 1e-5, "model": "vit_large"}),
            ]

            for run_name, hparams in run_configs:
                w = _make_writer(logdir, run_name, hparams=hparams)
                for step in range(20):
                    w.add_scalar("loss/train", 1.0 - step * 0.04, step)
                w.flush()
                w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Verify all 3 runs are listed
                resp = client.get("/api/runs")
                assert resp.status_code == 200
                runs = resp.json()
                assert len(runs) == 3
                names = {r["name"] for r in runs}
                assert names == {"exp_lr1e3", "exp_lr1e4", "exp_lr1e5"}

                # All should be complete
                for r in runs:
                    assert r["status"] == "complete"
                    assert r["start_time"] is not None

                # Check hparams were written correctly for each run
                for run_name, expected_hp in run_configs:
                    resp = client.get(f"/api/runs/{run_name}/hparams")
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["hparams"]["lr"] == pytest.approx(expected_hp["lr"])
                    assert data["hparams"]["model"] == expected_hp["model"]

                # Each run should have the same tags
                for run_name, _ in run_configs:
                    resp = client.get(f"/api/runs/{run_name}/tags")
                    assert resp.status_code == 200
                    tags = resp.json()
                    assert "loss/train" in tags["scalars"]

                # Scalars for each run should have 20 points
                for run_name, _ in run_configs:
                    resp = client.get(
                        f"/api/runs/{run_name}/scalars",
                        params={"tag": "loss/train", "downsample": 0},
                    )
                    assert resp.status_code == 200
                    assert len(resp.json()) == 20

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestCompareEval:
    """Write eval results across runs, compare via /api/compare/eval."""

    def test_compare_eval(self):
        logdir = tempfile.mkdtemp(prefix="sb_compare_eval_")
        try:
            w1 = _make_writer(logdir, "eval_a")
            w1.add_eval("fid", "case_0", step=100, score_name="fid_score", score_value=42.0)
            w1.flush()
            w1.close()

            w2 = _make_writer(logdir, "eval_b")
            w2.add_eval("fid", "case_0", step=100, score_name="fid_score", score_value=38.5)
            w2.flush()
            w2.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                resp = client.get(
                    "/api/compare/eval",
                    params={"suite": "fid", "runs": "eval_a,eval_b"},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert "eval_a" in data
                assert "eval_b" in data
                assert len(data["eval_a"]) == 1
                assert len(data["eval_b"]) == 1
                assert data["eval_a"][0]["score_value"] == pytest.approx(42.0)
                assert data["eval_b"][0]["score_value"] == pytest.approx(38.5)

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestScalarValueAccuracy:
    """Verify scalar values survive the full round-trip with float precision."""

    def test_scalar_value_accuracy(self):
        logdir = tempfile.mkdtemp(prefix="sb_accuracy_")
        try:
            w = _make_writer(logdir, "accuracy_run")
            # Write known values at specific steps
            expected = {
                0: 1.0,
                1: 0.999,
                2: 0.5,
                3: 0.001,
                4: 1e-8,
                5: 0.0,
                6: -0.5,
                7: 100.0,
                8: 1e6,
            }
            for step, value in expected.items():
                w.add_scalar("precise_loss", value, step)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                resp = client.get(
                    "/api/runs/accuracy_run/scalars",
                    params={"tag": "precise_loss", "downsample": 0},
                )
                assert resp.status_code == 200
                scalars = resp.json()
                assert len(scalars) == len(expected)
                for row in scalars:
                    step = row[0]
                    value = row[2]
                    assert value == pytest.approx(expected[step], rel=1e-7, abs=1e-12)

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestImageArtifactBlobIntegrity:
    """Verify that image blobs written by SummaryWriter exist on disk."""

    def test_image_blob_exists(self):
        logdir = tempfile.mkdtemp(prefix="sb_blob_")
        try:
            w = _make_writer(logdir, "blob_run")
            rng = np.random.default_rng(7)
            img = rng.random((3, 32, 32), dtype=np.float32)
            w.add_image("sample_img", img, step=0)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                resp = client.get(
                    "/api/runs/blob_run/artifacts", params={"tag": "sample_img"}
                )
                assert resp.status_code == 200
                artifacts = resp.json()
                assert len(artifacts) == 1
                blob_key = artifacts[0]["blob_key"]

                # Verify the blob file exists on disk
                blobs_dir = os.path.join(logdir, "blob_run", "blobs")
                blob_path = os.path.join(blobs_dir, blob_key)
                assert os.path.isfile(blob_path)
                # Verify it's a valid PNG (starts with PNG magic bytes)
                with open(blob_path, "rb") as f:
                    header = f.read(8)
                assert header[:4] == b"\x89PNG"

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestEmptyTagQuery:
    """Query scalars for a tag that exists but has no data should return empty list."""

    def test_empty_tag_query(self):
        logdir = tempfile.mkdtemp(prefix="sb_empty_")
        try:
            w = _make_writer(logdir, "empty_run")
            w.add_scalar("loss", 0.5, 0)
            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Query a tag that does not exist
                resp = client.get(
                    "/api/runs/empty_run/scalars",
                    params={"tag": "nonexistent_tag", "downsample": 0},
                )
                assert resp.status_code == 200
                assert resp.json() == []

                # Query artifacts for a tag that does not exist
                resp = client.get(
                    "/api/runs/empty_run/artifacts",
                    params={"tag": "nonexistent_tag"},
                )
                assert resp.status_code == 200
                assert resp.json() == []

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestContextManagerWriter:
    """Verify SummaryWriter works as a context manager and closes cleanly."""

    def test_context_manager(self):
        logdir = tempfile.mkdtemp(prefix="sb_ctx_")
        try:
            with SummaryWriter(logdir, run_name="ctx_run") as w:
                for step in range(10):
                    w.add_scalar("loss", 1.0 / (step + 1), step)

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                resp = client.get("/api/runs")
                assert resp.status_code == 200
                runs = resp.json()
                assert len(runs) == 1
                assert runs[0]["name"] == "ctx_run"
                assert runs[0]["status"] == "complete"

                resp = client.get(
                    "/api/runs/ctx_run/scalars",
                    params={"tag": "loss", "downsample": 0},
                )
                assert resp.status_code == 200
                assert len(resp.json()) == 10

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestMixedDataTypes:
    """Write scalars, images, text, histograms, traces, eval all in one run."""

    def test_mixed_data_types(self):
        logdir = tempfile.mkdtemp(prefix="sb_mixed_")
        try:
            w = _make_writer(logdir, "mixed_run")
            rng = np.random.default_rng(0)

            # Scalars
            for step in range(50):
                w.add_scalar("loss/train", 1.0 - step * 0.01, step)
                w.add_scalar("loss/val", 1.2 - step * 0.012, step)

            # Images
            for step in [10, 30]:
                img = rng.random((3, 32, 32), dtype=np.float32)
                w.add_image("generated", img, step)

            # Text
            w.add_text("logs", "Epoch 1 complete", step=10)
            w.add_text("logs", "Epoch 2 complete", step=20)

            # Histograms
            w.add_histogram("grads/conv1", rng.standard_normal(200), step=25)

            # Traces
            w.add_trace(step=5, phase="forward", duration_ms=15.0)
            w.add_trace(step=5, phase="backward", duration_ms=30.0)

            # Eval
            w.add_eval("quality", "sample_0", step=50, score_name="psnr", score_value=28.5)

            w.flush()
            w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Tags should show everything
                resp = client.get("/api/runs/mixed_run/tags")
                assert resp.status_code == 200
                tags = resp.json()
                assert "loss/train" in tags["scalars"]
                assert "loss/val" in tags["scalars"]
                assert "generated" in tags["artifacts"]
                assert "logs" in tags["text_events"]
                assert "grads/conv1" in tags["tensors"]
                assert "forward" in tags["trace_events"]
                assert "backward" in tags["trace_events"]
                assert "quality" in tags["eval_suites"]

                # Verify counts for scalars
                resp = client.get(
                    "/api/runs/mixed_run/scalars",
                    params={"tag": "loss/train", "downsample": 0},
                )
                assert resp.status_code == 200
                assert len(resp.json()) == 50

                resp = client.get(
                    "/api/runs/mixed_run/scalars",
                    params={"tag": "loss/val", "downsample": 0},
                )
                assert resp.status_code == 200
                assert len(resp.json()) == 50

                # Verify images
                resp = client.get(
                    "/api/runs/mixed_run/artifacts", params={"tag": "generated"}
                )
                assert resp.status_code == 200
                assert len(resp.json()) == 2

                # Verify text
                resp = client.get(
                    "/api/runs/mixed_run/text", params={"tag": "logs"}
                )
                assert resp.status_code == 200
                assert len(resp.json()) == 2

                # Verify histograms
                resp = client.get(
                    "/api/runs/mixed_run/histograms",
                    params={"tag": "grads/conv1"},
                )
                assert resp.status_code == 200
                assert len(resp.json()) == 1

                # Verify traces
                resp = client.get("/api/runs/mixed_run/traces")
                assert resp.status_code == 200
                assert len(resp.json()) == 2

                # Verify eval
                resp = client.get(
                    "/api/runs/mixed_run/eval", params={"suite": "quality"}
                )
                assert resp.status_code == 200
                evals = resp.json()
                assert len(evals) == 1
                assert evals[0]["score_value"] == pytest.approx(28.5)

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestDeleteAndRequery:
    """Delete a run mid-session, verify all endpoints reflect the deletion."""

    def test_delete_and_requery(self):
        logdir = tempfile.mkdtemp(prefix="sb_delreq_")
        try:
            # Create two runs
            for name in ("keep_run", "delete_me"):
                w = _make_writer(logdir, name)
                w.add_scalar("loss", 0.5, 0)
                w.flush()
                w.close()

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                # Both exist
                resp = client.get("/api/runs")
                assert len(resp.json()) == 2

                # Delete one
                resp = client.delete("/api/runs/delete_me")
                assert resp.status_code == 200

                # Only one remains
                resp = client.get("/api/runs")
                runs = resp.json()
                assert len(runs) == 1
                assert runs[0]["name"] == "keep_run"

                # Deleted run returns 404
                resp = client.get("/api/runs/delete_me/tags")
                assert resp.status_code == 404

                # Remaining run still works
                resp = client.get(
                    "/api/runs/keep_run/scalars",
                    params={"tag": "loss", "downsample": 0},
                )
                assert resp.status_code == 200
                assert len(resp.json()) == 1

        finally:
            shutil.rmtree(logdir, ignore_errors=True)


class TestRunDirectoryCleanup:
    """Verify DELETE /api/runs/{run} removes the directory from disk."""

    def test_directory_removed_on_delete(self):
        logdir = tempfile.mkdtemp(prefix="sb_cleanup_")
        try:
            w = _make_writer(logdir, "doomed_run")
            w.add_scalar("loss", 0.5, 0)
            w.flush()
            w.close()

            run_dir = os.path.join(logdir, "doomed_run")
            assert os.path.isdir(run_dir)

            with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
                client = _make_app_client(logdir)

                resp = client.delete("/api/runs/doomed_run")
                assert resp.status_code == 200

                # Directory should be gone
                assert not os.path.exists(run_dir)

        finally:
            shutil.rmtree(logdir, ignore_errors=True)
