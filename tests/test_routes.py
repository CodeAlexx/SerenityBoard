"""Comprehensive tests for all SerenityBoard API sub-router endpoints."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
import wave
from unittest.mock import patch

import numpy as np
import pytest
from starlette.testclient import TestClient

from serenityboard.server.app import create_app
from serenityboard.server.data_provider import RunDataProvider
from serenityboard.writer.schema import create_tables, set_pragmas


# ---------------------------------------------------------------------------
# Thread-safety patch for TestClient
# ---------------------------------------------------------------------------
# Starlette's TestClient runs the ASGI app in a background thread, but
# RunDataProvider opens sqlite3 connections with the default
# check_same_thread=True.  We patch __init__ for tests only.

_orig_rdp_init = RunDataProvider.__init__


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
    self._audio_analysis_cache: dict[tuple[str, int, int, int], dict[str, object] | None] = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(sample_run_dir):
    """Create a logdir containing a copy of sample_run_dir, build the app,
    yield (TestClient, run_name).

    The sample_run_dir fixture (from conftest.py) creates a temp directory
    with board.db + blobs/ already populated.  We copy it into a fresh
    parent logdir so create_app discovers exactly one run.
    (os.walk in _find_run_dbs does not follow symlinks, so we copy.)
    """
    logdir = tempfile.mkdtemp(prefix="sb_logdir_")
    run_name = "test_run"
    shutil.copytree(sample_run_dir, os.path.join(logdir, run_name))

    with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
        app = create_app(logdir)
        tc = TestClient(app, raise_server_exceptions=False)
        yield tc, run_name

    shutil.rmtree(logdir, ignore_errors=True)


@pytest.fixture()
def rich_client(sample_run_dir):
    """Like ``client`` but injects additional data types into the database
    that are absent from the base sample_db: audio, graphs, meshes,
    embeddings, hparam_metrics, pr_curves, and custom_scalar_layouts.

    Yields (TestClient, run_name).
    """
    db_path = os.path.join(sample_run_dir, "board.db")
    now = time.time()

    # Open in read-write mode to insert extra rows.
    conn = sqlite3.connect(db_path)

    with conn:
        # -- audio --
        conn.execute(
            "INSERT INTO audio "
            "(tag, step, seq_index, wall_time, blob_key, sample_rate, "
            "num_channels, duration_ms, mime_type, label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("audio/speech", 3, 0, now - 570, "aabbccdd11223344.wav",
             22050, 1, 1500.0, "audio/wav", "sample speech"),
        )

        # -- graphs (inline JSON legacy) --
        graph_json = json.dumps({
            "nodes": [{"name": "conv1"}, {"name": "relu1"}],
            "edges": [{"source": "conv1", "target": "relu1"}],
        })
        conn.execute(
            "INSERT INTO graphs (tag, step, wall_time, graph_blob_key) "
            "VALUES (?, ?, ?, ?)",
            ("model/architecture", 0, now - 600, graph_json),
        )

        # -- meshes --
        conn.execute(
            "INSERT INTO meshes "
            "(tag, step, wall_time, num_vertices, has_faces, has_colors, "
            "num_faces, vertices_blob_key, faces_blob_key, colors_blob_key, config_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("mesh/pointcloud", 1, now - 590, 100, 0, 0, 0,
             "1122334455667788.bin", None, None, None),
        )

        # -- embeddings --
        conn.execute(
            "INSERT INTO embeddings "
            "(tag, step, wall_time, num_points, dimensions, "
            "tensor_blob_key, metadata_json, metadata_header, "
            "sprite_blob_key, sprite_single_h, sprite_single_w) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("emb/layer4", 2, now - 580, 50, 128,
             "aabb112233445566.bin", '["label1","label2"]',
             '["class"]', None, None, None),
        )

        # -- hparam_metrics --
        conn.execute(
            "INSERT INTO hparam_metrics (metric_tag, value, step, wall_time) "
            "VALUES (?, ?, ?, ?)",
            ("final_loss", 0.05, 19, now - 410),
        )
        conn.execute(
            "INSERT INTO hparam_metrics (metric_tag, value, step, wall_time) "
            "VALUES (?, ?, ?, ?)",
            ("accuracy", 0.98, 19, now - 410),
        )

        # -- hparams in metadata --
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("hparams", json.dumps({"lr": 1e-4, "batch_size": 32})),
        )

        # -- pr_curves --
        num_thresh = 5
        # 6 rows of num_thresh each: TP, FP, TN, FN, precision, recall
        pr_data = np.zeros((6, num_thresh), dtype=np.float64)
        pr_data[4, :] = [1.0, 0.9, 0.8, 0.7, 0.6]  # precision
        pr_data[5, :] = [0.2, 0.4, 0.6, 0.8, 1.0]   # recall
        conn.execute(
            "INSERT INTO pr_curves (tag, step, class_index, wall_time, "
            "num_thresholds, data) VALUES (?, ?, ?, ?, ?, ?)",
            ("pr/binary", 5, 0, now - 550, num_thresh, pr_data.tobytes()),
        )

        # -- custom_scalar_layouts --
        layout = {
            "categories": [
                {
                    "title": "Losses",
                    "charts": [
                        {"title": "Train Loss", "tag_regexes": ["loss/.*"]},
                    ],
                },
            ],
        }
        conn.execute(
            "INSERT OR REPLACE INTO custom_scalar_layouts (layout_name, config) "
            "VALUES (?, ?)",
            ("default", json.dumps(layout)),
        )

    conn.close()

    # Create a real WAV blob for audio route + blob route tests.
    blobs_dir = os.path.join(sample_run_dir, "blobs")
    wav_path = os.path.join(blobs_dir, "aabbccdd11223344.wav")
    sample_rate = 22050
    duration_s = 0.25
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    wave_data = (0.4 * np.sin(2 * np.pi * 220.0 * t) * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wave_data.tobytes())

    logdir = tempfile.mkdtemp(prefix="sb_richlogdir_")
    run_name = "rich_run"
    shutil.copytree(sample_run_dir, os.path.join(logdir, run_name))

    with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
        app = create_app(logdir)
        tc = TestClient(app, raise_server_exceptions=False)
        yield tc, run_name

    shutil.rmtree(logdir, ignore_errors=True)


# ===================================================================
# Top-level endpoints
# ===================================================================


class TestListRuns:
    def test_returns_run_list(self, client):
        tc, run_name = client
        resp = tc.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        names = {r["name"] for r in data}
        assert run_name in names

    def test_run_info_fields(self, client):
        tc, run_name = client
        resp = tc.get("/api/runs")
        data = resp.json()
        run = next(r for r in data if r["name"] == run_name)
        assert "start_time" in run
        assert "status" in run
        assert "last_activity" in run
        assert "last_step" in run


class TestGetTags:
    def test_returns_tag_groups(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert "scalars" in data
        assert "artifacts" in data
        assert "text_events" in data
        assert "trace_events" in data
        assert "eval_suites" in data
        assert "loss/train" in data["scalars"]
        assert "lr" in data["scalars"]
        assert "samples" in data["artifacts"]
        assert "config" in data["text_events"]

    def test_tags_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent_run_xyz/tags")
        assert resp.status_code == 404


class TestDeleteRun:
    def test_delete_existing_run(self, client):
        tc, run_name = client
        resp = tc.delete(f"/api/runs/{run_name}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] == run_name

        # Verify the run is gone from listing.
        resp = tc.get("/api/runs")
        names = {r["name"] for r in resp.json()}
        assert run_name not in names

    def test_delete_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.delete("/api/runs/nonexistent_run_xyz")
        assert resp.status_code == 404


# ===================================================================
# Scalars
# ===================================================================


class TestScalars:
    def test_get_scalars(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/scalars", params={"tag": "loss/train"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 20  # sample_db has 20 loss/train points
        # Each element is [step, wall_time, value].
        assert len(data[0]) == 3
        assert data[0][0] == 0  # first step

    def test_get_scalars_lr_tag(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/scalars", params={"tag": "lr"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 10

    def test_get_scalars_empty_tag(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/scalars", params={"tag": "nonexistent"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_scalars_wall_time_xaxis(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/scalars",
            params={"tag": "loss/train", "x_axis": "wall_time"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # x value (first element) should equal wall_time (second element).
        assert data[0][0] == data[0][1]

    def test_get_scalars_relative_xaxis(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/scalars",
            params={"tag": "loss/train", "x_axis": "relative"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data[0][0] == pytest.approx(0.0, abs=1e-6)
        assert data[1][0] > 0

    def test_scalars_last(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/scalars/last",
            params={"tags": "loss/train,lr"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "loss/train" in data
        assert "lr" in data
        assert data["loss/train"]["step"] == 19
        assert data["lr"]["step"] == 9

    def test_scalars_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/scalars", params={"tag": "loss/train"})
        assert resp.status_code == 404


# ===================================================================
# Images
# ===================================================================


class TestImages:
    def test_get_images(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/images", params={"tag": "samples"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["blob_key"] == "abc123.png"
        assert entry["width"] == 512
        assert entry["height"] == 512
        assert "step" in entry
        assert "wall_time" in entry

    def test_images_empty_tag(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/images", params={"tag": "nonexistent"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_images_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/images", params={"tag": "samples"})
        assert resp.status_code == 404


# ===================================================================
# Blob
# ===================================================================


class TestBlob:
    def test_get_blob_invalid_key_format(self, client):
        """Blob key must match ^[a-f0-9]{16}\\.[a-z0-9]+$ regex."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/blob/abc123.png")
        assert resp.status_code == 400

    def test_get_blob_not_found_valid_key(self, client):
        """Valid key format but blob does not exist in DB."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/blob/0000000000000000.png")
        assert resp.status_code == 404

    def test_get_blob_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/blob/0000000000000000.png")
        assert resp.status_code == 404

    def test_get_blob_success(self, rich_client):
        """Audio blob was inserted with a valid 16-hex-char key."""
        tc, run_name = rich_client
        resp = tc.get(f"/api/runs/{run_name}/blob/aabbccdd11223344.wav")
        assert resp.status_code == 200
        assert resp.content.startswith(b"RIFF")
        assert b"WAVE" in resp.content[:32]


# ===================================================================
# Histograms
# ===================================================================


class TestHistograms:
    def test_get_histograms(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/histograms",
            params={"tag": "weights/layer1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert "step" in entry
        assert "wall_time" in entry
        assert "bins" in entry
        assert isinstance(entry["bins"], list)

    def test_histograms_empty_tag(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/histograms",
            params={"tag": "nonexistent"},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_histograms_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/histograms", params={"tag": "x"})
        assert resp.status_code == 404


# ===================================================================
# Distributions
# ===================================================================


class TestDistributions:
    def test_get_distributions(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/distributions",
            params={"tag": "weights/layer1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        entry = data[0]
        assert "step" in entry
        assert "wall_time" in entry
        assert "percentiles" in entry
        # 9 basis points expected.
        assert len(entry["percentiles"]) == 9
        for p in entry["percentiles"]:
            assert "bp" in p
            assert "value" in p

    def test_distributions_empty_tag(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/distributions",
            params={"tag": "nonexistent"},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_distributions_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/distributions", params={"tag": "x"})
        assert resp.status_code == 404


# ===================================================================
# Text
# ===================================================================


class TestText:
    def test_get_text(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/text", params={"tag": "config"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        entry = data[0]
        assert "step" in entry
        assert "wall_time" in entry
        assert "value" in entry
        # The first text entry should be valid JSON.
        parsed = json.loads(entry["value"])
        assert "lr" in parsed

    def test_text_with_limit(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/text",
            params={"tag": "config", "limit": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_text_empty_tag(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/text", params={"tag": "nonexistent"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_text_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/text", params={"tag": "config"})
        assert resp.status_code == 404


# ===================================================================
# HParams
# ===================================================================


class TestHParams:
    def test_get_hparams_empty(self, client):
        """Base sample_db has no hparam metadata or hparam_metrics rows."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/hparams")
        assert resp.status_code == 200
        data = resp.json()
        assert "hparams" in data
        assert "metrics" in data
        assert isinstance(data["hparams"], dict)
        assert isinstance(data["metrics"], dict)

    def test_get_hparams_with_data(self, rich_client):
        """Rich DB has hparams metadata and hparam_metrics rows."""
        tc, run_name = rich_client
        resp = tc.get(f"/api/runs/{run_name}/hparams")
        assert resp.status_code == 200
        data = resp.json()
        assert data["hparams"]["lr"] == pytest.approx(1e-4)
        assert data["hparams"]["batch_size"] == 32
        assert data["metrics"]["final_loss"] == pytest.approx(0.05)
        assert data["metrics"]["accuracy"] == pytest.approx(0.98)

    def test_hparams_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/hparams")
        assert resp.status_code == 404


# ===================================================================
# Traces
# ===================================================================


class TestTraces:
    def test_get_traces(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        entry = data[0]
        assert "step" in entry
        assert "phase" in entry
        assert "duration_ms" in entry
        assert "details" in entry

    def test_traces_filter_by_step_range(self, client):
        tc, run_name = client
        # Only step 0.
        resp = tc.get(
            f"/api/runs/{run_name}/traces",
            params={"step_from": 0, "step_to": 0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["step"] == 0
        assert data[0]["phase"] == "forward"

    def test_traces_phases(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/traces")
        data = resp.json()
        phases = {e["phase"] for e in data}
        assert "forward" in phases
        assert "backward" in phases

    def test_traces_details_parsed(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/traces")
        data = resp.json()
        backward = next(e for e in data if e["phase"] == "backward")
        assert backward["details"]["grad_norm"] == pytest.approx(1.2)

    def test_traces_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/traces")
        assert resp.status_code == 404


# ===================================================================
# Eval
# ===================================================================


class TestEval:
    def test_get_eval(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/eval", params={"suite": "fid"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["suite_name"] == "fid"
        assert entry["score_name"] == "fid_score"
        assert entry["score_value"] == pytest.approx(42.5)

    def test_eval_empty_suite(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/eval", params={"suite": "nonexistent"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_eval_filter_by_step(self, client):
        tc, run_name = client
        # The eval result is at step 10.
        resp = tc.get(
            f"/api/runs/{run_name}/eval",
            params={"suite": "fid", "step": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        # Step that does not exist.
        resp = tc.get(
            f"/api/runs/{run_name}/eval",
            params={"suite": "fid", "step": 999},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_eval_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/eval", params={"suite": "fid"})
        assert resp.status_code == 404


# ===================================================================
# Audio
# ===================================================================


class TestAudio:
    def test_audio_empty_in_base_db(self, client):
        """Base sample_db has no audio rows."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/audio", params={"tag": "anything"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_audio_with_data(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(f"/api/runs/{run_name}/audio", params={"tag": "audio/speech"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["blob_key"] == "aabbccdd11223344.wav"
        assert entry["sample_rate"] == 22050
        assert entry["num_channels"] == 1
        assert entry["duration_ms"] == pytest.approx(1500.0)
        assert entry["mime_type"] == "audio/wav"
        assert entry["label"] == "sample speech"
        assert entry["tag"] == "audio/speech"
        assert isinstance(entry["waveform"], list)
        assert len(entry["waveform"]) > 0
        assert len(entry["waveform"][0]) == 2
        assert isinstance(entry["spectrogram"], list)
        assert len(entry["spectrogram"]) > 0
        assert isinstance(entry["peak_db"], float)
        assert isinstance(entry["rms_db"], float)

    def test_audio_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/audio", params={"tag": "x"})
        assert resp.status_code == 404


# ===================================================================
# Graphs
# ===================================================================


class TestGraphs:
    def test_graphs_empty_in_base_db(self, client):
        """Base sample_db has no graph rows."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/graphs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_graphs_with_data(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(f"/api/runs/{run_name}/graphs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["tag"] == "model/architecture"
        assert entry["step"] == 0
        assert "graph_data" in entry
        assert "nodes" in entry["graph_data"]
        assert "edges" in entry["graph_data"]

    def test_graphs_filter_by_tag(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(
            f"/api/runs/{run_name}/graphs",
            params={"tag": "model/architecture"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_graphs_nonexistent_tag(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(
            f"/api/runs/{run_name}/graphs",
            params={"tag": "nonexistent"},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_graphs_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/graphs")
        assert resp.status_code == 404


# ===================================================================
# Meshes
# ===================================================================


class TestMeshes:
    def test_meshes_empty_in_base_db(self, client):
        """Base sample_db has no mesh rows."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/meshes")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_meshes_with_data(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(f"/api/runs/{run_name}/meshes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["tag"] == "mesh/pointcloud"
        assert entry["step"] == 1
        assert entry["num_vertices"] == 100
        assert entry["has_faces"] is False
        assert entry["has_colors"] is False
        assert entry["vertices_blob_key"] == "1122334455667788.bin"

    def test_meshes_filter_by_tag(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(
            f"/api/runs/{run_name}/meshes",
            params={"tag": "mesh/pointcloud"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_meshes_specific_tag_and_step(self, rich_client):
        """When both tag and step are given, a single mesh dict is returned
        (not a list)."""
        tc, run_name = rich_client
        resp = tc.get(
            f"/api/runs/{run_name}/meshes",
            params={"tag": "mesh/pointcloud", "step": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Route returns the single entry directly (results[0]).
        assert isinstance(data, dict)
        assert data["tag"] == "mesh/pointcloud"

    def test_meshes_tag_and_step_not_found(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(
            f"/api/runs/{run_name}/meshes",
            params={"tag": "mesh/pointcloud", "step": 999},
        )
        assert resp.status_code == 404

    def test_meshes_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/meshes")
        assert resp.status_code == 404


# ===================================================================
# Embeddings
# ===================================================================


class TestEmbeddings:
    def test_embeddings_empty_in_base_db(self, client):
        """Base sample_db has no embedding rows."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/embeddings")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_embeddings_with_data(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(f"/api/runs/{run_name}/embeddings")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["tag"] == "emb/layer4"
        assert entry["step"] == 2
        assert entry["num_points"] == 50
        assert entry["dimensions"] == 128
        assert entry["tensor_blob_key"] == "aabb112233445566.bin"

    def test_embeddings_filter_by_tag(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(
            f"/api/runs/{run_name}/embeddings",
            params={"tag": "emb/layer4"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_embeddings_specific_tag_and_step(self, rich_client):
        """When both tag and step are given, a single embedding dict is returned."""
        tc, run_name = rich_client
        resp = tc.get(
            f"/api/runs/{run_name}/embeddings",
            params={"tag": "emb/layer4", "step": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert data["tag"] == "emb/layer4"

    def test_embeddings_tag_and_step_not_found(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(
            f"/api/runs/{run_name}/embeddings",
            params={"tag": "emb/layer4", "step": 999},
        )
        assert resp.status_code == 404

    def test_embeddings_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/embeddings")
        assert resp.status_code == 404


# ===================================================================
# Artifacts
# ===================================================================


class TestArtifacts:
    def test_get_artifacts(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/artifacts", params={"tag": "samples"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["blob_key"] == "abc123.png"
        assert entry["kind"] == "image"
        assert entry["mime_type"] == "image/png"
        assert entry["step"] == 5
        assert entry["width"] == 512
        assert entry["height"] == 512

    def test_artifacts_with_kind_filter(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/artifacts",
            params={"tag": "samples", "kind": "image"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

        # Non-matching kind should return empty.
        resp = tc.get(
            f"/api/runs/{run_name}/artifacts",
            params={"tag": "samples", "kind": "video"},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_artifacts_empty_tag(self, client):
        tc, run_name = client
        resp = tc.get(
            f"/api/runs/{run_name}/artifacts",
            params={"tag": "nonexistent"},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_artifacts_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/artifacts", params={"tag": "samples"})
        assert resp.status_code == 404


# ===================================================================
# PR Curves
# ===================================================================


class TestPRCurves:
    def test_pr_curves_empty_in_base_db(self, client):
        """Base sample_db has no PR curve rows."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/pr-curves", params={"tag": "pr/binary"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_pr_curves_with_data(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(f"/api/runs/{run_name}/pr-curves", params={"tag": "pr/binary"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["step"] == 5
        assert entry["class_index"] == 0
        assert entry["num_thresholds"] == 5
        assert "precision" in entry
        assert "recall" in entry
        assert "thresholds" in entry
        assert len(entry["precision"]) == 5
        assert len(entry["recall"]) == 5

    def test_pr_curves_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/pr-curves", params={"tag": "x"})
        assert resp.status_code == 404


# ===================================================================
# Custom Scalars
# ===================================================================


class TestCustomScalars:
    def test_custom_scalars_layout_empty(self, client):
        """Base sample_db has no custom scalar layouts."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/custom-scalars/layout")
        assert resp.status_code == 200
        data = resp.json()
        assert "categories" in data
        assert data["categories"] == []

    def test_custom_scalars_layout_with_data(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(f"/api/runs/{run_name}/custom-scalars/layout")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["categories"]) == 1
        assert data["categories"][0]["title"] == "Losses"

    def test_custom_scalars_data(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(
            f"/api/runs/{run_name}/custom-scalars/data",
            params={"tags": "loss/.*"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "loss/train" in data
        assert len(data["loss/train"]) > 0

    def test_custom_scalars_layout_404(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/custom-scalars/layout")
        assert resp.status_code == 404


# ===================================================================
# Metrics (unified)
# ===================================================================


class TestMetrics:
    def test_get_metrics(self, client):
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "scalars" in data
        assert "tensors" in data
        assert "artifacts" in data
        assert "text_events" in data
        # Check scalar tags appear.
        scalar_tags = [t["tag"] for t in data["scalars"]]
        assert "loss/train" in scalar_tags
        assert "lr" in scalar_tags

    def test_metrics_timeseries(self, client):
        tc, run_name = client
        resp = tc.post(
            f"/api/runs/{run_name}/metrics/timeseries",
            json={
                "requests": [
                    {"plugin": "scalars", "tag": "loss/train", "downsample": 100},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["plugin"] == "scalars"
        assert data[0]["tag"] == "loss/train"
        assert len(data[0]["data"]) == 20

    def test_metrics_timeseries_empty_request(self, client):
        tc, run_name = client
        resp = tc.post(
            f"/api/runs/{run_name}/metrics/timeseries",
            json={"requests": []},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_metrics_404_nonexistent_run(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/metrics")
        assert resp.status_code == 404


# ===================================================================
# Plugins
# ===================================================================


class TestPlugins:
    def test_list_plugins(self, client):
        tc, _ = client
        resp = tc.get("/api/plugins")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        names = {p["name"] for p in data}
        assert "scalars" in names
        assert "images" in names
        assert "text" in names


# ===================================================================
# Compare endpoints (top-level)
# ===================================================================


class TestCompare:
    def test_compare_scalars(self, client):
        tc, run_name = client
        resp = tc.get(
            "/api/compare/scalars",
            params={"tag": "loss/train", "runs": run_name},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert run_name in data
        assert len(data[run_name]) == 20

    def test_compare_scalars_nonexistent_run(self, client):
        """Non-existent runs are silently skipped."""
        tc, _ = client
        resp = tc.get(
            "/api/compare/scalars",
            params={"tag": "loss/train", "runs": "ghost_run"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data == {}

    def test_compare_eval(self, client):
        tc, run_name = client
        resp = tc.get(
            "/api/compare/eval",
            params={"suite": "fid", "runs": run_name},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert run_name in data
        assert len(data[run_name]) == 1

    def test_compare_hparams(self, rich_client):
        tc, run_name = rich_client
        resp = tc.get(
            "/api/compare/hparams",
            params={"runs": run_name},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["run"] == run_name
        assert "hparams" in data[0]
        assert "metrics" in data[0]


# ===================================================================
# Error model
# ===================================================================


class TestErrorModel:
    """Verify the spec-compliant JSON error envelope."""

    def test_404_error_format(self, client):
        tc, _ = client
        resp = tc.get("/api/runs/nonexistent/tags")
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "not_found"
        assert "message" in body["error"]
        assert "details" in body["error"]

    def test_400_error_format(self, client):
        """Invalid blob key format should return a 400 with structured error."""
        tc, run_name = client
        resp = tc.get(f"/api/runs/{run_name}/blob/INVALID_KEY_FORMAT")
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "invalid_request"
