"""Comprehensive tests for every SummaryWriter method, verifying data in board.db."""
from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np
import pytest

from serenityboard.server.data_provider import RunDataProvider
from serenityboard.writer.summary_writer import SummaryWriter


def _db_path(logdir: str, run_name: str) -> str:
    return os.path.join(logdir, run_name, "board.db")


def _connect(logdir: str, run_name: str) -> sqlite3.Connection:
    return sqlite3.connect(_db_path(logdir, run_name))


class TestAddAudio:

    def test_add_audio(self, writer_logdir: str) -> None:
        """Write a 1-second sine wave, flush, verify audio table and blob on disk."""
        run = "audio_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        sample_rate = 16000
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        sine_wave = np.sin(2 * np.pi * 440.0 * t)  # 440 Hz, 1 second

        w.add_audio("tone", sine_wave, step=0, sample_rate=sample_rate)
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT tag, step, blob_key, sample_rate, num_channels, duration_ms, mime_type "
            "FROM audio WHERE tag = 'tone'"
        ).fetchone()
        conn.close()

        assert row is not None, "No audio row found"
        tag, step, blob_key, sr, ch, dur_ms, mime = row
        assert tag == "tone"
        assert step == 0
        assert sr == sample_rate
        assert ch == 1
        assert abs(dur_ms - 1000.0) < 1.0  # ~1 second
        assert mime == "audio/wav"

        # Verify blob file exists on disk
        blob_path = os.path.join(writer_logdir, run, "blobs", blob_key)
        assert os.path.exists(blob_path)
        assert os.path.getsize(blob_path) > 0

        provider = RunDataProvider(_db_path(writer_logdir, run))
        preview_rows = provider.read_audio("tone")
        assert len(preview_rows) == 1
        assert preview_rows[0]["tag"] == "tone"
        assert isinstance(preview_rows[0]["waveform"], list)
        assert len(preview_rows[0]["waveform"]) > 0
        assert len(preview_rows[0]["waveform"][0]) == 2
        assert isinstance(preview_rows[0]["spectrogram"], list)
        assert len(preview_rows[0]["spectrogram"]) > 0
        assert isinstance(preview_rows[0]["peak_db"], float)
        assert isinstance(preview_rows[0]["rms_db"], float)

        w.close()


class TestAddEmbedding:

    def test_add_embedding(self, writer_logdir: str) -> None:
        """Write a (50, 10) embedding matrix with metadata, verify DB columns."""
        run = "emb_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        rng = np.random.default_rng(42)
        mat = rng.random((50, 10), dtype=np.float32)
        labels = [f"point_{i}" for i in range(50)]

        w.add_embedding(mat, metadata=labels, global_step=5, tag="features")
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT tag, step, num_points, dimensions, tensor_blob_key, "
            "metadata_json, sprite_blob_key "
            "FROM embeddings WHERE tag = 'features'"
        ).fetchone()
        conn.close()

        assert row is not None, "No embedding row found"
        tag, step, n, d, blob_key, meta_json, sprite_key = row
        assert tag == "features"
        assert step == 5
        assert n == 50
        assert d == 10
        assert blob_key is not None and len(blob_key) > 0
        assert sprite_key is None  # no label_img provided

        # Verify metadata was stored
        stored_labels = json.loads(meta_json)
        assert len(stored_labels) == 50
        assert stored_labels[0] == "point_0"

        w.close()

    def test_add_embedding_with_sprites(self, writer_logdir: str) -> None:
        """Write embedding with (10, 3, 8, 8) label_img, verify sprite_blob_key is set."""
        run = "emb_sprite_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        rng = np.random.default_rng(42)
        mat = rng.random((10, 5), dtype=np.float32)
        label_img = rng.random((10, 3, 8, 8)).astype(np.float32)

        w.add_embedding(mat, label_img=label_img, global_step=0, tag="sprites")
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT num_points, dimensions, sprite_blob_key, sprite_single_h, sprite_single_w "
            "FROM embeddings WHERE tag = 'sprites'"
        ).fetchone()
        conn.close()

        assert row is not None
        n, d, sprite_key, sprite_h, sprite_w = row
        assert n == 10
        assert d == 5
        assert sprite_key is not None and len(sprite_key) > 0
        assert sprite_h == 8
        assert sprite_w == 8

        # Verify sprite blob file exists
        blob_path = os.path.join(writer_logdir, run, "blobs", sprite_key)
        assert os.path.exists(blob_path)

        w.close()


class TestAddMesh:

    def test_add_mesh(self, writer_logdir: str) -> None:
        """Write 100 vertices, verify meshes table entry."""
        run = "mesh_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        rng = np.random.default_rng(42)
        vertices = rng.random((100, 3), dtype=np.float32)

        w.add_mesh("cube", vertices, global_step=1)
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT tag, step, num_vertices, has_faces, has_colors, num_faces, "
            "vertices_blob_key, faces_blob_key, colors_blob_key "
            "FROM meshes WHERE tag = 'cube'"
        ).fetchone()
        conn.close()

        assert row is not None
        tag, step, nv, hf, hc, nf, vbk, fbk, cbk = row
        assert tag == "cube"
        assert step == 1
        assert nv == 100
        assert hf == 0
        assert hc == 0
        assert nf == 0
        assert vbk is not None
        assert fbk is None
        assert cbk is None

        w.close()

    def test_add_mesh_with_faces_and_colors(self, writer_logdir: str) -> None:
        """Write mesh with faces (20, 3) and colors (100, 3), verify has_faces/has_colors."""
        run = "mesh_full_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        rng = np.random.default_rng(42)
        vertices = rng.random((100, 3), dtype=np.float32)
        faces = rng.integers(0, 100, size=(20, 3), dtype=np.int32)
        colors = rng.integers(0, 256, size=(100, 3), dtype=np.uint8)

        w.add_mesh("colored_mesh", vertices, colors=colors, faces=faces, global_step=2)
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT num_vertices, has_faces, has_colors, num_faces, "
            "faces_blob_key, colors_blob_key "
            "FROM meshes WHERE tag = 'colored_mesh'"
        ).fetchone()
        conn.close()

        assert row is not None
        nv, hf, hc, nf, fbk, cbk = row
        assert nv == 100
        assert hf == 1
        assert hc == 1
        assert nf == 20
        assert fbk is not None
        assert cbk is not None

        w.close()


class TestAddPRCurve:

    def test_add_pr_curve(self, writer_logdir: str) -> None:
        """Write PR curve data, verify table entry and unpack precision/recall."""
        run = "pr_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        rng = np.random.default_rng(42)
        # 100 binary samples: 60 positive, 40 negative
        labels = np.array([1] * 60 + [0] * 40, dtype=np.float64)
        # Predictions: positives get higher scores on average
        predictions = np.concatenate([
            rng.uniform(0.4, 1.0, size=60),
            rng.uniform(0.0, 0.6, size=40),
        ])

        num_thresholds = 51
        w.add_pr_curve("binary_clf", labels, predictions, step=10, num_thresholds=num_thresholds)
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT tag, step, class_index, num_thresholds, data "
            "FROM pr_curves WHERE tag = 'binary_clf'"
        ).fetchone()
        conn.close()

        assert row is not None
        tag, step, cls_idx, nt, data_bytes = row
        assert tag == "binary_clf"
        assert step == 10
        assert cls_idx == 0
        assert nt == num_thresholds

        # Unpack the blob: [6, num_thresholds] float64
        data = np.frombuffer(data_bytes, dtype=np.float64).reshape(6, num_thresholds)
        tp, fp, tn, fn, precision, recall = data

        # At threshold=0.0, everything is predicted positive
        assert tp[0] == 60  # all positives predicted positive
        assert fp[0] == 40  # all negatives predicted positive
        assert recall[0] == 1.0  # all positives found

        # At threshold=1.0, nothing is predicted positive
        assert tp[-1] == 0
        assert fp[-1] == 0

        # Precision values should be in [0, 1]
        assert np.all(precision >= 0.0)
        assert np.all(precision <= 1.0)

        # Recall values should be in [0, 1]
        assert np.all(recall >= 0.0)
        assert np.all(recall <= 1.0)

        w.close()


class TestAddText:

    def test_add_text(self, writer_logdir: str) -> None:
        """Write text at step 0, flush, verify text_events table."""
        run = "text_method_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        w.add_text("log", "training started", step=0)
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT tag, step, value FROM text_events WHERE tag = 'log'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row == ("log", 0, "training started")

        w.close()


class TestAddHparams:

    def test_add_hparams(self, writer_logdir: str) -> None:
        """Write hparams + metrics, verify metadata and hparam_metrics tables."""
        run = "hparams_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        hparams = {"lr": 1e-4, "batch_size": 32, "optimizer": "adam"}
        metrics = {"final_loss": 0.05, "accuracy": 0.98}

        w.add_hparams(hparams, metrics)
        w.flush()

        conn = _connect(writer_logdir, run)

        # Verify hparams in metadata
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'hparams'"
        ).fetchone()
        assert row is not None
        stored_hp = json.loads(row[0])
        assert stored_hp["lr"] == 1e-4
        assert stored_hp["batch_size"] == 32
        assert stored_hp["optimizer"] == "adam"

        # Verify metric entries
        metric_rows = conn.execute(
            "SELECT metric_tag, value FROM hparam_metrics ORDER BY metric_tag"
        ).fetchall()
        conn.close()

        metric_dict = {r[0]: r[1] for r in metric_rows}
        assert abs(metric_dict["accuracy"] - 0.98) < 1e-9
        assert abs(metric_dict["final_loss"] - 0.05) < 1e-9

        w.close()


class TestAddCustomScalarsLayout:

    def test_add_custom_scalars_layout(self, writer_logdir: str) -> None:
        """Write layout config, flush, verify custom_scalar_layouts table."""
        run = "layout_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        layout = {
            "categories": [
                {
                    "title": "Losses",
                    "charts": [
                        {"title": "Train vs Val", "tags": ["loss/train", "loss/val"]},
                    ],
                }
            ]
        }
        w.add_custom_scalars_layout(layout)
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT layout_name, config FROM custom_scalar_layouts "
            "WHERE layout_name = 'default'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "default"
        stored_layout = json.loads(row[1])
        assert stored_layout["categories"][0]["title"] == "Losses"
        assert stored_layout["categories"][0]["charts"][0]["tags"] == ["loss/train", "loss/val"]

        w.close()


class TestAddPluginData:

    def test_add_plugin_data(self, writer_logdir: str) -> None:
        """Write plugin data, flush, verify plugin_data table."""
        run = "plugin_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        data = {"model_version": "v2", "checkpoint": "ckpt_100.pt"}
        w.add_plugin_data("custom_tracker", "metadata", data, step=100)
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT plugin_name, tag, step, data "
            "FROM plugin_data WHERE plugin_name = 'custom_tracker' AND tag = 'metadata'"
        ).fetchone()
        conn.close()

        assert row is not None
        pname, tag, step, data_json = row
        assert pname == "custom_tracker"
        assert tag == "metadata"
        assert step == 100
        stored_data = json.loads(data_json)
        assert stored_data["model_version"] == "v2"
        assert stored_data["checkpoint"] == "ckpt_100.pt"

        w.close()


class TestAddTrace:

    def test_add_trace(self, writer_logdir: str) -> None:
        """Write trace event, flush, verify trace_events table."""
        run = "trace_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        details = {"grad_norm": 1.5, "memory_mb": 4096}
        w.add_trace(step=7, phase="backward", duration_ms=25.3, details=details)
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT step, phase, duration_ms, details "
            "FROM trace_events WHERE step = 7 AND phase = 'backward'"
        ).fetchone()
        conn.close()

        assert row is not None
        step, phase, dur, det_json = row
        assert step == 7
        assert phase == "backward"
        assert abs(dur - 25.3) < 0.01
        det = json.loads(det_json)
        assert det["grad_norm"] == 1.5
        assert det["memory_mb"] == 4096

        w.close()


class TestAddEval:

    def test_add_eval(self, writer_logdir: str) -> None:
        """Write eval result, flush, verify eval_results table."""
        run = "eval_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        details = {"samples": 1000}
        w.add_eval(
            suite_name="fid",
            case_id="full_set",
            step=50,
            score_name="fid_score",
            score_value=35.2,
            details=details,
        )
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT suite_name, case_id, step, score_name, score_value, details "
            "FROM eval_results WHERE suite_name = 'fid' AND case_id = 'full_set'"
        ).fetchone()
        conn.close()

        assert row is not None
        suite, cid, step, sname, sval, det_json = row
        assert suite == "fid"
        assert cid == "full_set"
        assert step == 50
        assert sname == "fid_score"
        assert abs(sval - 35.2) < 0.01
        det = json.loads(det_json)
        assert det["samples"] == 1000

        w.close()


class TestCrossCuttingConcerns:

    def test_wall_time_set(self, writer_logdir: str) -> None:
        """For each data type, verify wall_time is a recent timestamp."""
        run = "walltime_run"
        before = time.time()
        w = SummaryWriter(writer_logdir, run_name=run)

        rng = np.random.default_rng(42)

        # Write one of each lossless type
        w.add_scalar("s", 1.0, step=0)
        w.add_text("t", "hello", step=0)
        w.add_trace(step=0, phase="forward", duration_ms=10.0)

        # Write one of each reservoir type
        w.add_histogram("h", rng.standard_normal(100), step=0)
        w.add_image("img", rng.random((3, 8, 8), dtype=np.float32), step=0)
        w.add_audio("aud", rng.random(1000, dtype=np.float32), step=0, sample_rate=8000)
        w.add_pr_curve("pr", np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]), step=0)
        w.add_embedding(rng.random((5, 3), dtype=np.float32), global_step=0, tag="emb")
        w.add_mesh("m", rng.random((10, 3), dtype=np.float32), global_step=0)
        w.add_plugin_data("p", "d", {"x": 1}, step=0)

        w.flush()
        after = time.time()

        conn = _connect(writer_logdir, run)

        tables_with_wall_time = [
            ("scalars", "wall_time"),
            ("text_events", "wall_time"),
            ("trace_events", "wall_time"),
            ("tensors", "wall_time"),
            ("artifacts", "wall_time"),
            ("audio", "wall_time"),
            ("pr_curves", "wall_time"),
            ("embeddings", "wall_time"),
            ("meshes", "wall_time"),
            ("plugin_data", "wall_time"),
        ]

        for table, col in tables_with_wall_time:
            row = conn.execute(f"SELECT {col} FROM {table} LIMIT 1").fetchone()
            assert row is not None, f"No row in {table}"
            wt = row[0]
            # Allow generous tolerance (60s) for slow CI environments
            assert before - 60 <= wt <= after + 60, (
                f"{table}.{col}={wt} not within [{before - 60}, {after + 60}]"
            )

        conn.close()
        w.close()

    def test_flush_commits_data(self, writer_logdir: str) -> None:
        """Data should be readable from the DB only after flush()."""
        run = "flush_commit_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        w.add_scalar("x", 42.0, step=0)
        w.flush()

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT value FROM scalars WHERE tag = 'x' AND step = 0"
        ).fetchone()
        conn.close()

        assert row is not None
        assert abs(row[0] - 42.0) < 1e-9

        w.close()

    def test_close_sets_complete(self, writer_logdir: str) -> None:
        """After close(), sessions table should have status='complete'."""
        run = "close_complete_run"
        w = SummaryWriter(writer_logdir, run_name=run)
        w.add_scalar("y", 1.0, step=0)
        w.close()

        conn = _connect(writer_logdir, run)

        # Check metadata status
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'status'"
        ).fetchone()
        assert row is not None
        assert json.loads(row[0]) == "complete"

        # Check sessions table
        row = conn.execute(
            "SELECT status FROM sessions WHERE status = 'complete'"
        ).fetchone()
        assert row is not None
        assert row[0] == "complete"

        conn.close()

    def test_context_manager(self, writer_logdir: str) -> None:
        """Using `with SummaryWriter(...)` should auto-close and mark complete."""
        run = "ctx_mgr_run"
        with SummaryWriter(writer_logdir, run_name=run) as w:
            w.add_scalar("z", 3.0, step=0)

        conn = _connect(writer_logdir, run)
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'status'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert json.loads(row[0]) == "complete"


class TestDataProviderRoundTrip:

    def test_data_readable_via_provider(self, writer_logdir: str) -> None:
        """Write scalars, text, histogram via SummaryWriter, read back via RunDataProvider."""
        run = "provider_run"
        w = SummaryWriter(writer_logdir, run_name=run)

        # Write scalars
        for step in range(10):
            w.add_scalar("loss/train", 1.0 - step * 0.1, step)

        # Write text
        w.add_text("config", '{"lr": 0.001}', step=0)

        # Write histogram
        rng = np.random.default_rng(42)
        w.add_histogram("weights/fc1", rng.standard_normal(500), step=0)

        w.flush()

        # Read back via RunDataProvider
        db = _db_path(writer_logdir, run)
        provider = RunDataProvider(db)

        # Scalars
        tags = provider.get_tags()
        assert "loss/train" in tags["scalars"]

        rows = provider.read_scalars_downsampled("loss/train", n=0)
        assert len(rows) == 10
        # First point
        assert rows[0][0] == 0  # step
        assert abs(rows[0][2] - 1.0) < 1e-9  # value
        # Last point
        assert rows[9][0] == 9
        assert abs(rows[9][2] - 0.1) < 1e-9

        # Text
        text_results = provider.read_text("config")
        assert len(text_results) == 1
        assert text_results[0]["value"] == '{"lr": 0.001}'

        # Histogram
        hist_results = provider.read_histograms("weights/fc1")
        assert len(hist_results) == 1
        assert hist_results[0]["step"] == 0
        # bins should be list of lists with [left, right, count]
        assert len(hist_results[0]["bins"]) == 64  # default bins
        assert len(hist_results[0]["bins"][0]) == 3

        provider.close()
        w.close()
