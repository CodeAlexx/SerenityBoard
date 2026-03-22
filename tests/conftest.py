"""Shared fixtures for SerenityBoard tests."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time

import numpy as np
import pytest

from serenityboard.writer.schema import create_tables, set_pragmas


@pytest.fixture()
def tmp_logdir():
    """Create a temporary directory, yield it, clean up after."""
    d = tempfile.mkdtemp(prefix="sb_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def writer_logdir():
    """A clean temp dir for writer tests."""
    d = tempfile.mkdtemp(prefix="sb_writer_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def sample_db():
    """Create a temp dir with a board.db containing representative sample data.

    Yields the path to board.db.
    """
    run_dir = tempfile.mkdtemp(prefix="sb_sample_")
    db_path = os.path.join(run_dir, "board.db")

    conn = sqlite3.connect(db_path)
    set_pragmas(conn)
    create_tables(conn)

    session_id = "test-session-001"
    now = time.time()

    with conn:
        # -- metadata --
        for key, value in [
            ("active_session_id", json.dumps(session_id)),
            ("status", json.dumps("complete")),
            ("run_name", json.dumps("sample_run")),
            ("start_time", json.dumps(now - 600)),
            ("schema_version", json.dumps("2")),
        ]:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )

        # -- sessions --
        conn.execute(
            "INSERT INTO sessions (session_id, start_time, resume_step, status) "
            "VALUES (?, ?, NULL, 'complete')",
            (session_id, now - 600),
        )

        # -- scalars: loss/train (20 points, decreasing) --
        for step in range(20):
            value = 1.0 - step * (0.9 / 19)  # 1.0 down to ~0.1
            conn.execute(
                "INSERT INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                ("loss/train", step, now - 600 + step * 10, value),
            )

        # -- scalars: lr (10 points) --
        for step in range(10):
            conn.execute(
                "INSERT INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                ("lr", step, now - 600 + step * 10, 1e-4 * (1 - step / 10)),
            )

        # -- artifacts: 1 image --
        conn.execute(
            "INSERT INTO artifacts "
            "(tag, step, seq_index, wall_time, kind, mime_type, blob_key, width, height, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("samples", 5, 0, now - 550, "image", "image/png", "abc123.png", 512, 512, "{}"),
        )

        # -- text_events: 2 entries --
        conn.execute(
            "INSERT INTO text_events (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
            ("config", 0, now - 600, '{"lr": 1e-4, "epochs": 10}'),
        )
        conn.execute(
            "INSERT INTO text_events (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
            ("config", 1, now - 590, '{"lr": 5e-5, "epochs": 10}'),
        )

        # -- tensors: 1 histogram entry --
        rng = np.random.default_rng(42)
        hist_data = rng.standard_normal((10, 3)).astype(np.float64)
        conn.execute(
            "INSERT INTO tensors (tag, step, wall_time, dtype, shape, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "weights/layer1",
                0,
                now - 600,
                "float64",
                json.dumps([10, 3]),
                hist_data.tobytes(),
            ),
        )

        # -- trace_events: 2 entries --
        conn.execute(
            "INSERT INTO trace_events (step, wall_time, phase, duration_ms, details) "
            "VALUES (?, ?, ?, ?, ?)",
            (0, now - 600, "forward", 12.5, "{}"),
        )
        conn.execute(
            "INSERT INTO trace_events (step, wall_time, phase, duration_ms, details) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, now - 590, "backward", 25.0, '{"grad_norm": 1.2}'),
        )

        # -- eval_results: 1 entry --
        conn.execute(
            "INSERT INTO eval_results "
            "(suite_name, case_id, step, wall_time, score_name, score_value, artifact_key, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("fid", "case_0", 10, now - 500, "fid_score", 42.5, None, "{}"),
        )

    conn.close()

    # -- blobs directory with dummy image --
    blobs_dir = os.path.join(run_dir, "blobs")
    os.makedirs(blobs_dir, exist_ok=True)
    with open(os.path.join(blobs_dir, "abc123.png"), "wb") as f:
        f.write(b"PNG_FAKE_DATA")

    yield db_path

    shutil.rmtree(run_dir, ignore_errors=True)


@pytest.fixture()
def sample_run_dir(sample_db):
    """Yield the path to the run directory containing board.db."""
    return os.path.dirname(sample_db)
