"""Tests for SummaryWriter."""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from serenityboard.writer.summary_writer import SummaryWriter


class TestSummaryWriter:

    def test_writer_creates_db(self, writer_logdir: str) -> None:
        """Creating a writer should produce a board.db in the run directory."""
        w = SummaryWriter(writer_logdir, run_name="test_run")
        db_path = os.path.join(writer_logdir, "test_run", "board.db")
        assert os.path.exists(db_path)
        w.close()

    def test_add_scalar(self, writer_logdir: str) -> None:
        """Add 5 scalars, flush, read them back from SQLite."""
        w = SummaryWriter(writer_logdir, run_name="scalar_run")
        for step in range(5):
            w.add_scalar("loss", 1.0 - step * 0.2, step)
        w.flush()

        db_path = os.path.join(writer_logdir, "scalar_run", "board.db")
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT step, value FROM scalars WHERE tag = 'loss' ORDER BY step"
        ).fetchall()
        conn.close()

        assert len(rows) == 5
        assert rows[0] == (0, 1.0)
        assert rows[4][0] == 4
        assert abs(rows[4][1] - 0.2) < 1e-9
        w.close()

    def test_add_image(self, writer_logdir: str) -> None:
        """Add a CHW image, flush, verify artifacts table + blob file."""
        w = SummaryWriter(writer_logdir, run_name="image_run")
        rng = np.random.default_rng(42)
        img = rng.random((3, 64, 64), dtype=np.float32)  # CHW
        w.add_image("test_image", img, step=0)
        w.flush()

        db_path = os.path.join(writer_logdir, "image_run", "board.db")
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT tag, step, blob_key, width, height, kind, mime_type FROM artifacts"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        tag, step, blob_key, width, height, kind, mime_type = rows[0]
        assert tag == "test_image"
        assert step == 0
        assert width == 64
        assert height == 64
        assert kind == "image"
        assert mime_type == "image/png"

        # Verify blob file exists
        blob_path = os.path.join(writer_logdir, "image_run", "blobs", blob_key)
        assert os.path.exists(blob_path)
        assert os.path.getsize(blob_path) > 0
        w.close()

    def test_add_histogram(self, writer_logdir: str) -> None:
        """Add a histogram of random values, flush, verify tensors table."""
        w = SummaryWriter(writer_logdir, run_name="hist_run")
        rng = np.random.default_rng(42)
        values = rng.standard_normal(1000)
        w.add_histogram("weights", values, step=0)
        w.flush()

        db_path = os.path.join(writer_logdir, "hist_run", "board.db")
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT tag, step, dtype, shape FROM tensors"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        tag, step, dtype, shape = rows[0]
        assert tag == "weights"
        assert step == 0
        assert dtype == "float64"
        # default bins=64, so shape should be [64, 3]
        import json
        assert json.loads(shape) == [64, 3]
        w.close()

    def test_add_text(self, writer_logdir: str) -> None:
        """Add text, flush, verify text_events table."""
        w = SummaryWriter(writer_logdir, run_name="text_run")
        w.add_text("notes", "Hello world", step=0)
        w.flush()

        db_path = os.path.join(writer_logdir, "text_run", "board.db")
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT tag, step, value FROM text_events"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0] == ("notes", 0, "Hello world")
        w.close()

    def test_flush_and_close(self, writer_logdir: str) -> None:
        """writer.close() should set status to 'complete' in metadata."""
        w = SummaryWriter(writer_logdir, run_name="close_run")
        w.add_scalar("x", 1.0, step=0)
        w.close()

        db_path = os.path.join(writer_logdir, "close_run", "board.db")
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'status'"
        ).fetchone()
        conn.close()

        assert row is not None
        import json
        assert json.loads(row[0]) == "complete"

    def test_context_manager(self, writer_logdir: str) -> None:
        """Using `with SummaryWriter(...)` should auto-close and mark complete."""
        with SummaryWriter(writer_logdir, run_name="ctx_run") as w:
            w.add_scalar("y", 2.0, step=0)

        db_path = os.path.join(writer_logdir, "ctx_run", "board.db")
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'status'"
        ).fetchone()
        conn.close()

        assert row is not None
        import json
        assert json.loads(row[0]) == "complete"

    def test_noop_rank(self, writer_logdir: str) -> None:
        """Writer with rank=1 should be a no-op -- no db created or data written."""
        w = SummaryWriter(writer_logdir, run_name="noop_run", rank=1)
        w.add_scalar("loss", 1.0, step=0)
        w.flush()

        db_path = os.path.join(writer_logdir, "noop_run", "board.db")
        assert not os.path.exists(db_path)
        w.close()
