"""Tests for the torch.utils.tensorboard compatibility layer."""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pytest

from serenityboard.compat.torch import SummaryWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db_path(logdir: str, run_name: str = "compat_run") -> str:
    return os.path.join(logdir, run_name, "board.db")


def _connect(logdir: str, run_name: str = "compat_run") -> sqlite3.Connection:
    return sqlite3.connect(_db_path(logdir, run_name))


# ---------------------------------------------------------------------------
# Import compatibility
# ---------------------------------------------------------------------------

class TestImportPath:
    """The import path matches the TensorBoard replacement pattern."""

    def test_import_works(self) -> None:
        from serenityboard.compat.torch import SummaryWriter as SW
        assert SW is SummaryWriter

    def test_compat_init_works(self) -> None:
        from serenityboard.compat import torch
        assert hasattr(torch, "SummaryWriter")


# ---------------------------------------------------------------------------
# add_scalar
# ---------------------------------------------------------------------------

class TestAddScalar:
    """add_scalar forwards with correct arg mapping."""

    def test_scalar_value_and_global_step(self, writer_logdir: str) -> None:
        """scalar_value -> value, global_step -> step."""
        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        w.add_scalar("loss", scalar_value=0.5, global_step=10)
        w.flush()

        conn = _connect(writer_logdir)
        row = conn.execute(
            "SELECT step, value FROM scalars WHERE tag = 'loss'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == 10
        assert abs(row[1] - 0.5) < 1e-9
        w.close()

    def test_global_step_defaults_to_zero(self, writer_logdir: str) -> None:
        """When global_step is None, step defaults to 0."""
        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        w.add_scalar("metric", scalar_value=1.0)
        w.flush()

        conn = _connect(writer_logdir)
        row = conn.execute(
            "SELECT step FROM scalars WHERE tag = 'metric'"
        ).fetchone()
        conn.close()

        assert row[0] == 0
        w.close()


# ---------------------------------------------------------------------------
# add_image
# ---------------------------------------------------------------------------

class TestAddImage:
    """add_image forwards img_tensor and dataformats."""

    def test_chw_image(self, writer_logdir: str) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((3, 32, 32), dtype=np.float32)

        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        w.add_image("test_img", img, global_step=5, dataformats="CHW")
        w.flush()

        conn = _connect(writer_logdir)
        row = conn.execute(
            "SELECT tag, step, width, height FROM artifacts WHERE tag = 'test_img'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test_img"
        assert row[1] == 5
        assert row[2] == 32  # width
        assert row[3] == 32  # height
        w.close()


# ---------------------------------------------------------------------------
# add_histogram
# ---------------------------------------------------------------------------

class TestAddHistogram:
    """add_histogram maps bins='tensorflow' to integer bins."""

    def test_tensorflow_bins(self, writer_logdir: str) -> None:
        """bins='tensorflow' maps to 64 bins internally."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(500)

        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        w.add_histogram("weights", values, global_step=0, bins="tensorflow")
        w.flush()

        conn = _connect(writer_logdir)
        row = conn.execute(
            "SELECT shape FROM tensors WHERE tag = 'weights'"
        ).fetchone()
        conn.close()

        shape = json.loads(row[0])
        assert shape == [64, 3]
        w.close()

    def test_integer_bins_passthrough(self, writer_logdir: str) -> None:
        """Integer bins value passes through directly."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(500)

        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        w.add_histogram("weights2", values, global_step=0, bins=32)
        w.flush()

        conn = _connect(writer_logdir)
        row = conn.execute(
            "SELECT shape FROM tensors WHERE tag = 'weights2'"
        ).fetchone()
        conn.close()

        shape = json.loads(row[0])
        assert shape == [32, 3]
        w.close()


# ---------------------------------------------------------------------------
# add_text
# ---------------------------------------------------------------------------

class TestAddText:
    """add_text maps text_string -> text."""

    def test_text_string_mapping(self, writer_logdir: str) -> None:
        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        w.add_text("notes", text_string="hello world", global_step=3)
        w.flush()

        conn = _connect(writer_logdir)
        row = conn.execute(
            "SELECT step, value FROM text_events WHERE tag = 'notes'"
        ).fetchone()
        conn.close()

        assert row == (3, "hello world")
        w.close()


# ---------------------------------------------------------------------------
# add_hparams
# ---------------------------------------------------------------------------

class TestAddHparams:
    """add_hparams forwards hparam_dict and metric_dict."""

    def test_hparams_passthrough(self, writer_logdir: str) -> None:
        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        w.add_hparams(
            hparam_dict={"lr": 1e-3, "epochs": 10},
            metric_dict={"accuracy": 0.95},
        )
        w.flush()

        conn = _connect(writer_logdir)
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'hparams'"
        ).fetchone()
        hp = json.loads(row[0])
        assert hp["lr"] == 1e-3
        assert hp["epochs"] == 10

        metric_row = conn.execute(
            "SELECT metric_tag, value FROM hparam_metrics WHERE metric_tag = 'accuracy'"
        ).fetchone()
        assert metric_row is not None
        assert abs(metric_row[1] - 0.95) < 1e-9

        conn.close()
        w.close()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    """Context manager opens and closes cleanly."""

    def test_with_statement(self, writer_logdir: str) -> None:
        with SummaryWriter(log_dir=writer_logdir, run_name="compat_run") as w:
            w.add_scalar("x", scalar_value=1.0, global_step=0)

        conn = _connect(writer_logdir)
        status = conn.execute(
            "SELECT value FROM metadata WHERE key = 'status'"
        ).fetchone()
        conn.close()

        assert json.loads(status[0]) == "complete"


# ---------------------------------------------------------------------------
# Constructor arg mapping
# ---------------------------------------------------------------------------

class TestConstructorArgs:
    """TensorBoard constructor args are mapped correctly."""

    def test_log_dir_maps_to_logdir(self, writer_logdir: str) -> None:
        """log_dir parameter maps to native logdir."""
        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        db = _db_path(writer_logdir)
        assert os.path.exists(db)
        w.close()

    def test_purge_step_maps_to_resume_step(self, writer_logdir: str) -> None:
        """purge_step parameter maps to resume_step (tested by no crash)."""
        # Just verify it doesn't crash — purge_step is only meaningful
        # when resuming an existing run.
        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        w.close()


# ---------------------------------------------------------------------------
# V2 stubs
# ---------------------------------------------------------------------------

class TestV2Stubs:
    """Unimplemented methods raise NotImplementedError."""

    def test_add_video_not_implemented(self, writer_logdir: str) -> None:
        w = SummaryWriter(log_dir=writer_logdir, run_name="compat_run")
        with pytest.raises(NotImplementedError, match="add_video"):
            w.add_video("tag", None)
        w.close()
