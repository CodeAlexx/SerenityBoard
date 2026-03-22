"""Tests for RunWatcher discovery, scanning, deletion, and status logic."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time

import pytest

from serenityboard.server.run_manager import RunWatcher, STALE_TIMEOUT_S
from serenityboard.writer.schema import create_tables, set_pragmas


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_run_db(
    parent_dir: str,
    run_name: str,
    *,
    status: str = "complete",
    start_time: float | None = None,
    wall_time_offset: float = 0.0,
    num_scalars: int = 5,
) -> str:
    """Create ``parent_dir/run_name/board.db`` with schema, metadata, a
    session row, and a handful of scalar points.

    Returns the absolute path to the ``board.db`` file.
    """
    run_dir = os.path.join(parent_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    db_path = os.path.join(run_dir, "board.db")

    now = time.time()
    if start_time is None:
        start_time = now - 600

    conn = sqlite3.connect(db_path)
    set_pragmas(conn)
    create_tables(conn)

    session_id = f"session-{run_name}"

    with conn:
        for key, value in [
            ("active_session_id", json.dumps(session_id)),
            ("status", json.dumps(status)),
            ("run_name", json.dumps(run_name)),
            ("start_time", json.dumps(start_time)),
        ]:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )

        conn.execute(
            "INSERT INTO sessions (session_id, start_time, resume_step, status) "
            "VALUES (?, ?, NULL, ?)",
            (session_id, start_time, "running" if status == "running" else "complete"),
        )

        for step in range(num_scalars):
            conn.execute(
                "INSERT INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                ("loss/train", step, start_time + step * 10 + wall_time_offset, 1.0 - step * 0.1),
            )

    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests: _find_run_dbs
# ---------------------------------------------------------------------------

class TestFindRunDbs:
    def test_find_run_dbs_flat(self, tmp_path):
        """Two runs directly in logdir should both be found."""
        logdir = str(tmp_path)
        _make_run_db(logdir, "run_alpha")
        _make_run_db(logdir, "run_beta")

        watcher = RunWatcher(logdir)
        found = watcher._find_run_dbs()

        assert len(found) == 2
        assert "run_alpha" in found
        assert "run_beta" in found
        for name, db_path in found.items():
            assert db_path.endswith("board.db")
            assert os.path.isfile(db_path)

    def test_find_run_dbs_nested(self, tmp_path):
        """A run at logdir/subdir1/subdir2/run1 should use __ separator in name."""
        logdir = str(tmp_path)
        nested = os.path.join(logdir, "subdir1", "subdir2", "run1")
        os.makedirs(nested, exist_ok=True)
        _make_run_db(os.path.join(logdir, "subdir1", "subdir2"), "run1")

        watcher = RunWatcher(logdir)
        found = watcher._find_run_dbs()

        assert len(found) == 1
        name = list(found.keys())[0]
        assert name == "subdir1__subdir2__run1"

    def test_find_run_dbs_depth_limit(self, tmp_path):
        """A run 5 levels deep should NOT be found (depth limit is 4)."""
        logdir = str(tmp_path)
        # 5 levels: a/b/c/d/e/board.db  -> depth from logdir = 5
        deep_dir = os.path.join(logdir, "a", "b", "c", "d", "e")
        os.makedirs(deep_dir, exist_ok=True)
        db_path = os.path.join(deep_dir, "board.db")
        conn = sqlite3.connect(db_path)
        set_pragmas(conn)
        create_tables(conn)
        conn.close()

        watcher = RunWatcher(logdir)
        found = watcher._find_run_dbs()
        assert len(found) == 0


# ---------------------------------------------------------------------------
# Tests: scan_once
# ---------------------------------------------------------------------------

class TestScanOnce:
    def test_scan_once_add(self, tmp_path):
        """Adding a run directory and calling scan_once should detect it."""
        logdir = str(tmp_path)
        watcher = RunWatcher(logdir)

        # Initially empty
        added, removed = watcher.scan_once()
        assert added == []
        assert removed == []
        assert len(watcher.known_runs) == 0

        # Create a run
        _make_run_db(logdir, "new_run")
        added, removed = watcher.scan_once()

        assert added == ["new_run"]
        assert removed == []
        assert "new_run" in watcher.known_runs

    def test_scan_once_remove(self, tmp_path):
        """Deleting a run directory should cause scan_once to report it removed."""
        logdir = str(tmp_path)
        _make_run_db(logdir, "doomed_run")

        watcher = RunWatcher(logdir)
        added, removed = watcher.scan_once()
        assert "doomed_run" in added
        assert "doomed_run" in watcher.known_runs

        # Remove the run directory
        shutil.rmtree(os.path.join(logdir, "doomed_run"))

        added, removed = watcher.scan_once()
        assert "doomed_run" in removed
        assert "doomed_run" not in watcher.known_runs


# ---------------------------------------------------------------------------
# Tests: delete_run
# ---------------------------------------------------------------------------

class TestDeleteRun:
    def test_delete_run(self, tmp_path):
        """delete_run should remove directory and clear known_runs."""
        logdir = str(tmp_path)
        _make_run_db(logdir, "delete_me")
        run_dir = os.path.join(logdir, "delete_me")

        watcher = RunWatcher(logdir)
        watcher.scan_once()
        assert "delete_me" in watcher.known_runs
        assert os.path.isdir(run_dir)

        result = watcher.delete_run("delete_me")
        assert result is True
        assert "delete_me" not in watcher.known_runs
        assert not os.path.isdir(run_dir)

    def test_delete_run_not_found(self, tmp_path):
        """delete_run on unknown run should return False."""
        logdir = str(tmp_path)
        watcher = RunWatcher(logdir)
        assert watcher.delete_run("nonexistent") is False


# ---------------------------------------------------------------------------
# Tests: get_runs status logic
# ---------------------------------------------------------------------------

class TestGetRunsStatus:
    def test_stale_running_becomes_stopped(self, tmp_path):
        """A 'running' run with old wall_time (>300s) and no max_steps
        should be reported as 'stopped'."""
        logdir = str(tmp_path)
        old_time = time.time() - STALE_TIMEOUT_S - 100
        _make_run_db(
            logdir,
            "stale_run",
            status="running",
            start_time=old_time,
            wall_time_offset=0.0,
            num_scalars=5,
        )

        watcher = RunWatcher(logdir)
        watcher.scan_once()

        runs = watcher.get_runs()
        assert len(runs) == 1
        run = runs[0]
        assert run["name"] == "stale_run"
        # wall_time is old_time + step*10, so max wall_time is old_time + 40
        # That is > 300s ago, so status should be "stopped" (no max_steps set)
        assert run["status"] == "stopped"

    def test_empty_run_detected(self, tmp_path):
        """A 'running' run with old start_time and NO data should become 'empty'."""
        logdir = str(tmp_path)
        old_time = time.time() - STALE_TIMEOUT_S - 100

        # Create run with no scalar data
        _make_run_db(
            logdir,
            "empty_run",
            status="running",
            start_time=old_time,
            num_scalars=0,
        )

        watcher = RunWatcher(logdir)
        watcher.scan_once()

        runs = watcher.get_runs()
        assert len(runs) == 1
        run = runs[0]
        assert run["name"] == "empty_run"
        assert run["status"] == "empty"

    def test_active_run_stays_running(self, tmp_path):
        """A 'running' run with recent wall_time should stay 'running'."""
        logdir = str(tmp_path)
        recent_time = time.time() - 10  # 10 seconds ago
        _make_run_db(
            logdir,
            "active_run",
            status="running",
            start_time=recent_time,
            num_scalars=3,
        )

        watcher = RunWatcher(logdir)
        watcher.scan_once()

        runs = watcher.get_runs()
        assert len(runs) == 1
        assert runs[0]["status"] == "running"
