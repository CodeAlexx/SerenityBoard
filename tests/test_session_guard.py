"""Tests for SessionGuard session lifecycle management."""
from __future__ import annotations

import json
import os
import sqlite3
import time

import pytest

from serenityboard.writer.schema import create_tables, set_pragmas
from serenityboard.writer.session_guard import SessionGuard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_dir: str) -> tuple[sqlite3.Connection, str]:
    """Create a board.db with schema, return (conn, db_path)."""
    db_path = os.path.join(tmp_dir, "board.db")
    conn = sqlite3.connect(db_path)
    set_pragmas(conn)
    create_tables(conn)
    return conn, db_path


def _seed_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Insert an active session + metadata to simulate an existing run."""
    now = time.time()
    with conn:
        conn.execute(
            "INSERT INTO sessions (session_id, start_time, resume_step, status) "
            "VALUES (?, ?, NULL, 'running')",
            (session_id, now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("active_session_id", json.dumps(session_id)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("status", json.dumps("running")),
        )


def _insert_scalars(conn: sqlite3.Connection, tag: str, steps: range) -> None:
    """Insert scalar rows at given steps."""
    now = time.time()
    with conn:
        for step in steps:
            conn.execute(
                "INSERT INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                (tag, step, now + step, float(step)),
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFirstSession:
    """First session creates a session record in DB."""

    def test_creates_session_record(self, tmp_logdir: str) -> None:
        conn, _ = _make_db(tmp_logdir)
        guard = SessionGuard(conn, "session-1", resume_step=None)
        guard.initialize()

        row = conn.execute(
            "SELECT session_id, status FROM sessions WHERE session_id = 'session-1'"
        ).fetchone()
        assert row is not None
        assert row[0] == "session-1"
        assert row[1] == "running"

    def test_sets_active_session_metadata(self, tmp_logdir: str) -> None:
        conn, _ = _make_db(tmp_logdir)
        guard = SessionGuard(conn, "session-1", resume_step=None)
        guard.initialize()

        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'active_session_id'"
        ).fetchone()
        assert row is not None
        assert json.loads(row[0]) == "session-1"

    def test_sets_status_running(self, tmp_logdir: str) -> None:
        conn, _ = _make_db(tmp_logdir)
        guard = SessionGuard(conn, "session-1", resume_step=None)
        guard.initialize()

        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'status'"
        ).fetchone()
        assert json.loads(row[0]) == "running"


class TestResumeWithStep:
    """Resume with resume_step purges data after that step."""

    def test_purges_data_after_resume_step(self, tmp_logdir: str) -> None:
        conn, _ = _make_db(tmp_logdir)
        _seed_session(conn, "old-session")
        _insert_scalars(conn, "loss", range(10))

        guard = SessionGuard(conn, "new-session", resume_step=5)
        guard.initialize()

        # Steps 0-5 should remain, steps 6-9 should be purged
        rows = conn.execute(
            "SELECT step FROM scalars WHERE tag = 'loss' ORDER BY step"
        ).fetchall()
        steps = [r[0] for r in rows]
        assert steps == [0, 1, 2, 3, 4, 5]

    def test_marks_old_session_crashed(self, tmp_logdir: str) -> None:
        conn, _ = _make_db(tmp_logdir)
        _seed_session(conn, "old-session")

        guard = SessionGuard(conn, "new-session", resume_step=0)
        guard.initialize()

        row = conn.execute(
            "SELECT status FROM sessions WHERE session_id = 'old-session'"
        ).fetchone()
        assert row[0] == "crashed"

    def test_creates_new_session_record(self, tmp_logdir: str) -> None:
        conn, _ = _make_db(tmp_logdir)
        _seed_session(conn, "old-session")

        guard = SessionGuard(conn, "new-session", resume_step=0)
        guard.initialize()

        row = conn.execute(
            "SELECT session_id, resume_step, status FROM sessions WHERE session_id = 'new-session'"
        ).fetchone()
        assert row is not None
        assert row[0] == "new-session"
        assert row[1] == 0
        assert row[2] == "running"

    def test_purges_multiple_tables(self, tmp_logdir: str) -> None:
        """Purge affects all data tables, not just scalars."""
        conn, _ = _make_db(tmp_logdir)
        _seed_session(conn, "old-session")

        now = time.time()
        with conn:
            # scalars
            for step in range(5):
                conn.execute(
                    "INSERT INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                    ("loss", step, now, float(step)),
                )
            # text_events
            for step in range(5):
                conn.execute(
                    "INSERT INTO text_events (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                    ("log", step, now, f"step {step}"),
                )
            # trace_events
            for step in range(5):
                conn.execute(
                    "INSERT INTO trace_events (step, wall_time, phase, duration_ms, details) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (step, now, "forward", 10.0, "{}"),
                )

        guard = SessionGuard(conn, "new-session", resume_step=2)
        guard.initialize()

        # Check each table: steps 0-2 kept, steps 3-4 purged
        for table in ("scalars", "text_events", "trace_events"):
            rows = conn.execute(f"SELECT step FROM {table} ORDER BY step").fetchall()
            steps = [r[0] for r in rows]
            assert max(steps) <= 2, f"{table} has steps beyond resume: {steps}"


class TestResumeWithoutStep:
    """Resume without resume_step raises ValueError."""

    def test_raises_value_error(self, tmp_logdir: str) -> None:
        conn, _ = _make_db(tmp_logdir)
        _seed_session(conn, "old-session")

        guard = SessionGuard(conn, "new-session", resume_step=None)
        with pytest.raises(ValueError, match="resume_step"):
            guard.initialize()


class TestPurgeTransactional:
    """Purge is transactional — all tables purged together."""

    def test_purge_is_atomic(self, tmp_logdir: str) -> None:
        """If the transaction commits, all tables are purged. We verify this
        by checking that the purge and new session insert are in the same
        transaction (both present after initialize)."""
        conn, _ = _make_db(tmp_logdir)
        _seed_session(conn, "old-session")
        _insert_scalars(conn, "loss", range(10))

        guard = SessionGuard(conn, "new-session", resume_step=3)
        guard.initialize()

        # Both purge and session insert happened
        scalar_count = conn.execute(
            "SELECT COUNT(*) FROM scalars"
        ).fetchone()[0]
        assert scalar_count == 4  # steps 0-3

        new_session = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = 'new-session'"
        ).fetchone()
        assert new_session is not None


class TestMarkComplete:

    def test_mark_complete_sets_status(self, tmp_logdir: str) -> None:
        conn, _ = _make_db(tmp_logdir)
        guard = SessionGuard(conn, "session-1", resume_step=None)
        guard.initialize()

        guard.mark_complete()

        row = conn.execute(
            "SELECT status FROM sessions WHERE session_id = 'session-1'"
        ).fetchone()
        assert row[0] == "complete"

        meta = conn.execute(
            "SELECT value FROM metadata WHERE key = 'status'"
        ).fetchone()
        assert json.loads(meta[0]) == "complete"


class TestOrphanedBlobCleanup:
    """Purge cleans up orphaned blob files from disk."""

    def test_cleans_orphaned_artifact_blobs(self, tmp_logdir: str) -> None:
        conn, _ = _make_db(tmp_logdir)
        _seed_session(conn, "old-session")

        blobs_dir = os.path.join(tmp_logdir, "blobs")
        os.makedirs(blobs_dir, exist_ok=True)

        # Insert artifacts at steps 3 and 7
        now = time.time()
        with conn:
            conn.execute(
                "INSERT INTO artifacts (tag, step, seq_index, wall_time, kind, mime_type, blob_key, width, height, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("img", 3, 0, now, "image", "image/png", "keep.png", 64, 64, "{}"),
            )
            conn.execute(
                "INSERT INTO artifacts (tag, step, seq_index, wall_time, kind, mime_type, blob_key, width, height, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("img", 7, 0, now, "image", "image/png", "orphan.png", 64, 64, "{}"),
            )

        # Create blob files
        for name in ("keep.png", "orphan.png"):
            with open(os.path.join(blobs_dir, name), "wb") as f:
                f.write(b"FAKE")

        guard = SessionGuard(conn, "new-session", resume_step=5, blobs_dir=blobs_dir)
        guard.initialize()

        # keep.png (step 3) should still exist
        assert os.path.exists(os.path.join(blobs_dir, "keep.png"))
        # orphan.png (step 7) should be removed
        assert not os.path.exists(os.path.join(blobs_dir, "orphan.png"))
