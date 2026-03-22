"""Tests for _WriterThread internals: retry, sticky errors, shutdown, draining."""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time

import pytest

from serenityboard.writer.async_writer import (
    _MARK_COMPLETE,
    _SHUTDOWN,
    WriteItem,
    WriterError,
    _WriterThread,
)
from serenityboard.writer.schema import create_tables, set_pragmas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scalar_item(tag: str, step: int, value: float) -> WriteItem:
    """Create a valid WriteItem for the scalars table."""
    return WriteItem(
        table="scalars",
        params=(tag, step, time.time(), value),
        step=step,
    )


def _make_bad_item(step: int = 0) -> WriteItem:
    """Create a WriteItem targeting a non-existent table to trigger ValueError."""
    return WriteItem(
        table="__nonexistent_table__",
        params=("x",),
        step=step,
    )


def _start_writer(
    db_path: str,
    q: queue.Queue,
    flush_event: threading.Event,
    flush_interval: float = 0.05,
    session_id: str = "test-session",
    resume_step: int | None = None,
) -> _WriterThread:
    """Create, start, and wait for a _WriterThread to finish initialisation."""
    ready = threading.Event()
    wt = _WriterThread(
        db_path=db_path,
        session_id=session_id,
        resume_step=resume_step,
        q=q,
        flush_event=flush_event,
        flush_interval=flush_interval,
        blob_storage=None,
        ready_event=ready,
    )
    wt.start()
    ready.wait(timeout=5.0)
    assert wt.init_error is None, f"Writer init failed: {wt.init_error}"
    return wt


def _db_path_for(tmp: str) -> str:
    return os.path.join(tmp, "board.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStickyErrorPropagation:
    """Force a commit failure and verify sticky_error is set."""

    def test_sticky_error_propagation(self, tmp_logdir: str) -> None:
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        # Put a bad item (unknown table) -- triggers ValueError inside _insert,
        # which is a non-retryable error caught by the generic except branch.
        bad_item = _make_bad_item(step=42)
        q.put(bad_item)
        q.put(_SHUTDOWN)
        wt.join(timeout=5.0)

        assert not wt.is_alive()
        assert wt.sticky_error is not None
        assert isinstance(wt.sticky_error, WriterError)
        assert "42" in str(wt.sticky_error) or "nonexistent" in str(wt.sticky_error).lower()


class TestFlushWithPendingWrites:
    """Put multiple items then shutdown -- verify they are committed."""

    def test_flush_with_pending_writes(self, tmp_logdir: str) -> None:
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        num_items = 10
        for step in range(num_items):
            q.put(_make_scalar_item("loss", step, 1.0 - step * 0.1))

        q.put(_SHUTDOWN)
        wt.join(timeout=5.0)

        assert not wt.is_alive()
        assert wt.sticky_error is None

        # Verify data was committed
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT step, value FROM scalars WHERE tag = 'loss' ORDER BY step"
        ).fetchall()
        conn.close()

        assert len(rows) == num_items
        assert rows[0] == (0, 1.0)
        assert rows[-1][0] == num_items - 1


class TestGracefulShutdown:
    """Shutdown sentinel causes clean exit with committed data."""

    def test_graceful_shutdown(self, tmp_logdir: str) -> None:
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        # Write a few items
        for step in range(3):
            q.put(_make_scalar_item("metric/a", step, float(step)))

        q.put(_SHUTDOWN)
        wt.join(timeout=5.0)

        assert not wt.is_alive(), "Writer thread did not exit"
        assert wt.sticky_error is None, f"Unexpected error: {wt.sticky_error}"

        # Confirm data persisted
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT step FROM scalars WHERE tag = 'metric/a' ORDER BY step"
        ).fetchall()
        conn.close()

        assert [r[0] for r in rows] == [0, 1, 2]


class TestMarkCompleteSentinel:
    """_MARK_COMPLETE sets session status to 'complete' in the DB."""

    def test_mark_complete_sentinel(self, tmp_logdir: str) -> None:
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        # Write some data so we know the thread is processing
        q.put(_make_scalar_item("x", 0, 1.0))
        q.put(_MARK_COMPLETE)
        q.put(_SHUTDOWN)
        wt.join(timeout=5.0)

        assert not wt.is_alive()
        assert wt.sticky_error is None

        # Verify session status is 'complete'
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT status FROM sessions WHERE session_id = 'test-session'"
        ).fetchone()
        meta_row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'status'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "complete"
        assert meta_row is not None
        assert json.loads(meta_row[0]) == "complete"


class TestDrainRemainingOnError:
    """After a sticky error, remaining items are drained so queue.join() returns."""

    def test_drain_remaining_on_error(self, tmp_logdir: str) -> None:
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        # Use a long flush_interval so the writer blocks on the first get(),
        # giving us time to fill the queue with all items before it wakes up.
        wt = _start_writer(db_path, q, flush_event, flush_interval=5.0)

        # Put the bad item (will be picked up as the first item of a batch),
        # followed by good items that will be drained non-blocking in the
        # same _drain_queue call, then the shutdown sentinel.
        q.put(_make_bad_item(step=0))
        for step in range(5):
            q.put(_make_scalar_item("after_error", step, float(step)))
        q.put(_SHUTDOWN)

        wt.join(timeout=10.0)

        assert not wt.is_alive(), "Writer thread did not exit"
        assert wt.sticky_error is not None, "Expected sticky error from bad item"

        # The critical check: queue.join() must not deadlock because
        # _drain_remaining called task_done for every item left in the queue.
        # We give it a short timeout -- if it blocks, the test fails.
        join_done = threading.Event()

        def try_join():
            q.join()
            join_done.set()

        t = threading.Thread(target=try_join, daemon=True)
        t.start()
        success = join_done.wait(timeout=3.0)
        assert success, "queue.join() deadlocked -- _drain_remaining did not call task_done for all items"
