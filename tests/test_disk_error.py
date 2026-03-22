"""Tests for error handling in the write pipeline: retries, backoff, sticky errors."""
from __future__ import annotations

import os
import queue
import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

from serenityboard.writer.async_writer import (
    _SHUTDOWN,
    WriteItem,
    WriterError,
    _WriterThread,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scalar_item(tag: str, step: int, value: float) -> WriteItem:
    return WriteItem(
        table="scalars",
        params=(tag, step, time.time(), value),
        step=step,
    )


def _db_path_for(tmp: str) -> str:
    return os.path.join(tmp, "board.db")


def _start_writer(
    db_path: str,
    q: queue.Queue,
    flush_event: threading.Event,
    flush_interval: float = 5.0,
    session_id: str = "test-session",
) -> _WriterThread:
    ready = threading.Event()
    wt = _WriterThread(
        db_path=db_path,
        session_id=session_id,
        resume_step=None,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOperationalErrorRetry:
    """OperationalError triggers retry with backoff delays."""

    def test_retries_on_operational_error(self, tmp_logdir: str) -> None:
        """Verify 3 attempts when OperationalError is raised."""
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        call_count = 0
        original_insert = wt._insert

        def failing_insert(item):
            nonlocal call_count
            if item.table == "scalars":
                call_count += 1
                raise sqlite3.OperationalError("database is locked")
            return original_insert(item)

        wt._insert = failing_insert

        with patch("serenityboard.writer.async_writer.time.sleep"):
            q.put(_make_scalar_item("loss", 0, 1.0))
            q.put(_SHUTDOWN)
            wt.join(timeout=10.0)

        assert not wt.is_alive()
        assert call_count == 3  # MAX_RETRIES = 3
        assert wt.sticky_error is not None

    def test_backoff_delays(self, tmp_logdir: str) -> None:
        """Verify backoff delays are [0.1, 0.5] (2 sleeps before final failure)."""
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        def always_fail_insert(item):
            raise sqlite3.OperationalError("database is locked")

        wt._insert = always_fail_insert

        with patch("serenityboard.writer.async_writer.time.sleep") as mock_sleep:
            q.put(_make_scalar_item("loss", 0, 1.0))
            q.put(_SHUTDOWN)
            wt.join(timeout=10.0)

        # 2 sleeps: after attempt 1 (0.1s) and attempt 2 (0.5s); no sleep after final failure
        assert mock_sleep.call_count == 2
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [0.1, 0.5]


class TestStickyError:
    """After 3 failures, sticky error is set."""

    def test_sticky_error_after_max_retries(self, tmp_logdir: str) -> None:
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        def always_fail_insert(item):
            raise sqlite3.OperationalError("database is locked")

        wt._insert = always_fail_insert

        with patch("serenityboard.writer.async_writer.time.sleep"):
            q.put(_make_scalar_item("loss", 0, 1.0))
            q.put(_SHUTDOWN)
            wt.join(timeout=10.0)

        assert wt.sticky_error is not None
        assert isinstance(wt.sticky_error, WriterError)
        assert "3" in str(wt.sticky_error)

    def test_sticky_error_contains_step_info(self, tmp_logdir: str) -> None:
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        def always_fail_insert(item):
            raise sqlite3.OperationalError("disk I/O error")

        wt._insert = always_fail_insert

        with patch("serenityboard.writer.async_writer.time.sleep"):
            q.put(_make_scalar_item("loss", 42, 1.0))
            q.put(_SHUTDOWN)
            wt.join(timeout=10.0)

        assert wt.sticky_error is not None
        assert "42" in str(wt.sticky_error)


class TestRecoveryOnRetry:
    """If retry succeeds on attempt 2, no sticky error."""

    def test_recovery_on_second_attempt(self, tmp_logdir: str) -> None:
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        original_insert = wt._insert
        attempt = [0]

        def fail_once_insert(item):
            if item.table == "scalars":
                attempt[0] += 1
                if attempt[0] == 1:
                    raise sqlite3.OperationalError("database is locked")
            return original_insert(item)

        wt._insert = fail_once_insert

        with patch("serenityboard.writer.async_writer.time.sleep"):
            q.put(_make_scalar_item("loss", 0, 1.0))
            q.put(_SHUTDOWN)
            wt.join(timeout=10.0)

        assert not wt.is_alive()
        assert wt.sticky_error is None  # recovered on retry

        # Data was written
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT step, value FROM scalars WHERE tag = 'loss'"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == (0, 1.0)


class TestNonRetryableError:
    """Non-OperationalError sets sticky error immediately (no retry)."""

    def test_non_retryable_error(self, tmp_logdir: str) -> None:
        db_path = _db_path_for(tmp_logdir)
        q: queue.Queue = queue.Queue()
        flush_event = threading.Event()

        wt = _start_writer(db_path, q, flush_event)

        bad_item = WriteItem(
            table="__nonexistent__",
            params=("x",),
            step=99,
        )

        with patch("serenityboard.writer.async_writer.time.sleep") as mock_sleep:
            q.put(bad_item)
            q.put(_SHUTDOWN)
            wt.join(timeout=5.0)

        assert wt.sticky_error is not None
        # No retries for non-OperationalError
        mock_sleep.assert_not_called()
