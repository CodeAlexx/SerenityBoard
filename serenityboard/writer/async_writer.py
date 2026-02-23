"""Async writer thread for SerenityBoard storage."""
from __future__ import annotations

import dataclasses
import logging
import queue
import sqlite3
import threading
import time

from serenityboard.writer.schema import create_tables, set_pragmas
from serenityboard.writer.session_guard import SessionGuard

__all__ = ["WriterError", "WriteItem", "_WriterThread"]

logger = logging.getLogger(__name__)

# Sentinels for queue communication
_SHUTDOWN = object()
_MARK_COMPLETE = object()


class WriterError(Exception):
    pass


@dataclasses.dataclass
class WriteItem:
    """A queued write operation."""

    table: str  # "scalars", "text_events", "artifacts", "tensors", "trace_events", "eval_results", "hparam_metrics", "plugin_data", "metadata", "custom_scalar_layouts", "pr_curves", "audio", "graphs", "embeddings", "meshes"
    params: tuple  # parameters for INSERT
    step: int | None  # for error reporting


class _WriterThread(threading.Thread):
    """Background thread that owns the SQLite connection and commits batches.

    Communication with the main thread is via a ``queue.Queue``.  Sentinels:
    - ``_SHUTDOWN``: exit the processing loop
    - ``_MARK_COMPLETE``: call session_guard.mark_complete() on the writer thread

    A ``ready_event`` is set once initialization completes (or fails), so the
    main thread can detect init errors like ``ValueError`` from ``SessionGuard``.
    """

    MAX_RETRIES = 3
    RETRY_BACKOFF = [0.1, 0.5, 2.0]

    def __init__(
        self,
        db_path: str,
        session_id: str,
        resume_step: int | None,
        q: queue.Queue,
        flush_event: threading.Event,
        flush_interval: float,
        blob_storage: object | None,
        ready_event: threading.Event | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._db_path = db_path
        self._session_id = session_id
        self._resume_step = resume_step
        self._queue = q
        self._flush_event = flush_event
        self._flush_interval = flush_interval
        self._blob_storage = blob_storage
        self._ready_event = ready_event
        self._conn: sqlite3.Connection | None = None
        self._sticky_error: WriterError | None = None
        self._init_error: Exception | None = None
        self._session_guard: SessionGuard | None = None

    @property
    def sticky_error(self) -> WriterError | None:
        return self._sticky_error

    @property
    def init_error(self) -> Exception | None:
        return self._init_error

    # ── thread entry point ───────────────────────────────────────────

    def run(self) -> None:
        try:
            self._conn = sqlite3.connect(str(self._db_path))
            set_pragmas(self._conn)
            create_tables(self._conn)
            blobs_dir = None
            if self._blob_storage is not None:
                blobs_dir = getattr(self._blob_storage, '_blobs_dir', None)
            guard = SessionGuard(self._conn, self._session_id, self._resume_step, blobs_dir=blobs_dir)
            guard.initialize()
            self._session_guard = guard
        except Exception as exc:
            self._init_error = exc
            if self._ready_event:
                self._ready_event.set()
            return

        if self._ready_event:
            self._ready_event.set()

        try:
            self._process_loop()
        finally:
            try:
                if self._conn:
                    self._conn.close()
            except Exception:
                pass

    # ── main loop ────────────────────────────────────────────────────

    def _process_loop(self) -> None:
        """Drain queue, commit in batches, until shutdown sentinel."""
        while True:
            batch, shutdown, got_count = self._drain_queue()
            if batch:
                self._commit_batch(batch)
            # Mark all pulled items as done *after* commit so that
            # queue.join() only returns once data is persisted.
            for _ in range(got_count):
                self._queue.task_done()
            if shutdown or self._sticky_error:
                # Drain remaining items so flush()/close() don't deadlock
                self._drain_remaining()
                break

    def _drain_remaining(self) -> None:
        """Drain and discard all remaining queue items, calling task_done for each.

        Prevents deadlock when the writer thread exits due to sticky error
        or shutdown — any items left in the queue would block queue.join() forever.
        """
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _drain_queue(self) -> tuple[list[WriteItem], bool, int]:
        """Collect items from the queue.

        Blocks on the first item up to ``flush_interval`` seconds, then
        drains any remaining items non-blocking.

        Returns ``(batch, shutdown, got_count)`` where *shutdown* is True
        when the shutdown sentinel was received and *got_count* is the total
        number of ``queue.get`` calls (including sentinels).  The caller
        must call ``task_done`` that many times after processing.
        """
        batch: list[WriteItem] = []
        shutdown = False
        got_count = 0

        # Block for the first item (or until timeout).
        try:
            item = self._queue.get(timeout=self._flush_interval)
        except queue.Empty:
            return batch, False, 0

        got_count += 1
        if item is _SHUTDOWN:
            return batch, True, got_count
        if item is _MARK_COMPLETE:
            self._do_mark_complete()
            return batch, False, got_count
        if item is None:
            # Legacy shutdown sentinel (backwards compat with tests)
            return batch, True, got_count

        batch.append(item)

        # Non-blocking drain of remaining items.
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            got_count += 1
            if item is _SHUTDOWN or item is None:
                shutdown = True
                break
            if item is _MARK_COMPLETE:
                self._do_mark_complete()
                continue
            batch.append(item)

        return batch, shutdown, got_count

    # ── commit with retry ────────────────────────────────────────────

    def _commit_batch(self, batch: list[WriteItem]) -> None:
        """Commit a batch with retry + backoff. Sets sticky error on final failure."""
        for attempt in range(self.MAX_RETRIES):
            try:
                with self._conn:
                    for item in batch:
                        self._insert(item)
                return  # success
            except sqlite3.OperationalError as exc:
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF[attempt]
                    logger.error(
                        "SerenityBoard: commit failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs. Batch of %d items held in memory.",
                        attempt + 1, self.MAX_RETRIES, exc, wait, len(batch),
                    )
                    time.sleep(wait)
                else:
                    self._set_sticky_error(batch, exc)
                    return
            except Exception as exc:
                # Non-retryable error (IntegrityError, ProgrammingError, etc.)
                self._set_sticky_error(batch, exc)
                return

    def _set_sticky_error(self, batch: list[WriteItem], exc: Exception) -> None:
        """Set the sticky error with diagnostic info."""
        steps = [i.step for i in batch if i.step is not None]
        if steps:
            step_info = f"steps {min(steps)}-{max(steps)}"
        else:
            step_info = "metadata only"
        self._sticky_error = WriterError(
            f"SerenityBoard: commit failed after {self.MAX_RETRIES} "
            f"attempts. Last error: {exc}. "
            f"Lost batch: {len(batch)} items, {step_info}."
        )
        logger.critical(str(self._sticky_error))

    def _insert(self, item: WriteItem) -> None:
        """Insert a single WriteItem into the database."""
        table = item.table
        p = item.params

        if table == "scalars":
            self._conn.execute(
                "INSERT OR REPLACE INTO scalars "
                "(tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                p,
            )
        elif table == "text_events":
            self._conn.execute(
                "INSERT OR REPLACE INTO text_events "
                "(tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                p,
            )
        elif table == "artifacts":
            self._conn.execute(
                "INSERT OR REPLACE INTO artifacts "
                "(tag, step, seq_index, wall_time, kind, mime_type, blob_key, width, height, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                p,
            )
        elif table == "trace_events":
            self._conn.execute(
                "INSERT OR REPLACE INTO trace_events "
                "(step, wall_time, phase, duration_ms, details) "
                "VALUES (?, ?, ?, ?, ?)",
                p,
            )
        elif table == "eval_results":
            self._conn.execute(
                "INSERT OR REPLACE INTO eval_results "
                "(suite_name, case_id, step, wall_time, score_name, score_value, artifact_key, details) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                p,
            )
        elif table == "tensors":
            self._conn.execute(
                "INSERT OR REPLACE INTO tensors "
                "(tag, step, wall_time, dtype, shape, data) VALUES (?, ?, ?, ?, ?, ?)",
                p,
            )
        elif table == "hparam_metrics":
            self._conn.execute(
                "INSERT OR REPLACE INTO hparam_metrics "
                "(metric_tag, value, step, wall_time) VALUES (?, ?, ?, ?)",
                p,
            )
        elif table == "metadata":
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                p,
            )
        elif table == "plugin_data":
            self._conn.execute(
                "INSERT OR REPLACE INTO plugin_data "
                "(plugin_name, tag, step, wall_time, data) VALUES (?, ?, ?, ?, ?)",
                p,
            )
        elif table == "custom_scalar_layouts":
            self._conn.execute(
                "INSERT OR REPLACE INTO custom_scalar_layouts "
                "(layout_name, config) VALUES (?, ?)",
                p,
            )
        elif table == "pr_curves":
            self._conn.execute(
                "INSERT OR REPLACE INTO pr_curves "
                "(tag, step, class_index, wall_time, num_thresholds, data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                p,
            )
        elif table == "audio":
            self._conn.execute(
                "INSERT OR REPLACE INTO audio "
                "(tag, step, seq_index, wall_time, blob_key, sample_rate, num_channels, duration_ms, mime_type, label) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                p,
            )
        elif table == "graphs":
            self._conn.execute(
                "INSERT OR REPLACE INTO graphs "
                "(tag, step, wall_time, graph_blob_key) VALUES (?, ?, ?, ?)",
                p,
            )
        elif table == "embeddings":
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(tag, step, wall_time, num_points, dimensions, tensor_blob_key, "
                "metadata_json, metadata_header, sprite_blob_key, sprite_single_h, sprite_single_w) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                p,
            )
        elif table == "meshes":
            self._conn.execute(
                "INSERT OR REPLACE INTO meshes "
                "(tag, step, wall_time, num_vertices, has_faces, has_colors, num_faces, "
                "vertices_blob_key, faces_blob_key, colors_blob_key, config_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                p,
            )
        else:
            raise ValueError(f"Unknown table: {table!r}")

    # ── called from writer thread via queue sentinel ──────────────────

    def _do_mark_complete(self) -> None:
        """Mark the session as complete. Runs on the writer thread."""
        if self._session_guard is not None:
            self._session_guard.mark_complete()

    # ── called from main thread ──────────────────────────────────────

    def request_mark_complete(self) -> None:
        """Enqueue a mark_complete request to be executed on the writer thread."""
        self._queue.put(_MARK_COMPLETE)
