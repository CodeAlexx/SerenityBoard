"""Session lifecycle management and restart purges."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time

__all__ = ["SessionGuard"]

logger = logging.getLogger(__name__)


class SessionGuard:
    """Manages session lifecycle and restart purges.

    All operations happen on the writer thread's connection.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        resume_step: int | None,
        blobs_dir: str | None = None,
    ) -> None:
        self._conn = conn
        self._session_id = session_id
        self._resume_step = resume_step
        self._blobs_dir = blobs_dir

    def initialize(self) -> None:
        """Called once during writer thread startup, before any data writes."""
        existing_session = self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'active_session_id'"
        ).fetchone()

        if existing_session is not None:
            old_session_id = json.loads(existing_session[0])
            if self._resume_step is None:
                raise ValueError(
                    "Existing run with active session found. "
                    "Provide resume_step= to continue, or use a new run_name."
                )
            self._purge_and_transition(old_session_id)
        else:
            self._create_first_session()

    def _purge_and_transition(self, old_session_id: str) -> None:
        """Atomic: mark old session, purge orphaned data, start new session."""
        # Collect blob keys that will be orphaned before deleting rows
        orphaned_blob_keys: list[str] = []
        try:
            rows = self._conn.execute(
                "SELECT blob_key FROM artifacts WHERE step > ?",
                (self._resume_step,),
            ).fetchall()
            orphaned_blob_keys = [r[0] for r in rows]
        except Exception:
            pass  # Table may not exist in V1 schema
        # Also collect orphaned embedding blobs (tensor + sprite)
        try:
            rows = self._conn.execute(
                "SELECT tensor_blob_key FROM embeddings WHERE step > ?",
                (self._resume_step,),
            ).fetchall()
            orphaned_blob_keys.extend(r[0] for r in rows)
            rows = self._conn.execute(
                "SELECT sprite_blob_key FROM embeddings WHERE step > ? AND sprite_blob_key IS NOT NULL",
                (self._resume_step,),
            ).fetchall()
            orphaned_blob_keys.extend(r[0] for r in rows)
        except Exception:
            pass  # Table may not exist in older databases
        # Also collect orphaned mesh blobs (vertices, faces, colors)
        try:
            rows = self._conn.execute(
                "SELECT vertices_blob_key FROM meshes WHERE step > ?",
                (self._resume_step,),
            ).fetchall()
            orphaned_blob_keys.extend(r[0] for r in rows)
            rows = self._conn.execute(
                "SELECT faces_blob_key FROM meshes WHERE step > ? AND faces_blob_key IS NOT NULL",
                (self._resume_step,),
            ).fetchall()
            orphaned_blob_keys.extend(r[0] for r in rows)
            rows = self._conn.execute(
                "SELECT colors_blob_key FROM meshes WHERE step > ? AND colors_blob_key IS NOT NULL",
                (self._resume_step,),
            ).fetchall()
            orphaned_blob_keys.extend(r[0] for r in rows)
        except Exception:
            pass  # Table may not exist in older databases
        # Also collect orphaned audio blobs
        try:
            rows = self._conn.execute(
                "SELECT blob_key FROM audio WHERE step > ?",
                (self._resume_step,),
            ).fetchall()
            orphaned_blob_keys.extend(r[0] for r in rows)
        except Exception:
            pass  # Table may not exist in older databases
        # Also collect orphaned graph blobs
        try:
            rows = self._conn.execute(
                "SELECT graph_blob_key FROM graphs WHERE step > ?",
                (self._resume_step,),
            ).fetchall()
            orphaned_blob_keys.extend(
                r[0] for r in rows if r[0] and not r[0].startswith("{")
            )
        except Exception:
            pass  # Table may not exist or column may have old name

        with self._conn:  # single transaction
            self._conn.execute(
                "UPDATE sessions SET status = 'crashed' "
                "WHERE session_id = ? AND status = 'running'",
                (old_session_id,),
            )
            tables = ["scalars", "tensors", "artifacts", "text_events", "trace_events", "eval_results", "plugin_data", "graphs", "embeddings", "meshes", "audio", "pr_curves"]
            total_purged = 0
            for table in tables:
                cursor = self._conn.execute(
                    f"DELETE FROM {table} WHERE step > ?",  # noqa: S608
                    (self._resume_step,),
                )
                total_purged += cursor.rowcount
            self._conn.execute(
                "INSERT INTO sessions "
                "(session_id, start_time, resume_step, status) "
                "VALUES (?, ?, ?, 'running')",
                (self._session_id, time.time(), self._resume_step),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("active_session_id", json.dumps(self._session_id)),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("status", json.dumps("running")),
            )

        if total_purged > 0:
            logger.warning(
                "Session transition: purged %d orphaned rows above step %d. "
                "Old session: %s → New session: %s",
                total_purged,
                self._resume_step,
                old_session_id,
                self._session_id,
            )

        # Clean up orphaned blob files (best-effort, after transaction commits)
        if orphaned_blob_keys and self._blobs_dir:
            cleaned = 0
            for key in orphaned_blob_keys:
                path = os.path.join(self._blobs_dir, key)
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        cleaned += 1
                except OSError:
                    logger.debug("Failed to remove orphaned blob: %s", path)
            if cleaned > 0:
                logger.info("Cleaned %d orphaned blob files", cleaned)

    def _create_first_session(self) -> None:
        """First session for a new run."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO sessions "
                "(session_id, start_time, resume_step, status) "
                "VALUES (?, ?, NULL, 'running')",
                (self._session_id, time.time()),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("active_session_id", json.dumps(self._session_id)),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("status", json.dumps("running")),
            )

    def mark_complete(self) -> None:
        """Called by writer.close()."""
        with self._conn:
            self._conn.execute(
                "UPDATE sessions SET status = 'complete' "
                "WHERE session_id = ?",
                (self._session_id,),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("status", json.dumps("complete")),
            )
