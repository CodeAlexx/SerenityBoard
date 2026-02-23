"""Ops database connection manager for SerenityBoard."""
from __future__ import annotations

import os
import sqlite3
import threading

from serenityboard.ops.schema import create_ops_tables, set_ops_pragmas

__all__ = ["OpsDB"]


class OpsDB:
    """Manages the ops.db connection at ``<logdir>/_ops/ops.db``.

    Thread-safe: each call creates or reuses a thread-local connection.
    """

    def __init__(self, logdir: str) -> None:
        self._ops_dir = os.path.join(logdir, "_ops")
        os.makedirs(self._ops_dir, exist_ok=True)
        self._db_path = os.path.join(self._ops_dir, "ops.db")
        self._local = threading.local()
        # Initialize on main thread
        self._ensure_conn()

    def _ensure_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            set_ops_pragmas(conn)
            create_ops_tables(conn)
            self._local.conn = conn
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._ensure_conn()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params_seq)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
