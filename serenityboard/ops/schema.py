"""SQLite DDL for SerenityBoard ops modules (Tables, Registry, Sweeps, Automations)."""
from __future__ import annotations

import sqlite3

__all__ = ["create_ops_tables", "set_ops_pragmas"]

_OPS_DDL = """
-- ── Tables Plugin ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tables_def (
  table_id      TEXT PRIMARY KEY,
  name          TEXT NOT NULL UNIQUE,
  description   TEXT NOT NULL DEFAULT '',
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL,
  schema_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tables_rows (
  row_id        TEXT PRIMARY KEY,
  table_id      TEXT NOT NULL,
  run_name      TEXT,
  step          INTEGER,
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL,
  values_json   TEXT NOT NULL,
  tags_json     TEXT NOT NULL DEFAULT '[]',
  FOREIGN KEY(table_id) REFERENCES tables_def(table_id)
);

CREATE TABLE IF NOT EXISTS tables_row_artifacts (
  row_id        TEXT NOT NULL,
  artifact_id   TEXT NOT NULL,
  role          TEXT NOT NULL,
  PRIMARY KEY(row_id, artifact_id, role)
);

CREATE INDEX IF NOT EXISTS idx_tables_rows_table_step
  ON tables_rows(table_id, step DESC);
CREATE INDEX IF NOT EXISTS idx_tables_rows_run_step
  ON tables_rows(run_name, step DESC);

-- ── Artifact Registry ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS artifact_collections (
  collection_id     TEXT PRIMARY KEY,
  name              TEXT NOT NULL UNIQUE,
  kind              TEXT NOT NULL,
  description       TEXT NOT NULL DEFAULT '',
  created_at        REAL NOT NULL,
  updated_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_versions (
  artifact_id       TEXT PRIMARY KEY,
  collection_id     TEXT NOT NULL,
  version_index     INTEGER NOT NULL,
  digest            TEXT NOT NULL,
  size_bytes        INTEGER NOT NULL,
  storage_uri       TEXT NOT NULL,
  metadata_json     TEXT NOT NULL DEFAULT '{}',
  created_by_run    TEXT,
  created_at        REAL NOT NULL,
  UNIQUE(collection_id, version_index),
  UNIQUE(collection_id, digest),
  FOREIGN KEY(collection_id) REFERENCES artifact_collections(collection_id)
);

CREATE TABLE IF NOT EXISTS artifact_aliases (
  collection_id     TEXT NOT NULL,
  alias             TEXT NOT NULL,
  artifact_id       TEXT NOT NULL,
  updated_at        REAL NOT NULL,
  PRIMARY KEY(collection_id, alias)
);

CREATE TABLE IF NOT EXISTS artifact_lineage (
  parent_artifact_id  TEXT NOT NULL,
  child_artifact_id   TEXT NOT NULL,
  relation            TEXT NOT NULL,
  created_at          REAL NOT NULL,
  PRIMARY KEY(parent_artifact_id, child_artifact_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_artifact_versions_collection
  ON artifact_versions(collection_id, version_index DESC);
CREATE INDEX IF NOT EXISTS idx_artifact_lineage_parent
  ON artifact_lineage(parent_artifact_id);
CREATE INDEX IF NOT EXISTS idx_artifact_lineage_child
  ON artifact_lineage(child_artifact_id);

-- ── Sweeps Service ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sweeps (
  sweep_id         TEXT PRIMARY KEY,
  name             TEXT NOT NULL UNIQUE,
  method           TEXT NOT NULL,
  objective_metric TEXT NOT NULL,
  objective_mode   TEXT NOT NULL,
  parameter_space  TEXT NOT NULL,
  max_trials       INTEGER NOT NULL,
  parallelism      INTEGER NOT NULL DEFAULT 1,
  status           TEXT NOT NULL,
  created_at       REAL NOT NULL,
  updated_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sweep_trials (
  trial_id         TEXT PRIMARY KEY,
  sweep_id         TEXT NOT NULL,
  run_name         TEXT,
  params_json      TEXT NOT NULL,
  status           TEXT NOT NULL,
  objective_value  REAL,
  started_at       REAL,
  finished_at      REAL,
  error_text       TEXT NOT NULL DEFAULT '',
  assigned_agent   TEXT,
  attempt_count    INTEGER NOT NULL DEFAULT 0,
  created_at       REAL NOT NULL,
  updated_at       REAL NOT NULL,
  FOREIGN KEY(sweep_id) REFERENCES sweeps(sweep_id)
);

CREATE TABLE IF NOT EXISTS sweep_agents (
  agent_id         TEXT PRIMARY KEY,
  labels_json      TEXT NOT NULL DEFAULT '[]',
  last_heartbeat   REAL NOT NULL,
  status           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trials_sweep_status
  ON sweep_trials(sweep_id, status);
CREATE INDEX IF NOT EXISTS idx_trials_sweep_objective
  ON sweep_trials(sweep_id, objective_value);

-- ── Automations Plugin ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS automation_rules (
  rule_id            TEXT PRIMARY KEY,
  name               TEXT NOT NULL UNIQUE,
  enabled            INTEGER NOT NULL,
  trigger_type       TEXT NOT NULL,
  trigger_json       TEXT NOT NULL,
  action_type        TEXT NOT NULL,
  action_json        TEXT NOT NULL,
  cooldown_seconds   INTEGER NOT NULL DEFAULT 0,
  dedupe_window_sec  INTEGER NOT NULL DEFAULT 300,
  created_at         REAL NOT NULL,
  updated_at         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS automation_events (
  event_id           TEXT PRIMARY KEY,
  event_type         TEXT NOT NULL,
  event_key          TEXT NOT NULL,
  payload_json       TEXT NOT NULL,
  created_at         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS automation_dispatch (
  dispatch_id        TEXT PRIMARY KEY,
  rule_id            TEXT NOT NULL,
  event_id           TEXT NOT NULL,
  status             TEXT NOT NULL,
  error_text         TEXT NOT NULL DEFAULT '',
  attempt_count      INTEGER NOT NULL DEFAULT 0,
  next_attempt_at    REAL,
  created_at         REAL NOT NULL,
  updated_at         REAL NOT NULL,
  UNIQUE(rule_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_automation_dispatch_status_next
  ON automation_dispatch(status, next_attempt_at);
"""


def set_ops_pragmas(conn: sqlite3.Connection) -> None:
    """Set WAL mode, NORMAL synchronous, and busy timeout for ops DB."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")


def create_ops_tables(conn: sqlite3.Connection) -> None:
    """Run all ops DDL statements. Idempotent."""
    conn.executescript(_OPS_DDL)
