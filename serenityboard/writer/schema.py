"""SQLite schema for SerenityBoard storage."""
from __future__ import annotations

import sqlite3

__all__ = ["create_tables", "set_pragmas"]

_DDL_V2 = """
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT    NOT NULL,
    start_time   REAL    NOT NULL,
    resume_step  INTEGER,
    status       TEXT    NOT NULL CHECK(status IN ('running','complete','crashed')),
    PRIMARY KEY (session_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS scalars (
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    value     REAL    NOT NULL,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS tensors (
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    dtype     TEXT    NOT NULL,
    shape     TEXT    NOT NULL,
    data      BLOB    NOT NULL,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS artifacts (
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    seq_index INTEGER NOT NULL DEFAULT 0,
    wall_time REAL    NOT NULL,
    kind      TEXT    NOT NULL,
    mime_type TEXT    NOT NULL,
    blob_key  TEXT    NOT NULL,
    width     INTEGER,
    height    INTEGER,
    meta      TEXT    NOT NULL DEFAULT '{}',
    PRIMARY KEY (tag, step, seq_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS text_events (
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    value     TEXT    NOT NULL,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS trace_events (
    step        INTEGER NOT NULL,
    wall_time   REAL    NOT NULL,
    phase       TEXT    NOT NULL,
    duration_ms REAL    NOT NULL,
    details     TEXT    NOT NULL DEFAULT '{}',
    PRIMARY KEY (step, phase)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS eval_results (
    suite_name   TEXT    NOT NULL,
    case_id      TEXT    NOT NULL,
    step         INTEGER NOT NULL,
    wall_time    REAL    NOT NULL,
    score_name   TEXT    NOT NULL,
    score_value  REAL    NOT NULL,
    artifact_key TEXT,
    details      TEXT    NOT NULL DEFAULT '{}',
    PRIMARY KEY (suite_name, case_id, step, score_name)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS hparam_metrics (
    metric_tag TEXT    NOT NULL,
    value      REAL    NOT NULL,
    step       INTEGER,
    wall_time  REAL,
    PRIMARY KEY (metric_tag)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS plugin_data (
    plugin_name TEXT    NOT NULL,
    tag         TEXT    NOT NULL,
    step        INTEGER NOT NULL,
    wall_time   REAL    NOT NULL,
    data        TEXT    NOT NULL,
    PRIMARY KEY (plugin_name, tag, step)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS custom_scalar_layouts (
    layout_name TEXT NOT NULL,
    config TEXT NOT NULL,
    PRIMARY KEY (layout_name)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS pr_curves (
    tag TEXT NOT NULL, step INTEGER NOT NULL, class_index INTEGER NOT NULL DEFAULT 0,
    wall_time REAL NOT NULL,
    num_thresholds INTEGER NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (tag, step, class_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS audio (
    tag TEXT NOT NULL, step INTEGER NOT NULL, seq_index INTEGER NOT NULL DEFAULT 0,
    wall_time REAL NOT NULL, blob_key TEXT NOT NULL,
    sample_rate INTEGER NOT NULL, num_channels INTEGER NOT NULL DEFAULT 1,
    duration_ms REAL, mime_type TEXT NOT NULL DEFAULT 'audio/wav',
    label TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tag, step, seq_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS graphs (
    tag            TEXT    NOT NULL,
    step           INTEGER NOT NULL,
    wall_time      REAL    NOT NULL,
    graph_blob_key TEXT    NOT NULL,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS embeddings (
    tag              TEXT    NOT NULL,
    step             INTEGER NOT NULL,
    wall_time        REAL    NOT NULL,
    num_points       INTEGER NOT NULL,
    dimensions       INTEGER NOT NULL,
    tensor_blob_key  TEXT    NOT NULL,
    metadata_json    TEXT,
    metadata_header  TEXT,
    sprite_blob_key  TEXT,
    sprite_single_h  INTEGER,
    sprite_single_w  INTEGER,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS meshes (
    tag               TEXT    NOT NULL,
    step              INTEGER NOT NULL,
    wall_time         REAL    NOT NULL,
    num_vertices      INTEGER NOT NULL,
    has_faces         INTEGER NOT NULL DEFAULT 0,
    has_colors        INTEGER NOT NULL DEFAULT 0,
    num_faces         INTEGER NOT NULL DEFAULT 0,
    vertices_blob_key TEXT    NOT NULL,
    faces_blob_key    TEXT,
    colors_blob_key   TEXT,
    config_json       TEXT,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_scalars_tag ON scalars(tag);
CREATE INDEX IF NOT EXISTS idx_scalars_tag_step ON scalars(tag, step DESC);
CREATE INDEX IF NOT EXISTS idx_tensors_tag_step ON tensors(tag, step DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_tag_step ON artifacts(tag, step DESC);
CREATE INDEX IF NOT EXISTS idx_text_tag_step ON text_events(tag, step DESC);
CREATE INDEX IF NOT EXISTS idx_eval_suite_step ON eval_results(suite_name, step DESC);
CREATE INDEX IF NOT EXISTS idx_plugin_name_tag ON plugin_data(plugin_name, tag);
CREATE INDEX IF NOT EXISTS idx_pr_curves_tag_step ON pr_curves(tag, step DESC);
CREATE INDEX IF NOT EXISTS idx_audio_tag_step ON audio(tag, step DESC);
CREATE INDEX IF NOT EXISTS idx_graphs_tag_step ON graphs(tag, step DESC);
CREATE INDEX IF NOT EXISTS idx_embeddings_tag_step ON embeddings(tag, step DESC);
CREATE INDEX IF NOT EXISTS idx_meshes_tag_step ON meshes(tag, step DESC);
"""


def set_pragmas(conn: sqlite3.Connection) -> None:
    """Set WAL mode, NORMAL synchronous, and busy timeout."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")


def _is_v1_schema(conn: sqlite3.Connection) -> bool:
    """Detect V1 schema by checking for the 'blobs' table."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='blobs'"
    ).fetchone()
    return row is not None


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """In-place migration from V1 (blobs/text) to V2 (artifacts/text_events).

    Uses ALTER TABLE RENAME (SQLite 3.25+, Python 3.10 ships 3.37+).
    """
    with conn:
        # Rename tables
        conn.execute("ALTER TABLE blobs RENAME TO artifacts")
        conn.execute("ALTER TABLE text RENAME TO text_events")

        # Add new columns to artifacts
        conn.execute("ALTER TABLE artifacts ADD COLUMN kind TEXT NOT NULL DEFAULT 'image'")
        conn.execute("ALTER TABLE artifacts ADD COLUMN meta TEXT NOT NULL DEFAULT '{}'")

        # Create new tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trace_events (
                step        INTEGER NOT NULL,
                wall_time   REAL    NOT NULL,
                phase       TEXT    NOT NULL,
                duration_ms REAL    NOT NULL,
                details     TEXT    NOT NULL DEFAULT '{}',
                PRIMARY KEY (step, phase)
            ) WITHOUT ROWID
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                suite_name   TEXT    NOT NULL,
                case_id      TEXT    NOT NULL,
                step         INTEGER NOT NULL,
                wall_time    REAL    NOT NULL,
                score_name   TEXT    NOT NULL,
                score_value  REAL    NOT NULL,
                artifact_key TEXT,
                details      TEXT    NOT NULL DEFAULT '{}',
                PRIMARY KEY (suite_name, case_id, step, score_name)
            ) WITHOUT ROWID
        """)

        # Drop old indexes and create new ones
        conn.execute("DROP INDEX IF EXISTS idx_blobs_tag")
        conn.execute("DROP INDEX IF EXISTS idx_text_tag")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_tag_step ON artifacts(tag, step DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_text_tag_step ON text_events(tag, step DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_suite_step ON eval_results(suite_name, step DESC)")


def _needs_v3_migration(conn: sqlite3.Connection) -> bool:
    """Detect V2 schema that needs V3 migration.

    V3 renames graphs.graph_data -> graph_json and
    embeddings.blob_key -> tensor_blob_key, and adds the meshes table.
    Returns True if graphs table exists with the old 'graph_data' column.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='graphs'"
    ).fetchone()
    if row is None:
        return False
    # Check column names via PRAGMA
    cols = conn.execute("PRAGMA table_info(graphs)").fetchall()
    col_names = {c[1] for c in cols}
    return "graph_data" in col_names


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """In-place migration from V2 to V3 schema.

    - Renames graphs.graph_data -> graph_json via table rebuild
    - Renames embeddings.blob_key -> tensor_blob_key and relaxes NOT NULL defaults
    - Creates the meshes table
    - Creates new indexes

    Uses ALTER TABLE RENAME COLUMN (SQLite 3.25+, Python 3.10 ships 3.37+).
    """
    with conn:
        # Rename graphs.graph_data -> graph_json
        conn.execute("ALTER TABLE graphs RENAME COLUMN graph_data TO graph_json")

        # Rename embeddings.blob_key -> tensor_blob_key (if embeddings exists)
        cols = conn.execute("PRAGMA table_info(embeddings)").fetchall()
        col_names = {c[1] for c in cols}
        if "blob_key" in col_names:
            conn.execute(
                "ALTER TABLE embeddings RENAME COLUMN blob_key TO tensor_blob_key"
            )

        # Create meshes table (IF NOT EXISTS for safety)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meshes (
                tag               TEXT    NOT NULL,
                step              INTEGER NOT NULL,
                wall_time         REAL    NOT NULL,
                num_vertices      INTEGER NOT NULL,
                has_faces         INTEGER NOT NULL DEFAULT 0,
                has_colors        INTEGER NOT NULL DEFAULT 0,
                num_faces         INTEGER NOT NULL DEFAULT 0,
                vertices_blob_key TEXT    NOT NULL,
                faces_blob_key    TEXT,
                colors_blob_key   TEXT,
                config_json       TEXT,
                PRIMARY KEY (tag, step)
            ) WITHOUT ROWID
        """)

        # Create new indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_tag_step "
            "ON embeddings(tag, step DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_meshes_tag_step "
            "ON meshes(tag, step DESC)"
        )


def _ensure_meshes_num_faces(conn: sqlite3.Connection) -> None:
    """Add num_faces column to meshes table if it doesn't exist yet."""
    try:
        cols = conn.execute("PRAGMA table_info(meshes)").fetchall()
    except Exception:
        return  # meshes table doesn't exist yet
    col_names = {c[1] for c in cols}
    if cols and "num_faces" not in col_names:
        with conn:
            conn.execute(
                "ALTER TABLE meshes ADD COLUMN num_faces INTEGER NOT NULL DEFAULT 0"
            )


def _needs_v4_migration(conn: sqlite3.Connection) -> bool:
    """Detect V3 schema that needs V4 migration (graphs.graph_json -> graph_blob_key)."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='graphs'"
    ).fetchone()
    if row is None:
        return False
    cols = conn.execute("PRAGMA table_info(graphs)").fetchall()
    col_names = {c[1] for c in cols}
    return "graph_json" in col_names


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Rename graphs.graph_json -> graph_blob_key.

    Existing rows keep their inline JSON values; the data_provider detects
    whether a value is a blob key or inline JSON at read time.

    Uses ALTER TABLE RENAME COLUMN (SQLite 3.25+, Python 3.10 ships 3.37+).
    """
    with conn:
        conn.execute("ALTER TABLE graphs RENAME COLUMN graph_json TO graph_blob_key")


def create_tables(conn: sqlite3.Connection) -> None:
    """Run all DDL statements. Idempotent.

    If a V1 database is detected (has 'blobs' table), migrate V1->V2 first.
    If a V2 database is detected (graphs.graph_data column), migrate V2->V3.
    If a V3 database is detected (graphs.graph_json column), migrate V3->V4.
    Fresh databases get the current schema directly.
    """
    if _is_v1_schema(conn):
        _migrate_v1_to_v2(conn)
    if _needs_v3_migration(conn):
        _migrate_v2_to_v3(conn)
    if _needs_v4_migration(conn):
        _migrate_v3_to_v4(conn)
    # Ensure num_faces column exists on meshes (for databases created before this column was added)
    _ensure_meshes_num_faces(conn)
    # Always run full DDL -- CREATE IF NOT EXISTS is idempotent for new tables
    conn.executescript(_DDL_V2)
