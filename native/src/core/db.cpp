#include "serenityboard/db.hpp"

#include <sqlite3.h>

#include <cstring>

namespace sb {

namespace {

bool retryable_code(int code) {
  const int primary = code & 0xFF;
  return primary == SQLITE_BUSY || primary == SQLITE_LOCKED || primary == SQLITE_IOERR ||
         primary == SQLITE_FULL || primary == SQLITE_CANTOPEN || primary == SQLITE_PROTOCOL;
}

[[noreturn]] void throw_db(sqlite3 *db, int code, const char *context) {
  std::string message = context;
  message += ": ";
  message += db ? sqlite3_errmsg(db) : sqlite3_errstr(code);
  throw DbError(message, code, retryable_code(code));
}

struct Stmt {
  sqlite3_stmt *stmt{nullptr};
  ~Stmt() {
    if (stmt) sqlite3_finalize(stmt);
  }
};

void bind_all(sqlite3 *db, sqlite3_stmt *stmt, const std::vector<Param> &params) {
  int index = 1;
  for (const auto &p : params) {
    int rc = SQLITE_OK;
    switch (p.kind) {
      case Param::Kind::Null: rc = sqlite3_bind_null(stmt, index); break;
      case Param::Kind::Int: rc = sqlite3_bind_int64(stmt, index, p.i); break;
      case Param::Kind::Real: rc = sqlite3_bind_double(stmt, index, p.d); break;
      case Param::Kind::Text:
        rc = sqlite3_bind_text64(stmt, index, p.s.data(), p.s.size(), SQLITE_TRANSIENT, SQLITE_UTF8);
        break;
      case Param::Kind::Blob:
        rc = sqlite3_bind_blob64(stmt, index, p.s.data(), p.s.size(), SQLITE_TRANSIENT);
        break;
    }
    if (rc != SQLITE_OK) throw_db(db, rc, "bind");
    ++index;
  }
}

}  // namespace

bool Row::is_null(int col) const { return sqlite3_column_type(stmt_, col) == SQLITE_NULL; }
std::int64_t Row::as_int(int col) const { return sqlite3_column_int64(stmt_, col); }
double Row::as_double(int col) const { return sqlite3_column_double(stmt_, col); }
std::string Row::as_text(int col) const {
  const unsigned char *text = sqlite3_column_text(stmt_, col);
  if (!text) return {};
  return std::string(reinterpret_cast<const char *>(text),
                     static_cast<std::size_t>(sqlite3_column_bytes(stmt_, col)));
}
std::string Row::as_blob(int col) const {
  const void *bytes = sqlite3_column_blob(stmt_, col);
  const int n = sqlite3_column_bytes(stmt_, col);
  if (!bytes || n <= 0) return {};
  return std::string(static_cast<const char *>(bytes), static_cast<std::size_t>(n));
}
int Row::column_count() const { return sqlite3_column_count(stmt_); }
std::string Row::column_name(int col) const { return sqlite3_column_name(stmt_, col); }

Db::~Db() { close(); }

void Db::close() {
  if (db_) {
    sqlite3_close_v2(db_);
    db_ = nullptr;
  }
}

Db Db::open_rw(const std::string &path) {
  Db db;
  const int rc = sqlite3_open_v2(path.c_str(), &db.db_,
                                 SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX, nullptr);
  if (rc != SQLITE_OK) {
    const std::string message = std::string("cannot open ") + path + ": " + sqlite3_errstr(rc);
    sqlite3_close_v2(db.db_);
    db.db_ = nullptr;
    throw DbError(message, rc, retryable_code(rc));
  }
  return db;
}

Db Db::open_ro(const std::string &path) {
  Db db;
  const int rc = sqlite3_open_v2(path.c_str(), &db.db_, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);
  if (rc != SQLITE_OK) {
    const std::string message = std::string("cannot open ") + path + ": " + sqlite3_errstr(rc);
    sqlite3_close_v2(db.db_);
    db.db_ = nullptr;
    throw DbError(message, rc, retryable_code(rc));
  }
  db.exec("PRAGMA query_only = ON");
  return db;
}

void Db::exec(std::string_view sql) {
  char *error = nullptr;
  const std::string text(sql);
  const int rc = sqlite3_exec(db_, text.c_str(), nullptr, nullptr, &error);
  if (rc != SQLITE_OK) {
    std::string message = "exec: ";
    message += error ? error : sqlite3_errstr(rc);
    if (error) sqlite3_free(error);
    throw DbError(message, rc, retryable_code(rc));
  }
}

void Db::query(std::string_view sql, const std::vector<Param> &params,
               const std::function<bool(const Row &)> &on_row) const {
  Stmt s;
  int rc = sqlite3_prepare_v2(db_, sql.data(), static_cast<int>(sql.size()), &s.stmt, nullptr);
  if (rc != SQLITE_OK) throw_db(db_, rc, "prepare");
  bind_all(db_, s.stmt, params);
  while (true) {
    rc = sqlite3_step(s.stmt);
    if (rc == SQLITE_ROW) {
      if (!on_row(Row(s.stmt))) break;
      continue;
    }
    if (rc == SQLITE_DONE) break;
    throw_db(db_, rc, "step");
  }
}

void Db::run(std::string_view sql, const std::vector<Param> &params) {
  Stmt s;
  int rc = sqlite3_prepare_v2(db_, sql.data(), static_cast<int>(sql.size()), &s.stmt, nullptr);
  if (rc != SQLITE_OK) throw_db(db_, rc, "prepare");
  bind_all(db_, s.stmt, params);
  rc = sqlite3_step(s.stmt);
  if (rc != SQLITE_DONE && rc != SQLITE_ROW) throw_db(db_, rc, "step");
}

std::int64_t Db::changes() const { return sqlite3_changes64(db_); }
void Db::begin() { exec("BEGIN"); }
void Db::commit() { exec("COMMIT"); }
void Db::rollback() { exec("ROLLBACK"); }

bool Db::table_exists(const std::string &name) const {
  bool found = false;
  query("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", {Param::text(name)},
        [&](const Row &) { found = true; return false; });
  return found;
}

bool Db::column_exists(const std::string &table, const std::string &column) const {
  bool found = false;
  query("PRAGMA table_info(" + table + ")", {}, [&](const Row &row) {
    if (row.as_text(1) == column) { found = true; return false; }
    return true;
  });
  return found;
}

// schema.py:8-173 — verbatim column sets and constraints.
const char *ddl_v2() {
  return R"SQL(
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    resume_step INTEGER,
    status TEXT NOT NULL CHECK(status IN ('running','complete','crashed')),
    PRIMARY KEY (session_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS scalars (
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS tensors (
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    dtype TEXT NOT NULL,
    shape TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS artifacts (
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    seq_index INTEGER NOT NULL DEFAULT 0,
    wall_time REAL NOT NULL,
    kind TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    blob_key TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    meta TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (tag, step, seq_index)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS text_events (
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS trace_events (
    step INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    phase TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    details TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (step, phase)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS eval_results (
    suite_name TEXT NOT NULL,
    case_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    score_name TEXT NOT NULL,
    score_value REAL NOT NULL,
    artifact_key TEXT,
    details TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (suite_name, case_id, step, score_name)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS hparam_metrics (
    metric_tag TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER,
    wall_time REAL,
    PRIMARY KEY (metric_tag)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS plugin_data (
    plugin_name TEXT NOT NULL,
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (plugin_name, tag, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS custom_scalar_layouts (
    layout_name TEXT NOT NULL,
    config TEXT NOT NULL,
    PRIMARY KEY (layout_name)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS pr_curves (
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    class_index INTEGER NOT NULL DEFAULT 0,
    wall_time REAL NOT NULL,
    num_thresholds INTEGER NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (tag, step, class_index)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS audio (
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    seq_index INTEGER NOT NULL DEFAULT 0,
    wall_time REAL NOT NULL,
    blob_key TEXT NOT NULL,
    sample_rate INTEGER NOT NULL,
    num_channels INTEGER NOT NULL DEFAULT 1,
    duration_ms REAL,
    mime_type TEXT NOT NULL DEFAULT 'audio/wav',
    label TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tag, step, seq_index)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS graphs (
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    graph_blob_key TEXT NOT NULL,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS embeddings (
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    num_points INTEGER NOT NULL,
    dimensions INTEGER NOT NULL,
    tensor_blob_key TEXT NOT NULL,
    metadata_json TEXT,
    metadata_header TEXT,
    sprite_blob_key TEXT,
    sprite_single_h INTEGER,
    sprite_single_w INTEGER,
    PRIMARY KEY (tag, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS meshes (
    tag TEXT NOT NULL,
    step INTEGER NOT NULL,
    wall_time REAL NOT NULL,
    num_vertices INTEGER NOT NULL,
    has_faces INTEGER NOT NULL DEFAULT 0,
    has_colors INTEGER NOT NULL DEFAULT 0,
    num_faces INTEGER NOT NULL DEFAULT 0,
    vertices_blob_key TEXT NOT NULL,
    faces_blob_key TEXT,
    colors_blob_key TEXT,
    config_json TEXT,
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
)SQL";
}

void set_pragmas(Db &db) {
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA synchronous = NORMAL");
  db.exec("PRAGMA busy_timeout = 5000");
}

namespace {

// schema.py:183-235 (V1→V2), 238-304 (V2→V3), 307-342 (meshes.num_faces, V3→V4)
void migrate_v1_to_v2(Db &db) {
  db.exec("ALTER TABLE blobs RENAME TO artifacts");
  if (db.table_exists("text")) db.exec("ALTER TABLE text RENAME TO text_events");
  if (!db.column_exists("artifacts", "kind"))
    db.exec("ALTER TABLE artifacts ADD COLUMN kind TEXT NOT NULL DEFAULT 'image'");
  if (!db.column_exists("artifacts", "meta"))
    db.exec("ALTER TABLE artifacts ADD COLUMN meta TEXT NOT NULL DEFAULT '{}'");
  db.exec("DROP INDEX IF EXISTS idx_blobs_tag");
  db.exec("DROP INDEX IF EXISTS idx_text_tag");
}

void migrate_v2_to_v3(Db &db) {
  db.exec("ALTER TABLE graphs RENAME COLUMN graph_data TO graph_json");
  if (db.table_exists("embeddings") && db.column_exists("embeddings", "blob_key"))
    db.exec("ALTER TABLE embeddings RENAME COLUMN blob_key TO tensor_blob_key");
}

void migrate_v3_to_v4(Db &db) { db.exec("ALTER TABLE graphs RENAME COLUMN graph_json TO graph_blob_key"); }

}  // namespace

void create_tables(Db &db) {
  if (db.table_exists("blobs")) migrate_v1_to_v2(db);
  if (db.table_exists("graphs") && db.column_exists("graphs", "graph_data")) migrate_v2_to_v3(db);
  if (db.table_exists("graphs") && db.column_exists("graphs", "graph_json")) migrate_v3_to_v4(db);
  if (db.table_exists("meshes") && !db.column_exists("meshes", "num_faces"))
    db.exec("ALTER TABLE meshes ADD COLUMN num_faces INTEGER NOT NULL DEFAULT 0");
  db.exec(ddl_v2());
}

}  // namespace sb
