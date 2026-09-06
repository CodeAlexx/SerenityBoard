#include "serenityboard/session_guard.hpp"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace sb {

double wall_time_now() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

void SessionGuard::initialize() {
  std::optional<std::string> existing;
  db_.query("SELECT value FROM metadata WHERE key = 'active_session_id'", {}, [&](const Row &row) {
    existing = row.as_text(0);
    return false;
  });
  if (existing) {
    const std::string old_id = nlohmann::json::parse(*existing).get<std::string>();
    if (!resume_step_)
      throw std::invalid_argument(
          "Existing run with active session found. Provide resume_step= to continue, or use a new run_name.");
    purge_and_transition(old_id);
  } else {
    create_first_session();
  }
}

void SessionGuard::purge_and_transition(const std::string &old_session_id) {
  const long long resume = *resume_step_;
  std::vector<std::string> orphaned;
  auto collect = [&](const char *sql) {
    try {
      db_.query(sql, {Param::integer(resume)}, [&](const Row &row) {
        if (!row.is_null(0)) orphaned.push_back(row.as_text(0));
        return true;
      });
    } catch (const DbError &) {
      // Table may not exist in older schemas.
    }
  };
  collect("SELECT blob_key FROM artifacts WHERE step > ?");
  collect("SELECT tensor_blob_key FROM embeddings WHERE step > ?");
  collect("SELECT sprite_blob_key FROM embeddings WHERE step > ? AND sprite_blob_key IS NOT NULL");
  collect("SELECT vertices_blob_key FROM meshes WHERE step > ?");
  collect("SELECT faces_blob_key FROM meshes WHERE step > ? AND faces_blob_key IS NOT NULL");
  collect("SELECT colors_blob_key FROM meshes WHERE step > ? AND colors_blob_key IS NOT NULL");
  collect("SELECT blob_key FROM audio WHERE step > ?");
  try {
    db_.query("SELECT graph_blob_key FROM graphs WHERE step > ?", {Param::integer(resume)}, [&](const Row &row) {
      const std::string key = row.as_text(0);
      if (!key.empty() && key[0] != '{') orphaned.push_back(key);
      return true;
    });
  } catch (const DbError &) {
  }

  db_.begin();
  try {
    db_.run("UPDATE sessions SET status = 'crashed' WHERE session_id = ? AND status = 'running'",
            {Param::text(old_session_id)});
    static const char *tables[] = {"scalars",   "tensors",    "artifacts", "text_events", "trace_events", "eval_results",
                                   "plugin_data", "graphs",   "embeddings", "meshes",   "audio",        "pr_curves"};
    purged_rows_ = 0;
    for (const char *table : tables) {
      db_.run(std::string("DELETE FROM ") + table + " WHERE step > ?", {Param::integer(resume)});
      purged_rows_ += db_.changes();
    }
    db_.run("INSERT INTO sessions (session_id, start_time, resume_step, status) VALUES (?, ?, ?, 'running')",
            {Param::text(session_id_), Param::real(wall_time_now()), Param::integer(resume)});
    db_.run("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            {Param::text("active_session_id"), Param::text(nlohmann::json(session_id_).dump())});
    db_.run("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            {Param::text("status"), Param::text(nlohmann::json("running").dump())});
    db_.commit();
  } catch (...) {
    db_.rollback();
    throw;
  }

  if (!blobs_dir_.empty()) {
    for (const auto &key : orphaned) {
      std::error_code ec;
      std::filesystem::remove(std::filesystem::path(blobs_dir_) / key, ec);
    }
  }
}

void SessionGuard::create_first_session() {
  db_.begin();
  try {
    db_.run("INSERT INTO sessions (session_id, start_time, resume_step, status) VALUES (?, ?, NULL, 'running')",
            {Param::text(session_id_), Param::real(wall_time_now())});
    db_.run("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            {Param::text("active_session_id"), Param::text(nlohmann::json(session_id_).dump())});
    db_.run("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            {Param::text("status"), Param::text(nlohmann::json("running").dump())});
    db_.commit();
  } catch (...) {
    db_.rollback();
    throw;
  }
}

void SessionGuard::mark_complete() {
  db_.begin();
  try {
    db_.run("UPDATE sessions SET status = 'complete' WHERE session_id = ?", {Param::text(session_id_)});
    db_.run("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            {Param::text("status"), Param::text(nlohmann::json("complete").dump())});
    db_.commit();
  } catch (...) {
    db_.rollback();
    throw;
  }
}

}  // namespace sb
