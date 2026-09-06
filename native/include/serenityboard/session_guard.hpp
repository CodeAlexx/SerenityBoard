// Session lifecycle on board.db (serenityboard/writer/session_guard.py):
// first session, resume purge above resume_step, crashed/complete transitions.
#pragma once

#include <optional>
#include <string>

#include "serenityboard/db.hpp"

namespace sb {

class SessionGuard {
public:
  SessionGuard(Db &db, std::string session_id, std::optional<long long> resume_step, std::string blobs_dir)
      : db_(db), session_id_(std::move(session_id)), resume_step_(resume_step), blobs_dir_(std::move(blobs_dir)) {}

  /// Throws std::invalid_argument with the Python message when an active
  /// session exists and no resume_step was given.
  void initialize();
  void mark_complete();
  /// Rows purged by the last resume (0 for a first session).
  long long purged_rows() const { return purged_rows_; }

private:
  void purge_and_transition(const std::string &old_session_id);
  void create_first_session();

  Db &db_;
  std::string session_id_;
  std::optional<long long> resume_step_;
  std::string blobs_dir_;
  long long purged_rows_{0};
};

double wall_time_now();

}  // namespace sb
