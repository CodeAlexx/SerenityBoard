#include "serenityboard/live_updates.hpp"

namespace sb {

void LiveUpdateManager::subscribe(std::uint64_t id, Sender sender, SubscriptionFilter filt) {
  std::lock_guard<std::mutex> lock(mutex_);
  subscribers_[id] = Sub{std::move(sender), std::move(filt)};
}

void LiveUpdateManager::unsubscribe(std::uint64_t id) {
  std::lock_guard<std::mutex> lock(mutex_);
  subscribers_.erase(id);
}

std::size_t LiveUpdateManager::subscriber_count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return subscribers_.size();
}

std::vector<std::string> LiveUpdateManager::match_tags(const std::vector<std::string> &tags,
                                                       const std::set<std::string> &patterns) {
  if (patterns.empty()) return tags;
  std::vector<std::string> matched;
  for (const auto &tag : tags)
    for (const auto &p : patterns)
      if (glob_match(p, tag)) {
        matched.push_back(tag);
        break;
      }
  return matched;
}

void LiveUpdateManager::push_to_run(const std::string &run, const Json &message) {
  const std::string text = message.dump();
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = subscribers_.begin(); it != subscribers_.end();) {
    const auto &runs = it->second.filt.runs;
    bool cares = std::find(runs.begin(), runs.end(), run) != runs.end();
    if (cares && !it->second.send(text)) it = subscribers_.erase(it);
    else ++it;
  }
}

void LiveUpdateManager::push_to_kind(const std::string &run, const std::string &kind, const Json &message) {
  const std::string text = message.dump();
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = subscribers_.begin(); it != subscribers_.end();) {
    const auto &f = it->second.filt;
    bool cares = std::find(f.runs.begin(), f.runs.end(), run) != f.runs.end() && f.kinds.count(kind);
    if (cares && !it->second.send(text)) it = subscribers_.erase(it);
    else ++it;
  }
}

void LiveUpdateManager::push_scalar(const std::string &run, const std::string &tag, const Json &message) {
  const std::string text = message.dump();
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = subscribers_.begin(); it != subscribers_.end();) {
    const auto &f = it->second.filt;
    bool cares = std::find(f.runs.begin(), f.runs.end(), run) != f.runs.end() && f.kinds.count("scalar");
    if (cares && !f.tag_patterns.empty()) {
      cares = false;
      for (const auto &p : f.tag_patterns)
        if (glob_match(p, tag)) { cares = true; break; }
    }
    if (cares && !it->second.send(text)) it = subscribers_.erase(it);
    else ++it;
  }
}

void LiveUpdateManager::poll_and_push() {
  if (!watcher_) return;
  std::map<std::string, std::set<std::string>> active, active_kinds;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (subscribers_.empty()) return;
    for (const auto &[_, sub] : subscribers_)
      for (const auto &run : sub.filt.runs) {
        active[run].insert(sub.filt.tag_patterns.begin(), sub.filt.tag_patterns.end());
        active_kinds[run].insert(sub.filt.kinds.begin(), sub.filt.kinds.end());
      }
  }
  if (active.empty()) return;

  for (const auto &[run_name, patterns] : active) {
    auto provider = watcher_->get_provider(run_name);
    if (!provider) continue;

    std::optional<std::string> current_session;
    try {
      current_session = provider->get_active_session_id();
    } catch (...) {
    }
    auto known = known_sessions_.find(run_name);
    if (known != known_sessions_.end() && known->second && current_session != known->second) {
      Json resume = nullptr;
      try {
        if (current_session)
          if (auto rs = provider->get_resume_step(*current_session)) resume = *rs;
      } catch (...) {
      }
      push_to_run(run_name, Json{{"type", "session_changed"},
                                 {"run", run_name},
                                 {"old_session_id", *known->second},
                                 {"new_session_id", current_session ? Json(*current_session) : Json(nullptr)},
                                 {"resume_step", resume}});
    }
    known_sessions_[run_name] = current_session;
    const Json session_json = current_session ? Json(*current_session) : Json(nullptr);

    const auto &kinds_for_run = active_kinds[run_name];
    if (kinds_for_run.count("scalar")) {
      std::vector<std::string> scalar_tags;
      try {
        scalar_tags = provider->get_tags()["scalars"].get<std::vector<std::string>>();
      } catch (...) {
      }
      for (const auto &tag : match_tags(scalar_tags, patterns)) {
        std::vector<ScalarRow> rows;
        try {
          rows = provider->read_scalars_incremental(tag);
        } catch (...) {
          continue;
        }
        if (rows.empty()) continue;
        Json points = Json::array();
        for (const auto &r : rows) points.push_back({{"step", r.step}, {"wall_time", r.wall_time}, {"value", r.value}});
        push_scalar(run_name, tag, Json{{"type", "scalar"}, {"run", run_name}, {"tag", tag}, {"session_id", session_json}, {"points", points}});
      }
    }
    if (kinds_for_run.count("trace")) {
      Json events = Json::array();
      try {
        events = provider->read_trace_events_incremental();
      } catch (...) {
      }
      if (!events.empty())
        push_to_kind(run_name, "trace", Json{{"type", "trace"}, {"run", run_name}, {"session_id", session_json}, {"events", events}});
    }
    if (kinds_for_run.count("eval")) {
      Json results = Json::array();
      try {
        results = provider->read_eval_results_incremental();
      } catch (...) {
      }
      if (!results.empty())
        push_to_kind(run_name, "eval", Json{{"type", "eval"}, {"run", run_name}, {"session_id", session_json}, {"results", results}});
    }
  }
}

}  // namespace sb
