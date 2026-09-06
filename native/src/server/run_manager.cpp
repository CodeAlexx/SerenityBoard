#include "serenityboard/run_manager.hpp"

#include <filesystem>
#include <functional>

#include "serenityboard/session_guard.hpp"

namespace sb {

namespace fs = std::filesystem;

std::map<std::string, std::string> RunWatcher::find_run_dbs() const {
  std::map<std::string, std::string> found;
  std::error_code ec;
  if (!fs::is_directory(logdir_, ec)) return found;
  std::string base = logdir_;
  while (base.size() > 1 && base.back() == '/') base.pop_back();
  // os.walk with depth limit: depth = root[len(logdir):].count('/'); stop descending at depth >= 4
  std::function<void(const fs::path &, int)> walk = [&](const fs::path &root, int depth) {
    if (depth >= 4) return;
    std::error_code e;
    if (fs::exists(root / "board.db", e)) {
      const std::string root_text = root.string();
      std::string rel = root_text.size() > base.size() + 1 ? root_text.substr(base.size() + 1) : "";
      std::string name;
      if (!rel.empty()) {
        name = rel;
        std::size_t pos;
        while ((pos = name.find('/')) != std::string::npos) name.replace(pos, 1, "__");
      } else {
        name = root.filename().string();
      }
      found[name] = (root / "board.db").string();
    }
    for (const auto &entry : fs::directory_iterator(root, e)) {
      if (entry.is_directory(e)) walk(entry.path(), depth + 1);
    }
  };
  walk(fs::path(base), 0);
  return found;
}

std::pair<std::vector<std::string>, std::vector<std::string>> RunWatcher::scan_once() {
  std::vector<std::string> added, removed;
  const auto found = find_run_dbs();
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &[name, db_path] : found) {
    if (known_runs_.count(name)) continue;
    try {
      auto provider = std::make_shared<RunDataProvider>(db_path);
      provider->get_run_info();
      known_runs_[name] = provider;
      added.push_back(name);
    } catch (const std::exception &) {
      // Failed to open (mid-creation, corrupt): retry on the next scan.
    }
  }
  for (auto it = known_runs_.begin(); it != known_runs_.end();) {
    if (!found.count(it->first)) {
      it->second->close();
      removed.push_back(it->first);
      it = known_runs_.erase(it);
    } else {
      ++it;
    }
  }
  return {added, removed};
}

Json RunWatcher::get_runs() {
  const double now = wall_time_now();
  std::vector<std::pair<std::string, std::shared_ptr<RunDataProvider>>> snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &[name, p] : known_runs_) snapshot.emplace_back(name, p);
  }
  Json result = Json::array();
  for (const auto &[name, provider] : snapshot) {
    try {
      Json info = provider->get_run_info();
      std::string status = info.contains("status") && info["status"].is_string() ? info["status"].get<std::string>() : "unknown";
      const auto [last_wt, last_step] = provider->get_last_activity();
      Json hparams = info.contains("hparams") && info["hparams"].is_object() ? info["hparams"] : Json::object();
      Json max_steps = hparams.contains("max_steps") ? hparams["max_steps"] : Json(nullptr);
      if (status == "running") {
        if (last_wt && now - *last_wt > kStaleTimeoutSeconds) {
          const bool reached = max_steps.is_number() && last_step && static_cast<double>(*last_step) >= max_steps.get<double>();
          status = reached ? "completed" : "stopped";
        } else if (!last_wt) {
          if (info.contains("start_time") && info["start_time"].is_number() && now - info["start_time"].get<double>() > kStaleTimeoutSeconds)
            status = "empty";
        }
      }
      result.push_back({{"name", name},
                        {"start_time", info.contains("start_time") ? info["start_time"] : Json(nullptr)},
                        {"status", status},
                        {"last_activity", last_wt ? Json(*last_wt) : Json(nullptr)},
                        {"last_step", last_step ? Json(*last_step) : Json(nullptr)},
                        {"max_steps", max_steps},
                        {"active_session_id", info.contains("active_session_id") ? info["active_session_id"] : Json(nullptr)},
                        {"hparams", hparams}});
    } catch (const std::exception &) {
      result.push_back({{"name", name}, {"start_time", nullptr}, {"status", "error"}, {"last_activity", nullptr},
                        {"last_step", nullptr}, {"max_steps", nullptr}, {"active_session_id", nullptr},
                        {"hparams", Json::object()}});
    }
  }
  return result;
}

std::shared_ptr<RunDataProvider> RunWatcher::get_provider(const std::string &run_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = known_runs_.find(run_name);
  return it == known_runs_.end() ? nullptr : it->second;
}

bool RunWatcher::delete_run(const std::string &run_name) {
  std::shared_ptr<RunDataProvider> provider;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = known_runs_.find(run_name);
    if (it == known_runs_.end()) return false;
    provider = it->second;
    known_runs_.erase(it);
  }
  const std::string run_dir = provider->run_dir();
  provider->close();
  std::error_code ec;
  if (fs::is_directory(run_dir, ec)) fs::remove_all(run_dir, ec);
  return true;
}

void RunWatcher::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &[_, p] : known_runs_) p->close();
  known_runs_.clear();
}

}  // namespace sb
