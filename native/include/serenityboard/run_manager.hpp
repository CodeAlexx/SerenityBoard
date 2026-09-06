// Run discovery + status derivation (serenityboard/server/run_manager.py).
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "serenityboard/data_provider.hpp"

namespace sb {

class RunWatcher {
public:
  explicit RunWatcher(std::string logdir, double poll_interval = 5.0)
      : logdir_(std::move(logdir)), poll_interval_(poll_interval) {}
  const std::string &logdir() const { return logdir_; }
  double poll_interval() const { return poll_interval_; }

  /// {run_name: db_path}; depth <= 4; nested names joined with "__".
  std::map<std::string, std::string> find_run_dbs() const;
  /// Returns (added, removed).
  std::pair<std::vector<std::string>, std::vector<std::string>> scan_once();
  Json get_runs();
  std::shared_ptr<RunDataProvider> get_provider(const std::string &run_name);
  bool delete_run(const std::string &run_name);
  void close();

private:
  std::string logdir_;
  double poll_interval_;
  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<RunDataProvider>> known_runs_;
};

constexpr double kStaleTimeoutSeconds = 300.0;

}  // namespace sb
