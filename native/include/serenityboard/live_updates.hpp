// WebSocket live-update manager (serenityboard/server/live_updates.py):
// subscription-scoped 1 s polling, fnmatch tag globs, session-change notices.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "serenityboard/run_manager.hpp"

namespace sb {

struct SubscriptionFilter {
  std::vector<std::string> runs;
  std::set<std::string> tag_patterns;
  std::set<std::string> kinds{"scalar"};
};

/// A subscriber is an opaque sender: returns false when the socket is dead.
using Sender = std::function<bool(const std::string &text)>;

class LiveUpdateManager {
public:
  void set_watcher(RunWatcher *watcher) { watcher_ = watcher; }
  void subscribe(std::uint64_t id, Sender sender, SubscriptionFilter filt);
  void unsubscribe(std::uint64_t id);
  std::size_t subscriber_count();
  /// One poll: reads increments for subscribed runs and pushes messages.
  void poll_and_push();

  static std::vector<std::string> match_tags(const std::vector<std::string> &tags, const std::set<std::string> &patterns);

private:
  struct Sub {
    Sender send;
    SubscriptionFilter filt;
  };
  void push_to_run(const std::string &run, const Json &message);
  void push_to_kind(const std::string &run, const std::string &kind, const Json &message);
  void push_scalar(const std::string &run, const std::string &tag, const Json &message);

  RunWatcher *watcher_{nullptr};
  std::mutex mutex_;
  std::map<std::uint64_t, Sub> subscribers_;
  std::map<std::string, std::optional<std::string>> known_sessions_;
};

}  // namespace sb
