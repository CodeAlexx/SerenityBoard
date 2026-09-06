// The SerenityBoard native application: routes + live updates + scanner.
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "serenityboard/http.hpp"
#include "serenityboard/live_updates.hpp"
#include "serenityboard/run_manager.hpp"

namespace sb {

struct AppOptions {
  std::string logdir;
  std::string frontend_dir;  // serves index.html etc.; empty = API only
  std::string host{"0.0.0.0"};
  int port{6006};
  double live_poll_seconds{1.0};
  double scan_seconds{5.0};
};

class App {
public:
  explicit App(AppOptions options);
  ~App();
  /// Bind; returns the bound port (0 on failure).
  int start();
  void run_forever();
  void stop();
  RunWatcher &watcher() { return watcher_; }
  LiveUpdateManager &live() { return live_; }
  Response handle(const Request &req);
  int port() const { return server_.port(); }

private:
  void ws_live(const Request &req, WsConnection &ws);
  Response serve_static(const Request &req);
  AppOptions options_;
  RunWatcher watcher_;
  LiveUpdateManager live_;
  HttpServer server_;
  std::atomic<bool> running_{false};
  std::thread poll_thread_, scan_thread_;
};

}  // namespace sb
