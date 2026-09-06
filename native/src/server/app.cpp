#include "serenityboard/app.hpp"

#include <chrono>
#include <filesystem>

namespace sb {

namespace fs = std::filesystem;

App::App(AppOptions options) : options_(std::move(options)), watcher_(options_.logdir, options_.scan_seconds) {
  watcher_.scan_once();
  live_.set_watcher(&watcher_);
  server_.set_handler([this](const Request &req) { return handle(req); });
  server_.set_ws_handler("/ws/live", [this](const Request &req, WsConnection &ws) { ws_live(req, ws); });
}

App::~App() { stop(); }

int App::start() {
  const int port = server_.listen(options_.host, options_.port);
  if (port == 0) return 0;
  running_ = true;
  poll_thread_ = std::thread([this] {
    while (running_) {
      try {
        live_.poll_and_push();
      } catch (...) {
      }
      const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(options_.live_poll_seconds);
      while (running_ && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  scan_thread_ = std::thread([this] {
    while (running_) {
      const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(options_.scan_seconds);
      while (running_ && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (!running_) break;
      try {
        watcher_.scan_once();
      } catch (...) {
      }
    }
  });
  server_.start_background();
  return port;
}

void App::run_forever() {
  while (running_) std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

void App::stop() {
  if (!running_.exchange(false)) return;
  server_.stop();
  if (poll_thread_.joinable()) poll_thread_.join();
  if (scan_thread_.joinable()) scan_thread_.join();
  watcher_.close();
}

// app.py ws_live: accept, loop on text frames, `{"subscribe": {...}}` replaces the filter.
void App::ws_live(const Request &, WsConnection &ws) {
  const std::uint64_t id = ws.id();
  std::string text;
  while (ws.recv_text(text)) {
    Json msg;
    try {
      msg = Json::parse(text);
    } catch (...) {
      continue;
    }
    if (!msg.is_object() || !msg.contains("subscribe") || !msg["subscribe"].is_object()) continue;
    const Json &sub = msg["subscribe"];
    SubscriptionFilter filt;
    if (sub.contains("runs") && sub["runs"].is_array())
      for (const auto &r : sub["runs"]) if (r.is_string()) filt.runs.push_back(r.get<std::string>());
    filt.tag_patterns.clear();
    if (sub.contains("tags") && sub["tags"].is_array() && !sub["tags"].empty())
      for (const auto &t : sub["tags"]) if (t.is_string()) filt.tag_patterns.insert(t.get<std::string>());
    if (filt.tag_patterns.empty()) filt.tag_patterns.insert("*");
    filt.kinds.clear();
    if (sub.contains("kinds") && sub["kinds"].is_array() && !sub["kinds"].empty())
      for (const auto &k : sub["kinds"]) if (k.is_string()) filt.kinds.insert(k.get<std::string>());
    if (filt.kinds.empty()) filt.kinds.insert("scalar");
    WsConnection *conn = &ws;
    live_.subscribe(id, [conn](const std::string &frame) { return conn->send_text(frame); }, std::move(filt));
  }
  live_.unsubscribe(id);
}

Response App::serve_static(const Request &req) {
  if (options_.frontend_dir.empty()) return Response::json(404, R"({"error":{"code":"not_found","message":"Not Found","details":{}}})");
  std::string rel = req.path;
  if (rel == "/" || rel.empty()) rel = "/index.html";
  if (rel.find("..") != std::string::npos) return Response::text(400, "bad path");
  const fs::path candidate = fs::path(options_.frontend_dir) / rel.substr(1);
  std::error_code ec;
  if (fs::is_directory(candidate, ec)) {
    const fs::path index = candidate / "index.html";
    if (fs::is_regular_file(index, ec)) return Response::file(index.string(), mime_for_path(index.string()));
  }
  if (!fs::is_regular_file(candidate, ec))
    return Response::json(404, R"({"error":{"code":"not_found","message":"Not Found","details":{}}})");
  return Response::file(candidate.string(), mime_for_path(candidate.string()));
}

}  // namespace sb
