// Minimal HTTP/1.1 server with WebSocket (RFC 6455) upgrade, thread per
// connection. Enough for a dashboard: GET/PUT/POST/DELETE, JSON bodies,
// static files, query strings, keep-alive, WebSocket text frames.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sb {

struct Request {
  std::string method;
  std::string path;                            // percent-decoded, no query
  std::string raw_target;
  std::map<std::string, std::string> query;    // percent-decoded
  std::map<std::string, std::string> headers;  // lowercase keys
  std::string body;
  std::string header(const std::string &name) const;
  std::string query_or(const std::string &key, const std::string &fallback) const;
  bool has_query(const std::string &key) const { return query.count(key) != 0; }
};

struct Response {
  int status{200};
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;
  std::string file_path;  // when set, the file is streamed instead of body
  static Response json(int status, const std::string &json_text);
  static Response text(int status, const std::string &body, const std::string &content_type = "text/plain; charset=utf-8");
  static Response file(const std::string &path, const std::string &content_type);
  void set_header(const std::string &name, const std::string &value);
};

class WsConnection {
public:
  explicit WsConnection(int fd) : fd_(fd) {}
  ~WsConnection();
  /// Blocks for the next text frame; false on close/error. Ping/pong handled internally.
  bool recv_text(std::string &out);
  bool send_text(const std::string &text);
  void close();
  bool alive() const { return alive_; }
  std::uint64_t id() const { return id_; }

private:
  bool read_exact(std::string &buf, std::size_t n);
  bool send_frame(std::uint8_t opcode, const std::string &payload);
  int fd_;
  std::atomic<bool> alive_{true};
  std::mutex send_mutex_;
  std::uint64_t id_{next_id()};
  static std::uint64_t next_id();
};

using Handler = std::function<Response(const Request &)>;
using WsHandler = std::function<void(const Request &, WsConnection &)>;

class HttpServer {
public:
  HttpServer() = default;
  ~HttpServer();
  void set_handler(Handler h) { handler_ = std::move(h); }
  void set_ws_handler(const std::string &path, WsHandler h) { ws_path_ = path; ws_handler_ = std::move(h); }
  /// Binds and listens; returns the bound port (0 on failure).
  int listen(const std::string &host, int port);
  /// Accept loop (blocking) until stop().
  void run();
  void start_background();
  void stop();
  int port() const { return port_; }

private:
  void handle_connection(int fd);
  Handler handler_;
  WsHandler ws_handler_;
  std::string ws_path_;
  int listen_fd_{-1};
  int port_{0};
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
};

std::string percent_decode(std::string_view text);
std::string mime_for_path(const std::string &path);
std::string http_status_text(int status);

}  // namespace sb
