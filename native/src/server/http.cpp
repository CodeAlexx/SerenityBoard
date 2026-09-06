#include "serenityboard/http.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>

#include "serenityboard/hash.hpp"

namespace sb {

namespace {

std::string lower(std::string s) {
  for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool send_all(int fd, const char *data, std::size_t n) {
  while (n > 0) {
    const ssize_t k = ::send(fd, data, n, MSG_NOSIGNAL);
    if (k <= 0) {
      if (k < 0 && errno == EINTR) continue;
      return false;
    }
    data += k;
    n -= static_cast<std::size_t>(k);
  }
  return true;
}

bool parse_request_head(const std::string &head, Request &req) {
  std::istringstream in(head);
  std::string line;
  if (!std::getline(in, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  std::istringstream start(line);
  std::string version;
  if (!(start >> req.method >> req.raw_target >> version)) return false;
  const auto q = req.raw_target.find('?');
  req.path = percent_decode(req.raw_target.substr(0, q));
  if (q != std::string::npos) {
    std::string qs = req.raw_target.substr(q + 1);
    std::size_t pos = 0;
    while (pos <= qs.size()) {
      const auto amp = qs.find('&', pos);
      const std::string pair = qs.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
      if (!pair.empty()) {
        const auto eq = pair.find('=');
        std::string key = percent_decode(pair.substr(0, eq));
        std::string value = eq == std::string::npos ? "" : percent_decode(pair.substr(eq + 1));
        req.query[key] = value;
      }
      if (amp == std::string::npos) break;
      pos = amp + 1;
    }
  }
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) break;
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string name = lower(line.substr(0, colon));
    std::string value = line.substr(colon + 1);
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(value.begin());
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();
    req.headers[name] = value;
  }
  return true;
}

}  // namespace

std::string percent_decode(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '%' && i + 2 < text.size() + 0 && i + 2 <= text.size() - 1 + 0) {
      auto hexval = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };
      const int hi = hexval(text[i + 1]), lo = hexval(text[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>(hi * 16 + lo));
        i += 2;
        continue;
      }
    }
    if (text[i] == '+') out.push_back(' ');
    else out.push_back(text[i]);
  }
  return out;
}

std::string mime_for_path(const std::string &path) {
  const auto dot = path.rfind('.');
  const std::string ext = dot == std::string::npos ? "" : lower(path.substr(dot + 1));
  static const std::map<std::string, std::string> table = {
      {"html", "text/html; charset=utf-8"}, {"htm", "text/html; charset=utf-8"}, {"js", "text/javascript; charset=utf-8"},
      {"mjs", "text/javascript; charset=utf-8"}, {"css", "text/css; charset=utf-8"}, {"json", "application/json"},
      {"png", "image/png"}, {"jpg", "image/jpeg"}, {"jpeg", "image/jpeg"}, {"gif", "image/gif"}, {"svg", "image/svg+xml"},
      {"ico", "image/x-icon"}, {"webp", "image/webp"}, {"wav", "audio/wav"}, {"mp4", "video/mp4"}, {"webm", "video/webm"},
      {"txt", "text/plain; charset=utf-8"}, {"map", "application/json"}, {"woff", "font/woff"}, {"woff2", "font/woff2"}};
  auto it = table.find(ext);
  return it == table.end() ? "application/octet-stream" : it->second;
}

std::string http_status_text(int status) {
  switch (status) {
    case 200: return "OK";
    case 101: return "Switching Protocols";
    case 204: return "No Content";
    case 301: return "Moved Permanently";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 415: return "Unsupported Media Type";
    case 422: return "Unprocessable Entity";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Unknown";
  }
}

std::string Request::header(const std::string &name) const {
  auto it = headers.find(lower(name));
  return it == headers.end() ? "" : it->second;
}

std::string Request::query_or(const std::string &key, const std::string &fallback) const {
  auto it = query.find(key);
  return it == query.end() ? fallback : it->second;
}

Response Response::json(int status, const std::string &json_text) {
  Response r;
  r.status = status;
  r.body = json_text;
  r.headers.emplace_back("Content-Type", "application/json");
  return r;
}

Response Response::text(int status, const std::string &body, const std::string &content_type) {
  Response r;
  r.status = status;
  r.body = body;
  r.headers.emplace_back("Content-Type", content_type);
  return r;
}

Response Response::file(const std::string &path, const std::string &content_type) {
  Response r;
  r.file_path = path;
  r.headers.emplace_back("Content-Type", content_type);
  return r;
}

void Response::set_header(const std::string &name, const std::string &value) {
  for (auto &[k, v] : headers)
    if (lower(k) == lower(name)) {
      v = value;
      return;
    }
  headers.emplace_back(name, value);
}

// ── server ─────────────────────────────────────────────────────────────

HttpServer::~HttpServer() { stop(); }

int HttpServer::listen(const std::string &host, int port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) return 0;
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(port));
  if (host.empty() || host == "0.0.0.0") addr.sin_addr.s_addr = htonl(INADDR_ANY);
  else if (host == "localhost") addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  else if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) return 0;
  if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof addr) != 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    return 0;
  }
  if (::listen(listen_fd_, 64) != 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    return 0;
  }
  sockaddr_in bound{};
  socklen_t len = sizeof bound;
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr *>(&bound), &len);
  port_ = ntohs(bound.sin_port);
  return port_;
}

void HttpServer::run() {
  running_ = true;
  while (running_) {
    sockaddr_in peer{};
    socklen_t len = sizeof peer;
    const int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr *>(&peer), &len);
    if (fd < 0) {
      if (errno == EINTR) continue;
      if (!running_) break;
      continue;
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    std::thread([this, fd] { handle_connection(fd); }).detach();
  }
}

void HttpServer::start_background() {
  accept_thread_ = std::thread([this] { run(); });
}

void HttpServer::stop() {
  if (!running_ && listen_fd_ < 0) return;
  running_ = false;
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (accept_thread_.joinable()) accept_thread_.join();
}

void HttpServer::handle_connection(int fd) {
  std::string buffer;
  char chunk[16384];
  bool keep_alive = true;
  while (keep_alive && running_) {
    // read head
    std::size_t head_end = std::string::npos;
    while ((head_end = buffer.find("\r\n\r\n")) == std::string::npos) {
      const ssize_t k = ::recv(fd, chunk, sizeof chunk, 0);
      if (k <= 0) {
        ::close(fd);
        return;
      }
      buffer.append(chunk, static_cast<std::size_t>(k));
      if (buffer.size() > (1u << 20)) {
        ::close(fd);
        return;
      }
    }
    Request req;
    if (!parse_request_head(buffer.substr(0, head_end + 4), req)) {
      ::close(fd);
      return;
    }
    buffer.erase(0, head_end + 4);
    std::size_t content_length = 0;
    if (const auto cl = req.header("content-length"); !cl.empty()) content_length = std::stoull(cl);
    while (buffer.size() < content_length) {
      const ssize_t k = ::recv(fd, chunk, sizeof chunk, 0);
      if (k <= 0) {
        ::close(fd);
        return;
      }
      buffer.append(chunk, static_cast<std::size_t>(k));
    }
    req.body = buffer.substr(0, content_length);
    buffer.erase(0, content_length);
    keep_alive = lower(req.header("connection")) != "close";

    // WebSocket upgrade
    if (ws_handler_ && req.path == ws_path_ && lower(req.header("upgrade")) == "websocket") {
      const std::string key = req.header("sec-websocket-key");
      const auto digest = sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
      const std::string accept = base64(std::string(reinterpret_cast<const char *>(digest.data()), digest.size()));
      const std::string head = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                               "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
      if (!send_all(fd, head.data(), head.size())) {
        ::close(fd);
        return;
      }
      WsConnection conn(fd);
      try {
        ws_handler_(req, conn);
      } catch (...) {
      }
      conn.close();
      return;  // WsConnection closed the fd
    }

    Response resp;
    try {
      resp = handler_ ? handler_(req) : Response::text(404, "no handler");
    } catch (const std::exception &e) {
      resp = Response::json(500, std::string("{\"error\":{\"code\":\"error\",\"message\":") +
                                     "\"" + std::string(e.what()) + "\",\"details\":{}}}");
    }
    std::string body;
    if (!resp.file_path.empty()) {
      std::ifstream in(resp.file_path, std::ios::binary);
      if (!in) {
        resp = Response::json(404, "{\"error\":{\"code\":\"not_found\",\"message\":\"file not found\",\"details\":{}}}");
        body = resp.body;
      } else {
        body.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      }
    } else {
      body = resp.body;
    }
    std::string head = "HTTP/1.1 " + std::to_string(resp.status) + " " + http_status_text(resp.status) + "\r\n";
    bool has_type = false;
    for (const auto &[k, v] : resp.headers) {
      if (lower(k) == "content-type") has_type = true;
      head += k + ": " + v + "\r\n";
    }
    if (!has_type) head += "Content-Type: application/octet-stream\r\n";
    head += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    head += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
    head += "Server: serenityboard-native\r\n\r\n";
    if (!send_all(fd, head.data(), head.size()) || (req.method != "HEAD" && !send_all(fd, body.data(), body.size()))) {
      ::close(fd);
      return;
    }
  }
  ::close(fd);
}

// ── WebSocket connection ───────────────────────────────────────────────

std::uint64_t WsConnection::next_id() {
  static std::atomic<std::uint64_t> counter{1};
  return counter++;
}

WsConnection::~WsConnection() { close(); }

void WsConnection::close() {
  if (!alive_.exchange(false)) return;
  ::shutdown(fd_, SHUT_RDWR);
  ::close(fd_);
}

bool WsConnection::read_exact(std::string &buf, std::size_t n) {
  buf.resize(n);
  std::size_t got = 0;
  while (got < n) {
    const ssize_t k = ::recv(fd_, buf.data() + got, n - got, 0);
    if (k <= 0) {
      if (k < 0 && errno == EINTR) continue;
      return false;
    }
    got += static_cast<std::size_t>(k);
  }
  return true;
}

bool WsConnection::send_frame(std::uint8_t opcode, const std::string &payload) {
  std::lock_guard<std::mutex> lock(send_mutex_);
  if (!alive_) return false;
  std::string frame;
  frame.push_back(static_cast<char>(0x80 | opcode));
  const std::size_t n = payload.size();
  if (n < 126) frame.push_back(static_cast<char>(n));
  else if (n < 65536) {
    frame.push_back(126);
    frame.push_back(static_cast<char>(n >> 8));
    frame.push_back(static_cast<char>(n & 0xFF));
  } else {
    frame.push_back(127);
    for (int i = 7; i >= 0; --i) frame.push_back(static_cast<char>((static_cast<std::uint64_t>(n) >> (8 * i)) & 0xFF));
  }
  frame += payload;
  if (!send_all(fd_, frame.data(), frame.size())) {
    alive_ = false;
    return false;
  }
  return true;
}

bool WsConnection::send_text(const std::string &text) { return send_frame(0x1, text); }

bool WsConnection::recv_text(std::string &out) {
  std::string message;
  while (alive_) {
    std::string hdr;
    if (!read_exact(hdr, 2)) return false;
    const std::uint8_t b0 = static_cast<std::uint8_t>(hdr[0]), b1 = static_cast<std::uint8_t>(hdr[1]);
    const bool fin = b0 & 0x80;
    const std::uint8_t opcode = b0 & 0x0F;
    const bool masked = b1 & 0x80;
    std::uint64_t len = b1 & 0x7F;
    if (len == 126) {
      std::string ext;
      if (!read_exact(ext, 2)) return false;
      len = (std::uint64_t(std::uint8_t(ext[0])) << 8) | std::uint8_t(ext[1]);
    } else if (len == 127) {
      std::string ext;
      if (!read_exact(ext, 8)) return false;
      len = 0;
      for (int i = 0; i < 8; ++i) len = (len << 8) | std::uint8_t(ext[static_cast<std::size_t>(i)]);
    }
    if (len > (16u << 20)) return false;
    std::string mask;
    if (masked && !read_exact(mask, 4)) return false;
    std::string payload;
    if (!read_exact(payload, static_cast<std::size_t>(len))) return false;
    if (masked)
      for (std::size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<char>(payload[i] ^ mask[i % 4]);
    switch (opcode) {
      case 0x8:  // close
        send_frame(0x8, payload.substr(0, 2));
        return false;
      case 0x9:  // ping
        send_frame(0xA, payload);
        continue;
      case 0xA:  // pong
        continue;
      case 0x1:
      case 0x2:
      case 0x0:
        message += payload;
        if (fin) {
          out = std::move(message);
          return true;
        }
        continue;
      default:
        return false;
    }
  }
  return false;
}

}  // namespace sb
