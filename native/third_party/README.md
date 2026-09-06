# Vendored third-party sources (native backend)

| library | version | license | copied from |
|---|---|---|---|
| SQLite amalgamation | 3.46.0 (`sqlite/sqlite3.h` SQLITE_VERSION) | public domain | libsqlite3-sys-0.30.1 crate sources (cargo registry) |
| nlohmann/json | 3.11.3 | MIT | diffusion-compiler/third_party/cudnn_frontend thirdparty copy |

System libraries used: zlib (PNG encoding), pthread. HTTP/1.1, WebSocket (RFC 6455)
and SHA-1 are implemented in-tree (`src/server/http.cpp`, `src/server/ws.cpp`): the
frontend upgrades `/ws/live` on the same port, which header-only HTTP libraries on this
box do not support, and no OpenSSL headers are installed.
