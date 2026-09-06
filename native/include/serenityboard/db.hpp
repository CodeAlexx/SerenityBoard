// SQLite access and the SerenityBoard board.db schema (v2 DDL, migrations,
// pragmas) — a faithful port of serenityboard/writer/schema.py.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

struct sqlite3;
struct sqlite3_stmt;

namespace sb {

class DbError : public std::runtime_error {
public:
  DbError(std::string message, int code, bool retryable)
      : std::runtime_error(std::move(message)), code_(code), retryable_(retryable) {}
  int code() const { return code_; }
  /// True for SQLITE_BUSY / SQLITE_LOCKED / SQLITE_IOERR-class failures — the
  /// class Python's writer retries (sqlite3.OperationalError).
  bool retryable() const { return retryable_; }

private:
  int code_;
  bool retryable_;
};

/// One bound value for a prepared statement.
struct Param {
  enum class Kind { Null, Int, Real, Text, Blob };
  Kind kind{Kind::Null};
  std::int64_t i{};
  double d{};
  std::string s;   // Text or Blob bytes
  static Param null() { return {}; }
  static Param integer(std::int64_t v) { Param p; p.kind = Kind::Int; p.i = v; return p; }
  static Param real(double v) { Param p; p.kind = Kind::Real; p.d = v; return p; }
  static Param text(std::string v) { Param p; p.kind = Kind::Text; p.s = std::move(v); return p; }
  static Param blob(std::string v) { Param p; p.kind = Kind::Blob; p.s = std::move(v); return p; }
};

/// One result row: columns as strings/ints/doubles/blobs, addressed by index.
class Row {
public:
  explicit Row(sqlite3_stmt *stmt) : stmt_(stmt) {}
  bool is_null(int col) const;
  std::int64_t as_int(int col) const;
  double as_double(int col) const;
  std::string as_text(int col) const;
  std::string as_blob(int col) const;
  int column_count() const;
  std::string column_name(int col) const;

private:
  sqlite3_stmt *stmt_;
};

class Db {
public:
  Db() = default;
  ~Db();
  Db(const Db &) = delete;
  Db &operator=(const Db &) = delete;
  Db(Db &&other) noexcept : db_(other.db_) { other.db_ = nullptr; }
  Db &operator=(Db &&other) noexcept {
    if (this != &other) {
      close();
      db_ = other.db_;
      other.db_ = nullptr;
    }
    return *this;
  }

  /// Open read-write (creating), or read-only (`?mode=ro` + query_only).
  static Db open_rw(const std::string &path);
  static Db open_ro(const std::string &path);
  bool is_open() const { return db_ != nullptr; }
  void close();

  void exec(std::string_view sql);
  /// Run a statement with params; callback per row (return false to stop).
  void query(std::string_view sql, const std::vector<Param> &params,
             const std::function<bool(const Row &)> &on_row) const;
  void run(std::string_view sql, const std::vector<Param> &params);
  std::int64_t changes() const;
  void begin();
  void commit();
  void rollback();
  bool table_exists(const std::string &name) const;
  bool column_exists(const std::string &table, const std::string &column) const;
  sqlite3 *raw() const { return db_; }

private:
  sqlite3 *db_{nullptr};
};

/// schema.py: set_pragmas + create_tables (with V1→V2→V3→V4 migrations).
void set_pragmas(Db &db);
void create_tables(Db &db);
/// The v2 DDL text (for tests and diagnostics).
const char *ddl_v2();

}  // namespace sb
