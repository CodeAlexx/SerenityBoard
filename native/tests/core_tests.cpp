// Core gates: schema, writer round-trip, encoders, reservoir, session guard.
// Numeric cross-checks against NumPy live in tests/py/ (run by scripts/gate.sh).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "serenityboard/blob_storage.hpp"
#include "serenityboard/db.hpp"
#include "serenityboard/encoders.hpp"
#include "serenityboard/hash.hpp"
#include "serenityboard/reservoir.hpp"
#include "serenityboard/session_guard.hpp"
#include "serenityboard/summary_writer.hpp"

namespace fs = std::filesystem;
using sb::Param;

static int failures = 0;
#define CHECK(cond)                                                                        \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      ++failures;                                                                          \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n";           \
    }                                                                                      \
  } while (0)

static fs::path temp_dir(const std::string &name) {
  const char *keep = std::getenv("SB_KEEP_DIR");
  const fs::path p = keep ? fs::path(keep) / name
                          : fs::temp_directory_path() / ("sb_native_" + name + "_" + std::to_string(::getpid()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

static long long count_rows(sb::Db &db, const std::string &table) {
  long long n = 0;
  db.query("SELECT COUNT(*) FROM " + table, {}, [&](const sb::Row &r) { n = r.as_int(0); return false; });
  return n;
}

static void test_hashes() {
  CHECK(sb::hex(sb::sha256("abc").data(), 32) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(sb::hex(sb::sha1("abc").data(), 20) == "a9993e364706816aba3e25717850c26c9cd0d89d");
  CHECK(sb::base64("dGhlIHNhbXBsZSBub25jZQ==") == "ZEdobElITmhiWEJzWlNCdWIyNWpaUT09");
  CHECK(sb::base64("a") == "YQ==");
  CHECK(sb::base64("ab") == "YWI=");
}

static void test_schema() {
  const auto dir = temp_dir("schema");
  sb::Db db = sb::Db::open_rw((dir / "board.db").string());
  sb::set_pragmas(db);
  sb::create_tables(db);
  for (const char *t : {"metadata", "sessions", "scalars", "tensors", "artifacts", "text_events", "trace_events",
                        "eval_results", "hparam_metrics", "plugin_data", "custom_scalar_layouts", "pr_curves",
                        "audio", "graphs", "embeddings", "meshes"})
    CHECK(db.table_exists(t));
  std::string mode;
  db.query("PRAGMA journal_mode", {}, [&](const sb::Row &r) { mode = r.as_text(0); return false; });
  CHECK(mode == "wal");
  // idempotent
  sb::create_tables(db);
  // v1 migration path
  sb::Db v1 = sb::Db::open_rw((dir / "v1.db").string());
  v1.exec("CREATE TABLE blobs (tag TEXT NOT NULL, step INTEGER NOT NULL, seq_index INTEGER NOT NULL DEFAULT 0, "
          "wall_time REAL NOT NULL, mime_type TEXT NOT NULL, blob_key TEXT NOT NULL, width INTEGER, height INTEGER, "
          "PRIMARY KEY (tag, step, seq_index)) WITHOUT ROWID; CREATE TABLE text (tag TEXT NOT NULL, step INTEGER NOT "
          "NULL, wall_time REAL NOT NULL, value TEXT NOT NULL, PRIMARY KEY (tag, step)) WITHOUT ROWID;");
  sb::create_tables(v1);
  CHECK(v1.table_exists("artifacts") && v1.table_exists("text_events") && !v1.table_exists("blobs"));
  CHECK(v1.column_exists("artifacts", "kind") && v1.column_exists("artifacts", "meta"));
  fs::remove_all(dir);
}

static void test_encoders() {
  // histogram: 10 values, 5 bins over [0, 9]
  std::vector<double> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto rows = sb::histogram_rows(v, 5);
  CHECK(rows.size() == 15);
  CHECK(rows[0] == 0.0 && rows[1] == 1.8 && rows[2] == 2.0);   // [0,1.8): 0,1
  CHECK(rows[12] == 7.2 && rows[13] == 9.0 && rows[14] == 2.0);  // [7.2,9]: 8,9
  double total = 0;
  for (std::size_t i = 2; i < rows.size(); i += 3) total += rows[i];
  CHECK(total == 10.0);
  // constant input widens to [x-0.5, x+0.5]
  auto c = sb::histogram_rows({5, 5, 5}, 2);
  CHECK(c[0] == 4.5 && c[4] == 5.5 && c[2] + c[5] == 3.0);
  // non-finite dropped; empty -> empty
  CHECK(sb::histogram_rows({NAN, INFINITY}, 4).empty());
  // pr curve: perfect predictions, 3 thresholds
  auto pr = sb::pr_curve_rows({1, 0, 1, 0}, {0.9, 0.1, 0.8, 0.2}, 3);
  CHECK(pr.size() == 18);
  // rows: tp, fp, tn, fn, precision, recall; columns: thresholds 0, 0.5, 1
  CHECK(pr[0] == 2 && pr[3] == 2 && pr[6] == 0 && pr[9] == 0);  // t=0: everything predicted positive
  CHECK(pr[12] == 0.5 && pr[15] == 1.0);                          // precision 0.5, recall 1
  CHECK(pr[1] == 2 && pr[4] == 0);                                // t=0.5: exact split
  CHECK(pr[14] == 1.0 && pr[17] == 0.0);                          // t=1: nothing predicted -> precision 1, recall 0
  // wav
  const std::string wav = sb::encode_wav_pcm16({1, -1, 2, -2}, 2, 8000);
  CHECK(wav.size() == 44 + 8 && wav.compare(0, 4, "RIFF") == 0 && wav.compare(8, 4, "WAVE") == 0);
  // png
  const std::uint8_t px[12] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
  const std::string png = sb::encode_png(2, 2, 3, px);
  CHECK(png.size() > 33 && png.compare(0, 8, "\x89PNG\r\n\x1a\n") == 0);
  CHECK(sb::linspace(0, 1, 5)[2] == 0.5 && sb::linspace(0, 1, 5)[4] == 1.0);
}

static void test_reservoir() {
  sb::Reservoir<int> unbounded(0);
  for (int i = 0; i < 1000; ++i) unbounded.add("k", i);
  CHECK(unbounded.get_items("k").size() == 1000);
  sb::Reservoir<int> bounded(10, 0, true);
  for (int i = 0; i < 1000; ++i) bounded.add("k", i);
  auto items = bounded.get_items("k");
  CHECK(items.size() == 10 && items.back() == 999);  // always_keep_last
  sb::Reservoir<int> a(10, 7, true), b(10, 7, true);
  for (int i = 0; i < 500; ++i) { a.add("k", i); b.add("k", i); }
  CHECK(a.get_items("k") == b.get_items("k"));  // deterministic per seed
  auto drained = bounded.drain_items("k");
  CHECK(drained.size() == 10 && bounded.get_items("k").empty());
  sb::Reservoir<int> one(1, 0, false);
  for (int i = 0; i < 50; ++i) one.add("k", i);
  CHECK(one.get_items("k").size() == 1);
}

static void test_writer_roundtrip() {
  const auto dir = temp_dir("writer");
  std::string db_path;
  {
    sb::WriterOptions o;
    o.run_name = "run1";
    o.hparams = nlohmann::json{{"lr", 1e-4}, {"batch", 4}};
    o.system_metrics = false;
    sb::SummaryWriter w(dir.string(), o);
    db_path = w.db_path();
    for (int s = 0; s < 20; ++s) w.add_scalar("loss/train", 1.0 / (s + 1), s);
    w.add_scalars("lr", {{"a", 1.0}, {"b", 2.0}}, 3);
    w.add_histogram("weights", {0.1, 0.2, 0.3, 0.4, 5.0}, 1, 8);
    const std::uint8_t px[4 * 3] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    w.add_image("img", 2, 2, 3, px, 5);
    w.add_text("notes", "hello", 2);
    w.add_trace(1, "forward", 12.5, nlohmann::json{{"k", 1}});
    w.add_eval("suite", "case-1", 1, "score", 0.75, std::nullopt, std::nullopt);
    w.add_pr_curve("pr", {1, 0}, {0.9, 0.2}, 1, 5);
    w.add_audio("clip", {0, 100, -100, 0}, 1, 8000, 1);
    w.add_plugin_data("plug", "t", nlohmann::json{{"x", 1}}, 1);
    w.add_custom_scalars_layout(nlohmann::json{{"categories", nlohmann::json::array()}});
    w.add_hparams(nlohmann::json{{"lr", 1e-4}}, {{"final_loss", 0.05}});
    w.flush();
    w.close();
  }
  sb::Db db = sb::Db::open_ro(db_path);
  CHECK(count_rows(db, "scalars") == 22);
  CHECK(count_rows(db, "tensors") == 1);
  CHECK(count_rows(db, "artifacts") == 1);
  CHECK(count_rows(db, "text_events") == 1);
  CHECK(count_rows(db, "trace_events") == 1);
  CHECK(count_rows(db, "eval_results") == 1);
  CHECK(count_rows(db, "pr_curves") == 1);
  CHECK(count_rows(db, "audio") == 1);
  CHECK(count_rows(db, "plugin_data") == 1);
  CHECK(count_rows(db, "custom_scalar_layouts") == 1);
  CHECK(count_rows(db, "hparam_metrics") == 1);
  std::string status, hparams, run_name;
  db.query("SELECT key, value FROM metadata", {}, [&](const sb::Row &r) {
    const auto k = r.as_text(0);
    if (k == "status") status = r.as_text(1);
    if (k == "hparams") hparams = r.as_text(1);
    if (k == "run_name") run_name = r.as_text(1);
    return true;
  });
  CHECK(status == "\"complete\"");
  CHECK(run_name == "\"run1\"");
  CHECK(nlohmann::json::parse(hparams)["lr"] == 1e-4);
  std::string session_status;
  db.query("SELECT status FROM sessions", {}, [&](const sb::Row &r) { session_status = r.as_text(0); return false; });
  CHECK(session_status == "complete");
  // blob on disk
  std::string blob_key;
  db.query("SELECT blob_key, width, height, mime_type FROM artifacts", {}, [&](const sb::Row &r) {
    blob_key = r.as_text(0);
    CHECK(r.as_int(1) == 2 && r.as_int(2) == 2 && r.as_text(3) == "image/png");
    return false;
  });
  CHECK(fs::exists(dir / "run1" / "blobs" / blob_key));
  CHECK(blob_key.size() == 20 && blob_key.substr(16) == ".png");
  // histogram shape
  db.query("SELECT dtype, shape, length(data) FROM tensors", {}, [&](const sb::Row &r) {
    CHECK(r.as_text(0) == "float64" && r.as_text(1) == "[8,3]" && r.as_int(2) == 8 * 3 * 8);
    return false;
  });
  db.close();

  // resume: purge above step 10, old session crashed
  {
    sb::WriterOptions o;
    o.run_name = "run1";
    o.resume_step = 10;
    o.system_metrics = false;
    sb::SummaryWriter w(dir.string(), o);
    w.add_scalar("loss/train", 0.5, 11);
    w.flush();
    w.close();
  }
  sb::Db db2 = sb::Db::open_ro(db_path);
  long long above = 0, sessions = 0, crashed = 0;
  db2.query("SELECT COUNT(*) FROM scalars WHERE step > 10", {}, [&](const sb::Row &r) { above = r.as_int(0); return false; });
  db2.query("SELECT COUNT(*) FROM sessions", {}, [&](const sb::Row &r) { sessions = r.as_int(0); return false; });
  db2.query("SELECT COUNT(*) FROM sessions WHERE status='crashed'", {}, [&](const sb::Row &r) { crashed = r.as_int(0); return false; });
  CHECK(above == 1);  // only the new step 11
  CHECK(sessions == 2);
  CHECK(crashed == 0);  // the first session had completed cleanly, so it stays 'complete'
  db2.close();

  // no resume_step on an active session -> ValueError message
  {
    sb::WriterOptions o;
    o.run_name = "run2";
    o.system_metrics = false;
    { sb::SummaryWriter w(dir.string(), o); w.add_scalar("x", 1, 0); w.flush(); }  // destructor closes -> complete
    // simulate an unfinished session: reopen and set metadata status running, session running
    sb::Db raw = sb::Db::open_rw((dir / "run2" / "board.db").string());
    raw.exec("UPDATE sessions SET status='running'");
    raw.close();
    bool threw = false;
    try {
      sb::SummaryWriter w(dir.string(), o);
    } catch (const std::invalid_argument &e) {
      threw = std::string(e.what()).find("Provide resume_step=") != std::string::npos;
    }
    CHECK(threw);
  }
  if (std::getenv("SB_KEEP_DIR")) return;  // leave run1/run2 on disk for the Python cross-check
  // rank gate -> no-op
  {
    sb::WriterOptions o;
    o.rank = 1;
    sb::SummaryWriter w(dir.string(), o);
    CHECK(w.is_noop());
    w.add_scalar("x", 1, 0);
    w.close();
  }
  fs::remove_all(dir);
}

int main() {
  test_hashes();
  test_schema();
  test_encoders();
  test_reservoir();
  test_writer_roundtrip();
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "core tests: all pass\n";
  return 0;
}
