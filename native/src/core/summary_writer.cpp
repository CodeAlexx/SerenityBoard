#include "serenityboard/summary_writer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

#include "serenityboard/encoders.hpp"
#include "serenityboard/session_guard.hpp"
#include "serenityboard/system_metrics.hpp"

namespace sb {

namespace {

const std::map<std::string, std::size_t> kDefaultReservoir = {
    {"scalars", 0},   {"text_events", 0}, {"trace_events", 0}, {"tensors", 500},   {"artifacts", 100}, {"audio", 100},
    {"plugin_data", 500}, {"pr_curves", 100}, {"graphs", 10},  {"embeddings", 20}, {"meshes", 50}};

const std::vector<std::string> kLossyClasses = {"tensors", "artifacts",  "audio", "plugin_data",
                                                "pr_curves", "graphs", "embeddings", "meshes"};

std::string utc_run_name() {
  const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
  gmtime_r(&now, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y%m%d_%H%M%S", &tm);
  return buf;
}

std::string uuid4() {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<std::uint64_t> dist;
  std::uint64_t a = dist(gen), b = dist(gen);
  a = (a & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;  // version 4
  b = (b & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;  // variant
  char buf[40];
  std::snprintf(buf, sizeof buf, "%08llx-%04llx-%04llx-%04llx-%012llx", static_cast<unsigned long long>(a >> 32),
                static_cast<unsigned long long>((a >> 16) & 0xFFFF), static_cast<unsigned long long>(a & 0xFFFF),
                static_cast<unsigned long long>(b >> 48), static_cast<unsigned long long>(b & 0xFFFFFFFFFFFFULL));
  return buf;
}

int resolve_rank(const std::optional<int> &explicit_rank) {
  if (explicit_rank) return *explicit_rank;
  for (const char *name : {"SB_RANK", "RANK", "LOCAL_RANK"}) {
    if (const char *v = std::getenv(name)) {
      try {
        return std::stoi(v);
      } catch (...) {
      }
    }
  }
  return 0;
}

std::string f64_bytes(const std::vector<double> &values) {
  std::string out(values.size() * sizeof(double), '\0');
  if (!values.empty()) std::memcpy(out.data(), values.data(), out.size());
  return out;
}

}  // namespace

SummaryWriter::SummaryWriter(const std::string &logdir, WriterOptions options)
    : logdir_(logdir), options_(std::move(options)) {
  if (resolve_rank(options_.rank) != 0) {
    noop_ = true;
    return;
  }
  run_name_ = options_.run_name.empty() ? utc_run_name() : options_.run_name;
  run_dir_ = (std::filesystem::path(logdir_) / run_name_).string();
  std::filesystem::create_directories(run_dir_);
  db_path_ = (std::filesystem::path(run_dir_) / "board.db").string();
  session_id_ = uuid4();
  blobs_ = std::make_unique<BlobStorage>((std::filesystem::path(run_dir_) / "blobs").string());

  auto config = kDefaultReservoir;
  for (const auto &[k, v] : options_.reservoir_config) config[k] = v;
  for (const auto &[name, size] : config) reservoirs_[name] = std::make_unique<Reservoir<WriteItem>>(size, 0, true);

  thread_alive_ = true;
  thread_ = std::thread([this] { writer_loop(); });
  {
    std::unique_lock<std::mutex> lock(qmutex_);
    q_all_done_.wait(lock, [this] { return ready_; });
  }
  if (init_error_) {
    thread_alive_ = false;
    if (thread_.joinable()) thread_.join();
    std::rethrow_exception(init_error_);
  }

  if (options_.hparams)
    queue_put({"metadata", {Param::text("hparams"), Param::text(options_.hparams->dump())}, std::nullopt});
  queue_put({"metadata", {Param::text("run_name"), Param::text(nlohmann::json(run_name_).dump())}, std::nullopt});
  queue_put({"metadata", {Param::text("start_time"), Param::text(nlohmann::json(wall_time_now()).dump())}, std::nullopt});
  queue_put({"metadata", {Param::text("schema_version"), Param::text(nlohmann::json("2").dump())}, std::nullopt});

  if (options_.system_metrics) {
    sys_metrics_ = std::make_unique<SystemMetricsCollector>(*this, options_.system_metrics_interval_secs);
    sys_metrics_->start();
  }
}

SummaryWriter::~SummaryWriter() {
  try {
    close();
  } catch (...) {
  }
}

// ── queue ──────────────────────────────────────────────────────────────

void SummaryWriter::queue_put(WriteItem item) {
  std::unique_lock<std::mutex> lock(qmutex_);
  q_not_full_.wait(lock, [this] { return queue_.size() < options_.max_queue_size || !thread_alive_; });
  if (!thread_alive_) return;  // writer gone: drop silently (sticky error surfaces on check)
  queue_.push_back(std::move(item));
  ++unfinished_;
  q_not_empty_.notify_one();
}

void SummaryWriter::queue_join() {
  std::unique_lock<std::mutex> lock(qmutex_);
  q_all_done_.wait(lock, [this] { return unfinished_ == 0 || !thread_alive_; });
}

void SummaryWriter::enqueue(const std::string &data_class, const std::string &tag, WriteItem item) {
  if (std::find(kLossyClasses.begin(), kLossyClasses.end(), data_class) != kLossyClasses.end()) {
    reservoirs_[data_class]->add(tag, std::move(item));
  } else {
    queue_put(std::move(item));
  }
}

void SummaryWriter::drain_reservoirs() {
  for (const auto &name : kLossyClasses) {
    auto &res = reservoirs_[name];
    for (const auto &key : res->keys())
      for (auto &item : res->drain_items(key)) queue_put(std::move(item));
  }
}

void SummaryWriter::check_error() {
  std::lock_guard<std::mutex> lock(error_mutex_);
  if (sticky_error_) throw WriterError(*sticky_error_);
}

// ── writer thread ──────────────────────────────────────────────────────

void SummaryWriter::writer_loop() {
  Db db;
  std::unique_ptr<SessionGuard> guard;
  try {
    db = Db::open_rw(db_path_);
    set_pragmas(db);
    create_tables(db);
    guard = std::make_unique<SessionGuard>(db, session_id_, options_.resume_step, blobs_->dir());
    guard->initialize();
  } catch (...) {
    init_error_ = std::current_exception();
    std::lock_guard<std::mutex> lock(qmutex_);
    ready_ = true;
    thread_alive_ = false;
    q_all_done_.notify_all();
    return;
  }
  {
    std::lock_guard<std::mutex> lock(qmutex_);
    ready_ = true;
    q_all_done_.notify_all();
  }

  const auto interval = std::chrono::duration<double>(options_.flush_interval_secs);
  while (true) {
    std::vector<WriteItem> batch;
    bool shutdown = false, mark_complete = false;
    {
      std::unique_lock<std::mutex> lock(qmutex_);
      q_not_empty_.wait_for(lock, interval, [this] {
        return !queue_.empty() || shutdown_requested_ || mark_complete_requested_;
      });
      while (!queue_.empty()) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
      shutdown = shutdown_requested_;
      mark_complete = mark_complete_requested_;
      mark_complete_requested_ = false;
    }
    bool errored = false;
    if (!batch.empty()) {
      commit_batch(batch);
      {
        std::lock_guard<std::mutex> lock(error_mutex_);
        errored = sticky_error_.has_value();
      }
      // task_done after commit so join() == durability
      std::lock_guard<std::mutex> lock(qmutex_);
      unfinished_ -= batch.size();
      q_not_full_.notify_all();
      q_all_done_.notify_all();
    }
    if (mark_complete && !errored) {
      try {
        guard->mark_complete();
      } catch (const std::exception &e) {
        set_sticky_error({}, e.what());
      }
      std::lock_guard<std::mutex> lock(qmutex_);
      unfinished_ -= 1;  // the mark-complete request counted as one unit
      q_all_done_.notify_all();
    }
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      errored = sticky_error_.has_value();
    }
    if (shutdown || errored) {
      std::lock_guard<std::mutex> lock(qmutex_);
      queue_.clear();
      unfinished_ = 0;
      thread_alive_ = false;
      q_not_full_.notify_all();
      q_all_done_.notify_all();
      break;
    }
  }
}

void SummaryWriter::commit_batch(std::vector<WriteItem> &batch) {
  static const double backoff[] = {0.1, 0.5, 2.0};
  Db db = Db::open_rw(db_path_);  // fresh handle per batch keeps the loop simple and retry-safe
  db.exec("PRAGMA busy_timeout = 5000");
  const int retries = std::max(1, options_.max_retries_on_disk_error);
  for (int attempt = 0; attempt < retries; ++attempt) {
    try {
      db.begin();
      for (const auto &item : batch) insert(db, item);
      db.commit();
      return;
    } catch (const DbError &e) {
      try {
        db.rollback();
      } catch (...) {
      }
      if (e.retryable() && attempt < retries - 1) {
        const double wait = backoff[std::min(attempt, 2)];
        std::this_thread::sleep_for(std::chrono::duration<double>(wait));
        continue;
      }
      set_sticky_error(batch, e.what());
      return;
    } catch (const std::exception &e) {
      try {
        db.rollback();
      } catch (...) {
      }
      set_sticky_error(batch, e.what());
      return;
    }
  }
}

void SummaryWriter::set_sticky_error(const std::vector<WriteItem> &batch, const std::string &what) {
  std::optional<long long> lo, hi;
  for (const auto &item : batch)
    if (item.step) {
      lo = lo ? std::min(*lo, *item.step) : *item.step;
      hi = hi ? std::max(*hi, *item.step) : *item.step;
    }
  std::ostringstream msg;
  msg << "SerenityBoard: commit failed after " << std::max(1, options_.max_retries_on_disk_error)
      << " attempts. Last error: " << what << ". Lost batch: " << batch.size() << " items, ";
  if (lo) msg << "steps " << *lo << "-" << *hi << ".";
  else msg << "metadata only.";
  std::lock_guard<std::mutex> lock(error_mutex_);
  sticky_error_ = msg.str();
}

void SummaryWriter::insert(Db &db, const WriteItem &item) {
  static const std::map<std::string, std::string> sql = {
      {"scalars", "INSERT OR REPLACE INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)"},
      {"text_events", "INSERT OR REPLACE INTO text_events (tag, step, wall_time, value) VALUES (?, ?, ?, ?)"},
      {"artifacts",
       "INSERT OR REPLACE INTO artifacts (tag, step, seq_index, wall_time, kind, mime_type, blob_key, width, height, "
       "meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"},
      {"trace_events",
       "INSERT OR REPLACE INTO trace_events (step, wall_time, phase, duration_ms, details) VALUES (?, ?, ?, ?, ?)"},
      {"eval_results",
       "INSERT OR REPLACE INTO eval_results (suite_name, case_id, step, wall_time, score_name, score_value, "
       "artifact_key, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"},
      {"tensors", "INSERT OR REPLACE INTO tensors (tag, step, wall_time, dtype, shape, data) VALUES (?, ?, ?, ?, ?, ?)"},
      {"hparam_metrics",
       "INSERT OR REPLACE INTO hparam_metrics (metric_tag, value, step, wall_time) VALUES (?, ?, ?, ?)"},
      {"metadata", "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)"},
      {"plugin_data",
       "INSERT OR REPLACE INTO plugin_data (plugin_name, tag, step, wall_time, data) VALUES (?, ?, ?, ?, ?)"},
      {"custom_scalar_layouts",
       "INSERT OR REPLACE INTO custom_scalar_layouts (layout_name, config) VALUES (?, ?)"},
      {"pr_curves",
       "INSERT OR REPLACE INTO pr_curves (tag, step, class_index, wall_time, num_thresholds, data) VALUES (?, ?, ?, ?, "
       "?, ?)"},
      {"audio",
       "INSERT OR REPLACE INTO audio (tag, step, seq_index, wall_time, blob_key, sample_rate, num_channels, "
       "duration_ms, mime_type, label) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"},
      {"graphs", "INSERT OR REPLACE INTO graphs (tag, step, wall_time, graph_blob_key) VALUES (?, ?, ?, ?)"},
      {"embeddings",
       "INSERT OR REPLACE INTO embeddings (tag, step, wall_time, num_points, dimensions, tensor_blob_key, "
       "metadata_json, metadata_header, sprite_blob_key, sprite_single_h, sprite_single_w) VALUES (?, ?, ?, ?, ?, ?, "
       "?, ?, ?, ?, ?)"},
      {"meshes",
       "INSERT OR REPLACE INTO meshes (tag, step, wall_time, num_vertices, has_faces, has_colors, num_faces, "
       "vertices_blob_key, faces_blob_key, colors_blob_key, config_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"},
  };
  auto it = sql.find(item.table);
  if (it == sql.end()) throw std::runtime_error("Unknown table: '" + item.table + "'");
  db.run(it->second, item.params);
}

// ── public API ─────────────────────────────────────────────────────────

void SummaryWriter::add_scalar(const std::string &tag, double value, long long step) {
  if (noop_) return;
  check_error();
  enqueue("scalars", tag,
          {"scalars", {Param::text(tag), Param::integer(step), Param::real(wall_time_now()), Param::real(value)}, step});
}

void SummaryWriter::add_scalars(const std::string &main_tag, const std::map<std::string, double> &values,
                                long long step) {
  for (const auto &[sub, v] : values) add_scalar(main_tag + "/" + sub, v, step);
}

void SummaryWriter::add_image(const std::string &tag, std::uint32_t width, std::uint32_t height,
                              std::uint32_t channels, const std::uint8_t *pixels, long long step) {
  if (noop_) return;
  check_error();
  const std::string png = encode_png(width, height, channels, pixels);
  const std::string key = blobs_->store(png, "png");
  enqueue("artifacts", tag,
          {"artifacts",
           {Param::text(tag), Param::integer(step), Param::integer(0), Param::real(wall_time_now()),
            Param::text("image"), Param::text("image/png"), Param::text(key), Param::integer(width),
            Param::integer(height), Param::text("{}")},
           step});
}

void SummaryWriter::add_image_file(const std::string &tag, const std::string &png_path, long long step) {
  if (noop_) return;
  check_error();
  std::ifstream in(png_path, std::ios::binary);
  if (!in) throw std::invalid_argument("cannot read " + png_path);
  std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (bytes.size() < 24 || bytes.compare(0, 8, "\x89PNG\r\n\x1a\n") != 0) throw std::invalid_argument("not a PNG: " + png_path);
  auto be32 = [&](std::size_t off) {
    return (std::uint32_t(std::uint8_t(bytes[off])) << 24) | (std::uint32_t(std::uint8_t(bytes[off + 1])) << 16) |
           (std::uint32_t(std::uint8_t(bytes[off + 2])) << 8) | std::uint32_t(std::uint8_t(bytes[off + 3]));
  };
  const std::uint32_t width = be32(16), height = be32(20);
  const std::string key = blobs_->store(bytes, "png");
  enqueue("artifacts", tag,
          {"artifacts",
           {Param::text(tag), Param::integer(step), Param::integer(0), Param::real(wall_time_now()),
            Param::text("image"), Param::text("image/png"), Param::text(key), Param::integer(width),
            Param::integer(height), Param::text("{}")},
           step});
}

void SummaryWriter::add_histogram(const std::string &tag, const std::vector<double> &values, long long step,
                                  int bins) {
  if (noop_) return;
  check_error();
  const auto rows = histogram_rows(values, bins);
  if (rows.empty()) return;
  const nlohmann::json shape = nlohmann::json::array({rows.size() / 3, 3});
  enqueue("tensors", tag,
          {"tensors",
           {Param::text(tag), Param::integer(step), Param::real(wall_time_now()), Param::text("float64"),
            Param::text(shape.dump()), Param::blob(f64_bytes(rows))},
           step});
}

void SummaryWriter::add_text(const std::string &tag, const std::string &text, long long step) {
  if (noop_) return;
  check_error();
  enqueue("text_events", tag,
          {"text_events", {Param::text(tag), Param::integer(step), Param::real(wall_time_now()), Param::text(text)}, step});
}

void SummaryWriter::add_hparams(const nlohmann::json &hparams, const std::map<std::string, double> &metrics) {
  if (noop_) return;
  check_error();
  queue_put({"metadata", {Param::text("hparams"), Param::text(hparams.dump())}, std::nullopt});
  const double wt = wall_time_now();
  for (const auto &[tag, value] : metrics)
    queue_put({"hparam_metrics", {Param::text(tag), Param::real(value), Param::null(), Param::real(wt)}, std::nullopt});
}

void SummaryWriter::add_trace(long long step, const std::string &phase, double duration_ms,
                              const std::optional<nlohmann::json> &details) {
  if (noop_) return;
  check_error();
  enqueue("trace_events", "trace/" + phase,
          {"trace_events",
           {Param::integer(step), Param::real(wall_time_now()), Param::text(phase), Param::real(duration_ms),
            Param::text(details ? details->dump() : "{}")},
           step});
}

void SummaryWriter::add_eval(const std::string &suite_name, const std::string &case_id, long long step,
                             const std::string &score_name, double score_value,
                             const std::optional<std::string> &artifact, const std::optional<nlohmann::json> &details) {
  if (noop_) return;
  check_error();
  queue_put({"eval_results",
             {Param::text(suite_name), Param::text(case_id), Param::integer(step), Param::real(wall_time_now()),
              Param::text(score_name), Param::real(score_value), artifact ? Param::text(*artifact) : Param::null(),
              Param::text(details ? details->dump() : "{}")},
             step});
}

void SummaryWriter::add_custom_scalars_layout(const nlohmann::json &layout) {
  if (noop_) return;
  check_error();
  queue_put({"custom_scalar_layouts", {Param::text("default"), Param::text(layout.dump())}, std::nullopt});
}

void SummaryWriter::add_pr_curve(const std::string &tag, const std::vector<double> &labels,
                                 const std::vector<double> &predictions, long long step, int num_thresholds,
                                 int class_index) {
  if (noop_) return;
  check_error();
  const auto data = pr_curve_rows(labels, predictions, num_thresholds);
  enqueue("pr_curves", tag,
          {"pr_curves",
           {Param::text(tag), Param::integer(step), Param::integer(class_index), Param::real(wall_time_now()),
            Param::integer(num_thresholds), Param::blob(f64_bytes(data))},
           step});
}

void SummaryWriter::add_audio(const std::string &tag, const std::vector<std::int16_t> &samples,
                              std::uint32_t channels, std::uint32_t sample_rate, long long step) {
  if (noop_) return;
  check_error();
  if (channels == 0) throw std::invalid_argument("channels must be >= 1");
  const std::string wav = encode_wav_pcm16(samples, channels, sample_rate);
  const double duration_ms = (static_cast<double>(samples.size()) / channels / sample_rate) * 1000.0;
  const std::string key = blobs_->store(wav, "wav");
  enqueue("audio", tag,
          {"audio",
           {Param::text(tag), Param::integer(step), Param::integer(0), Param::real(wall_time_now()), Param::text(key),
            Param::integer(sample_rate), Param::integer(channels), Param::real(duration_ms), Param::text("audio/wav"),
            Param::text("")},
           step});
}

void SummaryWriter::add_audio_float(const std::string &tag, const std::vector<float> &samples,
                                    std::uint32_t channels, std::uint32_t sample_rate, long long step) {
  std::vector<std::int16_t> pcm(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const float clipped = std::max(-1.0f, std::min(1.0f, samples[i]));
    pcm[i] = static_cast<std::int16_t>(clipped * 32767.0f);  // truncation, like astype(int16)
  }
  add_audio(tag, pcm, channels, sample_rate, step);
}

void SummaryWriter::add_plugin_data(const std::string &plugin_name, const std::string &tag,
                                    const nlohmann::json &data, long long step) {
  if (noop_) return;
  check_error();
  enqueue("plugin_data", plugin_name + "/" + tag,
          {"plugin_data",
           {Param::text(plugin_name), Param::text(tag), Param::integer(step), Param::real(wall_time_now()),
            Param::text(data.dump())},
           step});
}

void SummaryWriter::add_graph_json(const nlohmann::json &graph, const std::string &tag, long long step) {
  if (noop_) return;
  check_error();
  const std::string key = blobs_->store(graph.dump(), "json");
  enqueue("graphs", tag,
          {"graphs", {Param::text(tag), Param::integer(step), Param::real(wall_time_now()), Param::text(key)}, step});
}

void SummaryWriter::add_embedding(const std::vector<float> &matrix, std::size_t num_points, std::size_t dimensions,
                                  const std::optional<std::vector<std::string>> &metadata, long long step,
                                  const std::string &tag,
                                  const std::optional<std::vector<std::string>> &metadata_header) {
  if (noop_) return;
  check_error();
  if (matrix.size() != num_points * dimensions) throw std::invalid_argument("embedding matrix size mismatch");
  if (metadata && metadata->size() != num_points)
    throw std::invalid_argument("metadata length must equal the number of points");
  std::string bytes(matrix.size() * sizeof(float), '\0');
  if (!matrix.empty()) std::memcpy(bytes.data(), matrix.data(), bytes.size());
  const std::string key = blobs_->store(bytes, "emb");
  enqueue("embeddings", tag,
          {"embeddings",
           {Param::text(tag), Param::integer(step), Param::real(wall_time_now()), Param::integer(static_cast<long long>(num_points)),
            Param::integer(static_cast<long long>(dimensions)), Param::text(key),
            metadata ? Param::text(nlohmann::json(*metadata).dump()) : Param::null(),
            metadata_header ? Param::text(nlohmann::json(*metadata_header).dump()) : Param::null(), Param::null(),
            Param::null(), Param::null()},
           step});
}

void SummaryWriter::add_mesh(const std::string &tag, const std::vector<float> &vertices_xyz,
                             const std::optional<std::vector<std::uint8_t>> &colors_rgb,
                             const std::optional<std::vector<std::int32_t>> &faces,
                             const std::optional<nlohmann::json> &config, long long step) {
  if (noop_) return;
  check_error();
  if (vertices_xyz.size() % 3 != 0) throw std::invalid_argument("vertices must be (N,3)");
  const long long n = static_cast<long long>(vertices_xyz.size() / 3);
  auto bytes_of = [](const void *p, std::size_t n_bytes) {
    std::string s(n_bytes, '\0');
    if (n_bytes) std::memcpy(s.data(), p, n_bytes);
    return s;
  };
  const std::string vkey = blobs_->store(bytes_of(vertices_xyz.data(), vertices_xyz.size() * 4), "bin");
  Param fkey = Param::null(), ckey = Param::null();
  long long num_faces = 0;
  if (faces) {
    if (faces->size() % 3 != 0) throw std::invalid_argument("faces must be (F,3)");
    num_faces = static_cast<long long>(faces->size() / 3);
    fkey = Param::text(blobs_->store(bytes_of(faces->data(), faces->size() * 4), "bin"));
  }
  if (colors_rgb) {
    if (colors_rgb->size() != vertices_xyz.size()) throw std::invalid_argument("colors must be (N,3)");
    ckey = Param::text(blobs_->store(bytes_of(colors_rgb->data(), colors_rgb->size()), "bin"));
  }
  enqueue("meshes", tag,
          {"meshes",
           {Param::text(tag), Param::integer(step), Param::real(wall_time_now()), Param::integer(n),
            Param::integer(faces ? 1 : 0), Param::integer(colors_rgb ? 1 : 0), Param::integer(num_faces),
            Param::text(vkey), fkey, ckey, config ? Param::text(config->dump()) : Param::null()},
           step});
}

void SummaryWriter::flush() {
  if (noop_) return;
  check_error();
  if (!thread_alive_) {
    check_error();
    return;
  }
  drain_reservoirs();
  queue_join();
  check_error();
}

void SummaryWriter::close() {
  if (noop_ || closed_) return;
  closed_ = true;
  try {
    if (sys_metrics_) {
      sys_metrics_->stop();
      sys_metrics_.reset();
    }
    drain_reservoirs();
    if (thread_alive_) queue_join();
    if (thread_alive_) {
      {
        std::lock_guard<std::mutex> lock(qmutex_);
        mark_complete_requested_ = true;
        ++unfinished_;
        q_not_empty_.notify_one();
      }
      queue_join();
    }
  } catch (...) {
  }
  {
    std::lock_guard<std::mutex> lock(qmutex_);
    shutdown_requested_ = true;
    q_not_empty_.notify_all();
  }
  if (thread_.joinable()) thread_.join();
  check_error();
}

}  // namespace sb
