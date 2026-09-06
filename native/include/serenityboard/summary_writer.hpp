// Native SummaryWriter — a port of serenityboard/writer/summary_writer.py +
// async_writer.py. One background writer thread owns the SQLite connection;
// producers enqueue WriteItems (bounded, blocking queue); lossy classes go
// through per-tag reservoirs drained on flush()/close().
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "serenityboard/blob_storage.hpp"
#include "serenityboard/db.hpp"
#include "serenityboard/reservoir.hpp"

namespace sb {

class WriterError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct WriteItem {
  std::string table;
  std::vector<Param> params;
  std::optional<long long> step;
};

struct WriterOptions {
  std::string run_name;                       // default: UTC %Y%m%d_%H%M%S
  std::optional<nlohmann::json> hparams;      // metadata.hparams
  std::optional<long long> resume_step;
  std::size_t max_queue_size{1000};
  double flush_interval_secs{2.0};
  int max_retries_on_disk_error{3};
  std::map<std::string, std::size_t> reservoir_config;  // overrides per data class
  std::optional<int> rank;                    // else SB_RANK / RANK / LOCAL_RANK / 0
  bool system_metrics{true};
  double system_metrics_interval_secs{10.0};
};

class SystemMetricsCollector;

class SummaryWriter {
public:
  SummaryWriter(const std::string &logdir, WriterOptions options = {});
  ~SummaryWriter();
  SummaryWriter(const SummaryWriter &) = delete;
  SummaryWriter &operator=(const SummaryWriter &) = delete;

  bool is_noop() const { return noop_; }
  const std::string &run_dir() const { return run_dir_; }
  const std::string &db_path() const { return db_path_; }
  const std::string &session_id() const { return session_id_; }

  void add_scalar(const std::string &tag, double value, long long step);
  void add_scalars(const std::string &main_tag, const std::map<std::string, double> &values, long long step);
  /// HWC / HW pixels; channels 1, 3 or 4 (uint8). CHW callers transpose first.
  void add_image(const std::string &tag, std::uint32_t width, std::uint32_t height, std::uint32_t channels,
                 const std::uint8_t *pixels, long long step);
  void add_image_file(const std::string &tag, const std::string &png_path, long long step);
  void add_histogram(const std::string &tag, const std::vector<double> &values, long long step, int bins = 64);
  void add_text(const std::string &tag, const std::string &text, long long step);
  void add_hparams(const nlohmann::json &hparams, const std::map<std::string, double> &metrics);
  void add_trace(long long step, const std::string &phase, double duration_ms,
                 const std::optional<nlohmann::json> &details = std::nullopt);
  void add_eval(const std::string &suite_name, const std::string &case_id, long long step, const std::string &score_name,
                double score_value, const std::optional<std::string> &artifact = std::nullopt,
                const std::optional<nlohmann::json> &details = std::nullopt);
  void add_custom_scalars_layout(const nlohmann::json &layout);
  void add_pr_curve(const std::string &tag, const std::vector<double> &labels, const std::vector<double> &predictions,
                    long long step, int num_thresholds = 201, int class_index = 0);
  /// Interleaved int16 samples (frames * channels).
  void add_audio(const std::string &tag, const std::vector<std::int16_t> &samples, std::uint32_t channels,
                 std::uint32_t sample_rate, long long step);
  void add_audio_float(const std::string &tag, const std::vector<float> &samples, std::uint32_t channels,
                       std::uint32_t sample_rate, long long step);
  void add_plugin_data(const std::string &plugin_name, const std::string &tag, const nlohmann::json &data, long long step);
  /// Pre-built graph JSON (serenityboard graph document); stored as a blob.
  void add_graph_json(const nlohmann::json &graph, const std::string &tag = "default", long long step = 0);
  /// Row-major float32 matrix (num_points x dimensions) + optional metadata rows.
  void add_embedding(const std::vector<float> &matrix, std::size_t num_points, std::size_t dimensions,
                     const std::optional<std::vector<std::string>> &metadata = std::nullopt, long long step = 0,
                     const std::string &tag = "default",
                     const std::optional<std::vector<std::string>> &metadata_header = std::nullopt);
  void add_mesh(const std::string &tag, const std::vector<float> &vertices_xyz,
                const std::optional<std::vector<std::uint8_t>> &colors_rgb,
                const std::optional<std::vector<std::int32_t>> &faces,
                const std::optional<nlohmann::json> &config, long long step);

  void flush();
  void close();

private:
  friend class SystemMetricsCollector;
  void enqueue(const std::string &data_class, const std::string &tag, WriteItem item);
  void queue_put(WriteItem item);
  void queue_join();
  void drain_reservoirs();
  void check_error();
  void writer_loop();
  void commit_batch(std::vector<WriteItem> &batch);
  void insert(Db &db, const WriteItem &item);
  void set_sticky_error(const std::vector<WriteItem> &batch, const std::string &what);

  std::string logdir_, run_name_, run_dir_, db_path_, session_id_;
  WriterOptions options_;
  bool noop_{false};
  bool closed_{false};
  std::unique_ptr<BlobStorage> blobs_;

  // queue
  std::mutex qmutex_;
  std::condition_variable q_not_full_, q_not_empty_, q_all_done_;
  std::deque<WriteItem> queue_;
  std::size_t unfinished_{0};
  bool shutdown_requested_{false};
  bool mark_complete_requested_{false};
  std::thread thread_;
  std::exception_ptr init_error_;
  bool ready_{false};
  bool thread_alive_{false};

  // sticky error
  std::mutex error_mutex_;
  std::optional<std::string> sticky_error_;

  std::map<std::string, std::unique_ptr<Reservoir<WriteItem>>> reservoirs_;
  std::unique_ptr<SystemMetricsCollector> sys_metrics_;
};

}  // namespace sb
