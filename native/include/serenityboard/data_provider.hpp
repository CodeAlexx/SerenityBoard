// Read-only provider for one run's board.db (serenityboard/server/data_provider.py).
// One instance per run; all methods serialize on an internal mutex (the server
// answers requests from several threads).
#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "serenityboard/db.hpp"

namespace sb {

using Json = nlohmann::json;

struct ScalarRow {
  long long step;
  double wall_time;
  double value;
};

class RunDataProvider {
public:
  explicit RunDataProvider(const std::string &db_path);
  const std::string &run_dir() const { return run_dir_; }
  const std::string &db_path() const { return db_path_; }

  std::vector<ScalarRow> read_scalars_incremental(const std::string &tag);
  std::vector<ScalarRow> read_scalars_downsampled(const std::string &tag, long long n);
  Json read_scalars_last(const std::vector<std::string> &tags);
  Json get_tags();
  Json get_run_info();
  std::pair<std::optional<double>, std::optional<long long>> get_last_activity();
  Json get_hparams();
  Json read_histograms(const std::string &tag, long long downsample = 100);
  Json read_distributions(const std::string &tag, long long downsample = 100);
  Json read_pr_curves(const std::string &tag, long long downsample = 50);
  Json read_images(const std::string &tag, long long downsample = 100);
  std::optional<std::string> get_blob_mime(const std::string &blob_key);
  Json read_audio(const std::string &tag, long long downsample = 50);
  Json read_meshes(const std::optional<std::string> &tag, const std::optional<long long> &step, long long downsample = 50);
  Json read_embeddings(const std::optional<std::string> &tag, const std::optional<long long> &step,
                       long long downsample = 20);
  Json read_graphs(const std::optional<std::string> &tag, long long downsample = 10);
  Json read_text(const std::string &tag, const std::optional<long long> &limit);
  Json read_artifacts(const std::string &tag, long long downsample, const std::optional<std::string> &kind);
  Json read_trace_events(const std::optional<long long> &step_from, const std::optional<long long> &step_to);
  Json read_trace_events_incremental();
  Json read_eval_results(const std::string &suite_name, const std::optional<long long> &step);
  Json read_eval_results_incremental();
  std::optional<std::string> get_active_session_id();
  std::optional<long long> get_resume_step(const std::string &session_id);
  Json get_all_metric_tags();
  Json read_metric_timeseries(const Json &requests);
  std::optional<Json> get_custom_scalar_layout();
  Json read_custom_scalars(const std::vector<std::string> &tag_regexes, long long downsample = 5000);
  Json get_note();
  void set_note(const std::string &text);
  void close();

private:
  void check_session();
  Json histogram_rows_json(const std::string &tag, long long downsample, bool distributions);
  std::optional<Json> analyze_audio_blob(const std::string &blob_key);

  std::string db_path_, run_dir_;
  Db conn_;
  std::mutex mutex_;
  std::map<std::string, long long> last_seen_;
  std::optional<std::string> known_session_id_;
  std::map<std::string, std::optional<Json>> audio_cache_;
};

/// fnmatch-style glob (Python fnmatch semantics on POSIX: case-sensitive).
bool glob_match(const std::string &pattern, const std::string &text);

}  // namespace sb
