#include "serenityboard/data_provider.hpp"

#include <fnmatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>

#include "serenityboard/encoders.hpp"
#include "serenityboard/session_guard.hpp"

namespace sb {

namespace fs = std::filesystem;

bool glob_match(const std::string &pattern, const std::string &text) {
  return ::fnmatch(pattern.c_str(), text.c_str(), 0) == 0;
}

namespace {

Json json_or_empty_object(const std::string &text) {
  if (text.empty()) return Json::object();
  try {
    return Json::parse(text);
  } catch (...) {
    return Json::object();
  }
}

Json json_or_null(const std::string &text) {
  if (text.empty()) return nullptr;
  try {
    return Json::parse(text);
  } catch (...) {
    return nullptr;
  }
}

std::vector<double> f64_from_blob(const std::string &blob) {
  std::vector<double> out(blob.size() / sizeof(double));
  if (!out.empty()) std::memcpy(out.data(), blob.data(), out.size() * sizeof(double));
  return out;
}

Json null_or_int(const Row &r, int col) { return r.is_null(col) ? Json(nullptr) : Json(r.as_int(col)); }
Json null_or_double(const Row &r, int col) { return r.is_null(col) ? Json(nullptr) : Json(r.as_double(col)); }
Json null_or_text(const Row &r, int col) { return r.is_null(col) ? Json(nullptr) : Json(r.as_text(col)); }

// Python: indices = set(range(0, n, max(1, n // downsample))); add n-1; sorted[:downsample]
template <typename T>
std::vector<T> index_downsample(std::vector<T> rows, long long downsample) {
  const long long n = static_cast<long long>(rows.size());
  if (n <= downsample) return rows;
  std::set<long long> indices;
  const long long stride = std::max(1LL, n / downsample);
  for (long long i = 0; i < n; i += stride) indices.insert(i);
  indices.insert(n - 1);
  std::vector<T> out;
  for (long long i : indices) {
    if (static_cast<long long>(out.size()) >= downsample) break;
    out.push_back(rows[static_cast<std::size_t>(i)]);
  }
  return out;
}

// Python: rows[::max(1, len // downsample)] (when downsample truthy and len > downsample)
template <typename T>
std::vector<T> stride_downsample(std::vector<T> rows, long long downsample) {
  const long long n = static_cast<long long>(rows.size());
  if (downsample <= 0 || n <= downsample) return rows;
  const long long stride = std::max(1LL, n / downsample);
  std::vector<T> out;
  for (long long i = 0; i < n; i += stride) out.push_back(rows[static_cast<std::size_t>(i)]);
  return out;
}

}  // namespace

RunDataProvider::RunDataProvider(const std::string &db_path)
    : db_path_(db_path), run_dir_(fs::path(db_path).parent_path().string()), conn_(Db::open_ro(db_path)) {}

void RunDataProvider::close() { conn_.close(); }

void RunDataProvider::check_session() {
  std::optional<std::string> current;
  conn_.query("SELECT value FROM metadata WHERE key = 'active_session_id'", {}, [&](const Row &r) {
    current = Json::parse(r.as_text(0)).get<std::string>();
    return false;
  });
  if (!current) return;
  if (current != known_session_id_) {
    known_session_id_ = current;
    last_seen_.clear();
  }
}

std::vector<ScalarRow> RunDataProvider::read_scalars_incremental(const std::string &tag) {
  std::lock_guard<std::mutex> lock(mutex_);
  check_session();
  long long last = -1;
  if (auto it = last_seen_.find(tag); it != last_seen_.end()) last = it->second;
  std::vector<ScalarRow> rows;
  conn_.query("SELECT step, wall_time, value FROM scalars WHERE tag = ? AND step > ? ORDER BY step",
              {Param::text(tag), Param::integer(last)}, [&](const Row &r) {
                rows.push_back({r.as_int(0), r.as_double(1), r.as_double(2)});
                return true;
              });
  if (!rows.empty()) last_seen_[tag] = rows.back().step;
  return rows;
}

std::vector<ScalarRow> RunDataProvider::read_scalars_downsampled(const std::string &tag, long long n) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto fetch_all = [&]() {
    std::vector<ScalarRow> rows;
    conn_.query("SELECT step, wall_time, value FROM scalars WHERE tag = ? ORDER BY step", {Param::text(tag)},
                [&](const Row &r) {
                  rows.push_back({r.as_int(0), r.as_double(1), r.as_double(2)});
                  return true;
                });
    return rows;
  };
  if (n <= 0) return fetch_all();
  n = std::max(n, 3LL);
  std::optional<long long> min_step, max_step;
  conn_.query("SELECT step FROM scalars WHERE tag = ? ORDER BY step ASC LIMIT 1", {Param::text(tag)},
              [&](const Row &r) { min_step = r.as_int(0); return false; });
  if (!min_step) return {};
  conn_.query("SELECT step FROM scalars WHERE tag = ? ORDER BY step DESC LIMIT 1", {Param::text(tag)},
              [&](const Row &r) { max_step = r.as_int(0); return false; });
  const long long step_range = *max_step - *min_step;
  if (step_range == 0) {
    std::vector<ScalarRow> rows;
    conn_.query("SELECT step, wall_time, value FROM scalars WHERE tag = ? AND step = ?",
                {Param::text(tag), Param::integer(*min_step)}, [&](const Row &r) {
                  rows.push_back({r.as_int(0), r.as_double(1), r.as_double(2)});
                  return true;
                });
    return rows;
  }
  long long total = 0;
  conn_.query("SELECT COUNT(*) FROM scalars WHERE tag = ?", {Param::text(tag)},
              [&](const Row &r) { total = r.as_int(0); return false; });
  if (total <= n) return fetch_all();
  std::set<long long> sample_steps{*min_step, *max_step};
  const double step_interval = static_cast<double>(step_range) / static_cast<double>(n - 1);
  for (long long i = 1; i < n - 1; ++i) {
    const long long target = *min_step + static_cast<long long>(static_cast<double>(i) * step_interval);
    conn_.query("SELECT step FROM scalars WHERE tag = ? AND step >= ? ORDER BY step ASC LIMIT 1",
                {Param::text(tag), Param::integer(target)}, [&](const Row &r) {
                  sample_steps.insert(r.as_int(0));
                  return false;
                });
  }
  std::vector<long long> sorted_steps(sample_steps.begin(), sample_steps.end());
  if (static_cast<long long>(sorted_steps.size()) > n) sorted_steps.resize(static_cast<std::size_t>(n));
  std::string placeholders;
  std::vector<Param> params{Param::text(tag)};
  for (std::size_t i = 0; i < sorted_steps.size(); ++i) {
    placeholders += i ? ",?" : "?";
    params.push_back(Param::integer(sorted_steps[i]));
  }
  std::vector<ScalarRow> rows;
  conn_.query("SELECT step, wall_time, value FROM scalars WHERE tag = ? AND step IN (" + placeholders +
                  ") ORDER BY step",
              params, [&](const Row &r) {
                rows.push_back({r.as_int(0), r.as_double(1), r.as_double(2)});
                return true;
              });
  return rows;
}

Json RunDataProvider::read_scalars_last(const std::vector<std::string> &tags) {
  std::lock_guard<std::mutex> lock(mutex_);
  Json result = Json::object();
  for (const auto &tag : tags)
    conn_.query("SELECT step, wall_time, value FROM scalars WHERE tag = ? ORDER BY step DESC LIMIT 1",
                {Param::text(tag)}, [&](const Row &r) {
                  result[tag] = {{"step", r.as_int(0)}, {"wall_time", r.as_double(1)}, {"value", r.as_double(2)}};
                  return false;
                });
  return result;
}

Json RunDataProvider::get_tags() {
  std::lock_guard<std::mutex> lock(mutex_);
  Json result = Json::object();
  auto distinct = [&](const char *sql) {
    std::vector<std::string> out;
    try {
      conn_.query(sql, {}, [&](const Row &r) {
        out.push_back(r.as_text(0));
        return true;
      });
    } catch (const DbError &) {
    }
    std::sort(out.begin(), out.end());
    return Json(out);
  };
  result["scalars"] = distinct("SELECT DISTINCT tag FROM scalars");
  result["tensors"] = distinct("SELECT DISTINCT tag FROM tensors");
  result["artifacts"] = distinct("SELECT DISTINCT tag FROM artifacts");
  result["text_events"] = distinct("SELECT DISTINCT tag FROM text_events");
  result["audio"] = distinct("SELECT DISTINCT tag FROM audio");
  result["trace_events"] = distinct("SELECT DISTINCT phase FROM trace_events");
  result["eval_suites"] = distinct("SELECT DISTINCT suite_name FROM eval_results");
  result["pr_curves"] = distinct("SELECT DISTINCT tag FROM pr_curves");
  result["graphs"] = distinct("SELECT DISTINCT tag FROM graphs");
  result["meshes"] = distinct("SELECT DISTINCT tag FROM meshes");
  result["embeddings"] = distinct("SELECT DISTINCT tag FROM embeddings");
  return result;
}

Json RunDataProvider::get_run_info() {
  std::lock_guard<std::mutex> lock(mutex_);
  Json meta = Json::object();
  conn_.query("SELECT key, value FROM metadata", {}, [&](const Row &r) {
    meta[r.as_text(0)] = json_or_null(r.as_text(1));
    return true;
  });
  return meta;
}

std::pair<std::optional<double>, std::optional<long long>> RunDataProvider::get_last_activity() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::optional<double> latest_wt;
  std::optional<long long> latest_step;
  for (const char *table : {"scalars", "tensors", "trace_events"}) {
    try {
      conn_.query(std::string("SELECT MAX(wall_time), MAX(step) FROM ") + table, {}, [&](const Row &r) {
        if (!r.is_null(0)) {
          const double wt = r.as_double(0);
          if (!latest_wt || wt > *latest_wt) latest_wt = wt;
          if (!r.is_null(1)) {
            const long long st = r.as_int(1);
            if (!latest_step || st > *latest_step) latest_step = st;
          }
        }
        return false;
      });
    } catch (const DbError &) {
    }
  }
  return {latest_wt, latest_step};
}

Json RunDataProvider::get_hparams() {
  Json meta = get_run_info();
  std::lock_guard<std::mutex> lock(mutex_);
  Json hparams = meta.contains("hparams") ? meta["hparams"] : Json::object();
  Json metrics = Json::object();
  conn_.query("SELECT metric_tag, value FROM hparam_metrics", {}, [&](const Row &r) {
    metrics[r.as_text(0)] = r.as_double(1);
    return true;
  });
  return {{"hparams", hparams}, {"metrics", metrics}};
}

Json RunDataProvider::histogram_rows_json(const std::string &tag, long long downsample, bool distributions) {
  struct Rec {
    long long step;
    double wall_time;
    std::string dtype, shape, data;
  };
  std::vector<Rec> rows;
  conn_.query("SELECT step, wall_time, dtype, shape, data FROM tensors WHERE tag = ? ORDER BY step",
              {Param::text(tag)}, [&](const Row &r) {
                rows.push_back({r.as_int(0), r.as_double(1), r.as_text(2), r.as_text(3), r.as_blob(4)});
                return true;
              });
  rows = index_downsample(std::move(rows), downsample);
  static const std::vector<int> basis_points = {0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000};
  Json result = Json::array();
  for (const auto &rec : rows) {
    std::vector<long long> shape;
    try {
      shape = Json::parse(rec.shape).get<std::vector<long long>>();
    } catch (...) {
      continue;
    }
    if (rec.dtype != "float64" || shape.size() != 2 || shape[1] != 3) continue;
    const auto values = f64_from_blob(rec.data);
    if (static_cast<long long>(values.size()) != shape[0] * 3) continue;
    const std::size_t nbins = static_cast<std::size_t>(shape[0]);
    if (!distributions) {
      Json bins = Json::array();
      for (std::size_t i = 0; i < nbins; ++i)
        bins.push_back({values[i * 3], values[i * 3 + 1], values[i * 3 + 2]});
      result.push_back({{"step", rec.step}, {"wall_time", rec.wall_time}, {"bins", bins}});
      continue;
    }
    double total = 0;
    for (std::size_t i = 0; i < nbins; ++i) total += values[i * 3 + 2];
    if (total == 0) continue;
    std::vector<double> cumulative(nbins), centers(nbins);
    double acc = 0;
    for (std::size_t i = 0; i < nbins; ++i) {
      acc += values[i * 3 + 2];
      cumulative[i] = acc / total;
      centers[i] = (values[i * 3] + values[i * 3 + 1]) / 2;
    }
    Json percentiles = Json::array();
    for (int bp : basis_points) {
      const double target = bp / 10000.0;
      double value;
      if (target <= 0) value = values[0];
      else if (target >= 1) value = values[(nbins - 1) * 3 + 1];
      else {
        // np.searchsorted(cumulative, target) (left): first index with cumulative[idx] >= target
        const std::size_t idx = static_cast<std::size_t>(
            std::lower_bound(cumulative.begin(), cumulative.end(), target) - cumulative.begin());
        if (idx == 0) value = centers[0];
        else if (idx >= nbins) value = centers[nbins - 1];
        else {
          const double frac = (target - cumulative[idx - 1]) / (cumulative[idx] - cumulative[idx - 1] + 1e-12);
          value = centers[idx - 1] + frac * (centers[idx] - centers[idx - 1]);
        }
      }
      percentiles.push_back({{"bp", bp}, {"value", value}});
    }
    result.push_back({{"step", rec.step}, {"wall_time", rec.wall_time}, {"percentiles", percentiles}});
  }
  return result;
}

Json RunDataProvider::read_histograms(const std::string &tag, long long downsample) {
  std::lock_guard<std::mutex> lock(mutex_);
  return histogram_rows_json(tag, downsample, false);
}

Json RunDataProvider::read_distributions(const std::string &tag, long long downsample) {
  std::lock_guard<std::mutex> lock(mutex_);
  return histogram_rows_json(tag, downsample, true);
}

Json RunDataProvider::read_pr_curves(const std::string &tag, long long downsample) {
  std::lock_guard<std::mutex> lock(mutex_);
  struct Rec {
    long long step;
    double wall_time;
    long long class_index, num_thresholds;
    std::string data;
  };
  std::vector<Rec> rows;
  try {
    conn_.query("SELECT step, wall_time, class_index, num_thresholds, data FROM pr_curves WHERE tag = ? ORDER BY step",
                {Param::text(tag)}, [&](const Row &r) {
                  rows.push_back({r.as_int(0), r.as_double(1), r.as_int(2), r.as_int(3), r.as_blob(4)});
                  return true;
                });
  } catch (const DbError &) {
    return Json::array();
  }
  rows = stride_downsample(std::move(rows), downsample);
  Json results = Json::array();
  for (const auto &rec : rows) {
    const auto values = f64_from_blob(rec.data);
    const std::size_t n = static_cast<std::size_t>(rec.num_thresholds);
    if (values.size() != 6 * n) continue;
    Json precision = Json::array(), recall = Json::array(), thresholds = Json::array();
    for (std::size_t i = 0; i < n; ++i) {
      precision.push_back(values[4 * n + i]);
      recall.push_back(values[5 * n + i]);
    }
    for (double t : linspace(0.0, 1.0, static_cast<int>(n))) thresholds.push_back(t);
    results.push_back({{"step", rec.step},
                       {"wall_time", rec.wall_time},
                       {"class_index", rec.class_index},
                       {"num_thresholds", rec.num_thresholds},
                       {"precision", precision},
                       {"recall", recall},
                       {"thresholds", thresholds}});
  }
  return results;
}

Json RunDataProvider::read_images(const std::string &tag, long long downsample) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Json> rows;
  conn_.query("SELECT step, wall_time, blob_key, width, height, kind, meta FROM artifacts WHERE tag = ? ORDER BY step",
              {Param::text(tag)}, [&](const Row &r) {
                rows.push_back({{"step", r.as_int(0)},
                                {"wall_time", r.as_double(1)},
                                {"blob_key", r.as_text(2)},
                                {"width", null_or_int(r, 3)},
                                {"height", null_or_int(r, 4)}});
                return true;
              });
  return Json(index_downsample(std::move(rows), downsample));
}

std::optional<std::string> RunDataProvider::get_blob_mime(const std::string &blob_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::optional<std::string> mime;
  auto probe = [&](const char *sql, const std::vector<Param> &params, const char *fixed) {
    if (mime) return;
    try {
      conn_.query(sql, params, [&](const Row &r) {
        mime = fixed ? std::string(fixed) : r.as_text(0);
        return false;
      });
    } catch (const DbError &) {
    }
  };
  probe("SELECT mime_type FROM artifacts WHERE blob_key = ? LIMIT 1", {Param::text(blob_key)}, nullptr);
  probe("SELECT mime_type FROM audio WHERE blob_key = ? LIMIT 1", {Param::text(blob_key)}, nullptr);
  probe("SELECT tag FROM meshes WHERE vertices_blob_key = ? OR faces_blob_key = ? OR colors_blob_key = ? LIMIT 1",
        {Param::text(blob_key), Param::text(blob_key), Param::text(blob_key)}, "application/octet-stream");
  probe("SELECT tag FROM embeddings WHERE tensor_blob_key = ? LIMIT 1", {Param::text(blob_key)},
        "application/octet-stream");
  probe("SELECT tag FROM embeddings WHERE sprite_blob_key = ? LIMIT 1", {Param::text(blob_key)}, "image/png");
  probe("SELECT tag FROM graphs WHERE graph_blob_key = ? LIMIT 1", {Param::text(blob_key)}, "application/json");
  return mime;
}

// ── audio analysis (data_provider.py:455-589) ────────────────────────

namespace {

std::optional<std::vector<float>> load_audio_samples(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::nullopt;
  std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (bytes.size() < 44 || bytes.compare(0, 4, "RIFF") != 0 || bytes.compare(8, 4, "WAVE") != 0) return std::nullopt;
  auto u16 = [&](std::size_t o) { return std::uint16_t(std::uint8_t(bytes[o])) | (std::uint16_t(std::uint8_t(bytes[o + 1])) << 8); };
  auto u32 = [&](std::size_t o) { return std::uint32_t(u16(o)) | (std::uint32_t(u16(o + 2)) << 16); };
  std::size_t pos = 12;
  int channels = 1, sample_width = 2;
  std::string data;
  bool have_fmt = false;
  while (pos + 8 <= bytes.size()) {
    const std::string id = bytes.substr(pos, 4);
    const std::uint32_t size = u32(pos + 4);
    const std::size_t body = pos + 8;
    if (id == "fmt " && body + 16 <= bytes.size()) {
      channels = std::max(1, int(u16(body + 2)));
      sample_width = u16(body + 14) / 8;
      have_fmt = true;
    } else if (id == "data") {
      data = bytes.substr(body, std::min<std::size_t>(size, bytes.size() - body));
      break;
    }
    pos = body + size + (size & 1U);
  }
  if (!have_fmt || data.empty()) return std::nullopt;
  std::vector<float> samples;
  if (sample_width == 1) {
    samples.reserve(data.size());
    for (unsigned char c : data) samples.push_back((static_cast<float>(c) - 128.0f) / 128.0f);
  } else if (sample_width == 2) {
    samples.reserve(data.size() / 2);
    for (std::size_t i = 0; i + 1 < data.size(); i += 2) {
      std::int16_t v;
      std::memcpy(&v, data.data() + i, 2);
      samples.push_back(static_cast<float>(v) / 32767.0f);
    }
  } else if (sample_width == 4) {
    samples.reserve(data.size() / 4);
    for (std::size_t i = 0; i + 3 < data.size(); i += 4) {
      std::int32_t v;
      std::memcpy(&v, data.data() + i, 4);
      samples.push_back(static_cast<float>(static_cast<double>(v) / 2147483647.0));
    }
  } else {
    return std::nullopt;
  }
  if (samples.empty()) return std::nullopt;
  if (channels > 1) {
    const std::size_t usable = (samples.size() / static_cast<std::size_t>(channels)) * static_cast<std::size_t>(channels);
    std::vector<float> mono;
    mono.reserve(usable / static_cast<std::size_t>(channels));
    for (std::size_t i = 0; i < usable; i += static_cast<std::size_t>(channels)) {
      float s = 0;
      for (int c = 0; c < channels; ++c) s += samples[i + static_cast<std::size_t>(c)];
      mono.push_back(s / static_cast<float>(channels));
    }
    samples = std::move(mono);
  }
  return samples;
}

Json waveform_preview(const std::vector<float> &s, int bins) {
  const std::size_t block = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(static_cast<double>(s.size()) / std::max(16, bins))));
  Json peaks = Json::array();
  for (std::size_t start = 0; start < s.size(); start += block) {
    float lo = s[start], hi = s[start];
    for (std::size_t i = start; i < std::min(s.size(), start + block); ++i) {
      lo = std::min(lo, s[i]);
      hi = std::max(hi, s[i]);
    }
    peaks.push_back({static_cast<double>(lo), static_cast<double>(hi)});
  }
  return peaks.empty() ? Json(nullptr) : peaks;
}

Json spectrogram_preview(const std::vector<float> &s, int frames, int bins) {
  if (s.size() < 32) return nullptr;
  const std::size_t window = std::min<std::size_t>(1024, s.size());
  if (window < 32) return nullptr;
  std::vector<std::size_t> starts;
  if (s.size() <= window) {
    starts.push_back(0);
  } else {
    const std::size_t frame_count = std::max<std::size_t>(8, std::min<std::size_t>(static_cast<std::size_t>(frames), s.size() - window + 1));
    // np.linspace(0, size-window, frame_count, dtype=int32): truncation toward zero
    for (std::size_t i = 0; i < frame_count; ++i) {
      const double v = static_cast<double>(s.size() - window) * static_cast<double>(i) / static_cast<double>(frame_count - 1);
      starts.push_back(static_cast<std::size_t>(v));
    }
  }
  std::vector<float> hann(window);
  for (std::size_t n = 0; n < window; ++n)
    hann[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(n) / static_cast<double>(window - 1)));
  std::vector<std::vector<double>> columns;
  const std::size_t nfreq = window / 2 + 1;
  for (std::size_t start : starts) {
    std::vector<double> frame(window, 0.0);
    for (std::size_t n = 0; n < window && start + n < s.size(); ++n) frame[n] = static_cast<double>(s[start + n] * hann[n]);
    // rfft magnitudes (direct DFT; window <= 1024)
    std::vector<double> spectrum;
    spectrum.reserve(nfreq);
    for (std::size_t k = 0; k < nfreq; ++k) {
      double re = 0, im = 0;
      const double w = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(window);
      for (std::size_t n = 0; n < window; ++n) {
        re += frame[n] * std::cos(w * static_cast<double>(n));
        im += frame[n] * std::sin(w * static_cast<double>(n));
      }
      spectrum.push_back(std::sqrt(re * re + im * im));
    }
    if (spectrum.size() <= 1) continue;
    std::vector<double> logspec;
    for (std::size_t i = 1; i < spectrum.size(); ++i) logspec.push_back(std::log1p(spectrum[i]));
    const std::size_t band = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(static_cast<double>(logspec.size()) / std::max(8, bins))));
    std::vector<double> bands;
    for (std::size_t i = 0; i < logspec.size(); i += band) {
      double sum = 0;
      std::size_t cnt = 0;
      for (std::size_t j = i; j < std::min(logspec.size(), i + band); ++j, ++cnt) sum += logspec[j];
      if (cnt) bands.push_back(sum / static_cast<double>(cnt));
    }
    while (static_cast<int>(bands.size()) < bins) bands.push_back(0.0);
    bands.resize(static_cast<std::size_t>(bins));
    columns.push_back(std::move(bands));
  }
  if (columns.empty()) return nullptr;
  float max_value = 0;
  for (const auto &c : columns)
    for (double v : c) max_value = std::max(max_value, static_cast<float>(v));
  Json matrix = Json::array();
  for (const auto &c : columns) {
    Json row = Json::array();
    for (double v : c) row.push_back(static_cast<double>(max_value > 0 ? static_cast<float>(v) / max_value : static_cast<float>(v)));
    matrix.push_back(row);
  }
  return matrix;
}

double amplitude_to_db(double v) { return 20.0 * std::log10(std::max(std::fabs(v), 1.0e-6)); }

// NumPy's float32 pairwise summation (numpy/core/src/umath/loops_utils.h.src):
// n < 8 -> serial; n <= 128 -> 8 accumulators; else recurse on halves (multiple of 8).
float numpy_pairwise_sum_f32(const float *a, std::size_t n) {
  if (n < 8) {
    float res = 0.f;
    for (std::size_t i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= 128) {
    float r[8];
    for (int i = 0; i < 8; ++i) r[i] = a[static_cast<std::size_t>(i)];
    std::size_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + static_cast<std::size_t>(j)];
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  const std::size_t n2 = (n / 2) - (n / 2) % 8;
  return numpy_pairwise_sum_f32(a, n2) + numpy_pairwise_sum_f32(a + n2, n - n2);
}

}  // namespace

std::optional<Json> RunDataProvider::analyze_audio_blob(const std::string &blob_key) {
  if (auto it = audio_cache_.find(blob_key); it != audio_cache_.end()) return it->second;
  const auto samples = load_audio_samples((fs::path(run_dir_) / "blobs" / blob_key).string());
  if (!samples || samples->empty()) {
    audio_cache_[blob_key] = std::nullopt;
    return std::nullopt;
  }
  float peak = 0;
  std::vector<float> squares(samples->size());
  for (std::size_t i = 0; i < samples->size(); ++i) {
    peak = std::max(peak, std::fabs((*samples)[i]));
    squares[i] = (*samples)[i] * (*samples)[i];  // np.square(samples, dtype=float32)
  }
  // np.mean(float32) = pairwise float32 sum / n (float32); np.sqrt(float32) -> float32; then Python float
  const float mean_sq = numpy_pairwise_sum_f32(squares.data(), squares.size()) / static_cast<float>(squares.size());
  const float rms = std::sqrt(mean_sq);
  Json analysis = {{"waveform", waveform_preview(*samples, 160)},
                   {"spectrogram", spectrogram_preview(*samples, 96, 48)},
                   {"peak_db", amplitude_to_db(peak)},
                   {"rms_db", amplitude_to_db(rms)}};
  audio_cache_[blob_key] = analysis;
  return analysis;
}

Json RunDataProvider::read_audio(const std::string &tag, long long downsample) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Json> rows;
  try {
    conn_.query("SELECT step, wall_time, blob_key, sample_rate, num_channels, duration_ms, mime_type, label FROM audio "
                "WHERE tag = ? ORDER BY step",
                {Param::text(tag)}, [&](const Row &r) {
                  rows.push_back({{"tag", tag},
                                  {"step", r.as_int(0)},
                                  {"wall_time", r.as_double(1)},
                                  {"blob_key", r.as_text(2)},
                                  {"sample_rate", r.as_int(3)},
                                  {"num_channels", r.as_int(4)},
                                  {"duration_ms", null_or_double(r, 5)},
                                  {"mime_type", r.as_text(6)},
                                  {"label", r.as_text(7)}});
                  return true;
                });
  } catch (const DbError &) {
    return Json::array();
  }
  rows = stride_downsample(std::move(rows), downsample);
  Json out = Json::array();
  for (auto &row : rows) {
    const auto analysis = analyze_audio_blob(row["blob_key"].get<std::string>());
    for (const char *k : {"waveform", "spectrogram", "peak_db", "rms_db"})
      row[k] = analysis ? (*analysis)[k] : Json(nullptr);
    out.push_back(std::move(row));
  }
  return out;
}

Json RunDataProvider::read_meshes(const std::optional<std::string> &tag, const std::optional<long long> &step,
                                  long long downsample) {
  std::lock_guard<std::mutex> lock(mutex_);
  const char *cols = "SELECT tag, step, wall_time, num_vertices, has_faces, has_colors, num_faces, vertices_blob_key, "
                     "faces_blob_key, colors_blob_key, config_json FROM meshes";
  auto to_json = [](const Row &r) {
    return Json{{"tag", r.as_text(0)},
                {"step", r.as_int(1)},
                {"wall_time", r.as_double(2)},
                {"num_vertices", r.as_int(3)},
                {"has_faces", r.as_int(4) != 0},
                {"has_colors", r.as_int(5) != 0},
                {"num_faces", r.as_int(6)},
                {"vertices_blob_key", r.as_text(7)},
                {"faces_blob_key", null_or_text(r, 8)},
                {"colors_blob_key", null_or_text(r, 9)},
                {"config", r.is_null(10) || r.as_text(10).empty() ? Json(nullptr) : json_or_null(r.as_text(10))}};
  };
  std::vector<Json> rows;
  try {
    if (tag && step) {
      conn_.query(std::string(cols) + " WHERE tag = ? AND step = ?", {Param::text(*tag), Param::integer(*step)},
                  [&](const Row &r) { rows.push_back(to_json(r)); return false; });
      return Json(rows);
    }
    if (tag)
      conn_.query(std::string(cols) + " WHERE tag = ? ORDER BY step", {Param::text(*tag)},
                  [&](const Row &r) { rows.push_back(to_json(r)); return true; });
    else
      conn_.query(std::string(cols) + " ORDER BY tag, step", {}, [&](const Row &r) { rows.push_back(to_json(r)); return true; });
  } catch (const DbError &) {
    return Json::array();
  }
  return Json(stride_downsample(std::move(rows), downsample));
}

Json RunDataProvider::read_embeddings(const std::optional<std::string> &tag, const std::optional<long long> &step,
                                      long long downsample) {
  std::lock_guard<std::mutex> lock(mutex_);
  const char *cols = "SELECT tag, step, wall_time, num_points, dimensions, tensor_blob_key, metadata_json, "
                     "metadata_header, sprite_blob_key, sprite_single_h, sprite_single_w FROM embeddings";
  auto to_json = [](const Row &r) {
    return Json{{"tag", r.as_text(0)},
                {"step", r.as_int(1)},
                {"wall_time", r.as_double(2)},
                {"num_points", r.as_int(3)},
                {"dimensions", r.as_int(4)},
                {"tensor_blob_key", r.as_text(5)},
                {"metadata", r.is_null(6) || r.as_text(6).empty() ? Json(nullptr) : json_or_null(r.as_text(6))},
                {"metadata_header", r.is_null(7) || r.as_text(7).empty() ? Json(nullptr) : json_or_null(r.as_text(7))},
                {"sprite_blob_key", null_or_text(r, 8)},
                {"sprite_single_h", null_or_int(r, 9)},
                {"sprite_single_w", null_or_int(r, 10)}};
  };
  std::vector<Json> rows;
  try {
    if (tag && step) {
      conn_.query(std::string(cols) + " WHERE tag = ? AND step = ?", {Param::text(*tag), Param::integer(*step)},
                  [&](const Row &r) { rows.push_back(to_json(r)); return false; });
      return Json(rows);
    }
    if (tag)
      conn_.query(std::string(cols) + " WHERE tag = ? ORDER BY step", {Param::text(*tag)},
                  [&](const Row &r) { rows.push_back(to_json(r)); return true; });
    else
      conn_.query(std::string(cols) + " ORDER BY tag, step", {}, [&](const Row &r) { rows.push_back(to_json(r)); return true; });
  } catch (const DbError &) {
    return Json::array();
  }
  return Json(stride_downsample(std::move(rows), downsample));
}

Json RunDataProvider::read_graphs(const std::optional<std::string> &tag, long long downsample) {
  std::lock_guard<std::mutex> lock(mutex_);
  struct Rec {
    std::string tag;
    long long step;
    double wall_time;
    std::string value;
  };
  std::vector<Rec> rows;
  try {
    if (tag)
      conn_.query("SELECT tag, step, wall_time, graph_blob_key FROM graphs WHERE tag = ? ORDER BY step",
                  {Param::text(*tag)}, [&](const Row &r) {
                    rows.push_back({r.as_text(0), r.as_int(1), r.as_double(2), r.as_text(3)});
                    return true;
                  });
    else
      conn_.query("SELECT tag, step, wall_time, graph_blob_key FROM graphs ORDER BY tag, step", {}, [&](const Row &r) {
        rows.push_back({r.as_text(0), r.as_int(1), r.as_double(2), r.as_text(3)});
        return true;
      });
  } catch (const DbError &) {
    return Json::array();
  }
  rows = stride_downsample(std::move(rows), downsample);
  Json results = Json::array();
  for (const auto &rec : rows) {
    if (rec.value.empty()) continue;
    Json graph;
    try {
      if (rec.value[0] == '{') {
        graph = Json::parse(rec.value);
      } else {
        std::ifstream in(fs::path(run_dir_) / "blobs" / rec.value, std::ios::binary);
        if (!in) continue;
        graph = Json::parse(in);
      }
    } catch (...) {
      continue;
    }
    results.push_back({{"tag", rec.tag}, {"step", rec.step}, {"wall_time", rec.wall_time}, {"graph_data", graph}});
  }
  return results;
}

Json RunDataProvider::read_text(const std::string &tag, const std::optional<long long> &limit) {
  std::lock_guard<std::mutex> lock(mutex_);
  Json out = Json::array();
  auto on_row = [&](const Row &r) {
    out.push_back({{"step", r.as_int(0)}, {"wall_time", r.as_double(1)}, {"value", r.as_text(2)}});
    return true;
  };
  if (limit)
    conn_.query("SELECT step, wall_time, value FROM text_events WHERE tag = ? ORDER BY step LIMIT ?",
                {Param::text(tag), Param::integer(*limit)}, on_row);
  else
    conn_.query("SELECT step, wall_time, value FROM text_events WHERE tag = ? ORDER BY step", {Param::text(tag)}, on_row);
  return out;
}

Json RunDataProvider::read_artifacts(const std::string &tag, long long downsample, const std::optional<std::string> &kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Json> rows;
  auto on_row = [&](const Row &r) {
    rows.push_back({{"step", r.as_int(0)},
                    {"wall_time", r.as_double(1)},
                    {"blob_key", r.as_text(2)},
                    {"mime_type", r.as_text(3)},
                    {"width", null_or_int(r, 4)},
                    {"height", null_or_int(r, 5)},
                    {"kind", r.as_text(6)},
                    {"meta", json_or_empty_object(r.as_text(7))}});
    return true;
  };
  if (kind && !kind->empty())
    conn_.query("SELECT step, wall_time, blob_key, mime_type, width, height, kind, meta FROM artifacts WHERE tag = ? "
                "AND kind = ? ORDER BY step",
                {Param::text(tag), Param::text(*kind)}, on_row);
  else
    conn_.query("SELECT step, wall_time, blob_key, mime_type, width, height, kind, meta FROM artifacts WHERE tag = ? "
                "ORDER BY step",
                {Param::text(tag)}, on_row);
  return Json(index_downsample(std::move(rows), downsample));
}

Json RunDataProvider::read_trace_events(const std::optional<long long> &step_from, const std::optional<long long> &step_to) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string where = "1=1";
  std::vector<Param> params;
  if (step_from) { where = "step >= ?"; params.push_back(Param::integer(*step_from)); }
  if (step_to) {
    where = step_from ? where + " AND step <= ?" : "step <= ?";
    params.push_back(Param::integer(*step_to));
  }
  Json out = Json::array();
  conn_.query("SELECT step, wall_time, phase, duration_ms, details FROM trace_events WHERE " + where + " ORDER BY step, phase",
              params, [&](const Row &r) {
                out.push_back({{"step", r.as_int(0)}, {"phase", r.as_text(2)}, {"duration_ms", r.as_double(3)},
                               {"details", json_or_empty_object(r.as_text(4))}});
                return true;
              });
  return out;
}

Json RunDataProvider::read_trace_events_incremental() {
  std::lock_guard<std::mutex> lock(mutex_);
  check_session();
  long long last = -1;
  if (auto it = last_seen_.find("__trace_events"); it != last_seen_.end()) last = it->second;
  Json out = Json::array();
  long long last_step = last;
  conn_.query("SELECT step, wall_time, phase, duration_ms, details FROM trace_events WHERE step > ? ORDER BY step, phase",
              {Param::integer(last)}, [&](const Row &r) {
                last_step = r.as_int(0);
                out.push_back({{"step", r.as_int(0)}, {"phase", r.as_text(2)}, {"duration_ms", r.as_double(3)},
                               {"details", json_or_empty_object(r.as_text(4))}});
                return true;
              });
  if (!out.empty()) last_seen_["__trace_events"] = last_step;
  return out;
}

Json RunDataProvider::read_eval_results(const std::string &suite_name, const std::optional<long long> &step) {
  std::lock_guard<std::mutex> lock(mutex_);
  Json out = Json::array();
  auto on_row = [&](const Row &r) {
    out.push_back({{"suite_name", r.as_text(0)},
                   {"case_id", r.as_text(1)},
                   {"step", r.as_int(2)},
                   {"wall_time", r.as_double(3)},
                   {"score_name", r.as_text(4)},
                   {"score_value", r.as_double(5)},
                   {"artifact_key", null_or_text(r, 6)},
                   {"details", json_or_empty_object(r.as_text(7))}});
    return true;
  };
  if (step)
    conn_.query("SELECT suite_name, case_id, step, wall_time, score_name, score_value, artifact_key, details FROM "
                "eval_results WHERE suite_name = ? AND step = ? ORDER BY case_id, score_name",
                {Param::text(suite_name), Param::integer(*step)}, on_row);
  else
    conn_.query("SELECT suite_name, case_id, step, wall_time, score_name, score_value, artifact_key, details FROM "
                "eval_results WHERE suite_name = ? ORDER BY step, case_id, score_name",
                {Param::text(suite_name)}, on_row);
  return out;
}

Json RunDataProvider::read_eval_results_incremental() {
  std::lock_guard<std::mutex> lock(mutex_);
  check_session();
  long long last = -1;
  if (auto it = last_seen_.find("__eval_results"); it != last_seen_.end()) last = it->second;
  Json out = Json::array();
  long long last_step = last;
  conn_.query("SELECT suite_name, case_id, step, wall_time, score_name, score_value, artifact_key, details FROM "
              "eval_results WHERE step > ? ORDER BY step, suite_name, case_id",
              {Param::integer(last)}, [&](const Row &r) {
                last_step = r.as_int(2);
                out.push_back({{"suite_name", r.as_text(0)}, {"case_id", r.as_text(1)}, {"step", r.as_int(2)},
                               {"score_name", r.as_text(4)}, {"score_value", r.as_double(5)}});
                return true;
              });
  if (!out.empty()) last_seen_["__eval_results"] = last_step;
  return out;
}

std::optional<std::string> RunDataProvider::get_active_session_id() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::optional<std::string> id;
  conn_.query("SELECT value FROM metadata WHERE key = 'active_session_id'", {}, [&](const Row &r) {
    const Json v = json_or_null(r.as_text(0));
    if (v.is_string()) id = v.get<std::string>();
    return false;
  });
  return id;
}

std::optional<long long> RunDataProvider::get_resume_step(const std::string &session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::optional<long long> out;
  conn_.query("SELECT resume_step FROM sessions WHERE session_id = ?", {Param::text(session_id)}, [&](const Row &r) {
    if (!r.is_null(0)) out = r.as_int(0);
    return false;
  });
  return out;
}

Json RunDataProvider::get_all_metric_tags() {
  std::lock_guard<std::mutex> lock(mutex_);
  Json result = Json::object();
  for (const char *table : {"scalars", "tensors", "artifacts", "text_events", "audio", "pr_curves", "graphs", "meshes", "embeddings"}) {
    Json entries = Json::array();
    try {
      conn_.query(std::string("SELECT tag, COUNT(*), MAX(step), MAX(wall_time) FROM ") + table + " GROUP BY tag ORDER BY tag", {},
                  [&](const Row &r) {
                    entries.push_back({{"tag", r.as_text(0)}, {"count", r.as_int(1)}, {"last_step", null_or_int(r, 2)},
                                       {"last_wall_time", null_or_double(r, 3)}});
                    return true;
                  });
    } catch (const DbError &) {
      entries = Json::array();
    }
    result[table] = entries;
  }
  return result;
}

Json RunDataProvider::read_metric_timeseries(const Json &requests) {
  Json results = Json::array();
  if (!requests.is_array()) return results;
  for (const auto &req : requests) {
    const std::string plugin = req.value("plugin", "scalars");
    const std::string tag = req.value("tag", "");
    const long long downsample = req.value("downsample", 100LL);
    Json data;
    if (plugin == "scalars") {
      data = Json::array();
      for (const auto &r : read_scalars_downsampled(tag, downsample))
        data.push_back({{"step", r.step}, {"wall_time", r.wall_time}, {"value", r.value}});
    } else if (plugin == "tensors") data = read_histograms(tag, downsample);
    else if (plugin == "text_events") data = read_text(tag, downsample);
    else if (plugin == "artifacts") data = read_artifacts(tag, downsample, std::nullopt);
    else if (plugin == "audio") data = read_audio(tag, downsample);
    else if (plugin == "pr_curves") data = read_pr_curves(tag, downsample);
    else if (plugin == "graphs") data = read_graphs(tag, downsample);
    else if (plugin == "meshes") data = read_meshes(tag, std::nullopt, downsample);
    else if (plugin == "embeddings") data = read_embeddings(tag, std::nullopt, downsample);
    else continue;
    results.push_back({{"plugin", plugin}, {"tag", tag}, {"data", data}});
  }
  return results;
}

std::optional<Json> RunDataProvider::get_custom_scalar_layout() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::optional<Json> out;
  conn_.query("SELECT config FROM custom_scalar_layouts WHERE layout_name = 'default'", {}, [&](const Row &r) {
    out = json_or_null(r.as_text(0));
    return false;
  });
  return out;
}

Json RunDataProvider::read_custom_scalars(const std::vector<std::string> &tag_regexes, long long downsample) {
  std::vector<std::string> all_tags;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    conn_.query("SELECT DISTINCT tag FROM scalars", {}, [&](const Row &r) {
      all_tags.push_back(r.as_text(0));
      return true;
    });
  }
  std::vector<std::regex> compiled;
  for (const auto &p : tag_regexes) {
    try {
      compiled.emplace_back(p, std::regex::ECMAScript);
    } catch (const std::regex_error &) {
    }
  }
  std::set<std::string> matched;
  for (const auto &tag : all_tags)
    for (const auto &re : compiled)
      if (std::regex_search(tag, re)) {
        matched.insert(tag);
        break;
      }
  Json result = Json::object();
  for (const auto &tag : matched) {
    Json rows = Json::array();
    for (const auto &r : read_scalars_downsampled(tag, downsample)) rows.push_back({r.step, r.wall_time, r.value});
    result[tag] = rows;
  }
  return result;
}

Json RunDataProvider::get_note() {
  std::lock_guard<std::mutex> lock(mutex_);
  Json out = {{"note", ""}, {"updated_at", nullptr}};
  try {
    conn_.query("SELECT note, updated_at FROM run_notes ORDER BY id LIMIT 1", {}, [&](const Row &r) {
      out = {{"note", r.as_text(0)}, {"updated_at", r.as_double(1)}};
      return false;
    });
  } catch (const DbError &) {
  }
  return out;
}

void RunDataProvider::set_note(const std::string &text) {
  std::lock_guard<std::mutex> lock(mutex_);
  Db w = Db::open_rw(db_path_);
  w.exec("PRAGMA busy_timeout = 5000");
  w.exec("CREATE TABLE IF NOT EXISTS run_notes (id INTEGER PRIMARY KEY, note TEXT NOT NULL, updated_at REAL NOT NULL)");
  w.run("INSERT INTO run_notes (id, note, updated_at) VALUES (1, ?, ?) ON CONFLICT(id) DO UPDATE SET note = "
        "excluded.note, updated_at = excluded.updated_at",
        {Param::text(text), Param::real(wall_time_now())});
}

}  // namespace sb
