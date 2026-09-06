#include "serenityboard/serenityboard.h"

#include <cstring>
#include <map>
#include <string>

#include "serenityboard/summary_writer.hpp"

namespace {
thread_local std::string g_last_error;

template <typename F>
int guarded(F &&f) {
  try {
    f();
    return 0;
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return -1;
  }
}
}  // namespace

struct sb_writer {
  sb::SummaryWriter writer;
  sb_writer(const std::string &logdir, sb::WriterOptions options) : writer(logdir, std::move(options)) {}
};

extern "C" {

sb_writer *sb_writer_open(const char *logdir, const char *run_name, const char *hparams_json, long long resume_step,
                          int system_metrics) {
  try {
    sb::WriterOptions options;
    if (run_name) options.run_name = run_name;
    if (hparams_json && *hparams_json) options.hparams = nlohmann::json::parse(hparams_json);
    if (resume_step >= 0) options.resume_step = resume_step;
    options.system_metrics = system_metrics != 0;
    return new sb_writer(logdir ? logdir : ".", std::move(options));
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return nullptr;
  }
}

const char *sb_last_error(void) { return g_last_error.c_str(); }

int sb_add_scalar(sb_writer *w, const char *tag, double value, long long step) {
  return guarded([&] { w->writer.add_scalar(tag, value, step); });
}
int sb_add_text(sb_writer *w, const char *tag, const char *text, long long step) {
  return guarded([&] { w->writer.add_text(tag, text ? text : "", step); });
}
int sb_add_histogram(sb_writer *w, const char *tag, const double *values, size_t count, long long step, int bins) {
  return guarded([&] { w->writer.add_histogram(tag, std::vector<double>(values, values + count), step, bins); });
}
int sb_add_image(sb_writer *w, const char *tag, uint32_t width, uint32_t height, uint32_t channels,
                 const uint8_t *pixels, long long step) {
  return guarded([&] { w->writer.add_image(tag, width, height, channels, pixels, step); });
}
int sb_add_image_file(sb_writer *w, const char *tag, const char *png_path, long long step) {
  return guarded([&] { w->writer.add_image_file(tag, png_path, step); });
}
int sb_add_trace(sb_writer *w, long long step, const char *phase, double duration_ms, const char *details_json) {
  return guarded([&] {
    std::optional<nlohmann::json> details;
    if (details_json && *details_json) details = nlohmann::json::parse(details_json);
    w->writer.add_trace(step, phase, duration_ms, details);
  });
}
int sb_add_eval(sb_writer *w, const char *suite, const char *case_id, long long step, const char *score_name,
                double score_value, const char *artifact_key, const char *details_json) {
  return guarded([&] {
    std::optional<std::string> artifact;
    if (artifact_key && *artifact_key) artifact = artifact_key;
    std::optional<nlohmann::json> details;
    if (details_json && *details_json) details = nlohmann::json::parse(details_json);
    w->writer.add_eval(suite, case_id, step, score_name, score_value, artifact, details);
  });
}
int sb_add_hparams(sb_writer *w, const char *hparams_json, const char *metrics_json) {
  return guarded([&] {
    std::map<std::string, double> metrics;
    if (metrics_json && *metrics_json) {
      const nlohmann::json parsed = nlohmann::json::parse(metrics_json);
      for (const auto &[k, v] : parsed.items()) metrics[k] = v.get<double>();
    }
    w->writer.add_hparams(hparams_json && *hparams_json ? nlohmann::json::parse(hparams_json) : nlohmann::json::object(),
                          metrics);
  });
}
int sb_add_audio_pcm16(sb_writer *w, const char *tag, const int16_t *samples, size_t count, uint32_t channels,
                       uint32_t sample_rate, long long step) {
  return guarded([&] { w->writer.add_audio(tag, std::vector<int16_t>(samples, samples + count), channels, sample_rate, step); });
}
int sb_flush(sb_writer *w) { return guarded([&] { w->writer.flush(); }); }
int sb_writer_close(sb_writer *w) {
  const int rc = guarded([&] { w->writer.close(); });
  delete w;
  return rc;
}

}  // extern "C"
