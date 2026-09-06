#include "serenityboard/compiler_ingest.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace sb::ingest {

namespace {

bool parse_number(const std::string &text, double &out) {
  if (text.empty()) return false;
  char *end = nullptr;
  out = std::strtod(text.c_str(), &end);
  return end && *end == '\0' && std::isfinite(out);
}

std::string strip_quotes(std::string s) {
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') return s.substr(1, s.size() - 2);
  return s;
}

/// Flatten numeric leaves of a JSON object into "a/b/c" -> value scalars at `step`.
void flatten_numeric(SummaryWriter &w, const Json &node, const std::string &prefix, long long step, Counts &c,
                     int depth = 0) {
  if (depth > 6) return;
  if (node.is_object()) {
    for (const auto &[k, v] : node.items()) flatten_numeric(w, v, prefix.empty() ? k : prefix + "/" + k, step, c, depth + 1);
  } else if (node.is_number() && !node.is_boolean()) {
    w.add_scalar(prefix, node.get<double>(), step);
    ++c.scalars;
  }
}

Json strip_large(const Json &node, std::size_t max_array = 64) {
  if (node.is_array()) {
    if (node.size() > max_array) return Json("[" + std::to_string(node.size()) + " items]");
    Json out = Json::array();
    for (const auto &v : node) out.push_back(strip_large(v, max_array));
    return out;
  }
  if (node.is_object()) {
    Json out = Json::object();
    for (const auto &[k, v] : node.items()) out[k] = strip_large(v, max_array);
    return out;
  }
  return node;
}

}  // namespace

std::map<std::string, std::string> parse_kv_line(const std::string &line) {
  std::map<std::string, std::string> out;
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && line[i] == ' ') ++i;
    const std::size_t eq = line.find('=', i);
    if (eq == std::string::npos) break;
    const std::size_t key_start = line.rfind(' ', eq);
    const std::string key = line.substr(key_start == std::string::npos ? 0 : key_start + 1, eq - (key_start == std::string::npos ? 0 : key_start + 1));
    std::size_t j = eq + 1;
    std::string value;
    if (j < line.size() && line[j] == '"') {
      const std::size_t close = line.find('"', j + 1);
      value = line.substr(j, close == std::string::npos ? std::string::npos : close - j + 1);
      j = close == std::string::npos ? line.size() : close + 1;
    } else {
      const std::size_t sp = line.find(' ', j);
      value = line.substr(j, sp == std::string::npos ? std::string::npos : sp - j);
      j = sp == std::string::npos ? line.size() : sp;
    }
    if (!key.empty() && key.find(' ') == std::string::npos) out[key] = strip_quotes(value);
    i = j;
  }
  return out;
}

Counts ingest_step_lines(SummaryWriter &w, std::istream &in) {
  Counts c;
  std::string line;
  long long profile_index = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const std::string head = line.substr(0, line.find(' '));
    const auto kv = parse_kv_line(line);
    auto num = [&](const char *key, double &v) {
      auto it = kv.find(key);
      return it != kv.end() && parse_number(it->second, v);
    };
    double v = 0;
    if (head == "H3_STEP") {
      double idx;
      if (!num("index", idx)) continue;
      const long long step = static_cast<long long>(idx) + 1;
      if (num("denoiser_ms", v)) { w.add_scalar("h3/denoiser_ms", v, step); w.add_trace(step, "denoiser", v, Json{{"text_refiner_cache", kv.count("text_refiner_cache") ? kv.at("text_refiner_cache") : ""}}); ++c.scalars; ++c.traces; }
      if (num("video_t", v)) { w.add_scalar("h3/video_t", v, step); ++c.scalars; }
      if (num("audio_t", v)) { w.add_scalar("h3/audio_t", v, step); ++c.scalars; }
    } else if (head == "FLUX2_NATIVE_STEP" || head == "KREA2_NATIVE_STEP") {
      auto it = kv.find("step");
      if (it == kv.end()) continue;
      const long long step = std::stoll(it->second.substr(0, it->second.find('/')));
      const std::string prefix = head == "FLUX2_NATIVE_STEP" ? "flux2/" : "krea2/";
      for (const auto &[key, value] : kv) {
        if (key == "step" || !parse_number(value, v)) continue;
        w.add_scalar(prefix + key, v, step);
        ++c.scalars;
        if (key.size() > 3 && key.compare(key.size() - 3, 3, "_ms") == 0) {
          w.add_trace(step, key.substr(0, key.size() - 3), v);
          ++c.traces;
        }
      }
    } else if (head == "KREA2_STEP") {
      // KREA2_STEP <n>/<max> loss= ms= state_h2d= state_d2h=
      std::istringstream ss(line);
      std::string tok, frac;
      ss >> tok >> frac;
      const long long step = std::stoll(frac.substr(0, frac.find('/')));
      for (const auto &[key, value] : kv) {
        if (!parse_number(value, v)) continue;
        w.add_scalar(key == "loss" ? "loss/train" : "train/" + key, v, step);
        ++c.scalars;
      }
    } else if (head == "H3_PROFILE") {
      ++profile_index;
      for (const auto &[key, value] : kv)
        if (parse_number(value, v)) { w.add_scalar("h3_profile/" + key, v, profile_index); ++c.scalars; }
    } else if (head.size() > 5 && (head.compare(head.size() - 5, 5, "_PASS") == 0 || line.find(" PASS ") != std::string::npos)) {
      w.add_text("receipts/" + head, line, 0);
      ++c.texts;
    }
  }
  return c;
}

Counts ingest_report_json(SummaryWriter &w, const std::string &json_text, const std::string &source_name) {
  Counts c;
  Json doc = Json::parse(json_text);
  const std::string kind = doc.contains("kind") && doc["kind"].is_string() ? doc["kind"].get<std::string>() : "";
  Json hparams = Json::object();

  if (kind == "benchmark") {
    // difbench: per-run stage walls -> trace events (step = run index), GPU peaks -> scalars
    const Json &bench = doc["benchmark"];
    for (const char *k : {"label", "host", "boundary", "acceptance_metric", "recipe", "workload", "comparator"})
      if (bench.contains(k)) hparams[k] = strip_large(bench[k]);
    if (doc.contains("hardware")) hparams["hardware"] = strip_large(doc["hardware"]);
    for (const auto &run : doc.value("runs", Json::array())) {
      const long long step = run.value("index", 0LL);
      if (run.contains("wall_seconds") && run["wall_seconds"].is_number()) { w.add_scalar("bench/complete_wall_seconds", run["wall_seconds"].get<double>(), step); ++c.scalars; }
      for (const auto &stage : run.value("stages", Json::array())) {
        if (!stage.contains("wall_seconds") || !stage["wall_seconds"].is_number()) continue;
        const std::string name = stage.value("name", "stage");
        w.add_trace(step, name, stage["wall_seconds"].get<double>() * 1000.0,
                    Json{{"exit_status", stage.value("exit_status", Json(nullptr))},
                         {"max_rss_kib", stage.value("max_rss_kib", Json(nullptr))},
                         {"cache_status", stage.value("cache_status", Json(nullptr))},
                         {"user_cpu_seconds", stage.value("user_cpu_seconds", Json(nullptr))}});
        ++c.traces;
        w.add_scalar("bench/stage_wall_seconds/" + name, stage["wall_seconds"].get<double>(), step);
        ++c.scalars;
      }
      if (run.contains("gpu") && run["gpu"].is_object()) {
        for (const char *k : {"peak_used_memory_bytes", "mean_power_watts", "max_power_watts", "max_temperature_celsius"})
          if (run["gpu"].contains(k) && run["gpu"][k].is_number()) { w.add_scalar(std::string("bench/gpu/") + k, run["gpu"][k].get<double>(), step); ++c.scalars; }
      }
    }
    if (doc.contains("summary")) {
      w.add_text("bench/summary", doc["summary"].dump(2), 0);
      ++c.texts;
      const Json &s = doc["summary"];
      if (s.contains("complete_wall_seconds") && s["complete_wall_seconds"].is_object())
        for (const char *k : {"minimum", "median", "maximum"})
          if (s["complete_wall_seconds"].contains(k)) { w.add_scalar(std::string("bench/summary/") + k + "_wall_seconds", s["complete_wall_seconds"][k].get<double>(), 0); ++c.scalars; }
    }
    w.add_text("bench/status", doc.value("status", ""), 0);
    ++c.texts;
  } else if (kind == "runtime-trace" || kind == "trace") {
    flatten_numeric(w, doc.value("execution", Json::object()), "trace/execution", 0, c);
    flatten_numeric(w, doc.value("pipeline_profile", Json::object()), "trace/pipeline", 0, c);
    flatten_numeric(w, doc.value("run_launch_telemetry", Json::object()), "trace/run_launch", 0, c);
    if (doc.contains("trace") && doc["trace"].is_object()) {
      for (const char *section : {"preparation_attribution", "run_attribution"})
        if (doc["trace"].contains(section))
          for (const auto &[cat, v] : doc["trace"][section].items())
            if (v.is_object() && v.contains("host_ms")) { w.add_trace(0, std::string(section) + "/" + cat, v["host_ms"].get<double>(), strip_large(v)); ++c.traces; }
    }
    if (doc.contains("hardware")) hparams["hardware"] = strip_large(doc["hardware"]);
    if (doc.contains("program")) hparams["program"] = doc["program"];
  } else if (kind == "regression-report") {
    for (const auto &check : doc.value("checks", Json::array())) {
      const std::string name = check.value("name", "check");
      w.add_eval("regression", name, 0, "verdict_pass", check.value("verdict", "") == "PASS" ? 1.0 : 0.0, std::nullopt, strip_large(check));
      ++c.evals;
      if (check.contains("performance") && check["performance"].is_object() && check["performance"].contains("median") && check["performance"]["median"].is_number()) {
        w.add_scalar("regression/median/" + name, check["performance"]["median"].get<double>(), 0);
        ++c.scalars;
      }
    }
    if (doc.contains("summary")) { w.add_text("regression/summary", doc["summary"].dump(2), 0); ++c.texts; }
  } else if (kind == "device-probe") {
    hparams["hardware"] = strip_large(doc.value("hardware", Json::object()));
    flatten_numeric(w, doc.value("runtime_budget", Json::object()), "probe/runtime_budget", 0, c);
  } else if (doc.contains("steps_detail") || doc.contains("timing_ms") || doc.contains("model_family")) {
    // difkrea2sample / difflux2sample hand-rolled reports
    if (doc.contains("steps_detail") && doc["steps_detail"].is_array()) {
      for (const auto &s : doc["steps_detail"]) {
        const long long step = s.value("step", 0LL);
        for (const auto &[k, v] : s.items()) {
          if (k == "step" || !v.is_number()) continue;
          w.add_scalar("krea2/" + k, v.get<double>(), step);
          ++c.scalars;
          if (k.size() > 3 && k.compare(k.size() - 3, 3, "_ms") == 0) { w.add_trace(step, k.substr(0, k.size() - 3), v.get<double>()); ++c.traces; }
        }
      }
    }
    if (doc.contains("profile_pipeline") && doc["profile_pipeline"].is_object() && doc["profile_pipeline"].contains("steps"))
      for (const auto &s : doc["profile_pipeline"]["steps"]) {
        const long long step = s.value("step", 0LL);
        for (const auto &[k, v] : s.items())
          if (v.is_number() && k != "step") { w.add_scalar("krea2/pipeline/" + k, v.get<double>(), step); ++c.scalars; }
      }
    if (doc.contains("timing_ms") && doc["timing_ms"].is_object()) {
      for (const auto &[k, v] : doc["timing_ms"].items())
        if (v.is_number()) { w.add_scalar("flux2/timing_ms/" + k, v.get<double>(), 0); w.add_trace(0, k, v.get<double>()); ++c.scalars; ++c.traces; }
    }
    if (doc.contains("host_phases_ms") && doc["host_phases_ms"].is_object())
      for (const auto &[k, v] : doc["host_phases_ms"].items())
        if (v.is_number()) { w.add_trace(0, "host/" + k, v.get<double>()); ++c.traces; }
    for (const char *k : {"model_family", "prompt", "seed", "steps", "guidance", "scheduler", "width", "height", "device",
                          "transformer_checkpoint", "checkpoint", "mu", "dtype", "geometry", "conditioning_mode",
                          "creator_commit", "source_commit", "model_revision", "parity_reference", "initial_seed"})
      if (doc.contains(k) && !doc[k].is_object() && !doc[k].is_array()) hparams[k] = doc[k];
    for (const char *k : {"residency_plan", "streamed_weight_plan", "w8a8_candidate", "int8_weight_only_candidate", "resident_plan"})
      if (doc.contains(k)) hparams[k] = strip_large(doc[k]);
    for (const char *k : {"wall_ms", "denoiser_preparation_ms", "peak_prepared_resident_bytes", "denoiser_resident_bytes"})
      if (doc.contains(k) && doc[k].is_number()) { w.add_scalar(std::string("report/") + k, doc[k].get<double>(), 0); ++c.scalars; }
  } else {
    w.add_text("report/" + source_name, json_text.substr(0, 65536), 0);
    ++c.texts;
    return c;
  }
  if (doc.contains("provenance")) hparams["provenance"] = doc["provenance"];
  if (!hparams.empty()) w.add_hparams(hparams, {});
  w.add_text("report/source", source_name + (kind.empty() ? "" : " (" + kind + ")"), 0);
  ++c.texts;
  return c;
}

Counts ingest_jsonl(SummaryWriter &w, std::istream &in, const std::string &source_name) {
  Counts c;
  std::string line;
  long long index = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Json doc;
    try {
      doc = Json::parse(line);
    } catch (...) {
      continue;  // diftrace convention: skip malformed lines
    }
    if (!doc.is_object()) continue;
    if (doc.contains("kind")) {
      // runtime-trace sink: one document per prepared execution; step = document index
      ++index;
      if (doc.contains("execution") && doc["execution"].is_object()) {
        const Json &e = doc["execution"];
        for (const char *k : {"preparation_ms", "mean_ms", "min_ms", "max_ms", "resident_bytes", "free_bytes_before", "free_bytes_after"})
          if (e.contains(k) && e[k].is_number()) {
            // preparation_reported: only count preparation on the run that reported it
            if (std::string(k) == "preparation_ms" && e.contains("preparation_reported") && e["preparation_reported"].is_boolean() && !e["preparation_reported"].get<bool>()) continue;
            w.add_scalar(std::string("trace/") + k, e[k].get<double>(), index);
            ++c.scalars;
          }
      }
      if (doc.contains("trace") && doc["trace"].is_object() && doc["trace"].contains("run_attribution"))
        for (const auto &[cat, v] : doc["trace"]["run_attribution"].items())
          if (v.is_object() && v.contains("host_ms") && v["host_ms"].is_number()) { w.add_trace(index, cat, v["host_ms"].get<double>(), Json{{"count", v.value("count", 0)}, {"bytes", v.value("bytes", 0)}}); ++c.traces; }
      continue;
    }
    if (doc.contains("completed_steps") && doc.contains("loss")) {
      // TrainingStepReport (exp branch): one record per step
      const long long step = doc["completed_steps"].get<long long>();
      w.add_scalar("loss/train", doc["loss"].get<double>(), step);
      ++c.scalars;
      for (const char *k : {"grad_norm", "step_milliseconds", "nonfinites"})
        if (doc.contains(k) && doc[k].is_number()) { w.add_scalar(std::string("train/") + k, doc[k].get<double>(), step); ++c.scalars; }
      if (doc.contains("phases") && doc["phases"].is_object())
        for (const auto &[k, v] : doc["phases"].items())
          if (v.is_number()) {
            const std::string phase = k.size() > 13 && k.compare(k.size() - 13, 13, "_milliseconds") == 0 ? k.substr(0, k.size() - 13) : k;
            w.add_trace(step, phase, v.get<double>());
            ++c.traces;
          }
      if (doc.contains("memory") && doc["memory"].is_object())
        for (const auto &[k, v] : doc["memory"].items())
          if (v.is_number()) { w.add_scalar("memory/" + k, v.get<double>(), step); ++c.scalars; }
      if (doc.contains("submitted") && doc["submitted"].is_object())
        for (const auto &[k, v] : doc["submitted"].items())
          if (v.is_number()) { w.add_scalar("submitted/" + k, v.get<double>(), step); ++c.scalars; }
      if (index++ == 0) {
        Json hparams = Json::object();
        for (const char *k : {"model", "backend", "device", "target", "runtime_budget", "plan_fingerprint", "checkpoint",
                              "trainable_tensors", "trainable_parameters", "parameter_policy", "physical_formats"})
          if (doc.contains(k)) hparams[k] = doc[k];
        if (!hparams.empty()) w.add_hparams(hparams, {});
        w.add_text("report/source", source_name + " (training-report)", 0);
        ++c.texts;
      }
      continue;
    }
  }
  return c;
}

Counts ingest_losses_diftensor(SummaryWriter &w, const std::string &path, const std::string &tag) {
  Counts c;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (bytes.size() < 20 || bytes.compare(0, 8, "DIFTNS01") != 0) throw std::runtime_error("not a DIFTNS01 tensor: " + path);
  auto u32 = [&](std::size_t o) { std::uint32_t v; std::memcpy(&v, bytes.data() + o, 4); return v; };
  auto u64 = [&](std::size_t o) { std::uint64_t v; std::memcpy(&v, bytes.data() + o, 8); return v; };
  const std::uint32_t dtype = u32(12), rank = u32(16);
  if (dtype != 0) throw std::runtime_error("losses tensor must be F32 (dtype 0), got " + std::to_string(dtype));
  std::size_t off = 20;
  std::uint64_t count = 1;
  for (std::uint32_t i = 0; i < rank; ++i, off += 8) count *= u64(off);
  const std::uint64_t byte_count = u64(off);
  off += 8;
  if (byte_count != count * 4 || off + byte_count > bytes.size()) throw std::runtime_error("truncated tensor: " + path);
  for (std::uint64_t i = 0; i < count; ++i) {
    float v;
    std::memcpy(&v, bytes.data() + off + i * 4, 4);
    w.add_scalar(tag, static_cast<double>(v), static_cast<long long>(i));
    ++c.scalars;
  }
  return c;
}

}  // namespace sb::ingest
