// All /api routes (serenityboard/server/app.py + routes/*.py) on top of App.
#include <cmath>
#include <filesystem>
#include <regex>
#include <sstream>
#include <atomic>
#include <fstream>
#include <unistd.h>

#include "serenityboard/app.hpp"
#include "serenityboard/lora_analytics.hpp"

namespace sb {

namespace fs = std::filesystem;

namespace {

Response error_response(int status, const std::string &message) {
  static const std::map<int, std::string> codes = {{400, "invalid_request"}, {404, "not_found"}, {409, "conflict"},
                                                    {429, "rate_limited"}, {503, "service_unavailable"}};
  auto it = codes.find(status);
  Json body = {{"error", {{"code", it == codes.end() ? "error" : it->second}, {"message", message}, {"details", Json::object()}}}};
  return Response::json(status, body.dump());
}

Response ok_json(const Json &body) { return Response::json(200, body.dump()); }

std::vector<std::string> split_csv(const std::string &text) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const auto b = item.find_first_not_of(" \t"), e = item.find_last_not_of(" \t");
    if (b == std::string::npos) continue;
    out.push_back(item.substr(b, e - b + 1));
  }
  return out;
}

std::optional<long long> int_query(const Request &req, const std::string &key) {
  if (!req.has_query(key)) return std::nullopt;
  try {
    return std::stoll(req.query.at(key));
  } catch (...) {
    return std::nullopt;
  }
}

long long int_query_or(const Request &req, const std::string &key, long long fallback) {
  return int_query(req, key).value_or(fallback);
}

std::optional<std::string> str_query(const Request &req, const std::string &key) {
  if (!req.has_query(key)) return std::nullopt;
  return req.query.at(key);
}

/// Match "/api/runs/{run}/scalars" style patterns; captures into params.
bool match_route(const std::string &pattern, const std::string &path, std::map<std::string, std::string> &params) {
  std::vector<std::string> p_seg, u_seg;
  auto split = [](const std::string &s, std::vector<std::string> &out) {
    std::stringstream ss(s);
    std::string seg;
    while (std::getline(ss, seg, '/')) out.push_back(seg);
  };
  split(pattern, p_seg);
  split(path, u_seg);
  if (p_seg.size() != u_seg.size()) return false;
  params.clear();
  for (std::size_t i = 0; i < p_seg.size(); ++i) {
    if (p_seg[i].size() > 2 && p_seg[i].front() == '{' && p_seg[i].back() == '}') {
      if (u_seg[i].empty()) return false;
      params[p_seg[i].substr(1, p_seg[i].size() - 2)] = u_seg[i];
    } else if (p_seg[i] != u_seg[i]) {
      return false;
    }
  }
  return true;
}

Json rows_with_axis(const std::vector<ScalarRow> &rows, const std::string &x_axis) {
  Json out = Json::array();
  if (x_axis == "wall_time") {
    for (const auto &r : rows) out.push_back({r.wall_time, r.wall_time, r.value});
  } else if (x_axis == "relative") {
    if (rows.empty()) return out;
    const double t0 = rows[0].wall_time;
    for (const auto &r : rows) out.push_back({r.wall_time - t0, r.wall_time, r.value});
  } else {
    for (const auto &r : rows) out.push_back({r.step, r.wall_time, r.value});
  }
  return out;
}

std::string safe_name(const std::string &s) {
  std::string out;
  for (char c : s) out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') ? c : '_');
  return out;
}

// Python's repr() for floats in CSV: shortest round-trip. nlohmann dumps the
// same way for doubles; integers print as integers.
std::string number_text(double v) {
  if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) {
    // Python prints floats that are integral as "1.0"; steps are ints and go through the int path.
    return Json(v).dump();
  }
  return Json(v).dump();
}

Response export_scalars(RunDataProvider &p, const std::string &run, const Request &req) {
  std::vector<std::string> tag_list;
  if (const auto tags = str_query(req, "tags"); tags && !tags->empty()) tag_list = split_csv(*tags);
  else tag_list = p.get_tags()["scalars"].get<std::vector<std::string>>();
  if (tag_list.empty()) return error_response(404, "No scalar tags found");
  const std::string x_axis = req.query_or("x_axis", "step");
  const std::string format = req.query_or("format", "csv");
  std::map<std::string, std::vector<ScalarRow>> tag_data;
  for (const auto &tag : tag_list) tag_data[tag] = p.read_scalars_downsampled(tag, 0);
  std::optional<double> t0;
  if (x_axis == "relative")
    for (const auto &[_, rows] : tag_data)
      if (!rows.empty() && (!t0 || rows[0].wall_time < *t0)) t0 = rows[0].wall_time;
  // step -> (wall_time, {tag: value}); first writer wins for wall_time
  std::map<long long, std::pair<double, std::map<std::string, double>>> step_index;
  for (const auto &tag : tag_list)
    for (const auto &r : tag_data[tag]) {
      auto it = step_index.find(r.step);
      if (it == step_index.end()) it = step_index.emplace(r.step, std::make_pair(r.wall_time, std::map<std::string, double>{})).first;
      it->second.second[tag] = r.value;
    }
  if (format == "json") {
    Json result = Json::array();
    for (const auto &[step, entry] : step_index) {
      const double wt = entry.first;
      Json x = x_axis == "step" ? Json(step) : (x_axis == "time" ? Json(wt) : Json(wt - t0.value_or(0.0)));
      Json values = Json::object();
      for (const auto &[tag, v] : entry.second) values[tag] = v;
      result.push_back({{"step", step}, {"wall_time", wt}, {"x", x}, {"values", values}});
    }
    return ok_json(result);
  }
  std::string csv;
  const std::string x_label = x_axis == "time" ? "wall_time" : (x_axis == "relative" ? "relative_time" : "step");
  csv += x_label + ",wall_time";
  for (const auto &tag : tag_list) csv += "," + tag;
  csv += "\r\n";
  for (const auto &[step, entry] : step_index) {
    const double wt = entry.first;
    if (x_axis == "step") csv += std::to_string(step);
    else if (x_axis == "time") csv += number_text(wt);
    else csv += number_text(wt - t0.value_or(0.0));
    csv += "," + number_text(wt);
    for (const auto &tag : tag_list) {
      auto it = entry.second.find(tag);
      csv += ",";
      if (it != entry.second.end() && !std::isnan(it->second)) csv += number_text(it->second);
    }
    csv += "\r\n";
  }
  std::string safe_tags;
  for (std::size_t i = 0; i < std::min<std::size_t>(3, tag_list.size()); ++i) safe_tags += (i ? "_" : "") + safe_name(tag_list[i]);
  if (tag_list.size() > 3) safe_tags += "_+" + std::to_string(tag_list.size() - 3);
  Response r = Response::text(200, csv, "text/csv; charset=utf-8");
  r.set_header("Content-Disposition", "attachment; filename=\"" + safe_name(run) + "_" + safe_tags + ".csv\"");
  return r;
}

const std::regex kBlobKey("^[a-f0-9]{16}\\.[a-z0-9]+$");

}  // namespace

// ── /api/lora/* (serenityboard/server/routes/lora.py) ──
struct MultipartFile {
  std::string filename, content;
};

std::map<std::string, MultipartFile> parse_multipart(const Request &req) {
  std::map<std::string, MultipartFile> out;
  auto ct = req.headers.find("content-type");
  if (ct == req.headers.end()) return out;
  const auto bpos = ct->second.find("boundary=");
  if (bpos == std::string::npos) return out;
  std::string boundary = ct->second.substr(bpos + 9);
  if (!boundary.empty() && boundary.front() == '"') boundary = boundary.substr(1, boundary.size() - 2);
  const std::string delim = "--" + boundary;
  const std::string &body = req.body;
  std::size_t pos = body.find(delim);
  while (pos != std::string::npos) {
    pos += delim.size();
    if (body.compare(pos, 2, "--") == 0) break;
    if (body.compare(pos, 2, "\r\n") == 0) pos += 2;
    const auto hdr_end = body.find("\r\n\r\n", pos);
    if (hdr_end == std::string::npos) break;
    const std::string headers = body.substr(pos, hdr_end - pos);
    const std::size_t data_start = hdr_end + 4;
    std::size_t next = body.find("\r\n" + delim, data_start);
    if (next == std::string::npos) break;
    MultipartFile part;
    part.content = body.substr(data_start, next - data_start);
    std::string name;
    auto field = [&](const std::string &key) -> std::string {
      const auto k = headers.find(key + "=\"");
      if (k == std::string::npos) return "";
      const auto start = k + key.size() + 2, end = headers.find('"', start);
      return headers.substr(start, end - start);
    };
    name = field("name");
    part.filename = field("filename");
    out[name] = std::move(part);
    pos = next + 2;
  }
  return out;
}

bool ends_with(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct TempFile {
  std::string path;
  explicit TempFile(const std::string &content) {
    static std::atomic<unsigned> counter{0};
    path = (std::filesystem::temp_directory_path() /
            ("sb_lora_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + ".safetensors")).string();
    std::ofstream(path, std::ios::binary) << content;
  }
  ~TempFile() { std::error_code ec; std::filesystem::remove(path, ec); }
};

Response lora_json(const Json &body) { return Response::json(200, sb::lora::dump_json_py(body)); }

Response lora_route(const std::string &path, const Request &req) {
  using namespace sb::lora;
  try {
    if (path == "/api/lora/analyze" || path == "/api/lora/compare") {
      Json body = Json::parse(req.body, nullptr, false);
      if (!body.is_object()) return error_response(422, "Invalid JSON body");
      std::vector<std::string> paths;
      if (path == "/api/lora/analyze") {
        if (!body.contains("path") || !body["path"].is_string()) return error_response(422, "path is required");
        paths.push_back(body["path"].get<std::string>());
      } else {
        for (const char *k : {"path_a", "path_b"}) {
          if (!body.contains(k) || !body[k].is_string()) return error_response(422, std::string(k) + " is required");
          paths.push_back(body[k].get<std::string>());
        }
      }
      for (const auto &p : paths) {
        if (!std::filesystem::is_regular_file(p)) return error_response(404, "File not found: " + p);
        if (!ends_with(p, ".safetensors")) return error_response(400, "Only .safetensors files are supported");
      }
      if (paths.size() == 1) {
        const auto metrics = analyze_file(paths[0]);
        return lora_json({{"layers", metrics}, {"summary", summary_stats(metrics)}, {"diagnostics", diagnose(metrics)},
                          {"num_layers", metrics.size()}});
      }
      const auto m1 = analyze_file(paths[0]), m2 = analyze_file(paths[1]);
      const Json comparison = compare(m1, m2);
      auto diags = diagnose(m1);
      for (auto &d : diagnose(m2)) diags.push_back(d);
      return lora_json({{"layers", comparison}, {"summary_a", summary_stats(m1)}, {"summary_b", summary_stats(m2)},
                        {"diagnostics", diags}, {"num_layers", comparison.size()}});
    }
    if (path == "/api/lora/analyze-upload" || path == "/api/lora/compare-upload") {
      auto parts = parse_multipart(req);
      const bool single = path == "/api/lora/analyze-upload";
      std::vector<std::string> names = single ? std::vector<std::string>{"file"} : std::vector<std::string>{"file_a", "file_b"};
      for (const auto &n : names) {
        if (!parts.count(n)) return error_response(422, "Missing upload field: " + n);
        if (parts[n].filename.empty() || !ends_with(parts[n].filename, ".safetensors"))
          return error_response(400, "Only .safetensors files are supported");
      }
      if (single) {
        TempFile tmp(parts["file"].content);
        const auto metrics = analyze_file(tmp.path);
        return lora_json({{"layers", metrics}, {"summary", summary_stats(metrics)}, {"diagnostics", diagnose(metrics)},
                          {"num_layers", metrics.size()}, {"filename", parts["file"].filename}});
      }
      TempFile ta(parts["file_a"].content), tb(parts["file_b"].content);
      const auto m1 = analyze_file(ta.path), m2 = analyze_file(tb.path);
      const Json comparison = compare(m1, m2);
      auto diags = diagnose(m1);
      for (auto &d : diagnose(m2)) diags.push_back(d);
      return lora_json({{"layers", comparison}, {"summary_a", summary_stats(m1)}, {"summary_b", summary_stats(m2)},
                        {"diagnostics", diags}, {"num_layers", comparison.size()},
                        {"filename_a", parts["file_a"].filename}, {"filename_b", parts["file_b"].filename}});
    }
    return error_response(404, "Not Found");
  } catch (const std::exception &e) {
    return error_response(500, std::string("LoRA analysis failed: ") + e.what());
  }
}

Response App::handle(const Request &req) {
  Response resp = [&]() -> Response {
    std::map<std::string, std::string> pp;
    const std::string &path = req.path;
    const std::string &m = req.method;

    if (path == "/api/runs" && m == "GET") {
      watcher_.scan_once();
      return ok_json(watcher_.get_runs());
    }
    if (path == "/api/plugins" && m == "GET") {
      return ok_json(Json::array({{{"name", "scalars"}, {"display_name", "Scalars"}, {"active", true}},
                                  {{"name", "images"}, {"display_name", "Images"}, {"active", true}},
                                  {{"name", "text"}, {"display_name", "Text"}, {"active", true}},
                                  {{"name", "histograms"}, {"display_name", "Histograms"}, {"active", false}},
                                  {{"name", "hparams"}, {"display_name", "HParams"}, {"active", true}},
                                  {{"name", "traces"}, {"display_name", "Traces"}, {"active", true}},
                                  {{"name", "eval"}, {"display_name", "Eval"}, {"active", true}},
                                  {{"name", "artifacts"}, {"display_name", "Artifacts"}, {"active", true}},
                                  {{"name", "graphs"}, {"display_name", "Graphs"}, {"active", true}},
                                  {{"name", "embeddings"}, {"display_name", "Embeddings"}, {"active", true}},
                                  {{"name", "meshes"}, {"display_name", "Meshes"}, {"active", true}}}));
    }
    if (path == "/api/compare/scalars" && m == "GET") {
      const auto tag = str_query(req, "tag"), runs = str_query(req, "runs");
      if (!tag || !runs) return error_response(422, "tag and runs are required");
      const long long downsample = int_query_or(req, "downsample", 5000);
      const std::string x_axis = req.query_or("x_axis", "step");
      Json result = Json::object();
      for (const auto &name : split_csv(*runs))
        if (auto p = watcher_.get_provider(name)) result[name] = rows_with_axis(p->read_scalars_downsampled(*tag, downsample), x_axis);
      return ok_json(result);
    }
    if (path == "/api/compare/eval" && m == "GET") {
      const auto suite = str_query(req, "suite"), runs = str_query(req, "runs"), score = str_query(req, "score");
      if (!suite || !runs) return error_response(422, "suite and runs are required");
      Json result = Json::object();
      for (const auto &name : split_csv(*runs))
        if (auto p = watcher_.get_provider(name)) {
          Json evals = p->read_eval_results(*suite, std::nullopt);
          if (score && !score->empty()) {
            Json filtered = Json::array();
            for (const auto &e : evals) if (e["score_name"] == *score) filtered.push_back(e);
            evals = filtered;
          }
          result[name] = evals;
        }
      return ok_json(result);
    }
    if (path == "/api/compare/hparams" && m == "GET") {
      const auto runs = str_query(req, "runs");
      if (!runs) return error_response(422, "runs is required");
      Json result = Json::array();
      for (const auto &name : split_csv(*runs))
        if (auto p = watcher_.get_provider(name)) {
          Json hp = p->get_hparams();
          hp["run"] = name;
          result.push_back(hp);
        }
      return ok_json(result);
    }
    if (path.rfind("/api/lora/", 0) == 0 && m == "POST") return lora_route(path, req);

    // ── run-scoped ──
    if (match_route("/api/runs/{run}", path, pp) && m == "DELETE") {
      if (watcher_.delete_run(pp["run"])) return ok_json({{"deleted", pp["run"]}});
      return error_response(404, "Run not found");
    }
    static const char *run_routes[] = {"/api/runs/{run}/tags", "/api/runs/{run}/scalars", "/api/runs/{run}/scalars/last",
                                       "/api/runs/{run}/export", "/api/runs/{run}/images", "/api/runs/{run}/blob/{blob_key}",
                                       "/api/runs/{run}/text", "/api/runs/{run}/hparams", "/api/runs/{run}/histograms",
                                       "/api/runs/{run}/distributions", "/api/runs/{run}/traces", "/api/runs/{run}/eval",
                                       "/api/runs/{run}/artifacts", "/api/runs/{run}/metrics", "/api/runs/{run}/metrics/timeseries",
                                       "/api/runs/{run}/custom-scalars/layout", "/api/runs/{run}/custom-scalars/data",
                                       "/api/runs/{run}/pr-curves", "/api/runs/{run}/audio", "/api/runs/{run}/graphs",
                                       "/api/runs/{run}/meshes", "/api/runs/{run}/embeddings", "/api/runs/{run}/notes"};
    std::string route;
    for (const char *r : run_routes)
      if (match_route(r, path, pp)) { route = r; break; }
    if (route.empty()) {
      if (path.rfind("/api/", 0) == 0) return error_response(404, "Not Found");
      return serve_static(req);
    }
    const std::string sub = route.substr(std::string("/api/runs/{run}").size());
    if (sub == "/blob/{blob_key}" && !std::regex_match(pp["blob_key"], kBlobKey))
      return error_response(400, "Invalid blob key format");
    auto p = watcher_.get_provider(pp["run"]);
    if (!p) return error_response(404, "Run not found");
    auto require = [&](const char *key) -> std::optional<Response> {
      if (!req.has_query(key)) return error_response(422, std::string("missing query parameter: ") + key);
      return std::nullopt;
    };

    if (sub == "/tags" && m == "GET") return ok_json(p->get_tags());
    if (sub == "/scalars" && m == "GET") {
      if (auto e = require("tag")) return *e;
      return ok_json(rows_with_axis(p->read_scalars_downsampled(req.query.at("tag"), int_query_or(req, "downsample", 5000)),
                                    req.query_or("x_axis", "step")));
    }
    if (sub == "/scalars/last" && m == "GET") {
      if (auto e = require("tags")) return *e;
      return ok_json(p->read_scalars_last(split_csv(req.query.at("tags"))));
    }
    if (sub == "/export" && m == "GET") return export_scalars(*p, pp["run"], req);
    if (sub == "/images" && m == "GET") {
      if (auto e = require("tag")) return *e;
      return ok_json(p->read_images(req.query.at("tag"), int_query_or(req, "downsample", 100)));
    }
    if (sub == "/blob/{blob_key}" && m == "GET") {
      const auto mime = p->get_blob_mime(pp["blob_key"]);
      if (!mime) return error_response(404, "Blob not found");
      const fs::path blob = fs::path(p->run_dir()) / "blobs" / pp["blob_key"];
      std::error_code ec;
      if (!fs::is_regular_file(blob, ec)) return error_response(404, "Blob file missing");
      return Response::file(blob.string(), *mime);
    }
    if (sub == "/text" && m == "GET") {
      if (auto e = require("tag")) return *e;
      return ok_json(p->read_text(req.query.at("tag"), int_query(req, "limit")));
    }
    if (sub == "/hparams" && m == "GET") return ok_json(p->get_hparams());
    if (sub == "/histograms" && m == "GET") {
      if (auto e = require("tag")) return *e;
      return ok_json(p->read_histograms(req.query.at("tag"), int_query_or(req, "downsample", 100)));
    }
    if (sub == "/distributions" && m == "GET") {
      if (auto e = require("tag")) return *e;
      return ok_json(p->read_distributions(req.query.at("tag"), int_query_or(req, "downsample", 100)));
    }
    if (sub == "/traces" && m == "GET") return ok_json(p->read_trace_events(int_query(req, "step_from"), int_query(req, "step_to")));
    if (sub == "/eval" && m == "GET") {
      if (auto e = require("suite")) return *e;
      return ok_json(p->read_eval_results(req.query.at("suite"), int_query(req, "step")));
    }
    if (sub == "/artifacts" && m == "GET") {
      if (auto e = require("tag")) return *e;
      return ok_json(p->read_artifacts(req.query.at("tag"), int_query_or(req, "downsample", 100), str_query(req, "kind")));
    }
    if (sub == "/metrics" && m == "GET") return ok_json(p->get_all_metric_tags());
    if (sub == "/metrics/timeseries" && m == "POST") {
      Json body;
      try {
        body = Json::parse(req.body);
      } catch (...) {
        return error_response(400, "invalid JSON body");
      }
      const Json requests = body.is_object() && body.contains("requests") ? body["requests"] : Json::array();
      if (!requests.is_array() || requests.empty()) return ok_json(Json::array());
      return ok_json(p->read_metric_timeseries(requests));
    }
    if (sub == "/custom-scalars/layout" && m == "GET") {
      const auto layout = p->get_custom_scalar_layout();
      return ok_json(layout ? *layout : Json{{"categories", Json::array()}});
    }
    if (sub == "/custom-scalars/data" && m == "GET") {
      if (auto e = require("tags")) return *e;
      const auto regexes = split_csv(req.query.at("tags"));
      if (regexes.empty()) return ok_json(Json::object());
      return ok_json(p->read_custom_scalars(regexes, int_query_or(req, "downsample", 5000)));
    }
    if (sub == "/pr-curves" && m == "GET") {
      if (auto e = require("tag")) return *e;
      return ok_json(p->read_pr_curves(req.query.at("tag"), int_query_or(req, "downsample", 50)));
    }
    if (sub == "/audio" && m == "GET") {
      if (auto e = require("tag")) return *e;
      return ok_json(p->read_audio(req.query.at("tag"), int_query_or(req, "downsample", 50)));
    }
    if (sub == "/graphs" && m == "GET") return ok_json(p->read_graphs(str_query(req, "tag"), int_query_or(req, "downsample", 10)));
    if (sub == "/meshes" && m == "GET") {
      const auto tag = str_query(req, "tag");
      const auto step = int_query(req, "step");
      Json results = p->read_meshes(tag, step, int_query_or(req, "downsample", 50));
      if (tag && !tag->empty() && step) {
        if (results.empty()) return error_response(404, "Mesh not found for this tag and step");
        return ok_json(results[0]);
      }
      return ok_json(results);
    }
    if (sub == "/embeddings" && m == "GET") {
      const auto tag = str_query(req, "tag");
      const auto step = int_query(req, "step");
      Json results = p->read_embeddings(tag, step, int_query_or(req, "downsample", 20));
      if (tag && !tag->empty() && step) {
        if (results.empty()) return error_response(404, "Embedding not found");
        return ok_json(results[0]);
      }
      return ok_json(results);
    }
    if (sub == "/notes" && m == "GET") return ok_json(p->get_note());
    if (sub == "/notes" && m == "PUT") {
      Json body;
      try {
        body = Json::parse(req.body);
      } catch (...) {
        return error_response(422, "invalid JSON body");
      }
      if (!body.is_object() || !body.contains("note") || !body["note"].is_string()) return error_response(422, "note is required");
      p->set_note(body["note"].get<std::string>());
      return ok_json({{"ok", true}});
    }
    return error_response(405, "Method Not Allowed");
  }();

  // app.py no-cache middleware
  const std::string &path = req.path;
  auto ends_with = [&](const char *suffix) {
    const std::string s(suffix);
    return path.size() >= s.size() && path.compare(path.size() - s.size(), s.size(), s) == 0;
  };
  if (path == "/" || ends_with(".js") || ends_with(".css") || ends_with(".html")) {
    resp.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    resp.set_header("Pragma", "no-cache");
    resp.set_header("Expires", "0");
  }
  return resp;
}

}  // namespace sb
