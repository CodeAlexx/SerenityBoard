#include "serenityboard/lora_analytics.hpp"

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace sb::lora {

namespace {

float bf16_to_f32(std::uint16_t v) {
  std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1U, exp = (h >> 10) & 0x1FU, mant = h & 0x3FFU;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) bits = sign << 31;
    else {
      // subnormal
      float f = static_cast<float>(mant) / 1024.0f * std::ldexp(1.0f, -14);
      if (sign) f = -f;
      return f;
    }
  } else if (exp == 31) {
    bits = (sign << 31) | 0x7F800000U | (mant << 13);
  } else {
    bits = (sign << 31) | ((exp + 112U) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Symmetric eigen-decomposition: Householder tridiagonalisation followed by the
// implicit QL algorithm (the classic EISPACK tred2/tql2 formulation). a is n x n
// row-major and is overwritten with the eigenvectors (columns) when want_vectors.
// Returns eigenvalues in ascending order; vectors[k*n + j] is component k of vector j.
std::vector<double> sym_eig(std::vector<double> &V, std::size_t n, bool want_vectors) {
  std::vector<double> d(n), e(n);
  auto at = [&](std::size_t i, std::size_t j) -> double & { return V[i * n + j]; };
  for (std::size_t j = 0; j < n; ++j) d[j] = at(n - 1, j);
  for (std::size_t i = n - 1; i > 0; --i) {
    double scale = 0, h = 0;
    for (std::size_t k = 0; k < i; ++k) scale += std::fabs(d[k]);
    if (scale == 0.0) {
      e[i] = d[i - 1];
      for (std::size_t j = 0; j < i; ++j) { d[j] = at(i - 1, j); at(i, j) = 0; at(j, i) = 0; }
    } else {
      for (std::size_t k = 0; k < i; ++k) { d[k] /= scale; h += d[k] * d[k]; }
      double f = d[i - 1], g = std::sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (std::size_t j = 0; j < i; ++j) e[j] = 0;
      for (std::size_t j = 0; j < i; ++j) {
        f = d[j];
        at(j, i) = f;
        g = e[j] + at(j, j) * f;
        for (std::size_t k = j + 1; k < i; ++k) { g += at(k, j) * d[k]; e[k] += at(k, j) * f; }
        e[j] = g;
      }
      f = 0;
      for (std::size_t j = 0; j < i; ++j) { e[j] /= h; f += e[j] * d[j]; }
      const double hh = f / (h + h);
      for (std::size_t j = 0; j < i; ++j) e[j] -= hh * d[j];
      for (std::size_t j = 0; j < i; ++j) {
        f = d[j];
        g = e[j];
        for (std::size_t k = j; k < i; ++k) at(k, j) -= (f * e[k] + g * d[k]);
        d[j] = at(i - 1, j);
        at(i, j) = 0;
      }
    }
    d[i] = h;
  }
  // accumulate transformations
  for (std::size_t i = 0; i + 1 < n; ++i) {
    at(n - 1, i) = at(i, i);
    at(i, i) = 1;
    const double h = d[i + 1];
    if (h != 0.0) {
      for (std::size_t k = 0; k <= i; ++k) d[k] = at(k, i + 1) / h;
      for (std::size_t j = 0; j <= i; ++j) {
        double g = 0;
        for (std::size_t k = 0; k <= i; ++k) g += at(k, i + 1) * at(k, j);
        for (std::size_t k = 0; k <= i; ++k) at(k, j) -= g * d[k];
      }
    }
    for (std::size_t k = 0; k <= i; ++k) at(k, i + 1) = 0;
  }
  for (std::size_t j = 0; j < n; ++j) { d[j] = at(n - 1, j); at(n - 1, j) = 0; }
  at(n - 1, n - 1) = 1;
  e[0] = 0;
  // implicit QL
  for (std::size_t i = 1; i < n; ++i) e[i - 1] = e[i];
  e[n - 1] = 0;
  double f = 0, tst1 = 0;
  const double eps = std::ldexp(1.0, -52);
  for (std::size_t l = 0; l < n; ++l) {
    tst1 = std::max(tst1, std::fabs(d[l]) + std::fabs(e[l]));
    std::size_t m = l;
    while (m < n) {
      if (std::fabs(e[m]) <= eps * tst1) break;
      ++m;
    }
    if (m > l) {
      int iter = 0;
      do {
        if (++iter > 200) break;
        double g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = std::hypot(p, 1.0);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const double dl1 = d[l + 1];
        double h = g - d[l];
        for (std::size_t i = l + 2; i < n; ++i) d[i] -= h;
        f += h;
        p = d[m];
        double c = 1, c2 = c, c3 = c, el1 = e[l + 1], s = 0, s2 = 0;
        for (std::size_t i = m; i-- > l;) {
          c3 = c2; c2 = c; s2 = s;
          g = c * e[i];
          h = c * p;
          r = std::hypot(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          if (want_vectors)
            for (std::size_t k = 0; k < n; ++k) {
              h = at(k, i + 1);
              at(k, i + 1) = s * at(k, i) + c * h;
              at(k, i) = c * at(k, i) - s * h;
            }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (std::fabs(e[l]) > eps * tst1);
    }
    d[l] += f;
    e[l] = 0;
  }
  return d;
}

std::vector<double> sym_eigenvalues(std::vector<double> m, std::size_t n) {
  if (n == 0) return {};
  auto v = sym_eig(m, n, false);
  std::sort(v.begin(), v.end(), std::greater<double>());
  return v;
}

// Gram of the smaller side: M Mᵀ (rows x rows) if rows <= cols else Mᵀ M.
std::vector<double> gram_small(const Matrix &m, std::size_t &n) {
  if (m.rows <= m.cols) {
    n = m.rows;
    std::vector<double> g(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i; j < n; ++j) {
        double s = 0;
        const float *ri = &m.data[i * m.cols], *rj = &m.data[j * m.cols];
        for (std::size_t k = 0; k < m.cols; ++k) s += static_cast<double>(ri[k]) * rj[k];
        g[i * n + j] = g[j * n + i] = s;
      }
    return g;
  }
  n = m.cols;
  std::vector<double> g(n * n, 0.0);
  for (std::size_t r = 0; r < m.rows; ++r) {
    const float *row = &m.data[r * m.cols];
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i; j < n; ++j) g[i * n + j] += static_cast<double>(row[i]) * row[j];
  }
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) g[j * n + i] = g[i * n + j];
  return g;
}

}  // namespace

std::vector<double> singular_values(const Matrix &m) {
  std::size_t n;
  auto g = gram_small(m, n);
  auto eig = sym_eigenvalues(std::move(g), n);
  for (auto &e : eig) e = std::sqrt(std::max(0.0, e));
  return eig;
}

std::map<std::string, Matrix> read_safetensors_matrices(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::uint64_t header_len = 0;
  in.read(reinterpret_cast<char *>(&header_len), 8);
  if (!in || header_len == 0 || header_len > (256u << 20)) throw std::runtime_error("invalid safetensors header in " + path);
  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  const nlohmann::json meta = nlohmann::json::parse(header);
  const std::streamoff base = static_cast<std::streamoff>(8 + header_len);
  std::map<std::string, Matrix> out;
  for (const auto &[name, spec] : meta.items()) {
    if (name == "__metadata__" || !spec.is_object()) continue;
    const auto shape = spec.value("shape", std::vector<std::uint64_t>{});
    if (shape.size() < 2) continue;
    const std::string dtype = spec.value("dtype", "");
    const auto offsets = spec.value("data_offsets", std::vector<std::uint64_t>{0, 0});
    std::size_t elems = 1;
    for (auto d : shape) elems *= static_cast<std::size_t>(d);
    Matrix m;
    m.rows = static_cast<std::size_t>(shape[0]);
    m.cols = elems / m.rows;  // >2-D tensors flatten trailing dims (torch ndim>=2 rule keeps them)
    m.data.resize(elems);
    const std::size_t bytes = offsets[1] - offsets[0];
    std::string raw(bytes, '\0');
    in.seekg(base + static_cast<std::streamoff>(offsets[0]));
    in.read(raw.data(), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("truncated tensor " + name + " in " + path);
    if (dtype == "F32" && bytes == elems * 4) {
      std::memcpy(m.data.data(), raw.data(), bytes);
    } else if (dtype == "BF16" && bytes == elems * 2) {
      for (std::size_t i = 0; i < elems; ++i) {
        std::uint16_t v;
        std::memcpy(&v, raw.data() + i * 2, 2);
        m.data[i] = bf16_to_f32(v);
      }
    } else if (dtype == "F16" && bytes == elems * 2) {
      for (std::size_t i = 0; i < elems; ++i) {
        std::uint16_t v;
        std::memcpy(&v, raw.data() + i * 2, 2);
        m.data[i] = f16_to_f32(v);
      }
    } else if (dtype == "F64" && bytes == elems * 8) {
      for (std::size_t i = 0; i < elems; ++i) {
        double v;
        std::memcpy(&v, raw.data() + i * 8, 8);
        m.data[i] = static_cast<float>(v);
      }
    } else {
      continue;  // integer / unsupported dtypes are not LoRA weights
    }
    out[name] = std::move(m);
  }
  return out;
}

Metrics analyze_layer(const Matrix &a, const Matrix &b) {
  double a_l1 = 0, a_l2 = 0, b_l1 = 0, b_l2 = 0;
  for (float v : a.data) { a_l1 += std::fabs(v); a_l2 += static_cast<double>(v) * v; }
  for (float v : b.data) { b_l1 += std::fabs(v); b_l2 += static_cast<double>(v) * v; }
  a_l2 = std::sqrt(a_l2);
  b_l2 = std::sqrt(b_l2);
  const auto a_sv = singular_values(a), b_sv = singular_values(b);
  const double a_spectral = a_sv.empty() ? 0.0 : a_sv[0];
  const double b_spectral = b_sv.empty() ? 0.0 : b_sv[0];

  // BA = B (m x r) @ A (r x n). Singular values through the r x r core:
  // G_A = A Aᵀ = V D Vᵀ, G_B = Bᵀ B; sv(BA)² = eig(D^½ Vᵀ G_B V D^½).
  if (b.cols != a.rows) throw std::runtime_error("LoRA shape mismatch: B cols != A rows");
  const std::size_t r = a.rows, m_rows = b.rows, n_cols = a.cols;
  std::vector<double> ga(r * r, 0.0), gb(r * r, 0.0);
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = i; j < r; ++j) {
      double s = 0;
      for (std::size_t k = 0; k < n_cols; ++k) s += static_cast<double>(a.at(i, k)) * a.at(j, k);
      ga[i * r + j] = ga[j * r + i] = s;
    }
  for (std::size_t row = 0; row < m_rows; ++row)
    for (std::size_t i = 0; i < r; ++i)
      for (std::size_t j = i; j < r; ++j) gb[i * r + j] += static_cast<double>(b.at(row, i)) * b.at(row, j);
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = i + 1; j < r; ++j) gb[j * r + i] = gb[i * r + j];
  std::vector<double> V = ga;
  const auto ga_eig = sym_eig(V, r, true);
  std::vector<double> d(r);
  for (std::size_t i = 0; i < r; ++i) d[i] = std::sqrt(std::max(0.0, ga_eig[i]));
  // core = D^½ Vᵀ G_B V D^½
  std::vector<double> gbv(r * r, 0.0), core(r * r, 0.0);
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = 0; j < r; ++j) {
      double s = 0;
      for (std::size_t k = 0; k < r; ++k) s += gb[i * r + k] * V[k * r + j];
      gbv[i * r + j] = s;
    }
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = 0; j < r; ++j) {
      double s = 0;
      for (std::size_t k = 0; k < r; ++k) s += V[k * r + i] * gbv[k * r + j];
      core[i * r + j] = d[i] * s * d[j];
    }
  auto ba_sv = sym_eigenvalues(std::move(core), r);
  for (auto &e : ba_sv) e = std::sqrt(std::max(0.0, e));
  const double ba_spectral = ba_sv.empty() ? 0.0 : ba_sv[0];
  double nuclear = 0;
  for (double s : ba_sv) nuclear += s;
  const double effective_rank = ba_spectral > 1e-12 ? nuclear / ba_spectral : 0.0;
  // weight magnitude = mean |BA| without materializing more than one row
  double abs_sum = 0;
  std::vector<float> row(n_cols);
  for (std::size_t i = 0; i < m_rows; ++i) {
    std::fill(row.begin(), row.end(), 0.0f);
    float *rp = row.data();
    for (std::size_t k = 0; k < r; ++k) {
      const float bik = b.at(i, k);
      if (bik == 0.0f) continue;
      const float *ak = &a.data[k * n_cols];
      for (std::size_t j = 0; j < n_cols; ++j) rp[j] += bik * ak[j];
    }
    double row_sum = 0;
    for (float v : row) row_sum += std::fabs(v);
    abs_sum += row_sum;
  }
  const double weight_mag = abs_sum / static_cast<double>(m_rows * n_cols);
  std::vector<double> nonzero;
  for (double s : ba_sv) if (s > 1e-10) nonzero.push_back(s);
  const double condition = nonzero.size() > 1 ? nonzero.front() / nonzero.back() : 1.0;
  const double ab_ratio = a_spectral > 1e-12 ? b_spectral / a_spectral : std::numeric_limits<double>::infinity();
  return {{"a_l1_norm", a_l1},          {"a_l2_norm", a_l2},         {"a_spectral_norm", a_spectral},
          {"b_l1_norm", b_l1},          {"b_l2_norm", b_l2},         {"b_spectral_norm", b_spectral},
          {"effective_rank", effective_rank}, {"weight_magnitude", weight_mag}, {"ba_spectral_norm", ba_spectral},
          {"condition_number", condition}, {"ab_ratio", ab_ratio}};
}

namespace {
const char *kASuffixes[] = {".lora_down.weight", ".lora_a.weight", ".lora_A.weight", ".lora.down.weight"};
const char *kBSuffixes[] = {".lora_up.weight", ".lora_b.weight", ".lora_B.weight", ".lora.up.weight"};

std::string lower(std::string s) {
  for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}
bool ends_with_ci(const std::string &key, const char *suffix) {
  const std::string k = lower(key), s = lower(suffix);
  return k.size() >= s.size() && k.compare(k.size() - s.size(), s.size(), s) == 0;
}
}  // namespace

std::map<std::string, Metrics> analyze_file(const std::string &path) {
  auto tensors = read_safetensors_matrices(path);
  std::map<std::string, std::string> a_keys, b_keys;
  for (const auto &[key, _] : tensors) {
    bool matched = false;
    for (const char *s : kASuffixes)
      if (ends_with_ci(key, s)) { a_keys[key.substr(0, key.size() - std::strlen(s))] = key; matched = true; break; }
    if (matched) continue;
    for (const char *s : kBSuffixes)
      if (ends_with_ci(key, s)) { b_keys[key.substr(0, key.size() - std::strlen(s))] = key; break; }
  }
  std::vector<std::pair<std::string, std::pair<const Matrix *, const Matrix *>>> pairs;
  for (const auto &[base, akey] : a_keys) {
    auto bit = b_keys.find(base);
    if (bit == b_keys.end()) continue;
    pairs.emplace_back(base, std::make_pair(&tensors.at(akey), &tensors.at(bit->second)));
  }
  // Layers are independent: analyse them across hardware threads.
  std::vector<Metrics> results(pairs.size());
  std::atomic<std::size_t> next{0};
  std::mutex err_mutex;
  std::string first_error;
  auto worker = [&] {
    for (std::size_t i = next.fetch_add(1); i < pairs.size(); i = next.fetch_add(1)) {
      try {
        results[i] = analyze_layer(*pairs[i].second.first, *pairs[i].second.second);
      } catch (const std::exception &e) {
        std::lock_guard<std::mutex> lock(err_mutex);
        if (first_error.empty()) first_error = pairs[i].first + ": " + e.what();
      }
    }
  };
  const unsigned n_threads = std::max(1u, std::min<unsigned>(std::thread::hardware_concurrency(), static_cast<unsigned>(pairs.size())));
  std::vector<std::thread> threads;
  for (unsigned t = 1; t < n_threads; ++t) threads.emplace_back(worker);
  worker();
  for (auto &t : threads) t.join();
  if (!first_error.empty()) throw std::runtime_error(first_error);
  std::map<std::string, Metrics> out;
  for (std::size_t i = 0; i < pairs.size(); ++i) out[pairs[i].first] = std::move(results[i]);
  return out;
}

nlohmann::json summary_stats(const std::map<std::string, Metrics> &metrics) {
  if (metrics.empty()) return nlohmann::json::object();
  const double n = static_cast<double>(metrics.size());
  double ab_sum = 0, ab_count = 0, eff_sum = 0, b_max = -std::numeric_limits<double>::infinity(), b_sum = 0;
  for (const auto &[_, m] : metrics) {
    if (std::isfinite(m.at("ab_ratio"))) { ab_sum += m.at("ab_ratio"); ab_count += 1; }
    eff_sum += m.at("effective_rank");
    b_max = std::max(b_max, m.at("b_spectral_norm"));
    b_sum += m.at("b_spectral_norm");
  }
  return {{"mean_ab_ratio", ab_count > 0 ? ab_sum / ab_count : 0.0}, {"mean_effective_rank", eff_sum / n},
          {"max_b_spectral", b_max}, {"mean_b_spectral", b_sum / n}, {"num_layers", metrics.size()}};
}

std::vector<std::string> diagnose(const std::map<std::string, Metrics> &metrics) {
  std::vector<std::string> warnings;
  if (metrics.empty()) return warnings;
  const double n = static_cast<double>(metrics.size());
  double b_dominant = 0;
  for (const auto &[_, m] : metrics)
    if (m.at("ab_ratio") > 2.0 && std::isfinite(m.at("ab_ratio"))) b_dominant += 1;
  auto shortname = [](const std::string &name) {
    const auto dot = name.rfind('.');
    return dot == std::string::npos ? name : name.substr(dot + 1);
  };
  char buf[512];
  if (b_dominant > n * 0.5) {
    std::snprintf(buf, sizeof buf, "B matrices dominating A across %.0f%% of layers (ab_ratio > 2.0) — possible overtraining or LR too high", b_dominant / n * 100);
    warnings.emplace_back(buf);
  }
  for (const auto &[name, m] : metrics)
    if (m.at("effective_rank") < 1.5) {
      std::snprintf(buf, sizeof buf, "Effective rank collapsed to %.1f in %s — dead LoRA dimensions", m.at("effective_rank"), shortname(name).c_str());
      warnings.emplace_back(buf);
    }
  for (const auto &[name, m] : metrics)
    if (m.at("condition_number") > 1000) {
      std::snprintf(buf, sizeof buf, "Condition number > 1000 (%.0f) in %s — numerical instability risk", m.at("condition_number"), shortname(name).c_str());
      warnings.emplace_back(buf);
    }
  return warnings;
}

nlohmann::json compare(const std::map<std::string, Metrics> &a, const std::map<std::string, Metrics> &b) {
  std::set<std::string> layers;
  for (const auto &[k, _] : a) layers.insert(k);
  for (const auto &[k, _] : b) layers.insert(k);
  nlohmann::json result = nlohmann::json::object();
  for (const auto &layer : layers) {
    nlohmann::json entry = nlohmann::json::object();
    const auto ia = a.find(layer), ib = b.find(layer);
    entry["lora1"] = ia == a.end() ? nlohmann::json(nullptr) : nlohmann::json(ia->second);
    entry["lora2"] = ib == b.end() ? nlohmann::json(nullptr) : nlohmann::json(ib->second);
    if (ia != a.end() && ib != b.end())
      for (const auto &[key, va] : ia->second) {
        const double vb = ib->second.at(key);
        if (std::fabs(va) > 1e-12) entry["diff_" + key + "_pct"] = ((vb - va) / std::fabs(va)) * 100.0;
        else entry["diff_" + key + "_pct"] = std::fabs(vb) < 1e-12 ? 0.0 : std::numeric_limits<double>::infinity();
      }
    result[layer] = entry;
  }
  return result;
}

std::string dump_json_py(const nlohmann::json &j) {
  // nlohmann serializes inf/nan as null; Python's json module emits Infinity/NaN.
  std::function<nlohmann::json(const nlohmann::json &)> mark = [&](const nlohmann::json &node) -> nlohmann::json {
    if (node.is_number_float()) {
      const double v = node.get<double>();
      if (std::isinf(v)) return nlohmann::json(v > 0 ? "\x01INF" : "\x01-INF");
      if (std::isnan(v)) return nlohmann::json("\x01NAN");
      return node;
    }
    if (node.is_object()) {
      nlohmann::json o = nlohmann::json::object();
      for (const auto &[k, v] : node.items()) o[k] = mark(v);
      return o;
    }
    if (node.is_array()) {
      nlohmann::json a = nlohmann::json::array();
      for (const auto &v : node) a.push_back(mark(v));
      return a;
    }
    return node;
  };
  std::string text = mark(j).dump();
  auto replace_all = [&](const std::string &from, const std::string &to) {
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
      text.replace(pos, from.size(), to);
      pos += to.size();
    }
  };
  replace_all("\"\\u0001INF\"", "Infinity");
  replace_all("\"\\u0001-INF\"", "-Infinity");
  replace_all("\"\\u0001NAN\"", "NaN");
  return text;
}

}  // namespace sb::lora
