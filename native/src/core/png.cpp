#include <zlib.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "serenityboard/encoders.hpp"

namespace sb {

namespace {

void put_u32(std::string &out, std::uint32_t v) {
  out.push_back(static_cast<char>(v >> 24));
  out.push_back(static_cast<char>(v >> 16));
  out.push_back(static_cast<char>(v >> 8));
  out.push_back(static_cast<char>(v));
}

void chunk(std::string &out, const char *type, const std::string &data) {
  put_u32(out, static_cast<std::uint32_t>(data.size()));
  std::string body(type, 4);
  body += data;
  out += body;
  const auto crc = crc32(0L, reinterpret_cast<const Bytef *>(body.data()), static_cast<uInt>(body.size()));
  put_u32(out, static_cast<std::uint32_t>(crc));
}

}  // namespace

std::string encode_png(std::uint32_t width, std::uint32_t height, std::uint32_t channels,
                       const std::uint8_t *pixels) {
  if (width == 0 || height == 0) throw std::invalid_argument("png: empty image");
  std::uint8_t color_type;
  switch (channels) {
    case 1: color_type = 0; break;
    case 3: color_type = 2; break;
    case 4: color_type = 6; break;
    default: throw std::invalid_argument("png: unsupported channel count " + std::to_string(channels));
  }
  const std::size_t row = static_cast<std::size_t>(width) * channels;
  std::string raw;
  raw.reserve((row + 1) * height);
  for (std::uint32_t y = 0; y < height; ++y) {
    raw.push_back(0);  // filter: none
    raw.append(reinterpret_cast<const char *>(pixels + static_cast<std::size_t>(y) * row), row);
  }
  uLongf bound = compressBound(static_cast<uLong>(raw.size()));
  std::string z(bound, '\0');
  if (compress2(reinterpret_cast<Bytef *>(z.data()), &bound, reinterpret_cast<const Bytef *>(raw.data()),
                static_cast<uLong>(raw.size()), Z_DEFAULT_COMPRESSION) != Z_OK)
    throw std::runtime_error("png: zlib compression failed");
  z.resize(bound);

  std::string out("\x89PNG\r\n\x1a\n", 8);
  std::string ihdr;
  put_u32(ihdr, width);
  put_u32(ihdr, height);
  ihdr.push_back(8);
  ihdr.push_back(static_cast<char>(color_type));
  ihdr.push_back(0);
  ihdr.push_back(0);
  ihdr.push_back(0);
  chunk(out, "IHDR", ihdr);
  chunk(out, "IDAT", z);
  chunk(out, "IEND", "");
  return out;
}

std::vector<double> linspace(double start, double stop, int num) {
  std::vector<double> out;
  if (num <= 0) return out;
  out.resize(static_cast<std::size_t>(num));
  if (num == 1) {
    out[0] = start;
    return out;
  }
  const double step = (stop - start) / static_cast<double>(num - 1);
  for (int i = 0; i < num; ++i) out[static_cast<std::size_t>(i)] = start + static_cast<double>(i) * step;
  out[static_cast<std::size_t>(num - 1)] = stop;
  return out;
}

// np.histogram(a, bins=N): uniform edges over [min, max] (or [min-0.5, max+0.5]
// when min == max); values on the last right edge fall into the last bin;
// index = floor((x - first) * norm) with the round-off correction NumPy applies.
std::vector<double> histogram_rows(const std::vector<double> &values, int bins) {
  std::vector<double> finite;
  finite.reserve(values.size());
  for (double v : values)
    if (std::isfinite(v)) finite.push_back(v);
  if (finite.empty() || bins <= 0) return {};
  double lo = finite[0], hi = finite[0];
  for (double v : finite) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo == hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const auto edges = linspace(lo, hi, bins + 1);
  std::vector<double> counts(static_cast<std::size_t>(bins), 0.0);
  const double norm = static_cast<double>(bins) / (hi - lo);
  for (double v : finite) {
    long idx = static_cast<long>((v - lo) * norm);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    // NumPy's correction for floating-point rounding at the edges.
    while (idx > 0 && v < edges[static_cast<std::size_t>(idx)]) --idx;
    while (idx + 1 < bins && v >= edges[static_cast<std::size_t>(idx) + 1]) ++idx;
    counts[static_cast<std::size_t>(idx)] += 1.0;
  }
  std::vector<double> rows;
  rows.reserve(static_cast<std::size_t>(bins) * 3);
  for (int i = 0; i < bins; ++i) {
    rows.push_back(edges[static_cast<std::size_t>(i)]);
    rows.push_back(edges[static_cast<std::size_t>(i) + 1]);
    rows.push_back(counts[static_cast<std::size_t>(i)]);
  }
  return rows;
}

std::vector<double> pr_curve_rows(const std::vector<double> &labels, const std::vector<double> &predictions,
                                  int num_thresholds) {
  if (labels.size() != predictions.size()) throw std::invalid_argument("labels and predictions must have the same length");
  if (num_thresholds <= 0) throw std::invalid_argument("num_thresholds must be positive");
  const auto thresholds = linspace(0.0, 1.0, num_thresholds);
  const std::size_t n = static_cast<std::size_t>(num_thresholds);
  std::vector<double> data(6 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double tp = 0, fp = 0, tn = 0, fn = 0;
    for (std::size_t k = 0; k < labels.size(); ++k) {
      const bool predicted_pos = predictions[k] >= thresholds[i];
      const bool actual_pos = labels[k] >= 0.5;
      if (predicted_pos && actual_pos) tp += 1;
      else if (predicted_pos) fp += 1;
      else if (actual_pos) fn += 1;
      else tn += 1;
    }
    data[0 * n + i] = tp;
    data[1 * n + i] = fp;
    data[2 * n + i] = tn;
    data[3 * n + i] = fn;
    data[4 * n + i] = (tp + fp) > 0 ? tp / (tp + fp) : 1.0;
    data[5 * n + i] = (tp + fn) > 0 ? tp / (tp + fn) : 0.0;
  }
  return data;
}

}  // namespace sb
