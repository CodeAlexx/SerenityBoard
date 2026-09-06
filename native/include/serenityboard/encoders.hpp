// PNG (zlib), WAV (PCM16), NumPy-compatible histogram, PR curve packing.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sb {

/// Encode 8-bit pixels (HWC, channels 1/3/4) as PNG; returns the file bytes.
std::string encode_png(std::uint32_t width, std::uint32_t height, std::uint32_t channels,
                       const std::uint8_t *pixels);

/// PCM16 WAV; `samples` are interleaved int16 frames (len = frames * channels).
std::string encode_wav_pcm16(const std::vector<std::int16_t> &samples, std::uint32_t channels,
                             std::uint32_t sample_rate);

/// np.histogram(values, bins=bins) over finite values: rows of [left, right, count]
/// as float64 (bins x 3). Empty when no finite value.
std::vector<double> histogram_rows(const std::vector<double> &values, int bins);

/// summary_writer.add_pr_curve packing: (6, num_thresholds) float64
/// rows tp, fp, tn, fn, precision, recall. thresholds = linspace(0,1,n).
std::vector<double> pr_curve_rows(const std::vector<double> &labels, const std::vector<double> &predictions,
                                  int num_thresholds);

/// numpy.linspace(start, stop, num) semantics (num >= 2).
std::vector<double> linspace(double start, double stop, int num);

}  // namespace sb
