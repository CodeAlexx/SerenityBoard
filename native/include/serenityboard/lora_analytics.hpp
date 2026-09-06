// LoRA weight analytics (serenityboard/lora_analytics.py) without torch:
// safetensors reader + exact singular values through small Gram matrices.
#pragma once

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sb::lora {

struct Matrix {
  std::size_t rows{0}, cols{0};
  std::vector<float> data;  // row-major
  float at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

/// Read every >=2-D tensor of a safetensors file as float32 (F32/F16/BF16/F64 supported).
std::map<std::string, Matrix> read_safetensors_matrices(const std::string &path);

using Metrics = std::map<std::string, double>;

Metrics analyze_layer(const Matrix &a, const Matrix &b);
/// {layer: metrics} for every A/B pair (suffix rules identical to the Python module).
std::map<std::string, Metrics> analyze_file(const std::string &path);
nlohmann::json summary_stats(const std::map<std::string, Metrics> &metrics);
std::vector<std::string> diagnose(const std::map<std::string, Metrics> &metrics);
nlohmann::json compare(const std::map<std::string, Metrics> &a, const std::map<std::string, Metrics> &b);
/// JSON with inf rendered the way Python's json module does ("Infinity").
std::string dump_json_py(const nlohmann::json &j);

/// Singular values (descending) of a matrix, exact through the smaller Gram matrix.
std::vector<double> singular_values(const Matrix &m);

}  // namespace sb::lora
