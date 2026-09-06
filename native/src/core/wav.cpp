#include <cstring>

#include "serenityboard/encoders.hpp"

namespace sb {

namespace {
void put_u32le(std::string &out, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>(v >> (8 * i)));
}
void put_u16le(std::string &out, std::uint16_t v) {
  out.push_back(static_cast<char>(v));
  out.push_back(static_cast<char>(v >> 8));
}
}  // namespace

// Matches Python's wave module output (RIFF/WAVE, fmt 16 bytes, PCM, data).
std::string encode_wav_pcm16(const std::vector<std::int16_t> &samples, std::uint32_t channels,
                             std::uint32_t sample_rate) {
  const std::uint32_t data_bytes = static_cast<std::uint32_t>(samples.size() * 2);
  std::string out;
  out.reserve(44 + data_bytes);
  out += "RIFF";
  put_u32le(out, 36 + data_bytes);
  out += "WAVE";
  out += "fmt ";
  put_u32le(out, 16);
  put_u16le(out, 1);
  put_u16le(out, static_cast<std::uint16_t>(channels));
  put_u32le(out, sample_rate);
  put_u32le(out, sample_rate * channels * 2);
  put_u16le(out, static_cast<std::uint16_t>(channels * 2));
  put_u16le(out, 16);
  out += "data";
  put_u32le(out, data_bytes);
  for (std::int16_t s : samples) put_u16le(out, static_cast<std::uint16_t>(s));
  return out;
}

}  // namespace sb
