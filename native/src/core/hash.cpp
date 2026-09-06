#include "serenityboard/hash.hpp"

#include <cstring>
#include <vector>

namespace sb {

namespace {

inline std::uint32_t rotr(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32U - n)); }
inline std::uint32_t rotl(std::uint32_t x, unsigned n) { return (x << n) | (x >> (32U - n)); }

constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

std::vector<std::uint8_t> pad_message(std::string_view bytes) {
  std::vector<std::uint8_t> m(bytes.begin(), bytes.end());
  const std::uint64_t bit_length = static_cast<std::uint64_t>(bytes.size()) * 8U;
  m.push_back(0x80);
  while (m.size() % 64 != 56) m.push_back(0);
  for (int i = 7; i >= 0; --i) m.push_back(static_cast<std::uint8_t>(bit_length >> (8 * i)));
  return m;
}

}  // namespace

std::array<std::uint8_t, 32> sha256(std::string_view bytes) {
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const auto m = pad_message(bytes);
  for (std::size_t offset = 0; offset < m.size(); offset += 64) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (std::uint32_t(m[offset + 4 * i]) << 24) | (std::uint32_t(m[offset + 4 * i + 1]) << 16) |
             (std::uint32_t(m[offset + 4 * i + 2]) << 8) | std::uint32_t(m[offset + 4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = hh + S1 + ch + kSha256K[i] + w[i];
      const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  std::array<std::uint8_t, 32> out{};
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) out[static_cast<std::size_t>(4 * i + j)] = static_cast<std::uint8_t>(h[i] >> (24 - 8 * j));
  return out;
}

std::array<std::uint8_t, 20> sha1(std::string_view bytes) {
  std::uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
  const auto m = pad_message(bytes);
  for (std::size_t offset = 0; offset < m.size(); offset += 64) {
    std::uint32_t w[80];
    for (int i = 0; i < 16; ++i)
      w[i] = (std::uint32_t(m[offset + 4 * i]) << 24) | (std::uint32_t(m[offset + 4 * i + 1]) << 16) |
             (std::uint32_t(m[offset + 4 * i + 2]) << 8) | std::uint32_t(m[offset + 4 * i + 3]);
    for (int i = 16; i < 80; ++i) w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    std::uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
    for (int i = 0; i < 80; ++i) {
      std::uint32_t f, k;
      if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
      else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
      else { f = b ^ c ^ d; k = 0xCA62C1D6; }
      const std::uint32_t temp = rotl(a, 5) + f + e + k + w[i];
      e = d; d = c; c = rotl(b, 30); b = a; a = temp;
    }
    h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
  }
  std::array<std::uint8_t, 20> out{};
  const std::uint32_t hs[5] = {h0, h1, h2, h3, h4};
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 4; ++j) out[static_cast<std::size_t>(4 * i + j)] = static_cast<std::uint8_t>(hs[i] >> (24 - 8 * j));
  return out;
}

std::string hex(const std::uint8_t *bytes, std::size_t n) {
  static const char digits[] = "0123456789abcdef";
  std::string out;
  out.reserve(n * 2);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(digits[bytes[i] >> 4]);
    out.push_back(digits[bytes[i] & 0xF]);
  }
  return out;
}

std::string base64(std::string_view bytes) {
  static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  std::size_t i = 0;
  const auto *p = reinterpret_cast<const unsigned char *>(bytes.data());
  while (i + 2 < bytes.size()) {
    const std::uint32_t v = (std::uint32_t(p[i]) << 16) | (std::uint32_t(p[i + 1]) << 8) | p[i + 2];
    out.push_back(table[(v >> 18) & 63]); out.push_back(table[(v >> 12) & 63]);
    out.push_back(table[(v >> 6) & 63]); out.push_back(table[v & 63]);
    i += 3;
  }
  if (i + 1 == bytes.size()) {
    const std::uint32_t v = std::uint32_t(p[i]) << 16;
    out.push_back(table[(v >> 18) & 63]); out.push_back(table[(v >> 12) & 63]); out += "==";
  } else if (i + 2 == bytes.size()) {
    const std::uint32_t v = (std::uint32_t(p[i]) << 16) | (std::uint32_t(p[i + 1]) << 8);
    out.push_back(table[(v >> 18) & 63]); out.push_back(table[(v >> 12) & 63]); out.push_back(table[(v >> 6) & 63]); out += "=";
  }
  return out;
}

}  // namespace sb
