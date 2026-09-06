// SHA-256 (blob keys) and SHA-1 (WebSocket handshake), in-tree — no OpenSSL on the box.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace sb {

std::array<std::uint8_t, 32> sha256(std::string_view bytes);
std::array<std::uint8_t, 20> sha1(std::string_view bytes);
std::string hex(const std::uint8_t *bytes, std::size_t n);
std::string base64(std::string_view bytes);

}  // namespace sb
