#include "serenityboard/blob_storage.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "serenityboard/hash.hpp"

namespace sb {

BlobStorage::BlobStorage(std::string blobs_dir) : blobs_dir_(std::move(blobs_dir)) {
  std::filesystem::create_directories(blobs_dir_);
}

std::string BlobStorage::key_for(std::string_view data, const std::string &extension) {
  const auto digest = sha256(data);
  return hex(digest.data(), digest.size()).substr(0, 16) + "." + extension;
}

std::string BlobStorage::store(std::string_view data, const std::string &extension) const {
  const std::string key = key_for(data, extension);
  const std::filesystem::path path = std::filesystem::path(blobs_dir_) / key;
  if (std::filesystem::exists(path)) return key;
  const std::filesystem::path tmp = path.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write blob " + tmp.string());
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out) throw std::runtime_error("short write for blob " + tmp.string());
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) throw std::runtime_error("cannot rename blob " + tmp.string() + ": " + ec.message());
  return key;
}

std::string BlobStorage::get_path(const std::string &key) const {
  return (std::filesystem::path(blobs_dir_) / key).string();
}

}  // namespace sb
