// Content-addressed blob files: <blobs_dir>/<sha256(data)[:16]>.<ext>
// (serenityboard/writer/blob_storage.py). Dedup by key, atomic .tmp + rename.
#pragma once

#include <string>
#include <string_view>

namespace sb {

class BlobStorage {
public:
  explicit BlobStorage(std::string blobs_dir);
  /// Store bytes; returns the blob key. Existing key = no rewrite.
  std::string store(std::string_view data, const std::string &extension) const;
  std::string get_path(const std::string &key) const;
  const std::string &dir() const { return blobs_dir_; }
  static std::string key_for(std::string_view data, const std::string &extension);

private:
  std::string blobs_dir_;
};

}  // namespace sb
