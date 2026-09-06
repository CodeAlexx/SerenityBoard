// Bounded per-key reservoir sampling (serenityboard/writer/reservoir.py).
// Deterministic per seed. NOTE: the RNG is std::mt19937 (not CPython's
// random.Random), so the *sampled subset* can differ from the Python writer for
// the same stream; capacity, unbounded mode, always_keep_last and per-drain
// reset semantics are identical.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace sb {

template <typename T>
class Reservoir {
public:
  explicit Reservoir(std::size_t max_size, std::uint32_t seed = 0, bool always_keep_last = true)
      : max_size_(max_size), seed_(seed), always_keep_last_(always_keep_last) {}

  void add(const std::string &key, T item) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buckets_.find(key);
    if (it == buckets_.end()) {
      Bucket fresh;
      fresh.rng.seed(seed_);
      it = buckets_.emplace(key, std::move(fresh)).first;
    }
    Bucket &b = it->second;
    if (max_size_ == 0 || b.items.size() < max_size_) {
      b.items.push_back(std::move(item));
    } else {
      std::uniform_int_distribution<std::size_t> dist(0, b.num_seen);
      const std::size_t r = dist(b.rng);
      if (r < max_size_) {
        b.items.erase(b.items.begin() + static_cast<std::ptrdiff_t>(r));
        b.items.push_back(std::move(item));
      } else if (always_keep_last_) {
        b.items.back() = std::move(item);
      }
    }
    b.num_seen += 1;
  }

  std::vector<T> get_items(const std::string &key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buckets_.find(key);
    if (it == buckets_.end()) return {};
    return it->second.items;
  }

  std::vector<T> drain_items(const std::string &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buckets_.find(key);
    if (it == buckets_.end()) return {};
    std::vector<T> out;
    out.swap(it->second.items);
    it->second.num_seen = 0;
    return out;
  }

  std::vector<std::string> keys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    for (const auto &[k, _] : buckets_) out.push_back(k);
    return out;
  }

  std::size_t max_size() const { return max_size_; }

private:
  struct Bucket {
    std::mt19937 rng{};
    std::vector<T> items{};
    std::size_t num_seen{0};
  };
  std::size_t max_size_;
  std::uint32_t seed_;
  bool always_keep_last_;
  mutable std::mutex mutex_;
  std::map<std::string, Bucket> buckets_;
};

}  // namespace sb
