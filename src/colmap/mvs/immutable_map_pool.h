// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/mat.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace colmap::mvs {
struct MappedNormalStats {
  size_t capacity_bytes = 0, capacity_files = 0;
  // One owner holds one file handle; a stripe holds three mapped VMAs.
  size_t capacity_regions = 0;
  size_t resident_bytes = 0, resident_files = 0;
  size_t resident_regions = 0;
  size_t reserved_bytes = 0, reserved_files = 0;
  size_t reserved_regions = 0;
  size_t peak_charged_bytes = 0, peak_charged_files = 0;
  size_t peak_charged_regions = 0;
  size_t loads = 0, hits = 0, evictions = 0, waits = 0, active_copies = 0;
  // Immutable source descriptors are charged separately from stripe owners.
  size_t source_loads = 0, source_hits = 0, source_evictions = 0;
  size_t source_charged_files = 0, source_peak_charged_files = 0;
  size_t source_live_files = 0, source_peak_live_files = 0;
};
struct MappedNormalPool : std::enable_shared_from_this<MappedNormalPool> {
  struct SourceEntry {
    std::shared_ptr<const ImmutableMapSource> source;
    std::filesystem::path path;
    uint64_t last_access = 0;
    bool loading = true;
  };
  struct Entry {
    std::unique_ptr<ImmutableMapRegion> owner;
    size_t pins = 0, reserved_bytes = 0;
    size_t regions = 1;
    uint64_t last_access = 0;
    bool loading = true;
  };
  struct Lease {
    Lease(std::shared_ptr<MappedNormalPool> pool, Entry* entry);
    Lease(Lease&& other) noexcept;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease();
    const ImmutableMapRegion& operator*() const { return *entry->owner; }
    std::shared_ptr<MappedNormalPool> pool;
    Entry* entry;
  };
  MappedNormalPool(size_t bytes, size_t files, bool bind_sources = false);
  Lease Acquire(int image_idx, const std::filesystem::path& path);
  Lease Acquire(int image_idx,
                const std::filesystem::path& path,
                size_t row_begin,
                size_t row_count);
  using Key = std::tuple<int, size_t, size_t>;
  Lease AcquireImpl(Key key, const std::filesystem::path& path);
  std::shared_ptr<const ImmutableMapSource> AcquireSource(
      int image_idx, const std::filesystem::path& path);
  std::mutex mutex;
  std::condition_variable changed;
  std::map<Key, Entry> entries;
  std::map<int, SourceEntry> sources;
  bool bind_sources = false;
  size_t sources_closing = 0;
  MappedNormalStats stats;
  uint64_t clock = 0;
};
}  // namespace colmap::mvs
