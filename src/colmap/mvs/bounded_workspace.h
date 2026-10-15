// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/workspace.h"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace colmap {
namespace mvs {

// Whole-image bounded fusion cache. Decoded arrays and in-flight resize arrays
// are charged before loading; decoder-private scratch is outside this budget.
class BoundedWorkspace : public Workspace {
 private:
  struct CacheEntry;
  struct CacheState;

 public:
  enum class Component { kImage = 0, kDepth = 1, kNormal = 2, kBitmap = 3 };
  struct ImageData {
    Bitmap bitmap;
    DepthMap depth_map;
    NormalMap normal_map;
    bool bitmap_on_depth_grid = false;
    size_t normal_origin_x = 0;
    size_t normal_origin_y = 0;
    size_t NumBytes() const;
    Eigen::Vector3f GetNormal(size_t row, size_t col) const;
  };

  // Copies pin the same entry. The state outlives the workspace while leased.
  class ImageLease {
   public:
    ImageLease() noexcept = default;
    ImageLease(const ImageLease& other);
    ImageLease& operator=(const ImageLease& other);
    ImageLease(ImageLease&& other) noexcept = default;
    ImageLease& operator=(ImageLease&& other) noexcept;
    ~ImageLease();
    const ImageData* get() const noexcept;
    const ImageData* operator->() const noexcept { return get(); }
    explicit operator bool() const noexcept { return get() != nullptr; }
    void reset() noexcept;

   private:
    friend struct CacheState;
    ImageLease(std::shared_ptr<CacheState> state,
               std::shared_ptr<CacheEntry> entry) noexcept;
    std::shared_ptr<CacheState> state_;
    std::shared_ptr<CacheEntry> entry_;
  };

  struct CacheStats {
    size_t capacity_bytes = 0;
    size_t resident_bytes = 0;
    size_t reserved_bytes = 0;
    size_t peak_charged_bytes = 0;
    size_t loads = 0;
    size_t hits = 0;
    size_t waits = 0;
    size_t evictions = 0;
    std::array<size_t, 4> component_loads{};
    size_t normal_tile_loads = 0;
    size_t normal_tile_hits = 0;
  };

  struct MappedNormalStats {
    size_t capacity_bytes = 0, capacity_files = 0;
    size_t resident_bytes = 0, resident_files = 0;
    size_t reserved_bytes = 0, reserved_files = 0;
    size_t peak_charged_bytes = 0, peak_charged_files = 0;
    size_t loads = 0, hits = 0, evictions = 0, waits = 0, active_copies = 0;
  };
  struct TileMetadataStats {
    size_t slot_count = 0, free = 0, live = 0;
  };

  explicit BoundedWorkspace(const Options& options);
  void Load(const std::vector<std::string>&) override {}
  ImageLease AcquireImage(int image_idx);
  ImageLease AcquireComponent(int image_idx, Component component);
  ImageLease AcquireNormalTile(int image_idx,
                               size_t row,
                               size_t col,
                               size_t tile_size);
  std::pair<size_t, size_t> GetNormalTileGrid(int image_idx) const;
  size_t GetNormalTileReservationBytes(int image_idx, size_t tile_size) const;
  size_t GetNormalTileResidentBytes(int image_idx, size_t tile_size) const;
  size_t GetLoadReservationBytes(int image_idx,
                                 Component component = Component::kImage) const;
  size_t GetResidentBytesEstimate(
      int image_idx, Component component = Component::kImage) const;
  std::pair<size_t, size_t> GetDepthMapDimensions(int image_idx) const;
  CacheStats GetCacheStats() const;
  MappedNormalStats GetMappedNormalStats() const;
  TileMetadataStats GetTileMetadataStats() const;
  // Test-only, one-shot allocation-like failure at tile-key publication.
  void FailNextTileKeyPublicationForTesting();

  // The inherited borrowed getters cannot express a parallel cache's pin.
  const Bitmap& GetBitmap(int image_idx) override;
  const DepthMap& GetDepthMap(int image_idx) override;
  const NormalMap& GetNormalMap(int image_idx) override;

 private:
  struct NormalRegion {
    size_t x, y, width, height, tile_size;
  };
  struct MappedNormalPool : std::enable_shared_from_this<MappedNormalPool> {
    struct Entry {
      std::unique_ptr<ImmutableMapRegion> owner;
      size_t pins = 0, reserved_bytes = 0;
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
    MappedNormalPool(size_t bytes, size_t files);
    Lease Acquire(int image_idx, const std::filesystem::path& path);
    std::mutex mutex;
    std::condition_variable changed;
    std::map<int, Entry> entries;
    MappedNormalStats stats;
    uint64_t clock = 0;
  };
  int ComponentKey(int image_idx, Component component) const;
  size_t RequiredLoadBytes(int image_idx, Component component) const;
  ImageLease AcquireImpl(int image_idx,
                         Component component,
                         std::optional<NormalRegion> region);
  std::shared_ptr<CacheState> cache_state_;
  std::shared_ptr<MappedNormalPool> mapped_normal_pool_;
  // Map files and model paths are immutable for a Workspace lifetime. Publish
  // both validated shapes together; failed first admission remains retryable.
  mutable std::vector<std::once_flag> normal_tile_grid_once_;
  mutable std::vector<std::pair<size_t, size_t>> normal_tile_grids_;
};

}  // namespace mvs
}  // namespace colmap
