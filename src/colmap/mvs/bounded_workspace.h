// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/immutable_map_pool.h"
#include "colmap/mvs/workspace.h"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
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
    std::vector<uint8_t> positive_depth_mask;
    size_t exact_depth_width = 0, exact_depth_height = 0;
    static size_t PositiveDepthMaskBytes(size_t samples) {
      return samples / 8 + (samples % 8 != 0);
    }
    bool HasPositiveDepth(size_t row, size_t col) const;
    float GetDepth(size_t row, size_t col) const;
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

  using MappedNormalStats = ::colmap::mvs::MappedNormalStats;
  struct TileMetadataStats {
    size_t slot_count = 0, free = 0, live = 0;
  };

  explicit BoundedWorkspace(const Options& options);
  ~BoundedWorkspace() override;
  void Load(const std::vector<std::string>&) override {}
  ImageLease AcquireImage(int image_idx);
  ImageLease AcquireComponent(int image_idx, Component component);
  ImageLease AcquireNormalTile(int image_idx,
                               size_t row,
                               size_t col,
                               size_t tile_size);
  // Admit complete original FP32 depth before starting row workers. Stored
  // and consumed grids must agree; inputs remain immutable for the run.
  void PrepareOriginalDepth(int image_idx, size_t width, size_t height);
  float ReadOriginalDepth(int image_idx,
                          size_t row,
                          size_t col,
                          Mat<float>& scalar_scratch);
  struct DepthMappedNeighborhoodStats {
    size_t source_admissions = 0, root_preparations = 0;
    size_t complete_roots = 0, fallback_roots = 0, cancelled_roots = 0;
    size_t pin_acquires = 0, borrowed_reads = 0, scalar_acquires = 0;
    size_t metadata_peak_bytes = 0;
  };
  // Caller thread with joined row tasks: all reachable owners or fallback.
  // Keep worst excluded scalar byte/file headroom to avoid self-waiting.
  bool PrepareOriginalDepthNeighborhood(
      const std::vector<int>& views,
      const std::function<bool()>& stop_requested);
  void ReleaseOriginalDepthNeighborhood() noexcept;
  DepthMappedNeighborhoodStats GetDepthMappedNeighborhoodStats() const {
    return depth_mapped_neighborhood_stats_;
  }
  // Pins the complete FP32 normal file. The owner must outlive every sample.
  using MappedNormalLease = std::shared_ptr<const ImmutableMapRegion>;
  MappedNormalLease AcquireMappedNormal(int image_idx);
  MappedNormalLease AcquireMappedNormalWindow(int image_idx,
                                              size_t row_begin,
                                              size_t row_count);
  size_t GetMappedNormalReservationBytes(int image_idx) const;
  size_t GetMappedNormalWindowReservationBytes(int image_idx,
                                               size_t row_begin,
                                               size_t row_count) const;
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
  friend struct BoundedWorkspaceExactDepthPinTestPeer;
  struct NormalRegion {
    size_t x, y, width, height, tile_size;
  };
  using MappedNormalPool = ::colmap::mvs::MappedNormalPool;
  int ComponentKey(int image_idx, Component component) const;
  size_t RequiredLoadBytes(int image_idx, Component component) const;
  ImageLease AcquireImpl(int image_idx,
                         Component component,
                         std::optional<NormalRegion> region);
  std::shared_ptr<CacheState> cache_state_;
  std::shared_ptr<MappedNormalPool> mapped_normal_pool_;
  struct OriginalDepthSource {
    size_t width = 0, height = 0;
    std::filesystem::path path;
    size_t mapping_reservation_bytes = 0;
    const ImmutableMapRegion* borrowed_owner = nullptr;
  };
  std::vector<OriginalDepthSource> original_depth_sources_;
  std::vector<MappedNormalPool::Lease> original_depth_pins_;
  DepthMappedNeighborhoodStats depth_mapped_neighborhood_stats_;
  size_t DepthPinMetadataBytes(size_t selected_metadata_bytes) const;
  // Map files and model paths are immutable for a Workspace lifetime. Publish
  // both validated shapes together; failed first admission remains retryable.
  mutable std::vector<std::once_flag> normal_tile_grid_once_;
  mutable std::vector<std::pair<size_t, size_t>> normal_tile_grids_;
};

}  // namespace mvs
}  // namespace colmap
