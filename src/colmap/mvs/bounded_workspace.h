// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/workspace.h"

#include <cstddef>
#include <cstdint>
#include <memory>
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
  struct ImageData {
    Bitmap bitmap;
    DepthMap depth_map;
    NormalMap normal_map;
    bool bitmap_on_depth_grid = false;
    size_t NumBytes() const;
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
  };

  explicit BoundedWorkspace(const Options& options);
  void Load(const std::vector<std::string>&) override {}
  ImageLease AcquireImage(int image_idx);
  size_t GetLoadReservationBytes(int image_idx) const;
  size_t GetResidentBytesEstimate(int image_idx) const;
  std::pair<size_t, size_t> GetDepthMapDimensions(int image_idx) const;
  CacheStats GetCacheStats() const;

  // The inherited borrowed getters cannot express a parallel cache's pin.
  const Bitmap& GetBitmap(int image_idx) override;
  const DepthMap& GetDepthMap(int image_idx) override;
  const NormalMap& GetNormalMap(int image_idx) override;

 private:
  size_t RequiredLoadBytes(int image_idx) const;
  std::shared_ptr<CacheState> cache_state_;
};

}  // namespace mvs
}  // namespace colmap
