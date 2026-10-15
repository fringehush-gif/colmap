// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/bounded_workspace.h"
#include "colmap/mvs/mat.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/base_controller.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/ply.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <boost/container_hash/hash.hpp>

namespace colmap {
class ThreadPool;
namespace mvs {

struct StereoFusionOptions {
  // Path for PNG masks. Same format expected as ImageReaderOptions.
  std::filesystem::path mask_path = "";

  // The number of threads to use during fusion.
  int num_threads = -1;

  // Full-RAM workspace loading can use a separate positive thread count.
  // -1 retains the existing num_threads behavior.
  int full_ram_load_threads = -1;

  // Maximum image size in either dimension.
  int max_image_size = -1;

  // Minimum number of fused pixels to produce a point.
  int min_num_pixels = 5;

  // Maximum number of pixels to fuse into a single point.
  int max_num_pixels = 10000;

  // Maximum depth in consistency graph traversal.
  int max_traversal_depth = 100;

  // Maximum relative difference between measured and projected pixel.
  double max_reproj_error = 2.0f;

  // Maximum relative difference between measured and projected depth.
  double max_depth_error = 0.01f;

  // Maximum angular difference in degrees of normals of pixels to be fused.
  double max_normal_error = 10.0f;

  // Number of overlapping images to transitively check for fusing points.
  int check_num_images = 50;

  // Flag indicating whether to use LRU cache or pre-load all data
  bool use_cache = false;

  // Opt-in byte-bounded, pinned whole-image fusion cache.
  bool bounded_cache = false;

  // Load depth, normal and bitmap independently so rejected depth samples do
  // not decode the other components. All components share the byte budget.
  bool bounded_cache_components = false;

  // FP32 normal-map tiles in bounded component mode. With tile_size=0 and a
  // positive mapped budget, one fusion worker samples directly from pinned
  // FP32 maps, without decoded normal tiles. All defaults retain prior paths.
  // Mapped-owner limits are
  // independent virtual/file-backed byte and handle budgets, not decoded RSS.
  // The tile/neighborhood composition awaits a separate headroom policy.
  int bounded_cache_normal_tile_size = 0;
  int bounded_cache_normal_tile_slots = 4;
  bool bounded_cache_normal_tile_hints = true;
  int64_t bounded_cache_normal_mapped_bytes = 0;
  int bounded_cache_normal_mapped_files = 0;

  // Pin an admitted overlap set for the duration of each fused root. Row tasks
  // may borrow its immutable arrays only until the root's worker join.
  bool bounded_cache_neighborhood = false;

  // -1 uses the fusion pool. A positive value caps independent prefill lanes
  // without changing the number of fusion row workers.
  int bounded_cache_prefill_threads = -1;

  // Opt-in bounded-cache RGB compaction to the depth grid.
  bool bounded_cache_compact_rgb = false;

  // Cache size in gigabytes for fusion. The fusion keeps the bitmaps, depth
  // maps, normal maps, and consistency graphs of this number of images in
  // memory. A higher value leads to less disk access and faster fusion, while
  // a lower value leads to reduced memory usage. Note that a single image can
  // consume a lot of memory, if the consistency graph is dense.
  double cache_size = 32.0;

  std::pair<Eigen::Vector3f, Eigen::Vector3f> bounding_box =
      std::make_pair(Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX),
                     Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));

  // Emit one bounded, cold-path cache diagnostic row after CLI Fusion.
  bool bounded_cache_diagnostics = false;

  // Check the options for validity.
  bool Check() const;

  // Print the options to stdout.
  void Print() const;
};

namespace internal {
// Index only the already-owned serial mapped ring. Entries and last-image
// hints are bounded by the admitted slot count and model image count.
class MappedNormalRingIndex {
 public:
  using Key = std::pair<int, size_t>;

  void Reset(size_t images, size_t slot_count) {
    index_.clear();
    index_.reserve(slot_count);
    hints_.assign(images, Hint{slot_count, 0});
    slots_ = slot_count;
  }
  void Clear() {
    index_.clear();
    std::fill(hints_.begin(), hints_.end(), Hint{slots_, 0});
  }
  size_t Find(int image, size_t band) {
    const auto& hint = hints_.at(static_cast<size_t>(image));
    if (hint.slot < slots_ && hint.band == band) return hint.slot;
    const auto found = index_.find({image, band});
    if (found == index_.end()) return slots_;
    hints_[image] = {found->second, band};
    return found->second;
  }
  bool Insert(int image, size_t band, size_t slot) {
    auto& hint = hints_.at(static_cast<size_t>(image));
    if (slot >= slots_ || index_.size() >= slots_) return false;
    const auto inserted = index_.emplace(Key{image, band}, slot);
    if (inserted.second) hint = {slot, band};
    return inserted.second;
  }
  bool Erase(int image, size_t band, size_t slot) {
    auto& hint = hints_.at(static_cast<size_t>(image));
    const auto found = index_.find({image, band});
    if (found == index_.end() || found->second != slot) return false;
    index_.erase(found);
    if (hint.slot == slot && hint.band == band) hint.slot = slots_;
    return true;
  }

 private:
  struct Hint {
    size_t slot;
    size_t band;
  };
  size_t slots_ = 0;
  NodeHashMap<Key, size_t, boost::hash<Key>> index_;
  std::vector<Hint> hints_;
};
}  // namespace internal

// Fuses per-view depth and normal maps into a consistent 3D point cloud.
class StereoFusion : public BaseController {
 public:
  StereoFusion(const StereoFusionOptions& options,
               const std::filesystem::path& workspace_path,
               const std::string& workspace_format,
               const std::string& pmvs_option_name,
               const std::string& input_type);

  // Get the fused 3D points as PLY points.
  const std::vector<PlyPoint>& GetFusedPoints() const;
  // Get per-point visibility lists (indices of images that observe each point).
  const std::vector<std::vector<int>>& GetFusedPointsVisibility() const;
  std::optional<BoundedWorkspace::CacheStats> GetBoundedCacheStats() const;
  size_t GetMaxBoundedPrefillLanesScheduled() const;
  BoundedWorkspace::MappedNormalStats GetMappedNormalStats() const;
  struct NormalTileRingStats {
    size_t hits = 0;
    size_t misses = 0;
    size_t indexed_lookups = 0;
    size_t hint_checks = 0;
  };
  NormalTileRingStats GetNormalTileRingStats() const;
  std::optional<std::string> GetBoundedCacheDiagnosticRow() const;

  // Execute the depth map fusion pipeline.
  void Run();

 private:
  void PrepareBoundedNeighborhood(int image_idx,
                                  int num_threads,
                                  ThreadPool& thread_pool);
  void ClearNormalTileRings();
  const BoundedWorkspace::ImageData* AcquireWorkerNormalTile(int thread_id,
                                                             int image_idx,
                                                             int row,
                                                             int col);
  const ImmutableMapRegion* AcquireWorkerMappedNormal(int image_idx, int row);
  void ClearMappedNormalRing();
  void InitFusedPixelMask(int image_idx, size_t width, size_t height);
  void Fuse(int thread_id, int image_idx, int row, int col);

  const StereoFusionOptions options_;
  const std::filesystem::path workspace_path_;
  const std::string workspace_format_;
  const std::string pmvs_option_name_;
  const std::string input_type_;
  const float max_squared_reproj_error_;
  const float min_cos_normal_error_;

  std::unique_ptr<Workspace> workspace_;
  BoundedWorkspace* bounded_workspace_ = nullptr;
  std::vector<const BoundedWorkspace::ImageData*> neighborhood_data_;
  std::vector<BoundedWorkspace::ImageLease> neighborhood_leases_;
  size_t max_bounded_prefill_lanes_scheduled_ = 0;
  using NormalTileKey = std::tuple<int, size_t, size_t>;
  struct NormalTileSlot {
    int image_idx = -1;
    NormalTileKey key{};
    BoundedWorkspace::ImageLease lease;
  };
  struct alignas(64) NormalTileRing {
    std::vector<NormalTileSlot> tiles;
    NodeHashMap<NormalTileKey, size_t, boost::hash<NormalTileKey>> index;
    std::vector<size_t> last_slot_by_image;
    bool indexed = false;
    size_t next = 0, hits = 0, misses = 0;
    size_t indexed_lookups = 0, hint_checks = 0;
  };
  std::vector<NormalTileRing> normal_tile_rings_;
  struct MappedNormalSlot {
    int image_idx = -1;
    size_t row_begin = 0, row_count = 0;
    BoundedWorkspace::MappedNormalLease lease;
  };
  std::vector<MappedNormalSlot> mapped_normal_ring_;
  internal::MappedNormalRingIndex mapped_normal_ring_index_;
  size_t mapped_normal_next_ = 0;
  bool direct_mapped_normals_ = false;
  size_t normal_tile_worker_headroom_ = 0;
  std::vector<char> used_images_;
  std::vector<char> fused_images_;
  std::vector<std::vector<int>> overlapping_images_;
  // Contains image masks of pre-masked and already fused pixels.
  // Initialized from image masks if provided in StereoFusionOptions.
  std::vector<Mat<char>> fused_pixel_masks_;
  std::vector<std::pair<int, int>> depth_map_sizes_;
  std::vector<std::pair<float, float>> bitmap_scales_;
  std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> P_;
  std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> inv_P_;
  std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> inv_R_;

  struct FusionData {
    int image_idx = kInvalidImageId;
    int row = 0;
    int col = 0;
    int traversal_depth = -1;
    FusionData(int image_idx, int row, int col, int traversal_depth)
        : image_idx(image_idx),
          row(row),
          col(col),
          traversal_depth(traversal_depth) {}
    bool operator()(const FusionData& data1, const FusionData& data2) {
      return data1.image_idx > data2.image_idx;
    }
  };

  // Already fused points.
  std::vector<PlyPoint> fused_points_;
  std::vector<std::vector<int>> fused_points_visibility_;

  std::vector<std::vector<PlyPoint>> task_fused_points_;
  std::vector<std::vector<std::vector<int>>> task_fused_points_visibility_;
};

// Write the visibility information into a binary file of the following format:
//
//    <num_points : uint64_t>
//    <num_visible_images_for_point1 : uint32_t>
//    <point1_image_idx1 : uint32_t><point1_image_idx2 : uint32_t> ...
//    <num_visible_images_for_point2 : uint32_t>
//    <point2_image_idx2 : uint32_t><point2_image_idx2 : uint32_t> ...
//    ...
//
// Note that an image_idx in the case of the mvs::StereoFusion does not
// correspond to the image_id of a Reconstruction, but the index of the image in
// the mvs::Model, which is the location of the image in the images.bin/.txt.
void WritePointsVisibility(
    const std::filesystem::path& path,
    const std::vector<std::vector<int>>& points_visibility);

// Read per-point visibility data from a binary .vis file produced by the
// stereo fusion pipeline. Returns a vector of per-point image index lists.
// The format matches WritePointsVisibility().
std::vector<std::vector<int>> ReadPointsVisibility(
    const std::filesystem::path& path, size_t num_points);

}  // namespace mvs
}  // namespace colmap
