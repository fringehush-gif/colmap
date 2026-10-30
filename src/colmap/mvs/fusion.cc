// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/fusion.h"

#include "colmap/math/math.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace colmap {
namespace mvs {
namespace {
// A source hypothesis: one immutable row band per owner. This is not a
// measured optimum and does not alter any sampled FP32 value.
constexpr size_t kMappedNormalStripeRows = 256;

// Return the rounded median directly for nonnegative byte samples.
uint8_t RoundedByteMedian(std::vector<uint8_t>& elems) {
  THROW_CHECK(!elems.empty());
  const size_t middle = elems.size() / 2;
  std::nth_element(elems.begin(), elems.begin() + middle, elems.end());
  const unsigned int upper = elems[middle];
  if (elems.size() % 2 != 0) {
    return static_cast<uint8_t>(upper);
  }
  const unsigned int lower =
      *std::max_element(elems.begin(), elems.begin() + middle);
  return static_cast<uint8_t>((lower + upper + 1) / 2);
}

}  // namespace
namespace internal {

void FusionPixelMask::ExcludeNonpositiveDepth(const FusionFloatMapView& depth) {
  if constexpr (sizeof(float) != sizeof(uint32_t) ||
                !std::numeric_limits<float>::is_iec559) {
    return;
  }
  const float representation_probe = 0x1.2468acp0f;
  uint32_t representation_bits;
  std::memcpy(
      &representation_bits, &representation_probe, sizeof(representation_bits));
  if (representation_bits != 0x3f923456u) return;
  if (depth.width != width_ || depth.plane != size_ || depth.size < size_ ||
      (size_ != 0 && depth.data == nullptr)) {
    return;
  }
  for (size_t first = 0; first < size_;) {
    const size_t count = std::min(kWordBits, size_ - first);
    uint64_t excluded = 0;
    for (size_t bit = 0; bit < count; ++bit) {
      uint32_t bits;
      std::memcpy(&bits, depth.data + first + bit, sizeof(bits));
      const uint32_t magnitude = bits & 0x7fffffffu;
      // Leave NaNs and subnormals to the original worker-local comparison.
      // In particular, classification must not depend on the initializer's
      // floating-point flags or denormals-are-zero setting.
      const bool nonpositive = magnitude == 0 || ((bits & 0x80000000u) != 0 &&
                                                  magnitude >= 0x00800000u &&
                                                  magnitude <= 0x7f800000u);
      excluded |= static_cast<uint64_t>(nonpositive) << bit;
    }
    auto& word = data_[first / kWordBits];
    word.store(word.load(std::memory_order_relaxed) | excluded,
               std::memory_order_relaxed);
    first += count;
  }
}

void AppendFusionVisibilityBlock(std::vector<std::vector<int>>& blocks,
                                 const FlatHashSet<int>& visibility) {
  constexpr size_t kBlockWords = 64 * 1024 / sizeof(int);
  THROW_CHECK_LE(visibility.size(),
                 static_cast<size_t>(std::numeric_limits<int>::max()));
  THROW_CHECK_LT(visibility.size(), std::numeric_limits<size_t>::max());
  const size_t record_words = visibility.size() + 1;
  if (blocks.empty() || blocks.back().size() > kBlockWords ||
      record_words > kBlockWords - blocks.back().size()) {
    blocks.emplace_back();
    blocks.back().reserve(std::max(kBlockWords, record_words));
  }
  auto& block = blocks.back();
  block.push_back(static_cast<int>(visibility.size()));
  block.insert(block.end(), visibility.begin(), visibility.end());
}

size_t MergeFusionVisibilityBlocks(std::vector<std::vector<int>>& blocks,
                                   std::vector<std::vector<int>>& output) {
  const size_t first = output.size();
  for (auto& block : blocks) {
    size_t offset = 0;
    while (offset < block.size()) {
      const int count = block[offset++];
      if (count < 0 || static_cast<size_t>(count) > block.size() - offset) {
        throw std::invalid_argument("Malformed Fusion visibility block");
      }
      output.emplace_back(block.begin() + offset,
                          block.begin() + offset + count);
      offset += static_cast<size_t>(count);
    }
    // Release each payload before materializing the next block's lists.
    std::vector<int>().swap(block);
  }
  blocks.clear();
  return output.size() - first;
}

std::vector<int> FusionNeighborhood(
    const std::vector<std::vector<int>>& overlapping_images,
    const std::vector<char>& used_images,
    const std::vector<char>& fused_images,
    const int root_image_idx,
    const int max_traversal_depth) {
  THROW_CHECK_EQ(overlapping_images.size(), used_images.size());
  THROW_CHECK_EQ(used_images.size(), fused_images.size());
  THROW_CHECK_GT(max_traversal_depth, 0);
  if (!used_images.at(root_image_idx)) return {};
  std::vector<int> views{root_image_idx};
  std::vector<int> depths{0};
  std::vector<char> seen(used_images.size(), false);
  seen.at(root_image_idx) = true;
  for (size_t cursor = 0; cursor < views.size(); ++cursor) {
    if (depths[cursor] >= max_traversal_depth - 1) continue;
    for (const int next : overlapping_images.at(views[cursor])) {
      if (seen.at(next) || !used_images.at(next) || fused_images.at(next)) {
        continue;
      }
      seen[next] = true;
      views.push_back(next);
      depths.push_back(depths[cursor] + 1);
    }
  }
  return views;
}

FusionPinIndices DepthPriorityFusionPins(
    const std::vector<size_t>& depth_bytes,
    const std::vector<size_t>& bitmap_bytes,
    const size_t budget) {
  THROW_CHECK_EQ(depth_bytes.size(), bitmap_bytes.size());
  FusionPinIndices admitted;
  if (depth_bytes.empty() || depth_bytes[0] > budget ||
      bitmap_bytes[0] > budget - depth_bytes[0]) {
    return admitted;
  }
  admitted.depth.push_back(0);
  admitted.bitmap.push_back(0);
  size_t remaining = budget - depth_bytes[0] - bitmap_bytes[0];
  for (size_t index = 1; index < depth_bytes.size(); ++index) {
    if (depth_bytes[index] > remaining) continue;
    admitted.depth.push_back(index);
    remaining -= depth_bytes[index];
  }
  for (size_t index = 1; index < bitmap_bytes.size(); ++index) {
    if (bitmap_bytes[index] > remaining) continue;
    admitted.bitmap.push_back(index);
    remaining -= bitmap_bytes[index];
  }
  return admitted;
}

// Use the sparse model to find most connected image that has not yet been
// fused. This is used as a heuristic to ensure that the workspace cache reuses
// already cached images as efficient as possible.
int FindNextImage(const std::vector<std::vector<int>>& overlapping_images,
                  const std::vector<char>& used_images,
                  const std::vector<char>& fused_images,
                  const int prev_image_idx) {
  THROW_CHECK_EQ(used_images.size(), fused_images.size());

  for (const auto image_idx : overlapping_images.at(prev_image_idx)) {
    if (used_images.at(image_idx) && !fused_images.at(image_idx)) {
      return image_idx;
    }
  }

  // If none of the overlapping images are not yet fused, simply return the
  // first image that has not yet been fused.
  for (size_t image_idx = 0; image_idx < fused_images.size(); ++image_idx) {
    if (used_images[image_idx] && !fused_images[image_idx]) {
      return image_idx;
    }
  }

  return -1;
}

}  // namespace internal

void StereoFusionOptions::Print() const {
#define PrintOption(option) LOG(INFO) << #option ": " << (option);
  LOG_HEADING2("StereoFusion::Options");
  PrintOption(mask_path);
  PrintOption(full_ram_load_threads);
  PrintOption(max_image_size);
  PrintOption(min_num_pixels);
  PrintOption(max_num_pixels);
  PrintOption(max_traversal_depth);
  PrintOption(max_reproj_error);
  PrintOption(max_depth_error);
  PrintOption(max_normal_error);
  PrintOption(check_num_images);
  PrintOption(use_cache);
  PrintOption(bounded_cache);
  PrintOption(bounded_cache_components);
  PrintOption(bounded_cache_diagnostics);
  PrintOption(bounded_cache_normal_tile_size);
  PrintOption(bounded_cache_normal_tile_slots);
  PrintOption(bounded_cache_normal_tile_hints);
  PrintOption(bounded_cache_exact_depth);
  PrintOption(bounded_cache_normal_mapped_bytes);
  PrintOption(bounded_cache_normal_mapped_files);
  PrintOption(bounded_cache_neighborhood);
  PrintOption(bounded_cache_prefill_threads);
  PrintOption(bounded_cache_compact_rgb);
  PrintOption(cache_size);
  const auto& bbox_min = bounding_box.first.transpose().eval();
  const auto& bbox_max = bounding_box.second.transpose().eval();
  PrintOption(bbox_min);
  PrintOption(bbox_max);
#undef PrintOption
}

bool StereoFusionOptions::Check() const {
  CHECK_OPTION_EQ((use_cache && bounded_cache), false);
  CHECK_OPTION_IN(min_num_pixels, 0, max_num_pixels);
  CHECK_OPTION_GT(max_traversal_depth, 0);
  CHECK_OPTION_GE(max_reproj_error, 0);
  CHECK_OPTION_GE(max_depth_error, 0);
  CHECK_OPTION_GE(max_normal_error, 0);
  CHECK_OPTION_GT(check_num_images, 0);
  CHECK_OPTION_GT(cache_size, 0);
  CHECK_OPTION_EQ((bounded_cache_compact_rgb && !bounded_cache), false);
  CHECK_OPTION_EQ((bounded_cache_components && !bounded_cache), false);
  CHECK_OPTION_EQ((bounded_cache_diagnostics && !bounded_cache), false);
  CHECK_OPTION_GE(bounded_cache_normal_tile_size, 0);
  CHECK_OPTION_GT(bounded_cache_normal_tile_slots, 0);
  CHECK_OPTION_GE(bounded_cache_normal_mapped_bytes, 0);
  CHECK_OPTION_GE(bounded_cache_normal_mapped_files, 0);
  CHECK_OPTION_LE(static_cast<uint64_t>(bounded_cache_normal_mapped_bytes),
                  std::numeric_limits<size_t>::max());
  CHECK_OPTION_EQ((bounded_cache_normal_mapped_bytes == 0),
                  (bounded_cache_normal_mapped_files == 0));
  CHECK_OPTION_EQ(
      (bounded_cache_normal_tile_size > 0 &&
       (!bounded_cache_components || bounded_cache_normal_mapped_bytes == 0)),
      false);
  CHECK_OPTION_EQ(
      (!bounded_cache_exact_depth && bounded_cache_normal_tile_size == 0 &&
       bounded_cache_normal_mapped_bytes > 0 &&
       (!bounded_cache || !bounded_cache_components ||
        bounded_cache_neighborhood || num_threads != 1 ||
        bounded_cache_normal_mapped_files < 2)),
      false);
  CHECK_OPTION_EQ((bounded_cache_exact_depth &&
                   (!bounded_cache || !bounded_cache_components ||
                    bounded_cache_normal_tile_size != 0 || num_threads != 1 ||
                    bounded_cache_normal_mapped_bytes == 0 ||
                    bounded_cache_normal_mapped_files == 0)),
                  false);
  CHECK_OPTION_EQ((bounded_cache_neighborhood && !bounded_cache), false);
  CHECK_OPTION_GE(bounded_cache_prefill_threads, -1);
  CHECK_OPTION_NE(bounded_cache_prefill_threads, 0);
  CHECK_OPTION_EQ(
      (bounded_cache_prefill_threads != -1 && !bounded_cache_neighborhood),
      false);
  CHECK_OPTION_GE(full_ram_load_threads, -1);
  CHECK_OPTION_NE(full_ram_load_threads, 0);
  CHECK_OPTION_EQ((full_ram_load_threads != -1 && (use_cache || bounded_cache)),
                  false);
  return true;
}

StereoFusion::StereoFusion(const StereoFusionOptions& options,
                           const std::filesystem::path& workspace_path,
                           const std::string& workspace_format,
                           const std::string& pmvs_option_name,
                           const std::string& input_type)
    : options_(options),
      workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      pmvs_option_name_(pmvs_option_name),
      input_type_(input_type),
      max_squared_reproj_error_(options_.max_reproj_error *
                                options_.max_reproj_error),
      min_cos_normal_error_(std::cos(DegToRad(options_.max_normal_error))) {
  THROW_CHECK(options_.Check());
}

const std::vector<PlyPoint>& StereoFusion::GetFusedPoints() const {
  return fused_points_;
}

const std::vector<std::vector<int>>& StereoFusion::GetFusedPointsVisibility()
    const {
  return fused_points_visibility_;
}

std::optional<BoundedWorkspace::CacheStats> StereoFusion::GetBoundedCacheStats()
    const {
  if (!bounded_workspace_) return std::nullopt;
  return bounded_workspace_->GetCacheStats();
}

std::pair<size_t, size_t> StereoFusion::GetDepthExactStateReadCounts() const {
  std::pair<size_t, size_t> counts{};
  for (const auto& scratch : exact_depth_scratch_) {
    counts.first += scratch.root_reads;
    counts.second += scratch.candidate_reads;
  }
  return counts;  // Read only after all row workers have joined.
}

BoundedWorkspace::DepthMappedNeighborhoodStats
StereoFusion::GetDepthMappedNeighborhoodStats() const {
  return bounded_workspace_
             ? bounded_workspace_->GetDepthMappedNeighborhoodStats()
             : BoundedWorkspace::DepthMappedNeighborhoodStats{};
}

size_t StereoFusion::GetMaxBoundedPrefillLanesScheduled() const {
  return max_bounded_prefill_lanes_scheduled_;
}

BoundedWorkspace::MappedNormalStats StereoFusion::GetMappedNormalStats() const {
  return bounded_workspace_ ? bounded_workspace_->GetMappedNormalStats()
                            : BoundedWorkspace::MappedNormalStats{};
}

StereoFusion::NormalTileRingStats StereoFusion::GetNormalTileRingStats() const {
  NormalTileRingStats stats;
  for (const auto& ring : normal_tile_rings_) {
    stats.hits += ring.hits;
    stats.misses += ring.misses;
    stats.indexed_lookups += ring.indexed_lookups;
    stats.hint_checks += ring.hint_checks;
  }
  return stats;
}

std::optional<std::string> StereoFusion::GetBoundedCacheDiagnosticRow() const {
  if (!options_.bounded_cache_diagnostics) return std::nullopt;

  std::ostringstream row;
  row.imbue(std::locale::classic());
  row << "COLMAP_FUSION_CACHE_DIAGNOSTIC_V1"
      << " tile_mode=" << (options_.bounded_cache_normal_tile_size > 0)
      << " direct_mapped_mode=" << direct_mapped_normals_;
  if (direct_mapped_normals_) {
    row << " direct_component_pins_admitted=" << direct_component_pins_admitted_
        << " direct_depth_borrows=" << direct_depth_borrows_
        << " direct_bitmap_borrows=" << direct_bitmap_borrows_;
  }
  if (options_.bounded_cache_neighborhood &&
      options_.bounded_cache_normal_tile_size > 0) {
    size_t depth_reads = 0, depth_misses = 0, bitmap_reads = 0;
    size_t bitmap_misses = 0, bitmap_initializations = 0;
    for (const auto& reads : task_neighborhood_reads_) {
      depth_reads += reads.pointer_reads;
      depth_misses += reads.demand_acquires;
      bitmap_reads += reads.rgb_pointer_reads;
      bitmap_misses += reads.rgb_demand_acquires;
      bitmap_initializations += reads.rgb_init_loads;
    }
    row << " neighborhood_bitmap_admitted="
        << neighborhood_bitmap_admitted_views_
        << " neighborhood_bitmap_carried=" << neighborhood_bitmap_carried_views_
        << " neighborhood_depth_borrows=" << depth_reads
        << " neighborhood_depth_demand=" << depth_misses
        << " neighborhood_bitmap_borrows=" << bitmap_reads
        << " neighborhood_bitmap_demand=" << bitmap_misses
        << " neighborhood_bitmap_init=" << bitmap_initializations;
  }
  const auto cache = GetBoundedCacheStats();
  if (cache) {
    row << " decoded_capacity_bytes=" << cache->capacity_bytes
        << " decoded_peak_charged_bytes=" << cache->peak_charged_bytes
        << " decoded_resident_bytes=" << cache->resident_bytes
        << " decoded_reserved_bytes=" << cache->reserved_bytes
        << " decoded_loads=" << cache->loads << " decoded_hits=" << cache->hits
        << " decoded_waits=" << cache->waits
        << " decoded_evictions=" << cache->evictions
        << " decoded_normal_tile_loads=" << cache->normal_tile_loads
        << " decoded_normal_tile_hits=" << cache->normal_tile_hits;
  } else {
    row << " decoded_cache=unavailable";
  }

  if (options_.bounded_cache_normal_mapped_bytes > 0 && cache) {
    const auto mapped = GetMappedNormalStats();
    row << " mapped_capacity_bytes=" << mapped.capacity_bytes
        << " mapped_capacity_files=" << mapped.capacity_files
        << " mapped_capacity_regions=" << mapped.capacity_regions
        << " mapped_peak_charged_bytes=" << mapped.peak_charged_bytes
        << " mapped_peak_charged_files=" << mapped.peak_charged_files
        << " mapped_peak_charged_regions=" << mapped.peak_charged_regions
        << " mapped_loads=" << mapped.loads << " mapped_hits=" << mapped.hits
        << " mapped_waits=" << mapped.waits
        << " mapped_evictions=" << mapped.evictions
        << " mapped_active_copies=" << mapped.active_copies
        << " mapped_source_loads=" << mapped.source_loads
        << " mapped_source_hits=" << mapped.source_hits
        << " mapped_source_evictions=" << mapped.source_evictions
        << " mapped_source_charged_files=" << mapped.source_charged_files
        << " mapped_source_peak_charged_files="
        << mapped.source_peak_charged_files
        << " mapped_source_live_files=" << mapped.source_live_files
        << " mapped_source_peak_live_files=" << mapped.source_peak_live_files;
    if (direct_mapped_normals_) {
      row << " direct_ring_slots=" << mapped_normal_ring_.size()
          << " direct_window_rows=" << kMappedNormalStripeRows
          << " direct_ring_hits=unavailable";
    } else {
      const auto ring = GetNormalTileRingStats();
      row << " ring_hits=" << ring.hits << " ring_misses=" << ring.misses
          << " ring_indexed_lookups=" << ring.indexed_lookups
          << " ring_hint_checks=" << ring.hint_checks
          << " ring_hint_hits=unavailable";
    }
  } else {
    row << " mapped_normal=unavailable ring=unavailable";
  }
  return row.str();
}

void StereoFusion::Run() {
  Timer run_timer;
  run_timer.Start();

  full_ram_inputs_.clear();
  ClearDirectComponentPins();
  direct_component_pins_admitted_ = false;
  direct_depth_borrows_ = 0;
  direct_bitmap_borrows_ = 0;
  neighborhood_bitmap_pins_.reset();
  neighborhood_bitmap_admitted_views_ = 0;
  neighborhood_bitmap_carried_views_ = 0;
  task_neighborhood_reads_.clear();
  fused_points_.clear();
  fused_points_visibility_.clear();
  // A failed row may have appended points before the joined exception.
  // Reset every run while retaining the worker vectors' allocated capacity.
  for (auto& points : task_fused_points_) points.clear();
  for (auto& visibility : task_fused_points_visibility_) visibility.clear();
  max_bounded_prefill_lanes_scheduled_ = 0;
  normal_tile_rings_.clear();
  mapped_normal_ring_.clear();
  mapped_normal_ring_index_.Clear();
  mapped_normal_next_ = 0;
  exact_depth_scratch_.clear();
  direct_mapped_normals_ = !options_.bounded_cache_exact_depth &&
                           options_.bounded_cache_normal_tile_size == 0 &&
                           options_.bounded_cache_normal_mapped_bytes > 0;
  normal_tile_worker_headroom_ = 0;

  options_.Print();

  LOG(INFO) << "Reading workspace...";

  Workspace::Options workspace_options;
  auto workspace_format_lower_case = workspace_format_;
  StringToLower(&workspace_format_lower_case);
  if (workspace_format_lower_case == "pmvs") {
    workspace_options.stereo_folder =
        StringPrintf("stereo-%s", pmvs_option_name_.c_str());
  }

  workspace_options.num_threads = options_.num_threads;
  workspace_options.max_image_size = options_.max_image_size;
  workspace_options.image_as_rgb = true;
  workspace_options.compact_rgb_to_depth_grid =
      options_.bounded_cache_compact_rgb;
  workspace_options.exact_depth_state = options_.bounded_cache_exact_depth;
  workspace_options.mapped_normal_bytes_limit =
      static_cast<size_t>(options_.bounded_cache_normal_mapped_bytes);
  workspace_options.mapped_normal_files_limit =
      static_cast<size_t>(options_.bounded_cache_normal_mapped_files);
  workspace_options.cache_size = options_.cache_size;
  workspace_options.workspace_path = workspace_path_;
  workspace_options.workspace_format = workspace_format_;
  workspace_options.input_type = input_type_;

  const auto image_names = ReadTextFileLines(
      workspace_path_ / workspace_options.stereo_folder / "fusion.cfg");
  int num_threads = 1;
  bounded_workspace_ = nullptr;
  if (options_.bounded_cache) {
    auto workspace = std::make_unique<BoundedWorkspace>(workspace_options);
    bounded_workspace_ = workspace.get();
    workspace_ = std::move(workspace);
    num_threads = GetEffectiveNumThreads(options_.num_threads);
  } else if (options_.use_cache) {
    workspace_ = std::make_unique<CachedWorkspace>(workspace_options);
  } else {
    if (options_.full_ram_load_threads != -1) {
      // Workspace::Load writes one model slot per name. An opt-in parallel
      // loader must not schedule the same slot twice for duplicate cfg rows.
      std::vector<std::string> unique_names;
      FlatHashSet<std::string> seen;
      unique_names.reserve(image_names.size());
      for (const auto& name : image_names) {
        if (seen.insert(name).second) unique_names.push_back(name);
      }
      const size_t roster_size = std::max<size_t>(unique_names.size(), 1);
      workspace_options.num_threads = static_cast<int>(std::min(
          static_cast<size_t>(options_.full_ram_load_threads), roster_size));
      workspace_ = std::make_unique<Workspace>(workspace_options);
      workspace_->Load(unique_names);
    } else {
      workspace_ = std::make_unique<Workspace>(workspace_options);
      workspace_->Load(image_names);
    }
    num_threads = GetEffectiveNumThreads(options_.num_threads);
  }

  if (CheckIfStopped()) {
    run_timer.PrintMinutes();
    return;
  }

  LOG(INFO) << "Reading configuration...";

  const auto& model = workspace_->GetModel();
  if (!options_.use_cache && !options_.bounded_cache) {
    full_ram_inputs_.resize(model.images.size());
  }
  neighborhood_data_.assign(model.images.size(), nullptr);
  neighborhood_leases_.clear();
  task_neighborhood_reads_.resize(num_threads);

  const double kMinTriangulationAngle = 0;
  if (model.GetMaxOverlappingImagesFromPMVS().empty()) {
    overlapping_images_ = model.GetMaxOverlappingImages(
        options_.check_num_images, kMinTriangulationAngle);
  } else {
    overlapping_images_ = model.GetMaxOverlappingImagesFromPMVS();
  }

  task_fused_points_.resize(num_threads);
  task_fused_points_visibility_.resize(num_threads);
  if (options_.bounded_cache_exact_depth) {
    exact_depth_scratch_.resize(num_threads);
    for (auto& scratch : exact_depth_scratch_) {
      scratch.scalar = Mat<float>(1, 1, 1);
    }
  }

  used_images_.assign(model.images.size(), false);
  fused_images_.assign(model.images.size(), false);
  fused_pixel_masks_.resize(model.images.size());
  depth_map_sizes_.resize(model.images.size());
  bitmap_scales_.resize(model.images.size());
  P_.resize(model.images.size());
  inv_P_.resize(model.images.size());
  inv_R_.resize(model.images.size());
  for (const auto& image_name : image_names) {
    const int image_idx = model.GetImageIdx(image_name);

    if (!workspace_->HasBitmap(image_idx) ||
        !workspace_->HasDepthMap(image_idx) ||
        !workspace_->HasNormalMap(image_idx)) {
      LOG(WARNING) << StringPrintf(
          "Ignoring image %s, because input does not exist.",
          image_name.c_str());
      continue;
    }

    const auto& image = model.images.at(image_idx);
    size_t depth_width, depth_height;
    if (bounded_workspace_) {
      std::tie(depth_width, depth_height) =
          bounded_workspace_->GetDepthMapDimensions(image_idx);
      if (options_.bounded_cache_exact_depth) {
        bounded_workspace_->PrepareOriginalDepth(
            image_idx, depth_width, depth_height);
      }
    } else {
      const auto depth_map = workspace_->GetDepthMapLease(image_idx);
      depth_width = depth_map->GetWidth();
      depth_height = depth_map->GetHeight();
    }

    used_images_.at(image_idx) = true;

    if (!full_ram_inputs_.empty()) {
      full_ram_inputs_.at(image_idx) = {
          internal::FusionFloatMapView(workspace_->GetDepthMap(image_idx)),
          internal::FusionFloatMapView(workspace_->GetNormalMap(image_idx)),
          &workspace_->GetBitmap(image_idx)};
    }

    InitFusedPixelMask(image_idx, depth_width, depth_height);
    if (!full_ram_inputs_.empty()) {
      fused_pixel_masks_.at(image_idx).ExcludeNonpositiveDepth(
          full_ram_inputs_.at(image_idx).depth);
    }

    depth_map_sizes_.at(image_idx) = {depth_width, depth_height};

    bitmap_scales_.at(image_idx) =
        std::make_pair(static_cast<float>(depth_width) / image.GetWidth(),
                       static_cast<float>(depth_height) / image.GetHeight());

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            image.GetK());
    K(0, 0) *= bitmap_scales_.at(image_idx).first;
    K(0, 2) *= bitmap_scales_.at(image_idx).first;
    K(1, 1) *= bitmap_scales_.at(image_idx).second;
    K(1, 2) *= bitmap_scales_.at(image_idx).second;
    K(0, 2) -= 0.5f;
    K(1, 2) -= 0.5f;

    ComposeProjectionMatrix(
        K.data(), image.GetR(), image.GetT(), P_.at(image_idx).data());
    ComposeInverseProjectionMatrix(
        K.data(), image.GetR(), image.GetT(), inv_P_.at(image_idx).data());
    inv_R_.at(image_idx) =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            image.GetR())
            .transpose();
  }

  if (direct_mapped_normals_) {
    THROW_CHECK_EQ(num_threads, 1);
    const auto mapped = bounded_workspace_->GetMappedNormalStats();
    size_t largest = 0, used = 0;
    for (size_t index = 0; index < used_images_.size(); ++index) {
      if (!used_images_[index]) continue;
      ++used;
      const auto grid =
          bounded_workspace_->GetNormalTileGrid(static_cast<int>(index));
      for (size_t begin = 0; begin < grid.second;) {
        const size_t count =
            std::min(kMappedNormalStripeRows, grid.second - begin);
        largest =
            std::max(largest,
                     bounded_workspace_->GetMappedNormalWindowReservationBytes(
                         static_cast<int>(index), begin, count));
        begin += count;
      }
    }
    if (used) {
      THROW_CHECK_GT(largest, 0);
      THROW_CHECK_LE(largest, mapped.capacity_bytes);
      const size_t owners_by_bytes = mapped.capacity_bytes / largest;
      THROW_CHECK_GE(owners_by_bytes, 2)
          << "Mapped normal stripes need one retained owner and one incoming";
      THROW_CHECK_GE(mapped.capacity_files, 2);
      THROW_CHECK_GE(mapped.capacity_regions / 3, 2);
      const size_t retained = std::min(
          {static_cast<size_t>(options_.bounded_cache_normal_tile_slots),
           mapped.capacity_files - 1,
           mapped.capacity_regions / 3 - 1,
           owners_by_bytes - 1});
      THROW_CHECK_GT(retained, 0);
      mapped_normal_ring_.resize(retained);
      mapped_normal_ring_index_.Reset(model.images.size(), retained);
    }
  }
  if (options_.bounded_cache_normal_tile_size > 0) {
    size_t largest_component = 0, largest_tile = 0;
    using Component = BoundedWorkspace::Component;
    for (size_t index = 0; index < used_images_.size(); ++index) {
      if (!used_images_[index]) continue;
      const int view = static_cast<int>(index);
      largest_tile =
          std::max(largest_tile,
                   bounded_workspace_->GetNormalTileResidentBytes(
                       view, options_.bounded_cache_normal_tile_size));
      largest_component = std::max(
          {largest_component,
           bounded_workspace_->GetNormalTileReservationBytes(
               view, options_.bounded_cache_normal_tile_size),
           bounded_workspace_->GetLoadReservationBytes(view, Component::kDepth),
           bounded_workspace_->GetLoadReservationBytes(view,
                                                       Component::kBitmap)});
    }
    const size_t per_worker =
        bounded_workspace_->GetCacheStats().capacity_bytes /
        static_cast<size_t>(num_threads);
    const size_t slots =
        static_cast<size_t>(options_.bounded_cache_normal_tile_slots);
    THROW_CHECK_LE(largest_component, per_worker);
    THROW_CHECK_LE(largest_tile, (per_worker - largest_component) / slots)
        << "Normal tile rings need retained pins and component headroom";
    normal_tile_worker_headroom_ = largest_component + slots * largest_tile;
    normal_tile_rings_.resize(num_threads);
    for (auto& ring : normal_tile_rings_) {
      ring.tiles.resize(slots);
      ring.indexed = slots > 4;
      if (ring.indexed) ring.index.reserve(slots);
      if (ring.indexed && options_.bounded_cache_normal_tile_hints) {
        ring.last_slot_by_image.assign(model.images.size(), slots);
      }
    }
  }

  LOG(INFO) << StringPrintf("Starting fusion with %d threads", num_threads);
  // Construct before the pool: release after its destructor joins any tasks
  // left by enqueue, callback or Wait failure.
  struct DepthPinExit {
    BoundedWorkspace* workspace;
    ~DepthPinExit() {
      if (workspace) workspace->ReleaseOriginalDepthNeighborhood();
    }
  } depth_pin_exit{options_.bounded_cache_exact_depth &&
                           options_.bounded_cache_neighborhood
                       ? bounded_workspace_
                       : nullptr};
  ThreadPool thread_pool(num_threads);
  // Only an explicit larger prefill count needs another pool. It is created
  // once per run and bounded by the configured image roster.
  std::unique_ptr<ThreadPool> independent_prefill_pool;
  if (bounded_workspace_ && options_.bounded_cache_neighborhood &&
      options_.bounded_cache_prefill_threads > num_threads) {
    const size_t roster_size = std::max<size_t>(
        std::count(used_images_.begin(), used_images_.end(), true), 1);
    const size_t workers =
        std::min(static_cast<size_t>(options_.bounded_cache_prefill_threads),
                 roster_size);
    if (workers > static_cast<size_t>(num_threads)) {
      independent_prefill_pool =
          std::make_unique<ThreadPool>(static_cast<int>(workers));
    }
  }
  ThreadPool& prefill_pool =
      independent_prefill_pool ? *independent_prefill_pool : thread_pool;

  // Using a row stride of 10 to avoid starting parallel processing in rows that
  // are too close to each other which may lead to duplicated work, since nearby
  // pixels are likely to get fused into the same point.
  const int kRowStride = 10;
  auto ProcessImageRows = [&, this](const int row_start,
                                    const int height,
                                    const int width,
                                    const int image_idx,
                                    const std::vector<char>& fused_pixel_mask) {
    const int row_end = std::min(height, row_start + kRowStride);
    // One row-band callback stays on its executing pool worker.
    const int thread_id = thread_pool.GetThreadIndex();
    for (int row = row_start; row < row_end; ++row) {
      for (int col = 0; col < width; ++col) {
        const size_t offset =
            static_cast<size_t>(row - row_start) * width + col;
        if (fused_pixel_mask.at(offset) > 0) {
          continue;
        }
        Fuse(thread_id, image_idx, row, col);
      }
    }
  };

  try {
    // Direct component pins are acquired before row tasks can borrow them.
    // A failed admission leaves no published raw pointer or retained pin.
    PrepareDirectComponentPins();
    // Destination eligibility changes only after the row-task join. Keep the
    // ordered live adjacency here rather than checking both flags per pixel.
    std::vector<std::vector<int>> incoming_images(overlapping_images_.size());
    for (size_t source = 0; source < overlapping_images_.size(); ++source) {
      auto& views = overlapping_images_[source];
      views.erase(std::remove_if(
                      views.begin(),
                      views.end(),
                      [&](const int view) { return !used_images_.at(view); }),
                  views.end());
      for (const int view : views) {
        incoming_images.at(view).push_back(static_cast<int>(source));
      }
    }
    size_t num_fused_images = 0;
    size_t total_fused_points = 0;
    for (int image_idx = 0; image_idx >= 0;
         image_idx = internal::FindNextImage(
             overlapping_images_, used_images_, fused_images_, image_idx)) {
      if (CheckIfStopped()) {
        break;
      }

      Timer timer;
      timer.Start();

      LOG(INFO) << StringPrintf("Fusing image [%d/%d] with index %d",
                                num_fused_images + 1,
                                model.images.size(),
                                image_idx)
                << std::flush;

      const int width = depth_map_sizes_.at(image_idx).first;
      const int height = depth_map_sizes_.at(image_idx).second;
      const auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);

      if (bounded_workspace_ && options_.bounded_cache_neighborhood) {
        PrepareBoundedNeighborhood(image_idx, num_threads, prefill_pool);
        if (CheckIfStopped()) break;
      }

      for (int row_start = 0; row_start < height; row_start += kRowStride) {
        const int row_end = std::min(height, row_start + kRowStride);
        const size_t first = static_cast<size_t>(row_start) * width;
        const size_t last = static_cast<size_t>(row_end) * width;
        // A task reads only its own rows. Move that snapshot into the bound
        // task rather than copying the entire image mask for every band.
        std::vector<char> row_mask = fused_pixel_mask.Snapshot(first, last);
        thread_pool.AddTask(ProcessImageRows,
                            row_start,
                            height,
                            width,
                            image_idx,
                            std::move(row_mask));
      }
      thread_pool.Wait();
      if (depth_pin_exit.workspace) {
        depth_pin_exit.workspace->ReleaseOriginalDepthNeighborhood();
      }
      ClearNormalTileRings();
      ClearMappedNormalRing();

      num_fused_images += 1;
      fused_images_.at(image_idx) = true;
      for (const int source : incoming_images.at(image_idx)) {
        auto& views = overlapping_images_.at(source);
        views.erase(std::remove(views.begin(), views.end(), image_idx),
                    views.end());
      }

      if (!full_ram_inputs_.empty() && used_images_.at(image_idx)) {
        // All readers joined and no remaining adjacency can reach this image.
        // Drop borrowed views before releasing their eager FP32 owners.
        full_ram_inputs_.at(image_idx) = FullRamInput();
        workspace_->ReleaseFullRamImageInputs(image_idx);
      }

      total_fused_points = 0;
      for (const auto& task_fused_points : task_fused_points_) {
        total_fused_points += task_fused_points.size();
      }
      LOG(INFO) << StringPrintf(
          " in %.3fs (%d points)", timer.ElapsedSeconds(), total_fused_points);
    }

    if (depth_pin_exit.workspace) {
      depth_pin_exit.workspace->ReleaseOriginalDepthNeighborhood();
    }
    std::fill(neighborhood_data_.begin(), neighborhood_data_.end(), nullptr);
    neighborhood_leases_.clear();

    fused_points_.reserve(total_fused_points);
    fused_points_visibility_.reserve(total_fused_points);
    for (size_t thread_id = 0; thread_id < task_fused_points_.size();
         ++thread_id) {
      const size_t num_task_points = task_fused_points_[thread_id].size();
      fused_points_.insert(fused_points_.end(),
                           task_fused_points_[thread_id].begin(),
                           task_fused_points_[thread_id].end());
      task_fused_points_[thread_id].clear();

      const size_t num_visibility = internal::MergeFusionVisibilityBlocks(
          task_fused_points_visibility_[thread_id], fused_points_visibility_);
      THROW_CHECK_EQ(num_visibility, num_task_points);
    }

    if (fused_points_.empty()) {
      LOG(WARNING)
          << "Could not fuse any points. This is likely caused by "
             "incorrect settings - filtering must be enabled for the last "
             "call to patch match stereo.";
    }

    LOG(INFO) << "Number of fused points: " << fused_points_.size();
    if (options_.bounded_cache_exact_depth) {
      const auto counts = GetDepthExactStateReadCounts();
      LOG(INFO) << (options_.bounded_cache_neighborhood
                        ? "Exact FP32 depth state with mapped neighborhoods: "
                          "root_reads="
                        : "Exact FP32 depth state: root_reads=")
                << counts.first << " candidate_reads=" << counts.second
                << " scalar_payload_bytes="
                << 4 * (counts.first + counts.second)
                << " scratch_payload_bytes=" << 4 * exact_depth_scratch_.size();
      const auto pins = GetDepthMappedNeighborhoodStats();
      LOG(INFO) << "Exact depth mapped neighborhoods: pin_enabled="
                << int(options_.bounded_cache_neighborhood)
                << " source_admissions=" << pins.source_admissions
                << " roots=" << pins.root_preparations
                << " complete=" << pins.complete_roots
                << " fallback=" << pins.fallback_roots
                << " cancelled=" << pins.cancelled_roots
                << " pin_acquires=" << pins.pin_acquires
                << " borrowed_reads=" << pins.borrowed_reads
                << " scalar_acquires=" << pins.scalar_acquires
                << " metadata_peak_bytes=" << pins.metadata_peak_bytes;
    }
    run_timer.PrintMinutes();
  } catch (...) {
    const auto failure = std::current_exception();
    // AddTask or a row callback may fail with earlier tasks still queued.
    // Join before releasing any component or mapped owner they can borrow.
    try {
      thread_pool.Wait();
    } catch (...) {
    }
    std::fill(neighborhood_data_.begin(), neighborhood_data_.end(), nullptr);
    neighborhood_leases_.clear();
    neighborhood_bitmap_pins_.reset();
    try {
      ClearMappedNormalRing();
    } catch (...) {
      mapped_normal_ring_.clear();
      mapped_normal_ring_index_.Clear();
    }
    ClearDirectComponentPins();
    std::rethrow_exception(failure);
  }
  neighborhood_bitmap_pins_.reset();
  ClearDirectComponentPins();
}

bool StereoFusion::PrepareDirectComponentPins() {
  ClearDirectComponentPins();
  if (!direct_mapped_normals_ || !bounded_workspace_) return false;
  THROW_CHECK_EQ(options_.num_threads, 1);

  using Component = BoundedWorkspace::Component;
  const size_t capacity = bounded_workspace_->GetCacheStats().capacity_bytes;
  size_t total_resident = 0;
  size_t largest_extra = 0;
  size_t used = 0;
  for (size_t index = 0; index < used_images_.size(); ++index) {
    if (!used_images_[index]) continue;
    ++used;
    for (const Component component : {Component::kDepth, Component::kBitmap}) {
      const int view = static_cast<int>(index);
      const size_t resident =
          bounded_workspace_->GetResidentBytesEstimate(view, component);
      const size_t reserved =
          bounded_workspace_->GetLoadReservationBytes(view, component);
      THROW_CHECK_GE(reserved, resident);
      if (resident > capacity - total_resident) return false;
      total_resident += resident;
      largest_extra = std::max(largest_extra, reserved - resident);
    }
  }
  if (used == 0 || largest_extra > capacity - total_resident) return false;

  // Build the complete owner set locally. A load failure destroys these
  // leases without exposing any borrowed pointer to fusion workers.
  std::vector<BoundedWorkspace::ImageLease> depth(used_images_.size());
  std::vector<BoundedWorkspace::ImageLease> bitmap(used_images_.size());
  std::vector<const BoundedWorkspace::ImageData*> depth_data(
      used_images_.size(), nullptr);
  std::vector<const BoundedWorkspace::ImageData*> bitmap_data(
      used_images_.size(), nullptr);
  for (size_t index = 0; index < used_images_.size(); ++index) {
    if (!used_images_[index]) continue;
    const int view = static_cast<int>(index);
    depth[index] =
        bounded_workspace_->AcquireComponent(view, Component::kDepth);
    bitmap[index] =
        bounded_workspace_->AcquireComponent(view, Component::kBitmap);
    depth_data[index] = depth[index].get();
    bitmap_data[index] = bitmap[index].get();
  }
  direct_depth_leases_ = std::move(depth);
  direct_bitmap_leases_ = std::move(bitmap);
  direct_depth_data_ = std::move(depth_data);
  direct_bitmap_data_ = std::move(bitmap_data);
  direct_component_pins_admitted_ = true;
  return true;
}

void StereoFusion::ClearDirectComponentPins() {
  // Borrowed pointers must disappear before their owners are released.
  direct_depth_data_.clear();
  direct_bitmap_data_.clear();
  direct_depth_leases_.clear();
  direct_bitmap_leases_.clear();
}

void StereoFusion::ClearNormalTileRings() {
  // All row tasks have joined before retained decoded pins are released.
  for (auto& ring : normal_tile_rings_) {
    std::fill(ring.last_slot_by_image.begin(),
              ring.last_slot_by_image.end(),
              ring.tiles.size());
    ring.index.clear();
    for (auto& slot : ring.tiles) {
      slot.lease.reset();
      slot.image_idx = -1;
    }
  }
}

void StereoFusion::ClearMappedNormalRing() {
  // Only the single joined row worker may have borrowed these owners.
  mapped_normal_ring_index_.Clear();
  for (auto& slot : mapped_normal_ring_) {
    slot.lease.reset();
    slot.image_idx = -1;
    slot.row_begin = slot.row_count = 0;
  }
}

const ImmutableMapRegion* StereoFusion::AcquireWorkerMappedNormal(
    const int image_idx, const int row) {
  THROW_CHECK_GE(row, 0);
  const size_t sample_row = static_cast<size_t>(row);
  const size_t band = sample_row / kMappedNormalStripeRows;
  const size_t found = mapped_normal_ring_index_.Find(image_idx, band);
  if (found < mapped_normal_ring_.size()) {
    const auto& slot = mapped_normal_ring_[found];
    THROW_CHECK_EQ(slot.image_idx, image_idx);
    THROW_CHECK(slot.lease);
    THROW_CHECK_LE(slot.row_begin, sample_row);
    THROW_CHECK_LT(sample_row - slot.row_begin, slot.row_count);
    return slot.lease.get();
  }
  THROW_CHECK(!mapped_normal_ring_.empty());
  const size_t slot_index = mapped_normal_next_;
  auto& slot = mapped_normal_ring_[slot_index];
  mapped_normal_next_ = (mapped_normal_next_ + 1) % mapped_normal_ring_.size();
  if (slot.lease) {
    THROW_CHECK(mapped_normal_ring_index_.Erase(
        slot.image_idx, slot.row_begin / kMappedNormalStripeRows, slot_index));
  }
  slot.lease.reset();  // Release victim before a potentially waiting acquire.
  slot.image_idx = -1;
  slot.row_begin = slot.row_count = 0;
  const auto grid = bounded_workspace_->GetNormalTileGrid(image_idx);
  THROW_CHECK_LT(sample_row, grid.second);
  const size_t begin =
      sample_row / kMappedNormalStripeRows * kMappedNormalStripeRows;
  const size_t count = std::min(kMappedNormalStripeRows, grid.second - begin);
  slot.lease =
      bounded_workspace_->AcquireMappedNormalWindow(image_idx, begin, count);
  slot.image_idx = image_idx;
  slot.row_begin = begin;
  slot.row_count = count;
  try {
    THROW_CHECK(mapped_normal_ring_index_.Insert(image_idx, band, slot_index));
  } catch (...) {
    slot.lease.reset();
    slot.image_idx = -1;
    slot.row_begin = slot.row_count = 0;
    throw;
  }
  return slot.lease.get();
}

const BoundedWorkspace::ImageData* StereoFusion::AcquireWorkerNormalTile(
    const int thread_id, const int image_idx, const int row, const int col) {
  auto& ring = normal_tile_rings_.at(thread_id);
  const auto contains = [row, col](const BoundedWorkspace::ImageData* data) {
    return data && static_cast<size_t>(col) >= data->normal_origin_x &&
           static_cast<size_t>(row) >= data->normal_origin_y &&
           static_cast<size_t>(col) - data->normal_origin_x <
               data->normal_map.GetWidth() &&
           static_cast<size_t>(row) - data->normal_origin_y <
               data->normal_map.GetHeight();
  };
  if (!ring.last_slot_by_image.empty()) {
    ++ring.hint_checks;
    const size_t hinted = ring.last_slot_by_image.at(image_idx);
    if (hinted < ring.tiles.size()) {
      const auto& slot = ring.tiles[hinted];
      if (slot.image_idx == image_idx && contains(slot.lease.get())) {
        ++ring.hits;
        return slot.lease.get();
      }
    }
  }
  NormalTileKey key{};
  if (ring.indexed) {
    ++ring.indexed_lookups;
    const size_t size = options_.bounded_cache_normal_tile_size;
    key = {image_idx,
           static_cast<size_t>(col) / size,
           static_cast<size_t>(row) / size};
    const auto found = ring.index.find(key);
    if (found != ring.index.end()) {
      ++ring.hits;
      if (!ring.last_slot_by_image.empty())
        ring.last_slot_by_image[image_idx] = found->second;
      return ring.tiles[found->second].lease.get();
    }
  } else {
    for (const auto& slot : ring.tiles) {
      if (slot.image_idx == image_idx && contains(slot.lease.get())) {
        ++ring.hits;
        return slot.lease.get();
      }
    }
  }
  ++ring.misses;
  const size_t slot_index = ring.next;
  ring.next = (ring.next + 1) % ring.tiles.size();
  auto& slot = ring.tiles[slot_index];
  if (!ring.last_slot_by_image.empty() && slot.image_idx >= 0 &&
      ring.last_slot_by_image[slot.image_idx] == slot_index) {
    ring.last_slot_by_image[slot.image_idx] = ring.tiles.size();
  }
  if (ring.indexed && slot.lease) ring.index.erase(slot.key);
  slot.lease.reset();  // Never wait for capacity while holding this slot's pin.
  slot.image_idx = -1;
  slot.lease = bounded_workspace_->AcquireNormalTile(
      image_idx, row, col, options_.bounded_cache_normal_tile_size);
  slot.image_idx = image_idx;
  slot.key = key;
  if (ring.indexed) {
    try {
      ring.index.emplace(key, slot_index);
    } catch (...) {
      slot.lease.reset();
      slot.image_idx = -1;
      throw;
    }
  }
  if (!ring.last_slot_by_image.empty())
    ring.last_slot_by_image[image_idx] = slot_index;
  return slot.lease.get();
}

const BoundedWorkspace::ImageData* StereoFusion::AcquireNeighborhoodBitmap(
    const int thread_id, const int image_idx) {
  if (!neighborhood_bitmap_pins_) return nullptr;
  auto& pin = neighborhood_bitmap_pins_[image_idx];
  if (!pin.admitted) return nullptr;
  auto* data = pin.data.load(std::memory_order_acquire);
  if (!data) {
    std::call_once(pin.once, [this, &pin, thread_id, image_idx] {
      pin.lease = bounded_workspace_->AcquireComponent(
          image_idx, BoundedWorkspace::Component::kBitmap);
      ++task_neighborhood_reads_[thread_id].rgb_init_loads;
      pin.data.store(pin.lease.get(), std::memory_order_release);
    });
    data = pin.data.load(std::memory_order_acquire);
  }
  ++task_neighborhood_reads_[thread_id].rgb_pointer_reads;
  return data;
}

void StereoFusion::PrepareBoundedNeighborhood(const int image_idx,
                                              const int num_threads,
                                              ThreadPool& thread_pool) {
  // The preceding root has joined. Clear borrowed pointers before releasing
  // their owners, and release all old pins before a new load may need room.
  std::fill(neighborhood_data_.begin(), neighborhood_data_.end(), nullptr);
  neighborhood_leases_.clear();
  auto previous_bitmap_pins = std::move(neighborhood_bitmap_pins_);
  if (CheckIfStopped()) return;

  const auto views = internal::FusionNeighborhood(overlapping_images_,
                                                  used_images_,
                                                  fused_images_,
                                                  image_idx,
                                                  options_.max_traversal_depth);
  if (views.empty()) return;
  if (options_.bounded_cache_exact_depth) {
    bounded_workspace_->PrepareOriginalDepthNeighborhood(
        views, [this] { return CheckIfStopped(); });
    if (CheckIfStopped()) return;
  }
  using Component = BoundedWorkspace::Component;
  const size_t capacity = bounded_workspace_->GetCacheStats().capacity_bytes;
  const size_t workers = static_cast<size_t>(num_threads);
  auto component =
      options_.bounded_cache_components ? Component::kDepth : Component::kImage;

  // A complete pinned whole-view set already contains every reachable
  // depth, normal and bitmap. Reserve loader scratch, not another whole view
  // per worker. Partial neighborhoods still need component demand headroom.
  if (options_.bounded_cache_components &&
      options_.bounded_cache_normal_tile_size == 0) {
    size_t whole_total = 0, whole_extra = 0;
    bool whole_fits = true;
    for (const int view : views) {
      const size_t resident =
          bounded_workspace_->GetResidentBytesEstimate(view, Component::kImage);
      const size_t reserved =
          bounded_workspace_->GetLoadReservationBytes(view, Component::kImage);
      THROW_CHECK_GE(reserved, resident);
      whole_extra = std::max(whole_extra, reserved - resident);
      if (resident > capacity - whole_total) {
        whole_fits = false;
        break;
      }
      whole_total += resident;
    }
    if (whole_fits && whole_extra <= capacity - whole_total) {
      component = Component::kImage;
    }
  }

  std::vector<size_t> resident_sizes;
  resident_sizes.reserve(views.size());
  size_t total = 0, largest = 0, largest_extra = 0;
  bool complete_fits = true;
  for (const int view : views) {
    const size_t reserved =
        bounded_workspace_->GetLoadReservationBytes(view, component);
    const size_t resident =
        bounded_workspace_->GetResidentBytesEstimate(view, component);
    THROW_CHECK_GE(reserved, resident);
    resident_sizes.push_back(resident);
    largest = std::max(largest, reserved);
    largest_extra = std::max(largest_extra, reserved - resident);
    if (options_.bounded_cache_components && component != Component::kImage) {
      largest = std::max(largest,
                         bounded_workspace_->GetLoadReservationBytes(
                             view, Component::kNormal));
      largest = std::max(largest,
                         bounded_workspace_->GetLoadReservationBytes(
                             view, Component::kBitmap));
    }
    if (resident > capacity - total) {
      complete_fits = false;
      total = capacity;
    } else {
      total += resident;
    }
  }
  complete_fits = complete_fits && largest_extra <= capacity - total;
  if (options_.bounded_cache_components && component != Component::kImage) {
    complete_fits = complete_fits && largest <= (capacity - total) / workers;
  }
  const size_t budget =
      complete_fits
          ? capacity
          : (largest <= capacity / workers ? capacity - largest * workers : 0);
  std::vector<int> prefill_views;
  if (options_.bounded_cache_normal_tile_size > 0) {
    // The checked Run admission already establishes workers * headroom <= cap.
    // Pin root depth + eventual RGB, then favor other depths; an excluded RGB
    // stays demand-loaded after the geometry checks.
    const size_t joint_budget =
        capacity - workers * normal_tile_worker_headroom_;
    std::vector<size_t> bitmap_sizes;
    bitmap_sizes.reserve(views.size());
    for (const int view : views) {
      bitmap_sizes.push_back(bounded_workspace_->GetResidentBytesEstimate(
          view, Component::kBitmap));
    }
    const auto admitted = internal::DepthPriorityFusionPins(
        resident_sizes, bitmap_sizes, joint_budget);
    for (const size_t index : admitted.depth) {
      prefill_views.push_back(views[index]);
    }
    neighborhood_bitmap_pins_ =
        std::make_unique<LazyBitmapPin[]>(used_images_.size());
    for (const size_t index : admitted.bitmap) {
      const int view = views[index];
      auto& pin = neighborhood_bitmap_pins_[view];
      pin.admitted = true;
      if (previous_bitmap_pins && previous_bitmap_pins[view].lease) {
        pin.lease = std::move(previous_bitmap_pins[view].lease);
        pin.data.store(pin.lease.get(), std::memory_order_release);
        ++neighborhood_bitmap_carried_views_;
      }
      ++neighborhood_bitmap_admitted_views_;
    }
  } else {
    size_t pinned_resident = 0;
    for (size_t index = 0; index < views.size(); ++index) {
      const size_t resident = resident_sizes[index];
      if (index == 0 && resident > budget) break;
      if (resident > budget - pinned_resident) continue;
      prefill_views.push_back(views[index]);
      pinned_resident += resident;
    }
  }
  // Excluded bitmap owners from the previous root must be gone before a new
  // prefill task reserves a component load.
  previous_bitmap_pins.reset();
  if (prefill_views.empty()) return;

  // Distinct slots are allocated before tasks start. Even if Wait rethrows,
  // member leases outlive the pool's worker join and no pointer is published.
  neighborhood_leases_.resize(prefill_views.size());
  const size_t prefill_threads =
      options_.bounded_cache_prefill_threads == -1
          ? workers
          : static_cast<size_t>(options_.bounded_cache_prefill_threads);
  const size_t lane_count = std::min(prefill_threads, prefill_views.size());
  max_bounded_prefill_lanes_scheduled_ =
      std::max(max_bounded_prefill_lanes_scheduled_, lane_count);
  std::vector<std::vector<std::pair<size_t, int>>> lanes(lane_count);
  for (size_t index = 0; index < prefill_views.size(); ++index) {
    lanes[index % lane_count].emplace_back(index, prefill_views[index]);
  }
  try {
    for (auto& lane : lanes) {
      thread_pool.AddTask([this, component, lane = std::move(lane)]() {
        for (const auto& [slot, view] : lane) {
          neighborhood_leases_[slot] =
              bounded_workspace_->AcquireComponent(view, component);
        }
      });
    }
    thread_pool.Wait();
  } catch (...) {
    // AddTask can fail after earlier lanes were queued. Join them before
    // releasing their member-owned leases or closing a resident reader scope.
    const auto failure = std::current_exception();
    try {
      thread_pool.Wait();
    } catch (...) {
    }
    neighborhood_leases_.clear();
    std::rethrow_exception(failure);
  }
  if (CheckIfStopped()) {
    neighborhood_leases_.clear();
    return;
  }
  for (size_t index = 0; index < prefill_views.size(); ++index) {
    neighborhood_data_[prefill_views[index]] =
        neighborhood_leases_[index].get();
  }
}

void StereoFusion::InitFusedPixelMask(int image_idx,
                                      size_t width,
                                      size_t height) {
  Bitmap mask;
  auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);
  const std::string mask_image_name =
      workspace_->GetModel().GetImageName(image_idx);
  auto mask_path = options_.mask_path / (mask_image_name + ".png");
  if (!ExistsFile(mask_path) && HasFileExtension(mask_image_name, ".png")) {
    mask_path = options_.mask_path / mask_image_name;
  }
  fused_pixel_mask = internal::FusionPixelMask(width, height);
  if (!options_.mask_path.empty() && ExistsFile(mask_path) &&
      mask.Read(mask_path, false)) {
    mask.Rescale(static_cast<int>(width),
                 static_cast<int>(height),
                 Bitmap::RescaleFilter::kBox);
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        const auto color = mask.GetPixel(col, row);
        fused_pixel_mask.Set(row, col, (!color || color->r == 0) ? 1 : 0);
      }
    }
  }
}

void StereoFusion::Fuse(const int thread_id,
                        const int image_idx,
                        const int row,
                        const int col) {
  struct Scratch {
    std::vector<FusionData> queue;
    std::vector<float> x, y, z, nx, ny, nz;
    std::vector<uint8_t> r, g, b;
    std::vector<int> visibility;
    bool in_use = false;

    void Clear() {
      queue.clear();
      x.clear();
      y.clear();
      z.clear();
      nx.clear();
      ny.clear();
      nz.clear();
      r.clear();
      g.clear();
      b.clear();
      visibility.clear();
    }

    size_t CapacityBytes() const {
      return queue.capacity() * sizeof(FusionData) +
             (x.capacity() + y.capacity() + z.capacity() + nx.capacity() +
              ny.capacity() + nz.capacity()) *
                 sizeof(float) +
             (r.capacity() + g.capacity() + b.capacity()) * sizeof(uint8_t) +
             visibility.capacity() * sizeof(int);
    }
  };
  thread_local Scratch worker_scratch;
  // A bitmap reader can reenter Fusion; never lend its caller's scratch.
  Scratch nested_scratch;
  auto& scratch = worker_scratch.in_use ? nested_scratch : worker_scratch;
  struct ScratchLease {
    Scratch& scratch;
    ~ScratchLease() {
      // Bound retained capacity without limiting the current traversal.
      if (scratch.CapacityBytes() > 256 * 1024) {
        scratch = Scratch();
      } else {
        scratch.in_use = false;
      }
    }
  } lease{scratch};
  scratch.in_use = true;
  scratch.Clear();

  // Next points to fuse.
  auto& fusion_queue = scratch.queue;
  fusion_queue.push_back({image_idx, row, col, 0});

  Eigen::Vector4f fused_ref_point = Eigen::Vector4f::Zero();
  Eigen::Vector3f fused_ref_normal = Eigen::Vector3f::Zero();
  // Points of different pixels of the currently point to be fused.
  auto& fused_point_x = scratch.x;
  auto& fused_point_y = scratch.y;
  auto& fused_point_z = scratch.z;
  auto& fused_point_nx = scratch.nx;
  auto& fused_point_ny = scratch.ny;
  auto& fused_point_nz = scratch.nz;
  auto& fused_point_r = scratch.r;
  auto& fused_point_g = scratch.g;
  auto& fused_point_b = scratch.b;
  auto& fused_point_visibility = scratch.visibility;

  // Select the immutable workspace mode once for this root. Specializing the
  // traversal lets eager FP32 inputs avoid fallback ownership and cache checks.
  const auto traverse = [&](auto resident_tag) {
    constexpr bool kResident = decltype(resident_tag)::value;
    while (!fusion_queue.empty()) {
      const auto data = fusion_queue.back();
      const int image_idx = data.image_idx;
      const int row = data.row;
      const int col = data.col;
      const int traversal_depth = data.traversal_depth;

      fusion_queue.pop_back();

      // Check if pixel already fused.
      auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);
      if (fused_pixel_mask.Get(row, col) > 0) {
        continue;
      }

      // Root pointers are borrowed only until the row-task join. Demand leases
      // keep one component pinned while its value is copied, then release it
      // before another component acquisition can wait for capacity.
      // Ordinary workspaces need no bounded-cache lease, even an empty one.
      std::optional<BoundedWorkspace::ImageLease> image_lease;
      std::shared_ptr<const DepthMap> depth_map;
      const BoundedWorkspace::ImageData* image_data = nullptr;
      const auto* resident =
          kResident ? &full_ram_inputs_.at(image_idx) : nullptr;
      float depth;
      if constexpr (kResident) {
        depth = resident->depth.Get(row, col);
      } else {
        const auto* pinned =
            direct_component_pins_admitted_ ? direct_depth_data_.at(image_idx)
            : bounded_workspace_ && options_.bounded_cache_neighborhood
                ? neighborhood_data_.at(image_idx)
                : nullptr;
        if (direct_component_pins_admitted_) ++direct_depth_borrows_;
        if (pinned && options_.bounded_cache_neighborhood) {
          ++task_neighborhood_reads_[thread_id].pointer_reads;
        }
        if (bounded_workspace_ && !pinned) {
          image_lease.emplace(bounded_workspace_->AcquireComponent(
              image_idx,
              options_.bounded_cache_components
                  ? BoundedWorkspace::Component::kDepth
                  : BoundedWorkspace::Component::kImage));
        }
        if (image_lease && *image_lease &&
            options_.bounded_cache_neighborhood) {
          ++task_neighborhood_reads_[thread_id].demand_acquires;
        }
        image_data = pinned        ? pinned
                     : image_lease ? image_lease->get()
                                   : nullptr;
        depth_map = image_data ? std::shared_ptr<const DepthMap>()
                               : workspace_->GetDepthMapLease(image_idx);
        const bool exact_depth = options_.bounded_cache_exact_depth;
        const bool exact_nonpositive =
            exact_depth && !image_data->HasPositiveDepth(row, col);
        depth = exact_depth ? 0.0f
                            : (image_data ? image_data->depth_map.Get(row, col)
                                          : depth_map->Get(row, col));
        if (options_.bounded_cache_components && image_lease && *image_lease) {
          image_lease.reset();
          image_data = nullptr;
        }

        if (exact_depth) {
          if (exact_nonpositive) continue;
          auto& scratch = exact_depth_scratch_.at(thread_id);
          depth = bounded_workspace_->ReadOriginalDepth(
              image_idx, row, col, scratch.scalar);
          THROW_CHECK_GT(depth, 0.0f)
              << "Original depth/positive mask disagree";
          if (traversal_depth == 0)
            ++scratch.root_reads;
          else
            ++scratch.candidate_reads;
        }
      }
      // Original FP32 now feeds every unchanged predicate and output operation.
      // Pixels with negative depth are filtered.
      if (depth <= 0.0f) {
        continue;
      }

      // If the traversal depth is greater than zero, the initial reference
      // pixel has already been added and we need to check for consistency.
      if (traversal_depth > 0) {
        // Project reference point into current view.
        const Eigen::Vector3f proj = P_.at(image_idx) * fused_ref_point;

        // Depth error of reference depth with current depth.
        const float depth_error = std::abs((proj(2) - depth) / depth);
        if (depth_error > options_.max_depth_error) {
          continue;
        }

        // Reprojection error reference point in the current view.
        const float col_diff = proj(0) / proj(2) - col;
        const float row_diff = proj(1) / proj(2) - row;
        const float squared_reproj_error =
            col_diff * col_diff + row_diff * row_diff;
        if (squared_reproj_error > max_squared_reproj_error_) {
          continue;
        }
      }

      // Determine normal direction in global reference frame.
      std::optional<BoundedWorkspace::ImageLease> normal_lease;
      std::shared_ptr<const NormalMap> normal_map;
      Eigen::Vector3f local_normal;
      if constexpr (kResident) {
        // Eager FP32 owners cannot use optional normal-source routes.
        local_normal = Eigen::Vector3f(resident->normal.Get(row, col, 0),
                                       resident->normal.Get(row, col, 1),
                                       resident->normal.Get(row, col, 2));
      } else {
        const auto* tile_image =
            options_.bounded_cache_normal_tile_size > 0
                ? AcquireWorkerNormalTile(thread_id, image_idx, row, col)
                : nullptr;
        const auto* mapped_normal =
            direct_mapped_normals_ ? AcquireWorkerMappedNormal(image_idx, row)
                                   : nullptr;
        if (bounded_workspace_ && options_.bounded_cache_components &&
            !tile_image && !mapped_normal &&
            (!image_data || image_data->normal_map.GetWidth() == 0)) {
          normal_lease.emplace(bounded_workspace_->AcquireComponent(
              image_idx, BoundedWorkspace::Component::kNormal));
        }
        const auto* normal_image = tile_image ? tile_image
                                   : normal_lease && *normal_lease
                                       ? normal_lease->get()
                                       : image_data;
        normal_map = normal_image || mapped_normal
                         ? std::shared_ptr<const NormalMap>()
                         : workspace_->GetNormalMapLease(image_idx);
        const auto* fp32_normal = normal_map.get();
        local_normal = mapped_normal ? mapped_normal->ReadFloat3(row, col)
                       : normal_image
                           ? normal_image->GetNormal(row, col)
                           : Eigen::Vector3f(fp32_normal->Get(row, col, 0),
                                             fp32_normal->Get(row, col, 1),
                                             fp32_normal->Get(row, col, 2));
      }
      const Eigen::Vector3f normal = inv_R_.at(image_idx) * local_normal;
      normal_lease.reset();

      // Check for consistent normal direction with reference normal.
      if (traversal_depth > 0) {
        const float cos_normal_error = fused_ref_normal.dot(normal);
        if (cos_normal_error < min_cos_normal_error_) continue;
      }

      // Determine 3D location of current depth value.
      const Eigen::Vector3f xyz =
          inv_P_.at(image_idx) *
          Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

      // Read the color of the pixel.
      const auto& bitmap_scale = bitmap_scales_.at(image_idx);
      const auto* borrowed_bitmap =
          kResident ? nullptr
          : options_.bounded_cache_neighborhood &&
                  options_.bounded_cache_normal_tile_size > 0
              ? AcquireNeighborhoodBitmap(thread_id, image_idx)
          : direct_component_pins_admitted_ ? direct_bitmap_data_.at(image_idx)
                                            : nullptr;
      if (borrowed_bitmap && direct_component_pins_admitted_) {
        ++direct_bitmap_borrows_;
      }
      std::optional<BoundedWorkspace::ImageLease> bitmap_lease;
      if (!kResident && bounded_workspace_ &&
          options_.bounded_cache_components && !borrowed_bitmap &&
          (!image_data || image_data->bitmap.Width() == 0)) {
        bitmap_lease.emplace(bounded_workspace_->AcquireComponent(
            image_idx, BoundedWorkspace::Component::kBitmap));
      }
      if (bitmap_lease && *bitmap_lease &&
          options_.bounded_cache_neighborhood &&
          options_.bounded_cache_normal_tile_size > 0) {
        ++task_neighborhood_reads_[thread_id].rgb_demand_acquires;
      }
      const auto* bitmap_image = borrowed_bitmap ? borrowed_bitmap
                                 : bitmap_lease && *bitmap_lease
                                     ? bitmap_lease->get()
                                     : image_data;
      const auto bitmap = bitmap_image || resident
                              ? std::shared_ptr<const Bitmap>()
                              : workspace_->GetBitmapLease(image_idx);
      const Bitmap& bitmap_data = bitmap_image ? bitmap_image->bitmap
                                  : resident   ? *resident->bitmap
                                               : *bitmap;
      // Unit-scale float coordinates preserve these integer pixel positions.
      constexpr int kMaxExactFloatInteger = 1 << 24;
      const bool resident_on_depth_grid =
          resident && bitmap_scale.first == 1.0f &&
          bitmap_scale.second == 1.0f && col >= 0 &&
          col <= kMaxExactFloatInteger && row >= 0 &&
          row <= kMaxExactFloatInteger;
      const auto color =
          ((bitmap_image && bitmap_image->bitmap_on_depth_grid) ||
           resident_on_depth_grid)
              ? bitmap_data.GetPixel(col, row).value_or(BitmapColor<uint8_t>(0))
              : bitmap_data
                    .InterpolateNearestNeighbor(col / bitmap_scale.first,
                                                row / bitmap_scale.second)
                    .value_or(BitmapColor<uint8_t>(0));
      bitmap_lease.reset();

      // Set the current pixel as visited.
      fused_pixel_mask.Set(row, col, 1);

      // Pixels out of bounds are filtered
      if (xyz(0) < options_.bounding_box.first(0) ||
          xyz(1) < options_.bounding_box.first(1) ||
          xyz(2) < options_.bounding_box.first(2) ||
          xyz(0) > options_.bounding_box.second(0) ||
          xyz(1) > options_.bounding_box.second(1) ||
          xyz(2) > options_.bounding_box.second(2)) {
        continue;
      }

      // Accumulate statistics for fused point.
      fused_point_x.push_back(xyz(0));
      fused_point_y.push_back(xyz(1));
      fused_point_z.push_back(xyz(2));
      fused_point_nx.push_back(normal(0));
      fused_point_ny.push_back(normal(1));
      fused_point_nz.push_back(normal(2));
      fused_point_r.push_back(color.r);
      fused_point_g.push_back(color.g);
      fused_point_b.push_back(color.b);
      fused_point_visibility.push_back(image_idx);

      // Remember the first pixel as the reference.
      if (traversal_depth == 0) {
        fused_ref_point = Eigen::Vector4f(xyz(0), xyz(1), xyz(2), 1.0f);
        fused_ref_normal = normal;
      }

      if (fused_point_x.size() >=
          static_cast<size_t>(options_.max_num_pixels)) {
        break;
      }

      if (traversal_depth >= options_.max_traversal_depth - 1) {
        continue;
      }

      for (const auto next_image_idx : overlapping_images_.at(image_idx)) {
        const Eigen::Vector3f next_proj =
            P_.at(next_image_idx) * xyz.homogeneous();
        const int next_col =
            static_cast<int>(std::round(next_proj(0) / next_proj(2)));
        const int next_row =
            static_cast<int>(std::round(next_proj(1) / next_proj(2)));

        const auto& depth_map_size = depth_map_sizes_.at(next_image_idx);
        if (next_col < 0 || next_row < 0 || next_col >= depth_map_size.first ||
            next_row >= depth_map_size.second) {
          continue;
        }
        fusion_queue.push_back(
            {next_image_idx, next_row, next_col, traversal_depth + 1});
      }
    }
  };
  if (full_ram_inputs_.empty()) {
    traverse(std::false_type{});
  } else {
    traverse(std::true_type{});
  }

  const size_t num_pixels = fused_point_x.size();
  if (num_pixels >= static_cast<size_t>(options_.min_num_pixels)) {
    PlyPoint fused_point;

    Eigen::Vector3f fused_normal;
    fused_normal.x() = Median(fused_point_nx);
    fused_normal.y() = Median(fused_point_ny);
    fused_normal.z() = Median(fused_point_nz);
    const float fused_normal_norm = fused_normal.norm();
    if (fused_normal_norm < std::numeric_limits<float>::epsilon()) {
      return;
    }

    fused_point.x = Median(fused_point_x);
    fused_point.y = Median(fused_point_y);
    fused_point.z = Median(fused_point_z);

    fused_point.nx = fused_normal.x() / fused_normal_norm;
    fused_point.ny = fused_normal.y() / fused_normal_norm;
    fused_point.nz = fused_normal.z() / fused_normal_norm;

    fused_point.r = RoundedByteMedian(fused_point_r);
    fused_point.g = RoundedByteMedian(fused_point_g);
    fused_point.b = RoundedByteMedian(fused_point_b);

    // Preserve every insertion, including duplicates, so the fresh table has
    // the same iteration order. Rejected roots need no visibility table.
    FlatHashSet<int> visibility;
    for (const int visible_image_idx : fused_point_visibility) {
      visibility.insert(visible_image_idx);
    }
    task_fused_points_[thread_id].push_back(fused_point);
    internal::AppendFusionVisibilityBlock(
        task_fused_points_visibility_[thread_id], visibility);
  }
}

void WritePointsVisibility(
    const std::filesystem::path& path,
    const std::vector<std::vector<int>>& points_visibility) {
  // Coalesce scalar writes; the buffer must outlive the stream's final flush.
  std::vector<char> file_buffer(256 * 1024);
  std::fstream file;
  file.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
  file.open(path, std::ios::out | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  WriteBinaryLittleEndian<uint64_t>(&file, points_visibility.size());

  // Coalesce the existing uint32 wire fields without per-scalar stream calls.
  std::array<uint32_t, 8192> payload;
  const bool native_little_endian = IsLittleEndian();
  size_t used = 0;
  const auto flush = [&] {
    file.write(reinterpret_cast<const char*>(payload.data()),
               static_cast<std::streamsize>(used * sizeof(uint32_t)));
    used = 0;
  };
  const auto append = [&](const uint32_t value) {
    payload[used++] = native_little_endian ? value : ReverseBytes(value);
    if (used == payload.size()) flush();
  };
  for (const auto& visibility : points_visibility) {
    append(static_cast<uint32_t>(visibility.size()));
    for (const auto& image_idx : visibility) {
      append(static_cast<uint32_t>(image_idx));
    }
  }
  if (used != 0) flush();
}

std::vector<std::vector<int>> ReadPointsVisibility(
    const std::filesystem::path& path, size_t num_points) {
  // Coalesce scalar reads; the buffer must outlive the stream.
  std::vector<char> file_buffer(256 * 1024);
  std::fstream file;
  file.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
  file.open(path, std::ios::in | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  uint64_t file_num_points = 0;
  file.read(reinterpret_cast<char*>(&file_num_points), sizeof(file_num_points));
  THROW_CHECK(file.good()) << "Unexpected end of visibility file header";
  file_num_points = LittleEndianToNative(file_num_points);
  THROW_CHECK_EQ(file_num_points, num_points);

  std::vector<std::vector<int>> visibility(num_points);
  std::array<uint32_t, 8192> buffer;
  const bool native_little_endian = IsLittleEndian();
  for (size_t i = 0; i < num_points; ++i) {
    uint32_t num_visible = 0;
    file.read(reinterpret_cast<char*>(&num_visible), sizeof(num_visible));
    THROW_CHECK(file.good())
        << "Unexpected end of visibility file at point " << i;
    if (!native_little_endian) {
      num_visible = LittleEndianToNative(num_visible);
    }
    visibility[i].resize(num_visible);
    // Read only this list's payload, including when the path is a pipe.
    for (size_t offset = 0; offset < num_visible;) {
      const size_t count =
          std::min(buffer.size(), static_cast<size_t>(num_visible) - offset);
      file.read(reinterpret_cast<char*>(buffer.data()),
                count * sizeof(uint32_t));
      THROW_CHECK(file.good())
          << "Unexpected end of visibility file at point " << i;
      for (size_t j = 0; j < count; ++j) {
        const uint32_t value =
            native_little_endian ? buffer[j] : LittleEndianToNative(buffer[j]);
        visibility[i][offset + j] = static_cast<int>(value);
      }
      offset += count;
    }
  }
  return visibility;
}

}  // namespace mvs
}  // namespace colmap
