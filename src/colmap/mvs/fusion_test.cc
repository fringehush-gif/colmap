// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/fusion.h"

#include "colmap/math/math.h"
#include "colmap/mvs/model.h"
#include "colmap/mvs/normal_precision_guard.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/bitmap_reader.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
TEST(StereoFusion, VisibilityBlocksPreserveListsAcrossBoundariesAndReuse) {
  const FlatHashSet<int> visibility = {9, -3, 4};
  const std::vector<int> expected(visibility.begin(), visibility.end());
  std::vector<std::vector<int>> blocks, output;
  constexpr size_t kBlockWords = 64 * 1024 / sizeof(int);
  const size_t records = kBlockWords / (visibility.size() + 1) + 1;
  for (size_t index = 0; index < records; ++index) {
    internal::AppendFusionVisibilityBlock(blocks, visibility);
  }
  ASSERT_EQ(blocks.size(), 2);
  EXPECT_EQ(blocks.front().size(), kBlockWords);
  EXPECT_EQ(internal::MergeFusionVisibilityBlocks(blocks, output), records);
  EXPECT_TRUE(blocks.empty());
  ASSERT_EQ(output.size(), records);
  for (const auto& values : output) EXPECT_EQ(values, expected);

  internal::AppendFusionVisibilityBlock(blocks, FlatHashSet<int>());
  EXPECT_EQ(internal::MergeFusionVisibilityBlocks(blocks, output), 1);
  EXPECT_TRUE(output.back().empty());
  EXPECT_EQ(internal::MergeFusionVisibilityBlocks(blocks, output), 0);

  FlatHashSet<int> large;
  for (size_t index = 0; index < kBlockWords + 1; ++index) {
    large.insert(static_cast<int>(index));
  }
  const std::vector<int> large_expected(large.begin(), large.end());
  internal::AppendFusionVisibilityBlock(blocks, large);
  internal::AppendFusionVisibilityBlock(blocks, visibility);
  ASSERT_EQ(blocks.size(), 2);
  EXPECT_EQ(blocks.front().size(), large.size() + 1);
  EXPECT_EQ(internal::MergeFusionVisibilityBlocks(blocks, output), 2);
  EXPECT_EQ(output[output.size() - 2], large_expected);
  EXPECT_EQ(output.back(), expected);
}

TEST(StereoFusion, VisibilityBlocksRejectTruncatedOrNegativeCounts) {
  for (const auto& malformed : std::vector<std::vector<int>>{{-1}, {2, 9}}) {
    std::vector<std::vector<int>> blocks = {malformed}, output;
    EXPECT_THROW(internal::MergeFusionVisibilityBlocks(blocks, output),
                 std::invalid_argument);
    EXPECT_TRUE(output.empty());
  }
}

TEST(StereoFusion, FloatMapViewPreservesValuesAndFlatBounds) {
  const std::array<float, 8> values = {0.0f,
                                       -0.0f,
                                       1.25f,
                                       -2.5f,
                                       std::numeric_limits<float>::denorm_min(),
                                       std::numeric_limits<float>::max(),
                                       std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::quiet_NaN()};
  const auto bits = [](float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  };
  for (size_t depth : {size_t{1}, size_t{2}, size_t{3}}) {
    Mat<float> map(7, 3, depth);
    for (size_t slice = 0; slice < depth; ++slice) {
      for (size_t row = 0; row < map.GetHeight(); ++row) {
        for (size_t col = 0; col < map.GetWidth(); ++col) {
          map.Set(row,
                  col,
                  slice,
                  values[(slice * 21 + row * 7 + col) % values.size()]);
        }
      }
    }
    const internal::FusionFloatMapView view(map);
    EXPECT_EQ(view.data, map.GetData().data());
    for (size_t slice = 0; slice < depth; ++slice) {
      for (size_t row = 0; row < map.GetHeight(); ++row) {
        for (size_t col = 0; col < map.GetWidth(); ++col) {
          EXPECT_EQ(bits(view.Get(row, col, slice)),
                    bits(map.Get(row, col, slice)));
        }
      }
    }
    // Mat::Get checks the flattened offset, including aliases between rows.
    EXPECT_EQ(bits(view.Get(0, 7)), bits(map.Get(0, 7)));
    if (depth > 1) {
      EXPECT_EQ(bits(view.Get(3, 0)), bits(map.Get(3, 0)));
    }
    EXPECT_THROW(view.Get(0, 0, depth), std::out_of_range);
    EXPECT_THROW(map.Get(0, 0, depth), std::out_of_range);
    EXPECT_THROW(view.Get(3 * depth, 0), std::out_of_range);
    EXPECT_THROW(map.Get(3 * depth, 0), std::out_of_range);
    map.Set(0, 0, 4.5f);
    EXPECT_EQ(view.Get(0, 0), 4.5f);
  }
  const internal::FusionFloatMapView empty;
  EXPECT_THROW(empty.Get(0, 0), std::out_of_range);
}

TEST(StereoFusion, AtomicPixelMaskInitializesBoundsAndMoves) {
  static_assert(
      std::is_nothrow_move_constructible_v<internal::FusionPixelMask>);
  static_assert(!std::is_copy_constructible_v<internal::FusionPixelMask>);
  internal::FusionPixelMask empty;
  EXPECT_TRUE(empty.Snapshot(0, 0).empty());
  EXPECT_THROW(empty.Get(0, 0), std::out_of_range);
  std::vector<internal::FusionPixelMask> masks(1);
  masks[0] = internal::FusionPixelMask(3, 2);
  EXPECT_EQ(masks[0].Snapshot(0, 6), std::vector<char>(6, 0));
  masks[0].Set(0, 1, 1);
  masks[0].Set(1, 2, 1);
  masks.resize(4);
  EXPECT_EQ(masks[0].Get(0, 1), 1);
  EXPECT_EQ(masks[0].Get(1, 0), 0);
  EXPECT_EQ(masks[0].Snapshot(1, 6), (std::vector<char>{1, 0, 0, 0, 1}));
  EXPECT_TRUE(masks[0].Snapshot(6, 6).empty());
  EXPECT_THROW(masks[0].Snapshot(4, 3), std::out_of_range);
  EXPECT_THROW(masks[0].Snapshot(2, 7), std::out_of_range);
  EXPECT_THROW(masks[0].Set(2, 0, 1), std::out_of_range);
  EXPECT_THROW(masks[0].Get(2, 0), std::out_of_range);
}

TEST(StereoFusion, PackedAtomicPixelMaskPreservesTailsAliasesAndMoves) {
  constexpr size_t kWidth = 13, kHeight = 7, kPixels = kWidth * kHeight;
  internal::FusionPixelMask mask(kWidth, kHeight);
  std::vector<char> expected(kPixels, 0);
  for (const size_t index :
       {size_t{0}, size_t{12}, size_t{63}, size_t{64}, kPixels - 1}) {
    mask.Set(index / kWidth, index % kWidth, 1);
    expected[index] = 1;
  }
  EXPECT_EQ(mask.Snapshot(0, kPixels), expected);
  EXPECT_EQ(mask.Snapshot(61, 67), (std::vector<char>{0, 0, 1, 1, 0, 0}));
  EXPECT_EQ(mask.Snapshot(kPixels - 2, kPixels), (std::vector<char>{0, 1}));
  EXPECT_TRUE(mask.Snapshot(kPixels, kPixels).empty());
  EXPECT_THROW(mask.Snapshot(kPixels + 1, kPixels + 1), std::out_of_range);
  EXPECT_THROW(mask.Snapshot(64, 63), std::out_of_range);
  EXPECT_THROW(mask.Get(kHeight, 0), std::out_of_range);
  EXPECT_THROW(mask.Get(0, 127), std::out_of_range);
  EXPECT_THROW(mask.Set(0, 127, 0), std::out_of_range);
  mask.Set(0, kWidth, 1);  // Same flattened offset as (1, 0).
  expected[kWidth] = 1;
  EXPECT_EQ(mask.Get(1, 0), 1);
  const size_t wrapped_row = std::numeric_limits<size_t>::max();
  EXPECT_EQ(mask.Get(wrapped_row, kWidth), mask.Get(0, 0));
  EXPECT_THROW(mask.Get(wrapped_row, kWidth - 1), std::out_of_range);
  mask.Set(wrapped_row, kWidth, 0);
  expected[0] = 0;
  EXPECT_EQ(mask.Snapshot(0, kPixels), expected);
  EXPECT_THROW(mask.Set(0, 0, 2), std::invalid_argument);
  EXPECT_EQ(mask.Snapshot(0, kPixels), expected);

  internal::FusionPixelMask moved(std::move(mask));
  EXPECT_EQ(moved.Snapshot(0, kPixels), expected);
  EXPECT_TRUE(mask.Snapshot(0, 0).empty());
  EXPECT_THROW(mask.Snapshot(1, 1), std::out_of_range);
  EXPECT_THROW(mask.Get(0, 0), std::out_of_range);
  internal::FusionPixelMask assigned(1, 1);
  assigned = std::move(moved);
  EXPECT_EQ(assigned.Snapshot(0, kPixels), expected);
  EXPECT_TRUE(moved.Snapshot(0, 0).empty());
  EXPECT_THROW(moved.Snapshot(1, 1), std::out_of_range);
  EXPECT_THROW(moved.Set(0, 0, 1), std::out_of_range);
  internal::FusionPixelMask zero_height(std::numeric_limits<size_t>::max(), 0);
  EXPECT_TRUE(zero_height.Snapshot(0, 0).empty());
  EXPECT_THROW(zero_height.Get(0, 0), std::out_of_range);
}

TEST(StereoFusion, PackedAtomicPixelMaskKeepsDisjointConcurrentBits) {
  constexpr size_t kRows = 128, kBits = 64;
  internal::FusionPixelMask mask(kBits, kRows);
  std::atomic<size_t> ready{0};
  const auto write = [&](size_t parity) {
    for (size_t row = 0; row < kRows; ++row) {
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (ready.load(std::memory_order_acquire) < 2 * (row + 1))
        std::this_thread::yield();
      // Both workers target distinct bits of the same atomic word.
      for (size_t col = parity; col < kBits; col += 2) mask.Set(row, col, 1);
    }
  };
  std::thread first(write, 0), second(write, 1);
  first.join();
  second.join();
  const auto snapshot = mask.Snapshot(0, kRows * kBits);
  EXPECT_TRUE(std::all_of(
      snapshot.begin(), snapshot.end(), [](char value) { return value == 1; }));
  mask.Set(17, 31, 0);
  for (size_t col = 0; col < kBits; ++col)
    EXPECT_EQ(mask.Get(17, col), col == 31 ? 0 : 1);
  mask.Set(17, 31, 1);
  EXPECT_EQ(mask.Snapshot(17 * kBits + 29, 17 * kBits + 34),
            std::vector<char>(5, 1));
}

TEST(StereoFusion, AtomicPixelMaskAllowsConcurrentStoresAndSnapshots) {
  constexpr size_t kSide = 64;
  internal::FusionPixelMask mask(kSide, kSide);
  std::atomic<bool> start{false};
  const auto write = [&] {
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    for (int repeat = 0; repeat < 8; ++repeat) {
      for (size_t row = 0; row < kSide; ++row) {
        for (size_t col = 0; col < kSide; ++col) mask.Set(row, col, 1);
      }
    }
  };
  std::thread first(write);
  std::thread second(write);
  start.store(true, std::memory_order_release);
  for (int repeat = 0; repeat < 8; ++repeat) {
    const auto snapshot = mask.Snapshot(0, kSide * kSide);
    EXPECT_TRUE(std::all_of(snapshot.begin(), snapshot.end(), [](char value) {
      return value == 0 || value == 1;
    }));
    const char value = mask.Get(kSide / 2, kSide / 2);
    EXPECT_TRUE(value == 0 || value == 1);
  }
  first.join();
  second.join();
  const auto snapshot = mask.Snapshot(0, kSide * kSide);
  EXPECT_TRUE(std::all_of(
      snapshot.begin(), snapshot.end(), [](char value) { return value == 1; }));
}

// Inspect ring ownership through the Fusion test peer.
struct StereoFusionExactDepthTestPeer {
  static bool FullRamInputsMatchWorkspace(StereoFusion& fusion) {
    if (fusion.full_ram_inputs_.size() != fusion.used_images_.size())
      return false;
    for (size_t index = 0; index < fusion.used_images_.size(); ++index) {
      const auto& input = fusion.full_ram_inputs_[index];
      if (!fusion.used_images_[index] || fusion.fused_images_[index]) {
        if (input.depth.data || input.normal.data || input.bitmap) return false;
        continue;
      }
      const int view = static_cast<int>(index);
      const auto matches = [](const internal::FusionFloatMapView& input,
                              const Mat<float>& map) {
        return input.data == map.GetData().data() &&
               input.size == map.GetData().size() &&
               input.width == map.GetWidth() &&
               input.plane == map.GetWidth() * map.GetHeight();
      };
      if (!matches(input.depth, fusion.workspace_->GetDepthMap(view)) ||
          !matches(input.normal, fusion.workspace_->GetNormalMap(view)) ||
          input.bitmap != &fusion.workspace_->GetBitmap(view))
        return false;
    }
    return !fusion.full_ram_inputs_.empty();
  }
  static size_t NumRetiredFullRamInputs(StereoFusion& fusion) {
    size_t count = 0;
    for (size_t index = 0; index < fusion.full_ram_inputs_.size(); ++index) {
      if (!fusion.used_images_[index] || !fusion.fused_images_[index]) continue;
      const auto& input = fusion.full_ram_inputs_[index];
      if (input.depth.data || input.normal.data || input.bitmap) return 0;
      const auto absent = [](auto getter) {
        try {
          getter();
          return false;
        } catch (const std::invalid_argument&) {
          return true;
        }
      };
      const int view = static_cast<int>(index);
      if (!absent([&] { fusion.workspace_->GetBitmap(view); }) ||
          !absent([&] { fusion.workspace_->GetDepthMap(view); }) ||
          !absent([&] { fusion.workspace_->GetNormalMap(view); }))
        return 0;
      ++count;
    }
    return count;
  }

  static bool HasFullRamInputs(const StereoFusion& fusion) {
    return !fusion.full_ram_inputs_.empty();
  }
  static void PrepareRing(StereoFusion& fusion,
                          const Workspace::Options& options,
                          size_t slots,
                          bool hints) {
    fusion.workspace_ = std::make_unique<BoundedWorkspace>(options);
    fusion.bounded_workspace_ =
        static_cast<BoundedWorkspace*>(fusion.workspace_.get());
    fusion.normal_tile_rings_.resize(1);
    auto& ring = fusion.normal_tile_rings_.front();
    ring.tiles.resize(slots);
    ring.indexed = slots > 4;
    if (ring.indexed) ring.index.reserve(slots);
    if (ring.indexed && hints) ring.last_slot_by_image.assign(1, slots);
  }
  static const BoundedWorkspace::ImageData* Acquire(StereoFusion& fusion,
                                                    int row,
                                                    int col) {
    return fusion.AcquireWorkerNormalTile(0, 0, row, col);
  }
};

namespace {

TEST(StereoFusion, NormalTileRingReusesSnormAndFp32CroppedTiles) {
  if (!ImmutableMapRegion::Supported())
    GTEST_SKIP() << "Immutable FP32 map owners are unsupported on this host";
  const auto root = CreateTestDir();
  for (const auto& sub : {"sparse",
                          "images",
                          "stereo",
                          "stereo/depth_maps",
                          "stereo/normal_maps"}) {
    CreateDirIfNotExists(root / sub);
  }
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 1;
  dataset.camera_width = 17;
  dataset.camera_height = 11;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  Workspace::Options base;
  base.workspace_path = root;
  base.workspace_format = "COLMAP";
  base.input_type = "geometric";
  base.cache_size = 65536.0 / (1024.0 * 1024.0 * 1024.0);
  base.mapped_normal_bytes_limit = 1 << 20;
  base.mapped_normal_files_limit = 1;
  Workspace descriptor(base);
  Mat<float> depth(17, 11, 1);
  depth.Fill(5.0f);
  depth.Write(descriptor.GetDepthMapPath(0));
  Mat<float> normal(17, 11, 3);
  normal.Fill(0.0f);
  for (int row = 0; row < 11; ++row) {
    for (int col = 0; col < 17; ++col) normal.Set(row, col, 2, 1.0f);
  }
  normal.Write(descriptor.GetNormalMapPath(0));
  Bitmap bitmap(17, 11, true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 64, 128));
  ASSERT_TRUE(bitmap.Write(descriptor.GetBitmapPath(0)));
  // A/B alternate while resident; C and D cross both interior boundaries and
  // the cropped right/bottom edges. A is revisited after ring replacement.
  const std::vector<std::pair<int, int>> samples = {{0, 0},
                                                    {7, 7},
                                                    {8, 16},
                                                    {10, 16},
                                                    {1, 1},
                                                    {9, 16},
                                                    {8, 15},
                                                    {8, 16},
                                                    {7, 16},
                                                    {10, 15},
                                                    {7, 16},
                                                    {0, 0}};
  for (bool snorm : {false, true}) {
    for (int slots : {2, 4, 8}) {
      for (bool hints : {false, true}) {
        if (slots <= 4 && !hints) continue;  // No hint array in linear rings.
        SCOPED_TRACE(::testing::Message() << "snorm=" << snorm << " slots="
                                          << slots << " hints=" << hints);
        auto workspace_options = base;
        workspace_options.normal_tiles_snorm16 = snorm;
        StereoFusionOptions options;
        options.num_threads = 1;
        options.cache_size = base.cache_size;
        options.bounded_cache = true;
        options.bounded_cache_components = true;
        options.bounded_cache_normal_tile_size = 8;
        options.bounded_cache_normal_tile_slots = slots;
        options.bounded_cache_normal_tile_hints = hints;
        options.bounded_cache_normal_snorm16 = snorm;
        options.bounded_cache_normal_snorm16_guard = false;
        options.bounded_cache_normal_mapped_bytes = 1 << 20;
        options.bounded_cache_normal_mapped_files = 1;
        ASSERT_TRUE(options.Check());
        StereoFusion fusion(options, root, "COLMAP", "", "geometric");
        StereoFusionExactDepthTestPeer::PrepareRing(
            fusion, workspace_options, slots, hints);
        for (const auto& [row, col] : samples) {
          const auto* tile =
              StereoFusionExactDepthTestPeer::Acquire(fusion, row, col);
          ASSERT_NE(tile, nullptr);
          const size_t x = static_cast<size_t>(col) / 8 * 8;
          const size_t y = static_cast<size_t>(row) / 8 * 8;
          EXPECT_EQ(tile->normal_origin_x, x);
          EXPECT_EQ(tile->normal_origin_y, y);
          const size_t width = std::min(size_t{8}, 17 - x);
          const size_t height = std::min(size_t{8}, 11 - y);
          EXPECT_EQ(tile->snorm_normals.empty(), !snorm);
          if (snorm) {
            EXPECT_EQ(tile->snorm_width, width);
            EXPECT_EQ(tile->snorm_height, height);
            EXPECT_EQ(tile->normal_map.GetWidth(), 0);
          } else {
            EXPECT_EQ(tile->normal_map.GetWidth(), width);
            EXPECT_EQ(tile->normal_map.GetHeight(), height);
          }
          EXPECT_EQ(tile->GetNormal(row, col), Eigen::Vector3f(0, 0, 1));
        }
        const auto ring = fusion.GetNormalTileRingStats();
        const auto cache = fusion.GetBoundedCacheStats();
        ASSERT_TRUE(cache);
        const size_t expected_misses = slots == 2 ? 5 : 4;
        EXPECT_EQ(ring.hits, samples.size() - expected_misses);
        EXPECT_EQ(ring.misses, expected_misses);
        EXPECT_EQ(ring.hits + ring.misses, samples.size());
        EXPECT_EQ(cache->normal_tile_loads, 4);
        EXPECT_EQ(cache->normal_tile_hits, expected_misses - 4);
        EXPECT_EQ(cache->normal_tile_loads + cache->normal_tile_hits,
                  ring.misses);
        EXPECT_EQ(cache->reserved_bytes, 0);
        if (slots <= 4) {
          EXPECT_EQ(ring.indexed_lookups, 0);
          EXPECT_EQ(ring.hint_checks, 0);
        } else if (hints) {
          EXPECT_EQ(ring.hint_checks, samples.size());
          EXPECT_LT(ring.indexed_lookups, samples.size());
        } else {
          EXPECT_EQ(ring.hint_checks, 0);
          EXPECT_EQ(ring.indexed_lookups, samples.size());
        }
        std::cout << "NORMAL_RING_PROOF snorm=" << snorm << " slots=" << slots
                  << " hints=" << hints << " requests=" << samples.size()
                  << " hits=" << ring.hits << " misses=" << ring.misses
                  << " loads=" << cache->normal_tile_loads
                  << " cache_hits=" << cache->normal_tile_hits
                  << " indexed=" << ring.indexed_lookups
                  << " hint_checks=" << ring.hint_checks << '\n';
      }
    }
  }
}

TEST(StereoFusionOptions, FullRamLoadThreadPolicy) {
  StereoFusionOptions options;
  EXPECT_TRUE(options.Check());
  options.full_ram_load_threads = 0;
  EXPECT_FALSE(options.Check());
  options.full_ram_load_threads = 2;
  EXPECT_TRUE(options.Check());
  options.bounded_cache = true;
  EXPECT_FALSE(options.Check());
  options.bounded_cache = false;
  options.use_cache = true;
  EXPECT_FALSE(options.Check());
  options.use_cache = false;
  options.full_ram_load_threads = -2;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusionOptions, BoundedComponentPolicy) {
  StereoFusionOptions options;
  options.bounded_cache_components = true;
  EXPECT_FALSE(options.Check());
  options.bounded_cache_components = false;
  options.bounded_cache_neighborhood = true;
  EXPECT_FALSE(options.Check());
  options.bounded_cache = true;
  EXPECT_TRUE(options.Check());
  options.bounded_cache_components = true;
  EXPECT_TRUE(options.Check());

  options.bounded_cache_prefill_threads = 0;
  EXPECT_FALSE(options.Check());
  options.bounded_cache_prefill_threads = -2;
  EXPECT_FALSE(options.Check());
  options.bounded_cache_prefill_threads = 2;
  EXPECT_TRUE(options.Check());
  options.bounded_cache_neighborhood = false;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusionOptions, BoundedDiagnosticPolicy) {
  StereoFusionOptions options;
  EXPECT_FALSE(options.bounded_cache_diagnostics);
  options.bounded_cache_diagnostics = true;
  EXPECT_FALSE(options.Check());
  options.bounded_cache = true;
  EXPECT_TRUE(options.Check());
}

TEST(StereoFusion, MappedNormalRingIndexReplacementAndClear) {
  internal::MappedNormalRingIndex index;
  index.Reset(3, 2);
  EXPECT_EQ(index.Find(0, 0), 2);
  ASSERT_TRUE(index.Insert(0, 0, 0));
  ASSERT_TRUE(index.Insert(0, 1, 1));
  // The last-image hint changes bands, while the older stripe remains indexed.
  EXPECT_EQ(index.Find(0, 0), 0);
  EXPECT_EQ(index.Find(0, 1), 1);
  EXPECT_FALSE(index.Insert(0, 1, 0));
  EXPECT_FALSE(index.Insert(1, 4, 0));  // Metadata cannot exceed ring slots.
  EXPECT_FALSE(
      index.Erase(0, 0, 1));  // Stale victim cannot erase another slot.
  EXPECT_EQ(index.Find(0, 0), 0);
  ASSERT_TRUE(index.Erase(0, 0, 0));
  EXPECT_EQ(index.Find(0, 0), 2);
  ASSERT_TRUE(index.Insert(1, 4, 0));
  EXPECT_EQ(index.Find(1, 4), 0);
  // Reusing a victim slot must never return the old image's owner.
  ASSERT_TRUE(index.Erase(0, 1, 1));
  ASSERT_TRUE(index.Insert(2, 7, 1));
  EXPECT_EQ(index.Find(0, 1), 2);
  EXPECT_EQ(index.Find(2, 7), 1);
  index.Clear();
  EXPECT_EQ(index.Find(1, 4), 2);
  EXPECT_EQ(index.Find(2, 7), 2);
  ASSERT_TRUE(index.Insert(0, 1, 0));
  EXPECT_EQ(index.Find(0, 1), 0);
}

TEST(StereoFusion, DepthPriorityFusionPinsBudgetBoundaries) {
  using internal::DepthPriorityFusionPins;
  EXPECT_TRUE(DepthPriorityFusionPins({}, {}, 10).depth.empty());
  EXPECT_TRUE(DepthPriorityFusionPins({8}, {3}, 10).bitmap.empty());
  const auto exact = DepthPriorityFusionPins({8, 2}, {3, 1}, 11);
  EXPECT_EQ(exact.depth, std::vector<size_t>({0}));
  EXPECT_EQ(exact.bitmap, std::vector<size_t>({0}));
  const auto priority = DepthPriorityFusionPins({4, 9, 3, 2}, {2, 5, 1, 2}, 12);
  EXPECT_EQ(priority.depth, std::vector<size_t>({0, 2, 3}));
  EXPECT_EQ(priority.bitmap, std::vector<size_t>({0, 2}));
  const auto overflow =
      DepthPriorityFusionPins({std::numeric_limits<size_t>::max(), 1},
                              {1, 1},
                              std::numeric_limits<size_t>::max());
  EXPECT_TRUE(overflow.depth.empty());
  EXPECT_TRUE(overflow.bitmap.empty());
}

TEST(StereoFusion, IndependentPixelsPreserveCompleteWorkerOutput) {
  const auto root = CreateTestDir();
  for (const auto& sub : {"sparse",
                          "images",
                          "stereo",
                          "stereo/depth_maps",
                          "stereo/normal_maps"}) {
    CreateDirIfNotExists(root / sub);
  }
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 1;
  dataset.camera_width = 30;
  dataset.camera_height = 40;
  dataset.camera_params = {30.0, 15.0, 20.0, 0.0};
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  ASSERT_EQ(reconstruction.Images().size(), 1);
  const auto name = reconstruction.Images().begin()->second.Name();
  Mat<float> depth(30, 40, 1);
  depth.Fill(5.0f);
  depth.Write(root / "stereo/depth_maps" / (name + ".geometric.bin"));
  Mat<float> normal(30, 40, 3);
  normal.Fill(0.0f);
  for (int row = 0; row < 40; ++row) {
    for (int col = 0; col < 30; ++col) normal.Set(row, col, 2, 1.0f);
  }
  normal.Write(root / "stereo/normal_maps" / (name + ".geometric.bin"));
  Bitmap image(30, 40, true);
  for (int row = 0; row < 40; ++row) {
    for (int col = 0; col < 30; ++col) {
      image.SetPixel(
          col, row, BitmapColor<uint8_t>(col * 7, row * 5, col + row));
    }
  }
  ASSERT_TRUE(image.Write(root / "images" / name));
  {
    std::ofstream config(root / "stereo/fusion.cfg");
    config << name << '\n';
  }
  Model model;
  model.Read(root, "COLMAP");
  const auto overlapping = model.GetMaxOverlappingImages(10, 0.0);
  ASSERT_EQ(model.images.size(), 1);
  ASSERT_EQ(overlapping.size(), 1);
  ASSERT_TRUE(overlapping.front().empty());

  // Include every serialized attribute and its visibility association. Bits
  // distinguish signed zero as well; task completion order is irrelevant.
  using Record = std::tuple<uint32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t,
                            uint8_t,
                            uint8_t,
                            uint8_t,
                            std::vector<int>>;
  const auto bits = [](float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  };
  const auto capture = [&](const StereoFusion& fusion) {
    std::vector<Record> records;
    const auto& points = fusion.GetFusedPoints();
    const auto& visibility = fusion.GetFusedPointsVisibility();
    records.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      const auto& p = points[i];
      EXPECT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) &&
                  std::isfinite(p.z) && std::isfinite(p.nx) &&
                  std::isfinite(p.ny) && std::isfinite(p.nz));
      auto visible = visibility.at(i);
      EXPECT_EQ(visible, std::vector<int>{0});
      records.emplace_back(bits(p.x),
                           bits(p.y),
                           bits(p.z),
                           bits(p.nx),
                           bits(p.ny),
                           bits(p.nz),
                           p.r,
                           p.g,
                           p.b,
                           std::move(visible));
    }
    std::sort(records.begin(), records.end());
    return records;
  };
  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.num_threads = 1;
  StereoFusion serial(options, root, "COLMAP", "", "geometric");
  serial.Run();
  ASSERT_EQ(serial.GetFusedPoints().size(), 30 * 40);
  ASSERT_EQ(serial.GetFusedPointsVisibility().size(), 30 * 40);
  const auto expected = capture(serial);
  // Cached inputs retain nearest-neighbor sampling rather than the resident
  // unit-grid shortcut. Compare all pixel attributes and visibility exactly.
  options.use_cache = true;
  StereoFusion cached(options, root, "COLMAP", "", "geometric");
  cached.Run();
  EXPECT_EQ(capture(cached), expected);
  options.use_cache = false;
  options.num_threads = 2;
  StereoFusion parallel(options, root, "COLMAP", "", "geometric");
  for (int run = 0; run < 2; ++run) {
    parallel.Run();
    ASSERT_EQ(parallel.GetFusedPoints().size(), expected.size());
    ASSERT_EQ(parallel.GetFusedPointsVisibility().size(), expected.size());
    EXPECT_EQ(capture(parallel), expected);
    EXPECT_TRUE(
        StereoFusionExactDepthTestPeer::FullRamInputsMatchWorkspace(parallel));
    EXPECT_EQ(StereoFusionExactDepthTestPeer::NumRetiredFullRamInputs(parallel),
              1);
  }

  // All row bands execute, but every pixel is skipped before Fuse. The
  // unconditional task-entry worker lookup must remain valid in this case.
  const auto masks = root / "masks";
  CreateDirIfNotExists(masks);
  Bitmap mask(30, 40, true);
  mask.Fill(BitmapColor<uint8_t>(0, 0, 0));
  ASSERT_TRUE(mask.Write(masks / (name + ".png")));
  options.mask_path = masks;
  StereoFusion masked(options, root, "COLMAP", "", "geometric");
  masked.Run();
  EXPECT_TRUE(masked.GetFusedPoints().empty());
  EXPECT_TRUE(masked.GetFusedPointsVisibility().empty());
}

TEST(StereoFusion, Integration) {
  const auto temp_dir = CreateTestDir();
  CreateDirIfNotExists(temp_dir / "sparse");
  CreateDirIfNotExists(temp_dir / "images");
  CreateDirIfNotExists(temp_dir / "stereo");
  CreateDirIfNotExists(temp_dir / "stereo" / "depth_maps");
  CreateDirIfNotExists(temp_dir / "stereo" / "normal_maps");

  // Create synthetic reconstruction with 2 overlapping images.
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.camera_width = 30;
  synthetic_dataset_options.camera_height = 20;
  // Center the principal point in this small image so both frames observe
  // the synthetic points and form a real two-view fusion neighborhood.
  synthetic_dataset_options.camera_params = {30.0, 15.0, 10.0, 0.0};
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  reconstruction.Write(temp_dir / "sparse");

  // Create depth maps, normal maps, and consistency graphs for both images.
  std::vector<std::string> image_names;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    image_names.push_back(image.Name());

    // Create depth map with constant depth.
    Mat<float> depth_map(synthetic_dataset_options.camera_width,
                         synthetic_dataset_options.camera_height,
                         1);
    depth_map.Fill(5.0f);
    depth_map.Write(temp_dir / "stereo" / "depth_maps" /
                    (image.Name() + ".geometric.bin"));

    // Create normal map pointing in z direction.
    Mat<float> normal_map(synthetic_dataset_options.camera_width,
                          synthetic_dataset_options.camera_height,
                          3);
    const size_t num_pixels = normal_map.GetHeight() * normal_map.GetWidth();
    for (size_t i = 0; i < num_pixels; ++i) {
      normal_map.GetPtr()[3 * i + 0] = 0.0f;  // nx
      normal_map.GetPtr()[3 * i + 1] = 0.0f;  // ny
      normal_map.GetPtr()[3 * i + 2] = 1.0f;  // nz
    }
    normal_map.Write(temp_dir / "stereo" / "normal_maps" /
                     (image.Name() + ".geometric.bin"));

    // Create bitmap.
    Bitmap bitmap(synthetic_dataset_options.camera_width,
                  synthetic_dataset_options.camera_height,
                  true);
    bitmap.Fill(BitmapColor<uint8_t>(0, 64, 128));
    bitmap.Write(temp_dir / "images" / image.Name());
  }

  // Write fusion config
  std::ofstream fusion_cfg(temp_dir / "stereo" / "fusion.cfg");
  for (const auto& name : image_names) {
    fusion_cfg << name << "\n";
  }
  fusion_cfg.close();

  // The prefill gate below requires two actual neighboring views. Keep the
  // sparse model, rather than just fusion.cfg, responsible for that graph.
  Model model;
  model.Read(temp_dir, "COLMAP");
  ASSERT_EQ(model.images.size(), 2);
  const auto overlapping = model.GetMaxOverlappingImages(10, 0.0);
  ASSERT_EQ(overlapping.size(), 2);
  ASSERT_EQ(overlapping[0], std::vector<int>{1});
  ASSERT_EQ(overlapping[1], std::vector<int>{0});

  // Run fusion
  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.max_traversal_depth = 10;
  options.check_num_images = 10;
  options.use_cache = false;
  options.num_threads = 1;
  EXPECT_FALSE(options.bounded_cache);

  StereoFusion fusion(options, temp_dir, "COLMAP", "", "geometric");
  fusion.Run();
  EXPECT_FALSE(fusion.GetBoundedCacheDiagnosticRow());

  // Verify that some points were fused
  const auto& fused_points = fusion.GetFusedPoints();
  const auto& visibility = fusion.GetFusedPointsVisibility();

  EXPECT_GT(fused_points.size(), 0);
  EXPECT_EQ(fused_points.size(), visibility.size());

  for (const auto& point : fused_points) {
    EXPECT_GT(point.x, -10.0f);
    EXPECT_LT(point.x, 10.0f);
    EXPECT_GT(point.y, -10.0f);
    EXPECT_LT(point.y, 10.0f);
    EXPECT_GT(point.z, -10.0f);
    EXPECT_LT(point.z, 10.0f);
    EXPECT_EQ(point.r, 0);
    EXPECT_EQ(point.g, 64);
    EXPECT_EQ(point.b, 128);
    EXPECT_FLOAT_EQ(
        point.nx * point.nx + point.ny * point.ny + point.nz * point.nz, 1.0f);
  }

  for (const auto& vis : visibility) {
    EXPECT_GT(vis.size(), 0);
  }

  // Row bands execute on pool workers. Exercise worker-local fusion state
  // with more than one worker; output order may differ with scheduling.
  auto two_worker_options = options;
  two_worker_options.num_threads = 2;
  StereoFusion two_workers(
      two_worker_options, temp_dir, "COLMAP", "", "geometric");
  two_workers.Run();
  EXPECT_TRUE(
      StereoFusionExactDepthTestPeer::FullRamInputsMatchWorkspace(two_workers));
  const auto& parallel_points = two_workers.GetFusedPoints();
  const auto& parallel_visibility = two_workers.GetFusedPointsVisibility();
  ASSERT_FALSE(parallel_points.empty());
  ASSERT_EQ(parallel_points.size(), parallel_visibility.size());
  for (size_t i = 0; i < parallel_points.size(); ++i) {
    const auto& point = parallel_points[i];
    EXPECT_GT(point.x, -10.0f);
    EXPECT_LT(point.x, 10.0f);
    EXPECT_GT(point.y, -10.0f);
    EXPECT_LT(point.y, 10.0f);
    EXPECT_GT(point.z, -10.0f);
    EXPECT_LT(point.z, 10.0f);
    EXPECT_EQ(point.r, 0);
    EXPECT_EQ(point.g, 64);
    EXPECT_EQ(point.b, 128);
    EXPECT_FLOAT_EQ(
        point.nx * point.nx + point.ny * point.ny + point.nz * point.nz, 1.0f);
    ASSERT_FALSE(parallel_visibility[i].empty());
    for (const auto image_idx : parallel_visibility[i]) {
      EXPECT_GE(image_idx, 0);
      EXPECT_LT(image_idx, static_cast<int>(model.images.size()));
    }
  }

  // Aggregation consumes worker visibility buffers. A second run must
  // reconstruct the complete ordered output after those buffers were moved.
  const auto first_points = fused_points;
  const auto first_visibility = visibility;
  fusion.Run();
  ASSERT_EQ(fusion.GetFusedPoints().size(), first_points.size());
  EXPECT_TRUE(
      StereoFusionExactDepthTestPeer::FullRamInputsMatchWorkspace(fusion));
  EXPECT_EQ(fusion.GetFusedPointsVisibility(), first_visibility);
  for (size_t i = 0; i < first_points.size(); ++i) {
    const auto& a = first_points[i];
    const auto& b = fusion.GetFusedPoints()[i];
    EXPECT_FLOAT_EQ(a.x, b.x);
    EXPECT_FLOAT_EQ(a.y, b.y);
    EXPECT_FLOAT_EQ(a.z, b.z);
    EXPECT_FLOAT_EQ(a.nx, b.nx);
    EXPECT_FLOAT_EQ(a.ny, b.ny);
    EXPECT_FLOAT_EQ(a.nz, b.nz);
    EXPECT_EQ(a.r, b.r);
    EXPECT_EQ(a.g, b.g);
    EXPECT_EQ(a.b, b.b);
  }

  // Exercise default and explicit loaders with a duplicate cfg row. Fusion
  // retains the original roster, while common Workspace loading deduplicates.
  {
    std::ofstream duplicate_cfg(temp_dir / "stereo" / "fusion.cfg",
                                std::ios::app);
    duplicate_cfg << image_names.front() << '\n';
  }
  for (const int load_threads : {-1, 2}) {
    SCOPED_TRACE(load_threads);
    options.full_ram_load_threads = load_threads;
    StereoFusion parallel_load(options, temp_dir, "COLMAP", "", "geometric");
    parallel_load.Run();
    const auto& parallel_points = parallel_load.GetFusedPoints();
    ASSERT_EQ(parallel_points.size(), fused_points.size());
    EXPECT_EQ(parallel_load.GetFusedPointsVisibility(), visibility);
    for (size_t i = 0; i < fused_points.size(); ++i) {
      const auto& a = fused_points[i];
      const auto& b = parallel_points[i];
      EXPECT_FLOAT_EQ(a.x, b.x);
      EXPECT_FLOAT_EQ(a.y, b.y);
      EXPECT_FLOAT_EQ(a.z, b.z);
      EXPECT_FLOAT_EQ(a.nx, b.nx);
      EXPECT_FLOAT_EQ(a.ny, b.ny);
      EXPECT_FLOAT_EQ(a.nz, b.nz);
      EXPECT_EQ(a.r, b.r);
      EXPECT_EQ(a.g, b.g);
      EXPECT_EQ(a.b, b.b);
    }
  }
  {
    std::ofstream original_cfg(temp_dir / "stereo" / "fusion.cfg",
                               std::ios::trunc);
    for (const auto& name : image_names) original_cfg << name << '\n';
  }
  options.full_ram_load_threads = -1;

  // The opt-in whole-image cache must preserve the eager path's XYZ,
  // normals, RGB, visibility, and point order on this deterministic fixture.
  options.bounded_cache = true;
  options.cache_size = 12000.0 / (1024.0 * 1024.0 * 1024.0);
  StereoFusion bounded(options, temp_dir, "COLMAP", "", "geometric");
  bounded.Run();
  EXPECT_FALSE(StereoFusionExactDepthTestPeer::HasFullRamInputs(bounded));
  const auto& bounded_points = bounded.GetFusedPoints();
  const auto& bounded_visibility = bounded.GetFusedPointsVisibility();
  ASSERT_EQ(bounded_points.size(), fused_points.size());
  EXPECT_EQ(bounded_visibility, visibility);
  for (size_t i = 0; i < fused_points.size(); ++i) {
    const auto& a = fused_points[i];
    const auto& b = bounded_points[i];
    EXPECT_FLOAT_EQ(a.x, b.x);
    EXPECT_FLOAT_EQ(a.y, b.y);
    EXPECT_FLOAT_EQ(a.z, b.z);
    EXPECT_FLOAT_EQ(a.nx, b.nx);
    EXPECT_FLOAT_EQ(a.ny, b.ny);
    EXPECT_FLOAT_EQ(a.nz, b.nz);
    EXPECT_EQ(a.r, b.r);
    EXPECT_EQ(a.g, b.g);
    EXPECT_EQ(a.b, b.b);
  }

  const auto expect_same_fusion = [&](const StereoFusion& candidate,
                                      const char* arm) {
    SCOPED_TRACE(arm);
    const auto& actual = candidate.GetFusedPoints();
    ASSERT_EQ(actual.size(), fused_points.size());
    EXPECT_EQ(candidate.GetFusedPointsVisibility(), visibility);
    for (size_t i = 0; i < actual.size(); ++i) {
      const auto& a = fused_points[i];
      const auto& b = actual[i];
      EXPECT_FLOAT_EQ(a.x, b.x);
      EXPECT_FLOAT_EQ(a.y, b.y);
      EXPECT_FLOAT_EQ(a.z, b.z);
      EXPECT_FLOAT_EQ(a.nx, b.nx);
      EXPECT_FLOAT_EQ(a.ny, b.ny);
      EXPECT_FLOAT_EQ(a.nz, b.nz);
      EXPECT_EQ(a.r, b.r);
      EXPECT_EQ(a.g, b.g);
      EXPECT_EQ(a.b, b.b);
    }
  };

  if (ImmutableMapRegion::Supported() &&
      normal_precision_guard::SupportedBuild()) {
    // The guarded codec preserves ordered geometry and visibility. This
    // fixture uses exactly representable 0/1 normal components.
    StereoFusionOptions guarded_options = options;
    guarded_options.bounded_cache_components = true;
    guarded_options.bounded_cache_normal_tile_size = 8;
    guarded_options.bounded_cache_normal_tile_slots = 2;
    guarded_options.bounded_cache_normal_mapped_bytes = 1 << 20;
    guarded_options.bounded_cache_normal_mapped_files = 2;
    guarded_options.bounded_cache_normal_snorm16 = true;
    guarded_options.bounded_cache_normal_snorm16_guard = true;
    StereoFusion guarded(guarded_options, temp_dir, "COLMAP", "", "geometric");
    guarded.Run();
    expect_same_fusion(guarded, "guarded");
    guarded.Run();
    expect_same_fusion(guarded, "guarded repeated successful run");
    ASSERT_TRUE(guarded.GetBoundedCacheStats());
    EXPECT_GT(guarded.GetBoundedCacheStats()->normal_tile_loads, 0);
    EXPECT_GT(guarded.GetNormalTileRingStats().hits, 0);
  }

  options.bounded_cache_neighborhood = true;
  options.cache_size = 50000.0 / (1024.0 * 1024.0 * 1024.0);
  StereoFusion borrowed_whole(options, temp_dir, "COLMAP", "", "geometric");
  borrowed_whole.Run();
  expect_same_fusion(borrowed_whole, "borrowed_whole");
  ASSERT_TRUE(borrowed_whole.GetBoundedCacheStats());
  EXPECT_GT(borrowed_whole.GetBoundedCacheStats()
                ->component_loads[static_cast<size_t>(
                    BoundedWorkspace::Component::kImage)],
            0);

  // Fusion stays serial while two prefill lanes borrow the same admitted
  // neighborhood. The scheduled-lane count verifies activation, and the
  // complete ordered point/visibility oracle verifies its output.
  options.bounded_cache_prefill_threads = 2;
  StereoFusion parallel_prefill(options, temp_dir, "COLMAP", "", "geometric");
  parallel_prefill.Run();
  expect_same_fusion(parallel_prefill, "parallel_prefill");
  EXPECT_EQ(parallel_prefill.GetMaxBoundedPrefillLanesScheduled(), 2);

  // A worker failure must join the admitted prefill tasks and release their
  // reservations before the resident reader scope can close.
  std::atomic<int> failed_callback_calls{0};
  auto failed_scope = ScopedBitmapReader::Create(
      {temp_dir / "images" / image_names.front()},
      [&failed_callback_calls](
          const std::filesystem::path&, bool, bool, Bitmap*) {
        ++failed_callback_calls;
        return false;
      },
      {{30, 20}});
  StereoFusion failed_prefill(options, temp_dir, "COLMAP", "", "geometric");
  EXPECT_THROW(failed_prefill.Run(), std::exception);
  EXPECT_GT(failed_callback_calls.load(), 0);
  EXPECT_TRUE(failed_prefill.GetFusedPoints().empty());
  EXPECT_TRUE(failed_prefill.GetFusedPointsVisibility().empty());
  ASSERT_TRUE(failed_prefill.GetBoundedCacheStats());
  EXPECT_EQ(failed_prefill.GetBoundedCacheStats()->reserved_bytes, 0);
  EXPECT_LE(failed_prefill.GetBoundedCacheStats()->resident_bytes,
            failed_prefill.GetBoundedCacheStats()->capacity_bytes);
  // Removing the injected read failure repairs the input source. Retry the
  // same object and compare complete ordered points and visibility.
  failed_scope->Close();
  failed_prefill.Run();
  expect_same_fusion(failed_prefill, "failed_prefill");

  StereoFusion cancelled_prefill(options, temp_dir, "COLMAP", "", "geometric");
  cancelled_prefill.SetCheckIfStoppedFunc([&cancelled_prefill]() {
    const auto stats = cancelled_prefill.GetBoundedCacheStats();
    return stats && stats->loads > 0;
  });
  cancelled_prefill.Run();
  EXPECT_TRUE(cancelled_prefill.GetFusedPoints().empty());
  EXPECT_TRUE(cancelled_prefill.GetFusedPointsVisibility().empty());
  ASSERT_TRUE(cancelled_prefill.GetBoundedCacheStats());
  EXPECT_GT(cancelled_prefill.GetBoundedCacheStats()->loads, 0);
  EXPECT_EQ(cancelled_prefill.GetBoundedCacheStats()->reserved_bytes, 0);
  EXPECT_LE(cancelled_prefill.GetBoundedCacheStats()->resident_bytes,
            cancelled_prefill.GetBoundedCacheStats()->capacity_bytes);
  options.bounded_cache_prefill_threads = -1;

  options.bounded_cache_components = true;
  options.cache_size = 12000.0 / (1024.0 * 1024.0 * 1024.0);
  StereoFusion borrowed_depth(options, temp_dir, "COLMAP", "", "geometric");
  borrowed_depth.Run();
  expect_same_fusion(borrowed_depth, "borrowed_depth");
  ASSERT_TRUE(borrowed_depth.GetBoundedCacheStats());
  const auto component_loads =
      borrowed_depth.GetBoundedCacheStats()->component_loads;
  EXPECT_GT(
      component_loads[static_cast<size_t>(BoundedWorkspace::Component::kDepth)],
      0);
  EXPECT_GT(component_loads[static_cast<size_t>(
                BoundedWorkspace::Component::kNormal)],
            0);
  EXPECT_GT(component_loads[static_cast<size_t>(
                BoundedWorkspace::Component::kBitmap)],
            0);

  if (ImmutableMapRegion::Supported()) {
    options.bounded_cache_neighborhood = false;
    options.bounded_cache_normal_tile_size = 7;
    options.bounded_cache_normal_tile_slots = 2;
    options.bounded_cache_normal_mapped_bytes = 1 << 20;
    options.bounded_cache_normal_mapped_files = 2;
    options.bounded_cache_diagnostics = true;
    ASSERT_TRUE(options.Check());
    StereoFusion tiled(options, temp_dir, "COLMAP", "", "geometric");
    tiled.Run();
    expect_same_fusion(tiled, "tiled");
    ASSERT_TRUE(tiled.GetBoundedCacheStats());
    EXPECT_GT(tiled.GetBoundedCacheStats()->normal_tile_loads, 0);
    EXPECT_EQ(tiled.GetBoundedCacheStats()->reserved_bytes, 0);
    EXPECT_GT(tiled.GetMappedNormalStats().loads, 0);
    EXPECT_EQ(tiled.GetMappedNormalStats().active_copies, 0);
    EXPECT_LE(tiled.GetMappedNormalStats().peak_charged_bytes,
              static_cast<size_t>(options.bounded_cache_normal_mapped_bytes));
    const auto tile_row = tiled.GetBoundedCacheDiagnosticRow();
    ASSERT_TRUE(tile_row);
    EXPECT_EQ(tile_row->find("COLMAP_FUSION_CACHE_DIAGNOSTIC_V1"), 0);
    EXPECT_NE(
        tile_row->find(
            "decoded_normal_tile_loads=" +
            std::to_string(tiled.GetBoundedCacheStats()->normal_tile_loads)),
        std::string::npos);
    EXPECT_NE(
        tile_row->find("mapped_loads=" +
                       std::to_string(tiled.GetMappedNormalStats().loads)),
        std::string::npos);
    EXPECT_NE(tile_row->find("ring_hint_hits=unavailable"), std::string::npos);
    options.bounded_cache_normal_tile_slots = 64;
    options.bounded_cache_normal_tile_hints = true;
    options.cache_size = 50000.0 / (1024.0 * 1024.0 * 1024.0);
    StereoFusion indexed_tiles(options, temp_dir, "COLMAP", "", "geometric");
    indexed_tiles.Run();
    expect_same_fusion(indexed_tiles, "indexed_tiles");
    EXPECT_GT(indexed_tiles.GetNormalTileRingStats().indexed_lookups, 0);
    EXPECT_GT(indexed_tiles.GetNormalTileRingStats().hint_checks, 0);
    const auto indexed_row = indexed_tiles.GetBoundedCacheDiagnosticRow();
    ASSERT_TRUE(indexed_row);
    EXPECT_NE(indexed_row->find(
                  "ring_indexed_lookups=" +
                  std::to_string(
                      indexed_tiles.GetNormalTileRingStats().indexed_lookups)),
              std::string::npos);
    options.bounded_cache_neighborhood = true;
    options.cache_size = 1.0 / 1024.0;
    ASSERT_TRUE(options.Check());
    StereoFusion tile_neighborhood(
        options, temp_dir, "COLMAP", "", "geometric");
    tile_neighborhood.Run();
    expect_same_fusion(tile_neighborhood, "tile_neighborhood");
    const auto neighborhood_row =
        tile_neighborhood.GetBoundedCacheDiagnosticRow();
    ASSERT_TRUE(neighborhood_row);
    const auto diagnostic_count = [&](const std::string& diagnostic,
                                      const std::string& key) {
      const auto pos = diagnostic.find(key + "=");
      EXPECT_NE(pos, std::string::npos);
      return pos == std::string::npos
                 ? size_t{0}
                 : static_cast<size_t>(
                       std::stoull(diagnostic.substr(pos + key.size() + 1)));
    };
    EXPECT_GE(
        diagnostic_count(*neighborhood_row, "neighborhood_bitmap_admitted"), 2);
    EXPECT_GT(
        diagnostic_count(*neighborhood_row, "neighborhood_bitmap_carried"), 0);
    EXPECT_GT(
        diagnostic_count(*neighborhood_row, "neighborhood_bitmap_borrows"), 0);
    EXPECT_GT(diagnostic_count(*neighborhood_row, "neighborhood_bitmap_init"),
              0);
    EXPECT_GT(diagnostic_count(*neighborhood_row, "neighborhood_depth_borrows"),
              0);
    ASSERT_TRUE(tile_neighborhood.GetBoundedCacheStats());
    EXPECT_EQ(tile_neighborhood.GetBoundedCacheStats()->reserved_bytes, 0);
    std::atomic<int> lazy_bitmap_failures{0};
    auto failed_lazy_scope = ScopedBitmapReader::Create(
        {temp_dir / "images" / image_names.front()},
        [&lazy_bitmap_failures](
            const std::filesystem::path&, bool, bool, Bitmap*) {
          ++lazy_bitmap_failures;
          return false;
        },
        {{30, 20}});
    StereoFusion failed_lazy(options, temp_dir, "COLMAP", "", "geometric");
    EXPECT_THROW(failed_lazy.Run(), std::exception);
    EXPECT_GT(lazy_bitmap_failures.load(), 0);
    ASSERT_TRUE(failed_lazy.GetBoundedCacheStats());
    EXPECT_EQ(failed_lazy.GetBoundedCacheStats()->reserved_bytes, 0);
    failed_lazy_scope->Close();
    failed_lazy.Run();
    expect_same_fusion(failed_lazy, "failed_lazy");
    options.num_threads = 2;
    StereoFusion parallel_tile_neighborhood(
        options, temp_dir, "COLMAP", "", "geometric");
    parallel_tile_neighborhood.Run();
    EXPECT_GT(parallel_tile_neighborhood.GetFusedPoints().size(), 0);
    EXPECT_EQ(parallel_tile_neighborhood.GetFusedPoints().size(),
              parallel_tile_neighborhood.GetFusedPointsVisibility().size());
    ASSERT_TRUE(parallel_tile_neighborhood.GetBoundedCacheStats());
    EXPECT_EQ(parallel_tile_neighborhood.GetBoundedCacheStats()->reserved_bytes,
              0);
    const auto parallel_row =
        parallel_tile_neighborhood.GetBoundedCacheDiagnosticRow();
    ASSERT_TRUE(parallel_row);
    EXPECT_GT(diagnostic_count(*parallel_row, "neighborhood_bitmap_borrows"),
              0);
    EXPECT_GT(diagnostic_count(*parallel_row, "neighborhood_bitmap_init"), 0);
    options.num_threads = 1;
    options.bounded_cache_neighborhood = false;
    // tile_size=0 with an owned mapped pool is a distinct, serial FP32 route.
    options.bounded_cache_normal_tile_size = 0;
    options.bounded_cache_normal_tile_slots = 4;
    ASSERT_TRUE(options.Check());
    const auto normal_path = temp_dir / "stereo" / "normal_maps" /
                             (image_names[0] + ".geometric.bin");
    const auto one_owner =
        ImmutableMapRegion::GetMappingReservationBytes(normal_path);
    options.bounded_cache_normal_mapped_bytes = static_cast<int64_t>(one_owner);
    StereoFusion insufficient(options, temp_dir, "COLMAP", "", "geometric");
    EXPECT_THROW(insufficient.Run(), std::exception);
    EXPECT_EQ(insufficient.GetMappedNormalStats().loads, 0);
    options.bounded_cache_normal_mapped_bytes = 1 << 20;
    StereoFusion direct(options, temp_dir, "COLMAP", "", "geometric");
    direct.Run();
    expect_same_fusion(direct, "direct");
    ASSERT_TRUE(direct.GetBoundedCacheStats());
    EXPECT_EQ(direct.GetBoundedCacheStats()->normal_tile_loads, 0);
    EXPECT_EQ(
        direct.GetBoundedCacheStats()->component_loads[static_cast<size_t>(
            BoundedWorkspace::Component::kNormal)],
        0);
    EXPECT_GT(direct.GetMappedNormalStats().loads, 0);
    EXPECT_EQ(direct.GetMappedNormalStats().active_copies, 0);
    EXPECT_EQ(direct.GetMappedNormalStats().resident_regions,
              3 * direct.GetMappedNormalStats().resident_files);
    EXPECT_LE(direct.GetMappedNormalStats().peak_charged_bytes,
              static_cast<size_t>(options.bounded_cache_normal_mapped_bytes));
    const auto direct_row = direct.GetBoundedCacheDiagnosticRow();
    ASSERT_TRUE(direct_row);
    EXPECT_NE(direct_row->find("direct_mapped_mode=1"), std::string::npos);
    EXPECT_NE(direct_row->find("direct_window_rows=256"), std::string::npos);
    EXPECT_NE(direct_row->find("mapped_peak_charged_regions="),
              std::string::npos);
    EXPECT_NE(
        direct_row->find("mapped_loads=" +
                         std::to_string(direct.GetMappedNormalStats().loads)),
        std::string::npos);
    EXPECT_NE(direct_row->find("direct_ring_slots=1"), std::string::npos);
    EXPECT_NE(direct_row->find("direct_component_pins_admitted=1"),
              std::string::npos);
    EXPECT_EQ(direct.GetBoundedCacheStats()->hits, 0);
    EXPECT_EQ(
        direct.GetBoundedCacheStats()->component_loads[static_cast<size_t>(
            BoundedWorkspace::Component::kImage)],
        0);
    EXPECT_EQ(
        direct.GetBoundedCacheStats()->component_loads[static_cast<size_t>(
            BoundedWorkspace::Component::kDepth)],
        2);
    EXPECT_EQ(
        direct.GetBoundedCacheStats()->component_loads[static_cast<size_t>(
            BoundedWorkspace::Component::kBitmap)],
        2);
    EXPECT_EQ(direct_row->find("direct_depth_borrows=0"), std::string::npos);
    EXPECT_EQ(direct_row->find("direct_bitmap_borrows=0"), std::string::npos);

    // Both components of either view fit separately, but the complete two-view
    // owner set does not. Demand acquisition must still make progress and keep
    // the original ordered output under this tight decoded budget.
    options.cache_size = 5000.0 / (1024.0 * 1024.0 * 1024.0);
    StereoFusion direct_demand(options, temp_dir, "COLMAP", "", "geometric");
    direct_demand.Run();
    expect_same_fusion(direct_demand, "direct_demand");
    const auto demand_row = direct_demand.GetBoundedCacheDiagnosticRow();
    ASSERT_TRUE(demand_row);
    EXPECT_NE(demand_row->find("direct_component_pins_admitted=0"),
              std::string::npos);
    ASSERT_TRUE(direct_demand.GetBoundedCacheStats());
    EXPECT_GT(direct_demand.GetBoundedCacheStats()->hits, 0);
    EXPECT_EQ(direct_demand.GetBoundedCacheStats()->reserved_bytes, 0);

    // Failure during a bitmap load must unwind all locally acquired
    // pins before publication. Closing the failing reader then permits a fresh
    // Run on the same Fusion object with the normal exact-output oracle.
    options.cache_size = 50000.0 / (1024.0 * 1024.0 * 1024.0);
    std::atomic<int> direct_failed_calls{0};
    auto direct_failed_scope = ScopedBitmapReader::Create(
        {temp_dir / "images" / model.GetImageName(1)},
        [&direct_failed_calls](
            const std::filesystem::path&, bool, bool, Bitmap*) {
          ++direct_failed_calls;
          return false;
        },
        {{30, 20}});
    StereoFusion direct_failed(options, temp_dir, "COLMAP", "", "geometric");
    EXPECT_THROW(direct_failed.Run(), std::exception);
    EXPECT_GT(direct_failed_calls.load(), 0);
    const auto failed_row = direct_failed.GetBoundedCacheDiagnosticRow();
    ASSERT_TRUE(failed_row);
    EXPECT_NE(failed_row->find("direct_component_pins_admitted=0"),
              std::string::npos);
    ASSERT_TRUE(direct_failed.GetBoundedCacheStats());
    EXPECT_EQ(direct_failed.GetBoundedCacheStats()->reserved_bytes, 0);
    direct_failed_scope->Close();
    direct_failed.Run();
    expect_same_fusion(direct_failed, "direct_failed");

    StereoFusion direct_cancelled(options, temp_dir, "COLMAP", "", "geometric");
    direct_cancelled.SetCheckIfStoppedFunc([&direct_cancelled]() {
      const auto stats = direct_cancelled.GetBoundedCacheStats();
      return stats && stats->loads >= 4;
    });
    direct_cancelled.Run();
    EXPECT_TRUE(direct_cancelled.GetFusedPoints().empty());
    ASSERT_TRUE(direct_cancelled.GetBoundedCacheStats());
    EXPECT_EQ(direct_cancelled.GetBoundedCacheStats()->reserved_bytes, 0);
    direct_cancelled.SetCheckIfStoppedFunc([]() { return false; });
    direct_cancelled.Run();
    expect_same_fusion(direct_cancelled, "direct_cancelled");

    options.num_threads = 2;
    EXPECT_FALSE(options.Check());
    options.num_threads = 1;
    options.bounded_cache_normal_mapped_files = 1;
    EXPECT_FALSE(options.Check());
    options.bounded_cache_normal_mapped_files = 2;
    options.bounded_cache_neighborhood = true;
    EXPECT_FALSE(options.Check());
    options.bounded_cache_neighborhood = false;
    options.bounded_cache_components = false;
    EXPECT_FALSE(options.Check());
    options.bounded_cache_components = true;
    options.bounded_cache_normal_tile_size = 7;
    options.bounded_cache_normal_mapped_files = 0;
    EXPECT_FALSE(options.Check());
    options.bounded_cache_normal_mapped_files = 2;
  }

  options.bounded_cache_normal_tile_size = 0;
  options.bounded_cache_normal_mapped_bytes = 0;
  options.bounded_cache_normal_mapped_files = 0;

  options.bounded_cache_components = false;
  options.bounded_cache_neighborhood = false;

  options.bounded_cache_diagnostics = true;
  bool stop_checked = false;
  StereoFusion cancelled_fusion(options, temp_dir, "COLMAP", "", "geometric");
  cancelled_fusion.SetCheckIfStoppedFunc([&stop_checked]() {
    stop_checked = true;
    return true;
  });
  cancelled_fusion.Run();
  EXPECT_TRUE(stop_checked);
  EXPECT_TRUE(cancelled_fusion.GetFusedPoints().empty());
  const auto no_tile_row = cancelled_fusion.GetBoundedCacheDiagnosticRow();
  ASSERT_TRUE(no_tile_row);
  EXPECT_NE(no_tile_row->find("mapped_normal=unavailable ring=unavailable"),
            std::string::npos);

  // An unused initial view still leads to the configured overlapping view.
  // Its depth and normal maps must never enter the fusion traversal.
  {
    std::ofstream config(temp_dir / "stereo" / "fusion.cfg");
    config << model.images[1].GetPath().filename().string() << '\n';
  }
  StereoFusionOptions single_view_options;
  single_view_options.min_num_pixels = 1;
  single_view_options.num_threads = 1;
  StereoFusion single_view(
      single_view_options, temp_dir, "COLMAP", "", "geometric");
  single_view.Run();
  ASSERT_FALSE(single_view.GetFusedPoints().empty());
  ASSERT_EQ(single_view.GetFusedPoints().size(),
            single_view.GetFusedPointsVisibility().size());
  for (const auto& visible : single_view.GetFusedPointsVisibility()) {
    EXPECT_EQ(visible, std::vector<int>{1});
  }
}

TEST(StereoFusion, CompactResidentRgbMatchesDiskFusion) {
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "images");
  CreateDirIfNotExists(root / "stereo");
  CreateDirIfNotExists(root / "stereo" / "depth_maps");
  CreateDirIfNotExists(root / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 2;
  dataset.camera_width = 10;
  dataset.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  Workspace::Options workspace_options;
  workspace_options.workspace_path = root;
  workspace_options.workspace_format = "COLMAP";
  workspace_options.input_type = "geometric";
  Workspace descriptor(workspace_options);
  std::vector<std::string> names;
  std::vector<std::filesystem::path> paths;
  auto originals = std::make_shared<std::vector<Bitmap>>();
  for (int image_id = 1; image_id <= 2; ++image_id) {
    const auto name = reconstruction.Image(image_id).Name();
    names.push_back(name);
    const int image = descriptor.GetModel().GetImageIdx(name);
    paths.push_back(descriptor.GetBitmapPath(image));
    Mat<float> depth(7, 3, 1);
    depth.Fill(5.0f);
    depth.Write(descriptor.GetDepthMapPath(image));
    Mat<float> normal(7, 3, 3);
    normal.Fill(0.0f);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 7; ++col) {
        normal.Set(row, col, 2, 1.0f);
      }
    }
    normal.Write(descriptor.GetNormalMapPath(image));
    Bitmap bitmap(10, 5, true);
    for (int row = 0; row < 5; ++row) {
      for (int col = 0; col < 10; ++col) {
        bitmap.SetPixel(
            col,
            row,
            BitmapColor<uint8_t>(
                col * 19 + row, row * 31 + col, image_id * 20 + col));
      }
    }
    ASSERT_TRUE(bitmap.Write(paths.back()));
    originals->push_back(bitmap);
  }
  {
    std::ofstream config(root / "stereo" / "fusion.cfg");
    for (const auto& name : names) config << name << '\n';
  }
  StereoFusionOptions options;
  options.num_threads = 1;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.max_traversal_depth = 10;
  options.check_num_images = 10;
  StereoFusion disk(options, root, "COLMAP", "", "geometric");
  disk.Run();
  const auto& expected = disk.GetFusedPoints();
  ASSERT_FALSE(expected.empty());
  auto scope = ScopedBitmapReader::Create(
      paths,
      [strong = originals, paths](
          const auto& path, bool rgb, bool linear, Bitmap* target) {
        EXPECT_TRUE(rgb);
        EXPECT_FALSE(linear);
        for (size_t i = 0; i < paths.size(); ++i) {
          if (path == std::filesystem::absolute(paths[i]).lexically_normal()) {
            *target = strong->at(i);
            return true;
          }
        }
        return false;
      },
      {{10, 5}, {10, 5}});
  originals.reset();
  for (const auto& path : paths) ASSERT_TRUE(std::filesystem::remove(path));
  options.bounded_cache = true;
  options.bounded_cache_compact_rgb = true;
  options.cache_size = 600.0 / (1024.0 * 1024.0 * 1024.0);
  StereoFusion compact(options, root, "COLMAP", "", "geometric");
  compact.Run();  // The strong scope owner stays alive through worker join.
  const auto& actual = compact.GetFusedPoints();
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_EQ(compact.GetFusedPointsVisibility(),
            disk.GetFusedPointsVisibility());
  for (size_t i = 0; i < actual.size(); ++i) {
    const auto& a = actual[i];
    const auto& b = expected[i];
    EXPECT_FLOAT_EQ(a.x, b.x);
    EXPECT_FLOAT_EQ(a.y, b.y);
    EXPECT_FLOAT_EQ(a.z, b.z);
    EXPECT_FLOAT_EQ(a.nx, b.nx);
    EXPECT_FLOAT_EQ(a.ny, b.ny);
    EXPECT_FLOAT_EQ(a.nz, b.nz);
    EXPECT_EQ(a.r, b.r);
    EXPECT_EQ(a.g, b.g);
    EXPECT_EQ(a.b, b.b);
  }
  scope->Close();
  for (const auto& name : names) {
    EXPECT_FALSE(descriptor.HasBitmap(descriptor.GetModel().GetImageIdx(name)));
  }
}

TEST(ReadPointsVisibility, BinaryWireBytes) {
  const auto test_file = CreateTestDir() / "wire.vis";
  std::vector<std::vector<std::vector<int>>> cases = {
      {},
      {{},
       {std::numeric_limits<int>::min(),
        -1,
        0,
        1,
        std::numeric_limits<int>::max()}},
      {std::vector<int>(8191), {}, std::vector<int>(8193)}};
  for (size_t i = 0; i < cases[2][0].size(); ++i) {
    cases[2][0][i] = static_cast<int>(i % 11);
  }
  for (size_t i = 0; i < cases[2][2].size(); ++i) {
    cases[2][2][i] = static_cast<int>(10 - i % 11);
  }
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    const auto& visibility = cases[i];
    WritePointsVisibility(test_file, visibility);
    std::ifstream file(test_file, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    const std::string actual((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    // Preserve global/list counts, order, duplicates and uint32 conversions.
    std::ostringstream expected(std::ios::out | std::ios::binary);
    WriteBinaryLittleEndian<uint64_t>(&expected, visibility.size());
    for (const auto& images : visibility) {
      WriteBinaryLittleEndian<uint32_t>(&expected, images.size());
      for (const int image : images) {
        WriteBinaryLittleEndian<uint32_t>(&expected, image);
      }
    }
    EXPECT_EQ(actual, expected.str());
    EXPECT_EQ(ReadPointsVisibility(test_file, visibility.size()), visibility);
  }
}

TEST(ReadPointsVisibility, TrailingBytesAreIgnored) {
  const auto path = CreateTestDir() / "trailing.vis";
  const std::vector<std::vector<int>> expected = {{7, 7, 0, -1}, {}, {42}};
  const std::vector<char> tail(32769, 'x');
  for (const size_t count : {size_t{1}, size_t{3}, size_t{4}, tail.size()}) {
    SCOPED_TRACE(count);
    WritePointsVisibility(path, expected);
    {
      std::ofstream file(path, std::ios::binary | std::ios::app);
      file.write(tail.data(), count);
    }
    EXPECT_EQ(ReadPointsVisibility(path, expected.size()), expected);
  }
}

TEST(ReadPointsVisibility, TruncatedHeaderCountAndPayloadAreRejected) {
  const auto path = CreateTestDir() / "truncated.vis";
  const std::vector<std::vector<int>> expected = {{7, 8, 9}, {}, {3, 4}};
  std::ostringstream encoded(std::ios::out | std::ios::binary);
  WriteBinaryLittleEndian<uint64_t>(&encoded, expected.size());
  for (const auto& images : expected) {
    WriteBinaryLittleEndian<uint32_t>(&encoded, images.size());
    for (const int image : images) {
      WriteBinaryLittleEndian<uint32_t>(&encoded, image);
    }
  }
  const std::string bytes = encoded.str();
  for (size_t cut = 0; cut < bytes.size(); ++cut) {
    SCOPED_TRACE(cut);
    {
      std::ofstream file(path, std::ios::binary);
      file.write(bytes.data(), cut);
    }
    EXPECT_THROW(ReadPointsVisibility(path, expected.size()),
                 std::invalid_argument);
  }
}

TEST(ReadPointsVisibility, RoundTrip) {
  const auto test_dir = CreateTestDir();
  const auto vis_path = test_dir / "test.vis";

  std::vector<std::vector<int>> expected = {
      {0, 1, 2},
      {1, 3},
      {},
      {0, 2, 3, 4},
  };
  WritePointsVisibility(vis_path, expected);

  const auto actual = ReadPointsVisibility(vis_path, expected.size());

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(actual[i].size(), expected[i].size());
    for (size_t j = 0; j < expected[i].size(); ++j) {
      EXPECT_EQ(actual[i][j], expected[i][j]);
    }
  }
}

TEST(ReadPointsVisibility, LargeRoundTrip) {
  const auto vis_path = CreateTestDir() / "large.vis";
  // Variable records exceed 1 MiB and leave a partial final output buffer.
  std::vector<std::vector<int>> expected(32771);
  for (size_t i = 0; i < expected.size(); ++i) {
    auto& visibility = expected[i];
    visibility.resize(i % 17);
    for (size_t j = 0; j < visibility.size(); ++j) {
      // Retain repeated image indices and their order, as well as empty lists.
      visibility[j] = static_cast<int>((i * 13 + (j % 5) * 7) % 1024);
    }
  }

  WritePointsVisibility(vis_path, expected);
  ASSERT_GT(std::filesystem::file_size(vis_path), 1024 * 1024);
  const auto actual = ReadPointsVisibility(vis_path, expected.size());
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "point index: " << i;
  }
}

TEST(ReadPointsVisibility, SizeMismatch) {
  const auto test_dir = CreateTestDir();
  const auto vis_path = test_dir / "test.vis";

  std::vector<std::vector<int>> data = {{0, 1}, {2}};
  WritePointsVisibility(vis_path, data);

  EXPECT_THROW(ReadPointsVisibility(vis_path, 5), std::invalid_argument);
}

TEST(StereoFusion, Rotation180Invariance) {
  const size_t width = 40;
  const size_t height = 30;
  const double focal_length = 100.0;
  const double cx = 18.0;
  const double cy = 12.0;

  const int test_col = 12;
  const int test_row = 8;
  const float test_depth = 5.0f;

  auto CreateWorkspace = [&](const std::filesystem::path& dir,
                             const double cam_cx,
                             const double cam_cy,
                             const Rigid3d& cam_from_world,
                             const int col,
                             const int row) {
    CreateDirIfNotExists(dir / "sparse");
    CreateDirIfNotExists(dir / "images");
    CreateDirIfNotExists(dir / "stereo");
    CreateDirIfNotExists(dir / "stereo" / "depth_maps");
    CreateDirIfNotExists(dir / "stereo" / "normal_maps");

    Reconstruction reconstruction;
    colmap::Camera camera;
    camera.camera_id = 1;
    camera.model_id = CameraModelId::kPinhole;
    camera.width = width;
    camera.height = height;
    camera.params = {focal_length, focal_length, cam_cx, cam_cy};
    reconstruction.AddCameraWithTrivialRig(camera);

    colmap::Image image;
    image.SetImageId(1);
    image.SetName("image.png");
    image.SetCameraId(1);
    reconstruction.AddImageWithTrivialFrame(image, cam_from_world);
    reconstruction.Write(dir / "sparse");

    Mat<float> depth_map(width, height, 1);
    depth_map.Fill(0.0f);
    depth_map.Set(row, col, 0, test_depth);
    depth_map.Write(dir / "stereo" / "depth_maps" / "image.png.geometric.bin");

    Mat<float> normal_map(width, height, 3);
    normal_map.Fill(0.0f);
    normal_map.Set(row, col, 2, 1.0f);
    normal_map.Write(dir / "stereo" / "normal_maps" /
                     "image.png.geometric.bin");

    Bitmap bitmap(width, height, true);
    bitmap.Fill(BitmapColor<uint8_t>(100, 100, 100));
    bitmap.Write(dir / "images" / "image.png");

    std::ofstream fusion_cfg(dir / "stereo" / "fusion.cfg");
    fusion_cfg << "image.png\n";
    fusion_cfg.close();
  };

  const auto temp_dir1 = CreateTestDir();
  CreateWorkspace(temp_dir1, cx, cy, Rigid3d(), test_col, test_row);

  const auto temp_dir2 = CreateTestDir();
  const Eigen::Quaterniond q_z180(
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
  CreateWorkspace(temp_dir2,
                  width - cx,
                  height - cy,
                  Rigid3d(q_z180, Eigen::Vector3d::Zero()),
                  width - 1 - test_col,
                  height - 1 - test_row);

  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.use_cache = false;

  StereoFusion fusion1(options, temp_dir1, "COLMAP", "", "geometric");
  fusion1.Run();

  StereoFusion fusion2(options, temp_dir2, "COLMAP", "", "geometric");
  fusion2.Run();

  const auto& points1 = fusion1.GetFusedPoints();
  const auto& points2 = fusion2.GetFusedPoints();

  ASSERT_EQ(points1.size(), 1);
  ASSERT_EQ(points2.size(), 1);

  // Expected 3D point using pixel center at (col + 0.5, row + 0.5).
  const float expected_x = test_depth * (test_col + 0.5f - cx) / focal_length;
  const float expected_y = test_depth * (test_row + 0.5f - cy) / focal_length;
  const float expected_z = test_depth;

  EXPECT_NEAR(points1[0].x, expected_x, 1e-5f);
  EXPECT_NEAR(points1[0].y, expected_y, 1e-5f);
  EXPECT_NEAR(points1[0].z, expected_z, 1e-5f);

  EXPECT_NEAR(points2[0].x, expected_x, 1e-5f);
  EXPECT_NEAR(points2[0].y, expected_y, 1e-5f);
  EXPECT_NEAR(points2[0].z, expected_z, 1e-5f);

  EXPECT_NEAR(points1[0].x, points2[0].x, 1e-5f);
  EXPECT_NEAR(points1[0].y, points2[0].y, 1e-5f);
  EXPECT_NEAR(points1[0].z, points2[0].z, 1e-5f);
}

std::filesystem::path CreateTwoViewNormalDecisionFixture() {
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "images");
  CreateDirIfNotExists(root / "stereo");
  CreateDirIfNotExists(root / "stereo" / "depth_maps");
  CreateDirIfNotExists(root / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 2;
  dataset.camera_width = 30;
  dataset.camera_height = 20;
  dataset.camera_params = {30.0, 15.0, 10.0, 0.0};
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  // Both views project the same raster point. Sparse tracks still establish
  // the two-view neighborhood; only the normal decision differs.
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    reconstruction.Frame(frame_id).SetRigFromWorld(Rigid3d());
  }
  reconstruction.Write(root / "sparse");
  std::vector<std::string> names;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    names.push_back(image.Name());
    Mat<float> depth(30, 20, 1);
    depth.Fill(5.0f);
    depth.Write(root / "stereo" / "depth_maps" /
                (image.Name() + ".geometric.bin"));
    Mat<float> normal(30, 20, 3);
    normal.Fill(0.0f);
    const float z = names.size() == 1 ? 1.0f : 0.99999f;
    for (size_t row = 0; row < 20; ++row) {
      for (size_t col = 0; col < 30; ++col) normal.Set(row, col, 2, z);
    }
    normal.Write(root / "stereo" / "normal_maps" /
                 (image.Name() + ".geometric.bin"));
    Bitmap bitmap(30, 20, true);
    bitmap.Fill(BitmapColor<uint8_t>(0, 64, 128));
    bitmap.Write(root / "images" / image.Name());
  }
  THROW_CHECK_EQ(names.size(), 2);
  {
    std::ofstream config(root / "stereo" / "fusion.cfg");
    for (const auto& name : names) config << name << '\n';
  }
  return root;
}

TEST(StereoFusion, RetiredEagerInputsSurviveInterruptionAndRepeat) {
  const auto root = CreateTwoViewNormalDecisionFixture();
  StereoFusionOptions options;
  options.num_threads = 1;
  options.min_num_pixels = options.max_num_pixels = 2;
  StereoFusion fusion(options, root, "COLMAP", "", "geometric");
  fusion.SetCheckIfStoppedFunc([&fusion] {
    return StereoFusionExactDepthTestPeer::NumRetiredFullRamInputs(fusion) > 0;
  });
  fusion.Run();
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NumRetiredFullRamInputs(fusion), 1);
  EXPECT_TRUE(
      StereoFusionExactDepthTestPeer::FullRamInputsMatchWorkspace(fusion));
  ASSERT_FALSE(fusion.GetFusedPoints().empty());
  const auto partial = fusion.GetFusedPoints();
  const auto partial_visibility = fusion.GetFusedPointsVisibility();
  fusion.SetCheckIfStoppedFunc([] { return false; });
  for (int run = 0; run < 2; ++run) {
    fusion.Run();
    EXPECT_EQ(StereoFusionExactDepthTestPeer::NumRetiredFullRamInputs(fusion),
              2);
    EXPECT_TRUE(
        StereoFusionExactDepthTestPeer::FullRamInputsMatchWorkspace(fusion));
    ASSERT_EQ(fusion.GetFusedPoints().size(), partial.size());
    EXPECT_EQ(fusion.GetFusedPointsVisibility(), partial_visibility);
    for (size_t i = 0; i < partial.size(); ++i) {
      const auto& actual = fusion.GetFusedPoints()[i];
      const auto& expected = partial[i];
      EXPECT_TRUE(actual.x == expected.x && actual.y == expected.y &&
                  actual.z == expected.z && actual.nx == expected.nx &&
                  actual.ny == expected.ny && actual.nz == expected.nz &&
                  actual.r == expected.r && actual.g == expected.g &&
                  actual.b == expected.b);
    }
  }
}

TEST(StereoFusion, RejectedRootsDoNotLeakVisibility) {
  const auto root = CreateTwoViewNormalDecisionFixture();
  for (const int num_samples : {3, 1, 2}) {
    SCOPED_TRACE(num_samples);
    StereoFusionOptions options;
    options.num_threads = 1;
    options.use_cache = false;
    options.min_num_pixels = num_samples;
    options.max_num_pixels = num_samples;
    StereoFusion fusion(options, root, "COLMAP", "", "geometric");
    fusion.Run();

    const auto& points = fusion.GetFusedPoints();
    const auto& visibility = fusion.GetFusedPointsVisibility();
    if (num_samples == 3) {
      EXPECT_TRUE(points.empty());
      EXPECT_TRUE(visibility.empty());
      continue;
    }
    ASSERT_EQ(points.size(), num_samples == 1 ? 1200 : 600);
    ASSERT_EQ(visibility.size(), points.size());
    std::vector<size_t> counts(2, 0);
    for (auto visible : visibility) {
      ASSERT_EQ(visible.size(), num_samples);
      std::sort(visible.begin(), visible.end());
      if (num_samples == 2) {
        EXPECT_EQ(visible, (std::vector<int>{0, 1}));
      }
      for (const int image_idx : visible) {
        ASSERT_GE(image_idx, 0);
        ASSERT_LT(image_idx, 2);
        ++counts[image_idx];
      }
    }
    EXPECT_EQ(counts, (std::vector<size_t>{600, 600}));
  }
}

TEST(StereoFusion, RoundedByteMedianColors) {
  const auto root = CreateTwoViewNormalDecisionFixture();
  Model model;
  model.Read(root, "COLMAP");
  ASSERT_EQ(model.images.size(), 2);
  const std::vector<BitmapColor<uint8_t>> colors = {
      BitmapColor<uint8_t>(0, 0, 254), BitmapColor<uint8_t>(255, 1, 255)};
  for (size_t image_idx = 0; image_idx < model.images.size(); ++image_idx) {
    Bitmap bitmap(30, 20, true);
    bitmap.Fill(colors[image_idx]);
    ASSERT_TRUE(bitmap.Write(root / "images" / model.GetImageName(image_idx)));
  }

  for (const int num_samples : {2, 1}) {
    SCOPED_TRACE(num_samples);
    StereoFusionOptions options;
    options.num_threads = 1;
    options.use_cache = false;
    options.min_num_pixels = num_samples;
    options.max_num_pixels = num_samples;
    StereoFusion fusion(options, root, "COLMAP", "", "geometric");
    fusion.Run();

    const auto& points = fusion.GetFusedPoints();
    const auto& visibility = fusion.GetFusedPointsVisibility();
    ASSERT_EQ(points.size(), num_samples == 2 ? 600 : 1200);
    ASSERT_EQ(visibility.size(), points.size());
    std::vector<std::vector<int>> seen(2, std::vector<int>(600, 0));
    for (size_t point_idx = 0; point_idx < points.size(); ++point_idx) {
      const auto& point = points[point_idx];
      ASSERT_TRUE(std::isfinite(point.x) && std::isfinite(point.y) &&
                  std::isfinite(point.z));
      const int col = static_cast<int>(std::lround(point.x * 6 + 15 - 0.5f));
      const int row = static_cast<int>(std::lround(point.y * 6 + 10 - 0.5f));
      ASSERT_GE(col, 0);
      ASSERT_LT(col, 30);
      ASSERT_GE(row, 0);
      ASSERT_LT(row, 20);
      EXPECT_NEAR(point.x, (col + 0.5f - 15) / 6.0f, 1e-6f);
      EXPECT_NEAR(point.y, (row + 0.5f - 10) / 6.0f, 1e-6f);
      EXPECT_FLOAT_EQ(point.z, 5.0f);
      EXPECT_FLOAT_EQ(point.nx, 0.0f);
      EXPECT_FLOAT_EQ(point.ny, 0.0f);
      EXPECT_FLOAT_EQ(point.nz, 1.0f);

      auto visible = visibility[point_idx];
      ASSERT_EQ(visible.size(), num_samples);
      std::sort(visible.begin(), visible.end());
      if (num_samples == 2) {
        EXPECT_EQ(visible, (std::vector<int>{0, 1}));
        // Extreme averaging and half-integer medians round upward.
        EXPECT_EQ(point.r, 128);
        EXPECT_EQ(point.g, 1);
        EXPECT_EQ(point.b, 255);
      } else {
        ASSERT_GE(visible.front(), 0);
        ASSERT_LT(visible.front(), 2);
        const auto& expected = colors[visible.front()];
        EXPECT_EQ(point.r, expected.r);
        EXPECT_EQ(point.g, expected.g);
        EXPECT_EQ(point.b, expected.b);
      }
      for (const int image_idx : visible) {
        ASSERT_GE(image_idx, 0);
        ASSERT_LT(image_idx, 2);
        ++seen[image_idx][row * 30 + col];
      }
    }
    // Every input pixel contributes exactly once in both arms.
    for (const auto& image_seen : seen) {
      for (const int count : image_seen) EXPECT_EQ(count, 1);
    }
  }
}

TEST(StereoFusion, SignedNormalGuardRestoresLossyTwoViewDecision) {
  if (!ImmutableMapRegion::Supported())
    GTEST_SKIP() << "Immutable FP32 map owners are unsupported on this host";
  const auto root = CreateTwoViewNormalDecisionFixture();
  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.max_traversal_depth = 10;
  options.check_num_images = 10;
  options.num_threads = 1;
  options.max_normal_error = 0;
  StereoFusion exact(options, root, "COLMAP", "", "geometric");
  exact.Run();
  ASSERT_GT(exact.GetFusedPoints().size(), 0);
  options.bounded_cache = true;
  options.bounded_cache_components = true;
  options.cache_size = 1;
  options.bounded_cache_normal_tile_size = 8;
  options.bounded_cache_normal_tile_slots = 2;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 3;
  options.bounded_cache_normal_snorm16 = true;
  ASSERT_TRUE(options.Check());
  StereoFusion raw(options, root, "COLMAP", "", "geometric");
  raw.Run();
  EXPECT_LT(raw.GetFusedPoints().size(), exact.GetFusedPoints().size());
  if (!normal_precision_guard::SupportedBuild()) {
    options.bounded_cache_normal_snorm16_guard = true;
    EXPECT_FALSE(options.Check());
    return;
  }
  options.bounded_cache_normal_snorm16_guard = true;
  StereoFusion guarded(options, root, "COLMAP", "", "geometric");
  guarded.Run();
  const auto& expected = exact.GetFusedPoints();
  const auto& actual = guarded.GetFusedPoints();
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_EQ(guarded.GetFusedPointsVisibility(),
            exact.GetFusedPointsVisibility());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(actual[i].x, expected[i].x);
    EXPECT_FLOAT_EQ(actual[i].y, expected[i].y);
    EXPECT_FLOAT_EQ(actual[i].z, expected[i].z);
    EXPECT_EQ(actual[i].r, expected[i].r);
    EXPECT_EQ(actual[i].g, expected[i].g);
    EXPECT_EQ(actual[i].b, expected[i].b);
  }
  ASSERT_TRUE(guarded.GetBoundedCacheStats());
  EXPECT_EQ(guarded.GetBoundedCacheStats()->reserved_bytes, 0);
  EXPECT_EQ(guarded.GetMappedNormalStats().active_copies, 0);
}

TEST(StereoFusion, ResidentOct4GuardRestoresOriginalFileDecision) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  ASSERT_TRUE(normal_precision_guard::SupportedBuild());
  const auto root = CreateTwoViewNormalDecisionFixture();
  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.max_traversal_depth = 10;
  options.check_num_images = 10;
  options.num_threads = 1;
  options.full_ram_load_threads = 1;
  options.max_normal_error = 0;
  StereoFusion exact(options, root, "COLMAP", "", "geometric");
  exact.Run();
  ASSERT_FALSE(exact.GetFusedPoints().empty());
  options.full_ram_normal_storage = "oct2x16";
  StereoFusion lossy(options, root, "COLMAP", "", "geometric");
  lossy.Run();
  ASSERT_LT(lossy.GetFusedPoints().size(), exact.GetFusedPoints().size());
  options.full_ram_normal_guard = true;
  options.full_ram_normal_guard_bytes = 1 << 20;
  options.full_ram_normal_guard_files = 2;
  StereoFusion guarded(options, root, "COLMAP", "", "geometric");
  guarded.Run();
  ASSERT_EQ(guarded.GetFusedPoints().size(), exact.GetFusedPoints().size());
  EXPECT_EQ(guarded.GetFusedPointsVisibility(),
            exact.GetFusedPointsVisibility());
  for (size_t i = 0; i < exact.GetFusedPoints().size(); ++i) {
    const auto& actual = guarded.GetFusedPoints()[i];
    const auto& expected = exact.GetFusedPoints()[i];
    EXPECT_FLOAT_EQ(actual.x, expected.x);
    EXPECT_FLOAT_EQ(actual.y, expected.y);
    EXPECT_FLOAT_EQ(actual.z, expected.z);
    EXPECT_EQ(actual.r, expected.r);
    EXPECT_EQ(actual.g, expected.g);
    EXPECT_EQ(actual.b, expected.b);
  }
  EXPECT_FALSE(guarded.GetBoundedCacheStats());
  EXPECT_EQ(guarded.GetResidentNormalMappingStatsSnapshot().active_copies, 0);
  EXPECT_EQ(
      guarded.GetResidentNormalMappingStatsSnapshot().source_peak_live_files,
      2);
}

TEST(StereoFusion, ResidentCodecCertificateBoundsBothOperands) {
  ASSERT_TRUE(normal_precision_guard::SupportedBuild());
  using namespace normal_precision_guard;
  const Eigen::Vector3f original(0, 0, 0.5f);
  OctNormalPair packed;
  EncodeNormalOct2x16Unchecked(original, &packed);
  const auto decoded = DecodeNormalOct2x16(packed.x, packed.y);
  CodecErrorCertificate certificate(1);
  certificate.AddDecoded(original, decoded);
  const auto error = certificate.Seal();
  const Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
  const auto operand =
      BoundedOperandFromRotated(rotation, error, decoded, decoded);
  const auto missing =
      BoundedOperandFromRotated(rotation, Error::Zero(), decoded, decoded);
  EXPECT_EQ(ScreenNormal(missing, operand, 0.4f), Screen::kAccept);
  EXPECT_EQ(ScreenNormal(operand, missing, 0.4f), Screen::kAccept);
  EXPECT_EQ(ScreenNormal(operand, operand, 0.4f), Screen::kExact);
  for (const float threshold :
       {std::nextafter(0.25f, 0.f), 0.25f, std::nextafter(0.25f, 1.f), 0.4f}) {
    auto reference = operand;
    size_t root_reads = 0, candidate_reads = 0;
    const auto read_root = [&]() -> Eigen::Vector3f {
      ++root_reads;
      return original;
    };
    const auto read_candidate = [&]() -> Eigen::Vector3f {
      ++candidate_reads;
      return original;
    };
    EXPECT_EQ(
        NormalAccept(reference, operand, threshold, read_root, read_candidate),
        !(original.dot(original) < threshold));
    EXPECT_EQ(root_reads, 1);
    EXPECT_EQ(candidate_reads, 1);
    EXPECT_EQ(
        NormalAccept(reference, operand, threshold, read_root, read_candidate),
        !(original.dot(original) < threshold));
    EXPECT_EQ(root_reads, 1);
    EXPECT_EQ(candidate_reads, 2);
  }
}

TEST(StereoFusion, GuardFastUpMatchesLibmOnBoundaries) {
  if (!normal_precision_guard::SupportedBuild())
    GTEST_SKIP() << "Exact normal screen is unqualified for this compiler math";
  const double infinity = std::numeric_limits<double>::infinity();
  const double normal_min = std::numeric_limits<double>::min();
  const double denormal_min = std::numeric_limits<double>::denorm_min();
  const double cases[] = {0.0,
                          -0.0,
                          denormal_min,
                          -denormal_min,
                          normal_min,
                          1.0,
                          1.0 / 65534.0,
                          std::numeric_limits<double>::max(),
                          -1.0,
                          -infinity,
                          infinity};
  for (const double value : cases) {
    const double expected = std::nextafter(value, infinity);
    const double actual = normal_precision_guard::Up(value);
    EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(double)), 0)
        << "input=" << value;
  }
  EXPECT_TRUE(std::isnan(
      normal_precision_guard::Up(std::numeric_limits<double>::quiet_NaN())));
}

TEST(StereoFusion, GuardRotationReusePreservesOperandBitsAndDecision) {
  if (!normal_precision_guard::SupportedBuild())
    GTEST_SKIP() << "Exact normal screen is unqualified for this compiler math";
  using namespace normal_precision_guard;
  ASSERT_TRUE(SupportedArithmetic());
  const Rot rotations[] = {
      Rot::Identity(),
      Eigen::AngleAxisf(0.37f, Vec(1.0f, 2.0f, 3.0f).normalized())
          .toRotationMatrix()};
  const Vec decoded[] = {Vec(0.0f, 0.0f, 1.0f),
                         Vec(0.375f, -0.625f, 0.875f),
                         Vec(-1.0f, 0.25f, 0.0f)};
  // Independent conservative arithmetic: libm Up and the original r*decoded
  // expression. The production helper receives that exact rotated value.
  const auto reference_up = [](double x) {
    return std::nextafter(x, std::numeric_limits<double>::infinity());
  };
  const auto reference_add = [&](double a, double b) {
    return reference_up(a + b);
  };
  const auto reference_mul = [&](double a, double b) {
    return reference_up(a * b);
  };
  for (const Rot& rotation : rotations) {
    for (const Vec& normal : decoded) {
      const Vec rotated = rotation * normal;
      const Operand candidate =
          CompressedOperandFromRotated(rotation, normal, rotated);
      Operand reference{rotation * normal, Error::Zero(), false};
      const double e = reference_up(1.0 / 65534.0 + 0x1p-24);
      const double u = 0x1p-24;
      const double gamma = reference_up(7 * u / (1 - 7 * u));
      for (int i = 0; i < 3; ++i) {
        double delta = 0, magnitudes = 0;
        for (int j = 0; j < 3; ++j) {
          delta = reference_add(
              delta, reference_mul(std::abs(double(rotation(i, j))), e));
          magnitudes =
              reference_add(magnitudes,
                            reference_mul(std::abs(double(rotation(i, j))),
                                          std::abs(double(normal(j)))));
        }
        reference.error(i) = reference_add(
            reference_add(
                delta,
                reference_mul(gamma, reference_add(2 * magnitudes, delta))),
            32 * double(std::numeric_limits<float>::denorm_min()));
      }
      for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(std::memcmp(
                      &candidate.value(i), &reference.value(i), sizeof(float)),
                  0);
        EXPECT_EQ(std::memcmp(
                      &candidate.error(i), &reference.error(i), sizeof(double)),
                  0);
      }
      EXPECT_EQ(candidate.exact, reference.exact);
      EXPECT_EQ(ScreenNormal(candidate, candidate, 0.5f),
                ScreenNormal(reference, reference, 0.5f));
    }
  }
}

TEST(StereoFusion, GuardedRotationRefusesUnsupportedRounding) {
  if (!normal_precision_guard::SupportedBuild())
    GTEST_SKIP() << "Exact normal screen is unqualified for this compiler math";
  using namespace normal_precision_guard;
  const int saved_rounding = std::fegetround();
  ASSERT_NE(saved_rounding, -1);
  struct RoundingRestorer {
    int saved;
    ~RoundingRestorer() { std::fesetround(saved); }
  } restore{saved_rounding};
  ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
  EXPECT_FALSE(SupportedArithmetic());
  auto root = CompressedOperand(Rot::Identity(), Vec(0.0f, 0.0f, 1.0f));
  const auto candidate = CompressedOperandFromRotated(
      Rot::Identity(), Vec(0.0f, 0.0f, 1.0f), Vec(0.0f, 0.0f, 1.0f));
  int reads = 0;
  bool exact = false;
  EXPECT_TRUE(NormalAccept(
      root,
      candidate,
      0.5f,
      [&] {
        ++reads;
        return Vec(0.0f, 0.0f, 1.0f);
      },
      [&] {
        ++reads;
        return Vec(0.0f, 0.0f, 1.0f);
      },
      &exact));
  EXPECT_TRUE(exact);
  EXPECT_EQ(reads, 2);
}

#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

TEST(StereoFusion, PreparedNormalRotationMatchesOrdinaryBounds) {
  if (!normal_precision_guard::SupportedBuild())
    GTEST_SKIP() << "Exact normal screen is unqualified for this compiler math";
  using namespace normal_precision_guard;
  const auto check = [](const Rot& rotation, const Vec& decoded) {
    const Vec rotated = rotation * decoded;
    const auto ordinary =
        CompressedOperandFromRotated(rotation, decoded, rotated);
    const auto prepared = CompressedOperandFromRotated(
        rotation, PrepareRotation(rotation), decoded, rotated);
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(
          std::memcmp(&ordinary.error(i), &prepared.error(i), sizeof(double)),
          0);
    }
    for (const float threshold : {-1.0f, 0.0f, 0.984807753f, 1.0f}) {
      EXPECT_EQ(ScreenNormal(ordinary, ordinary, threshold),
                ScreenNormal(prepared, prepared, threshold));
    }
  };
  for (int sample = 0; sample < 128; ++sample) {
    SCOPED_TRACE(sample);
    Rot rotation;
    Vec decoded;
    for (int i = 0; i < 9; ++i) {
      rotation.data()[i] = float((sample * 31 + i * 17) % 257 - 128) / 128;
    }
    for (int i = 0; i < 3; ++i) {
      decoded(i) = float((sample * 43 + i * 19) % 251 - 125) / 125;
    }
    check(rotation, decoded);
  }
  for (const std::uint32_t bits : {0u,
                                   0x80000000u,
                                   1u,
                                   0x007fffffu,
                                   0x00800000u,
                                   0x7f7fffffu,
                                   0xff7fffffu,
                                   0x7f800000u,
                                   0xff800000u,
                                   0x7fc00001u}) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    for (int axis = 0; axis < 12; ++axis) {
      Rot rotation = Rot::Identity();
      Vec decoded(0.0f, 0.0f, 1.0f);
      if (axis < 9) {
        rotation.data()[axis] = value;
      } else {
        decoded(axis - 9) = value;
      }
      check(rotation, decoded);
    }
  }
}

TEST(StereoFusion, PreparedNormalRotationRechecksArithmeticControls) {
  if (!normal_precision_guard::SupportedBuild())
    GTEST_SKIP() << "Exact normal screen is unqualified for this compiler math";
  using namespace normal_precision_guard;
  struct ControlsRestorer {
    int rounding = std::fegetround();
#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    unsigned csr = _mm_getcsr();
#endif
    ~ControlsRestorer() {
      std::fesetround(rounding);
#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
      _mm_setcsr(csr);
#endif
    }
  } restore;
  ASSERT_NE(restore.rounding, -1);
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
  _mm_setcsr(_mm_getcsr() & ~((1u << 15) | (3u << 13) | (1u << 6)));
#endif
  const Rot rotation = Rot::Identity();
  const Vec decoded(0.25f, -0.5f, 1.0f);
  const Vec rotated = rotation * decoded;
  const auto prepared_nearest = PrepareRotation(rotation);
  ASSERT_TRUE(prepared_nearest.valid);
  for (const int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    if (std::fesetround(mode) != 0) {
      continue;
    }
    const auto prepared_unsupported = PrepareRotation(rotation);
    EXPECT_FALSE(prepared_unsupported.valid);
    const auto changed = CompressedOperandFromRotated(
        rotation, prepared_nearest, decoded, rotated);
    EXPECT_TRUE(
        (changed.error.array() == std::numeric_limits<double>::infinity())
            .all());
    EXPECT_EQ(ScreenNormal(changed, changed, 0.5f), Screen::kExact);
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    const auto restored = CompressedOperandFromRotated(
        rotation, prepared_unsupported, decoded, rotated);
    const auto ordinary =
        CompressedOperandFromRotated(rotation, decoded, rotated);
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(
          std::memcmp(&restored.error(i), &ordinary.error(i), sizeof(double)),
          0);
    }
  }
}

#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
TEST(StereoFusion, PreparedNormalRotationRechecksIndependentSseControls) {
  if (!normal_precision_guard::SupportedBuild())
    GTEST_SKIP() << "Exact normal screen is unqualified for this compiler math";
  using namespace normal_precision_guard;
  struct ControlsRestorer {
    int rounding = std::fegetround();
    unsigned csr = _mm_getcsr();
    ~ControlsRestorer() {
      std::fesetround(rounding);
      _mm_setcsr(csr);
    }
  } restore;
  ASSERT_NE(restore.rounding, -1);
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  const unsigned nearest_csr =
      _mm_getcsr() & ~((1u << 15) | (3u << 13) | (1u << 6));
  _mm_setcsr(nearest_csr);
  const Rot rotation = Rot::Identity();
  const Vec decoded(0.25f, -0.5f, 1.0f);
  const Vec rotated = rotation * decoded;
  const auto prepared_nearest = PrepareRotation(rotation);
  ASSERT_TRUE(prepared_nearest.valid);
  for (const unsigned controls : {1u << 13,
                                  2u << 13,
                                  3u << 13,
                                  1u << 15,
                                  1u << 6,
                                  (1u << 15) | (1u << 6)}) {
    SCOPED_TRACE(controls);
    _mm_setcsr(nearest_csr | controls);
    for (const bool prepare_now : {false, true}) {
      const auto prepared =
          prepare_now ? PrepareRotation(rotation) : prepared_nearest;
      if (prepare_now) {
        EXPECT_FALSE(prepared.valid);
      }
      const auto unsupported =
          CompressedOperandFromRotated(rotation, prepared, decoded, rotated);
      EXPECT_TRUE(
          (unsupported.error.array() == std::numeric_limits<double>::infinity())
              .all());
      EXPECT_EQ(ScreenNormal(unsupported, unsupported, 0.5f), Screen::kExact);
    }
    const auto prepared_unsupported = PrepareRotation(rotation);
    _mm_setcsr(nearest_csr);
    const auto restored = CompressedOperandFromRotated(
        rotation, prepared_unsupported, decoded, rotated);
    const auto ordinary =
        CompressedOperandFromRotated(rotation, decoded, rotated);
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(
          std::memcmp(&restored.error(i), &ordinary.error(i), sizeof(double)),
          0);
    }
  }
}
#endif

TEST(StereoFusionOptions, CachePoliciesAreMutuallyExclusive) {
  StereoFusionOptions options;
  EXPECT_TRUE(options.Check());
  options.use_cache = true;
  EXPECT_TRUE(options.Check());
  options.bounded_cache = true;
  EXPECT_FALSE(options.Check());
  options.use_cache = false;
  EXPECT_TRUE(options.Check());
}

TEST(StereoFusionOptions, SignedNormalGuardIsExplicitlyBounded) {
  StereoFusionOptions options;
  EXPECT_FALSE(options.bounded_cache_normal_snorm16);
  EXPECT_FALSE(options.bounded_cache_normal_snorm16_guard);
  options.bounded_cache_normal_snorm16_guard = true;
  EXPECT_FALSE(options.Check());
  options.bounded_cache_normal_snorm16 = true;
  EXPECT_FALSE(options.Check());
  options.bounded_cache = true;
  options.bounded_cache_components = true;
  options.bounded_cache_normal_tile_size = 8;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 2;
  EXPECT_EQ(options.Check(), normal_precision_guard::SupportedBuild());
  options.bounded_cache_normal_snorm16_guard = false;
  EXPECT_TRUE(options.Check());  // Approximate SNORM remains independent.
}

TEST(StereoFusion, UnsupportedBuildNeverScreensNormalDecisions) {
  using namespace normal_precision_guard;
  if (SupportedBuild())
    GTEST_SKIP() << "This regression exercises an unsupported compiler build";
  EXPECT_FALSE(SupportedArithmetic());
  Operand a{Vec(0, 0, 1), Error::Zero(), false};
  Operand b{Vec(0, 0, -1), Error::Zero(), false};
  // Even an obvious rejection must defer to the original FP32 callback.
  EXPECT_EQ(ScreenNormal(a, b, 0.0f), Screen::kExact);
  b.value = a.value;
  EXPECT_EQ(ScreenNormal(a, b, 0.0f), Screen::kExact);
  EXPECT_FALSE(PrepareRotation(Rot::Identity()).valid);
}

TEST(StereoFusion, SignedNormalExactCallbackOwnsRotatedTemporary) {
  using namespace normal_precision_guard;
  Rot rotation;
  rotation << 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f;
  const Vec local(1.0f, 0.0f, 0.0f);
  const Vec expected = rotation * local;
  auto root = CompressedOperand(rotation, local);
  const auto candidate = CompressedOperand(rotation, local);
  int reads = 0;
  const auto read_rotated = [&]() -> Vec {
    ++reads;
    // Match production's by-value source, which a lazy return would outlive.
    return rotation * Vec(1.0f, 0.0f, 0.0f);
  };
  bool used_exact = false;
  EXPECT_TRUE(NormalAccept(
      root, candidate, 1.0f, read_rotated, read_rotated, &used_exact));
  EXPECT_TRUE(used_exact);
  EXPECT_EQ(reads, 2);
  EXPECT_EQ(root.value, expected);
  EXPECT_TRUE(root.exact);
}

TEST(StereoFusion, SignedNormalDecisionFallsBackAtBoundary) {
  using namespace normal_precision_guard;
  const Rot rotation = Rot::Identity();
  const Vec root(0.0f, 0.0f, 1.0f);
  const Vec candidate(0.0f, 0.0f, 1.0f);
  auto operand = CompressedOperand(rotation, root);
  const auto next = CompressedOperand(rotation, candidate);
  int reads = 0;
  bool used_exact = false;
  EXPECT_TRUE(NormalAccept(
      operand,
      next,
      1.0f,
      [&] {
        ++reads;
        return root;
      },
      [&] {
        ++reads;
        return candidate;
      },
      &used_exact));
  EXPECT_TRUE(used_exact);
  EXPECT_EQ(reads, 2);
  EXPECT_TRUE(operand.exact);
  // A raw SNORM decode rounds 0.99999 to 1 and would accept equality here.
  // The guarded decision must consult the original FP32 normal and reject.
  const Vec original_below(0.0f, 0.0f, 0.99999f);
  const Vec decoded_below(0.0f, 0.0f, 1.0f);
  auto exact_root = CompressedOperand(rotation, root);
  const auto rounded_candidate = CompressedOperand(rotation, decoded_below);
  int exact_reads = 0;
  EXPECT_FALSE(NormalAccept(
      exact_root,
      rounded_candidate,
      1.0f,
      [&] {
        ++exact_reads;
        return root;
      },
      [&] {
        ++exact_reads;
        return original_below;
      },
      &used_exact));
  EXPECT_TRUE(used_exact);
  EXPECT_EQ(exact_reads, 2);
}

// Compare compact resident maps with their decoded FP32 equivalents.
TEST(StereoFusion, FullRamNormalStorageDecodedFileOracle) {
  const auto root = CreateTestDir() / "fusion";
  const int num_clusters = 1;
  ASSERT_TRUE(!std::filesystem::exists(root))
      << "plane fixture root must be fresh";
  for (const auto& dir : {root / "sparse",
                          root / "images",
                          root / "stereo" / "depth_maps",
                          root / "stereo" / "normal_maps"}) {
    std::filesystem::create_directories(dir);
  }
  std::ofstream cameras(root / "sparse" / "cameras.txt");
  cameras << "1 PINHOLE 96 96 96 96 48 48\n";
  cameras.close();
  std::ofstream images(root / "sparse" / "images.txt");
  std::ofstream points(root / "sparse" / "points3D.txt");
  std::ofstream roster(root / "stereo" / "fusion.cfg");
  const int map_dimensions[] = {96, 48, 96, 72, 96};
  const int num_views = 5 * num_clusters;
  for (int i = 0; i < num_views; ++i) {
    const int cluster = i / 5;
    const int local_view = i % 5;
    const double cluster_base = 4 * cluster;
    const double local_center = (local_view - 2) * 0.04;
    const double center_x = cluster_base + local_center;
    const std::string name = "plane-" + std::to_string(i) + ".png";
    images << i + 1 << " 1 0 0 0 " << -center_x << " 0 0 1 " << name << '\n';
    for (int k = 0; k < 9; ++k) {
      const double x = cluster_base + (k % 3 - 1) * 0.2;
      const double y = (k / 3 - 1) * 0.2;
      images << 48 + 48 * (x - center_x) << ' ' << 48 + 48 * y << ' '
             << cluster * 9 + k + 1 << ' ';
    }
    images << '\n';
    roster << name << '\n';
    colmap::Bitmap bitmap(96, 96, true);
    for (int row = 0; row < 96; ++row) {
      for (int col = 0; col < 96; ++col) {
        bitmap.SetPixel(col,
                        row,
                        colmap::BitmapColor<uint8_t>(
                            std::round(col + 80 + 48 * local_center),
                            row + 80,
                            32 + 16 * local_view + cluster));
      }
    }
    ASSERT_TRUE(bitmap.Write(root / "images" / name)) << "plane bitmap write";
    const size_t map_size = num_clusters == 1 ? map_dimensions[i] : 96;
    Mat<float> depth(map_size, map_size, 1);
    depth.Fill(2.0f);
    depth.Write(root / "stereo" / "depth_maps" / (name + ".geometric.bin"));
    Mat<float> normal(map_size, map_size, 3);
    normal.Fill(0.0f);
    for (size_t row = 0; row < map_size; ++row) {
      for (size_t col = 0; col < map_size; ++col)
        normal.Set(row, col, 2, -1.0f);
    }
    normal.Write(root / "stereo" / "normal_maps" / (name + ".geometric.bin"));
  }
  for (int cluster = 0; cluster < num_clusters; ++cluster) {
    for (int k = 0; k < 9; ++k) {
      points << cluster * 9 + k + 1 << ' ' << 4 * cluster + (k % 3 - 1) * 0.2
             << ' ' << (k / 3 - 1) * 0.2 << " 2 12 34 56 0";
      for (int i = 0; i < 5; ++i)
        points << ' ' << cluster * 5 + i + 1 << ' ' << k;
      points << '\n';
    }
  }
  images.close();
  points.close();
  roster.close();
  Workspace::Options workspace_options;
  workspace_options.workspace_path = root;
  workspace_options.workspace_format = "COLMAP";
  workspace_options.input_type = "geometric";
  Workspace descriptor(workspace_options);
  std::vector<Mat<float>> originals;
  std::vector<std::string> names;
  for (size_t view = 0; view < descriptor.GetModel().images.size(); ++view) {
    names.push_back(descriptor.GetModel().GetImageName(view));
    Mat<float> normal;
    normal.Read(descriptor.GetNormalMapPath(view));
    for (size_t row = 0; row < normal.GetHeight(); ++row)
      for (size_t col = 0; col < normal.GetWidth(); ++col) {
        normal.Set(row, col, 0, 0.00017f * col + 0.00031f * view);
        normal.Set(row, col, 1, -0.00013f * row);
        normal.Set(row, col, 2, -0.995f);
      }
    // Explicit zero rejection is part of the complete-fusion oracle.
    for (size_t channel = 0; channel < 3; ++channel)
      normal.Set(0, 0, channel, 0);
    normal.Write(descriptor.GetNormalMapPath(view));
    originals.push_back(normal);
  }
  colmap::mvs::StereoFusionOptions options;
  options.num_threads = 1;
  options.full_ram_load_threads = 4;
  options.min_num_pixels = 2;
  // Also exercise the native full-RAM resize-before-encode route.
  options.max_image_size = 48;
  for (const auto* name : {"oct2x16"}) {
    options.full_ram_normal_storage = name;
    colmap::mvs::StereoFusion candidate(
        options, root, "COLMAP", "", "geometric");
    candidate.Run();
    ASSERT_TRUE(!candidate.GetFusedPoints().empty() &&
                !candidate.GetBoundedCacheStats())
        << "full-RAM codec produced no points or selected cache machinery";
    // Explicit decoded files use existing scalar codec primitives, independent
    // of Workspace's pack/publication branch. Resize once before encoding.
    auto consumed_options = workspace_options;
    consumed_options.max_image_size = options.max_image_size;
    Workspace consumed(consumed_options);
    consumed.Load(names);
    for (size_t view = 0; view < originals.size(); ++view) {
      const auto& source = consumed.GetNormalMap(view);
      Mat<float> decoded(source.GetWidth(), source.GetHeight(), 3);
      for (size_t row = 0; row < source.GetHeight(); ++row)
        for (size_t col = 0; col < source.GetWidth(); ++col) {
          Eigen::Vector3f value(source.Get(row, col, 0),
                                source.Get(row, col, 1),
                                source.Get(row, col, 2));
          OctNormalPair encoded;
          EncodeNormalOct2x16Unchecked(value, &encoded);
          value = DecodeNormalOct2x16(encoded.x, encoded.y);
          for (size_t channel = 0; channel < 3; ++channel)
            decoded.Set(row, col, channel, value[channel]);
        }
      decoded.Write(descriptor.GetNormalMapPath(view));
    }
    auto reference_options = options;
    reference_options.full_ram_normal_storage = "fp32";
    colmap::mvs::StereoFusion reference(
        reference_options, root, "COLMAP", "", "geometric");
    reference.Run();
    ASSERT_TRUE(reference.GetFusedPointsVisibility() ==
                    candidate.GetFusedPointsVisibility() &&
                reference.GetFusedPoints().size() ==
                    candidate.GetFusedPoints().size())
        << "full-RAM compact fusion differs from decoded-file contributor "
           "oracle";
    for (size_t index = 0; index < reference.GetFusedPoints().size(); ++index) {
      const auto& a = reference.GetFusedPoints()[index];
      const auto& b = candidate.GetFusedPoints()[index];
      ASSERT_TRUE(a.x == b.x && a.y == b.y && a.z == b.z && a.nx == b.nx &&
                  a.ny == b.ny && a.nz == b.nz && a.r == b.r && a.g == b.g &&
                  a.b == b.b)
          << "full-RAM compact fusion differs from decoded-file FP32 "
             "arithmetic";
    }
    for (size_t view = 0; view < originals.size(); ++view)
      originals[view].Write(descriptor.GetNormalMapPath(view));
  }
}

TEST(StereoFusionOptions, ResidentOct4GuardBoundedWorkerPolicy) {
  StereoFusionOptions options;
  EXPECT_FALSE(options.full_ram_normal_guard);
  options.full_ram_normal_storage = "oct2x16";
  options.full_ram_normal_guard = true;
  options.full_ram_normal_guard_bytes = 1 << 20;
  options.full_ram_normal_guard_files = 24;
  options.full_ram_load_threads = 1;
  for (const int workers : {1, 2, 24}) {
    options.num_threads = workers;
    EXPECT_EQ(options.Check(), normal_precision_guard::SupportedBuild());
  }
  for (const int workers : {-1, 0, 25}) {
    options.num_threads = workers;
    EXPECT_FALSE(options.Check());
  }
  options.num_threads = 24;
  for (const int loaders : {-1, 2, 24}) {
    options.full_ram_load_threads = loaders;
    EXPECT_FALSE(options.Check());
  }
  options.full_ram_load_threads = 1;
  options.full_ram_normal_storage = "fp32";
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusion, ResidentOct4Guard24WorkersOriginalFileDecision) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  ASSERT_TRUE(normal_precision_guard::SupportedBuild());
  const auto root = CreateTwoViewNormalDecisionFixture();
  // Height 240 supplies 24 native ten-row tasks; every pixel requires the
  // exact two-view boundary decision. Distinct rows have no competing pixels.
  Reconstruction reconstruction;
  reconstruction.Read(root / "sparse");
  for (const auto& [camera_id, camera] : reconstruction.Cameras())
    reconstruction.Camera(camera_id).height = 240;
  reconstruction.Write(root / "sparse");
  int view = 0;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    Mat<float> depth(30, 240, 1);
    depth.Fill(5.0f);
    depth.Write(root / "stereo/depth_maps" / (image.Name() + ".geometric.bin"));
    Mat<float> normal(30, 240, 3);
    normal.Fill(0.0f);
    for (size_t row = 0; row < 240; ++row)
      for (size_t col = 0; col < 30; ++col)
        normal.Set(row, col, 2, view == 0 ? 1.0f : 0.99999f);
    normal.Write(root / "stereo/normal_maps" /
                 (image.Name() + ".geometric.bin"));
    Bitmap bitmap(30, 240, true);
    bitmap.Fill(BitmapColor<uint8_t>(0, 64, 128));
    ASSERT_TRUE(bitmap.Write(root / "images" / image.Name()));
    ++view;
  }
  using Record = std::
      tuple<float, float, float, uint8_t, uint8_t, uint8_t, std::vector<int>>;
  const auto capture = [](const StereoFusion& fusion) {
    std::vector<Record> records;
    const auto& points = fusion.GetFusedPoints();
    const auto& vis = fusion.GetFusedPointsVisibility();
    for (size_t i = 0; i < points.size(); ++i) {
      const auto& p = points[i];
      EXPECT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) &&
                  std::isfinite(p.z) && std::isfinite(p.nx) &&
                  std::isfinite(p.ny) && std::isfinite(p.nz));
      records.emplace_back(p.x, p.y, p.z, p.r, p.g, p.b, vis.at(i));
    }
    std::sort(records.begin(), records.end());
    return records;
  };
  StereoFusionOptions options;
  options.num_threads = 24;
  options.full_ram_load_threads = 1;
  options.min_num_pixels = 1;
  options.max_normal_error = 0;
  StereoFusion exact(options, root, "COLMAP", "", "geometric");
  exact.Run();
  ASSERT_EQ(exact.GetFusedPoints().size(), 2 * 30 * 240);
  const auto expected = capture(exact);
  options.full_ram_normal_storage = "oct2x16";
  options.full_ram_normal_guard = true;
  options.full_ram_normal_guard_files = 24;
  auto source = std::make_shared<ImmutableMapSource>(
      root / "stereo/normal_maps" /
          (reconstruction.Images().begin()->second.Name() + ".geometric.bin"),
      ImmutableMapSource::BindIdentity{});
  size_t maximum = 0;
  for (size_t row = 0; row < 240; ++row)
    maximum = std::max(maximum, source->GetWindowReservationBytes(row, 1));
  options.full_ram_normal_guard_bytes = static_cast<int64_t>(24 * maximum);
  StereoFusion guarded(options, root, "COLMAP", "", "geometric");
  for (int run = 0; run < 2; ++run) {
    guarded.Run();
    EXPECT_EQ(capture(guarded), expected);
    EXPECT_FALSE(guarded.GetBoundedCacheStats());
    const auto stats = guarded.GetResidentNormalMappingStatsSnapshot();
    EXPECT_EQ(stats.loads, 0);
    EXPECT_EQ(stats.resident_bytes, 0);
    EXPECT_EQ(stats.active_copies, 0);
    EXPECT_EQ(stats.reserved_bytes, 0);
    EXPECT_EQ(stats.reserved_files, 0);
    EXPECT_EQ(stats.reserved_regions, 0);
    EXPECT_LE(stats.peak_charged_bytes, 24 * maximum);
    EXPECT_LE(stats.peak_charged_files, 24);
    EXPECT_LE(stats.peak_charged_regions, 72);
    EXPECT_EQ(stats.source_peak_live_files, 2);
  }
}

TEST(StereoFusion, PreparedCodecRotationPreservesBoundsAndExactCallbacks) {
  ASSERT_TRUE(normal_precision_guard::SupportedArithmetic());
  using namespace normal_precision_guard;
  const auto check = [](const Rot& r, const Error& error, const Vec& decoded) {
    const Vec rotated = r * decoded;
    const auto ordinary = BoundedOperandFromRotated(r, error, decoded, rotated);
    const auto cached = BoundedOperandFromRotated(
        r, error, PrepareCodecRotation(r, error), decoded, rotated);
    EXPECT_EQ(
        std::memcmp(
            ordinary.value.data(), cached.value.data(), 3 * sizeof(float)),
        0);
    EXPECT_EQ(
        std::memcmp(
            ordinary.error.data(), cached.error.data(), 3 * sizeof(double)),
        0);
    EXPECT_EQ(ordinary.exact, cached.exact);
    const float dot = ordinary.value.dot(ordinary.value);
    for (const float threshold :
         {-1.0f,
          0.0f,
          0.984807753f,
          1.0f,
          dot,
          std::nextafter(dot, -std::numeric_limits<float>::infinity()),
          std::nextafter(dot, std::numeric_limits<float>::infinity())}) {
      EXPECT_EQ(ScreenNormal(ordinary, ordinary, threshold),
                ScreenNormal(cached, cached, threshold));
      if (!rotated.allFinite() || !std::isfinite(dot)) continue;
      auto a = ordinary, b = cached;
      int reads_a = 0, reads_b = 0;
      bool exact_a = false, exact_b = false;
      const bool decision_a = NormalAccept(
          a,
          ordinary,
          threshold,
          [&]() -> Vec {
            ++reads_a;
            return rotated;
          },
          [&]() -> Vec {
            ++reads_a;
            return rotated;
          },
          &exact_a);
      const bool decision_b = NormalAccept(
          b,
          cached,
          threshold,
          [&]() -> Vec {
            ++reads_b;
            return rotated;
          },
          [&]() -> Vec {
            ++reads_b;
            return rotated;
          },
          &exact_b);
      EXPECT_EQ(decision_a, decision_b);
      EXPECT_EQ(reads_a, reads_b);
      EXPECT_EQ(exact_a, exact_b);
      EXPECT_EQ(a.exact, b.exact);
    }
  };
  for (int sample = 0; sample < 128; ++sample) {
    Rot r;
    Vec decoded;
    Error error;
    for (int i = 0; i < 9; ++i)
      r.data()[i] = float((sample * 31 + i * 17) % 257 - 128) / 128;
    for (int i = 0; i < 3; ++i) {
      decoded(i) = float((sample * 43 + i * 19) % 251 - 125) / 125;
      error(i) = (sample % 4 == 0)
                     ? 0.0
                     : std::ldexp(double((sample * 7 + i * 13) % 101), -20 - i);
    }
    check(r, error, decoded);
  }
  const std::vector<Vec> originals{{0.0f, 0.0f, 0.0f},
                                   {0.2f, -0.3f, 0.7f},
                                   {-0.4f, 0.1f, -0.2f},
                                   {1.0f, -1.0f, 1.0f}};
  CodecErrorCertificate certificate(originals.size());
  std::vector<Vec> actual_decodes;
  for (const auto& original : originals) {
    OctNormalPair pair;
    EncodeNormalOct2x16Unchecked(original, &pair);
    const Vec decoded = DecodeNormalOct2x16(pair.x, pair.y);
    certificate.AddDecoded(original, decoded);
    actual_decodes.push_back(decoded);
  }
  const Error certified_error = certificate.Seal();
  for (const auto& decoded : actual_decodes)
    check(Rot::Identity(), certified_error, decoded);
  for (const double e : {0.0,
                         std::numeric_limits<double>::denorm_min(),
                         0x1p-24,
                         1.0,
                         std::numeric_limits<double>::max()}) {
    check(Rot::Identity(), Error::Constant(e), Vec(0.0f, -0.0f, 1.0f));
    check(Rot::Identity(),
          Error::Constant(e),
          Vec(std::numeric_limits<float>::denorm_min(), 0.0f, 1.0f));
    check(Rot::Identity(),
          Error::Constant(e),
          Vec::Constant(std::numeric_limits<float>::max()));
  }
}

TEST(StereoFusion, PreparedCodecRotationRefusesStaleOrInvalidTerms) {
  ASSERT_TRUE(normal_precision_guard::SupportedArithmetic());
  using namespace normal_precision_guard;
  const Rot r = Rot::Identity();
  const Error error(0x1p-20, 0x1p-21, 0x1p-22);
  const auto prepared = PrepareCodecRotation(r, error);
  ASSERT_TRUE(prepared.valid());
  const Vec decoded(0.2f, -0.3f, 0.7f);
  const auto check = [&](const Rot& actual_r,
                         const Error& actual_error,
                         const PreparedCodecRotation& cache) {
    const Vec rotated = actual_r * decoded;
    const auto ordinary =
        BoundedOperandFromRotated(actual_r, actual_error, decoded, rotated);
    const auto cached = BoundedOperandFromRotated(
        actual_r, actual_error, cache, decoded, rotated);
    EXPECT_EQ(
        std::memcmp(
            ordinary.error.data(), cached.error.data(), 3 * sizeof(double)),
        0);
  };
  Rot changed_r = r;
  changed_r(0, 1) = 0.25f;
  check(changed_r, error, prepared);
  changed_r = r;
  changed_r(0, 1) = -0.0f;  // Bitwise binding also covers signed zero.
  check(changed_r, error, prepared);
  Error changed_error = error;
  changed_error(1) *= 1024;
  check(r, changed_error, prepared);
  check(r, error, PreparedCodecRotation{});
  for (const double bad : {-1.0,
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    const Error invalid = Error::Constant(bad);
    const auto refused = PrepareCodecRotation(r, invalid);
    EXPECT_FALSE(refused.valid());
    check(r, invalid, refused);
  }
  changed_r(0, 0) = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(PrepareCodecRotation(changed_r, error).valid());
  check(changed_r, error, prepared);
}

TEST(StereoFusion, PreparedCodecRotationRechecksAllArithmeticControls) {
  ASSERT_TRUE(normal_precision_guard::SupportedArithmetic());
  using namespace normal_precision_guard;
  struct Restore {
    int rounding = std::fegetround();
#if defined(__SSE__)
    unsigned int sse = _mm_getcsr();
#endif
    ~Restore() {
      std::fesetround(rounding);
#if defined(__SSE__)
      _mm_setcsr(sse);
#endif
    }
  } restore;
  const Rot r = Rot::Identity();
  const Error error = Error::Constant(0x1p-20);
  const auto prepared = PrepareCodecRotation(r, error);
  const Vec decoded(0.2f, -0.3f, 0.7f), rotated = r * decoded;
  const auto check = [&](const PreparedCodecRotation& cache) {
    const auto ordinary = BoundedOperandFromRotated(r, error, decoded, rotated);
    const auto cached =
        BoundedOperandFromRotated(r, error, cache, decoded, rotated);
    EXPECT_EQ(
        std::memcmp(
            ordinary.error.data(), cached.error.data(), 3 * sizeof(double)),
        0);
    EXPECT_EQ(ScreenNormal(ordinary, ordinary, 1.0f),
              ScreenNormal(cached, cached, 1.0f));
  };
  for (const int mode : {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    EXPECT_FALSE(SupportedArithmetic());
    check(prepared);
    const auto unavailable = PrepareCodecRotation(r, error);
    EXPECT_FALSE(unavailable.valid());
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    check(unavailable);  // Invalid preparation remains the ordinary path.
  }
#if defined(__SSE__)
  const unsigned int nearest =
      _mm_getcsr() & ~((1u << 15) | (3u << 13) | (1u << 6));
  for (const unsigned int bits :
       {1u << 15, 1u << 6, 1u << 13, 2u << 13, 3u << 13}) {
    _mm_setcsr(nearest | bits);
    EXPECT_FALSE(SupportedArithmetic());
    check(prepared);
    EXPECT_FALSE(PrepareCodecRotation(r, error).valid());
    _mm_setcsr(nearest);
  }
#endif
  check(prepared);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
