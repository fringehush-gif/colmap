// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/fusion.h"

#include "colmap/math/math.h"
#include "colmap/mvs/model.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/bitmap_reader.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#ifdef __linux__
#include <csignal>

#include <sys/resource.h>
#endif

namespace colmap {
namespace mvs {
namespace {
std::vector<char> UnpackSnapshot(
    const internal::FusionPixelMask::PackedSnapshot& snapshot) {
  std::vector<char> result(snapshot.size(), 1);
  snapshot.ForEachUnset([&](size_t offset) { result.at(offset) = 0; });
  return result;
}
}  // namespace

TEST(StereoFusion, ResidentNonpositiveDepthSeedsMaskWithoutChangingSpecials) {
  if (sizeof(float) != sizeof(uint32_t) ||
      !std::numeric_limits<float>::is_iec559) {
    GTEST_SKIP() << "The bit-pattern oracle requires IEEE binary32";
  }
  const float probe = 0x1.2468acp0f;
  uint32_t probe_bits;
  std::memcpy(&probe_bits, &probe, sizeof(probe_bits));
  if (probe_bits != 0x3f923456u) {
    GTEST_SKIP() << "The bit-pattern oracle requires matching float/int layout";
  }
  constexpr size_t kWidth = 13, kHeight = 7, kSize = kWidth * kHeight;
  const std::vector<uint32_t> patterns = {0x00000000u,
                                          0x80000000u,
                                          0xbf800000u,
                                          0x3f800000u,
                                          0xff800000u,
                                          0x7f800000u,
                                          0xff7fffffu,
                                          0x7f7fffffu,
                                          0x80800000u,
                                          0x00800000u,
                                          0x80000001u,
                                          0x00000001u,
                                          0xffc12345u,
                                          0x7fc12345u,
                                          0xff812345u,
                                          0x7f812345u};
  Mat<float> depth(kWidth, kHeight, 1);
  std::vector<char> expected(kSize);
  for (size_t index = 0; index < kSize; ++index) {
    const uint32_t bits = patterns[index % patterns.size()];
    float value;
    std::memcpy(&value, &bits, sizeof(bits));
    depth.Set(index / kWidth, index % kWidth, value);
    // Pattern indices intentionally specify the independent expected oracle.
    const size_t pattern = index % patterns.size();
    expected[index] = pattern == 0 || pattern == 1 || pattern == 2 ||
                      pattern == 4 || pattern == 6 || pattern == 8;
  }
  expected[3] = 1;
  const auto original_depth = depth.GetData();
  std::fenv_t original;
  ASSERT_EQ(std::fegetenv(&original), 0);
  struct RestoreEnvironment {
    std::fenv_t saved;
    ~RestoreEnvironment() { std::fesetenv(&saved); }
  } restore{original};
  for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    internal::FusionPixelMask mask(kWidth, kHeight);
    mask.Set(0, 3, 1);  // Retain a PNG exclusion on positive depth.
    ASSERT_EQ(std::fesetround(mode), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    mask.ExcludeNonpositiveDepth(internal::FusionFloatMapView(depth));
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0);
    EXPECT_EQ(UnpackSnapshot(mask.Snapshot(0, kSize)), expected);
  }
  EXPECT_EQ(
      std::memcmp(
          depth.GetData().data(), original_depth.data(), kSize * sizeof(float)),
      0);
  auto view = internal::FusionFloatMapView(depth);
  for (size_t failure = 0; failure < 4; ++failure) {
    auto bad = view;
    if (failure == 0) --bad.width;
    if (failure == 1) --bad.plane;
    if (failure == 2) --bad.size;
    if (failure == 3) bad.data = nullptr;
    internal::FusionPixelMask untouched(kWidth, kHeight);
    untouched.ExcludeNonpositiveDepth(bad);
    EXPECT_EQ(UnpackSnapshot(untouched.Snapshot(0, kSize)),
              std::vector<char>(kSize, 0));
  }
  internal::FusionPixelMask empty;
  empty.ExcludeNonpositiveDepth(internal::FusionFloatMapView());
  EXPECT_TRUE(UnpackSnapshot(empty.Snapshot(0, 0)).empty());
}

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
  EXPECT_TRUE(UnpackSnapshot(empty.Snapshot(0, 0)).empty());
  EXPECT_THROW(empty.Get(0, 0), std::out_of_range);
  std::vector<internal::FusionPixelMask> masks(1);
  masks[0] = internal::FusionPixelMask(3, 2);
  EXPECT_EQ(UnpackSnapshot(masks[0].Snapshot(0, 6)), std::vector<char>(6, 0));
  masks[0].Set(0, 1, 1);
  masks[0].Set(1, 2, 1);
  masks.resize(4);
  EXPECT_EQ(masks[0].Get(0, 1), 1);
  EXPECT_EQ(masks[0].Get(1, 0), 0);
  EXPECT_EQ(UnpackSnapshot(masks[0].Snapshot(1, 6)),
            (std::vector<char>{1, 0, 0, 0, 1}));
  EXPECT_TRUE(UnpackSnapshot(masks[0].Snapshot(6, 6)).empty());
  EXPECT_THROW(UnpackSnapshot(masks[0].Snapshot(4, 3)), std::out_of_range);
  EXPECT_THROW(UnpackSnapshot(masks[0].Snapshot(2, 7)), std::out_of_range);
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
  EXPECT_EQ(UnpackSnapshot(mask.Snapshot(0, kPixels)), expected);
  EXPECT_EQ(UnpackSnapshot(mask.Snapshot(61, 67)),
            (std::vector<char>{0, 0, 1, 1, 0, 0}));
  EXPECT_EQ(UnpackSnapshot(mask.Snapshot(kPixels - 2, kPixels)),
            (std::vector<char>{0, 1}));
  EXPECT_TRUE(UnpackSnapshot(mask.Snapshot(kPixels, kPixels)).empty());
  EXPECT_THROW(UnpackSnapshot(mask.Snapshot(kPixels + 1, kPixels + 1)),
               std::out_of_range);
  EXPECT_THROW(UnpackSnapshot(mask.Snapshot(64, 63)), std::out_of_range);
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
  EXPECT_EQ(UnpackSnapshot(mask.Snapshot(0, kPixels)), expected);
  EXPECT_THROW(mask.Set(0, 0, 2), std::invalid_argument);
  EXPECT_EQ(UnpackSnapshot(mask.Snapshot(0, kPixels)), expected);

  internal::FusionPixelMask moved(std::move(mask));
  EXPECT_EQ(UnpackSnapshot(moved.Snapshot(0, kPixels)), expected);
  EXPECT_TRUE(UnpackSnapshot(mask.Snapshot(0, 0)).empty());
  EXPECT_THROW(UnpackSnapshot(mask.Snapshot(1, 1)), std::out_of_range);
  EXPECT_THROW(mask.Get(0, 0), std::out_of_range);
  internal::FusionPixelMask assigned(1, 1);
  assigned = std::move(moved);
  EXPECT_EQ(UnpackSnapshot(assigned.Snapshot(0, kPixels)), expected);
  EXPECT_TRUE(UnpackSnapshot(moved.Snapshot(0, 0)).empty());
  EXPECT_THROW(UnpackSnapshot(moved.Snapshot(1, 1)), std::out_of_range);
  EXPECT_THROW(moved.Set(0, 0, 1), std::out_of_range);
  internal::FusionPixelMask zero_height(std::numeric_limits<size_t>::max(), 0);
  EXPECT_TRUE(UnpackSnapshot(zero_height.Snapshot(0, 0)).empty());
  EXPECT_THROW(zero_height.Get(0, 0), std::out_of_range);
}

TEST(StereoFusion, PackedSnapshotBoundariesMovesAndImmutability) {
  internal::FusionPixelMask mask(160, 1);
  for (const size_t index : {size_t{1}, size_t{63}, size_t{64}, size_t{129}})
    mask.Set(0, index, 1);
  auto snapshot = mask.Snapshot(1, 130);
  ASSERT_EQ(snapshot.size(), 129);
  const auto before = UnpackSnapshot(snapshot);
  std::vector<char> expected(129, 0);
  for (const size_t offset : {size_t{0}, size_t{62}, size_t{63}, size_t{128}})
    expected[offset] = 1;
  EXPECT_EQ(before, expected);
  std::atomic<bool> start{false};
  std::thread writer([&] {
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    for (size_t pass = 0; pass < 128; ++pass) {
      for (size_t index = 0; index < 160; ++index)
        mask.Set(0, index, static_cast<char>(pass & 1));
    }
  });
  start.store(true, std::memory_order_release);
  for (int pass = 0; pass < 128; ++pass)
    EXPECT_EQ(UnpackSnapshot(snapshot), before);
  writer.join();
  EXPECT_EQ(UnpackSnapshot(snapshot), before);
  auto moved = std::move(snapshot);
  EXPECT_TRUE(snapshot.empty());
  snapshot.ForEachUnset([](size_t) { ADD_FAILURE() << "moved-from snapshot"; });
  EXPECT_EQ(UnpackSnapshot(moved), before);
  internal::FusionPixelMask::PackedSnapshot assigned;
  assigned = std::move(moved);
  EXPECT_TRUE(moved.empty());
  EXPECT_EQ(UnpackSnapshot(assigned), before);
  const auto empty = mask.Snapshot(159, 159);
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0);
  empty.ForEachUnset([](size_t) { ADD_FAILURE() << "empty snapshot"; });
}

TEST(StereoFusion, PackedSnapshotVisitsUnsetInOrderAcrossBoundaries) {
  constexpr size_t kSize = 193;
  for (const int pattern : {0, 1, 2, 3}) {
    internal::FusionPixelMask mask(kSize, 1);
    std::vector<char> expected(kSize);
    for (size_t i = 0; i < kSize; ++i) {
      const bool set = pattern == 1 || (pattern == 2 && (i % 3 == 0)) ||
                       (pattern == 3 && i != 63 && i != 64 && i != 129);
      if (set) mask.Set(0, i, 1);
      expected[i] = static_cast<char>(set);
    }
    for (const auto& [first, last] : {std::pair<size_t, size_t>{0, 0},
                                      {1, 2},
                                      {1, 63},
                                      {1, 64},
                                      {1, 65},
                                      {63, 130},
                                      {64, 128},
                                      {65, 193},
                                      {0, kSize},
                                      {kSize, kSize}}) {
      SCOPED_TRACE(pattern);
      SCOPED_TRACE(first);
      SCOPED_TRACE(last);
      const auto snapshot = mask.Snapshot(first, last);
      std::vector<size_t> actual, reference;
      snapshot.ForEachUnset([&](size_t offset) { actual.push_back(offset); });
      for (size_t i = first; i < last; ++i)
        if (!expected[i]) reference.push_back(i - first);
      EXPECT_EQ(actual, reference);
    }
  }
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
  const auto snapshot = UnpackSnapshot(mask.Snapshot(0, kRows * kBits));
  EXPECT_TRUE(std::all_of(
      snapshot.begin(), snapshot.end(), [](char value) { return value == 1; }));
  mask.Set(17, 31, 0);
  for (size_t col = 0; col < kBits; ++col)
    EXPECT_EQ(mask.Get(17, col), col == 31 ? 0 : 1);
  mask.Set(17, 31, 1);
  EXPECT_EQ(UnpackSnapshot(mask.Snapshot(17 * kBits + 29, 17 * kBits + 34)),
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
    const auto snapshot = UnpackSnapshot(mask.Snapshot(0, kSide * kSide));
    EXPECT_TRUE(std::all_of(snapshot.begin(), snapshot.end(), [](char value) {
      return value == 0 || value == 1;
    }));
    const char value = mask.Get(kSide / 2, kSide / 2);
    EXPECT_TRUE(value == 0 || value == 1);
  }
  first.join();
  second.join();
  const auto snapshot = UnpackSnapshot(mask.Snapshot(0, kSide * kSide));
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

  static char PixelMaskValue(const StereoFusion& fusion,
                             size_t image,
                             size_t row,
                             size_t col) {
    return fusion.fused_pixel_masks_.at(image).Get(row, col);
  }
  static size_t DepthBorrowCount(const StereoFusion& fusion) {
    size_t reads = 0;
    for (const auto& worker : fusion.task_neighborhood_reads_) {
      reads += worker.pointer_reads;
    }
    return reads;
  }
  static bool HasFullRamInputs(const StereoFusion& fusion) {
    return !fusion.full_ram_inputs_.empty();
  }
  static bool HasCompactFullRamInputs(const StereoFusion& fusion) {
    return std::any_of(fusion.full_ram_inputs_.begin(),
                       fusion.full_ram_inputs_.end(),
                       [](const auto& input) {
                         return input.bitmap && input.depth.width > 0 &&
                                static_cast<size_t>(input.bitmap->Width()) ==
                                    input.depth.width &&
                                static_cast<size_t>(input.bitmap->Height()) ==
                                    input.depth.plane / input.depth.width;
                       });
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
  static auto Acquire(StereoFusion& fusion, int row, int col) {
    return fusion.AcquireWorkerNormalTile(0, 0, row, col);
  }
  static void EnableSoftRing(StereoFusion& fusion) {
    auto& ring = fusion.normal_tile_rings_.front();
    ring.soft_tiles.resize(8);
    ring.soft_index.reserve(9);
  }
  static void EnableIndexedRing(StereoFusion& fusion) {
    auto& ring = fusion.normal_tile_rings_.front();
    ring.indexed = true;
    ring.index.reserve(ring.tiles.size());
    ring.last_slot_by_image.assign(1, ring.tiles.size());
  }
  static void ClearRing(StereoFusion& fusion) { fusion.ClearNormalTileRings(); }
  static size_t RetainedOwners(const StereoFusion& fusion) {
    size_t count = 0;
    for (const auto& slot : fusion.normal_tile_rings_.front().tiles) {
      if (slot.retained) ++count;
    }
    return count;
  }
  static BoundedWorkspace& Cache(StereoFusion& fusion) {
    return *fusion.bounded_workspace_;
  }
  static bool HasPackedResident(const StereoFusion& fusion) {
    return std::any_of(
        fusion.neighborhood_data_.begin(),
        fusion.neighborhood_data_.end(),
        [](const auto* data) { return data && data->sparse_depth; });
  }
  static void PrepareNormalWindow(StereoFusion& fusion, size_t root = 0) {
    auto& cache = *fusion.bounded_workspace_;
    const size_t count = cache.GetModel().images.size();
    fusion.used_images_.assign(count, true);
    fusion.fused_images_.assign(count, false);
    for (size_t index = 0; index < root; ++index)
      fusion.fused_images_[index] = true;
    fusion.overlapping_images_.assign(count, {});
    for (size_t index = 0; index < count; ++index) {
      for (size_t other = 0; other < count; ++other) {
        if (index != other)
          fusion.overlapping_images_[index].push_back(static_cast<int>(other));
      }
    }
    fusion.neighborhood_data_.assign(count, nullptr);
    fusion.task_neighborhood_reads_.resize(1);
    const size_t tile = cache.GetNormalTileResidentBytes(0, 7);
    const size_t largest = std::max(
        {cache.GetNormalTileReservationBytes(0, 7),
         cache.GetLoadReservationBytes(0, BoundedWorkspace::Component::kDepth),
         cache.GetLoadReservationBytes(0,
                                       BoundedWorkspace::Component::kBitmap)});
    fusion.normal_tile_worker_headroom_ = largest + 2 * tile;
    ThreadPool pool(fusion.options_.bounded_cache_prefill_threads == -1
                        ? 1
                        : fusion.options_.bounded_cache_prefill_threads);
    fusion.PrepareBoundedNeighborhood(static_cast<int>(root), 1, pool);
  }
  static auto AcquireBitmap(StereoFusion& fusion, int view) {
    return fusion.AcquireNeighborhoodBitmap(0, view);
  }
  static size_t CarriedBitmaps(const StereoFusion& fusion) {
    return fusion.neighborhood_bitmap_carried_views_;
  }
  static auto AcquireView(StereoFusion& fusion, int view, int row, int col) {
    return fusion.AcquireWorkerNormalTile(0, view, row, col);
  }
  static size_t NormalWindowOwners(const StereoFusion& fusion) {
    return fusion.normal_prefill_leases_.size();
  }
  static size_t NormalWindowPointers(const StereoFusion& fusion) {
    return std::count_if(fusion.normal_prefill_data_.begin(),
                         fusion.normal_prefill_data_.end(),
                         [](const auto* data) { return data != nullptr; });
  }
  static void ClearNormalWindow(StereoFusion& fusion) {
    fusion.ClearNormalPrefillWindow();
    std::fill(fusion.neighborhood_data_.begin(),
              fusion.neighborhood_data_.end(),
              nullptr);
    fusion.neighborhood_leases_.clear();
    fusion.neighborhood_bitmap_pins_.reset();
  }
};

namespace {

Workspace::Options CreateNormalTileRingFixture(const bool varied = false,
                                               const size_t views = 1) {
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
  dataset.num_frames_per_rig = views;
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

  Mat<float> normal(17, 11, 3);
  normal.Fill(0.0f);
  for (int row = 0; row < 11; ++row) {
    for (int col = 0; col < 17; ++col) {
      if (varied) {
        normal.Set(row, col, 0, row * 17.0f + col + 0.25f);
        normal.Set(row, col, 1, -col - 0.5f);
      }
      normal.Set(row, col, 2, 1.0f);
    }
  }

  Bitmap bitmap(17, 11, true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 64, 128));
  for (size_t view = 0; view < views; ++view) {
    depth.Write(descriptor.GetDepthMapPath(static_cast<int>(view)));
    normal.Write(descriptor.GetNormalMapPath(static_cast<int>(view)));
    THROW_CHECK(bitmap.Write(descriptor.GetBitmapPath(static_cast<int>(view))));
  }
  return base;
}

TEST(StereoFusion, NormalWindowUsesExactWholeOwnerThenTiledFallback) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  auto base = CreateNormalTileRingFixture(true, 4);
  base.cache_size = 12000.0 / (1024.0 * 1024.0 * 1024.0);
  Workspace descriptor(base);
  NormalMap expected;
  expected.Read(descriptor.GetNormalMapPath(0));
  const uint32_t special[] = {0x80000000u, 0x7fc12345u, 0xbf800000u};
  for (size_t i = 0; i < 3; ++i)
    std::memcpy(expected.GetPtr() + i, &special[i], sizeof(float));
  expected.Write(descriptor.GetNormalMapPath(0));
  StereoFusionOptions options;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_diagnostics = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = 2;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 1;
  options.num_threads = 1;
  StereoFusion fusion(options, base.workspace_path, "COLMAP", "", "geometric");
  StereoFusionExactDepthTestPeer::PrepareRing(fusion, base, 2, false);
  StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowOwners(fusion), 2);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(fusion), 2);
  const auto before = fusion.GetBoundedCacheStats();
  ASSERT_TRUE(before);
  EXPECT_EQ(before->normal_tile_loads, 0);
  const auto whole = StereoFusionExactDepthTestPeer::Acquire(fusion, 10, 16);
  ASSERT_TRUE(whole);
  EXPECT_EQ(whole->normal_map.GetWidth(), 17);
  ASSERT_EQ(whole->normal_map.GetData().size(), expected.GetData().size());
  EXPECT_EQ(std::memcmp(whole->normal_map.GetData().data(),
                        expected.GetData().data(),
                        expected.GetData().size() * sizeof(float)),
            0);
  EXPECT_FLOAT_EQ(whole->GetNormal(10, 16)(0), 186.25f);
  EXPECT_FLOAT_EQ(whole->GetNormal(10, 16)(1), -16.5f);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->hits, before->hits);
  EXPECT_EQ(fusion.GetNormalTileRingStats().misses, 0);
  const auto diagnostic = fusion.GetBoundedCacheDiagnosticRow();
  ASSERT_TRUE(diagnostic);
  EXPECT_NE(diagnostic->find("normal_prefill_borrows=1 "), std::string::npos);
  // View2 is outside the hot window but its original depth remains admitted.
  // Exercise its exact tile while the whole-normal owner is still pinned.
  auto tile = StereoFusionExactDepthTestPeer::AcquireView(fusion, 2, 10, 16);
  ASSERT_TRUE(tile);
  EXPECT_LT(tile->normal_map.GetWidth(), 17);
  EXPECT_FLOAT_EQ(tile->GetNormal(10, 16)(0), 186.25f);
  const auto after = fusion.GetBoundedCacheStats();
  ASSERT_TRUE(after);
  EXPECT_EQ(after->normal_tile_loads, 1);
  EXPECT_EQ(after->reserved_bytes, 0);
  EXPECT_LE(after->peak_charged_bytes, after->capacity_bytes);
  StereoFusionExactDepthTestPeer::ClearRing(fusion);
  StereoFusionExactDepthTestPeer::ClearNormalWindow(fusion);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowOwners(fusion), 0);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(fusion), 0);
  // No caller reads the borrowed wrappers after the joined owners clear.
}

TEST(StereoFusion, NormalWindowCarriesPartialNeighborhoodBitmap) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  auto base = CreateNormalTileRingFixture(false, 4);
  base.cache_size = 7000.0 / (1024.0 * 1024.0 * 1024.0);
  StereoFusionOptions options;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = 2;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 1;
  options.num_threads = 1;
  StereoFusion fusion(options, base.workspace_path, "COLMAP", "", "geometric");
  StereoFusionExactDepthTestPeer::PrepareRing(fusion, base, 2, false);
  StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion, 0);
  const auto* bitmap = StereoFusionExactDepthTestPeer::AcquireBitmap(fusion, 1);
  ASSERT_NE(bitmap, nullptr);
  const auto color = bitmap->bitmap.GetPixel(0, 0);
  ASSERT_TRUE(color);
  EXPECT_EQ(color->r, 0);
  EXPECT_EQ(color->g, 64);
  EXPECT_EQ(color->b, 128);
  // No readers remain when the next root reuses the warmed owner.
  StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion, 1);
  EXPECT_GT(StereoFusionExactDepthTestPeer::CarriedBitmaps(fusion), 0);
  const auto* carried =
      StereoFusionExactDepthTestPeer::AcquireBitmap(fusion, 1);
  ASSERT_EQ(carried, bitmap);
  const auto carried_color = carried->bitmap.GetPixel(0, 0);
  ASSERT_TRUE(carried_color);
  EXPECT_EQ(carried_color->r, 0);
  EXPECT_EQ(carried_color->g, 64);
  EXPECT_EQ(carried_color->b, 128);
  const auto stats = fusion.GetBoundedCacheStats();
  ASSERT_TRUE(stats);
  EXPECT_EQ(stats->component_loads[static_cast<size_t>(
                BoundedWorkspace::Component::kImage)],
            0);
  EXPECT_EQ(stats->reserved_bytes, 0);
  EXPECT_LE(stats->peak_charged_bytes, stats->capacity_bytes);
  StereoFusionExactDepthTestPeer::ClearNormalWindow(fusion);
}

TEST(StereoFusion, NormalWindowStopAndLoadFailurePublishNothing) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  auto base = CreateNormalTileRingFixture(true, 6);
  base.cache_size = 20000.0 / (1024.0 * 1024.0 * 1024.0);
  StereoFusionOptions options;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = 2;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 1;
  options.num_threads = 1;
  options.bounded_cache_prefill_threads = 4;
  StereoFusion fusion(options, base.workspace_path, "COLMAP", "", "geometric");
  StereoFusionExactDepthTestPeer::PrepareRing(fusion, base, 2, false);
  bool stopped = false;
  fusion.SetCheckIfStoppedFunc([&] {
    const auto stats = fusion.GetBoundedCacheStats();
    stopped =
        stopped || (stats && stats->component_loads[static_cast<size_t>(
                                 BoundedWorkspace::Component::kDepth)] > 0);
    return stopped;
  });
  StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowOwners(fusion), 0);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(fusion), 0);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
  fusion.SetCheckIfStoppedFunc([] { return false; });
  Workspace paths(base);
  const auto normal_path = paths.GetNormalMapPath(0);
  std::ifstream original(normal_path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(original)),
                          std::istreambuf_iterator<char>());
  original.close();
  const auto header_end =
      bytes.find('&', bytes.find('&', bytes.find('&') + 1) + 1);
  ASSERT_NE(header_end, std::string::npos);
  // A fresh cache ensures planning sees the valid original header/body first.
  // This main-thread stop callback runs after the depth tasks join, immediately
  // before the selected normal load. Inject failure there without a race.
  StereoFusionExactDepthTestPeer::ClearNormalWindow(fusion);
  StereoFusionExactDepthTestPeer::PrepareRing(fusion, base, 2, false);
  bool truncated = false;
  fusion.SetCheckIfStoppedFunc([&] {
    const auto stats = fusion.GetBoundedCacheStats();
    if (!truncated && stats &&
        stats->component_loads[static_cast<size_t>(
            BoundedWorkspace::Component::kDepth)] > 0) {
      std::ofstream broken(normal_path, std::ios::binary | std::ios::trunc);
      broken.write(bytes.data(), header_end + 1);
      truncated = true;
    }
    return false;
  });
  EXPECT_THROW(StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion),
               std::exception);
  EXPECT_TRUE(truncated);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowOwners(fusion), 0);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(fusion), 0);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
  fusion.SetCheckIfStoppedFunc([] { return false; });
  {
    std::ofstream restored(normal_path, std::ios::binary | std::ios::trunc);
    restored.write(bytes.data(), bytes.size());
  }
  StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowOwners(fusion), 4);
  StereoFusionExactDepthTestPeer::ClearNormalWindow(fusion);
}
TEST(StereoFusion, CompleteTileNeighborhoodAliasesWholeImageOwners) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  auto base = CreateNormalTileRingFixture(true, 2);
  base.cache_size = 12000.0 / (1024.0 * 1024.0 * 1024.0);
  Workspace paths(base);
  NormalMap expected;
  expected.Read(paths.GetNormalMapPath(0));
  const uint32_t special[] = {0x80000000u, 0x7fc12345u, 0xbf800000u};
  for (size_t i = 0; i < 3; ++i)
    std::memcpy(expected.GetPtr() + i, &special[i], sizeof(float));
  expected.Write(paths.GetNormalMapPath(0));
  StereoFusionOptions options;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = 2;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 1;
  options.num_threads = 1;
  options.bounded_cache_prefill_threads = 4;
  StereoFusion fusion(options, base.workspace_path, "COLMAP", "", "geometric");
  StereoFusionExactDepthTestPeer::PrepareRing(fusion, base, 2, false);
  StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion);
  const auto stats = fusion.GetBoundedCacheStats();
  ASSERT_TRUE(stats);
  EXPECT_EQ(stats->component_loads[static_cast<size_t>(
                BoundedWorkspace::Component::kImage)],
            2);
  EXPECT_EQ(stats->component_loads[static_cast<size_t>(
                BoundedWorkspace::Component::kNormal)],
            0);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowOwners(fusion), 0);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(fusion), 2);
  const auto whole = StereoFusionExactDepthTestPeer::Acquire(fusion, 10, 16);
  ASSERT_TRUE(whole);
  ASSERT_EQ(whole->normal_map.GetData().size(), expected.GetData().size());
  EXPECT_EQ(std::memcmp(whole->normal_map.GetData().data(),
                        expected.GetData().data(),
                        expected.GetData().size() * sizeof(float)),
            0);
  const auto second =
      StereoFusionExactDepthTestPeer::AcquireView(fusion, 1, 10, 16);
  ASSERT_TRUE(second);
  EXPECT_EQ(second->normal_map.GetWidth(), 17);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->normal_tile_loads, 0);
  EXPECT_LE(stats->peak_charged_bytes, stats->capacity_bytes);
  StereoFusionExactDepthTestPeer::ClearNormalWindow(fusion);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(fusion), 0);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
}

TEST(StereoFusion, CompleteTileNeighborhoodStopFailureAndRetry) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  auto base = CreateNormalTileRingFixture(true, 2);
  base.cache_size = 12000.0 / (1024.0 * 1024.0 * 1024.0);
  StereoFusionOptions options;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = 2;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 1;
  options.num_threads = 1;
  options.bounded_cache_prefill_threads = 4;
  StereoFusion fusion(options, base.workspace_path, "COLMAP", "", "geometric");
  StereoFusionExactDepthTestPeer::PrepareRing(fusion, base, 2, false);
  fusion.SetCheckIfStoppedFunc([&] {
    const auto stats = fusion.GetBoundedCacheStats();
    return stats && stats->component_loads[static_cast<size_t>(
                        BoundedWorkspace::Component::kImage)] > 0;
  });
  StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(fusion), 0);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
  fusion.SetCheckIfStoppedFunc([] { return false; });
  StereoFusionExactDepthTestPeer::PrepareRing(fusion, base, 2, false);
  Workspace paths(base);
  std::atomic<int> attempts{0};
  auto failed = ScopedBitmapReader::Create(
      {paths.GetBitmapPath(0)},
      [&attempts](const std::filesystem::path&, bool, bool, Bitmap*) {
        ++attempts;
        return false;
      },
      {{17, 11}});
  EXPECT_THROW(StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion),
               std::exception);
  EXPECT_GT(attempts.load(), 0);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(fusion), 0);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
  failed->Close();
  StereoFusionExactDepthTestPeer::PrepareNormalWindow(fusion);
  EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(fusion), 2);
  StereoFusionExactDepthTestPeer::ClearNormalWindow(fusion);
}

TEST(StereoFusion, NormalWindowPinsPreserveDepthAndBudgetLoaderExtra) {
  using internal::NormalWindowFusionPins;
  const std::vector<size_t> depth(4, 10), bitmap(4, 8), normal(4, 16);
  const auto pins =
      NormalWindowFusionPins(depth, bitmap, normal, {20, 20, 20, 20}, 100);
  EXPECT_EQ(pins.components.depth,
            internal::DepthPriorityFusionPins(depth, bitmap, 100).depth);
  EXPECT_EQ(pins.normal, (std::vector<size_t>{0}));
  EXPECT_EQ(pins.components.bitmap, (std::vector<size_t>{0, 1, 2, 3}));
  EXPECT_EQ(pins.largest_normal_extra, 4);
  size_t charge = pins.largest_normal_extra;
  for (auto i : pins.components.depth) charge += depth[i];
  for (auto i : pins.components.bitmap) charge += bitmap[i];
  for (auto i : pins.normal) charge += normal[i];
  EXPECT_LE(charge, 100);
  // A large loader extra cannot displace an admitted depth or root RGB.
  const auto scratch =
      NormalWindowFusionPins(depth, bitmap, normal, {80, 80, 80, 80}, 100);
  EXPECT_TRUE(scratch.normal.empty());
  EXPECT_EQ(scratch.components.depth, pins.components.depth);
  EXPECT_EQ(scratch.components.bitmap, pins.components.bitmap);
}

TEST(StereoFusion, NormalWindowPinsUseAvailableBudgetAndTinyFallback) {
  using internal::NormalWindowFusionPins;
  EXPECT_TRUE(NormalWindowFusionPins({}, {}, {}, {}, 0).normal.empty());
  const auto tiny = NormalWindowFusionPins({10}, {8}, {16}, {16}, 17);
  EXPECT_TRUE(tiny.components.depth.empty());
  EXPECT_TRUE(tiny.normal.empty());
  const auto cap = NormalWindowFusionPins(std::vector<size_t>(12, 1),
                                          std::vector<size_t>(12, 1),
                                          std::vector<size_t>(12, 1),
                                          std::vector<size_t>(12, 1),
                                          100);
  EXPECT_EQ(cap.normal,
            (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
  EXPECT_EQ(cap.components.depth.size(), 12);
  // Normals may replace only nonroot bitmap pins, never the baseline depths.
  const auto mixed = NormalWindowFusionPins(
      {10, 10, 10}, {20, 20, 20}, {20, 20, 20}, {20, 20, 20}, 100);
  EXPECT_EQ(mixed.normal, (std::vector<size_t>{0}));
  EXPECT_EQ(mixed.components.depth, (std::vector<size_t>{0, 1, 2}));
  EXPECT_EQ(mixed.components.bitmap, (std::vector<size_t>{0, 1}));
  EXPECT_ANY_THROW(NormalWindowFusionPins({1}, {1}, {2}, {1}, 16));
  const size_t max = std::numeric_limits<size_t>::max();
  EXPECT_TRUE(NormalWindowFusionPins({max}, {1}, {1}, {1}, max).normal.empty());
}

TEST(StereoFusion, NormalWindowPinsShareScratchAndSkipOversizedLoads) {
  using internal::NormalWindowFusionPins;
  const auto exact =
      NormalWindowFusionPins({1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {3, 7, 3}, 15);
  EXPECT_EQ(exact.normal, (std::vector<size_t>{0, 1}));
  EXPECT_EQ(exact.components.bitmap, (std::vector<size_t>{0, 1, 2}));
  EXPECT_EQ(exact.largest_normal_extra, 5);
  const auto smaller =
      NormalWindowFusionPins({1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {3, 7, 3}, 14);
  EXPECT_EQ(smaller.normal, (std::vector<size_t>{0, 2}));
  EXPECT_EQ(smaller.components.bitmap, exact.components.bitmap);
  EXPECT_EQ(smaller.largest_normal_extra, 1);
}

TEST(StereoFusion, NormalTileRingReusesFp32CroppedTiles) {
  if (!ImmutableMapRegion::Supported())
    GTEST_SKIP() << "Immutable FP32 map owners are unsupported on this host";
  const auto base = CreateNormalTileRingFixture();
  const auto& root = base.workspace_path;
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
  for (int slots : {2, 4, 8}) {
    for (bool hints : {false, true}) {
      if (slots <= 4 && !hints) continue;  // No hint array in linear rings.
      SCOPED_TRACE(::testing::Message()
                   << "slots=" << slots << " hints=" << hints);
      auto workspace_options = base;
      StereoFusionOptions options;
      options.num_threads = 1;
      options.cache_size = base.cache_size;
      options.bounded_cache = true;
      options.bounded_cache_components = true;
      options.bounded_cache_normal_tile_size = 8;
      options.bounded_cache_normal_tile_slots = slots;
      options.bounded_cache_normal_tile_hints = hints;
      options.bounded_cache_normal_mapped_bytes = 1 << 20;
      options.bounded_cache_normal_mapped_files = 1;
      ASSERT_TRUE(options.Check());
      StereoFusion fusion(options, root, "COLMAP", "", "geometric");
      StereoFusionExactDepthTestPeer::PrepareRing(
          fusion, workspace_options, slots, hints);
      for (const auto& [row, col] : samples) {
        auto tile = StereoFusionExactDepthTestPeer::Acquire(fusion, row, col);
        ASSERT_TRUE(tile);
        const size_t x = static_cast<size_t>(col) / 8 * 8;
        const size_t y = static_cast<size_t>(row) / 8 * 8;
        EXPECT_EQ(tile->normal_origin_x, x);
        EXPECT_EQ(tile->normal_origin_y, y);
        const size_t width = std::min(size_t{8}, 17 - x);
        const size_t height = std::min(size_t{8}, 11 - y);
        EXPECT_EQ(tile->normal_map.GetWidth(), width);
        EXPECT_EQ(tile->normal_map.GetHeight(), height);
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
      std::cout << "NORMAL_RING_PROOF slots=" << slots << " hints=" << hints
                << " requests=" << samples.size() << " hits=" << ring.hits
                << " misses=" << ring.misses
                << " loads=" << cache->normal_tile_loads
                << " cache_hits=" << cache->normal_tile_hits
                << " indexed=" << ring.indexed_lookups
                << " hint_checks=" << ring.hint_checks << '\n';
    }
  }
}

TEST(StereoFusion, SoftNormalTilesAvoidSharedHitsAndRejectStaleHandles) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  auto base = CreateNormalTileRingFixture(true);
  // Two hard tiles plus one largest transient component fit admission.
  // A/B/C initially fit, but adding a depth component must evict soft A.
  base.cache_size = 2304.0 / (1ULL << 30);
  StereoFusionOptions options;
  options.num_threads = 1;
  options.cache_size = base.cache_size;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_normal_tile_size = 8;
  options.bounded_cache_normal_tile_slots = 2;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 1;
  StereoFusion fusion(options, base.workspace_path, "COLMAP", "", "geometric");
  StereoFusionExactDepthTestPeer::PrepareRing(fusion, base, 2, false);
  StereoFusionExactDepthTestPeer::EnableSoftRing(fusion);
  const auto read = [&](const int row, const int col) -> Eigen::Vector3f {
    auto tile = StereoFusionExactDepthTestPeer::Acquire(fusion, row, col);
    EXPECT_TRUE(tile);
    return tile ? tile->GetNormal(row, col) : Eigen::Vector3f::Zero().eval();
  };
  const auto expected = [](const int row, const int col) {
    return Eigen::Vector3f(row * 17.0f + col + 0.25f, -col - 0.5f, 1.0f);
  };
  for (const int col : {0, 8, 16}) EXPECT_EQ(read(0, col), expected(0, col));
  const auto before = fusion.GetBoundedCacheStats();
  ASSERT_TRUE(before);
  for (int repeat = 0; repeat < 2; ++repeat)
    EXPECT_EQ(read(0, 0), expected(0, 0));
  const auto after = fusion.GetBoundedCacheStats();
  ASSERT_TRUE(after);
  EXPECT_EQ(after->normal_tile_loads, 3);
  EXPECT_EQ(after->normal_tile_hits, before->normal_tile_hits);
  EXPECT_EQ(after->resident_bytes, before->resident_bytes);
  EXPECT_EQ(fusion.GetNormalTileRingStats().soft_hits, 1);
  EXPECT_EQ(fusion.GetNormalTileRingStats().hits, 1);
  {
    auto depth = StereoFusionExactDepthTestPeer::Cache(fusion).AcquireComponent(
        0, BoundedWorkspace::Component::kDepth);
    EXPECT_EQ(depth->depth_map.Get(0, 0), 5.0f);
  }
  // A is now retained; demoted B was evicted for the depth load.
  // Its old metadata must reject the retired generation before reloading.
  EXPECT_EQ(read(0, 8), expected(0, 8));
  EXPECT_EQ(fusion.GetNormalTileRingStats().soft_expired, 1);
  const auto final = fusion.GetBoundedCacheStats();
  ASSERT_TRUE(final);
  EXPECT_EQ(final->normal_tile_loads, 4);
  EXPECT_EQ(final->reserved_bytes, 0);
  EXPECT_LE(final->peak_charged_bytes, final->capacity_bytes);
}

TEST(StereoFusion, SoftPromotionCyclesOwnersAndPreservesHardHits) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  for (bool indexed : {false, true}) {
    auto base = CreateNormalTileRingFixture(true);
    base.cache_size = 2304.0 / (1ULL << 30);
    StereoFusionOptions options;
    options.num_threads = 1;
    options.cache_size = base.cache_size;
    options.bounded_cache = options.bounded_cache_components = true;
    options.bounded_cache_normal_tile_size = 8;
    options.bounded_cache_normal_tile_slots = 2;
    options.bounded_cache_normal_mapped_bytes = 1 << 20;
    options.bounded_cache_normal_mapped_files = 1;
    StereoFusion fusion(
        options, base.workspace_path, "COLMAP", "", "geometric");
    StereoFusionExactDepthTestPeer::PrepareRing(fusion, base, 2, false);
    StereoFusionExactDepthTestPeer::EnableSoftRing(fusion);
    if (indexed) StereoFusionExactDepthTestPeer::EnableIndexedRing(fusion);
    const auto read = [&](int col) {
      auto tile = StereoFusionExactDepthTestPeer::Acquire(fusion, 0, col);
      EXPECT_TRUE(tile);
      if (tile) {
        EXPECT_EQ(tile->GetNormal(0, col),
                  Eigen::Vector3f(col + 0.25f, -col - 0.5f, 1.0f));
      }
    };
    for (int col : {0, 8, 16}) read(col);
    for (int repeat = 0; repeat < 4; ++repeat) {
      for (int col : {0, 8, 16}) {
        read(col);  // Borrow the demoted resident generation into the victim
                    // slot.
        const auto before = fusion.GetNormalTileRingStats();
        read(col);  // The promoted owner must now take the cheap hard-hit path.
        const auto after = fusion.GetNormalTileRingStats();
        EXPECT_EQ(after.hits, before.hits + 1);
        EXPECT_EQ(after.soft_hits, before.soft_hits);
        EXPECT_LE(StereoFusionExactDepthTestPeer::RetainedOwners(fusion), 2);
      }
    }
    EXPECT_EQ(StereoFusionExactDepthTestPeer::RetainedOwners(fusion), 2);
    const auto stats = fusion.GetBoundedCacheStats();
    ASSERT_TRUE(stats);
    EXPECT_EQ(stats->normal_tile_loads, 3);
    EXPECT_EQ(stats->normal_tile_hits, 0);
    EXPECT_EQ(stats->reserved_bytes, 0);
    EXPECT_LE(stats->peak_charged_bytes, stats->capacity_bytes);
    StereoFusionExactDepthTestPeer::ClearRing(fusion);
    EXPECT_EQ(StereoFusionExactDepthTestPeer::RetainedOwners(fusion), 0);
    // Cleanup releases both kinds of owners before another waiting load.
    auto depth = StereoFusionExactDepthTestPeer::Cache(fusion).AcquireComponent(
        0, BoundedWorkspace::Component::kDepth);
    EXPECT_EQ(depth->depth_map.Get(0, 0), 5.0f);
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

TEST(StereoFusionOptions, ExperimentalDepthCompactionPolicy) {
  StereoFusionOptions options;
  EXPECT_FALSE(options.bounded_cache_compact_depth);
  options.bounded_cache_compact_depth = true;
  EXPECT_FALSE(options.Check());
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 2;
  EXPECT_TRUE(options.Check());
  options.bounded_cache_neighborhood = false;
  EXPECT_FALSE(options.Check());
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_tile_size = 0;
  EXPECT_FALSE(options.Check());
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_exact_depth = true;
  EXPECT_FALSE(options.Check());
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
  // Compare resident and cached sampling, including every pixel attribute
  // and visibility entry.
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

// Analytic rays and planes provide an oracle independent of Fusion's helpers.
class StereoFusionPlaneOracle : public testing::Test {
 protected:
  struct ExpectedPoint {
    Eigen::Vector3d xyz;
    Eigen::Vector3d normal;
    std::array<int, 3> rgb;
    std::vector<int> visibility;
  };

  void SetUp() override {
    root_ = CreateTestDir();
    for (const auto* dir :
         {"sparse", "images", "stereo/depth_maps", "stereo/normal_maps"}) {
      CreateDirIfNotExists(root_ / dir, true);
    }
    constexpr int kWidth = 41, kHeight = 29;
    constexpr int kBitmapWidth = 83, kBitmapHeight = 61;
    constexpr double kFx = 90, kFy = 75, kCx = 43.2, kCy = 28.4;
    const std::array<image_t, 4> image_ids{31, 7, 99, 42};
    const std::array<double, 4> yaw{-0.03, -0.01, 0.015, 0.04};
    const std::array<double, 4> roll{0.015, -0.01, 0.02, -0.015};
    std::array<Eigen::Matrix3d, 4> rotations;
    std::array<Eigen::Vector3d, 4> centers;
    std::array<std::string, 4> names;
    std::array<Mat<float>, 4> depths, normals;
    for (size_t view = 0; view < names.size(); ++view) {
      rotations[view] =
          (Eigen::AngleAxisd(roll[view], Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(yaw[view], Eigen::Vector3d::UnitY()))
              .toRotationMatrix();
      centers[view] =
          Eigen::Vector3d(-0.15 + 0.1 * view, 0.025 * (view % 2), 0.015 * view);
      names[view] = "view_" + std::to_string(image_ids[view]) + ".png";
      depths[view] = Mat<float>(kWidth, kHeight, 1);
      normals[view] = Mat<float>(kWidth, kHeight, 3);
      depths[view].Fill(0);
      normals[view].Fill(0);
    }
    const auto project = [&](int view, const Eigen::Vector3d& xyz) {
      const Eigen::Vector3d q = rotations[view] * (xyz - centers[view]);
      return Eigen::Vector2d(
          (kFx * q.x() / q.z() + kCx) * kWidth / kBitmapWidth - 0.5,
          (kFy * q.y() / q.z() + kCy) * kHeight / kBitmapHeight - 0.5);
    };
    const auto intersect = [&](int view,
                               int col,
                               int row,
                               const Eigen::Vector3d& normal,
                               double offset) {
      const Eigen::Vector3d ray =
          rotations[view].transpose() *
          Eigen::Vector3d(((col + 0.5) * kBitmapWidth / kWidth - kCx) / kFx,
                          ((row + 0.5) * kBitmapHeight / kHeight - kCy) / kFy,
                          1);
      return Eigen::Vector3d(
          centers[view] +
          ((offset - normal.dot(centers[view])) / normal.dot(ray)) * ray);
    };
    const auto rgb = [](int view, int col, int row) {
      return std::array<int, 3>{(13 * view + 3 * col + 5 * row) % 256,
                                (47 * view + 11 * col + 7 * row) % 256,
                                (91 * view + col + 17 * row) % 256};
    };
    Reconstruction reconstruction;
    colmap::Camera camera;
    camera.camera_id = 5;
    camera.model_id = CameraModelId::kPinhole;
    camera.width = kBitmapWidth;
    camera.height = kBitmapHeight;
    camera.params = {kFx, kFy, kCx, kCy};
    reconstruction.AddCameraWithTrivialRig(camera);
    const std::array<Eigen::Vector3d, 3> sparse_points{
        Eigen::Vector3d(-1, -0.4, 6),
        Eigen::Vector3d(0, 0.5, 6.5),
        Eigen::Vector3d(1, 0, 7)};
    for (size_t view = 0; view < names.size(); ++view) {
      colmap::Image image;
      image.SetImageId(image_ids[view]);
      image.SetName(names[view]);
      image.SetCameraId(camera.camera_id);
      std::vector<Eigen::Vector2d> observations;
      for (const auto& xyz : sparse_points) {
        const Eigen::Vector2d pixel = project(view, xyz);
        observations.emplace_back((pixel.x() + 0.5) * kBitmapWidth / kWidth,
                                  (pixel.y() + 0.5) * kBitmapHeight / kHeight);
      }
      image.SetPoints2D(observations);
      reconstruction.AddImageWithTrivialFrame(
          image,
          Rigid3d(Eigen::Quaterniond(rotations[view]),
                  -rotations[view] * centers[view]));
    }
    // Only adjacent physical views share sparse tracks.
    for (size_t edge = 0; edge < sparse_points.size(); ++edge) {
      Track track;
      track.AddElement(image_ids[edge], edge);
      track.AddElement(image_ids[edge + 1], edge);
      reconstruction.AddPoint3D(sparse_points[edge], std::move(track));
    }
    reconstruction.Write(root_ / "sparse");
    Model model;
    model.Read(root_, "COLMAP");
    std::array<int, 4> image_indices;
    for (size_t view = 0; view < names.size(); ++view) {
      image_indices[view] = model.GetImageIdx(names[view]);
    }
    const auto overlap = model.GetMaxOverlappingImages(3, 0);
    for (int view = 0; view < 4; ++view) {
      std::vector<int> expected;
      if (view > 0) expected.push_back(image_indices[view - 1]);
      if (view < 3) expected.push_back(image_indices[view + 1]);
      auto actual = overlap[image_indices[view]];
      std::sort(expected.begin(), expected.end());
      std::sort(actual.begin(), actual.end());
      ASSERT_EQ(actual, expected);
    }
    const std::array<std::vector<int>, 4> support{
        std::vector<int>{0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3}};
    const std::array<std::array<std::pair<int, int>, 3>, 4> anchors{{
        {{{4, 5}, {11, 14}, {11, 23}}},
        {{{12, 5}, {12, 13}, {12, 23}}},
        {{{17, 8}, {20, 14}, {29, 21}}},
        {{{35, 5}, {30, 14}, {28, 23}}},
    }};
    for (size_t group = 0; group < support.size(); ++group) {
      const Eigen::Vector3d plane = group % 2 == 0
                                        ? Eigen::Vector3d(-0.12, 0.18, 1)
                                        : Eigen::Vector3d(0.16, -0.09, 1);
      const double offset = group % 2 == 0 ? 5 : 7.5;
      for (const auto& [anchor_col, anchor_row] : anchors[group]) {
        const Eigen::Vector3d truth =
            intersect(0, anchor_col, anchor_row, plane, offset);
        ExpectedPoint expected;
        expected.normal = plane.normalized();
        std::array<std::vector<double>, 3> coordinates;
        std::array<std::vector<int>, 3> colors;
        std::vector<Eigen::Vector3d> samples;
        std::vector<Eigen::Vector2i> pixels;
        for (const int view : support[group]) {
          const Eigen::Vector2d projected = project(view, truth);
          const int col = std::lround(projected.x());
          const int row = std::lround(projected.y());
          ASSERT_GT(col, 0);
          ASSERT_LT(col, kWidth - 1);
          ASSERT_GT(row, 0);
          ASSERT_LT(row, kHeight - 1);
          ASSERT_EQ(depths[view].Get(row, col), 0);
          const Eigen::Vector3d xyz = intersect(view, col, row, plane, offset);
          const Eigen::Vector3d q = rotations[view] * (xyz - centers[view]);
          const Eigen::Vector3d normal = rotations[view] * expected.normal;
          depths[view].Set(row, col, q.z());
          for (int axis = 0; axis < 3; ++axis) {
            normals[view].Set(row, col, axis, normal[axis]);
            coordinates[axis].push_back(xyz[axis]);
          }
          // Nearest bitmap centers at ((pixel+.5)/scale-.5). Integer arithmetic
          // fixes expected source indices independently of either RGB reader.
          const auto color = rgb(view,
                                 (2 * col + 1) * kBitmapWidth / (2 * kWidth),
                                 (2 * row + 1) * kBitmapHeight / (2 * kHeight));
          for (int axis = 0; axis < 3; ++axis)
            colors[axis].push_back(color[axis]);
          expected.visibility.push_back(image_indices[view]);
          samples.push_back(xyz);
          pixels.emplace_back(col, row);
        }
        // Establish unambiguous correspondence and depth acceptance before
        // running Fusion. Every permitted traversal order must reach this set.
        for (size_t index = 0; index < support[group].size(); ++index) {
          const int view = support[group][index];
          const double depth =
              (rotations[view] * (samples[index] - centers[view])).z();
          for (const auto& xyz : samples) {
            ASSERT_LT((project(view, xyz) - pixels[index].cast<double>())
                          .cwiseAbs()
                          .maxCoeff(),
                      0.45);
            ASSERT_LT(std::abs((rotations[view] * (xyz - centers[view])).z() -
                               depth) /
                          depth,
                      0.005);
          }
        }
        for (int axis = 0; axis < 3; ++axis) {
          auto& xyz = coordinates[axis];
          auto& color = colors[axis];
          std::sort(xyz.begin(), xyz.end());
          std::sort(color.begin(), color.end());
          const size_t lower = (xyz.size() - 1) / 2, upper = xyz.size() / 2;
          expected.xyz[axis] = (xyz[lower] + xyz[upper]) / 2;
          expected.rgb[axis] = (color[lower] + color[upper] + 1) / 2;
        }
        ASSERT_LT((expected.xyz - truth).norm(), 0.12);
        std::sort(expected.visibility.begin(), expected.visibility.end());
        expected_.push_back(std::move(expected));
      }
    }
    // An exact plane sample without a second supporting view must not become
    // output. Check its intended support before writing the production inputs.
    const Eigen::Vector3d plane(-0.12, 0.18, 1);
    const Eigen::Vector3d orphan = intersect(0, 3, 26, plane, 5);
    depths[0].Set(26, 3, (rotations[0] * (orphan - centers[0])).z());
    const Eigen::Vector3d orphan_normal = rotations[0] * plane.normalized();
    for (int axis = 0; axis < 3; ++axis) {
      normals[0].Set(26, 3, axis, orphan_normal[axis]);
    }
    const Eigen::Vector2d orphan_pixel = project(1, orphan);
    ASSERT_EQ(depths[1].Get(std::lround(orphan_pixel.y()),
                            std::lround(orphan_pixel.x())),
              0);
    std::ofstream config(root_ / "stereo/fusion.cfg");
    for (int view = 3; view >= 0; --view) {
      config << names[view] << '\n';
      depths[view].Write(root_ / "stereo/depth_maps" /
                         (names[view] + ".geometric.bin"));
      normals[view].Write(root_ / "stereo/normal_maps" /
                          (names[view] + ".geometric.bin"));
      Bitmap bitmap(kBitmapWidth, kBitmapHeight, true);
      for (int row = 0; row < kBitmapHeight; ++row) {
        for (int col = 0; col < kBitmapWidth; ++col) {
          const auto color = rgb(view, col, row);
          ASSERT_TRUE(bitmap.SetPixel(
              col, row, BitmapColor<uint8_t>(color[0], color[1], color[2])));
        }
      }
      ASSERT_TRUE(bitmap.Write(root_ / "images" / names[view]));
    }
  }
  StereoFusionOptions Options() const {
    StereoFusionOptions options;
    options.num_threads = 1;
    options.min_num_pixels = 2;
    options.max_num_pixels = 4;
    options.max_traversal_depth = 4;
    options.check_num_images = 3;
    options.max_reproj_error = 1;
    options.max_depth_error = 0.005;
    options.max_normal_error = 1;
    return options;
  }

  void ExpectPoints(const StereoFusion& fusion, bool first_root_only = false) {
    ExpectPoints(fusion.GetFusedPoints(),
                 fusion.GetFusedPointsVisibility(),
                 first_root_only);
  }

  void ExpectPoints(const std::vector<PlyPoint>& points,
                    const std::vector<std::vector<int>>& visibility,
                    bool first_root_only = false) {
    std::vector<const ExpectedPoint*> expected;
    for (const auto& point : expected_) {
      if (!first_root_only ||
          std::find(point.visibility.begin(), point.visibility.end(), 0) !=
              point.visibility.end())
        expected.push_back(&point);
    }
    if (first_root_only) {
      ASSERT_GT(expected.size(), 0);
      ASSERT_LT(expected.size(), expected_.size());
    }
    ASSERT_EQ(points.size(), expected.size());
    ASSERT_EQ(visibility.size(), expected.size());
    std::vector<int> counts(expected.size(), 0);
    for (size_t index = 0; index < points.size(); ++index) {
      const auto& point = points[index];
      const Eigen::Vector3d xyz(point.x, point.y, point.z);
      const auto closest =
          std::min_element(expected.begin(),
                           expected.end(),
                           [&](const auto* lhs, const auto* rhs) {
                             return (xyz - lhs->xyz).squaredNorm() <
                                    (xyz - rhs->xyz).squaredNorm();
                           });
      ASSERT_NE(closest, expected.end());
      ++counts[closest - expected.begin()];
      // At depths 5--8, 5e-5 permits FP32 pose/inverse rounding while remaining
      // over 1000 times smaller than a pixel. Normals have unit magnitude.
      for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(xyz[axis], (*closest)->xyz[axis], 5e-5);
      }
      EXPECT_NEAR(point.nx, (*closest)->normal.x(), 2e-6);
      EXPECT_NEAR(point.ny, (*closest)->normal.y(), 2e-6);
      EXPECT_NEAR(point.nz, (*closest)->normal.z(), 2e-6);
      EXPECT_EQ(point.r, (*closest)->rgb[0]);
      EXPECT_EQ(point.g, (*closest)->rgb[1]);
      EXPECT_EQ(point.b, (*closest)->rgb[2]);
      auto actual_visibility = visibility[index];
      std::sort(actual_visibility.begin(), actual_visibility.end());
      EXPECT_EQ(actual_visibility, (*closest)->visibility);
    }
    for (const int count : counts) EXPECT_EQ(count, 1);
  }

  std::filesystem::path root_;
  std::vector<ExpectedPoint> expected_;
};

TEST_F(StereoFusionPlaneOracle, WorkspaceModes) {
  std::vector<std::pair<std::string, StereoFusionOptions>> variants;
  auto options = Options();
  variants.emplace_back("eager", options);
  options.num_threads = 3;
  options.full_ram_load_threads = 2;
  variants.emplace_back("eager_parallel", options);
  options = Options();
  options.use_cache = true;
  options.cache_size = 40000.0 / (1024.0 * 1024.0 * 1024.0);
  variants.emplace_back("legacy_cache", options);
  options.use_cache = false;
  options.bounded_cache = true;
  options.cache_size = 60000.0 / (1024.0 * 1024.0 * 1024.0);
  variants.emplace_back("bounded_whole", options);
  options.bounded_cache_components = true;
  variants.emplace_back("bounded_components", options);
  options.bounded_cache_compact_rgb = true;
  options.bounded_cache_neighborhood = true;
  options.cache_size = 200000.0 / (1024.0 * 1024.0 * 1024.0);
  variants.emplace_back("complete_neighborhood_compact_rgb", options);
  if (ImmutableMapRegion::Supported()) {
    options.cache_size = 50000.0 / (1024.0 * 1024.0 * 1024.0);
    options.bounded_cache_neighborhood = false;
    options.bounded_cache_normal_mapped_bytes = 1 << 20;
    options.bounded_cache_normal_mapped_files = 2;
    variants.emplace_back("direct_mapped_compact_rgb", options);
    options.num_threads = 2;
    options.bounded_cache_neighborhood = true;
    options.bounded_cache_prefill_threads = 3;
    options.bounded_cache_normal_tile_size = 7;
    options.bounded_cache_compact_depth = true;
    variants.emplace_back("tiled_neighborhood_compact_depth_rgb", options);
  }
  for (const auto& [name, variant] : variants) {
    SCOPED_TRACE(name);
    ASSERT_TRUE(variant.Check());
    StereoFusion fusion(variant, root_, "COLMAP", "", "geometric");
    bool compact_borrowed = false, sparse_pinned = false;
    fusion.SetCheckIfStoppedFunc([&] {
      compact_borrowed |=
          StereoFusionExactDepthTestPeer::HasCompactFullRamInputs(fusion);
      sparse_pinned |=
          StereoFusionExactDepthTestPeer::HasPackedResident(fusion);
      return false;
    });
    fusion.Run();
    ASSERT_NO_FATAL_FAILURE(ExpectPoints(fusion));
    if (name == "eager") {
      const auto path = root_ / "fused.ply";
      WriteFusedPointCloud(
          path, fusion.GetFusedPoints(), fusion.GetFusedPointsVisibility());
      ASSERT_NO_FATAL_FAILURE(
          ExpectPoints(ReadPly(path),
                       ReadPointsVisibility(AddFileExtension(path, ".vis"),
                                            expected_.size())));
    }
    if (!variant.use_cache && !variant.bounded_cache) {
      EXPECT_EQ(StereoFusionExactDepthTestPeer::NumRetiredFullRamInputs(fusion),
                4);
    }
    if (variant.bounded_cache) {
      const auto stats = fusion.GetBoundedCacheStats();
      ASSERT_TRUE(stats);
      EXPECT_GT(stats->loads, 0);
      EXPECT_EQ(stats->reserved_bytes, 0);
      EXPECT_LE(stats->peak_charged_bytes, stats->capacity_bytes);
      if (name == "bounded_whole" || name == "bounded_components") {
        EXPECT_GT(stats->evictions, 0);
      }
      if (name == "complete_neighborhood_compact_rgb") {
        // The current complete compact-RGB neighborhood stays on the generic
        // path. A change to borrowed traversal must qualify the same oracle.
        EXPECT_FALSE(compact_borrowed);
        EXPECT_GT(stats->component_loads[static_cast<size_t>(
                      BoundedWorkspace::Component::kImage)],
                  0);
      }
      if (variant.bounded_cache_normal_tile_size > 0) {
        EXPECT_GT(stats->normal_tile_loads, 0);
        EXPECT_TRUE(sparse_pinned);
        EXPECT_GT(fusion.GetMaxBoundedPrefillLanesScheduled(), 1);
        EXPECT_LE(fusion.GetMaxBoundedPrefillLanesScheduled(),
                  static_cast<size_t>(variant.bounded_cache_prefill_threads));
      }
    }
    if (variant.bounded_cache_normal_mapped_bytes > 0) {
      const auto stats = fusion.GetMappedNormalStats();
      EXPECT_GT(stats.loads, 0);
      EXPECT_EQ(stats.active_copies, 0);
      EXPECT_LE(stats.peak_charged_bytes,
                static_cast<size_t>(variant.bounded_cache_normal_mapped_bytes));
    }
  }
}

TEST_F(StereoFusionPlaneOracle, RetirementCancellationAndRepeat) {
  for (const int workers : {1, 3}) {
    SCOPED_TRACE(workers);
    auto options = Options();
    options.num_threads = workers;
    StereoFusion fusion(options, root_, "COLMAP", "", "geometric");
    fusion.SetCheckIfStoppedFunc([&] {
      return StereoFusionExactDepthTestPeer::NumRetiredFullRamInputs(fusion) >
             0;
    });
    fusion.Run();
    EXPECT_EQ(StereoFusionExactDepthTestPeer::NumRetiredFullRamInputs(fusion),
              1);
    ASSERT_NO_FATAL_FAILURE(ExpectPoints(fusion, true));
    fusion.SetCheckIfStoppedFunc([] { return false; });
    for (int run = 0; run < 2; ++run) {
      fusion.Run();
      ASSERT_NO_FATAL_FAILURE(ExpectPoints(fusion));
      EXPECT_EQ(StereoFusionExactDepthTestPeer::NumRetiredFullRamInputs(fusion),
                4);
    }
  }
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

  // A reader may start another fusion while a worker is inside Fuse. The
  // second Run uses its own ThreadPool, so its Fuse must have another TLS
  // scratch even though the outer worker is still waiting in the callback.
  {
    const auto image_path = temp_dir / "images" / image_names.front();
    Bitmap bitmap;
    ASSERT_TRUE(bitmap.Read(image_path, true));
    auto cached_options = options;
    cached_options.bounded_cache = false;
    cached_options.use_cache = true;
    cached_options.num_threads = 1;
    ASSERT_TRUE(cached_options.Check());
    std::atomic<int> callback_calls{0};
    std::thread::id outer_worker;
    std::thread::id nested_worker;
    std::unique_ptr<StereoFusion> nested;
    auto reader = ScopedBitmapReader::Create(
        {image_path},
        [&](const std::filesystem::path&,
            bool as_rgb,
            bool linear,
            Bitmap* target) {
          EXPECT_TRUE(as_rgb);
          EXPECT_FALSE(linear);
          const int call = ++callback_calls;
          if (call == 1) {
            outer_worker = std::this_thread::get_id();
            nested = std::make_unique<StereoFusion>(
                cached_options, temp_dir, "COLMAP", "", "geometric");
            nested->Run();
          } else if (call == 2) {
            nested_worker = std::this_thread::get_id();
          }
          *target = bitmap;
          return true;
        },
        {{30, 20}});
    StereoFusion outer(cached_options, temp_dir, "COLMAP", "", "geometric");
    outer.Run();
    ASSERT_TRUE(nested);
    EXPECT_GE(callback_calls.load(), 2);
    EXPECT_NE(outer_worker, std::this_thread::get_id());
    EXPECT_NE(outer_worker, nested_worker);
    expect_same_fusion(*nested, "nested_reader_fusion");
    expect_same_fusion(outer, "outer_reader_fusion");
    reader->Close();
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
    // Keep this cohort on the partial tile path: two whole images do not fit.
    options.bounded_cache_normal_tile_slots = 2;
    options.cache_size = 20000.0 / (1024.0 * 1024.0 * 1024.0);
    ASSERT_TRUE(options.Check());
    StereoFusion tile_neighborhood(
        options, temp_dir, "COLMAP", "", "geometric");
    tile_neighborhood.Run();
    expect_same_fusion(tile_neighborhood, "tile_neighborhood");
    for (const int prefill_threads : {1, 4}) {
      SCOPED_TRACE(prefill_threads);
      auto hybrid_options = options;
      hybrid_options.bounded_cache_normal_tile_slots = 2;
      hybrid_options.bounded_cache_prefill_threads = prefill_threads;
      hybrid_options.cache_size = 80000.0 / (1024.0 * 1024.0 * 1024.0);
      StereoFusion hybrid(hybrid_options, temp_dir, "COLMAP", "", "geometric");
      for (int repeat = 0; repeat < 2; ++repeat) {
        hybrid.Run();
        expect_same_fusion(hybrid, "hybrid-normal-window");
        ASSERT_EQ(hybrid.GetFusedPoints().size(), first_points.size());
        EXPECT_EQ(hybrid.GetFusedPointsVisibility(), first_visibility);
        for (size_t index = 0; index < first_points.size(); ++index) {
          const auto& a = first_points[index];
          const auto& b = hybrid.GetFusedPoints()[index];
          const float expected_values[] = {a.x, a.y, a.z, a.nx, a.ny, a.nz};
          const float actual_values[] = {b.x, b.y, b.z, b.nx, b.ny, b.nz};
          std::array<uint32_t, 6> expected_bits, actual_bits;
          static_assert(sizeof(expected_bits) == sizeof(expected_values));
          static_assert(sizeof(actual_bits) == sizeof(actual_values));
          std::memcpy(
              expected_bits.data(), expected_values, sizeof(expected_bits));
          std::memcpy(actual_bits.data(), actual_values, sizeof(actual_bits));
          EXPECT_EQ(expected_bits, actual_bits);
          EXPECT_EQ(a.r, b.r);
          EXPECT_EQ(a.g, b.g);
          EXPECT_EQ(a.b, b.b);
        }
        const auto stats = hybrid.GetBoundedCacheStats();
        ASSERT_TRUE(stats);
        EXPECT_GE(stats->component_loads[static_cast<size_t>(
                      BoundedWorkspace::Component::kImage)],
                  2);
        EXPECT_EQ(stats->component_loads[static_cast<size_t>(
                      BoundedWorkspace::Component::kNormal)],
                  0);
        EXPECT_EQ(stats->normal_tile_loads, 0);
        EXPECT_EQ(stats->reserved_bytes, 0);
        EXPECT_LE(stats->peak_charged_bytes, stats->capacity_bytes);
        EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowOwners(hybrid),
                  0);
        EXPECT_EQ(StereoFusionExactDepthTestPeer::NormalWindowPointers(hybrid),
                  0);
        const auto row = hybrid.GetBoundedCacheDiagnosticRow();
        ASSERT_TRUE(row);
        EXPECT_NE(row->find("normal_prefill_admitted="), std::string::npos);
      }
    }
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
    EXPECT_GT(tile_neighborhood.GetBoundedCacheStats()
                  ->component_loads[static_cast<size_t>(
                      BoundedWorkspace::Component::kImage)],
              0);
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

#ifdef __linux__
TEST(WritePointsVisibility, FailedOutputIsReported) {
  EXPECT_THROW(WritePointsVisibility("/dev/full", {{1, 2}, {}, {3}}),
               std::invalid_argument);
}
#endif  // __linux__

namespace {
std::string ReadFusionTestBytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK(file.is_open()) << path;
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}
}  // namespace

TEST(WriteFusedPointCloud, ReplacesPairAndRejectsCountMismatch) {
  const auto dir = CreateTestDir();
  const auto path = dir / "fused.ply";
  const auto vis = AddFileExtension(path, ".vis");
  PlyPoint point;
  point.x = 1.25f;
  point.g = 17;
  const std::vector<PlyPoint> points{point};
  const std::vector<std::vector<int>> visibility{{1, 3}};
  WriteFusedPointCloud(path, points, visibility);
  const auto old_ply = ReadFusionTestBytes(path);
  const auto old_vis = ReadFusionTestBytes(vis);
  EXPECT_THROW(WriteFusedPointCloud(path, points, {}), std::invalid_argument);
  EXPECT_EQ(ReadFusionTestBytes(path), old_ply);
  EXPECT_EQ(ReadFusionTestBytes(vis), old_vis);

  point.x = 2.5f;
  point.g = 31;
  const std::vector<PlyPoint> replacement{point};
  const std::vector<std::vector<int>> replacement_vis{{2}};
  WriteFusedPointCloud(path, replacement, replacement_vis);
  const auto actual = ReadPly(path);
  ASSERT_EQ(actual.size(), 1);
  EXPECT_FLOAT_EQ(actual[0].x, 2.5f);
  EXPECT_EQ(actual[0].g, 31);
  EXPECT_EQ(ReadPointsVisibility(vis, 1), replacement_vis);
  const auto expected_ply = dir / "expected.ply";
  const auto expected_vis = dir / "expected.vis";
  WriteBinaryPlyPoints(expected_ply, replacement);
  WritePointsVisibility(expected_vis, replacement_vis);
  EXPECT_EQ(ReadFusionTestBytes(path), ReadFusionTestBytes(expected_ply));
  EXPECT_EQ(ReadFusionTestBytes(vis), ReadFusionTestBytes(expected_vis));
}

TEST(WriteFusedPointCloud,
     VisibilityPublishFailureRemovesCompletionAndRetries) {
  const auto dir = CreateTestDir();
  const auto path = dir / "fused.ply";
  const auto vis = AddFileExtension(path, ".vis");
  const std::vector<PlyPoint> points(1);
  const std::vector<std::vector<int>> visibility{{1}};
  WriteFusedPointCloud(path, points, visibility);
  ASSERT_TRUE(std::filesystem::remove(vis));
  ASSERT_TRUE(std::filesystem::create_directory(vis));
  EXPECT_THROW(WriteFusedPointCloud(path, points, visibility), std::exception);
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_TRUE(std::filesystem::is_directory(vis));
  ASSERT_TRUE(std::filesystem::remove(vis));
  EXPECT_NO_THROW(WriteFusedPointCloud(path, points, visibility));
  EXPECT_EQ(ReadPly(path).size(), 1);
  EXPECT_EQ(ReadPointsVisibility(vis, 1), visibility);
}

#ifdef __linux__
TEST(WriteFusedPointCloud, StagingWriteFailurePreservesExistingPair) {
  const auto dir = CreateTestDir();
  const auto path = dir / "fused.ply";
  const auto vis = AddFileExtension(path, ".vis");
  const std::vector<PlyPoint> points(1);
  const std::vector<std::vector<int>> visibility{{1}};
  WriteFusedPointCloud(path, points, visibility);
  const auto old_ply = ReadFusionTestBytes(path);
  const auto old_vis = ReadFusionTestBytes(vis);
  struct LimitGuard {
    struct rlimit old_limit;
    using Handler = void (*)(int);
    Handler old_handler;
    explicit LimitGuard(rlim_t max_bytes) {
      THROW_CHECK_EQ(getrlimit(RLIMIT_FSIZE, &old_limit), 0);
      old_handler = std::signal(SIGXFSZ, SIG_IGN);
      THROW_CHECK(old_handler != SIG_ERR);
      struct rlimit limited = old_limit;
      limited.rlim_cur = max_bytes;
      if (setrlimit(RLIMIT_FSIZE, &limited) != 0) {
        std::signal(SIGXFSZ, old_handler);
        throw std::runtime_error("Cannot set file size limit");
      }
    }
    ~LimitGuard() {
      setrlimit(RLIMIT_FSIZE, &old_limit);
      std::signal(SIGXFSZ, old_handler);
    }
  };
  // Verify the 1024-byte boundary reaches payload rather than header.
  const std::vector<PlyPoint> large_points(512);
  const std::vector<std::vector<int>> large_visibility(512, {1});
  const auto sizing_ply = dir / "sizing.ply";
  WriteBinaryPlyPoints(sizing_ply, large_points);
  const auto sizing_bytes = ReadFusionTestBytes(sizing_ply);
  const auto header_end = sizing_bytes.find("end_header\n");
  ASSERT_NE(header_end, std::string::npos);
  ASSERT_LT(header_end + std::string("end_header\n").size(), 1024);
  ASSERT_GT(sizing_bytes.size(), 1024);
  ASSERT_TRUE(std::filesystem::remove(sizing_ply));

  // One-point PLY fits; a long visibility list forces its later stage to fail.
  const std::vector<std::vector<int>> long_visibility(1,
                                                      std::vector<int>(512, 1));
  const auto small_ply = dir / "small.ply";
  const auto large_vis = dir / "large.vis";
  WriteBinaryPlyPoints(small_ply, points);
  WritePointsVisibility(large_vis, long_visibility);
  ASSERT_LT(std::filesystem::file_size(small_ply), 1024);
  ASSERT_GT(std::filesystem::file_size(large_vis), 1024);
  ASSERT_TRUE(std::filesystem::remove(small_ply));
  ASSERT_TRUE(std::filesystem::remove(large_vis));

  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    std::string error;
    {
      LimitGuard guard(mode == 0 ? 0 : 1024);
      try {
        if (mode == 2) {
          WriteFusedPointCloud(path, points, long_visibility);
        } else {
          WriteFusedPointCloud(path, large_points, large_visibility);
        }
      } catch (const std::exception& exception) {
        error = exception.what();
      }
    }
    EXPECT_NE(error.find(mode == 2 ? "Failed to write visibility file"
                                   : "Failed to write PLY file"),
              std::string::npos);
    EXPECT_EQ(ReadFusionTestBytes(path), old_ply);
    EXPECT_EQ(ReadFusionTestBytes(vis), old_vis);
  }
  EXPECT_NO_THROW(WriteFusedPointCloud(path, points, visibility));
}
#endif  // __linux__

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

TEST(ReadPointsVisibility, ExtremeTruncatedListDoesNotAllocateDeclaredCount) {
  const auto path = CreateTestDir() / "extreme-count.vis";
  for (const size_t payload_words : {size_t{0}, size_t{8193}}) {
    SCOPED_TRACE(payload_words);
    {
      std::ofstream file(path, std::ios::binary);
      WriteBinaryLittleEndian<uint64_t>(&file, 1);
      WriteBinaryLittleEndian<uint32_t>(&file,
                                        std::numeric_limits<uint32_t>::max());
      for (size_t i = 0; i < payload_words; ++i) {
        WriteBinaryLittleEndian<uint32_t>(&file, 7);
      }
    }
    EXPECT_THROW(ReadPointsVisibility(path, 1), std::invalid_argument);
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

TEST(StereoFusion, BoundingBoxControlsLazyRgbDemand) {
  const auto root = CreateTwoViewNormalDecisionFixture();
  const auto names = ReadTextFileLines(root / "stereo/fusion.cfg");
  ASSERT_EQ(names.size(), 2);
  std::vector<std::filesystem::path> paths;
  paths.reserve(names.size());
  for (const auto& name : names) paths.push_back(root / "images" / name);
  Bitmap rgb(30, 20, true);
  rgb.Fill(BitmapColor<uint8_t>(0, 64, 128));
  for (const bool fail_rgb : {false, true}) {
    // Check included geometry first in both the successful and failing cases.
    for (const bool excluded : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "excluded=" << excluded << ", fail_rgb=" << fail_rgb);
      std::atomic<int> reads{0};
      auto reader = ScopedBitmapReader::Create(
          paths,
          [&](const std::filesystem::path&, bool, bool, Bitmap* target) {
            ++reads;
            if (fail_rgb) return false;
            *target = rgb;
            return true;
          },
          std::vector<std::pair<int, int>>(paths.size(), {30, 20}));
      StereoFusionOptions options;
      options.num_threads = 1;
      options.min_num_pixels = options.max_num_pixels = 2;
      options.bounded_cache = options.bounded_cache_components = true;
      if (excluded) {
        options.bounding_box.first.setConstant(1e6f);
        options.bounding_box.second.setConstant(2e6f);
      }
      StereoFusion fusion(options, root, "COLMAP", "", "geometric");
      if (fail_rgb && !excluded) {
        try {
          fusion.Run();
          FAIL() << "Expected required RGB input to report the read failure";
        } catch (const AggregateException& error) {
          ASSERT_FALSE(error.exceptions().empty());
          for (const auto& exception : error.exceptions()) {
            try {
              std::rethrow_exception(exception);
            } catch (const std::invalid_argument& cause) {
              EXPECT_NE(std::string(cause.what()).find("data->bitmap.Read"),
                        std::string::npos);
            }
          }
        }
        EXPECT_GT(reads.load(), 0);
      } else {
        ASSERT_NO_THROW(fusion.Run());
        if (excluded) {
          EXPECT_TRUE(fusion.GetFusedPoints().empty());
          EXPECT_TRUE(fusion.GetFusedPointsVisibility().empty());
          EXPECT_EQ(reads.load(), 0);
        } else {
          EXPECT_EQ(reads.load(), 2);
          EXPECT_EQ(fusion.GetFusedPoints().size(), 600);
          EXPECT_EQ(fusion.GetFusedPointsVisibility().size(), 600);
          for (const auto& point : fusion.GetFusedPoints()) {
            EXPECT_FLOAT_EQ(point.z, 5.0f);
            EXPECT_EQ(point.r, 0);
            EXPECT_EQ(point.g, 64);
            EXPECT_EQ(point.b, 128);
          }
        }
      }
      const auto stats = fusion.GetBoundedCacheStats();
      ASSERT_TRUE(stats);
      const auto bitmap_loads = stats->component_loads[static_cast<size_t>(
          BoundedWorkspace::Component::kBitmap)];
      if (fail_rgb && !excluded) {
        EXPECT_GT(bitmap_loads, 0);
      } else {
        EXPECT_EQ(bitmap_loads, excluded ? 0 : 2);
      }
      EXPECT_EQ(stats->reserved_bytes, 0);
      // Rejection still visits; required-color failure must not visit pixels.
      for (size_t view = 0; view < names.size(); ++view) {
        for (size_t row = 0; row < 20; ++row) {
          for (size_t col = 0; col < 30; ++col) {
            EXPECT_EQ(StereoFusionExactDepthTestPeer::PixelMaskValue(
                          fusion, view, row, col),
                      fail_rgb && !excluded ? 0 : 1);
          }
        }
      }
      reader->Close();
    }
  }
}

TEST(StereoFusion, SparseDepthCancellationFailureAndReuse) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = CreateTwoViewNormalDecisionFixture();
  const auto names = ReadTextFileLines(root / "stereo/fusion.cfg");
  for (const auto& name : names) {
    Mat<float> depth(30, 20, 1);
    depth.Fill(0.0f);
    depth.Set(10, 15, 0, 5.0f);
    depth.Write(root / "stereo/depth_maps" / (name + ".geometric.bin"));
  }
  StereoFusionOptions options;
  options.num_threads = 1;
  options.use_cache = false;
  options.min_num_pixels = options.max_num_pixels = 2;
  StereoFusion reference(options, root, "COLMAP", "", "geometric");
  reference.Run();
  ASSERT_EQ(reference.GetFusedPoints().size(), 1);
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = 4;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 4;
  options.cache_size = 20000.0 / (1ULL << 30);
  {
    StereoFusion uncompressed(options, root, "COLMAP", "", "geometric");
    uncompressed.Run();
    EXPECT_FALSE(
        StereoFusionExactDepthTestPeer::Cache(uncompressed).UsesCompactDepth());
    EXPECT_EQ(uncompressed.GetFusedPointsVisibility(),
              reference.GetFusedPointsVisibility());
  }
  options.bounded_cache_compact_depth = true;
  StereoFusion fusion(options, root, "COLMAP", "", "geometric");
  bool cancelled = false;
  fusion.SetCheckIfStoppedFunc([&] {
    const auto stats = fusion.GetBoundedCacheStats();
    cancelled |=
        stats && stats->component_loads[1] > 0 && stats->reserved_bytes == 0;
    return cancelled;
  });
  fusion.Run();
  EXPECT_TRUE(cancelled);
  EXPECT_TRUE(fusion.GetFusedPoints().empty());
  ASSERT_TRUE(fusion.GetBoundedCacheStats());
  EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
  const auto path = root / "stereo/normal_maps" / (names[0] + ".geometric.bin");
  Mat<float> saved;
  saved.Read(path);
  bool injected = false;
  fusion.SetCheckIfStoppedFunc([&] {
    // Run replaces the old workspace before this callback. Windows cannot
    // truncate inputs while that workspace still retains their mappings.
    if (!injected) {
      std::ofstream file;
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.open(path, std::ios::binary | std::ios::trunc);
      file << "30&20&3&";
      file.close();
      injected = true;
    }
    return false;
  });
  EXPECT_THROW(fusion.Run(), std::exception);
  ASSERT_TRUE(injected);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
  saved.Write(path);
  bool packed_observed = false;
  fusion.SetCheckIfStoppedFunc([&] {
    packed_observed |=
        StereoFusionExactDepthTestPeer::HasPackedResident(fusion);
    return false;
  });
  for (int repeat = 0; repeat < 2; ++repeat) {
    packed_observed = false;
    fusion.Run();
    EXPECT_TRUE(packed_observed);
    ASSERT_EQ(fusion.GetFusedPoints().size(), 1);
    EXPECT_EQ(fusion.GetFusedPointsVisibility(),
              reference.GetFusedPointsVisibility());
    const auto& actual = fusion.GetFusedPoints()[0];
    const auto& expected = reference.GetFusedPoints()[0];
    EXPECT_EQ(actual.x, expected.x);
    EXPECT_EQ(actual.y, expected.y);
    EXPECT_EQ(actual.z, expected.z);
    EXPECT_EQ(actual.nx, expected.nx);
    EXPECT_EQ(actual.ny, expected.ny);
    EXPECT_EQ(actual.nz, expected.nz);
    EXPECT_EQ(actual.r, expected.r);
    EXPECT_EQ(actual.g, expected.g);
    EXPECT_EQ(actual.b, expected.b);
    EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
  }
}

TEST(StereoFusion, SparseDepthKeepsWholeViewFallbackWithoutTileHeadroom) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = CreateTwoViewNormalDecisionFixture();
  Workspace::Options workspace_options;
  workspace_options.workspace_path = root;
  workspace_options.workspace_format = "COLMAP";
  workspace_options.input_type = "geometric";
  BoundedWorkspace estimates(workspace_options);
  using Component = BoundedWorkspace::Component;
  size_t total = 0, extra = 0;
  for (const int view : {0, 1}) {
    const size_t resident = estimates.GetResidentBytesEstimate(view);
    total += resident;
    extra = std::max(extra, estimates.GetLoadReservationBytes(view) - resident);
  }
  StereoFusionOptions options;
  options.num_threads = 24;
  options.min_num_pixels = options.max_num_pixels = 2;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_compact_depth = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = 4;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 2;
  const size_t capacity = total + extra + 64;
  options.cache_size = static_cast<double>(capacity) / (1ULL << 30);
  ASSERT_GT(estimates.GetLoadReservationBytes(0, Component::kDepth),
            capacity / 24);
  StereoFusion fusion(options, root, "COLMAP", "", "geometric");
  ASSERT_NO_THROW(fusion.Run());
  EXPECT_FALSE(fusion.GetFusedPoints().empty());
  ASSERT_TRUE(fusion.GetBoundedCacheStats());
  EXPECT_EQ(fusion.GetBoundedCacheStats()->component_loads[0], 2);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->normal_tile_loads, 0);
  EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
}

TEST(StereoFusion, SparseDepthKeepsRawPartialFallbackAtTightAdmission) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = CreateTwoViewNormalDecisionFixture();
  StereoFusionOptions options;
  options.num_threads = 2;
  options.min_num_pixels = options.max_num_pixels = 2;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_compact_depth = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = 4;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 2;
  options.cache_size = 14000.0 / (1ULL << 30);
  StereoFusion fusion(options, root, "COLMAP", "", "geometric");
  ASSERT_NO_THROW(fusion.Run());
  EXPECT_FALSE(fusion.GetFusedPoints().empty());
  EXPECT_FALSE(
      StereoFusionExactDepthTestPeer::Cache(fusion).UsesCompactDepth());
  const auto stats = fusion.GetBoundedCacheStats();
  ASSERT_TRUE(stats);
  // The last remaining view fits whole after the initial partial root.
  EXPECT_EQ(stats->component_loads[0], 1);
  EXPECT_GT(stats->component_loads[1], 0);
  EXPECT_EQ(stats->reserved_bytes, 0);
  EXPECT_LE(stats->peak_charged_bytes, 14000);
}

TEST(StereoFusion, ZeroMinimumSkipsEmptySamples) {
  for (const bool valid_sample : {false, true}) {
    SCOPED_TRACE(valid_sample);
    const auto root = CreateTwoViewNormalDecisionFixture();
    const auto names = ReadTextFileLines(root / "stereo" / "fusion.cfg");
    ASSERT_EQ(names.size(), 2);
    for (size_t index = 0; index < names.size(); ++index) {
      Mat<float> depth(30, 20, 1);
      depth.Fill(0.0f);
      if (valid_sample && index == 0) depth.Set(0, 0, 0, 5.0f);
      depth.Write(root / "stereo" / "depth_maps" /
                  (names[index] + ".geometric.bin"));
    }

    for (const int mode : {0, 1, 2}) {
      SCOPED_TRACE(mode);
      StereoFusionOptions options;
      options.num_threads = 2;
      options.min_num_pixels = 0;
      options.use_cache = mode == 1;
      options.bounded_cache = mode == 2;
      options.cache_size = 1.0 / 1024.0;
      ASSERT_TRUE(options.Check());
      StereoFusion fusion(options, root, "COLMAP", "", "geometric");
      ASSERT_NO_THROW(fusion.Run());
      const auto& points = fusion.GetFusedPoints();
      const auto& visibility = fusion.GetFusedPointsVisibility();
      ASSERT_EQ(points.size(), valid_sample ? 1 : 0);
      ASSERT_EQ(visibility.size(), points.size());
      if (valid_sample) {
        EXPECT_FLOAT_EQ(points[0].z, 5.0f);
        EXPECT_EQ(points[0].r, 0);
        EXPECT_EQ(points[0].g, 64);
        EXPECT_EQ(points[0].b, 128);
        ASSERT_EQ(visibility[0].size(), 1);
      }
    }
  }
}

TEST(StereoFusion, BoundedNeighborhoodSeedsNonpositiveDepthBeforeRows) {
  const auto root = CreateTwoViewNormalDecisionFixture();
  const auto names = ReadTextFileLines(root / "stereo" / "fusion.cfg");
  ASSERT_EQ(names.size(), 2);
  const auto write_depth = [&](float value) {
    Mat<float> depth(30, 20, 1);
    depth.Fill(value);
    for (const auto& name : names) {
      depth.Write(root / "stereo" / "depth_maps" / (name + ".geometric.bin"));
    }
  };
  Mat<float> probe_depth(1, 1, 1);
  probe_depth.Fill(0.0f);
  internal::FusionPixelMask probe_mask(1, 1);
  probe_mask.ExcludeNonpositiveDepth(internal::FusionFloatMapView(probe_depth));
  const bool seed_supported = probe_mask.Get(0, 0) != 0;
  for (const bool components : {false, true}) {
    SCOPED_TRACE(components);
    write_depth(0.0f);
    StereoFusionOptions options;
    options.num_threads = 4;
    options.min_num_pixels = options.max_num_pixels = 2;
    options.bounded_cache = true;
    options.bounded_cache_components = components;
    options.bounded_cache_neighborhood = true;
    options.bounded_cache_diagnostics = true;
    options.cache_size = 1.0;
    StereoFusion fusion(options, root, "COLMAP", "", "geometric");
    fusion.Run();
    EXPECT_TRUE(fusion.GetFusedPoints().empty());
    EXPECT_TRUE(fusion.GetFusedPointsVisibility().empty());
    const auto stats = fusion.GetBoundedCacheStats();
    ASSERT_TRUE(stats);
    EXPECT_GT(stats->loads, 0);
    EXPECT_EQ(stats->reserved_bytes, 0);
    EXPECT_LE(stats->peak_charged_bytes, stats->capacity_bytes);
    for (size_t image = 0; image < 2; ++image) {
      for (size_t row = 0; row < 20; ++row) {
        for (size_t col = 0; col < 30; ++col) {
          EXPECT_EQ(StereoFusionExactDepthTestPeer::PixelMaskValue(
                        fusion, image, row, col),
                    seed_supported ? 1 : 0);
        }
      }
    }
    if (seed_supported) {
      EXPECT_EQ(StereoFusionExactDepthTestPeer::DepthBorrowCount(fusion), 0);
    }
    // A fresh Run must not retain exclusions from a previous depth generation.
    write_depth(5.0f);
    fusion.Run();
    EXPECT_FALSE(fusion.GetFusedPoints().empty());
  }
}

TEST(StereoFusion, CompleteBoundedWindowBorrowsResidentViewsAndCancels) {
  const auto root = CreateTwoViewNormalDecisionFixture();
  StereoFusionOptions options;
  options.num_threads = 1;
  options.min_num_pixels = options.max_num_pixels = 2;
  options.bounded_cache = true;
  options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.cache_size = 1.0;
  StereoFusion fusion(options, root, "COLMAP", "", "geometric");
  bool observed = false;
  fusion.SetCheckIfStoppedFunc([&] {
    observed |= StereoFusionExactDepthTestPeer::HasFullRamInputs(fusion);
    return observed;
  });
  fusion.Run();
  EXPECT_TRUE(observed);
  EXPECT_TRUE(fusion.GetFusedPoints().empty());
  EXPECT_FALSE(StereoFusionExactDepthTestPeer::HasFullRamInputs(fusion));
  ASSERT_TRUE(fusion.GetBoundedCacheStats());
  EXPECT_EQ(fusion.GetBoundedCacheStats()->reserved_bytes, 0);
  observed = false;
  fusion.SetCheckIfStoppedFunc([&] {
    observed |= StereoFusionExactDepthTestPeer::HasFullRamInputs(fusion);
    return false;
  });
  fusion.Run();
  EXPECT_TRUE(observed);
  EXPECT_FALSE(fusion.GetFusedPoints().empty());
  EXPECT_FALSE(StereoFusionExactDepthTestPeer::HasFullRamInputs(fusion));
  const auto names = ReadTextFileLines(root / "stereo" / "fusion.cfg");
  Mat<float> depth(15, 10, 1);
  depth.Fill(5.0f);
  Mat<float> normal(15, 10, 3);
  normal.Fill(0.0f);
  for (size_t row = 0; row < 10; ++row) {
    for (size_t col = 0; col < 15; ++col) normal.Set(row, col, 2, 1.0f);
  }
  for (const auto& name : names) {
    depth.Write(root / "stereo" / "depth_maps" / (name + ".geometric.bin"));
    normal.Write(root / "stereo" / "normal_maps" / (name + ".geometric.bin"));
  }
  options.bounded_cache_compact_rgb = true;
  StereoFusion compact(options, root, "COLMAP", "", "geometric");
  bool compact_resident = false;
  compact.SetCheckIfStoppedFunc([&] {
    compact_resident |=
        StereoFusionExactDepthTestPeer::HasFullRamInputs(compact);
    return false;
  });
  compact.Run();
  EXPECT_FALSE(compact_resident);
  EXPECT_FALSE(compact.GetFusedPoints().empty());
  auto generic_options = options;
  generic_options.bounded_cache_neighborhood = false;
  StereoFusion generic(generic_options, root, "COLMAP", "", "geometric");
  generic.Run();
  ASSERT_EQ(compact.GetFusedPoints().size(), generic.GetFusedPoints().size());
  EXPECT_EQ(compact.GetFusedPointsVisibility(),
            generic.GetFusedPointsVisibility());
  for (size_t i = 0; i < compact.GetFusedPoints().size(); ++i) {
    const auto& actual = compact.GetFusedPoints()[i];
    const auto& expected = generic.GetFusedPoints()[i];
    EXPECT_FLOAT_EQ(actual.x, expected.x);
    EXPECT_FLOAT_EQ(actual.y, expected.y);
    EXPECT_FLOAT_EQ(actual.z, expected.z);
    EXPECT_FLOAT_EQ(actual.nx, expected.nx);
    EXPECT_FLOAT_EQ(actual.ny, expected.ny);
    EXPECT_FLOAT_EQ(actual.nz, expected.nz);
    EXPECT_EQ(actual.r, expected.r);
    EXPECT_EQ(actual.g, expected.g);
    EXPECT_EQ(actual.b, expected.b);
  }
}

TEST(StereoFusion, CompleteViewSetDoesNotReserveWorkerCopies) {
  const auto root = CreateTwoViewNormalDecisionFixture();
  Workspace::Options workspace_options;
  workspace_options.workspace_path = root;
  workspace_options.workspace_format = "COLMAP";
  workspace_options.input_type = "geometric";
  workspace_options.image_as_rgb = true;
  workspace_options.cache_size = 1.0;
  BoundedWorkspace estimates(workspace_options);
  using Component = BoundedWorkspace::Component;
  size_t resident_total = 0, loader_extra = 0;
  for (const int view : {0, 1}) {
    const size_t resident =
        estimates.GetResidentBytesEstimate(view, Component::kImage);
    const size_t reserved =
        estimates.GetLoadReservationBytes(view, Component::kImage);
    ASSERT_GE(reserved, resident);
    resident_total += resident;
    loader_extra = std::max(loader_extra, reserved - resident);
  }
  for (const int workers : {1, 4, 12, 24}) {
    SCOPED_TRACE(workers);
    StereoFusionOptions options;
    options.num_threads = workers;
    options.min_num_pixels = options.max_num_pixels = 2;
    options.bounded_cache = true;
    options.bounded_cache_components = true;
    options.bounded_cache_neighborhood = true;
    options.cache_size =
        static_cast<double>(resident_total + loader_extra + 64) / (1ULL << 30);
    StereoFusion fusion(options, root, "COLMAP", "", "geometric");
    fusion.Run();
    ASSERT_FALSE(fusion.GetFusedPoints().empty());
    const auto stats = fusion.GetBoundedCacheStats();
    ASSERT_TRUE(stats);
    EXPECT_EQ(stats->component_loads[static_cast<size_t>(Component::kImage)],
              2);
    for (const auto part :
         {Component::kDepth, Component::kNormal, Component::kBitmap}) {
      EXPECT_EQ(stats->component_loads[static_cast<size_t>(part)], 0);
    }
    EXPECT_LE(stats->peak_charged_bytes, stats->capacity_bytes);
    EXPECT_EQ(stats->reserved_bytes, 0);
  }
}

// Root 0 is a leaf, root 1 a hub; after the first root is fused the
// second root must not inherit its complete-neighborhood admission.
TEST(StereoFusion, CompleteTileRootThenPartialRootNeedsFallbackHeadroom) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = CreateTestDir();
  for (const auto* dir :
       {"sparse", "images", "stereo/depth_maps", "stereo/normal_maps"}) {
    CreateDirIfNotExists(root / dir, true);
  }
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 4;
  dataset.num_points3D = 90;
  dataset.camera_width = 30;
  dataset.camera_height = 20;
  dataset.camera_params = {30.0, 15.0, 10.0, 0.0};
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    reconstruction.Frame(frame_id).SetRigFromWorld(Rigid3d());
  }
  reconstruction.Write(root / "sparse");
  Model initial;
  initial.Read(root, "COLMAP");
  ASSERT_EQ(initial.images.size(), 4);
  std::vector<image_t> image_ids;
  std::vector<std::string> image_names;
  for (int index = 0; index < 4; ++index) {
    const auto& name = initial.GetImageName(index);
    image_names.push_back(name);
    bool found = false;
    for (const auto& [id, image] : reconstruction.Images()) {
      if (image.Name() == name) {
        image_ids.push_back(id);
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }
  std::vector<point3D_t> point_ids;
  for (const auto& [id, point] : reconstruction.Points3D()) {
    ASSERT_EQ(point.track.Length(), 4);
    point_ids.push_back(id);
  }
  std::sort(point_ids.begin(), point_ids.end());
  ASSERT_GE(point_ids.size(), 3);
  const std::array<int, 3> leaves{0, 2, 3};
  for (size_t index = 0; index < point_ids.size(); ++index) {
    const image_t leaf = image_ids[leaves[index % leaves.size()]];
    const auto elements =
        reconstruction.Point3D(point_ids[index]).track.Elements();
    for (const auto& element : elements) {
      if (element.image_id != image_ids[1] && element.image_id != leaf) {
        reconstruction.DeleteObservation(element.image_id, element.point2D_idx);
      }
    }
  }
  reconstruction.Write(root / "sparse");
  {
    std::ofstream config(root / "stereo/fusion.cfg");
    for (const auto& name : image_names) {
      config << name << '\n';
      Mat<float> depth(30, 20, 1);
      depth.Fill(5.0f);
      depth.Write(root / "stereo/depth_maps" / (name + ".geometric.bin"));
      Mat<float> normal(30, 20, 3);
      normal.Fill(0.0f);
      for (size_t row = 0; row < 20; ++row) {
        for (size_t col = 0; col < 30; ++col) normal.Set(row, col, 2, 1.0f);
      }
      normal.Write(root / "stereo/normal_maps" / (name + ".geometric.bin"));
      Bitmap bitmap(30, 20, true);
      bitmap.Fill(BitmapColor<uint8_t>(0, 64, 128));
      bitmap.Write(root / "images" / name);
    }
  }
  Model model;
  model.Read(root, "COLMAP");
  for (int index = 0; index < 4; ++index) {
    ASSERT_EQ(model.GetImageName(index), image_names[index]);
  }
  const auto adjacency = model.GetMaxOverlappingImages(50, 0.0);
  ASSERT_EQ(adjacency[0], std::vector<int>({1}));
  ASSERT_EQ(adjacency[1].size(), 3);
  EXPECT_NE(std::find(adjacency[1].begin(), adjacency[1].end(), 0),
            adjacency[1].end());
  EXPECT_NE(std::find(adjacency[1].begin(), adjacency[1].end(), 2),
            adjacency[1].end());
  EXPECT_NE(std::find(adjacency[1].begin(), adjacency[1].end(), 3),
            adjacency[1].end());

  Workspace::Options workspace_options;
  workspace_options.workspace_path = root;
  workspace_options.workspace_format = "COLMAP";
  workspace_options.input_type = "geometric";
  workspace_options.image_as_rgb = true;
  workspace_options.cache_size = 1.0;
  BoundedWorkspace estimates(workspace_options);
  using Component = BoundedWorkspace::Component;
  const size_t first = estimates.GetResidentBytesEstimate(0, Component::kImage);
  const size_t hub = estimates.GetResidentBytesEstimate(1, Component::kImage);
  const size_t extra =
      std::max(estimates.GetLoadReservationBytes(0, Component::kImage) - first,
               estimates.GetLoadReservationBytes(1, Component::kImage) - hub);
  const size_t capacity = first + hub + extra + 64;
  const size_t later =
      hub + estimates.GetResidentBytesEstimate(2, Component::kImage) +
      estimates.GetResidentBytesEstimate(3, Component::kImage);
  ASSERT_GT(later, capacity);
  const size_t tile = estimates.GetNormalTileResidentBytes(0, 7);
  const size_t component =
      std::max({estimates.GetNormalTileReservationBytes(0, 7),
                estimates.GetLoadReservationBytes(0, Component::kDepth),
                estimates.GetLoadReservationBytes(0, Component::kBitmap)});
  ASSERT_GT(component + 4 * tile, capacity / 12);
  StereoFusionOptions options;
  options.num_threads = 12;
  options.max_traversal_depth = 2;
  options.min_num_pixels = options.max_num_pixels = 2;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = 4;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 1;
  options.cache_size = static_cast<double>(capacity) / (1ULL << 30);
  ASSERT_TRUE(options.Check());
  StereoFusion fusion(options, root, "COLMAP", "", "geometric");
  try {
    fusion.Run();
    FAIL() << "Expected the later partial root to refuse tile fallback";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(
        std::string(error.what()).find("Partial normal-tile neighborhoods"),
        std::string::npos);
  }
  const auto stats = fusion.GetBoundedCacheStats();
  ASSERT_TRUE(stats);
  // Two whole owners prove the first root reached admission before the later
  // partial root was refused; no partial prefill or tile load may survive.
  EXPECT_EQ(stats->component_loads[static_cast<size_t>(Component::kImage)], 2);
  EXPECT_EQ(stats->normal_tile_loads, 0);
  EXPECT_LE(stats->peak_charged_bytes, stats->capacity_bytes);
  EXPECT_EQ(stats->reserved_bytes, 0);
  EXPECT_TRUE(fusion.GetFusedPoints().empty());
}

// A tight complete owner set must not be rejected merely because hypothetical
// per-worker fallback tiles would not fit. The estimator keeps this fixture
// independent of hardcoded cache accounting.
TEST(StereoFusion, CompleteTileOwnersDoNotRequireFallbackHeadroom) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = CreateTwoViewNormalDecisionFixture();
  Workspace::Options workspace_options;
  workspace_options.workspace_path = root;
  workspace_options.workspace_format = "COLMAP";
  workspace_options.input_type = "geometric";
  workspace_options.image_as_rgb = true;
  workspace_options.cache_size = 1.0;
  BoundedWorkspace estimates(workspace_options);
  using Component = BoundedWorkspace::Component;
  size_t resident_total = 0, loader_extra = 0, largest_component = 0;
  size_t largest_tile = 0;
  for (const int view : {0, 1}) {
    const size_t resident =
        estimates.GetResidentBytesEstimate(view, Component::kImage);
    const size_t reserved =
        estimates.GetLoadReservationBytes(view, Component::kImage);
    ASSERT_GE(reserved, resident);
    resident_total += resident;
    loader_extra = std::max(loader_extra, reserved - resident);
    largest_tile =
        std::max(largest_tile, estimates.GetNormalTileResidentBytes(view, 7));
    largest_component =
        std::max({largest_component,
                  estimates.GetNormalTileReservationBytes(view, 7),
                  estimates.GetLoadReservationBytes(view, Component::kDepth),
                  estimates.GetLoadReservationBytes(view, Component::kBitmap)});
  }
  constexpr size_t kWorkers = 12;
  constexpr size_t kSlots = 4;
  const size_t capacity = resident_total + loader_extra + 64;
  ASSERT_GT(largest_component + kSlots * largest_tile, capacity / kWorkers);
  StereoFusionOptions options;
  options.num_threads = kWorkers;
  options.min_num_pixels = options.max_num_pixels = 2;
  options.bounded_cache = options.bounded_cache_components = true;
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_tile_size = 7;
  options.bounded_cache_normal_tile_slots = kSlots;
  options.bounded_cache_normal_mapped_bytes = 1 << 20;
  options.bounded_cache_normal_mapped_files = 1;
  options.cache_size = static_cast<double>(capacity) / (1ULL << 30);
  ASSERT_TRUE(options.Check());
  StereoFusion complete(options, root, "COLMAP", "", "geometric");
  ASSERT_NO_THROW(complete.Run());
  EXPECT_FALSE(complete.GetFusedPoints().empty());
  const auto stats = complete.GetBoundedCacheStats();
  ASSERT_TRUE(stats);
  EXPECT_EQ(stats->capacity_bytes, capacity);
  EXPECT_EQ(stats->component_loads[static_cast<size_t>(Component::kImage)], 2);
  EXPECT_EQ(stats->normal_tile_loads, 0);
  EXPECT_LE(stats->peak_charged_bytes, stats->capacity_bytes);
  EXPECT_EQ(stats->reserved_bytes, 0);

  auto generous_options = options;
  const size_t generous_capacity =
      resident_total + loader_extra +
      kWorkers * (largest_component + kSlots * largest_tile) + 64;
  generous_options.cache_size =
      static_cast<double>(generous_capacity) / (1ULL << 30);
  StereoFusion generous(generous_options, root, "COLMAP", "", "geometric");
  ASSERT_NO_THROW(generous.Run());
  const auto sorted_points = [](const auto& points) {
    std::vector<std::array<float, 9>> result;
    result.reserve(points.size());
    for (const auto& p : points) {
      result.push_back({p.x,
                        p.y,
                        p.z,
                        p.nx,
                        p.ny,
                        p.nz,
                        static_cast<float>(p.r),
                        static_cast<float>(p.g),
                        static_cast<float>(p.b)});
    }
    std::sort(result.begin(), result.end());
    return result;
  };
  EXPECT_EQ(sorted_points(complete.GetFusedPoints()),
            sorted_points(generous.GetFusedPoints()));
  for (const auto& visibility : complete.GetFusedPointsVisibility()) {
    auto sorted = visibility;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted, (std::vector<int>{0, 1}));
  }

  // Whole owners no longer fit, and the same tile fallback is still too large.
  ASSERT_GT(resident_total, 1);
  const size_t partial_capacity = resident_total - 1;
  ASSERT_GT(largest_component + kSlots * largest_tile,
            partial_capacity / kWorkers);
  options.cache_size = static_cast<double>(partial_capacity) / (1ULL << 30);
  StereoFusion partial(options, root, "COLMAP", "", "geometric");
  EXPECT_THROW(partial.Run(), std::exception);
  EXPECT_TRUE(partial.GetFusedPoints().empty());
  const auto partial_stats = partial.GetBoundedCacheStats();
  ASSERT_TRUE(partial_stats);
  EXPECT_EQ(
      partial_stats->component_loads[static_cast<size_t>(Component::kImage)],
      0);
  EXPECT_EQ(partial_stats->normal_tile_loads, 0);
  EXPECT_EQ(partial_stats->reserved_bytes, 0);

  // Without neighborhoods there is no complete-owner admission to defer to.
  options.bounded_cache_neighborhood = false;
  options.cache_size = static_cast<double>(capacity) / (1ULL << 30);
  StereoFusion no_neighborhood(options, root, "COLMAP", "", "geometric");
  EXPECT_THROW(no_neighborhood.Run(), std::exception);
  ASSERT_TRUE(no_neighborhood.GetBoundedCacheStats());
  EXPECT_EQ(no_neighborhood.GetBoundedCacheStats()->reserved_bytes, 0);
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

TEST(StereoFusion, EmptyModelProducesNoOutput) {
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "images");
  CreateDirIfNotExists(root / "stereo");
  Reconstruction reconstruction;
  reconstruction.Write(root / "sparse");
  {
    std::ofstream config(root / "stereo/fusion.cfg");
    ASSERT_TRUE(config.is_open());
  }
  for (const int mode : {0, 1, 2, 3}) {
    SCOPED_TRACE(mode);
    StereoFusionOptions options;
    options.num_threads = 1;
    options.use_cache = mode == 1;
    options.bounded_cache = mode >= 2;
    options.bounded_cache_components = mode == 3;
    options.bounded_cache_neighborhood = mode == 3;
    StereoFusion fusion(options, root, "COLMAP", "", "geometric");
    ASSERT_NO_THROW(fusion.Run());
    EXPECT_TRUE(fusion.GetFusedPoints().empty());
    EXPECT_TRUE(fusion.GetFusedPointsVisibility().empty());
  }
}

TEST(StereoFusion, ChangedRosterAfterCancellationSelectsOnlyCurrentRoots) {
  const bool mapped_storage_supported = ImmutableMapRegion::Supported();
  RecordProperty("mapped_storage_supported",
                 mapped_storage_supported ? "true" : "false");
  const auto root = CreateTwoViewNormalDecisionFixture();
  Model model;
  model.Read(root, "COLMAP");
  ASSERT_EQ(model.images.size(), 2);
  const auto write_roster = [&](const std::vector<int>& image_indices) {
    std::ofstream config(root / "stereo/fusion.cfg");
    for (const int image_idx : image_indices) {
      config << model.GetImageName(image_idx) << '\n';
    }
    config.close();
    ASSERT_TRUE(config);
  };

  for (const std::string mode :
       {"eager", "lru", "bounded", "components", "mapped"}) {
    SCOPED_TRACE(mode);
    if (mode == "mapped" && !mapped_storage_supported) continue;
    StereoFusionOptions options;
    options.num_threads = 1;
    options.min_num_pixels = options.max_num_pixels = 1;
    options.cache_size = 1.0 / 1024;
    options.use_cache = mode == "lru";
    options.bounded_cache =
        mode == "bounded" || mode == "components" || mode == "mapped";
    options.bounded_cache_components = mode == "components" || mode == "mapped";
    options.bounded_cache_neighborhood = mode == "components";
    options.bounded_cache_normal_tile_size =
        mode == "components" && mapped_storage_supported ? 4 : 0;
    if ((mode == "components" || mode == "mapped") &&
        mapped_storage_supported) {
      options.bounded_cache_normal_mapped_bytes = 1 << 20;
      options.bounded_cache_normal_mapped_files = 2;
    }
    StereoFusion fusion(options, root, "COLMAP", "", "geometric");
    write_roster({0, 1});
    int stop_checks = 0;
    fusion.SetCheckIfStoppedFunc([&] { return ++stop_checks == 2; });
    fusion.Run();
    ASSERT_EQ(stop_checks, 2);
    ASSERT_TRUE(fusion.GetFusedPoints().empty());

    // Cancellation has initialized both per-image rasters without consuming
    // either root. Reusing that state must not admit a now-excluded image.
    write_roster({1});
    fusion.SetCheckIfStoppedFunc([] { return false; });
    ASSERT_NO_THROW(fusion.Run());
    EXPECT_EQ(fusion.GetFusedPoints().size(), 600);
    ASSERT_EQ(fusion.GetFusedPointsVisibility().size(), 600);
    for (const auto& visibility : fusion.GetFusedPointsVisibility()) {
      EXPECT_EQ(visibility, (std::vector<int>{1}));
    }

    write_roster({});
    ASSERT_NO_THROW(fusion.Run());
    EXPECT_TRUE(fusion.GetFusedPoints().empty());
    EXPECT_TRUE(fusion.GetFusedPointsVisibility().empty());

    write_roster({0, 1});
    ASSERT_NO_THROW(fusion.Run());
    EXPECT_EQ(fusion.GetFusedPoints().size(), 1200);
    ASSERT_EQ(fusion.GetFusedPointsVisibility().size(), 1200);
    std::vector<size_t> coverage(2, 0);
    for (const auto& visibility : fusion.GetFusedPointsVisibility()) {
      ASSERT_EQ(visibility.size(), 1);
      ASSERT_GE(visibility.front(), 0);
      ASSERT_LT(visibility.front(), 2);
      ++coverage[visibility.front()];
    }
    EXPECT_EQ(coverage, (std::vector<size_t>{600, 600}));
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

TEST(StereoFusion, MixedResolutionColorsFollowPixelCenters) {
  const auto root = CreateTestDir();
  for (const auto* dir : {"sparse",
                          "images",
                          "stereo",
                          "stereo/depth_maps",
                          "stereo/normal_maps"}) {
    CreateDirIfNotExists(root / dir);
  }
  colmap::Camera camera;
  camera.camera_id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 10;
  camera.height = 5;
  camera.params = {8.0, 11.0, 4.3, 2.2};
  Reconstruction reconstruction;
  reconstruction.AddCameraWithTrivialRig(camera);
  colmap::Image image;
  image.SetImageId(1);
  image.SetCameraId(1);
  image.SetName("view.png");
  reconstruction.AddImageWithTrivialFrame(image, Rigid3d());
  reconstruction.Write(root / "sparse");
  Bitmap bitmap(10, 5, true);
  for (int row = 0; row < 5; ++row) {
    for (int col = 0; col < 10; ++col) {
      bitmap.SetPixel(
          col, row, BitmapColor<uint8_t>(20 * col, 30 * row, col + row));
    }
  }
  ASSERT_TRUE(bitmap.Write(root / "images/view.png"));
  {
    std::ofstream config(root / "stereo/fusion.cfg");
    config << "view.png\n";
  }
  struct Grid {
    std::vector<int> source_columns;
    std::vector<int> source_rows;
  };
  // Hand-derived nearest source indices. The 7x3 raster has unequal scales;
  // 1x1 and 5x1 include exact halfway samples, including a non-binary ratio.
  // Keeping every grid cell in the expected set makes lost coverage fail too.
  const std::vector<Grid> grids = {
      {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 3, 4}},
      {{0, 2, 3, 5, 6, 7, 9}, {0, 2, 4}},
      {{5}, {2}},
      {{1, 3, 5, 7, 9}, {2}},
  };
  for (const auto& grid : grids) {
    const int width = grid.source_columns.size();
    const int height = grid.source_rows.size();
    SCOPED_TRACE(::testing::Message() << width << "x" << height);
    Mat<float> depth(width, height, 1);
    depth.Fill(5.0f);
    depth.Write(root / "stereo/depth_maps/view.png.geometric.bin");
    Mat<float> normal(width, height, 3);
    normal.Fill(0.0f);
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) normal.Set(row, col, 2, -1.0f);
    }
    normal.Write(root / "stereo/normal_maps/view.png.geometric.bin");
    for (const std::string mode : {"eager",
                                   "lru",
                                   "bounded",
                                   "components",
                                   "compact",
                                   "compact_neighborhood"}) {
      SCOPED_TRACE(mode);
      StereoFusionOptions options;
      options.num_threads = 1;
      options.min_num_pixels = 1;
      options.max_num_pixels = 1;
      options.use_cache = mode == "lru";
      options.bounded_cache = mode != "eager" && mode != "lru";
      options.bounded_cache_components =
          mode == "components" || mode == "compact_neighborhood";
      options.bounded_cache_compact_rgb =
          mode == "compact" || mode == "compact_neighborhood";
      options.bounded_cache_neighborhood = mode == "compact_neighborhood";
      options.cache_size = 1.0 / 1024;
      StereoFusion fusion(options, root, "COLMAP", "", "geometric");
      fusion.Run();
      const auto& points = fusion.GetFusedPoints();
      const auto& visibility = fusion.GetFusedPointsVisibility();
      ASSERT_EQ(points.size(), width * height);
      ASSERT_EQ(visibility.size(), points.size());
      std::vector<int> coverage(width * height, 0);
      for (size_t index = 0; index < points.size(); ++index) {
        const auto& point = points[index];
        ASSERT_TRUE(std::isfinite(point.x));
        ASSERT_TRUE(std::isfinite(point.y));
        ASSERT_TRUE(std::isfinite(point.z));
        ASSERT_GT(point.z, 0.0f);
        // Project published XYZ with the original camera, independent of
        // Fusion's float K and the integer rescaling helper.
        const int col =
            std::lround((8.0 * point.x / point.z + 4.3) * width / 10 - 0.5);
        const int row =
            std::lround((11.0 * point.y / point.z + 2.2) * height / 5 - 0.5);
        ASSERT_GE(col, 0);
        ASSERT_LT(col, width);
        ASSERT_GE(row, 0);
        ASSERT_LT(row, height);
        ++coverage[row * width + col];
        EXPECT_NEAR(point.x, 5 * ((col + 0.5) * 10 / width - 4.3) / 8, 2e-6);
        EXPECT_NEAR(point.y, 5 * ((row + 0.5) * 5 / height - 2.2) / 11, 2e-6);
        EXPECT_FLOAT_EQ(point.z, 5.0f);
        EXPECT_FLOAT_EQ(point.nx, 0.0f);
        EXPECT_FLOAT_EQ(point.ny, 0.0f);
        EXPECT_FLOAT_EQ(point.nz, -1.0f);
        const auto expected =
            bitmap.GetPixel(grid.source_columns[col], grid.source_rows[row]);
        ASSERT_TRUE(expected);
        EXPECT_EQ(point.r, expected->r);
        EXPECT_EQ(point.g, expected->g);
        EXPECT_EQ(point.b, expected->b);
        EXPECT_EQ(visibility[index], std::vector<int>{0});
      }
      EXPECT_EQ(coverage, std::vector<int>(width * height, 1));
    }
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

}  // namespace
}  // namespace mvs
}  // namespace colmap
