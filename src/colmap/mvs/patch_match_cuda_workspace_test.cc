// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_cuda_workspace.h"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace internal {
namespace {

TEST(PatchMatchLookAheadPlan, RejectsIneligibleAndUnrepresentableWork) {
  const size_t max_size = std::numeric_limits<size_t>::max();
  for (const bool geometry : {false, true}) {
    for (const auto& input : std::vector<std::array<size_t, 2>>{
             {0, 1}, {1, 0}, {1, 33}, {max_size, 1}, {1, max_size}}) {
      const auto plan =
          PatchMatchLookAheadPlan::Make(input[0], input[1], 33, geometry);
      EXPECT_EQ(plan.rows, 0);
      EXPECT_EQ(plan.num_bytes, 0);
    }
    for (const int samples : {std::numeric_limits<int>::min(), -1, 0, 30}) {
      EXPECT_EQ(PatchMatchLookAheadPlan::Make(35, 31, samples, geometry).rows,
                0);
    }
    EXPECT_EQ(PatchMatchLookAheadPlan::Make(
                  1, 1, std::numeric_limits<int>::max(), geometry)
                  .rows,
              0);
    EXPECT_GT(PatchMatchLookAheadPlan::Make(35, 31, 31, geometry).rows, 0);
    EXPECT_GT(PatchMatchLookAheadPlan::Make(35, 32, 33, geometry).rows, 0);
  }
}

TEST(PatchMatchLookAheadPlan, RetainsCurrentDispatchAndBandSizes) {
  struct Example {
    size_t width;
    size_t sources;
    int samples;
    bool geometry;
    int rows;
    size_t bytes;
  };
  for (const Example& example : {
           Example{1536, 10, 15, false, 128, 38559744},
           Example{1536, 10, 15, true, 128, 54288384},
           Example{3072, 10, 15, false, 111, 66883584},
           Example{3072, 10, 15, true, 79, 67031040},
           Example{3072, 32, 33, false, 41, 67055616},
           Example{3072, 32, 33, true, 27, 65409024},
       }) {
    const auto plan = PatchMatchLookAheadPlan::Make(
        example.width, example.sources, example.samples, example.geometry);
    EXPECT_EQ(plan.rows, example.rows);
    EXPECT_EQ(plan.num_bytes, example.bytes);
  }
}

TEST(PatchMatchLookAheadPlan, ChoosesLargestBandWithinBudget) {
  for (const size_t width : {1, 8, 35, 1536, 3072, 65537, 1048576, 4194305}) {
    for (const size_t sources : {1, 10, 16, 17, 31, 32}) {
      for (const int samples : {1, 15, 17, 33, 257}) {
        for (const bool geometry : {false, true}) {
          const auto plan =
              PatchMatchLookAheadPlan::Make(width, sources, samples, geometry);
          int expected_rows = 0;
          if (sources <= static_cast<size_t>(samples)) {
            // Independently enumerate each candidate size with separate plane
            // charges, including the carry that remains live between bands.
            for (int rows = 1; rows <= 128; ++rows) {
              const uint64_t pixels = uint64_t{width} * rows;
              const uint64_t bytes =
                  sizeof(float) *
                  (4 * pixels + samples * pixels + 3 * sources * pixels +
                   (geometry ? 2 * sources * pixels : 0) + 4 * width);
              if (bytes <= PatchMatchLookAheadPlan::kMaxBytes) {
                expected_rows = rows;
              }
            }
          }
          EXPECT_EQ(plan.rows, expected_rows)
              << width << ", " << sources << ", " << samples << ", "
              << geometry;
          EXPECT_LE(plan.num_bytes, PatchMatchLookAheadPlan::kMaxBytes);
        }
      }
    }
  }
}

TEST(PatchMatchLookAheadPlan, OneRowBoundaryIncludesPersistentCarry) {
  constexpr size_t kFloats = PatchMatchLookAheadPlan::kMaxBytes / sizeof(float);
  // PHOTO, S=P=1: four random, one uniform, three cost planes and four
  // persistent carry values per column. The next width cannot fit one row.
  const auto fits = PatchMatchLookAheadPlan::Make(kFloats / 12, 1, 1, false);
  ASSERT_EQ(fits.rows, 1);
  EXPECT_EQ(fits.num_bytes, (kFloats / 12) * 12 * sizeof(float));
  const auto refused =
      PatchMatchLookAheadPlan::Make(kFloats / 12 + 1, 1, 1, false);
  EXPECT_EQ(refused.rows, 0);
  EXPECT_EQ(refused.num_bytes, 0);
}

TEST(PatchMatchLookAheadPlan, EveryPlaneHasExclusiveStorage) {
  for (const size_t width : {1, 8, 35}) {
    for (const size_t sources : {1, 17, 32}) {
      for (const bool geometry : {false, true}) {
        const auto plan =
            PatchMatchLookAheadPlan::Make(width, sources, 33, geometry);
        ASSERT_GT(plan.rows, 0);
        std::vector<uint8_t> written(plan.num_bytes / sizeof(float), 0);
        const auto write_planes =
            [&](size_t first, size_t rows, size_t planes) {
              for (size_t plane = 0; plane < planes; ++plane) {
                for (size_t row = 0; row < rows; ++row) {
                  for (size_t col = 0; col < width; ++col) {
                    const size_t offset =
                        first + (plane * rows + row) * width + col;
                    ASSERT_LT(offset, written.size());
                    ASSERT_EQ(written[offset], 0)
                        << "Overlapping offset " << offset;
                    written[offset] = 1;
                  }
                }
              }
            };
        write_planes(0, plan.rows, 4);
        write_planes(plan.uniforms_offset, plan.rows, 33);
        write_planes(plan.costs_offset, plan.rows, 3 * sources);
        write_planes(
            plan.geometry_offset, plan.rows, geometry ? 2 * sources : 0);
        write_planes(plan.previous_offset, 1, 4);
        // No hidden padding or unaccounted allocation remains.
        for (const uint8_t count : written) ASSERT_EQ(count, 1);
      }
    }
  }
}

}  // namespace
}  // namespace internal
}  // namespace mvs
}  // namespace colmap
