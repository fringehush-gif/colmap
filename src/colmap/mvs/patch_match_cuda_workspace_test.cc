// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_cuda_workspace.h"

#include <algorithm>
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
    for (const auto& input :
         std::vector<std::array<size_t, 3>>{{0, 1, 1},
                                            {1, 0, 1},
                                            {1, 1, 0},
                                            {1, 1, 33},
                                            {max_size, 1, 1},
                                            {1, max_size, 1},
                                            {1, 1, max_size}}) {
      const auto plan = PatchMatchLookAheadPlan::Make(
          input[0], input[1], input[2], 33, geometry);
      EXPECT_EQ(plan.rows, 0);
      EXPECT_EQ(plan.num_bytes, 0);
      for (const auto& layout : plan.layouts) {
        EXPECT_EQ(layout.width, 0);
        EXPECT_EQ(layout.plane_rows, 0);
        EXPECT_EQ(layout.num_bytes, 0);
      }
    }
    for (const int samples : {std::numeric_limits<int>::min(), -1, 0, 30}) {
      EXPECT_EQ(
          PatchMatchLookAheadPlan::Make(35, 259, 31, samples, geometry).rows,
          0);
    }
    EXPECT_EQ(PatchMatchLookAheadPlan::Make(
                  1, 1, 1, std::numeric_limits<int>::max(), geometry)
                  .rows,
              0);
    EXPECT_GT(PatchMatchLookAheadPlan::Make(35, 259, 31, 31, geometry).rows, 0);
    EXPECT_GT(PatchMatchLookAheadPlan::Make(35, 259, 32, 33, geometry).rows, 0);
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
    const auto plan = PatchMatchLookAheadPlan::Make(example.width,
                                                    example.width,
                                                    example.sources,
                                                    example.samples,
                                                    example.geometry);
    EXPECT_EQ(plan.rows, example.rows);
    EXPECT_EQ(plan.num_bytes, example.bytes);
    for (const auto& layout : plan.layouts) {
      EXPECT_EQ(layout.width, example.width);
      EXPECT_EQ(layout.plane_rows, example.rows);
      EXPECT_EQ(layout.num_bytes, example.bytes);
    }
  }
}

TEST(PatchMatchLookAheadPlan, ChoosesLargestBandWithinBudget) {
  for (const size_t width : {1, 8, 35, 1536, 3072, 65537, 1048576, 4194305}) {
    for (const size_t height : {size_t{1}, width}) {
      for (const size_t sources : {1, 10, 16, 17, 31, 32}) {
        for (const int samples : {1, 15, 17, 33, 257}) {
          for (const bool geometry : {false, true}) {
            const auto plan = PatchMatchLookAheadPlan::Make(
                width, height, sources, samples, geometry);
            int expected_rows = 0;
            uint64_t legacy_bytes = 0;
            if (sources <= static_cast<size_t>(samples)) {
              // Independently enumerate the original maximum-dimension band
              // schedule. A smaller live allocation must not change it.
              for (int rows = 1; rows <= 128; ++rows) {
                const uint64_t pixels = uint64_t{width} * rows;
                const uint64_t bytes =
                    sizeof(float) *
                    (4 * pixels + samples * pixels + 3 * sources * pixels +
                     (geometry ? 2 * sources * pixels : 0) + 4 * width);
                if (bytes <= PatchMatchLookAheadPlan::kMaxBytes) {
                  expected_rows = rows;
                  legacy_bytes = bytes;
                }
              }
            }
            EXPECT_EQ(plan.rows, expected_rows)
                << width << ", " << height << ", " << sources << ", " << samples
                << ", " << geometry;
            EXPECT_LE(plan.num_bytes, legacy_bytes);
            EXPECT_LE(plan.num_bytes, PatchMatchLookAheadPlan::kMaxBytes);
          }
        }
      }
    }
  }
}

TEST(PatchMatchLookAheadPlan, OneRowBoundaryIncludesPersistentCarry) {
  constexpr size_t kFloats = PatchMatchLookAheadPlan::kMaxBytes / sizeof(float);
  // PHOTO, S=P=1: four random, one uniform, three cost planes and four
  // persistent carry values per column. The next width cannot fit one row.
  const auto fits = PatchMatchLookAheadPlan::Make(kFloats / 12, 1, 1, 1, false);
  ASSERT_EQ(fits.rows, 1);
  EXPECT_EQ(fits.num_bytes, (kFloats / 12) * 12 * sizeof(float));
  const auto refused =
      PatchMatchLookAheadPlan::Make(kFloats / 12 + 1, 1, 1, 1, false);
  EXPECT_EQ(refused.rows, 0);
  EXPECT_EQ(refused.num_bytes, 0);
}

TEST(PatchMatchLookAheadPlan, ThinImagesShrinkWithoutChangingBandSchedule) {
  struct Example {
    size_t width;
    size_t height;
    bool geometry;
    int rows;
    size_t bytes;
    std::array<size_t, 2> pitches;
    std::array<size_t, 2> plane_rows;
    std::array<int, 2> band_counts;
  };
  for (const Example& example : {
           Example{3072, 1, false, 41, 1683456, {3072, 32}, {1, 41}, {1, 75}},
           Example{1, 3072, false, 41, 1683456, {32, 3072}, {41, 1}, {75, 1}},
           Example{3072, 1, true, 27, 2469888, {3072, 32}, {1, 27}, {1, 114}},
           Example{1, 3072, true, 27, 2469888, {32, 3072}, {27, 1}, {114, 1}},
           Example{17, 35, false, 128, 596352, {32, 64}, {35, 17}, {1, 1}},
           Example{35, 17, true, 128, 883072, {64, 32}, {17, 35}, {1, 1}},
       }) {
    SCOPED_TRACE(::testing::Message() << example.width << "x" << example.height
                                      << ", geometry=" << example.geometry);
    const auto plan = PatchMatchLookAheadPlan::Make(
        example.width, example.height, 32, 33, example.geometry);
    ASSERT_EQ(plan.rows, example.rows);
    EXPECT_EQ(plan.num_bytes, example.bytes);
    EXPECT_EQ(plan.sources, 32);
    EXPECT_EQ(plan.samples, 33);
    EXPECT_EQ(plan.geometry, example.geometry);
    for (int rotation = 0; rotation < 4; ++rotation) {
      const int orientation = rotation % 2;
      const auto& layout = plan.layouts[orientation];
      EXPECT_EQ(layout.width, example.pitches[orientation]);
      EXPECT_EQ(layout.plane_rows, example.plane_rows[orientation]);
      const size_t height = orientation ? example.width : example.height;
      EXPECT_EQ((height + plan.rows - 1) / plan.rows,
                example.band_counts[orientation]);
    }
  }
}

TEST(PatchMatchLookAheadPlan, KeepsLegacyLayoutWhenPaddingErasesSaving) {
  // The short dimension is below B=128, but both padded 128x127 bands need
  // 16 more bytes than the legacy 127x128 band because of persistent carry.
  const auto plan = PatchMatchLookAheadPlan::Make(127, 127, 32, 33, false);
  ASSERT_EQ(plan.rows, 128);
  EXPECT_EQ(plan.num_bytes, 8650224);
  for (const auto& layout : plan.layouts) {
    EXPECT_EQ(layout.width, 127);
    EXPECT_EQ(layout.plane_rows, 128);
    EXPECT_EQ(layout.num_bytes, 8650224);
  }
}

TEST(PatchMatchLookAheadPlan, EveryPlaneHasExclusiveStorage) {
  struct Shape {
    size_t width;
    size_t height;
    std::array<size_t, 2> pitches;
    std::array<size_t, 2> plane_rows;
  };
  for (const Shape& shape : {
           Shape{1, 8, {32, 32}, {8, 1}},
           Shape{17, 35, {32, 64}, {35, 17}},
           Shape{35, 259, {64, 288}, {128, 35}},
           Shape{127, 127, {127, 127}, {128, 128}},
       }) {
    for (const size_t sources : {1, 17, 32}) {
      for (const bool geometry : {false, true}) {
        const auto plan = PatchMatchLookAheadPlan::Make(
            shape.width, shape.height, sources, 33, geometry);
        ASSERT_EQ(plan.rows, 128);
        size_t largest_layout = 0;
        for (int orientation = 0; orientation < 2; ++orientation) {
          SCOPED_TRACE(::testing::Message()
                       << shape.width << "x" << shape.height
                       << ", sources=" << sources << ", geometry=" << geometry
                       << ", orientation=" << orientation);
          const auto& layout = plan.layouts[orientation];
          const size_t pitch = shape.pitches[orientation];
          const size_t rows = shape.plane_rows[orientation];
          ASSERT_EQ(layout.width, pitch);
          ASSERT_EQ(layout.plane_rows, rows);
          const size_t pixels = pitch * rows;
          const size_t expected_floats =
              4 * pixels + 33 * pixels + 3 * sources * pixels +
              (geometry ? 2 * sources * pixels : 0) + 4 * pitch;
          ASSERT_EQ(layout.num_bytes, expected_floats * sizeof(float));
          ASSERT_LE(layout.num_bytes, plan.num_bytes);
          largest_layout = std::max(largest_layout, layout.num_bytes);
          std::vector<uint8_t> written(plan.num_bytes / sizeof(float), 0);
          const auto write_planes =
              [&](size_t first, size_t rows, size_t planes) {
                for (size_t plane = 0; plane < planes; ++plane) {
                  for (size_t row = 0; row < rows; ++row) {
                    for (size_t col = 0; col < pitch; ++col) {
                      const size_t offset =
                          first + (plane * rows + row) * pitch + col;
                      ASSERT_LT(offset, written.size());
                      ASSERT_EQ(written[offset], 0)
                          << "Overlapping offset " << offset;
                      written[offset] = 1;
                    }
                  }
                }
              };
          write_planes(0, rows, 4);
          write_planes(layout.uniforms_offset, rows, 33);
          write_planes(layout.costs_offset, rows, 3 * sources);
          write_planes(
              layout.geometry_offset, rows, geometry ? 2 * sources : 0);
          write_planes(layout.previous_offset, 1, 4);
          // The selected layout is exactly accounted for, including row
          // padding; unused capacity from the other orientation stays
          // untouched.
          for (size_t i = 0; i < written.size(); ++i) {
            ASSERT_EQ(written[i], i < expected_floats ? 1 : 0) << i;
          }
        }
        EXPECT_EQ(plan.num_bytes, largest_layout);
      }
    }
  }
}

TEST(PatchMatchLookAheadPlan, CarriesRemainLiveAcrossTailsAndRotations) {
  // This fixture has two full bands followed by a three-row tail in rotations
  // 0 and 2. The other rotations use a different pitch and one short band.
  // Mark real layout addresses, without relying on a copy of the CUDA helper.
  for (const bool geometry : {false, true}) {
    const auto plan = PatchMatchLookAheadPlan::Make(35, 259, 17, 17, geometry);
    ASSERT_EQ(plan.rows, 128);
    std::vector<uint32_t> storage(plan.num_bytes / sizeof(float), 0);
    int observed_tails = 0;
    for (int rotation = 0; rotation < 4; ++rotation) {
      const auto& layout = plan.layouts[rotation % 2];
      const size_t width = rotation % 2 ? 259 : 35;
      const size_t height = rotation % 2 ? 35 : 259;
      const size_t fixed_rows = std::min<size_t>(128, height);
      ASSERT_EQ(layout.plane_rows, fixed_rows);
      const size_t pitch = rotation % 2 ? 288 : 64;
      ASSERT_EQ(layout.width, pitch);
      const size_t carry_offset =
          (4 + 17 + 3 * 17 + (geometry ? 2 * 17 : 0)) * pitch * fixed_rows;
      ASSERT_EQ(layout.previous_offset, carry_offset);
      const auto tag = [&](size_t boundary, size_t plane, size_t col) {
        return static_cast<uint32_t>(0x40000000 + (rotation + 1) * 1000000 +
                                     boundary * 1000 + plane * 300 + col);
      };
      for (size_t plane = 0; plane < 4; ++plane) {
        for (size_t col = 0; col < width; ++col) {
          storage[carry_offset + plane * pitch + col] = tag(0, plane, col);
        }
      }
      for (size_t first_row = 0; first_row < height; first_row += plan.rows) {
        const size_t active_rows =
            std::min<size_t>(plan.rows, height - first_row);
        const auto write_active = [&](size_t offset, size_t planes) {
          for (size_t plane = 0; plane < planes; ++plane) {
            for (size_t row = 0; row < active_rows; ++row) {
              for (size_t col = 0; col < width; ++col) {
                const size_t index =
                    offset + (plane * fixed_rows + row) * pitch + col;
                ASSERT_LT(index, carry_offset);
                storage[index] = 1;
              }
            }
          }
        };
        write_active(0, 4);
        write_active(layout.uniforms_offset, 17);
        write_active(layout.costs_offset, 3 * 17);
        write_active(layout.geometry_offset, geometry ? 2 * 17 : 0);
        for (size_t plane = 0; plane < 4; ++plane) {
          for (size_t col = 0; col < width; ++col) {
            const size_t index = layout.previous_offset + plane * pitch + col;
            EXPECT_EQ(storage[index], tag(first_row, plane, col));
            storage[index] = tag(first_row + active_rows, plane, col);
          }
        }
        if (active_rows < fixed_rows) {
          ++observed_tails;
          // Repacking the carry after active rows would silently read band
          // data instead of the previous winner. Establish this fixture is
          // sensitive to that mistake, rather than merely checking capacity.
          const size_t wrong_carry = carry_offset / fixed_rows * active_rows;
          ASSERT_NE(wrong_carry, carry_offset);
          EXPECT_NE(storage[wrong_carry], tag(first_row + active_rows, 0, 0));
        }
      }
    }
    EXPECT_EQ(observed_tails, 2);
  }
}

}  // namespace
}  // namespace internal
}  // namespace mvs
}  // namespace colmap
