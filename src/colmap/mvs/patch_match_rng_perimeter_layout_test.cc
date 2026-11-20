// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_rng_perimeter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(PatchMatchRngPerimeterLayout, EveryPixelMatchesPhysicalRotations) {
  const std::array<std::array<size_t, 2>, 7> shapes{
      {{1, 1}, {1, 259}, {259, 1}, {35, 259}, {259, 35}, {17, 33}, {33, 17}}};
  size_t checked_pixels = 0;
  for (const auto& shape : shapes) {
    const size_t original_width = shape[0];
    const size_t original_height = shape[1];
    const PatchMatchRngPerimeter layout{original_width, original_height};
    std::vector<size_t> original(original_width * original_height);
    std::iota(original.begin(), original.end(), size_t{0});
    auto physical = original;
    size_t width = original_width, height = original_height;
    for (unsigned int rotation = 0; rotation < 4; ++rotation) {
      SCOPED_TRACE(::testing::Message()
                   << original_width << "x" << original_height
                   << " rotation=" << rotation);
      for (size_t first_row = 0; first_row < height; first_row += 128) {
        for (size_t local_row = 0;
             local_row < std::min<size_t>(128, height - first_row);
             ++local_row) {
          const size_t row = first_row + local_row;
          for (size_t col = 0; col < width; ++col) {
            size_t original_row, original_col;
            layout.OriginalCoordinates(
                rotation, row, col, &original_row, &original_col);
            ASSERT_LT(original_row, original_height);
            ASSERT_LT(original_col, original_width);
            EXPECT_EQ(original[original_row * original_width + original_col],
                      physical[row * width + col]);
            if (row == 0) {
              EXPECT_EQ(layout.Index(original_row, original_col),
                        layout.SweepIndex(rotation, col));
            }
            ++checked_pixels;
          }
        }
      }
      // Independent oracle: copy each tagged source cell into its CCW
      // destination. It does not call the production coordinate helper.
      std::vector<size_t> rotated(physical.size());
      for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
          rotated[(width - 1 - col) * height + row] =
              physical[row * width + col];
        }
      }
      physical.swap(rotated);
      std::swap(width, height);
    }
    EXPECT_EQ(physical, original);
  }
  EXPECT_EQ(checked_pixels, 79084u);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
