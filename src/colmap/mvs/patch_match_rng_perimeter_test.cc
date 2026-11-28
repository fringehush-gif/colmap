// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_rng_perimeter.h"

#include <numeric>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {

TEST(PatchMatchRngPerimeter, CoordinateAndSweepRoundTrip) {
  for (const auto [width, height] : {std::pair<size_t, size_t>{1, 1},
                                     {1, 5},
                                     {5, 1},
                                     {2, 2},
                                     {3, 5},
                                     {5, 3}}) {
    SCOPED_TRACE(::testing::Message() << width << "x" << height);
    const PatchMatchRngPerimeter layout{width, height};
    std::vector<bool> seen(layout.Count(), false);
    size_t count = 0;
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        if (row != 0 && col != 0 && row + 1 != height && col + 1 != width) {
          continue;
        }
        const size_t index = layout.Index(row, col);
        ASSERT_LT(index, seen.size());
        EXPECT_FALSE(seen[index]);
        seen[index] = true;
        ++count;
        size_t decoded_row, decoded_col;
        layout.Coordinates(index, &decoded_row, &decoded_col);
        EXPECT_EQ(decoded_row, row);
        EXPECT_EQ(decoded_col, col);
      }
    }
    EXPECT_EQ(count, layout.Count());

    std::vector<size_t> pixels(width * height);
    std::iota(pixels.begin(), pixels.end(), 0);
    size_t current_width = width;
    size_t current_height = height;
    for (unsigned rotation = 0; rotation < 4; ++rotation) {
      for (size_t col = 0; col < current_width; ++col) {
        const size_t index = layout.SweepIndex(rotation, col);
        ASSERT_LT(index, layout.Count());
        size_t row, original_col;
        layout.Coordinates(index, &row, &original_col);
        EXPECT_EQ(row * width + original_col, pixels[col]);
      }
      std::vector<size_t> rotated(pixels.size());
      for (size_t row = 0; row < current_height; ++row) {
        for (size_t col = 0; col < current_width; ++col) {
          rotated[(current_width - 1 - col) * current_height + row] =
              pixels[row * current_width + col];
        }
      }
      pixels.swap(rotated);
      std::swap(current_width, current_height);
    }
  }
}

}  // namespace mvs
}  // namespace colmap
