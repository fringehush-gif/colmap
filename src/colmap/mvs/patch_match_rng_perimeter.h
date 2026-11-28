// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLMAP_RNG_HD __host__ __device__
#else
#define COLMAP_RNG_HD
#endif

namespace colmap {
namespace mvs {

// Canonical coordinates of the original image, with positive dimensions.
// Adjacent sweep orientations share the same state at each corner.
struct PatchMatchRngPerimeter {
  const size_t width;
  const size_t height;

  COLMAP_RNG_HD constexpr size_t Count() const {
    if (width == 1) return height;
    if (height == 1) return width;
    return 2 * (width - 1) + 2 * (height - 1);
  }

  // The pixel must lie on the perimeter.
  COLMAP_RNG_HD constexpr size_t Index(size_t row, size_t col) const {
    if (width == 1) return row;
    if (height == 1 || row == 0) return col;
    if (col == width - 1) return width + row - 1;
    if (row == height - 1) return 2 * width + height - 3 - col;
    return 2 * width + 2 * height - 4 - row;
  }

  COLMAP_RNG_HD constexpr void Coordinates(size_t index,
                                           size_t* row,
                                           size_t* col) const {
    if (width == 1) {
      *row = index;
      *col = 0;
    } else if (height == 1 || index < width) {
      *row = 0;
      *col = index;
    } else if (index < width + height - 1) {
      *row = index - width + 1;
      *col = width - 1;
    } else if (index < 2 * width + height - 2) {
      *row = height - 1;
      *col = 2 * width + height - 3 - index;
    } else {
      *row = 2 * width + 2 * height - 4 - index;
      *col = 0;
    }
  }

  // Row zero of the current orientation, after 0..3 counter-clockwise turns.
  COLMAP_RNG_HD constexpr size_t SweepIndex(unsigned rotation,
                                            size_t col) const {
    switch (rotation) {
      case 0:
        return Index(0, col);
      case 1:
        return Index(col, width - 1);
      case 2:
        return Index(height - 1, width - 1 - col);
      default:
        return Index(height - 1 - col, 0);
    }
  }
};

}  // namespace mvs
}  // namespace colmap

#undef COLMAP_RNG_HD
