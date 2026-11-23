// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace colmap {
namespace mvs {
namespace internal {

// Keep ordinary float rounding, then enforce integer downsize bounds.
inline std::pair<size_t, size_t> ComputeDownsizeDimensions(
    const size_t width,
    const size_t height,
    const size_t max_width,
    const size_t max_height) {
  const float factor = std::min(static_cast<float>(max_width) / width,
                                static_cast<float>(max_height) / height);
  const auto compute_dimension = [factor](const size_t dimension,
                                          const size_t max_dimension) {
    // Retain the existing one-pixel floor when a maximum is zero.
    const size_t limit =
        std::max(size_t{1}, std::min(dimension, max_dimension));
    const float rounded = std::max(1.0f, std::round(dimension * factor));
    // A large size_t can round up to 2^digits as a float. Bound it before
    // conversion, using the exact power of two rather than a rounded limit.
    if (rounded >= std::ldexp(1.0f, std::numeric_limits<size_t>::digits)) {
      return limit;
    }
    return std::min(limit, static_cast<size_t>(rounded));
  };
  return {compute_dimension(width, max_width),
          compute_dimension(height, max_height)};
}

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
