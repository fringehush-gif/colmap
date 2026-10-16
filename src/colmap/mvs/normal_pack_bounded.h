// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/mat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <Eigen/Core>
#if defined(__FAST_MATH__)
#error Bounded rounding requires unchanged IEEE arithmetic
#endif
namespace colmap::mvs::normal_pack::bounded {
// Precondition: finite codec intermediate, abs(x) < 65536, supported FP state.
// Keep the fraction comparison separate: adding 0.5 first can round across a
// tie.
inline double RoundNearestTiesAwayBounded(const double x) {
  const double a = std::abs(x);
  const double f = std::floor(a);
  const double r = f + static_cast<double>(a - f >= 0.5);
  return std::copysign(r, x);
}
// Encode validated finite [-1, 1] components with the signed-normal codec,
// without clipping or vector renormalization.
inline int16_t EncodeNormalSnorm16Unchecked(const float value) {
  return static_cast<int16_t>(
      RoundNearestTiesAwayBounded(static_cast<double>(value) * 32767.0));
}

inline void EncodeNormalOct2x16Unchecked(const Eigen::Vector3f& normal,
                                         OctNormalPair* encoded) {
  const double nx = normal.x(), ny = normal.y(), nz = normal.z();
  const double l1 = std::abs(nx) + std::abs(ny) + std::abs(nz);
  if (l1 == 0.0) {
    encoded->x = encoded->y = std::numeric_limits<int16_t>::min();
    return;
  }
  double u = nx / l1, v = ny / l1;
  if (nz < 0.0) {
    const double old_u = u;
    u = (1.0 - std::abs(v)) * (u >= 0.0 ? 1.0 : -1.0);
    v = (1.0 - std::abs(old_u)) * (v >= 0.0 ? 1.0 : -1.0);
  }
  encoded->x = static_cast<int16_t>(RoundNearestTiesAwayBounded(u * 32767.0));
  encoded->y = static_cast<int16_t>(RoundNearestTiesAwayBounded(v * 32767.0));
}

inline uint16_t EncodeNormalMagnitudeUnorm16Unchecked(
    const Eigen::Vector3f& normal) {
  constexpr double kMaximum = 1.7320508075688772935;  // sqrt(3)
  const double x = normal.x(), y = normal.y(), z = normal.z();
  return static_cast<uint16_t>(RoundNearestTiesAwayBounded(
      std::sqrt(x * x + y * y + z * z) * 65535.0 / kMaximum));
}

}  // namespace colmap::mvs::normal_pack::bounded
