// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

namespace colmap {
namespace mvs {
namespace unorm16 {

#if defined(__CUDACC__) || defined(__HIPCC__)
// Caller admits a finite nonnegative value whose exact scaled value is at
// most 65535. Correct the half-integer cases where rounding the product first
// would change the subsequent nearest-even integer rounding.
__device__ inline uint16_t RoundScaled(float value, float scale) {
  const float scaled = __fmul_rn(value, scale);
  unsigned rounded = __float2uint_rn(scaled);
  const float delta = __fsub_rn(scaled, static_cast<float>(rounded));
  if (delta == 0.5f || delta == -0.5f) {
    const float residual = __fmaf_rn(value, scale, -scaled);
    if (residual > 0.0f && delta > 0.0f) {
      ++rounded;
    } else if (residual < 0.0f && delta < 0.0f) {
      --rounded;
    }
  }
  return static_cast<uint16_t>(rounded);
}
#endif

}  // namespace unorm16
}  // namespace mvs
}  // namespace colmap
