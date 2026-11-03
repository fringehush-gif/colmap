// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace oct_revisit {

// Direction only. A consumer must explicitly own its magnitude policy.
struct alignas(4) Code {
  int16_t x, y;
};
static_assert(sizeof(Code) == 4);
constexpr int kScale = 32767;
constexpr int16_t kInvalid = -32768;

__device__ inline float3 RawDecode(Code c) {
  float x = static_cast<float>(c.x) / kScale;
  float y = static_cast<float>(c.y) / kScale;
  const float z = 1.0f - fabsf(x) - fabsf(y);
  if (z < 0) {
    const float old_x = x;
    x = copysignf(1.0f - fabsf(y), old_x);
    y = copysignf(1.0f - fabsf(old_x), y);
  }
  return make_float3(x, y, z);
}

__device__ inline float3 Decode(Code c) {
  if (c.x == kInvalid || c.y == kInvalid) return make_float3(0, 0, 0);
  float3 n = RawDecode(c);
  const float inv_length = rsqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
  return make_float3(n.x * inv_length, n.y * inv_length, n.z * inv_length);
}

__device__ inline Code QuarterTurn(Code c) {
  if (c.x == kInvalid || c.y == kInvalid) return {kInvalid, kInvalid};
  return {c.y, static_cast<int16_t>(-c.x)};
}

// Compare angular error through cross products, avoiding near-one FP32 dots.
// This does four cheap corner evaluations; the read-side decoder is unchanged.
__device__ inline bool Encode(float3 input, Code& out) {
  out = {kInvalid, kInvalid};
  if (!isfinite(input.x) || !isfinite(input.y) || !isfinite(input.z))
    return false;
  const float l1 = fabsf(input.x) + fabsf(input.y) + fabsf(input.z);
  if (l1 == 0) {
    // FTZ must not silently turn a nonzero subnormal vector into missing data.
    return ((__float_as_uint(input.x) | __float_as_uint(input.y) |
             __float_as_uint(input.z)) &
            0x7fffffffu) == 0;
  }
  if (!isfinite(l1)) return false;
  const float3 n = make_float3(input.x / l1, input.y / l1, input.z / l1);
  float x = n.x, y = n.y;
  if (n.z < 0) {
    x = copysignf(1.0f - fabsf(n.y), n.x);
    y = copysignf(1.0f - fabsf(n.x), n.y);
  }
  const float qx = fminf(float(kScale), fmaxf(float(-kScale), x * kScale));
  const float qy = fminf(float(kScale), fmaxf(float(-kScale), y * kScale));
  const int lo_x = static_cast<int>(floorf(qx)),
            hi_x = static_cast<int>(ceilf(qx));
  const int lo_y = static_cast<int>(floorf(qy)),
            hi_y = static_cast<int>(ceilf(qy));
  float best_error = 0, best_length = 1;
  bool found = false;
  for (int i = 0; i < 4; ++i) {
    const Code c{static_cast<int16_t>((i & 1) ? hi_x : lo_x),
                 static_cast<int16_t>((i & 2) ? hi_y : lo_y)};
    const float3 v = RawDecode(c);
    const float dot = n.x * v.x + n.y * v.y + n.z * v.z;
    if (!(dot > 0)) continue;
    const float a = n.y * v.z - n.z * v.y;
    const float b = n.z * v.x - n.x * v.z;
    const float d = n.x * v.y - n.y * v.x;
    const float error = a * a + b * b + d * d;
    const float length = v.x * v.x + v.y * v.y + v.z * v.z;
    if (!found || error * best_length < best_error * length) {
      found = true;
      out = c;
      best_error = error;
      best_length = length;
    }
  }
  return found;
}
}  // namespace oct_revisit
