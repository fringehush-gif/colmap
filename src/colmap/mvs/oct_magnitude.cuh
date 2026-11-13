// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <math_constants.h>
#include "oct_precise.cuh"

namespace oct_magnitude {

constexpr float kStep = 0x1p-24f;
constexpr float kEnvelope = 127.0f * kStep;
constexpr float kEstimatorBudget = 0x1p-38f;
constexpr float kDefaultNormErrorLimit = kStep;
constexpr uint8_t kInvalidMagnitude = 255;

// Logical payload is four direction bytes plus one magnitude byte. This LOCAL
// value has C++ padding: never memcpy sizeof(Encoded) to a five-byte surface.
// Surface owners must store direction and magnitude in separate typed planes.
struct Encoded {
  oct_revisit::Code direction;
  uint8_t magnitude;
};

enum class Status : uint8_t {
  kOk,
  kZero,
  kNonfinite,
  kOutOfEnvelope,
  kNoCertifiedCandidate
};

__device__ inline Encoded Invalid() {
  return {{oct_revisit::kInvalid, oct_revisit::kInvalid}, kInvalidMagnitude};
}

__device__ inline float AddError(float a, float b, float sum) {
  const float bv = __fsub_rn(sum, a);
  const float av = __fsub_rn(sum, bv);
  return __fadd_rn(__fsub_rn(a, av), __fsub_rn(b, bv));
}

// All arithmetic which preserves cancellation is explicitly RN, including
// FMA product residuals. Surrounding --use_fast_math must not reassociate it.
// Caller establishes finite near-unit inputs. No FP64 or square root here.
__device__ inline float ResidualUnchecked(float3 n) {
  const float xx = __fmul_rn(n.x, n.x);
  const float yy = __fmul_rn(n.y, n.y);
  const float zz = __fmul_rn(n.z, n.z);
  const float xy = __fadd_rn(xx, yy);
  const float sum = __fadd_rn(xy, zz);
  float tail = __fadd_rn(__fmaf_rn(n.x, n.x, -xx), __fmaf_rn(n.y, n.y, -yy));
  tail = __fadd_rn(tail, __fmaf_rn(n.z, n.z, -zz));
  tail = __fadd_rn(tail, AddError(xx, yy, xy));
  tail = __fadd_rn(tail, AddError(xy, zz, sum));
  const float delta = __fadd_rn(__fsub_rn(sum, 1.0f), tail);
  return __fmaf_rn(__fmul_rn(-0.125f, delta), delta, __fmul_rn(0.5f, delta));
}

__device__ inline bool NearUnit(float3 n) {
  if (!isfinite(n.x) || !isfinite(n.y) || !isfinite(n.z)) return false;
  const float sum = __fadd_rn(
      __fadd_rn(__fmul_rn(n.x, n.x), __fmul_rn(n.y, n.y)), __fmul_rn(n.z, n.z));
  // This prefilter is deliberately wider than the admitted envelope, but
  // bounded so the compensated residual's Taylor/error budget applies.
  return sum > 1.0f - 0x1p-16f && sum < 1.0f + 0x1p-16f;
}

__device__ inline bool IsZero(float3 n) {
  // Preserve fail-closed semantics for nonzero subnormal input under FTZ.
  return ((__float_as_uint(n.x) | __float_as_uint(n.y) | __float_as_uint(n.z)) &
          0x7fffffffu) == 0;
}

__device__ inline float3 Correct(float3 base, int q) {
  const float rho = __fmul_rn(static_cast<float>(q), kStep);
  // Computing factor=1+rho first would lose positive odd q values near one.
  return make_float3(__fmaf_rn(base.x, rho, base.x),
                     __fmaf_rn(base.y, rho, base.y),
                     __fmaf_rn(base.z, rho, base.z));
}

// Valid canonical zero decodes successfully; malformed one-sided sentinels
// fail closed. On any refusal the returned vector is exactly zero.
__device__ inline bool Decode(Encoded code, float3& out) {
  out = make_float3(0, 0, 0);
  const bool bad_x = code.direction.x == oct_revisit::kInvalid;
  const bool bad_y = code.direction.y == oct_revisit::kInvalid;
  if (code.magnitude == kInvalidMagnitude || bad_x || bad_y)
    return code.magnitude == kInvalidMagnitude && bad_x && bad_y;
  const float3 base = oct_revisit::Decode(code.direction);
  out = Correct(base, static_cast<int>(code.magnitude) - 127);
  return true;
}

// Radius=0: one candidate, minimum work. Radius=1: three candidates.
// Radius=2: five candidates, default fidelity. No candidate array or FP64.
// The certificate covers actual output rounding and actual base-decoder norm.
// Limit is caller policy, never silently widened on refusal. This is a
// per-vector approximation; callers must not silently normalize on failure.
template <int Radius = 2>
__device__ inline Status Encode(float3 source,
                                Encoded& out,
                                float norm_error_limit = kDefaultNormErrorLimit,
                                float3* prepared = nullptr) {
  static_assert(Radius >= 0 && Radius <= 2, "bounded candidate search only");
  out = Invalid();
  if (prepared) *prepared = make_float3(0, 0, 0);
  if (!isfinite(source.x) || !isfinite(source.y) || !isfinite(source.z))
    return Status::kNonfinite;
  if (IsZero(source)) return Status::kZero;
  if (!NearUnit(source)) return Status::kOutOfEnvelope;
  const float target = ResidualUnchecked(source);
  if (!(fabsf(target) <= __fsub_rn(kEnvelope, kEstimatorBudget)))
    return Status::kOutOfEnvelope;
  if (!(norm_error_limit >= 2.0f * kEstimatorBudget) ||
      !isfinite(norm_error_limit))
    return Status::kNoCertifiedCandidate;
  oct_revisit::Code direction;
  if (!oct_revisit::Encode(source, direction))
    return Status::kNoCertifiedCandidate;
  const float3 base = oct_revisit::Decode(direction);
  const float base_mu = ResidualUnchecked(base);
  const float difference = __fsub_rn(target, base_mu);
  const float rho = __fmaf_rn(-base_mu, difference, difference);
  const int center = __float2int_rn(__fmul_rn(rho, 0x1p24f));
  float best_error = CUDART_INF_F;
  int best_q = 0;
  bool found = false;
#pragma unroll
  for (int offset = -Radius; offset <= Radius; ++offset) {
    const int q = center + offset;
    if (q < -127 || q > 127) continue;
    const float3 candidate = Correct(base, q);
    const float error = fabsf(__fsub_rn(ResidualUnchecked(candidate), target));
    if (!found || error < best_error) {
      found = true;
      best_error = error;
      best_q = q;
    }
  }
  if (!found ||
      __fadd_rn(best_error, 2.0f * kEstimatorBudget) > norm_error_limit)
    return Status::kNoCertifiedCandidate;
  out = {direction, static_cast<uint8_t>(best_q + 127)};
  if (prepared) *prepared = Correct(base, best_q);
  return Status::kOk;
}

}  // namespace oct_magnitude
