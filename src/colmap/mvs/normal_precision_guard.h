// SPDX-License-Identifier: BSD-3-Clause
#pragma once
// Conservative SNORM decision screen with exact FP32 callbacks.
#include "colmap/mvs/mat.h"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#endif

namespace colmap {
namespace mvs {
namespace normal_precision_guard {
using Vec = Eigen::Vector3f;
using Rot = Eigen::Matrix3f;
using Error = Eigen::Vector3d;
// Eligible positive normal doubles use an exact one-ULP increment;
// exceptional and unsupported cases retain libm nextafter.
inline double Up(double x) {
#if defined(__BYTE_ORDER__) && defined(__FLOAT_WORD_ORDER__) && \
    (__BYTE_ORDER__ == __FLOAT_WORD_ORDER__)
  if constexpr (sizeof(double) == sizeof(std::uint64_t) &&
                std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::radix == 2 &&
                std::numeric_limits<double>::digits == 53 &&
                std::numeric_limits<double>::min_exponent == -1021 &&
                std::numeric_limits<double>::max_exponent == 1024) {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    // Positive normal -> positive finite normal only. Integer classification
    // leaves NaNs and DAZ-sensitive values untouched until the libm fallback.
    // Max finite stays with libm to preserve overflow flags and errno.
    if (bits >= UINT64_C(0x0010000000000000) &&
        bits < UINT64_C(0x7fefffffffffffff)) {
      ++bits;
      std::memcpy(&x, &bits, sizeof(x));
      return x;
    }
  }
#endif
  return std::nextafter(x, std::numeric_limits<double>::infinity());
}
inline double Add(double a, double b) { return Up(a + b); }
inline double Mul(double a, double b) { return Up(a * b); }
// The interval screen is admitted only for ordinary GCC/Clang x86 SSE
// arithmetic. MSVC and other architectures require independent compiler/control
// coverage before enabling the exact-screen option. This gate does not prevent
// ordinary Fusion builds, including builds that select fast math.
inline constexpr bool SupportedBuild() {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__SSE__) && \
    !defined(_MSC_VER) && !defined(__FAST_MATH__) &&                 \
    (!defined(__FINITE_MATH_ONLY__) || __FINITE_MATH_ONLY__ == 0) && \
    (!defined(__FLT_EVAL_METHOD__) || __FLT_EVAL_METHOD__ == 0)
  return true;
#else
  return false;
#endif
}
inline bool SupportedArithmetic() {
  if (!SupportedBuild()) return false;
  if (std::fegetround() != FE_TONEAREST) return false;
#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
  // SSE rounding may differ from the x87 mode reported by fegetround().
  if (_mm_getcsr() & ((1u << 15) | (3u << 13) | (1u << 6))) return false;
#endif
  return std::numeric_limits<float>::is_iec559;
}
inline double Gamma() {
  constexpr double u = 0x1p-24;
  return Up(7 * u / (1 - 7 * u));
}
struct Operand {
  Vec value;
  Error error;
  bool exact = false;
};
// Reuse the concrete FP32 rotation already computed by Fusion; the value
// under unchanged controls across internal lease release, never normalized
// or an exact fallback. All accepted interval/control checks remain here.
inline Operand CompressedOperandFromRotated(const Rot& r,
                                            const Vec& decoded,
                                            const Vec& rotated) {
  Operand result{rotated, Error::Zero(), false};
  const double e = Up(1.0 / 65534.0 + 0x1p-24);
  // No callbacks or FP-control writes occur within this component loop.
  const bool arithmetic_supported = SupportedArithmetic();
  const bool finite_value = result.value.allFinite();
  const double gamma = Gamma();
  for (int i = 0; i < 3; ++i) {
    double delta = 0, magnitudes = 0;
    for (int j = 0; j < 3; ++j) {
      delta = Add(delta, Mul(std::abs(double(r(i, j))), e));
      magnitudes =
          Add(magnitudes,
              Mul(std::abs(double(r(i, j))), std::abs(double(decoded(j)))));
    }
    // Both original and decoded FP32 matvec rounding; gradual underflow term.
    result.error(i) =
        Add(Add(delta, Mul(gamma, Add(2 * magnitudes, delta))),
            32 * double(std::numeric_limits<float>::denorm_min()));
    if (!arithmetic_supported || !finite_value ||
        !std::isfinite(result.error(i)) ||
        magnitudes + delta > double(std::numeric_limits<float>::max()) / 16) {
      result.error.setConstant(std::numeric_limits<double>::infinity());
      break;
    }
  }
  return result;
}

inline Operand CompressedOperand(const Rot& r, const Vec& decoded) {
  return CompressedOperandFromRotated(r, decoded, r * decoded);
}

// Immutable image terms. Only prepare under the controls supported by the
// screen; a caller that later restores supported controls can use the ordinary
// path when preparation was unavailable. Never cache the per-call control test.
struct PreparedRotation {
  Eigen::Matrix3d absolute = Eigen::Matrix3d::Zero();
  Error delta = Error::Zero();
  bool valid = false;
};
inline PreparedRotation PrepareRotation(const Rot& r) {
  PreparedRotation prepared;
  if (!SupportedArithmetic()) return prepared;
  const double e = Up(1.0 / 65534.0 + 0x1p-24);
  for (int i = 0; i < 3; ++i) {
    double delta = 0;
    for (int j = 0; j < 3; ++j) {
      prepared.absolute(i, j) = std::abs(double(r(i, j)));
      delta = Add(delta, Mul(prepared.absolute(i, j), e));
    }
    prepared.delta(i) = delta;
  }
  prepared.valid = true;
  return prepared;
}
inline Operand CompressedOperandFromRotated(const Rot& r,
                                            const PreparedRotation& prepared,
                                            const Vec& decoded,
                                            const Vec& rotated) {
  if (!prepared.valid || !SupportedArithmetic()) {
    return CompressedOperandFromRotated(r, decoded, rotated);
  }
  Operand result{rotated, Error::Zero(), false};
  const bool finite_value = result.value.allFinite();
  const double gamma = Gamma();
  for (int i = 0; i < 3; ++i) {
    double magnitudes = 0;
    for (int j = 0; j < 3; ++j) {
      magnitudes =
          Add(magnitudes,
              Mul(prepared.absolute(i, j), std::abs(double(decoded(j)))));
    }
    const double delta = prepared.delta(i);
    result.error(i) =
        Add(Add(delta, Mul(gamma, Add(2 * magnitudes, delta))),
            32 * double(std::numeric_limits<float>::denorm_min()));
    if (!finite_value || !std::isfinite(result.error(i)) ||
        magnitudes + delta > double(std::numeric_limits<float>::max()) / 16) {
      result.error.setConstant(std::numeric_limits<double>::infinity());
      break;
    }
  }
  return result;
}

enum class Screen { kAccept, kReject, kExact };
inline Screen ScreenNormal(const Operand& a,
                           const Operand& b,
                           float threshold) {
  if (!SupportedArithmetic() || !std::isfinite(threshold) ||
      !a.value.allFinite() || !b.value.allFinite() || !a.error.allFinite() ||
      !b.error.allFinite())
    return Screen::kExact;
  const float dot = a.value.dot(b.value);
  double q = 0, approx = 0, original = 0;
  for (int i = 0; i < 3; ++i) {
    const double av = std::abs(double(a.value(i))),
                 bv = std::abs(double(b.value(i)));
    q = Add(q,
            Add(Add(Mul(av, b.error(i)), Mul(bv, a.error(i))),
                Mul(a.error(i), b.error(i))));
    approx = Add(approx, Mul(av, bv));
    original = Add(original, Mul(Add(av, a.error(i)), Add(bv, b.error(i))));
  }
  const double error =
      Add(Add(q, Mul(Gamma(), Add(approx, original))),
          32 * double(std::numeric_limits<float>::denorm_min()));
  if (!std::isfinite(dot) || !std::isfinite(error)) return Screen::kExact;
  if (Up(double(dot) + error) < double(threshold)) return Screen::kReject;
  if (std::nextafter(double(dot) - error,
                     -std::numeric_limits<double>::infinity()) >=
      double(threshold))
    return Screen::kAccept;
  return Screen::kExact;
}

template <typename ReadRoot, typename ReadCandidate>
bool NormalAccept(Operand& root,
                  const Operand& candidate,
                  float threshold,
                  ReadRoot read_root,
                  ReadCandidate read_candidate,
                  bool* used_exact = nullptr) {
  // Lazy Eigen expressions can borrow a vector destroyed by the callback.
  static_assert(std::is_same_v<std::invoke_result_t<ReadRoot>, Vec> &&
                    std::is_same_v<std::invoke_result_t<ReadCandidate>, Vec>,
                "Exact normal callbacks must return an owning Vector3f");
  const Screen screen = ScreenNormal(root, candidate, threshold);
  if (used_exact) *used_exact = screen == Screen::kExact;
  if (screen != Screen::kExact) return screen == Screen::kAccept;
  if (!root.exact) {
    root.value = read_root();
    root.error.setZero();
    root.exact = true;
  }
  const Vec normal = read_candidate();
  if (!root.value.allFinite() || !normal.allFinite() ||
      !std::isfinite(threshold))
    throw std::runtime_error("invalid exact normal guard");
  const float dot = root.value.dot(normal);
  if (!std::isfinite(dot))
    throw std::runtime_error("nonfinite exact normal dot");
  return !(dot < threshold);  // Preserve native strict rejection and equality.
}
}  // namespace normal_precision_guard
}  // namespace mvs
}  // namespace colmap
