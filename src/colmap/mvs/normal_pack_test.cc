// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/normal_pack.h"

#include "colmap/mvs/normal_pack_bounded.h"

#include <gtest/gtest.h>
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK
#include "normal_pack_ispc.h"
#endif
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace colmap::mvs::normal_pack {
namespace {
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK
static_assert(!std::is_same_v<OctNormalPair, ispc::NormalPackOctPair>);
static_assert(std::is_trivially_copyable_v<OctNormalPair> &&
              std::is_trivially_copyable_v<ispc::NormalPackOctPair>);
static_assert(std::is_standard_layout_v<OctNormalPair> &&
              std::is_standard_layout_v<ispc::NormalPackOctPair>);
static_assert(sizeof(ispc::NormalPackOctPair) == 4);
static_assert(offsetof(ispc::NormalPackOctPair, x) == 0 &&
              offsetof(ispc::NormalPackOctPair, y) == 2);
static_assert(std::is_same_v<decltype(ispc::NormalPackOctPair::x), int16_t> &&
              std::is_same_v<decltype(ispc::NormalPackOctPair::y), int16_t>);
static_assert(sizeof(OctNormalPair) == 4 &&
              alignof(OctNormalPair) == alignof(int16_t));
static_assert(offsetof(OctNormalPair, x) == 0 &&
              offsetof(OctNormalPair, y) == 2);
#endif
void Compare(size_t plane, Backend backend);
TEST(NormalPack, EnabledBuildActuallyAdmitsQualifiedSimd) {
  // Keep this case in both ON/OFF builds: bounded C++ must be admitted even
  // when the generated ISPC ABI is absent. The dispatcher uses this same gate.
#if defined(__SSE__)
  const int rounding = std::fegetround();
  const unsigned csr = _mm_getcsr();
  EXPECT_EQ(std::fesetround(FE_TONEAREST), 0);
  _mm_setcsr((_mm_getcsr() & ~((3u << 13) | (1u << 15) | (1u << 6))) |
             (0x3fu << 7));
  EXPECT_TRUE(BoundedCppAvailable());
  // Exercise the admitted bounded path across the ISPC tile boundary under
  // this known FP state, including SNORM, oct pairs and optional magnitude.
  Compare(4097, Backend::kBoundedCpp);
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK && \
    (defined(__GNUC__) || defined(__clang__)) &&                             \
    (defined(__x86_64__) || defined(__i386__))
  __builtin_cpu_init();
  EXPECT_EQ(Available(), static_cast<bool>(__builtin_cpu_supports("avx2")));
#else
  EXPECT_FALSE(Available());
#endif
  _mm_setcsr(csr);
  std::fesetround(rounding);
#else
  EXPECT_FALSE(BoundedCppAvailable());
  EXPECT_FALSE(Available());
#endif
}
void Compare(size_t plane, Backend backend = Backend::kAutomatic) {
  const float fixed[] = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      0.5f,
      -0.5f,
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::nextafter(1.0f, 0.0f),
      static_cast<float>(0.5 / 32767.0),
      std::nextafter(static_cast<float>(0.5 / 32767.0), 0.0f),
      std::nextafter(static_cast<float>(0.5 / 32767.0), 1.0f)};
  std::vector<float> src(3 * plane);
  for (size_t i = 0; i < src.size(); ++i) src[i] = fixed[i % std::size(fixed)];
  // Explicit zero sentinel and fold/tie cases, in addition to arbitrary tails.
  const float xyz[][3] = {{0, 0, 0},
                          {-0.0f, 0, -0.0f},
                          {1, 1, 0},
                          {-1, -1, 0},
                          {1, 1, -1},
                          {-1, 0, -1},
                          {0, 1, -1},
                          {1, 1, 1}};
  for (size_t i = 0; i < std::min(plane, std::size(xyz)); ++i)
    for (size_t c = 0; c < 3; ++c) src[c * plane + i] = xyz[i][c];
  ASSERT_TRUE(Validate(src.data(), src.size(), backend));
  std::vector<int16_t> a(src.size()), b(src.size());
  SnormUnchecked(src.data(), src.size(), a.data(), Backend::kCpp);
  SnormUnchecked(src.data(), src.size(), b.data(), backend);
  EXPECT_EQ(a, b);
  std::vector<OctNormalPair> x(plane), y(plane);
  std::vector<uint16_t> mx(plane), my(plane);
  for (bool magnitude : {false, true}) {
    OctUnchecked(src.data(),
                 plane,
                 x.data(),
                 magnitude ? mx.data() : nullptr,
                 Backend::kCpp);
    OctUnchecked(
        src.data(), plane, y.data(), magnitude ? my.data() : nullptr, backend);
    for (size_t i = 0; i < plane; ++i) {
      ASSERT_EQ(x[i].x, y[i].x) << i;
      ASSERT_EQ(x[i].y, y[i].y) << i;
    }
    if (magnitude) EXPECT_EQ(mx, my);
  }
}
TEST(NormalPack, OriginalCodeParityAcrossTailsAndChunks) {
  for (Backend backend : {Backend::kAutomatic, Backend::kBoundedCpp})
    for (size_t n : {1u,
                     3u,
                     4u,
                     7u,
                     8u,
                     9u,
                     15u,
                     17u,
                     65u,
                     257u,
                     4095u,
                     4096u,
                     4097u,
                     8193u,
                     1048577u})
      Compare(n, backend);
  if (BoundedCppAvailable()) {
    // Independent original std::round oracle across every bounded half-code
    // tie and its immediately adjacent representable FP64 values.
    for (int n = 0; n < 65536; ++n) {
      const double tie = static_cast<double>(n) + 0.5;
      for (double x :
           {std::nextafter(tie, 0.0),
            tie,
            std::nextafter(tie, std::numeric_limits<double>::infinity())})
        for (double sign : {-1.0, 1.0})
          ASSERT_EQ(bounded::RoundNearestTiesAwayBounded(sign * x),
                    std::round(sign * x));
    }
    for (double x : {0.0,
                     -0.0,
                     std::numeric_limits<double>::denorm_min(),
                     -std::numeric_limits<double>::denorm_min(),
                     65535.0,
                     -65535.0}) {
      const double actual = bounded::RoundNearestTiesAwayBounded(x);
      const double expected = std::round(x);
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(std::signbit(actual), std::signbit(expected));
    }
  }
}
TEST(NormalPack, InvalidValidationIncludingChunkBoundary) {
  std::vector<float> src(1048579, 0.0f);
  for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::nextafter(1.0f, 2.0f),
                        std::nextafter(-1.0f, -2.0f)}) {
    for (size_t index : {0u, 1048575u, 1048576u, 1048578u}) {
      src[index] = invalid;
      EXPECT_FALSE(Validate(src.data(), src.size()));
      EXPECT_FALSE(Validate(src.data(), src.size(), Backend::kCpp));
      EXPECT_FALSE(Validate(src.data(), src.size(), Backend::kBoundedCpp));
      src[index] = 0;
    }
  }
}
TEST(NormalPack, UnsupportedFpStateUsesExactCppFallback) {
  const int saved = std::fegetround();
  for (int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    if (std::fesetround(mode) == 0) {
      EXPECT_FALSE(Available());
      EXPECT_FALSE(BoundedCppAvailable());
      Compare(65);
      Compare(65, Backend::kBoundedCpp);
    }
  }
  std::fesetround(saved);
#if defined(__SSE__)
  const unsigned csr = _mm_getcsr();
  for (unsigned mask : {1u << 15, 1u << 6, 1u << 13}) {
    _mm_setcsr(csr | mask);
    EXPECT_FALSE(Available());
    EXPECT_FALSE(BoundedCppAvailable());
    Compare(65);
    Compare(65, Backend::kBoundedCpp);
  }
  // Clear sticky flags before unmasking; do not execute a codec in this state.
  for (unsigned exception = 7; exception <= 12; ++exception) {
    _mm_setcsr((csr & ~0x3fu) & ~(1u << exception));
    EXPECT_FALSE(Available());
    EXPECT_FALSE(BoundedCppAvailable());
  }
  _mm_setcsr(csr);
#endif
}
TEST(NormalPack, EmptyRangeDoesNotTouchNullPointers) {
  for (Backend backend :
       {Backend::kAutomatic, Backend::kCpp, Backend::kBoundedCpp}) {
    EXPECT_TRUE(Validate(nullptr, 0, backend));
    SnormUnchecked(nullptr, 0, nullptr, backend);
    OctUnchecked(nullptr, 0, nullptr, nullptr, backend);
  }
}
}  // namespace
}  // namespace colmap::mvs::normal_pack
