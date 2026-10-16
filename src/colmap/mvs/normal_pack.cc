// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/normal_pack.h"

#include "colmap/mvs/normal_pack_bounded.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK
#include "normal_pack_ispc.h"
#endif
#include <cfenv>
#include <cmath>
#include <limits>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__FAST_MATH__)
#error Normal packing must retain IEEE validation and rounding behavior
#endif
namespace colmap::mvs::normal_pack {
namespace {
constexpr size_t kChunk =
    1u << 20;  // Original C++ chunk and SNORM/validation ABI.
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK
constexpr size_t kOctIsPcTile = 4096;
static_assert(!std::is_same_v<OctNormalPair, ispc::NormalPackOctPair>);
static_assert(std::is_trivially_copyable_v<OctNormalPair> &&
              std::is_trivially_copyable_v<ispc::NormalPackOctPair>);
static_assert(std::is_standard_layout_v<OctNormalPair> &&
              std::is_standard_layout_v<ispc::NormalPackOctPair>);
static_assert(sizeof(OctNormalPair) == 4 &&
              sizeof(ispc::NormalPackOctPair) == 4);
static_assert(offsetof(OctNormalPair, x) == 0 &&
              offsetof(ispc::NormalPackOctPair, x) == 0);
static_assert(offsetof(OctNormalPair, y) == 2 &&
              offsetof(ispc::NormalPackOctPair, y) == 2);
static_assert(std::is_same_v<decltype(OctNormalPair::x), int16_t> &&
              std::is_same_v<decltype(OctNormalPair::y), int16_t> &&
              std::is_same_v<decltype(ispc::NormalPackOctPair::x), int16_t> &&
              std::is_same_v<decltype(ispc::NormalPackOctPair::y), int16_t>);
static_assert(kOctIsPcTile * sizeof(ispc::NormalPackOctPair) == 16 * 1024);
static_assert(sizeof(std::array<ispc::NormalPackOctPair, kOctIsPcTile>) ==
              16 * 1024);
#endif
}  // namespace
bool BoundedCppAvailable() {
#if defined(__SSE__)
  // Require nearest rounding, gradual underflow and masked SIMD exceptions.
  // The C++ admission gate is independent of ISPC build and AVX2 support.
  const unsigned csr = _mm_getcsr();
  return std::fegetround() == FE_TONEAREST &&
         (csr & ((3u << 13) | (1u << 15) | (1u << 6))) == 0 &&
         (csr & (0x3fu << 7)) == (0x3fu << 7);
#else
  // Other FP control registers have not been qualified for the bounded codec.
  return false;
#endif
}
bool Available() {
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK && \
    (defined(__GNUC__) || defined(__clang__)) &&                             \
    (defined(__x86_64__) || defined(__i386__)) && defined(__SSE__)
  __builtin_cpu_init();  // Includes AVX OS-state support, no vendor
                         // restriction.
  return __builtin_cpu_supports("avx2") && BoundedCppAvailable();
#else
  return false;
#endif
}
bool Validate(const float* source, const size_t count, const Backend backend) {
  const bool use_ispc = backend == Backend::kAutomatic && Available();
  for (size_t begin = 0; begin < count;) {
    const size_t n = std::min(kChunk, count - begin);
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK
    if (use_ispc) {
      if (ispc::normal_pack_validate_normals(source + begin,
                                             static_cast<int32_t>(n)) !=
          std::numeric_limits<int32_t>::max())
        return false;
    } else
#endif
    {
      for (size_t i = 0; i < n; ++i) {
        const float x = source[begin + i];
        if (!(std::isfinite(x) && x >= -1.0f && x <= 1.0f)) return false;
      }
    }
    begin += n;
  }
  (void)use_ispc;
  return true;
}
void SnormUnchecked(const float* source,
                    const size_t count,
                    int16_t* output,
                    const Backend backend) {
  const bool use_ispc = backend == Backend::kAutomatic && Available();
  const bool use_bounded =
      backend == Backend::kBoundedCpp && BoundedCppAvailable();
  for (size_t begin = 0; begin < count;) {
    const size_t n = std::min(kChunk, count - begin);
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK
    if (use_ispc) {
      ispc::normal_pack_snorm(
          source + begin, output + begin, static_cast<int32_t>(n));
    } else
#endif
    {
      for (size_t i = begin; i < begin + n; ++i)
        output[i] = use_bounded
                        ? bounded::EncodeNormalSnorm16Unchecked(source[i])
                        : EncodeNormalSnorm16Unchecked(source[i]);
    }
    begin += n;
  }
  (void)use_ispc;
}
void OctUnchecked(const float* source,
                  const size_t plane,
                  OctNormalPair* output,
                  uint16_t* magnitude,
                  const Backend backend) {
  const bool use_ispc = backend == Backend::kAutomatic && Available();
  const bool use_bounded =
      backend == Backend::kBoundedCpp && BoundedCppAvailable();
  for (size_t begin = 0; begin < plane;) {
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK
    const size_t n = std::min(use_ispc ? kOctIsPcTile : kChunk, plane - begin);
#else
    const size_t n = std::min(kChunk, plane - begin);
#endif
#if defined(COLMAP_HAVE_ISPC_NORMAL_PACK) && COLMAP_HAVE_ISPC_NORMAL_PACK
    if (use_ispc) {
      // A bounded 16 KiB tile bridges distinct trivially-copyable pair types.
      alignas(64) std::array<ispc::NormalPackOctPair, kOctIsPcTile> scratch;
      ispc::normal_pack_oct(source + begin,
                            source + plane + begin,
                            source + 2 * plane + begin,
                            scratch.data(),
                            magnitude ? magnitude + begin : nullptr,
                            static_cast<int32_t>(n),
                            magnitude != nullptr);
      std::memcpy(output + begin, scratch.data(), n * sizeof(OctNormalPair));
    } else
#endif
    {
      for (size_t i = begin; i < begin + n; ++i) {
        const Eigen::Vector3f normal(
            source[i], source[plane + i], source[2 * plane + i]);
        if (use_bounded) {
          bounded::EncodeNormalOct2x16Unchecked(normal, &output[i]);
          if (magnitude)
            magnitude[i] =
                bounded::EncodeNormalMagnitudeUnorm16Unchecked(normal);
        } else {
          EncodeNormalOct2x16Unchecked(normal, &output[i]);
          if (magnitude)
            magnitude[i] = EncodeNormalMagnitudeUnorm16Unchecked(normal);
        }
      }
    }
    begin += n;
  }
  (void)use_ispc;
}
}  // namespace colmap::mvs::normal_pack
