// SPDX-License-Identifier: BSD-3-Clause

#include "thirdparty/VLFeat/imopv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

class VLFeatPadding : public testing::Test {
 protected:
  void SetUp() override { simd_enabled_ = vl_get_simd_enabled(); }
  void TearDown() override { vl_set_simd_enabled(simd_enabled_); }

 private:
  vl_bool simd_enabled_;
};

template <typename T>
void Convolve(T* dst, size_t dst_stride, const T* src, int width, int height,
              int src_stride, const T* filter, int begin, int end, int step,
              unsigned int flags) {
  if constexpr (std::is_same_v<T, float>) {
    vl_imconvcol_vf(dst, dst_stride, src, width, height, src_stride,
                    filter, begin, end, step, flags);
  } else {
    vl_imconvcol_vd(dst, dst_stride, src, width, height, src_stride,
                    filter, begin, end, step, flags);
  }
}

// Match the documented convolution and descending tap/store order. The same
// reference also accepts aliased source/output to check sequential fallback.
template <typename T>
void ReferenceConvolve(T* dst, int dst_stride, const T* src, int width,
                       int height, int src_stride, const T* filter, int begin,
                       int end, int step, unsigned int flags,
                       bool reverse_taps = false) {
  for (int x = 0; x < width; ++x) {
    for (int y = 0, output_y = 0; y < height; y += step, ++output_y) {
      T sum = 0;
      for (int i = 0; i <= end - begin; ++i) {
        const int tap = reverse_taps ? begin + i : end - i;
        const int row = y - tap;
        const bool outside = row < 0 || row >= height;
        const T value = outside && (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO
                            ? T(0)
                            : src[std::clamp(row, 0, height - 1) * src_stride + x];
        sum += value * filter[tap - begin];
      }
      dst[(flags & VL_TRANSPOSE) ? x * dst_stride + output_y
                                : output_y * dst_stride + x] = sum;
    }
  }
}

template <typename T>
void CheckSupportsAndLayouts() {
  struct Layout {
    int width, height, step, padding;
    bool transpose;
  };
  const Layout layouts[] = {{1, 2, 1, 0, false},
                            {1, 1, 2, 0, true},
                            {3, 7, 2, 4, false},
                            {4, 2, 1, 0, true},
                            {5, 7, 2, 0, false},
                            {16, 32, 1, 4, false},
                            {17, 33, 2, 0, true},
                            {127, 2, 1, 0, false},
                            {128, 7, 2, 4, true},
                            {129, 33, 1, 0, false},
                            {160, 33, 2, 0, true},
                            {160, 33, 1, 4, false}};
  for (const int simd : {0, 1}) {
    vl_set_simd_enabled(simd);
    SCOPED_TRACE(simd);
    for (const auto& layout : layouts) {
      const auto [width, height, step, padding, transpose] = layout;
      SCOPED_TRACE(testing::Message() << width << "x" << height << " step="
                   << step << " padding=" << padding << " transpose=" << transpose);
      const int src_stride = width + padding;
      const int output_height = (height - 1) / step + 1;
      const int dst_stride = (transpose ? output_height : width) + padding;
      const int output_size = transpose ? (width - 1) * dst_stride + output_height
                                        : (output_height - 1) * dst_stride + width;
      std::vector<T> source((height - 1) * src_stride + width);
      for (size_t i = 0; i < source.size(); ++i) {
        source[i] = T(int(i % 29) - 14) * T(0.125);
      }
      const auto original = source;
      for (const auto support : {std::pair{-3, 3}, std::pair{0, 0},
                                 std::pair{3, 4}, std::pair{-4, -3},
                                 std::pair{-9, 7}}) {
        const auto [begin, end] = support;
        SCOPED_TRACE(testing::Message() << "support=" << begin << "," << end);
        std::vector<T> filter(end - begin + 1);
        for (size_t i = 0; i < filter.size(); ++i) {
          filter[i] = T(int(i % 5) - 2) * T(0.0625);
        }
        for (const unsigned int mode : {VL_PAD_BY_ZERO, VL_PAD_BY_CONTINUITY}) {
          SCOPED_TRACE(mode);
          const unsigned int flags = mode | (transpose ? VL_TRANSPOSE : 0);
          // Include untouched row padding and a suffix guard in exact equality.
          std::vector<T> expected(output_size + 16, T(-999));
          auto actual = expected;
          ReferenceConvolve(expected.data(), dst_stride, source.data(), width,
                            height, src_stride, filter.data(), begin, end, step, flags);
          Convolve(actual.data(), dst_stride, source.data(), width, height,
                   src_stride, filter.data(), begin, end, step, flags);
          ASSERT_EQ(std::memcmp(actual.data(), expected.data(),
                                actual.size() * sizeof(T)), 0);
          ASSERT_EQ(source, original);
        }
      }
    }
  }
}

TEST_F(VLFeatPadding, FloatSupportsAndLayouts) { CheckSupportsAndLayouts<float>(); }
TEST_F(VLFeatPadding, DoubleSupportsAndLayouts) { CheckSupportsAndLayouts<double>(); }

TEST_F(VLFeatPadding, InPlaceContinuityRetainsSequentialFallback) {
  for (const int simd : {0, 1}) {
    vl_set_simd_enabled(simd);
    constexpr int width = 160, height = 33;
    std::vector<float> actual(width * height);
    for (size_t i = 0; i < actual.size(); ++i) actual[i] = float(i % 19) / 8;
    auto expected = actual;
    const float filter[] = {0.125f, 0.25f, 0.125f};
    ReferenceConvolve(expected.data(), width, expected.data(), width, height,
                      width, filter, -1, 1, 1, VL_PAD_BY_CONTINUITY);
    Convolve(actual.data(), width, actual.data(), width, height, width,
             filter, -1, 1, 1, VL_PAD_BY_CONTINUITY);
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(),
                          actual.size() * sizeof(float)), 0);
  }
}

template <typename T>
void CheckRounding() {
  for (const int simd : {0, 1}) {
    vl_set_simd_enabled(simd);
    for (const int width : {5, 128, 160}) {
      constexpr int height = 33;
      std::vector<T> input(width * height);
      for (size_t i = 0; i < input.size(); ++i) {
        // Power-of-two inputs make each product exact, isolating sum order
        // from multiply-add contraction without changing the filter.
        input[i] = std::ldexp((i % 3 == 0) ? T(1) : T(-1),
                              6 * static_cast<int>(i % 7) - 18);
      }
      const T filter[] = {T(0.1), T(-0.7), T(1.3), T(0.9), T(-0.1)};
      std::vector<T> actual(input.size()), expected(input.size());
      std::vector<T> reverse_expected(input.size());
      ReferenceConvolve(expected.data(), width, input.data(), width, height,
                        width, filter, -2, 2, 1, VL_PAD_BY_CONTINUITY);
      ReferenceConvolve(reverse_expected.data(), width, input.data(), width,
                        height, width, filter, -2, 2, 1,
                        VL_PAD_BY_CONTINUITY, true);
      ASSERT_NE(std::memcmp(expected.data(), reverse_expected.data(),
                            expected.size() * sizeof(T)), 0)
          << "element bytes=" << sizeof(T) << " simd=" << simd << " width=" << width;
      Convolve(actual.data(), width, input.data(), width, height, width,
               filter, -2, 2, 1, VL_PAD_BY_CONTINUITY);
      EXPECT_EQ(std::memcmp(actual.data(), expected.data(),
                            actual.size() * sizeof(T)), 0)
          << "element bytes=" << sizeof(T) << " simd=" << simd << " width=" << width;
    }
  }
}

TEST_F(VLFeatPadding, NonDyadicCancellationRetainsTapOrderBits) {
  CheckRounding<float>();
  CheckRounding<double>();
}

}  // namespace
