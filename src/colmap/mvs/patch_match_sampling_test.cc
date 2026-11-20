// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_sampling.cuh"

#include <array>
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(PatchMatchSampling, EmptySupportAndLowerDrawEndpoint) {
  for (const int stride : {1, 8, 23, 32, 47}) {
    SCOPED_TRACE(stride);
    std::vector<float> values(3 * stride, -17.0f);
    values[0] = -0.0f;
    values[stride] = 0.0f;
    values[2 * stride] = -0.0f;
    const auto empty = values;
    EXPECT_FALSE(TransformPDFToCDF(values.data(), 3, stride));
    EXPECT_EQ(
        std::memcmp(values.data(), empty.data(), values.size() * sizeof(float)),
        0);
    EXPECT_EQ(SampleSource(values.data(), 3, stride, FLT_EPSILON / 2), -1);
    EXPECT_EQ(SampleSource(values.data(), 3, stride, 1.0f), -1);
    values[0] = 0.0f;
    values[stride] = 1.0f;
    values[2 * stride] = 3.0f;
    EXPECT_TRUE(TransformPDFToCDF(values.data(), 3, stride));
    EXPECT_EQ(values[0], 0.0f);
    EXPECT_EQ(values[stride], .25f);
    EXPECT_EQ(values[2 * stride], 1.0f);
    EXPECT_EQ(SampleSource(values.data(), 3, stride, FLT_EPSILON / 2), 1);
    EXPECT_EQ(SampleSource(values.data(), 3, stride, .5f), 2);
    EXPECT_EQ(SampleSource(values.data(), 3, stride, 1.0f), 2);
    for (size_t i = 0; i < values.size(); ++i) {
      if (i % stride != 0) {
        EXPECT_EQ(values[i], -17.0f);
      }
    }
  }
}

TEST(PatchMatchSampling, NonemptyCdfRetainsOrderedArithmetic) {
  const std::array<std::array<float, 3>, 3> masses{
      {{{0.0f, 1.0f, 3.0f}},
       {{5.0f, 2.0f, 0.0f}},
       {{0x1p20f, 1.0f, 0x1p-20f}}}};
  for (const int stride : {1, 8, 23, 32, 47}) {
    for (const auto& mass : masses) {
      std::vector<float> expected(3 * stride, -17.0f);
      for (int i = 0; i < 3; ++i) expected[i * stride] = mass[i];
      auto actual = expected;
      float total = 0.0f;
      for (int i = 0; i < 3; ++i) total += expected[i * stride];
      const float inverse = 1.0f / total;
      float cumulative = 0.0f;
      for (int i = 0; i < 3; ++i) {
        const float probability = expected[i * stride] * inverse;
        cumulative += probability;
        expected[i * stride] = cumulative;
      }
      ASSERT_TRUE(TransformPDFToCDF(actual.data(), 3, stride));
      EXPECT_EQ(
          std::memcmp(
              actual.data(), expected.data(), actual.size() * sizeof(float)),
          0);
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
