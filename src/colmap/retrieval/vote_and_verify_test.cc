// SPDX-License-Identifier: BSD-3-Clause

#define _USE_MATH_DEFINES

#include "colmap/retrieval/vote_and_verify.h"

#include "colmap/math/random.h"
#include "colmap/retrieval/geometry.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

struct SyntheticData {
  FeatureGeometryTransform img2_from_img1;
  std::vector<FeatureGeometryMatch> matches;
};

SyntheticData SynthesizeData(size_t num_inliers, size_t num_outliers) {
  SyntheticData data;
  data.img2_from_img1.scale = RandomUniformReal<float>(0.1, 10);
  data.img2_from_img1.angle = RandomUniformReal<float>(0, 2 * M_PI);
  data.img2_from_img1.tx = RandomUniformReal<float>(-100, 100);
  data.img2_from_img1.ty = RandomUniformReal<float>(-100, 100);

  const float sin_angle = std::sin(data.img2_from_img1.angle);
  const float cos_angle = std::cos(data.img2_from_img1.angle);

  data.matches.resize(num_inliers + num_outliers);
  for (size_t i = 0; i < num_inliers; ++i) {
    FeatureGeometryMatch& match = data.matches[i];
    match.geometry1.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry1.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry1.x = RandomUniformReal<float>(-100, 100);
    match.geometry1.y = RandomUniformReal<float>(-100, 100);
    match.geometry2.scale = data.img2_from_img1.scale * match.geometry1.scale;
    match.geometry2.orientation =
        data.img2_from_img1.angle + match.geometry1.orientation;
    match.geometry2.x =
        data.img2_from_img1.scale *
        (cos_angle * match.geometry1.x - sin_angle * match.geometry1.y);
    match.geometry2.y =
        data.img2_from_img1.scale *
        (sin_angle * match.geometry1.x + cos_angle * match.geometry1.y);
  }

  for (size_t i = 0; i < num_outliers; ++i) {
    FeatureGeometryMatch& match = data.matches[i + num_inliers];
    match.geometry1.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry1.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry1.x = RandomUniformReal<float>(-100, 100);
    match.geometry1.y = RandomUniformReal<float>(-100, 100);
    match.geometry2.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry2.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry2.x = RandomUniformReal<float>(-100, 100);
    match.geometry2.y = RandomUniformReal<float>(-100, 100);
  }

  return data;
}

TEST(VoteAndVerify, NoMatches) {
  EXPECT_EQ(VoteAndVerify(VoteAndVerifyOptions(), {}), 0);
}

TEST(VoteAndVerify, NoEffectiveInliers) {
  const size_t kNumInliers = 100;
  const size_t kNumOutliers = 50;
  const auto data = SynthesizeData(kNumInliers, kNumOutliers);
  VoteAndVerifyOptions options;
  options.eff_inlier_count = false;
  const int num_inliers = VoteAndVerify(options, data.matches);
  EXPECT_EQ(num_inliers, kNumInliers);
}

TEST(VoteAndVerify, EffectiveInliers) {
  const size_t kNumInliers = 100;
  const size_t kNumOutliers = 50;
  const auto data = SynthesizeData(kNumInliers, kNumOutliers);
  VoteAndVerifyOptions options;
  options.eff_inlier_count = true;
  const int num_inliers = VoteAndVerify(options, data.matches);
  EXPECT_GT(num_inliers, 0.8 * kNumInliers);
}

TEST(VoteAndVerify, EffectiveInliersWithZeroCoordinateSpan) {
  struct TestCase {
    float x;
    float y;
    float step_x;
    float step_y;
    int num_effective_inliers;
  };
  constexpr float kSmallStep = std::numeric_limits<float>::epsilon() / 16;
  const TestCase test_cases[] = {{16, 16, 0, 16, 4},
                                 {16, 16, 16, 0, 4},
                                 {16, 16, 0, 0, 1},
                                 {0, 16, kSmallStep, 0, 4},
                                 {16, 0, 0, kSmallStep, 4}};
  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(::testing::Message()
                 << test_case.step_x << ", " << test_case.step_y);
    std::vector<FeatureGeometryMatch> matches(4);
    for (size_t i = 0; i < matches.size(); ++i) {
      auto& match = matches[i];
      match.geometry1.x =
          test_case.x + static_cast<float>(i) * test_case.step_x;
      match.geometry1.y =
          test_case.y + static_cast<float>(i) * test_case.step_y;
      match.geometry1.scale = 1;
      match.geometry1.orientation = 0;
      match.geometry2 = match.geometry1;
    }

    VoteAndVerifyOptions options;
    EXPECT_EQ(VoteAndVerify(options, matches), test_case.num_effective_inliers);
    options.eff_inlier_count = false;
    EXPECT_EQ(VoteAndVerify(options, matches), 4);
  }
}

TEST(VoteAndVerify, EffectiveInlierCountIsTranslationInvariant) {
  for (const float offset_x : {-100.0f, 0.0f}) {
    for (const float offset_y : {-100.0f, 0.0f}) {
      SCOPED_TRACE(::testing::Message() << offset_x << ", " << offset_y);
      std::vector<FeatureGeometryMatch> matches(4);
      size_t i = 0;
      for (const float y : {0.0f, 1.0f}) {
        for (const float x : {0.0f, 1.0f}) {
          auto& match = matches[i++];
          match.geometry1.x = offset_x + x;
          match.geometry1.y = offset_y + y;
          match.geometry1.scale = 1;
          match.geometry1.orientation = 0;
          match.geometry2 = match.geometry1;
        }
      }

      VoteAndVerifyOptions options;
      EXPECT_EQ(VoteAndVerify(options, matches), 4);
      options.eff_inlier_count = false;
      EXPECT_EQ(VoteAndVerify(options, matches), 4);
    }
  }
}

TEST(VoteAndVerify, EffectiveInliersWithExtremeCoordinateSpan) {
  constexpr float kSmall = std::numeric_limits<float>::min();
  constexpr float kLarge = std::numeric_limits<float>::max();
  const float coordinates[][3] = {{0, kSmall, 2 * kSmall},
                                  {-kLarge, 0, kLarge}};
  for (const auto& coords : coordinates) {
    for (const bool swap_xy : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << coords[0] << ", " << coords[2] << ", " << swap_xy);
      std::vector<FeatureGeometryMatch> matches(3);
      for (size_t i = 0; i < matches.size(); ++i) {
        auto& match = matches[i];
        match.geometry1.x = swap_xy ? 0.25f : coords[i];
        match.geometry1.y = swap_xy ? coords[i] : 0.25f;
        match.geometry1.scale = 1;
        match.geometry1.orientation = 0;
        match.geometry2 = match.geometry1;
      }

      VoteAndVerifyOptions options;
      options.local_optimization = false;
      EXPECT_EQ(VoteAndVerify(options, matches), 3);
      options.eff_inlier_count = false;
      EXPECT_EQ(VoteAndVerify(options, matches), 3);
    }
  }
}

TEST(VoteAndVerify, VotingPreservesWideHistogramKeys) {
  std::vector<FeatureGeometryMatch> matches;
  for (const float translation_y : {0.0f, 2048.0f}) {
    for (const float y : {100.0f, 200.0f}) {
      for (const float x : {100.0f, 200.0f}) {
        FeatureGeometryMatch match;
        match.geometry1.x = x;
        match.geometry1.y = y;
        match.geometry1.scale = 1;
        match.geometry2 = match.geometry1;
        match.geometry2.y += translation_y;
        matches.push_back(match);
      }
    }
  }
  for (const int num_bins : {4096, 8192}) {
    SCOPED_TRACE(num_bins);
    VoteAndVerifyOptions options;
    options.num_trans_bins = num_bins;
    options.local_optimization = false;
    options.eff_inlier_count = false;
    if (num_bins == 8192 && std::numeric_limits<size_t>::digits < 64) {
      EXPECT_THROW(VoteAndVerify(options, matches), std::invalid_argument);
    } else {
      EXPECT_EQ(VoteAndVerify(options, matches), 4);
    }
  }
}

TEST(VoteAndVerify, VotingClampsLargeBinEndpointsBeforeConversion) {
  constexpr int kLargeBins = std::numeric_limits<int>::max() - 1;
  struct TestCase {
    int trans_bins;
    int scale_bins;
    int angle_bins;
    float tx;
    float ty;
    float scale;
    float angle;
  };
  const TestCase test_cases[] = {
      {kLargeBins, 2, 2, 4096, 0, 1, 0},
      {kLargeBins, 2, 2, 0, 4096, 1, 0},
      {2, kLargeBins, 2, 0, 0, 10, 0},
      {2, 2, kLargeBins, 0, 0, 1, static_cast<float>(EIGEN_PI)},
  };
  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(::testing::Message()
                 << test_case.trans_bins << ", " << test_case.scale_bins << ", "
                 << test_case.angle_bins << ", " << test_case.tx << ", "
                 << test_case.ty);
    VoteAndVerifyOptions options;
    options.num_trans_bins = test_case.trans_bins;
    options.num_scale_bins = test_case.scale_bins;
    options.num_angle_bins = test_case.angle_bins;
    options.local_optimization = false;
    options.eff_inlier_count = false;
    // These sparse grids fit 64-bit keys but exceed the 32-bit key domain.
    if (std::numeric_limits<size_t>::digits < 64) {
      EXPECT_THROW(VoteAndVerify(options, {}), std::invalid_argument);
      continue;
    }
    std::vector<FeatureGeometryMatch> matches;
    const float cosine = std::cos(test_case.angle);
    const float sine = std::sin(test_case.angle);
    for (const float y : {0.0f, 1.0f}) {
      for (const float x : {0.0f, 1.0f}) {
        FeatureGeometryMatch match;
        match.geometry1.x = x;
        match.geometry1.y = y;
        match.geometry1.scale = 1;
        match.geometry2.x =
            test_case.scale * (cosine * x - sine * y) + test_case.tx;
        match.geometry2.y =
            test_case.scale * (sine * x + cosine * y) + test_case.ty;
        match.geometry2.scale = test_case.scale;
        match.geometry2.orientation = test_case.angle;
        matches.push_back(match);
      }
    }
    EXPECT_EQ(VoteAndVerify(options, matches), 4);
  }
}

TEST(VoteAndVerify, VotingValidatesMaximumHistogramKey) {
  constexpr int kRadix =
      static_cast<int>(size_t{1} << (std::numeric_limits<size_t>::digits / 4));
  VoteAndVerifyOptions options;
  options.num_trans_bins = kRadix;
  options.num_scale_bins = kRadix;
  options.num_angle_bins = kRadix;
  // Empty input exercises the option contract without allocating histogram
  // bins.
  EXPECT_EQ(VoteAndVerify(options, {}), 0);
  options.num_angle_bins += 2;
  EXPECT_THROW(VoteAndVerify(options, {}), std::invalid_argument);
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap
