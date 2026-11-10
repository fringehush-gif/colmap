// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/sensor/bitmap.h"

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

class PatchMatchHmmCuda : public ::testing::Test {
 protected:
  void SetUp() override {
    int num_devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&num_devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && num_devices == 0)) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

// Compute the declared cost density independently in host double precision.
double NccEmission(const double cost, const float sigma) {
  const double s = sigma;
  const double integral =
      s * std::sqrt(std::acos(-1.0) / 2) * std::erf(std::sqrt(2.0) / s);
  return std::exp(-cost * cost / (2 * s * s)) / integral;
}

double SingletonPosterior(const double emission) {
  return emission / (emission + 0.5);
}

double TwoPixelPosterior(const double emission) {
  // The model keeps its state with probability 0.99999f. Preserve the declared
  // float transition values before promoting them to the host oracle's double.
  const double stay = 0.99999f;
  const double change = 1.0f - 0.99999f;
  double total = 0;
  double selected = 0;
  // Enumerate all four assignments, rather than copy the message recurrence.
  for (const bool first : {false, true}) {
    for (const bool second : {false, true}) {
      const double mass = 0.5 * (first == second ? stay : change) *
                          (first ? emission : 0.5) * (second ? emission : 0.5);
      total += mass;
      if (first) {
        selected += mass;
      }
    }
  }
  return selected / total;
}

void ExpectBlackSquarePosteriors(const int side, double (*posterior)(double)) {
  constexpr double kProbabilityTolerance = 2e-5;
  const float K[9] = {8, 0, 0.5f, 0, 11, 0.5f, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<Image> images;
  for (int image_idx = 0; image_idx < 3; ++image_idx) {
    const float T[3] = {-0.1f * image_idx, 0, 0};
    Image image("hmm-black-" + std::to_string(image_idx), side, side, K, R, T);
    Bitmap bitmap(side, side, false);
    bitmap.Fill(BitmapColor<uint8_t>(0));
    image.SetBitmap(std::move(bitmap));
    images.push_back(std::move(image));
  }
  PatchMatch::Problem problem;
  problem.images = &images;
  problem.ref_image_idx = 0;
  for (const float sigma : {0.6f, 1.2f}) {
    const double expected = posterior(NccEmission(2, sigma));
    ASSERT_GT(expected, kProbabilityTolerance);  // Zero outputs must fail.
    for (const int radius : {2, 5}) {
      for (const std::vector<int>& sources :
           {std::vector<int>{1}, std::vector<int>{2, 1}}) {
        SCOPED_TRACE(::testing::Message()
                     << "side=" << side << " sigma=" << sigma
                     << " radius=" << radius << " sources=" << sources.size());
        problem.src_image_idxs = sources;
        PatchMatchOptions options;
        options.gpu_index = "0";
        options.depth_min = 1;
        options.depth_max = 3.5f;
        options.window_radius = radius;
        options.window_step = 1;
        options.sigma_spatial = 3;
        options.ncc_sigma = sigma;
        options.num_iterations = 1;
        options.num_samples = 1;
        options.geom_consistency = false;
        options.filter = false;
        PatchMatch patch_match(options, problem);
        patch_match.Run();
        const auto selection = patch_match.GetSelProbMap();
        ASSERT_EQ(selection.GetWidth(), side);
        ASSERT_EQ(selection.GetHeight(), side);
        ASSERT_EQ(selection.GetDepth(), sources.size());
        ASSERT_EQ(selection.GetData().size(), side * side * sources.size());
        for (const float probability : selection.GetData()) {
          ASSERT_TRUE(std::isfinite(probability));
          EXPECT_NEAR(probability, expected, kProbabilityTolerance);
        }
      }
    }
  }
}

TEST_F(PatchMatchHmmCuda, SingletonSweepCountsCurrentEmissionOnce) {
  ExpectBlackSquarePosteriors(1, SingletonPosterior);
}

// Every rotated sweep of a black 2x2 square has a symmetric two-node chain.
// Both positions have the same marginal, so rotations and convex blends retain
// that value. This tests multi-row storage without copying the sweep schedule.
TEST_F(PatchMatchHmmCuda, TwoByTwoSweepsMatchEnumeratedPairPosterior) {
  ExpectBlackSquarePosteriors(2, TwoPixelPosterior);
}

// Matched coincident views keep the ideal warp independent of the hypothesis.
// A singleton is bright; larger squares contain only a lower-right impulse.
Mat<float> IdentitySquareSelection(const int side, const float sigma) {
  const float K[9] = {1, 0, 0.5f, 0, 1, 0.5f, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::vector<Image> images;
  std::vector<DepthMap> depths;
  std::vector<NormalMap> normals;
  for (int index = 0; index < 2; ++index) {
    Image image("hmm-identity-" + std::to_string(index), side, side, K, R, T);
    Bitmap bitmap(side, side, false);
    bitmap.Fill(BitmapColor<uint8_t>(0));
    bitmap.SetPixel(side - 1, side - 1, BitmapColor<uint8_t>(255));
    image.SetBitmap(std::move(bitmap));
    images.push_back(std::move(image));
    depths.emplace_back(side, side, 1, 3.5f);
    normals.emplace_back(side, side);
    for (int row = 0; row < side; ++row) {
      for (int col = 0; col < side; ++col) {
        depths.back().Set(row, col, 2);
        normals.back().Set(row, col, 0, 0);
        normals.back().Set(row, col, 1, 0);
        normals.back().Set(row, col, 2, -1);
      }
    }
  }
  PatchMatch::Problem problem;
  problem.images = &images;
  problem.depth_maps = &depths;
  problem.normal_maps = &normals;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {1};
  PatchMatchOptions options;
  options.gpu_index = "0";
  options.depth_min = 1;
  options.depth_max = 3.5f;
  options.window_radius = 2;
  options.window_step = 1;
  options.sigma_spatial = 3;
  options.sigma_color = 1;
  options.ncc_sigma = sigma;
  options.min_triangulation_angle = 0;
  options.num_iterations = 1;
  options.num_samples = 1;
  options.geom_consistency = true;
  options.geom_consistency_regularizer = 0;
  options.filter = false;
  PatchMatch patch_match(options, problem);
  patch_match.Run();
  return patch_match.GetSelProbMap();
}

// Enumerate all 16 binary chains, independently of the message recurrence.
double FourPixelPosterior(const std::array<double, 4>& emissions,
                          const unsigned position) {
  const double stay = 0.99999f;
  const double change = 1.0f - 0.99999f;
  double total = 0;
  double selected = 0;
  for (unsigned assignment = 0; assignment < 16; ++assignment) {
    double mass = 0.5;
    for (unsigned node = 0; node < 4; ++node) {
      const bool state = (assignment & (1u << node)) != 0;
      if (node > 0) {
        const bool previous = (assignment & (1u << (node - 1u))) != 0;
        mass *= state == previous ? stay : change;
      }
      mass *= state ? emissions[node] : 0.5;
    }
    total += mass;
    if ((assignment & (1u << position)) != 0) {
      selected += mass;
    }
  }
  return selected / total;
}

// Retained bright-singleton control: matched pixels and texture borders give
// a nonconstant patch and ideal NCC cost zero with public geometric inputs.
TEST_F(PatchMatchHmmCuda, IdentityBrightSingletonControl) {
  for (const float sigma : {0.6f, 1.2f}) {
    SCOPED_TRACE(sigma);
    const auto selection = IdentitySquareSelection(1, sigma);
    ASSERT_EQ(selection.GetWidth(), 1);
    ASSERT_EQ(selection.GetHeight(), 1);
    ASSERT_EQ(selection.GetDepth(), 1);
    ASSERT_EQ(selection.GetData().size(), 1);
    ASSERT_TRUE(std::isfinite(selection.GetData().front()));
    EXPECT_NEAR(selection.GetData().front(),
                SingletonPosterior(NccEmission(0, sigma)),
                2e-5);
  }
}

// Row/column zero exclude the impulse, giving ideal cost two; all other
// patches contain it and have ideal cost zero. At each checked diagonal pixel,
// horizontal/vertical chains are [2, 0, 0, 0]. Symmetric transitions and
// uniform boundaries give the same marginal in reverse, so every sweep's convex
// blend retains that value. No rotated arrays or temporal weights enter the
// oracle.
TEST_F(PatchMatchHmmCuda, CornerImpulseSweepsMatchEnumeratedPosteriors) {
  constexpr double kProbabilityTolerance = 2e-5;
  for (const float sigma : {0.6f, 1.2f}) {
    SCOPED_TRACE(::testing::Message() << "side=4 sigma=" << sigma);
    const auto selection = IdentitySquareSelection(4, sigma);
    ASSERT_EQ(selection.GetWidth(), 4);
    ASSERT_EQ(selection.GetHeight(), 4);
    ASSERT_EQ(selection.GetDepth(), 1);
    ASSERT_EQ(selection.GetData().size(), 16);
    for (const float probability : selection.GetData()) {
      ASSERT_TRUE(std::isfinite(probability));
    }
    const std::array<double, 4> emissions = {NccEmission(2, sigma),
                                             NccEmission(0, sigma),
                                             NccEmission(0, sigma),
                                             NccEmission(0, sigma)};
    for (unsigned position = 1; position < 4; ++position) {
      SCOPED_TRACE(::testing::Message() << "diagonal=" << position);
      const double expected = FourPixelPosterior(emissions, position);
      ASSERT_GT(expected, kProbabilityTolerance);  // Zero outputs must fail.
      ASSERT_GT(std::abs(expected - SingletonPosterior(emissions[position])),
                kProbabilityTolerance);  // Ignoring the chain must fail.
      EXPECT_NEAR(
          selection.Get(position, position), expected, kProbabilityTolerance);
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
