// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/image.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/sensor/bitmap.h"

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

// A black reference has zero variance, hence NCC cost 2 for every hypothesis.
// Compute its selected-state density independently in host double precision.
double BlackImageEmission(const float sigma) {
  const double s = sigma;
  const double integral =
      s * std::sqrt(std::acos(-1.0) / 2) * std::erf(std::sqrt(2.0) / s);
  return std::exp(-2 / (s * s)) / integral;
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
                          (first ? emission : 0.5) *
                          (second ? emission : 0.5);
      total += mass;
      if (first) {
        selected += mass;
      }
    }
  }
  return selected / total;
}

void ExpectBlackSquarePosteriors(const int side,
                                double (*posterior)(double)) {
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
    const double expected = posterior(BlackImageEmission(sigma));
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

}  // namespace
}  // namespace mvs
}  // namespace colmap
