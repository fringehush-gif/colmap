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

// A black image has zero reference variance, hence NCC cost 2 for every
// hypothesis and source. A one-pixel chain has no neighboring emissions:
// its posterior is the selected-state density divided by that density + 1/2.
// Compute the truncated Gaussian density independently in host double precision.
// This runs the actual solver, including backward storage, all four sweeps,
// temporal blending, and the published selection map.
TEST_F(PatchMatchHmmCuda, SingletonSweepCountsCurrentEmissionOnce) {
  const float K[9] = {8, 0, 0.5f, 0, 11, 0.5f, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<Image> images;
  for (int image_idx = 0; image_idx < 3; ++image_idx) {
    const float T[3] = {-0.1f * image_idx, 0, 0};
    Image image("hmm-singleton-" + std::to_string(image_idx), 1, 1, K, R, T);
    Bitmap bitmap(1, 1, false);
    bitmap.Fill(BitmapColor<uint8_t>(0));
    image.SetBitmap(std::move(bitmap));
    images.push_back(std::move(image));
  }
  PatchMatch::Problem problem;
  problem.images = &images;
  problem.ref_image_idx = 0;
  for (const float sigma : {0.6f, 1.2f}) {
    const double s = sigma;
    const double integral =
        s * std::sqrt(std::acos(-1.0) / 2) * std::erf(std::sqrt(2.0) / s);
    const double emission = std::exp(-2 / (s * s)) / integral;
    const double expected = emission / (emission + 0.5);
    ASSERT_GT(expected, 0.001);  // A zero-output implementation must fail.
    for (const int radius : {2, 5}) {
      for (const std::vector<int>& sources :
           {std::vector<int>{1}, std::vector<int>{2, 1}}) {
        SCOPED_TRACE(::testing::Message()
                     << "sigma=" << sigma << " radius=" << radius
                     << " sources=" << sources.size());
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
        ASSERT_EQ(selection.GetWidth(), 1);
        ASSERT_EQ(selection.GetHeight(), 1);
        ASSERT_EQ(selection.GetDepth(), sources.size());
        ASSERT_EQ(selection.GetData().size(), sources.size());
        for (const float probability : selection.GetData()) {
          ASSERT_TRUE(std::isfinite(probability));
          EXPECT_NEAR(probability, expected, 2e-5);
        }
      }
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
