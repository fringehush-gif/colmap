// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_cuda.h"

#include "colmap/util/cuda.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(PatchMatchCuda, ReferenceCameraIsOwnedByEachLaunch) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "No CUDA or HIP device available";
  }

  constexpr int kSize = 64;
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<Image> images;
  std::vector<DepthMap> depths;
  std::vector<NormalMap> normals;
  for (int i = 0; i < 3; ++i) {
    const float K[9] = {i == 2 ? 95.0f : 60.0f,
                        0,
                        i == 2 ? 25.0f : 32.0f,
                        0,
                        i == 2 ? 85.0f : 70.0f,
                        30.0f,
                        0,
                        0,
                        1};
    const float T[3] = {0.2f * i, 0, 0};
    images.emplace_back("", kSize, kSize, K, R, T);
    Bitmap bitmap(kSize, kSize, false);
    for (int y = 0; y < kSize; ++y) {
      for (int x = 0; x < kSize; ++x) {
        bitmap.RowMajorData()[y * kSize + x] = (19 * x + 37 * y + x * y) % 256;
      }
    }
    images.back().SetBitmap(std::move(bitmap));
    depths.emplace_back(kSize, kSize, 1.0f, 4.0f);
    depths.back().Fill(2.0f);
    normals.emplace_back(kSize, kSize);
    normals.back().Fill(0.0f);
    for (int y = 0; y < kSize; ++y) {
      for (int x = 0; x < kSize; ++x) {
        normals.back().Set(y, x, 2, -1.0f);
      }
    }
  }

  PatchMatchOptions options;
  options.gpu_index = "0";
  options.depth_min = 1.0f;
  options.depth_max = 4.0f;
  options.window_radius = 1;
  options.sigma_spatial = 1.0f;
  options.num_samples = 3;
  options.num_iterations = 1;
  options.filter = false;
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {1};
  problem.images = &images;
  problem.depth_maps = &depths;
  problem.normal_maps = &normals;
  auto other_problem = problem;
  other_problem.ref_image_idx = 2;

  for (const bool geom_consistency : {false, true}) {
    SCOPED_TRACE(geom_consistency);
    options.geom_consistency = geom_consistency;
    PatchMatchCuda baseline(options, problem);
    baseline.Run();
    const auto expected_depth = baseline.GetDepthMap();
    const auto expected_normal = baseline.GetNormalMap();
    const auto expected_prob = baseline.GetSelProbMap();
    ASSERT_EQ(expected_depth.GetData().size(), kSize * kSize);
    ASSERT_TRUE(std::all_of(expected_depth.GetData().begin(),
                            expected_depth.GetData().end(),
                            [](float depth) { return std::isfinite(depth); }));

    ASSERT_TRUE(std::any_of(expected_depth.GetData().begin(),
                            expected_depth.GetData().end(),
                            [](float depth) { return depth > 0; }));

    for (const bool interleave : {false, true}) {
      SCOPED_TRACE(interleave);
      PatchMatchCuda candidate(options, problem);
      // Another worker may construct its owner between construction and Run.
      std::unique_ptr<PatchMatchCuda> other;
      if (interleave) {
        other = std::make_unique<PatchMatchCuda>(options, other_problem);
      }
      candidate.Run();
      EXPECT_EQ(candidate.GetDepthMap().GetData(), expected_depth.GetData());
      EXPECT_EQ(candidate.GetNormalMap().GetData(), expected_normal.GetData());
      EXPECT_EQ(candidate.GetSelProbMap().GetData(), expected_prob.GetData());
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
