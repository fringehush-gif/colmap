// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

struct Output {
  DepthMap depth;
  NormalMap normal;
  Mat<float> probabilities;
  std::vector<int> graph;
};

Output RunPatchMatch(const PatchMatchOptions& options,
                     const PatchMatch::Problem& problem) {
  PatchMatch(options, problem).Check();
  PatchMatchCuda owner(options, problem);
  owner.Run();
  return {owner.GetDepthMap(),
          owner.GetNormalMap(),
          owner.GetSelProbMap(),
          owner.GetConsistentImageIdxs()};
}

void ExpectEqual(const Output& expected, const Output& actual) {
  const std::array<const Mat<float>*, 3> expected_maps = {
      &expected.depth, &expected.normal, &expected.probabilities};
  const std::array<const Mat<float>*, 3> actual_maps = {
      &actual.depth, &actual.normal, &actual.probabilities};
  for (size_t i = 0; i < expected_maps.size(); ++i) {
    ASSERT_GT(expected_maps[i]->GetNumBytes(), 0);
    ASSERT_EQ(expected_maps[i]->GetWidth(), actual_maps[i]->GetWidth());
    ASSERT_EQ(expected_maps[i]->GetHeight(), actual_maps[i]->GetHeight());
    ASSERT_EQ(expected_maps[i]->GetDepth(), actual_maps[i]->GetDepth());
    EXPECT_EQ(std::memcmp(expected_maps[i]->GetPtr(),
                          actual_maps[i]->GetPtr(),
                          expected_maps[i]->GetNumBytes()),
              0);
  }
  EXPECT_EQ(expected.graph, actual.graph);
}

TEST(PatchMatchSourceDepth, PreservesGeometricOutput) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "No CUDA/HIP device available";
  }

  std::vector<Image> images;
  std::vector<DepthMap> depths;
  std::vector<NormalMap> normals;
  std::vector<std::shared_ptr<const DepthMap>> leases;
  const float K[9] = {40, 0, 8, 0, 40, 4, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int view = 0; view < 3; ++view) {
    const int width = 17 + 2 * view;
    const int height = 9 + view;
    const float T[3] = {-0.05f * view, 0, 0};
    images.emplace_back("", width, height, K, R, T);
    Bitmap bitmap(width, height, false);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const uint8_t value = ((x + view) * 37 + y * 53) % 251;
        bitmap.SetPixel(x, y, BitmapColor<uint8_t>(value));
      }
    }
    images.back().SetBitmap(std::move(bitmap));
    depths.emplace_back(width, height, 1, 3);
    depths.back().Fill(2.0f);
    normals.emplace_back(width, height);
    normals.back().Fill(0.0f);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) normals.back().Set(y, x, 2, -1.0f);
    }
    leases.push_back(std::make_shared<DepthMap>(depths.back()));
  }

  PatchMatchOptions options;
  options.geom_consistency = true;
  options.gpu_index = "0";
  options.depth_min = 1;
  options.depth_max = 3;
  options.window_radius = 1;
  options.sigma_spatial = 1;
  options.num_samples = 3;
  options.num_iterations = 2;
  options.filter_min_ncc = -1;
  options.filter_min_triangulation_angle = 0;
  options.filter_geom_consistency_max_cost = 1000;
  options.filter_min_num_consistent = 1;
  options.write_consistency_graph = true;
  for (const bool filter : {false, true}) {
    options.filter = filter;
    SCOPED_TRACE(filter);
    for (int ref = 0; ref < 3; ++ref) {
      SCOPED_TRACE(ref);
      PatchMatch::Problem copied;
      copied.ref_image_idx = ref;
      copied.images = &images;
      copied.depth_maps = &depths;
      copied.normal_maps = &normals;
      for (int source = 0; source < 3; ++source) {
        if (source != ref) copied.src_image_idxs.push_back(source);
      }
      const Output expected = RunPatchMatch(options, copied);
      ASSERT_TRUE(std::any_of(
          expected.depth.GetData().begin(),
          expected.depth.GetData().end(),
          [](const float depth) { return std::isfinite(depth) && depth > 0; }));
      if (filter) {
        ASSERT_FALSE(expected.graph.empty());
      }
      ExpectEqual(expected, RunPatchMatch(options, copied));

      std::vector<DepthMap> owned_depths(3);
      owned_depths[ref] = depths[ref];
      auto shared = copied;
      shared.depth_maps = &owned_depths;
      shared.source_depth_leases = &leases;
      ExpectEqual(expected, RunPatchMatch(options, shared));
    }
  }
  for (size_t i = 0; i < leases.size(); ++i) {
    EXPECT_EQ(
        std::memcmp(
            leases[i]->GetPtr(), depths[i].GetPtr(), depths[i].GetNumBytes()),
        0);
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
