// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"

#include <memory>
#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

struct PatchMatchInputs {
  PatchMatchInputs() : normal_maps(3) {
    const float K[] = {10, 0, 1, 0, 10, 1, 0, 0, 1};
    const float R[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float T[] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
      const int width = 3 + i;
      const int height = 5 + i;
      images.emplace_back("", width, height, K, R, T);
      images.back().SetBitmap(Bitmap(width, height, false));
      depth_maps.emplace_back(width, height, 1, 3);
    }
    depth_maps[2] = DepthMap();
    normal_maps[1] = NormalMap(4, 6);
    options.geom_consistency = true;
    options.gpu_index = "0";
    problem.ref_image_idx = 1;
    problem.src_image_idxs = {0};
    problem.images = &images;
    problem.depth_maps = &depth_maps;
    problem.normal_maps = &normal_maps;
  }

  PatchMatchOptions options;
  PatchMatch::Problem problem;
  std::vector<Image> images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
};

TEST(PatchMatch, CheckNormalMapChannels) {
  PatchMatchInputs inputs;
  ASSERT_EQ(inputs.normal_maps[0].GetNumBytes(), 0);
  ASSERT_NO_THROW(PatchMatch(inputs.options, inputs.problem).Check());

  for (const size_t depth : {1, 2, 4}) {
    SCOPED_TRACE(depth);
    static_cast<Mat<float>&>(inputs.normal_maps[1]) = Mat<float>(4, 6, depth);
    EXPECT_THAT([&] { PatchMatch(inputs.options, inputs.problem).Check(); },
                testing::ThrowsMessage<std::invalid_argument>(
                    testing::HasSubstr("ref_normal_map.GetDepth() == 3")));
  }
}

TEST(PatchMatch, CheckDepthMapChannels) {
  PatchMatchInputs inputs;
  ASSERT_EQ(inputs.depth_maps[2].GetNumBytes(), 0);
  ASSERT_NO_THROW(PatchMatch(inputs.options, inputs.problem).Check());

  for (const int image_idx : {0, 1}) {
    SCOPED_TRACE(image_idx);
    const DepthMap previous = inputs.depth_maps[image_idx];
    for (const size_t depth : {2, 3, 4}) {
      SCOPED_TRACE(depth);
      static_cast<Mat<float>&>(inputs.depth_maps[image_idx]) =
          Mat<float>(previous.GetWidth(), previous.GetHeight(), depth);
      EXPECT_THAT([&] { PatchMatch(inputs.options, inputs.problem).Check(); },
                  testing::ThrowsMessage<std::invalid_argument>(
                      testing::HasSubstr("depth_map.GetDepth() == 1")));
    }
    inputs.depth_maps[image_idx] = previous;
  }
}

TEST(PatchMatch, CheckSourceDepthLeaseChannels) {
  PatchMatchInputs inputs;
  std::vector<std::shared_ptr<const DepthMap>> leases(3);
  inputs.problem.source_depth_leases = &leases;
  auto depth_map = std::make_shared<DepthMap>(3, 5, 1, 3);
  leases[0] = depth_map;
  ASSERT_NO_THROW(PatchMatch(inputs.options, inputs.problem).Check());
  for (const size_t depth : {2, 3, 4}) {
    SCOPED_TRACE(depth);
    static_cast<Mat<float>&>(*depth_map) = Mat<float>(3, 5, depth);
    EXPECT_THAT([&] { PatchMatch(inputs.options, inputs.problem).Check(); },
                testing::ThrowsMessage<std::invalid_argument>(
                    testing::HasSubstr("depth_map.GetDepth() == 1")));
  }
}

TEST(PatchMatch, CheckPhotometricWithoutNormalMaps) {
  PatchMatchInputs inputs;
  inputs.options.geom_consistency = false;
  inputs.problem.depth_maps = nullptr;
  inputs.problem.normal_maps = nullptr;
  EXPECT_NO_THROW(PatchMatch(inputs.options, inputs.problem).Check());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
