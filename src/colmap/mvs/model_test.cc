// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/model.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(Model, ReadCOLMAP) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 10;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  std::filesystem::create_directories(sparse_path);
  reconstruction.WriteBinary(sparse_path);

  Model model;
  model.ReadFromCOLMAP(test_dir);
  EXPECT_EQ(model.images.size(), reconstruction.NumRegImages());
  EXPECT_EQ(model.points.size(), reconstruction.NumPoints3D());

  Model model_lower;
  model_lower.Read(test_dir, "colmap");
  EXPECT_EQ(model_lower.images.size(), reconstruction.NumRegImages());
  EXPECT_EQ(model_lower.points.size(), reconstruction.NumPoints3D());

  // Verify case insensitivity.
  Model model_upper;
  model_upper.Read(test_dir, "COLMAP");
  EXPECT_EQ(model_upper.images.size(), reconstruction.NumRegImages());
  EXPECT_EQ(model_upper.points.size(), reconstruction.NumPoints3D());

  const std::string name = model.GetImageName(0);
  EXPECT_FALSE(name.empty());
  EXPECT_EQ(model.GetImageIdx(name), 0);

  EXPECT_THROW(model.GetImageIdx("nonexistent"), std::exception);
  EXPECT_THROW(model.GetImageName(-1), std::exception);
  EXPECT_THROW(model.GetImageName(model.images.size()), std::exception);
}

TEST(Model, ComputeSharedPoints) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  // Projection center = -R^T * T = -T (for R=I).
  const float T1[3] = {0, 0, 0};
  const float T2[3] = {-1, 0, 0};
  const float T3[3] = {-2, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T1);
  model.images.emplace_back("img1.jpg", 100, 100, K, R, T2);
  model.images.emplace_back("img2.jpg", 100, 100, K, R, T3);
  model.points.emplace_back(Model::Point{5.0f, 0.0f, 10.0f, {0, 1}});

  // Point seen by images 0, 1, and 2.
  model.points.emplace_back(Model::Point{6.0f, 0.0f, 10.0f, {0, 1, 2}});

  const std::vector<std::map<int, int>> shared = model.ComputeSharedPoints();
  EXPECT_EQ(shared.size(), 3);
  EXPECT_EQ(shared[0].at(1), 2);  // images 0,1 share 2 points
  EXPECT_EQ(shared[1].at(0), 2);
  EXPECT_EQ(shared[0].at(2), 1);  // images 0,2 share 1 point
  EXPECT_EQ(shared[2].at(0), 1);
  EXPECT_EQ(shared[1].at(2), 1);  // images 1,2 share 1 point
  EXPECT_EQ(shared[2].at(1), 1);
}

TEST(Model, ComputeDepthRanges) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T);

  for (int i = 1; i <= 100; ++i) {
    model.points.emplace_back(
        Model::Point{0.0f, 0.0f, static_cast<float>(i), {0}});
  }

  const std::vector<std::pair<float, float>> depth_ranges =
      model.ComputeDepthRanges();
  EXPECT_EQ(depth_ranges.size(), 1);

  // The points span depths in the range [1..100] and
  // the range is computed at percentiles 1% and 99%
  // with some additional padding.
  EXPECT_FLOAT_EQ(depth_ranges[0].first, 1.5f);
  EXPECT_FLOAT_EQ(depth_ranges[0].second, 125.0f);
}

TEST(Model, ComputeDepthRangesNoPoints) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T);

  const std::vector<std::pair<float, float>> depth_ranges =
      model.ComputeDepthRanges();
  EXPECT_EQ(depth_ranges.size(), 1);
  EXPECT_FLOAT_EQ(depth_ranges[0].first, -1.0f);
  EXPECT_FLOAT_EQ(depth_ranges[0].second, -1.0f);
}

TEST(Model, ComputeTriangulationAngles) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  // Projection center = -R^T * T = -T (for R=I).
  // Place cameras at (0,0,0) and (1,0,0).
  const float T1[3] = {0, 0, 0};
  const float T2[3] = {-1, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T1);
  model.images.emplace_back("img1.jpg", 100, 100, K, R, T2);

  // Point at midpoint between cameras, at depth 10.
  model.points.emplace_back(Model::Point{0.5f, 0.0f, 10.0f, {0, 1}});

  const std::vector<std::map<int, float>> angles =
      model.ComputeTriangulationAngles(50);
  EXPECT_EQ(angles.size(), 2);
  const float expected_angle = 2.0f * std::atan(0.5f / 10.0f);
  EXPECT_NEAR(angles[0].at(1), expected_angle, 1e-5f);
  // Angles should be symmetric.
  EXPECT_FLOAT_EQ(angles[0].at(1), angles[1].at(0));

  // Repeated observations and both pair orientations retain every sample.
  model.points = {{0.5f, 0, 10, {1, 0, 1}},
                  {0.75f, 0.5f, 3, {0, 1}},
                  {-0.25f, 0.25f, 2, {1, 0}}};
  const auto angle = [&model](const size_t point_idx) {
    const auto& point = model.points[point_idx];
    return static_cast<float>(CalculateTriangulationAngle(
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::UnitX(),
        Eigen::Vector3d(point.x, point.y, point.z)));
  };
  std::vector<float> samples = {angle(0), angle(0), angle(1), angle(2)};
  const float p75 = Percentile(samples, 75);
  EXPECT_EQ(model.ComputeTriangulationAngles(75),
            (std::vector<std::map<int, float>>{{{1, p75}}, {{0, p75}}}));
}

TEST(Model, GetMaxOverlappingImages) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  // Projection center = -R^T * T = -T (for R=I).
  const float T1[3] = {0, 0, 0};
  const float T2[3] = {-1, 0, 0};
  const float T3[3] = {-2, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T1);
  model.images.emplace_back("img1.jpg", 100, 100, K, R, T2);
  model.images.emplace_back("img2.jpg", 100, 100, K, R, T3);
  for (int i = 0; i < 10; ++i) {
    model.points.emplace_back(Model::Point{0.5f, 0.0f, 5.0f + i, {0, 1}});
  }
  model.points.emplace_back(Model::Point{1.0f, 0.0f, 10.0f, {0, 2}});

  const std::vector<std::vector<int>> overlapping =
      model.GetMaxOverlappingImages(2, 0.0);
  EXPECT_EQ(overlapping.size(), 3);
  // Image 0 should have image 1 as top overlap.
  EXPECT_FALSE(overlapping[0].empty());
  EXPECT_EQ(overlapping[0][0], 1);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
