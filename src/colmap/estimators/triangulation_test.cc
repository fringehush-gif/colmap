// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/triangulation.h"

#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

std::vector<Rigid3d> PerspectivePoses() {
  const Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  return {Rigid3d(rotation, Eigen::Vector3d::Zero()),
          Rigid3d(rotation, Eigen::Vector3d(-0.6, 0.0, 0.0)),
          Rigid3d(rotation, Eigen::Vector3d(0.0, -0.6, 0.0)),
          Rigid3d(rotation, Eigen::Vector3d(-0.5, -0.4, 0.0))};
}

std::vector<Eigen::Vector2d> ProjectObservations(
    const Camera& camera,
    const std::vector<Rigid3d>& cams_from_world,
    const Eigen::Vector3d& xyz) {
  std::vector<Eigen::Vector2d> points;
  points.reserve(cams_from_world.size());
  for (const Rigid3d& cam_from_world : cams_from_world) {
    points.push_back(camera.ImgFromCam(cam_from_world * xyz).value());
  }
  return points;
}

TEST(EstimateTriangulation, PreservesValidObservationMask) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 800.0, 1000, 800);
  const auto poses = PerspectivePoses();
  const Eigen::Vector3d expected_xyz(0.2, 0.1, 4.0);
  const auto points = ProjectObservations(camera, poses, expected_xyz);
  const std::vector<const Camera*> cameras(poses.size(), &camera);
  std::vector<char> mask;
  Eigen::Vector3d xyz;
  ASSERT_TRUE(EstimateTriangulation(
      EstimateTriangulationOptions(), points, poses, cameras, &mask, &xyz));
  ASSERT_EQ(mask.size(), points.size());
  for (const char inlier : mask) EXPECT_TRUE(inlier);
  EXPECT_LT((xyz - expected_xyz).norm(), 1e-5);
}

TEST(EstimateTriangulation, SkipsInvalidBearingsAndRestoresOriginalMask) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 800.0, 1000, 800);
  const auto poses = PerspectivePoses();
  const Eigen::Vector3d expected_xyz(0.2, 0.1, 4.0);
  const auto valid_points = ProjectObservations(camera, poses, expected_xyz);
  const std::vector<const Camera*> cameras(poses.size(), &camera);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const size_t invalid_idx : {size_t(0), size_t(1), size_t(3)}) {
    auto points = valid_points;
    points[invalid_idx] = Eigen::Vector2d(nan, 100.0);
    std::vector<char> mask;
    Eigen::Vector3d xyz;
    ASSERT_TRUE(EstimateTriangulation(
        EstimateTriangulationOptions(), points, poses, cameras, &mask, &xyz));
    ASSERT_EQ(mask.size(), points.size());
    for (size_t i = 0; i < mask.size(); ++i) {
      EXPECT_EQ(static_cast<bool>(mask[i]), i != invalid_idx);
    }
    EXPECT_LT((xyz - expected_xyz).norm(), 1e-5);
  }
}

TEST(EstimateTriangulation, SkipsFailedCameraUnprojection) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 800.0, 1000, 800);
  Camera unprojectable_camera =
      Camera::CreateFromModelId(2, CameraModelId::kOpenCV, 100.0, 100, 200);
  unprojectable_camera.params[4] = -0.5;
  unprojectable_camera.params[5] = 0.5;
  unprojectable_camera.params[6] = -0.5;
  const Eigen::Vector2d unprojectable(50.0, 150.0);
  ASSERT_FALSE(unprojectable_camera.CamRayFromImg(unprojectable).has_value());
  const auto poses = PerspectivePoses();
  const Eigen::Vector3d expected_xyz(0.2, 0.1, 4.0);
  auto points = ProjectObservations(camera, poses, expected_xyz);
  points[1] = unprojectable;
  std::vector<const Camera*> cameras(poses.size(), &camera);
  cameras[1] = &unprojectable_camera;
  std::vector<char> mask;
  Eigen::Vector3d xyz;
  ASSERT_TRUE(EstimateTriangulation(
      EstimateTriangulationOptions(), points, poses, cameras, &mask, &xyz));
  ASSERT_EQ(mask.size(), points.size());
  EXPECT_FALSE(mask[1]);
  EXPECT_TRUE(mask[0]);
  EXPECT_TRUE(mask[2]);
  EXPECT_TRUE(mask[3]);
  EXPECT_LT((xyz - expected_xyz).norm(), 1e-5);
}

TEST(EstimateTriangulation, RejectsFewerThanTwoValidBearings) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 800.0, 1000, 800);
  const auto poses = PerspectivePoses();
  const Eigen::Vector3d expected_xyz(0.2, 0.1, 4.0);
  auto points = ProjectObservations(camera, poses, expected_xyz);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (size_t i = 0; i < 3; ++i) points[i] = Eigen::Vector2d(nan, 100.0);
  const std::vector<const Camera*> cameras(poses.size(), &camera);
  std::vector<char> mask;
  Eigen::Vector3d xyz;
  EXPECT_FALSE(EstimateTriangulation(
      EstimateTriangulationOptions(), points, poses, cameras, &mask, &xyz));
}

TEST(EstimateTriangulation, KeepsValidSphericalBackHemisphereBearings) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, 0.0, 1000, 500);
  const Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  const std::vector<Rigid3d> poses = {
      Rigid3d(rotation, Eigen::Vector3d::Zero()),
      Rigid3d(rotation, Eigen::Vector3d(-1.0, 0.0, 0.0))};
  const Eigen::Vector3d expected_xyz(0.2, 0.1, -4.0);
  const auto points = ProjectObservations(camera, poses, expected_xyz);
  for (const auto& point : points) {
    const auto bearing = camera.CamRayFromImg(point);
    ASSERT_TRUE(bearing.has_value());
    EXPECT_LT(bearing->z(), 0.0);
  }
  const std::vector<const Camera*> cameras(poses.size(), &camera);
  std::vector<char> mask;
  Eigen::Vector3d xyz;
  ASSERT_TRUE(EstimateTriangulation(
      EstimateTriangulationOptions(), points, poses, cameras, &mask, &xyz));
  ASSERT_EQ(mask.size(), points.size());
  EXPECT_TRUE(mask[0]);
  EXPECT_TRUE(mask[1]);
  EXPECT_LT((xyz - expected_xyz).norm(), 1e-4);
}

}  // namespace
}  // namespace colmap
