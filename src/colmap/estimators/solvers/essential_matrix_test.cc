// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/essential_matrix.h"

#include "colmap/estimators/solvers/essential_matrix.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/camera.h"
#include "colmap/util/eigen_alignment.h"

#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Rejection thresholds used by RandomEpipolarCorrespondences to condition a
// minimal sample: minimum depth in front of either camera, and minimum parallax
// between the two rays as sin^2 of their angle (1 - a^2 in the cheirality
// form).
constexpr double kMinDepth = 0.2;
constexpr double kMinParallax = 1e-2;  // ~5.7 degrees.

// Generates a random relative pose with a unit-norm baseline. Bounding the
// baseline away from zero avoids the (near) pure-rotation degeneracy, where the
// essential matrix vanishes and the cheirality of every correspondence becomes
// ill-defined.
Rigid3d TestCam2FromCam1() {
  return Rigid3d(RandomEigenQuaterniond(),
                 RandomEigenVectord<3>().normalized());
}

// When reject_degenerate is set, resamples correspondences that make a minimal
// 5-point solve ill-conditioned (near-zero depth or near-parallel rays). Only
// the minimal case needs this; larger samples are robust to such points.
void RandomEpipolarCorrespondences(const Rigid3d& cam2_from_cam1,
                                   size_t num_rays,
                                   bool reject_degenerate,
                                   std::vector<Eigen::Vector3d>& rays1,
                                   std::vector<Eigen::Vector3d>& rays2) {
  for (size_t i = 0; i < num_rays; ++i) {
    Eigen::Vector3d ray1;
    Eigen::Vector3d point_in_cam2;
    bool degenerate;
    do {
      ray1 = RandomEigenVectord<3>().normalized();
      const double random_depth = RandomUniformReal<double>(kMinDepth, 2.0);
      point_in_cam2 = cam2_from_cam1 * (random_depth * ray1);
      const Eigen::Vector3d ray1_in_cam2 = cam2_from_cam1.rotation() * ray1;
      const double cos_parallax = ray1_in_cam2.dot(point_in_cam2.normalized());
      degenerate = point_in_cam2.norm() < kMinDepth ||
                   1.0 - cos_parallax * cos_parallax < kMinParallax;
    } while (reject_degenerate && degenerate);
    rays1.push_back(ray1);
    rays2.push_back(point_in_cam2.normalized());
  }
}

template <typename Estimator>
void ExpectAtLeastOneValidModel(const Estimator& estimator,
                                const std::vector<Eigen::Vector3d>& rays1,
                                const std::vector<Eigen::Vector3d>& rays2,
                                Eigen::Matrix3d& expected_E,
                                std::vector<Eigen::Matrix3d>& models,
                                double E_eps = 1e-4,
                                double r_eps = 1e-5) {
  expected_E.normalize();
  for (size_t i = 0; i < models.size(); ++i) {
    const Eigen::Matrix3d E = models[i].normalized();
    if (std::min((E - expected_E).norm(), (E + expected_E).norm()) > E_eps) {
      continue;
    }

    // The five/eight-point solvers no longer expose Residuals (bearing Sampson
    // was retired). Verify the recovered model directly with the plain Sampson
    // error, which is ~0 for these noiseless, in-front correspondences.
    std::vector<double> residuals;
    ComputeSquaredSampsonError(rays1, rays2, E, &residuals);
    for (size_t j = 0; j < rays1.size(); ++j) {
      EXPECT_LT(residuals[j], r_eps);
    }

    return;
  }
  ADD_FAILURE() << "No essential matrix is equal up to scale.";
}

class EssentialMatrixFivePointEstimatorTests
    : public ::testing::TestWithParam<size_t> {};

TEST_P(EssentialMatrixFivePointEstimatorTests, Nominal) {
  const size_t kNumRays = GetParam();
  // The minimal case has no redundancy, so it conditions its sample to stay
  // well-posed and accepts the solver's numerical accuracy with a looser
  // tolerance.
  const bool is_minimal =
      kNumRays == EssentialMatrixFivePointEstimator::kMinNumSamples;
  for (size_t k = 0; k < 100; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    RandomEpipolarCorrespondences(cam2_from_cam1,
                                  kNumRays,
                                  /*reject_degenerate=*/is_minimal,
                                  rays1,
                                  rays2);

    EssentialMatrixFivePointEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(rays1, rays2, &models);

    ExpectAtLeastOneValidModel(estimator,
                               rays1,
                               rays2,
                               expected_E,
                               models,
                               /*E_eps=*/is_minimal ? 5e-3 : 1e-4);
  }
}

INSTANTIATE_TEST_SUITE_P(EssentialMatrixFivePointEstimator,
                         EssentialMatrixFivePointEstimatorTests,
                         ::testing::Values(5, 20, 1000));

class EssentialMatrixEightPointEstimatorTests
    : public ::testing::TestWithParam<size_t> {};

TEST_P(EssentialMatrixEightPointEstimatorTests, Nominal) {
  const size_t kNumRays = GetParam();
  for (size_t k = 0; k < 1; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    RandomEpipolarCorrespondences(
        cam2_from_cam1, kNumRays, /*reject_degenerate=*/false, rays1, rays2);

    EssentialMatrixEightPointEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(rays1, rays2, &models);

    ExpectAtLeastOneValidModel(estimator, rays1, rays2, expected_E, models);
  }
}

INSTANTIATE_TEST_SUITE_P(EssentialMatrixEightPointEstimator,
                         EssentialMatrixEightPointEstimatorTests,
                         ::testing::Values(8, 64, 1024));

// Attaches an unprojection Jacobian to each bearing via a spherical camera,
// which maps every direction to a valid pixel.
std::vector<CamRayWithJac> WithJacobians(
    const Camera& camera, const std::vector<Eigen::Vector3d>& rays) {
  std::vector<CamRayWithJac> cam_rays_with_jac(rays.size());
  for (size_t i = 0; i < rays.size(); ++i) {
    cam_rays_with_jac[i] =
        camera.CamRayFromImgWithJac(camera.ImgFromCam(rays[i]).value()).value();
  }
  return cam_rays_with_jac;
}

// A fixed, front-facing scene with substantial baseline; no random generator is
// involved in the data or the essential-matrix reference used by these tests.
Rigid3d DeterministicCam2FromCam1() {
  return Rigid3d(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitY())),
      Eigen::Vector3d(0.8, 0.1, 0.05).normalized());
}

void DeterministicEpipolarCorrespondences(const Rigid3d& cam2_from_cam1,
                                          size_t num_rays,
                                          std::vector<Eigen::Vector3d>* rays1,
                                          std::vector<Eigen::Vector3d>* rays2) {
  rays1->clear();
  rays2->clear();
  rays1->reserve(num_rays);
  rays2->reserve(num_rays);
  for (size_t i = 0; i < num_rays; ++i) {
    const Eigen::Vector3d point((static_cast<int>(i % 10) - 4.5) * 0.2,
                                (static_cast<int>(i / 10) - 4.0) * 0.12,
                                2.0 + 0.09 * (i % 13));
    rays1->push_back(point.normalized());
    rays2->push_back((cam2_from_cam1 * point).normalized());
  }
}

TEST(EssentialMatrixTangentSampsonEstimator,
     OptimisticResidualsBoundCheiralityGatedResiduals) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);
  const Rigid3d cam2_from_cam1 = DeterministicCam2FromCam1();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);
  std::vector<Eigen::Vector3d> rays1;
  std::vector<Eigen::Vector3d> rays2;
  DeterministicEpipolarCorrespondences(cam2_from_cam1, 40, &rays1, &rays2);
  // Negating the second bearing preserves the epipolar equation while moving
  // its triangulated point behind that camera for the majority-selected pose.
  for (size_t i = 0; i < 4; ++i) rays2[i] = -rays2[i];
  const auto crj1 = WithJacobians(camera, rays1);
  const auto crj2 = WithJacobians(camera, rays2);

  std::vector<double> optimistic;
  std::vector<double> exact;
  EssentialMatrixTangentSampsonEstimator::OptimisticResiduals(
      crj1, crj2, E, &optimistic);
  EssentialMatrixTangentSampsonEstimator::Residuals(crj1, crj2, E, &exact);
  ASSERT_EQ(optimistic.size(), rays1.size());
  ASSERT_EQ(exact.size(), rays1.size());
  size_t gated = 0;
  size_t accepted = 0;
  for (size_t i = 0; i < rays1.size(); ++i) {
    EXPECT_TRUE(std::isfinite(optimistic[i]));
    EXPECT_GE(optimistic[i], 0.0);
    EXPECT_LE(optimistic[i], exact[i]);
    if (exact[i] == std::numeric_limits<double>::max()) {
      ++gated;
    } else {
      ++accepted;
    }
  }
  EXPECT_GE(gated, 1);
  EXPECT_GE(accepted, 30);
}

TEST(EssentialMatrixTangentSampsonEstimator,
     OptimisticLORANSACMatchesExactMSACWithGrossOutliers) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);
  const Rigid3d cam2_from_cam1 = DeterministicCam2FromCam1();
  std::vector<Eigen::Vector3d> rays1;
  std::vector<Eigen::Vector3d> rays2;
  DeterministicEpipolarCorrespondences(cam2_from_cam1, 100, &rays1, &rays2);
  for (size_t i = 0; i < 30; ++i) {  // 30% gross outliers.
    rays2[i] =
        Eigen::Vector3d(0.6 + 0.03 * i, -0.7 + 0.02 * i, -0.4).normalized();
  }
  const auto crj1 = WithJacobians(camera, rays1);
  const auto crj2 = WithJacobians(camera, rays2);
  RANSACOptions options;
  options.max_error = 2.0;  // pixels
  options.min_num_trials = 64;
  options.max_num_trials = 64;
  options.random_seed = 20261010;
  options.num_threads = 1;
  const auto estimate = [&](const bool enabled) {
    LORANSAC<EssentialMatrixTangentSampsonEstimator,
             EssentialMatrixTangentSampsonEstimator,
             MEstimatorSupportMeasurer>
        ransac(options, EssentialMatrixTangentSampsonEstimator(enabled));
    return ransac.Estimate(crj1, crj2);
  };
  const auto exact = estimate(false);
  const auto screened = estimate(true);
  ASSERT_TRUE(exact.success);
  ASSERT_TRUE(screened.success);
  EXPECT_GE(exact.support.num_inliers, 60);
  EXPECT_EQ(screened.support.num_inliers, exact.support.num_inliers);
  EXPECT_DOUBLE_EQ(screened.support.score, exact.support.score);
  EXPECT_EQ(screened.inlier_mask, exact.inlier_mask);
  EXPECT_TRUE(screened.model.isApprox(exact.model, 0.0));
}

TEST(EssentialMatrixTangentSampsonEstimator,
     OptimisticResidualsBoundFullScoring) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, 0.0, 1000, 500);
  for (int trial = 0; trial < 8; ++trial) {
    const auto pose = TestCam2FromCam1();
    const Eigen::Matrix3d E = EssentialMatrixFromPose(pose);
    std::vector<Eigen::Vector3d> rays1, rays2;
    RandomEpipolarCorrespondences(pose, 30, false, rays1, rays2);
    // Include inconsistent pairs alongside the epipolar correspondences.
    for (size_t i = 1; i < rays2.size(); i += 3) rays2[i] = -rays2[i];
    const auto first = WithJacobians(camera, rays1);
    const auto second = WithJacobians(camera, rays2);
    for (int boundary = 0; boundary < 3; ++boundary) {
      auto X = first;
      auto Y = second;
      if (boundary == 1) {
        X[0].jacobian.setZero();
        Y[0].jacobian.setZero();
      } else if (boundary == 2) {
        X[0].jacobian.setConstant(std::numeric_limits<double>::quiet_NaN());
      }
      std::vector<double> lower, full;
      EssentialMatrixTangentSampsonEstimator::OptimisticResiduals(
          X, Y, E, &lower);
      EssentialMatrixTangentSampsonEstimator::Residuals(X, Y, E, &full);
      ASSERT_EQ(lower.size(), X.size());
      ASSERT_EQ(full.size(), X.size());
      if (boundary == 2) {
        EXPECT_TRUE(std::isnan(lower[0]));
        EXPECT_FALSE(internal::CannotImproveOptimisticMSAC(lower, 4.0, 1.0));
      }
      for (size_t i = 0; i < lower.size(); ++i) {
        if (std::isfinite(lower[i])) {
          EXPECT_GE(lower[i], 0.0);
          EXPECT_LE(lower[i], full[i]);
        } else {
          EXPECT_FALSE(internal::CannotImproveOptimisticMSAC(lower, 4.0, 1.0));
        }
      }
    }
  }
}

// Refine recovers the true pose from a perturbed initial E on exact rays.
TEST(EssentialMatrixTangentSampsonEstimator, RefineRecoversPose) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);
  for (size_t k = 0; k < 30; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    const Eigen::Matrix3d expected =
        EssentialMatrixFromPose(cam2_from_cam1).normalized();
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    RandomEpipolarCorrespondences(
        cam2_from_cam1, 50, /*reject_degenerate=*/false, rays1, rays2);
    const std::vector<CamRayWithJac> crj1 = WithJacobians(camera, rays1);
    const std::vector<CamRayWithJac> crj2 = WithJacobians(camera, rays2);

    // Seed the refinement from a slightly perturbed pose.
    const Rigid3d init(
        cam2_from_cam1.rotation() *
            Eigen::Quaterniond(
                Eigen::AngleAxisd(0.01, RandomEigenVectord<3>().normalized())),
        (cam2_from_cam1.translation() + 0.01 * RandomEigenVectord<3>())
            .normalized());
    Eigen::Matrix3d E = EssentialMatrixFromPose(init);

    auto dist = [&expected](const Eigen::Matrix3d& m) {
      const Eigen::Matrix3d n = m.normalized();
      return std::min((n - expected).norm(), (n + expected).norm());
    };
    const double init_dist = dist(E);
    ASSERT_TRUE(EssentialMatrixTangentSampsonEstimator::Refine(crj1, crj2, &E));
    EXPECT_LT(dist(E), 1e-4);
    EXPECT_LT(dist(E), init_dist);
  }
}

// The LO-RANSAC estimator recovers the pose despite 30% gross outliers.
TEST(EssentialMatrixTangentSampsonEstimator, LORANSACWithOutliers) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);
  for (size_t k = 0; k < 10; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    const Eigen::Matrix3d expected =
        EssentialMatrixFromPose(cam2_from_cam1).normalized();
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    RandomEpipolarCorrespondences(
        cam2_from_cam1, 200, /*reject_degenerate=*/false, rays1, rays2);
    for (size_t i = 0; i < 60; ++i) {  // 30% gross outliers.
      rays2[i] = RandomEigenVectord<3>().normalized();
    }
    const std::vector<CamRayWithJac> crj1 = WithJacobians(camera, rays1);
    const std::vector<CamRayWithJac> crj2 = WithJacobians(camera, rays2);

    RANSACOptions options;
    options.max_error = 2.0;  // pixels
    LORANSAC<EssentialMatrixTangentSampsonEstimator,
             EssentialMatrixTangentSampsonEstimator>
        ransac(options);
    const auto report = ransac.Estimate(crj1, crj2);

    ASSERT_TRUE(report.success);
    const Eigen::Matrix3d E = report.model.normalized();
    EXPECT_LT(std::min((E - expected).norm(), (E + expected).norm()), 1e-2);
  }
}
}  // namespace
}  // namespace colmap
