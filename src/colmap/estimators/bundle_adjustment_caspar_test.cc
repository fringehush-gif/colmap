// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

// Due to pose normalization operations, constant variables may not be perfectly
// fixed during bundle adjustment.
constexpr double kConstantPoseVarEps = 1e-9;

#define CheckVariableCamera(camera, orig_camera)       \
  {                                                    \
    const size_t focal_length_idx =                    \
        SimpleRadialCameraModel::focal_length_idxs[0]; \
    const size_t extra_param_idx =                     \
        SimpleRadialCameraModel::extra_params_idxs[0]; \
    EXPECT_NE((camera).params[focal_length_idx],       \
              (orig_camera).params[focal_length_idx]); \
    EXPECT_NE((camera).params[extra_param_idx],        \
              (orig_camera).params[extra_param_idx]);  \
  }

#define CheckConstantCamera(camera, orig_camera)       \
  {                                                    \
    const size_t focal_length_idx =                    \
        SimpleRadialCameraModel::focal_length_idxs[0]; \
    const size_t extra_param_idx =                     \
        SimpleRadialCameraModel::extra_params_idxs[0]; \
    EXPECT_EQ((camera).params[focal_length_idx],       \
              (orig_camera).params[focal_length_idx]); \
    EXPECT_EQ((camera).params[extra_param_idx],        \
              (orig_camera).params[extra_param_idx]);  \
  }

#define CheckVariableCamFromWorld(image, orig_image)                   \
  {                                                                    \
    EXPECT_THAT((image).CamFromWorld(),                                \
                testing::Not(Rigid3dEq((orig_image).CamFromWorld()))); \
  }

#define CheckConstantCamFromWorld(image, orig_image)     \
  {                                                      \
    EXPECT_THAT((image).CamFromWorld(),                  \
                Rigid3dNear((orig_image).CamFromWorld(), \
                            kConstantPoseVarEps,         \
                            kConstantPoseVarEps));       \
  }

#define CheckConstantPoint(point, orig_point) \
  {                                           \
    EXPECT_EQ((point).xyz, (orig_point).xyz); \
  }

namespace colmap {
namespace {

TEST(DefaultBundleAdjuster, RigThrowsErrorOnVariableSensorFromRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  BundleAdjustmentOptions options;
  BundleAdjustmentConfig config;
  options.refine_sensor_from_rig = true;  // Not supported yet
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  EXPECT_THROW(
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction),
      std::invalid_argument);
}

TEST(DefaultBundleAdjuster, MultiCameraRigLargeConstantSensorFromRig) {
  // Real-world multi-camera rigs (stereo, surround-view) have large
  // sensor_from_rig offsets — typically 20–90 degrees and 0.1–1 m baseline.
  // This test uses a 30-degree Z-rotation and 0.3 m translation to exercise
  // the non-identity sensor_from_rig path with realistic values.
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30.0;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.3;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  BundleAdjustmentOptions options;
  options.refine_sensor_from_rig = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(DefaultBundleAdjuster, MergedCalibConvergence) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.2,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(DefaultBundleAdjuster, MergedCalibFixedPose) {
  // Verifies that all four intrinsic parameters change.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t principal_point_idx_x =
      SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t principal_point_idx_y =
      SimpleRadialCameraModel::principal_point_idxs[1];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  for (const camera_t cam_id : {camera_t{1}, camera_t{2}}) {
    const auto& cam = reconstruction.Camera(cam_id);
    const auto& orig_cam = orig_reconstruction.Camera(cam_id);
    EXPECT_NE(cam.params[focal_length_idx], orig_cam.params[focal_length_idx]);
    EXPECT_NE(cam.params[extra_param_idx], orig_cam.params[extra_param_idx]);
    EXPECT_NE(cam.params[principal_point_idx_x],
              orig_cam.params[principal_point_idx_x]);
    EXPECT_NE(cam.params[principal_point_idx_y],
              orig_cam.params[principal_point_idx_y]);
  }

  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
}

TEST(DefaultBundleAdjuster, MergedCalibFixedPoint) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  // Fix all 3D points; only pose and calib are free.
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddConstantPoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckConstantPoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST(DefaultBundleAdjuster, ExternalImagePoseIsInvariant) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 3;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddVariablePoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
}

TEST(DefaultBundleAdjuster, ExternalCameraIntrinsicsOrderingIsConsistent) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 3;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddVariablePoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamera(reconstruction.Camera(3), orig_reconstruction.Camera(3));
}

TEST(DefaultBundleAdjuster, ExternalImageViaConstantPointsIsInvariant) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 3;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddConstantPoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
}

TEST(DefaultBundleAdjuster, MultipleExternalImagesAreInvariant) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 4;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddVariablePoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
  CheckConstantCamFromWorld(reconstruction.Image(4),
                            orig_reconstruction.Image(4));
}

TEST(DefaultBundleAdjuster, MergedCalibMatchesCeres) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.3;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;

  Reconstruction reconstruction_ceres = reconstruction;
  Reconstruction reconstruction_caspar = reconstruction;

  std::unique_ptr<BundleAdjuster> ceres_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction_ceres);
  ASSERT_NE(ceres_adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  std::unique_ptr<BundleAdjuster> caspar_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction_caspar);
  ASSERT_NE(caspar_adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Ceres uses exact dense Schur elimination with 7-DOF gauge fixing (scale
  // pinned), whereas Caspar uses iterative PCG with 6-DOF gauge fixing (scale
  // unpinned). Tolerances accommodate this gauge and solver difference while
  // ensuring convergence to the same calibration solution.
#ifdef CASPAR_USE_DOUBLE
  constexpr double kFocalTol = 1.0;
  constexpr double kPPTol = 2.0;
  constexpr double kExtraTol = 5e-4;
#else
  constexpr double kFocalTol = 20.0;
  constexpr double kPPTol = 10.0;
  constexpr double kExtraTol = 1.5e-2;
#endif

  const size_t f_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t cx_idx = SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t cy_idx = SimpleRadialCameraModel::principal_point_idxs[1];
  const size_t k_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  for (const auto& [cam_id, _] : reconstruction.Cameras()) {
    const auto& cam_ceres = reconstruction_ceres.Camera(cam_id);
    const auto& cam_caspar = reconstruction_caspar.Camera(cam_id);
    EXPECT_NEAR(cam_caspar.params[f_idx], cam_ceres.params[f_idx], kFocalTol)
        << "focal length mismatch for camera " << cam_id;
    EXPECT_NEAR(cam_caspar.params[cx_idx], cam_ceres.params[cx_idx], kPPTol)
        << "cx mismatch for camera " << cam_id;
    EXPECT_NEAR(cam_caspar.params[cy_idx], cam_ceres.params[cy_idx], kPPTol)
        << "cy mismatch for camera " << cam_id;
    EXPECT_NEAR(cam_caspar.params[k_idx], cam_ceres.params[k_idx], kExtraTol)
        << "radial distortion mismatch for camera " << cam_id;
  }
}

bool PoseExactlyUnchanged(const Image& a, const Image& b) {
  return a.CamFromWorld().rotation().coeffs() ==
             b.CamFromWorld().rotation().coeffs() &&
         a.CamFromWorld().translation() == b.CamFromWorld().translation();
}

TEST(DefaultBundleAdjuster, GaugeFixingWithOneFrameFromWorld) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Exactly one of the two frames must be pinned by gauge fixing.
  const int n_fixed =
      static_cast<int>(PoseExactlyUnchanged(reconstruction.Image(1),
                                            orig_reconstruction.Image(1))) +
      static_cast<int>(PoseExactlyUnchanged(reconstruction.Image(2),
                                            orig_reconstruction.Image(2)));
  EXPECT_EQ(n_fixed, 1);
}

TEST(DefaultBundleAdjuster,
     GaugeFixingWithOneFrameFromWorld_SkipsWhenAlreadyFixed) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);  // frame 1 explicitly constant
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Frame 1 is explicitly constant — must be unchanged.
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  // Frame 2 is not gauge-fixed (gauge fixer saw frame 1 already fixed) — must
  // change.
  CheckVariableCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));
}

TEST(DefaultBundleAdjuster, GaugeFixingWithThreePoints_PinsExactlyThreePoints) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  int n_unchanged = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.xyz == orig_reconstruction.Point3D(point3D_id).xyz) {
      ++n_unchanged;
    }
  }
  EXPECT_EQ(n_unchanged, 3);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
}

std::vector<image_t> GaugeTrackImageIds(const Point3D& point) {
  std::vector<image_t> ids;
  for (const auto& element : point.track.Elements())
    ids.push_back(element.image_id);
  std::sort(ids.begin(), ids.end());
  return ids;
}

TEST(DefaultBundleAdjuster,
     GaugeFixingWithThreePoints_CountsExistingConstantPoints) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  struct AnchorCase {
    int implicit_points;
    int explicit_points;
    bool refine_points;
    bool fixed_pose_only = false;
  };
  for (const AnchorCase test : {AnchorCase{0, 1, true},
                                AnchorCase{0, 0, true},
                                AnchorCase{1, 0, true},
                                AnchorCase{2, 0, true},
                                AnchorCase{3, 0, true},
                                AnchorCase{3, 0, false},
                                AnchorCase{3, 0, true, true}}) {
    SCOPED_TRACE(::testing::Message()
                 << "implicit=" << test.implicit_points
                 << " explicit=" << test.explicit_points
                 << " refine_points=" << test.refine_points
                 << " fixed_pose_only=" << test.fixed_pose_only);
    SetPRNGSeed(0);
    Reconstruction reconstruction;
    SyntheticDatasetOptions opts;
    opts.num_rigs = 3;
    opts.num_cameras_per_rig = 1;
    opts.num_frames_per_rig = 1;
    opts.num_points3D = 100;
    SynthesizeDataset(opts, &reconstruction);
    ASSERT_EQ(reconstruction.NumPoints3D(), opts.num_points3D);
    for (const auto& [id, point] : reconstruction.Points3D()) {
      ASSERT_GE(id, 1);
      ASSERT_LE(id, opts.num_points3D);
      ASSERT_EQ(GaugeTrackImageIds(point), (std::vector<image_t>{1, 2, 3}));
    }
    // Only the selected anchors retain an observation outside the local BA.
    // Their full tracks therefore exceed the two included observations.
    const Image& external_image = reconstruction.Image(3);
    for (point2D_t idx = 0; idx < external_image.NumPoints2D(); ++idx) {
      const Point2D& point = external_image.Point2D(idx);
      if (point.HasPoint3D() &&
          point.point3D_id > static_cast<point3D_t>(test.implicit_points)) {
        reconstruction.DeleteObservation(3, idx);
      }
    }
    if (test.fixed_pose_only) {
      const Image& image = reconstruction.Image(2);
      for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
        const Point2D& point = image.Point2D(idx);
        if (point.HasPoint3D() &&
            point.point3D_id <= static_cast<point3D_t>(test.implicit_points)) {
          reconstruction.DeleteObservation(2, idx);
        }
      }
    }
    ASSERT_EQ(reconstruction.NumPoints3D(), opts.num_points3D);
    for (const auto& [id, point] : reconstruction.Points3D()) {
      const bool implicit = id <= static_cast<point3D_t>(test.implicit_points);
      const std::vector<image_t> expected =
          !implicit              ? std::vector<image_t>{1, 2}
          : test.fixed_pose_only ? std::vector<image_t>{1, 3}
                                 : std::vector<image_t>{1, 2, 3};
      ASSERT_EQ(GaugeTrackImageIds(point), expected);
    }
    SyntheticNoiseOptions noise_opts;
    noise_opts.point2D_stddev = 1;
    noise_opts.point3D_stddev = 0.1;
    noise_opts.rig_from_world_rotation_stddev = 0.3;
    noise_opts.rig_from_world_translation_stddev = 0.05;
    SynthesizeNoise(noise_opts, &reconstruction);
    const Reconstruction original = reconstruction;
    ASSERT_GT((original.Point3D(2).xyz - original.Point3D(1).xyz)
                  .cross(original.Point3D(3).xyz - original.Point3D(1).xyz)
                  .norm(),
              1e-3);
    BundleAdjustmentConfig config;
    config.AddImage(1);
    config.AddImage(2);
    for (int idx = 1; idx <= test.explicit_points; ++idx) {
      config.AddConstantPoint(idx);
    }
    config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
    if (test.fixed_pose_only) {
      config.SetConstantRigFromWorldPose(reconstruction.Image(1).FrameId());
      config.SetConstantCamIntrinsics(reconstruction.Image(2).CameraId());
    }
    BundleAdjustmentOptions options;
    options.refine_focal_length = test.fixed_pose_only;
    options.refine_extra_params = test.fixed_pose_only;
    options.refine_principal_point = false;
    options.refine_points3D = test.refine_points;
    options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
    options.caspar->collect_iteration_data = true;
    auto adjuster =
        CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
    const auto summary =
        std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
            adjuster->Solve());
    ASSERT_NE(summary, nullptr);
    ASSERT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    EXPECT_EQ(summary->num_residuals,
              400 - (test.fixed_pose_only ? 2 * test.implicit_points : 0));
    ASSERT_TRUE(std::isfinite(summary->initial_score));
    ASSERT_GT(summary->initial_score, 0);
    EXPECT_TRUE(std::any_of(summary->iterations.begin(),
                            summary->iterations.end(),
                            [&](const auto& iteration) {
                              return iteration.step_accepted &&
                                     std::isfinite(iteration.score_best) &&
                                     iteration.score_best <
                                         summary->initial_score;
                            }));
    int unchanged = 0;
    double largest_change = 0;
    for (const auto& [id, point] : reconstruction.Points3D()) {
      EXPECT_TRUE(point.xyz.allFinite());
      if (point.xyz == original.Point3D(id).xyz) ++unchanged;
      largest_change = std::max(largest_change,
                                (point.xyz - original.Point3D(id).xyz).norm());
    }
    EXPECT_EQ(unchanged,
              test.refine_points
                  ? 3 + (test.fixed_pose_only ? test.implicit_points : 0)
                  : opts.num_points3D);
    if (test.refine_points) {
      EXPECT_GT(largest_change, 1e-3);
    }
    for (int idx = 1; idx <= test.implicit_points; ++idx) {
      CheckConstantPoint(reconstruction.Point3D(idx), original.Point3D(idx));
    }
    for (int idx = 1; idx <= test.explicit_points; ++idx) {
      CheckConstantPoint(reconstruction.Point3D(idx), original.Point3D(idx));
    }
  }
}

TEST(DefaultBundleAdjuster,
     GaugeFixingWithThreePoints_IgnoresInactiveImplicitExternalPoints) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 4;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  ASSERT_EQ(reconstruction.NumPoints3D(), opts.num_points3D);
  for (const auto& [id, point] : reconstruction.Points3D()) {
    ASSERT_EQ(GaugeTrackImageIds(point), (std::vector<image_t>{1, 2, 3, 4}));
  }
  for (const image_t image_id : {image_t{3}, image_t{4}}) {
    const Image& image = reconstruction.Image(image_id);
    for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
      if (image.Point2D(idx).HasPoint3D()) {
        reconstruction.DeleteObservation(image_id, idx);
      }
    }
  }
  ASSERT_EQ(reconstruction.NumPoints3D(), opts.num_points3D);
  for (const auto& [id, point] : reconstruction.Points3D()) {
    ASSERT_EQ(GaugeTrackImageIds(point), (std::vector<image_t>{1, 2}));
  }
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const camera_t unsupported_id = reconstruction.Image(4).CameraId();
  const Camera old_camera = reconstruction.Camera(unsupported_id);
  reconstruction.Camera(unsupported_id) =
      Camera::CreateFromModelId(unsupported_id,
                                CameraModelId::kOpenCV,
                                old_camera.MeanFocalLength(),
                                old_camera.width,
                                old_camera.height);
  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
  std::vector<point3D_t> external_ids;
  for (point2D_t idx = 0; idx < 3; ++idx) {
    ASSERT_FALSE(reconstruction.Image(3).Point2D(idx).HasPoint3D());
    ASSERT_FALSE(reconstruction.Image(4).Point2D(idx).HasPoint3D());
    Track track;
    track.AddElement(3, idx);
    track.AddElement(4, idx);
    const Eigen::Vector3d xyz(idx == 1 ? 1.0 : 0.0, idx == 2 ? 1.0 : 0.0, 4.0);
    const point3D_t id = reconstruction.AddPoint3D(xyz, track);
    external_ids.push_back(id);
    ASSERT_EQ(GaugeTrackImageIds(reconstruction.Point3D(id)),
              (std::vector<image_t>{3, 4}));
    config.AddVariablePoint(id);
  }
  const Reconstruction original = reconstruction;
  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  options.refine_principal_point = false;
  options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
  options.caspar->collect_iteration_data = true;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
      adjuster->Solve());
  ASSERT_NE(summary, nullptr);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);
  EXPECT_EQ(summary->num_residuals, 400);
  ASSERT_TRUE(std::isfinite(summary->initial_score));
  ASSERT_GT(summary->initial_score, 0);
  EXPECT_TRUE(std::any_of(summary->iterations.begin(),
                          summary->iterations.end(),
                          [&](const auto& iteration) {
                            return iteration.step_accepted &&
                                   std::isfinite(iteration.score_best) &&
                                   iteration.score_best <
                                       summary->initial_score;
                          }));
  int unchanged_local = 0;
  double largest_change = 0;
  for (const auto& [id, point] : reconstruction.Points3D()) {
    EXPECT_TRUE(point.xyz.allFinite());
    if (std::find(external_ids.begin(), external_ids.end(), id) !=
        external_ids.end()) {
      CheckConstantPoint(point, original.Point3D(id));
    } else {
      if (point.xyz == original.Point3D(id).xyz) ++unchanged_local;
      largest_change = std::max(largest_change,
                                (point.xyz - original.Point3D(id).xyz).norm());
    }
  }
  EXPECT_EQ(unchanged_local, 3);
  EXPECT_GT(largest_change, 1e-3);
}

TEST(DefaultBundleAdjuster, MultiCameraRigResidualCountConstantSensorFromRig) {
  // All sensor observations (ref and non-ref) must contribute residuals.
  // The old code skipped non-ref sensor observations when the pose was
  // variable, which would halve the residual count for a 2-camera rig.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_sensor_from_rig = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points × 4 images (2 sensors × 2 frames) × 2 residuals per obs
  EXPECT_EQ(summary->num_residuals, 800);
}

TEST(DefaultBundleAdjuster, MultiCameraRigConstantRigPoseHoldsAllSensors) {
  // When a frame's rig_from_world is held constant, and Caspar always holds
  // sensor_from_rig constant, ALL sensors in that frame (ref and non-ref)
  // must have invariant cam_from_world. Sensors in the variable frame must
  // change. This differs from the Ceres behaviour where non-ref sensors can
  // still move via a variable sensor_from_rig.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.3;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  const frame_t constant_frame_id = 1;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.SetConstantRigFromWorldPose(constant_frame_id);

  BundleAdjustmentOptions options;
  options.refine_sensor_from_rig = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(bundle_adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    if (image.FrameId() == constant_frame_id) {
      CheckConstantCamFromWorld(image, orig_reconstruction.Image(image_id));
    } else {
      CheckVariableCamFromWorld(image, orig_reconstruction.Image(image_id));
    }
  }
}

TEST(DefaultBundleAdjuster,
     MultiCameraRigLargeConvergenceConstantSensorFromRig) {
  // 2 rigs × 3 cameras × 5 frames = 30 images. Mirrors the Ceres
  // NominalMultiCameraRig test to verify Caspar converges to GT at the same
  // scale as the single-camera nominal test.
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_sensor_from_rig = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

}  // namespace
}  // namespace colmap
