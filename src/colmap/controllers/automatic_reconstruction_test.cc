// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <fstream>
#include <map>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

class ParameterizedAutomaticReconstructionTests
    : public ::testing::TestWithParam<
          AutomaticReconstructionController::Mapper> {};

TEST_P(ParameterizedAutomaticReconstructionTests, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 5;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.num_points2D_without_point3D = 10;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);
  SynthesizeImages(SyntheticImageOptions(), gt_reconstruction, image_path);

  AutomaticReconstructionController::Options options;
  options.workspace_path = workspace_path;
  options.image_path = image_path;
  options.data_type = AutomaticReconstructionController::DataType::INDIVIDUAL;
  options.quality = AutomaticReconstructionController::Quality::LOW;
  options.single_camera = false;
  options.dense = false;  // Disable dense reconstruction to avoid GPU
  options.use_gpu = false;
  options.random_seed = 1;
  options.mapper = GetParam();

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  EXPECT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(*reconstruction_manager->Get(0),
              ReconstructionNear(gt_reconstruction,
                                 /*max_rotation_error_deg=*/0.6,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.9,
                                 /*align=*/true));
}

// TODO: Add GLOBAL mapper test. Currently excluded because the test produces
// fewer observations than expected. The global pipeline is tested separately
// in global_pipeline_test.cc.
INSTANTIATE_TEST_SUITE_P(
    AutomaticReconstructionTests,
    ParameterizedAutomaticReconstructionTests,
    ::testing::Values(AutomaticReconstructionController::Mapper::INCREMENTAL,
                      AutomaticReconstructionController::Mapper::HIERARCHICAL));

class ParameterizedAutomaticReconstructionIntrinsicsTests
    : public ::testing::TestWithParam<
          AutomaticReconstructionController::Mapper> {};

TEST_P(ParameterizedAutomaticReconstructionIntrinsicsTests, FixedIntrinsics) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  // Like the benchmark, write the features and matches to the database
  // instead of extracting and matching them. The cameras have no prior focal
  // length, which the global mapper would otherwise calibrate.
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 5;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 200;
  std::map<camera_t, std::vector<double>> expected_params;
  {
    auto database = Database::Open(workspace_path / "database.db");
    SynthesizeDataset(
        synthetic_dataset_options, &gt_reconstruction, database.get());
    SyntheticNoiseOptions synthetic_noise_options;
    synthetic_noise_options.point2D_stddev = 0.5;
    SynthesizeNoise(
        synthetic_noise_options, &gt_reconstruction, database.get());
    // Perturb the focal lengths, which refinement or view graph calibration
    // would otherwise move back towards the GT.
    for (Camera camera : database->ReadAllCameras()) {
      ASSERT_FALSE(camera.HasPriorFocalLength());
      camera.SetFocalLength(1.1 * camera.FocalLength());
      database->UpdateCamera(camera);
      expected_params[camera.camera_id] = camera.params;
    }
  }

  AutomaticReconstructionController::Options options;
  options.workspace_path = workspace_path;
  options.image_path = image_path;
  options.extraction = false;
  options.matching = false;
  options.refine_intrinsics = false;
  options.dense = false;
  options.use_gpu = false;
  options.random_seed = 1;
  options.mapper = GetParam();

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  ASSERT_GE(reconstruction_manager->Size(), 1);
  const Reconstruction& reconstruction = *reconstruction_manager->Get(0);
  ASSERT_EQ(reconstruction.NumRegImages(),
            synthetic_dataset_options.num_rigs *
                synthetic_dataset_options.num_frames_per_rig);
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    EXPECT_EQ(camera.params, expected_params.at(camera_id));
  }
  // View graph calibration is skipped.
  auto database = Database::Open(workspace_path / "database.db");
  for (const auto& [camera_id, _] : expected_params) {
    EXPECT_EQ(database->ReadAllCameraSources(camera_id).count(
                  CameraSource::VIEW_GRAPH),
              0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    AutomaticReconstructionTests,
    ParameterizedAutomaticReconstructionIntrinsicsTests,
    ::testing::Values(AutomaticReconstructionController::Mapper::INCREMENTAL,
                      AutomaticReconstructionController::Mapper::HIERARCHICAL,
                      AutomaticReconstructionController::Mapper::GLOBAL));

#if defined(COLMAP_MVS_ENABLED) && \
    (defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED))
TEST(AutomaticReconstruction, PoissonRetriesIncompleteOutput) {
  const auto dir = CreateTestDir();
  const auto dense = dir / "dense" / "0";
  const auto fused = dense / "fused.ply";
  const auto mesh = dense / "meshed-poisson.ply";
  CreateDirIfNotExists(dense / "sparse", true);
  CreateDirIfNotExists(dense / "stereo");
  CreateDirIfNotExists(dir / "images");
  Reconstruction().Write(dense / "sparse");
  ASSERT_TRUE(std::ofstream(dense / "stereo" / "patch-match.cfg").is_open());
  std::ofstream(dense / "fused.ply.vis") << "cached visibility";
  WriteTextPlyPoints(fused,
                     std::vector<PlyPoint>(1),
                     /*write_normal=*/false,
                     /*write_rgb=*/false);

  AutomaticReconstructionController::Options options;
  options.workspace_path = dir;
  options.image_path = dir / "images";
  options.extraction = false;
  options.matching = false;
  options.single_view_calibration = false;
  options.sparse = false;
  options.dense = true;
  options.use_gpu = false;
  options.num_threads = 1;
  options.quality = AutomaticReconstructionController::Quality::LOW;
  options.mesher = AutomaticReconstructionController::Mesher::POISSON;
  // The empty scene submits no GPU jobs and explicit indices avoid enumeration.
  options.gpu_index = "0,1";

  const auto run = [&] {
    auto manager = std::make_shared<ReconstructionManager>();
    manager->Add();
    AutomaticReconstructionController controller(options, manager);
    controller.Setup();
    controller.Start();
    controller.Wait();
  };
  const auto expect_failure = [&](const char* message) {
    try {
      run();
      ADD_FAILURE() << "Expected failure: " << message;
    } catch (const std::exception& error) {
      EXPECT_THAT(error.what(), testing::HasSubstr(message));
    }
  };

  expect_failure("Poisson meshing failed");
  EXPECT_FALSE(std::filesystem::exists(mesh));
  // Simulate an empty output left behind by an older controller.
  ASSERT_TRUE(std::ofstream(mesh).is_open());
  expect_failure("Poisson meshing failed");
  EXPECT_EQ(std::filesystem::file_size(mesh), 0);

  std::vector<PlyPoint> points;
  for (int axis = 0; axis < 3; ++axis) {
    for (const float side : {-1.0f, 1.0f}) {
      for (int u = -3; u < 4; u += 2) {
        for (int v = -3; v < 4; v += 2) {
          float pos[3] = {};
          float normal[3] = {};
          pos[axis] = normal[axis] = side;
          pos[(axis + 1) % 3] = 0.25f * u;
          pos[(axis + 2) % 3] = 0.25f * v;
          PlyPoint point;
          point.x = pos[0];
          point.y = pos[1];
          point.z = pos[2];
          point.nx = normal[0];
          point.ny = normal[1];
          point.nz = normal[2];
          points.push_back(point);
        }
      }
    }
  }
  WriteBinaryPlyPoints(
      fused, points, /*write_normal=*/true, /*write_rgb=*/true);
  ASSERT_NO_THROW(run());
  ASSERT_GT(std::filesystem::file_size(mesh), 0);
  EXPECT_NO_THROW(ReadPly(mesh));
  const auto expected_size = std::filesystem::file_size(mesh);
  WriteTextPlyPoints(fused,
                     std::vector<PlyPoint>(1),
                     /*write_normal=*/false,
                     /*write_rgb=*/false);
  EXPECT_NO_THROW(run());
  EXPECT_EQ(std::filesystem::file_size(mesh), expected_size);
#ifndef _WIN32
  // Creating a symlink can require elevated privileges on Windows.
  const auto saved_mesh = dense / "preserved-mesh.ply";
  std::filesystem::rename(mesh, saved_mesh);
  std::filesystem::create_symlink(saved_mesh.filename(), mesh);
  EXPECT_NO_THROW(run());
  EXPECT_EQ(std::filesystem::read_symlink(mesh), saved_mesh.filename());
  EXPECT_EQ(std::filesystem::file_size(saved_mesh), expected_size);

  ASSERT_TRUE(std::filesystem::remove(saved_mesh));
  expect_failure("Poisson mesh symlink needs a regular target");
  EXPECT_EQ(std::filesystem::read_symlink(mesh), saved_mesh.filename());
  EXPECT_FALSE(std::filesystem::exists(saved_mesh));
#endif
  ASSERT_TRUE(std::filesystem::remove(mesh));
  ASSERT_TRUE(std::filesystem::create_directory(mesh));
  expect_failure("Poisson mesh destination is not a regular file");
  EXPECT_TRUE(std::filesystem::is_directory(mesh));
  for (const auto& entry : std::filesystem::directory_iterator(dense)) {
    EXPECT_NE(entry.path().filename().string().find(".colmap-poisson-"), 0);
  }
}
#endif

}  // namespace
}  // namespace colmap
