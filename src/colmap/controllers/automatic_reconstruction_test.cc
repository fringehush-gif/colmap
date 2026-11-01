// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <exception>
#include <fstream>
#include <string>

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

#if defined(COLMAP_MVS_ENABLED) && \
    (defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED))
TEST(AutomaticReconstruction, DenseCompletionRequiresVisibility) {
  for (const bool with_visibility : {true, false}) {
    SCOPED_TRACE(with_visibility);
    const auto dir =
        CreateTestDir() / (with_visibility ? "complete" : "missing");
    const auto dense = dir / "dense" / "0";
    CreateDirIfNotExists(dense, true);
    CreateDirIfNotExists(dir / "images");
    // Sentinel files must be skipped without reading an absent MVS workspace.
    std::ofstream(dense / "fused.ply") << "complete PLY sentinel";
    std::ofstream(dense / "meshed-poisson.ply") << "mesh sentinel";
    if (with_visibility) std::ofstream(dense / "fused.ply.vis") << "visibility";

    AutomaticReconstructionController::Options options;
    options.workspace_path = dir;
    options.image_path = dir / "images";
    options.extraction = false;
    options.matching = false;
    options.sparse = false;
    options.dense = true;
    options.use_gpu = false;
    options.num_threads = 1;
    auto manager = std::make_shared<ReconstructionManager>();
    manager->Add();
    AutomaticReconstructionController controller(options, manager);
    controller.Setup();
    controller.Start();
    if (with_visibility) {
      EXPECT_NO_THROW(controller.Wait());
    } else {
      // Missing VIS must resume rather than accept the sentinel pair. Reading
      // the absent workspace fails before PatchMatch queries or uses a GPU.
      try {
        controller.Wait();
        FAIL() << "Expected missing MVS workspace";
      } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("files do not exist at"),
                  std::string::npos);
      }
    }
  }
}
#endif

// TODO: Add GLOBAL mapper test. Currently excluded because the test produces
// fewer observations than expected. The global pipeline is tested separately
// in global_pipeline_test.cc.
INSTANTIATE_TEST_SUITE_P(
    AutomaticReconstructionTests,
    ParameterizedAutomaticReconstructionTests,
    ::testing::Values(AutomaticReconstructionController::Mapper::INCREMENTAL,
                      AutomaticReconstructionController::Mapper::HIERARCHICAL));

}  // namespace
}  // namespace colmap
