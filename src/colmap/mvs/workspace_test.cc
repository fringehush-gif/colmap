// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/workspace.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

class ParameterizedWorkspaceTests
    : public ::testing::TestWithParam<std::function<std::unique_ptr<Workspace>(
          const Workspace::Options&)>> {
 protected:
  void SetUp() override {
    temp_dir_ = CreateTestDir();
    CreateDirIfNotExists(temp_dir_ / "sparse");
    CreateDirIfNotExists(temp_dir_ / "images");
    CreateDirIfNotExists(temp_dir_ / "stereo");
    CreateDirIfNotExists(temp_dir_ / "stereo" / "depth_maps");
    CreateDirIfNotExists(temp_dir_ / "stereo" / "normal_maps");

    SyntheticDatasetOptions options;
    options.num_rigs = 1;
    options.num_cameras_per_rig = 1;
    options.num_frames_per_rig = 1;
    options.camera_width = 10;
    options.camera_height = 5;
    Reconstruction reconstruction;
    SynthesizeDataset(options, &reconstruction);
    reconstruction.Write(temp_dir_ / "sparse");

    image_name_ = reconstruction.Image(1).Name();

    Mat<float> depth_map(options.camera_width, options.camera_height, 1);
    depth_map.Fill(1.0f);
    depth_map.Write(temp_dir_ / "stereo" / "depth_maps" /
                    (image_name_ + ".geometric.bin"));

    Mat<float> normal_map(options.camera_width, options.camera_height, 3);
    normal_map.Fill(1.0f);
    normal_map.Write(temp_dir_ / "stereo" / "normal_maps" /
                     (image_name_ + ".geometric.bin"));

    Bitmap bitmap(options.camera_width, options.camera_height, true);
    bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
    bitmap.Write(temp_dir_ / "images" / image_name_);
  }

  Workspace::Options GetOptions() {
    Workspace::Options options;
    options.workspace_path = temp_dir_;
    options.workspace_format = "COLMAP";
    options.input_type = "geometric";
    return options;
  }

  std::filesystem::path temp_dir_;
  std::string image_name_;
};

TEST_P(ParameterizedWorkspaceTests, GetData) {
  auto workspace = GetParam()(GetOptions());
  const auto& model = workspace->GetModel();
  EXPECT_EQ(model.images.size(), 1);
  workspace->Load({image_name_});
  EXPECT_TRUE(workspace->HasBitmap(0));
  EXPECT_THAT(workspace->GetBitmapPath(0).string(),
              testing::HasSubstr(image_name_));
  EXPECT_FALSE(workspace->GetBitmap(0).IsEmpty());
  EXPECT_TRUE(workspace->HasDepthMap(0));
  EXPECT_THAT(workspace->GetDepthMapPath(0).string(),
              testing::HasSubstr(image_name_));
  EXPECT_GT(workspace->GetDepthMap(0).GetNumBytes(), 0);
  EXPECT_TRUE(workspace->HasNormalMap(0));
  EXPECT_THAT(workspace->GetNormalMapPath(0).string(),
              testing::HasSubstr(image_name_));
  EXPECT_GT(workspace->GetNormalMap(0).GetNumBytes(), 0);
}

TEST_P(ParameterizedWorkspaceTests, MaxImageSize) {
  Workspace::Options options = GetOptions();
  options.max_image_size = 4;
  auto workspace = GetParam()(options);
  workspace->Load({image_name_});
  ASSERT_EQ(workspace->GetModel().images.size(), 1);
  EXPECT_EQ(workspace->GetModel().images[0].GetWidth(), 4);
  EXPECT_EQ(workspace->GetModel().images[0].GetHeight(), 2);
  EXPECT_EQ(workspace->GetBitmap(0).Width(), 4);
  EXPECT_EQ(workspace->GetBitmap(0).Height(), 2);
  EXPECT_EQ(workspace->GetDepthMap(0).GetWidth(), 4);
  EXPECT_EQ(workspace->GetDepthMap(0).GetHeight(), 2);
  EXPECT_EQ(workspace->GetNormalMap(0).GetWidth(), 4);
  EXPECT_EQ(workspace->GetNormalMap(0).GetHeight(), 2);
}

TEST_P(ParameterizedWorkspaceTests, Load) {
  auto workspace = GetParam()(GetOptions());
  workspace->Load({image_name_});
}

TEST_P(ParameterizedWorkspaceTests, SourceNormalProbeDoesNotCacheOrResize) {
  auto options = GetOptions();
  options.max_image_size = 4;
  auto workspace = GetParam()(options);
  workspace->Load({image_name_});
  const bool cached =
      dynamic_cast<CachedWorkspace*>(workspace.get()) != nullptr;
  EXPECT_EQ(workspace->ValidateSourceNormalWithoutMaterialization(0), cached);
  Mat<float> changed(10, 5, 3);
  changed.Fill(-2.0f);
  changed.Write(workspace->GetNormalMapPath(0));
  const auto& reference = workspace->GetNormalMap(0);
  EXPECT_EQ(reference.GetWidth(), 4);
  EXPECT_EQ(reference.GetHeight(), 2);
  if (cached) EXPECT_LT(reference.Get(0, 0, 0), 0.0f);
}

TEST_P(ParameterizedWorkspaceTests,
       CachedSourceNormalProbePreservesResidentMap) {
  auto workspace = GetParam()(GetOptions());
  workspace->Load({image_name_});
  if (dynamic_cast<CachedWorkspace*>(workspace.get()) == nullptr) {
    EXPECT_FALSE(workspace->ValidateSourceNormalWithoutMaterialization(0));
    return;
  }
  const auto expected = workspace->GetNormalMap(0).GetData();
  std::ofstream(workspace->GetNormalMapPath(0),
                std::ios::binary | std::ios::trunc)
      << "bad map";
  EXPECT_TRUE(workspace->ValidateSourceNormalWithoutMaterialization(0));
  EXPECT_EQ(workspace->GetNormalMap(0).GetData(), expected);
}

TEST_P(ParameterizedWorkspaceTests, SourceNormalProbeFallsBackOnInvalidInput) {
  auto workspace = GetParam()(GetOptions());
  if (dynamic_cast<CachedWorkspace*>(workspace.get()) == nullptr) {
    EXPECT_FALSE(workspace->ValidateSourceNormalWithoutMaterialization(0));
    return;
  }
  const auto path = workspace->GetNormalMapPath(0);
  Mat<float> non_normal(10, 5, 2);
  non_normal.Fill(1.0f);
  non_normal.Write(path);
  EXPECT_FALSE(workspace->ValidateSourceNormalWithoutMaterialization(0));
  EXPECT_EQ(workspace->GetNormalMap(0).GetDepth(), 2);
  // A second instance avoids the intentionally cached two-channel map.
  auto uncached = GetParam()(GetOptions());
  for (const char* bytes : {"10&5&3&",
                            "0&5&3&",
                            "10,5&3&",
                            "18446744073709551615&5&3&",
                            "garbage"}) {
    std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
    EXPECT_FALSE(uncached->ValidateSourceNormalWithoutMaterialization(0));
  }
  std::ofstream(path, std::ios::binary | std::ios::trunc)
      << std::string(129, ' ') << "10&5&3&";
  EXPECT_FALSE(uncached->ValidateSourceNormalWithoutMaterialization(0));
  Mat<float> normal(10, 5, 3);
  normal.Fill(1.0f);
  normal.Write(path);
  std::ofstream(path, std::ios::binary | std::ios::app) << "tail";
  EXPECT_TRUE(uncached->ValidateSourceNormalWithoutMaterialization(0));
}

INSTANTIATE_TEST_SUITE_P(WorkspaceTests,
                         ParameterizedWorkspaceTests,
                         ::testing::Values(
                             [](const Workspace::Options& options) {
                               return std::make_unique<Workspace>(options);
                             },
                             [](const Workspace::Options& options) {
                               return std::make_unique<CachedWorkspace>(
                                   options);
                             }));

}  // namespace
}  // namespace mvs
}  // namespace colmap
