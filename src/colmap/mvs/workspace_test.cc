// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/workspace.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/bitmap_reader.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

class WorkspaceTests : public ::testing::Test {
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

class ParameterizedWorkspaceTests
    : public WorkspaceTests,
      public ::testing::WithParamInterface<std::function<
          std::unique_ptr<Workspace>(const Workspace::Options&)>> {};

void ExpectLoadedWorkspaceData(Workspace& workspace) {
  const auto& bitmap = workspace.GetBitmap(0);
  const auto& depth = workspace.GetDepthMap(0);
  const auto& normal = workspace.GetNormalMap(0);
  ASSERT_EQ(bitmap.Width(), 10);
  ASSERT_EQ(bitmap.Height(), 5);
  ASSERT_EQ(depth.GetWidth(), 10);
  ASSERT_EQ(depth.GetHeight(), 5);
  ASSERT_EQ(depth.GetDepth(), 1);
  ASSERT_EQ(normal.GetWidth(), 10);
  ASSERT_EQ(normal.GetHeight(), 5);
  ASSERT_EQ(normal.GetDepth(), 3);
  for (int row = 0; row < 5; ++row) {
    for (int col = 0; col < 10; ++col) {
      const auto pixel = bitmap.GetPixel(col, row);
      ASSERT_TRUE(pixel);
      EXPECT_EQ(pixel->r, 7);
      EXPECT_EQ(pixel->g, 8);
      EXPECT_EQ(pixel->b, 9);
      EXPECT_FLOAT_EQ(depth.Get(row, col), 1.0f);
      for (int channel = 0; channel < 3; ++channel) {
        EXPECT_FLOAT_EQ(normal.Get(row, col, channel), 1.0f);
      }
    }
  }
}

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

TEST_P(ParameterizedWorkspaceTests, ComponentLeasesPreserveOwnershipPolicy) {
  auto workspace = GetParam()(GetOptions());
  workspace->Load({image_name_});
  const auto bitmap = workspace->GetBitmapLease(0);
  const auto depth = workspace->GetDepthMapLease(0);
  const auto normal = workspace->GetNormalMapLease(0);
  ASSERT_TRUE(bitmap);
  ASSERT_TRUE(depth);
  ASSERT_TRUE(normal);
  EXPECT_EQ(bitmap.get(), &workspace->GetBitmap(0));
  EXPECT_EQ(depth.get(), &workspace->GetDepthMap(0));
  EXPECT_EQ(normal.get(), &workspace->GetNormalMap(0));
  if (dynamic_cast<CachedWorkspace*>(workspace.get()) == nullptr) {
    EXPECT_EQ(bitmap.use_count(), 0);
    EXPECT_EQ(depth.use_count(), 0);
    EXPECT_EQ(normal.use_count(), 0);
  } else {
    EXPECT_GT(bitmap.use_count(), 0);
    EXPECT_FALSE(bitmap.owner_before(depth));
    EXPECT_FALSE(depth.owner_before(bitmap));
    EXPECT_FALSE(bitmap.owner_before(normal));
    EXPECT_FALSE(normal.owner_before(bitmap));
  }
}

TEST(CachedWorkspaceLease, SurvivesConcurrentEviction) {
  const auto temp_dir = CreateTestDir();
  CreateDirIfNotExists(temp_dir / "sparse");
  CreateDirIfNotExists(temp_dir / "images");
  CreateDirIfNotExists(temp_dir / "stereo");
  CreateDirIfNotExists(temp_dir / "stereo" / "depth_maps");
  CreateDirIfNotExists(temp_dir / "stereo" / "normal_maps");

  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = 2;
  dataset_options.camera_width = 10;
  dataset_options.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset_options, &reconstruction);
  reconstruction.Write(temp_dir / "sparse");

  std::vector<std::string> image_names;
  for (int image_id = 1; image_id <= 2; ++image_id) {
    const std::string name = reconstruction.Image(image_id).Name();
    image_names.push_back(name);
    DepthMap depth_map(10, 5, 1.0f, 10.0f);
    depth_map.Fill(static_cast<float>(image_id + 1));
    depth_map.Write(temp_dir / "stereo" / "depth_maps" /
                    (name + ".geometric.bin"));
    NormalMap normal_map(10, 5);
    normal_map.Fill(static_cast<float>(image_id + 1));
    normal_map.Write(temp_dir / "stereo" / "normal_maps" /
                     (name + ".geometric.bin"));
    Bitmap bitmap(10, 5, true);
    bitmap.Fill(BitmapColor<uint8_t>(image_id, image_id, image_id));
    bitmap.Write(temp_dir / "images" / name);
  }

  Workspace::Options options;
  options.workspace_path = temp_dir;
  options.workspace_format = "COLMAP";
  options.input_type = "geometric";
  options.cache_size = 1000.0 / (1024.0 * 1024.0 * 1024.0);
  CachedWorkspace workspace(options);
  workspace.Load(image_names);
  const int first = workspace.GetModel().GetImageIdx(image_names[0]);
  const int second = workspace.GetModel().GetImageIdx(image_names[1]);

  const auto bitmap = workspace.GetBitmapLease(first);
  const auto depth = workspace.GetDepthMapLease(first);
  const auto normal = workspace.GetNormalMapLease(first);
  ASSERT_TRUE(bitmap);
  ASSERT_TRUE(depth);
  ASSERT_TRUE(normal);
  const size_t first_bytes =
      bitmap->NumBytes() + depth->GetNumBytes() + normal->GetNumBytes();
  ASSERT_LE(first_bytes, 1000);
  ASSERT_GT(first_bytes + depth->GetNumBytes(), 1000);
  ASSERT_FALSE(bitmap.owner_before(depth));
  ASSERT_FALSE(depth.owner_before(bitmap));
  ASSERT_FALSE(bitmap.owner_before(normal));
  ASSERT_FALSE(normal.owner_before(bitmap));
  ASSERT_EQ(bitmap.get(), workspace.GetBitmapLease(first).get());
  std::thread evictor([&] { workspace.GetDepthMapLease(second); });
  evictor.join();

  const auto replacement = workspace.GetBitmapLease(first);
  ASSERT_NE(bitmap.get(), replacement.get());
  const Bitmap bitmap_copy = *bitmap;
  const DepthMap depth_copy = *depth;
  const NormalMap normal_copy = *normal;
  EXPECT_EQ(bitmap_copy.Width(), 10);
  EXPECT_EQ(bitmap_copy.Height(), 5);
  EXPECT_FLOAT_EQ(depth_copy.GetData()[0], 2.0f);
  EXPECT_FLOAT_EQ(normal_copy.GetData()[0], 2.0f);
}

TEST_F(WorkspaceTests, EagerLoadDeduplicatesPerCall) {
  for (const int num_threads : {1, 2}) {
    SCOPED_TRACE(num_threads);
    auto options = GetOptions();
    options.num_threads = num_threads;
    Workspace workspace(options);
    std::atomic<int> reads{0};
    auto reader = ScopedBitmapReader::Create(
        {workspace.GetBitmapPath(0)},
        [&](const auto&, bool, bool, Bitmap* target) {
          ++reads;
          *target = Bitmap(10, 5, true);
          target->Fill(BitmapColor<uint8_t>(7, 8, 9));
          return true;
        },
        {{10, 5}});
    workspace.Load({image_name_, image_name_, image_name_});
    ASSERT_EQ(reads.load(), 1);
    ExpectLoadedWorkspaceData(workspace);
    workspace.Load({image_name_, image_name_});
    ASSERT_EQ(reads.load(), 2);
    ExpectLoadedWorkspaceData(workspace);
    reader->Close();
  }
}

TEST_F(WorkspaceTests, EagerLoadMissingInputCanBeRestored) {
  auto options = GetOptions();
  options.num_threads = 1;
  Workspace workspace(options);
  const auto depth_path = workspace.GetDepthMapPath(0);
  ASSERT_TRUE(std::filesystem::remove(depth_path));
  std::atomic<int> reads{0};
  auto reader =
      ScopedBitmapReader::Create({workspace.GetBitmapPath(0)},
                                 [&](const auto&, bool, bool, Bitmap* target) {
                                   ++reads;
                                   *target = Bitmap(10, 5, true);
                                   target->Fill(BitmapColor<uint8_t>(7, 8, 9));
                                   return true;
                                 },
                                 {{10, 5}});
  workspace.Load({image_name_, image_name_});
  EXPECT_EQ(reads.load(), 0);
  Mat<float> depth(10, 5, 1);
  depth.Fill(1.0f);
  depth.Write(depth_path);
  workspace.Load({image_name_, image_name_});
  EXPECT_EQ(reads.load(), 1);
  ExpectLoadedWorkspaceData(workspace);
  reader->Close();
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
