// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/workspace.h"

#include "colmap/mvs/patch_match_input.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <chrono>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <mutex>
#include <thread>

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
  EXPECT_EQ(workspace->CopyBitmap(0).RowMajorData(),
            workspace->GetBitmap(0).RowMajorData());
  EXPECT_EQ(workspace->CopyDepthMap(0).GetData(),
            workspace->GetDepthMap(0).GetData());
  EXPECT_EQ(workspace->CopyNormalMap(0).GetData(),
            workspace->GetNormalMap(0).GetData());
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

// The copy hook runs with the actual cached entry pinned and locked. Block the
// first copy, force eviction, then prove the same key can be loaded anew while
// the original owner is still alive.
TEST(CachedWorkspace, CopiesSurviveEvictionAndSameKeyReinsertion) {
  const auto temp_dir = CreateTestDir();
  CreateDirIfNotExists(temp_dir / "sparse");
  CreateDirIfNotExists(temp_dir / "images");
  CreateDirIfNotExists(temp_dir / "stereo");
  CreateDirIfNotExists(temp_dir / "stereo" / "depth_maps");
  CreateDirIfNotExists(temp_dir / "stereo" / "normal_maps");
  SyntheticDatasetOptions synthetic;
  synthetic.num_rigs = 1;
  synthetic.num_cameras_per_rig = 1;
  synthetic.num_frames_per_rig = 2;
  synthetic.camera_width = 10;
  synthetic.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic, &reconstruction);
  reconstruction.Write(temp_dir / "sparse");
  std::vector<std::string> names;
  for (const auto& [id, image] : reconstruction.Images()) {
    names.push_back(image.Name());
    Bitmap bitmap(10, 5, true);
    bitmap.Fill(BitmapColor<uint8_t>(13, 52, 97));
    bitmap.Write(temp_dir / "images" / image.Name());
    Mat<float> depth(10, 5, 1);
    depth.Fill(3.25f);
    depth.Write(temp_dir / "stereo" / "depth_maps" /
                (image.Name() + ".geometric.bin"));
    Mat<float> normal(10, 5, 3);
    normal.Fill(0.375f);
    normal.Write(temp_dir / "stereo" / "normal_maps" /
                 (image.Name() + ".geometric.bin"));
  }
  ASSERT_EQ(names.size(), 2);
  Workspace::Options options;
  options.workspace_path = temp_dir;
  options.workspace_format = "COLMAP";
  options.input_type = "geometric";
  options.cache_size = 1.0 / (1024.0 * 1024.0 * 1024.0);

  for (const auto payload : {internal::InputPayload::Bitmap,
                             internal::InputPayload::DepthMap,
                             internal::InputPayload::NormalMap}) {
    CachedWorkspace workspace(options);
    const int a = workspace.GetModel().GetImageIdx(names[0]);
    const int b = workspace.GetModel().GetImageIdx(names[1]);
    Bitmap expected_bitmap;
    expected_bitmap.Read(workspace.GetBitmapPath(a), true);
    DepthMap expected_depth;
    expected_depth.Read(workspace.GetDepthMapPath(a));
    NormalMap expected_normal;
    expected_normal.Read(workspace.GetNormalMapPath(a));

    std::mutex mutex;
    std::condition_variable cv;
    int copies_of_a = 0;
    bool release = false;
    bool hook_timeout = false;
    internal::SetInputCopyHookForTesting([&](int index, auto kind) {
      if (index != a || kind != payload) return;
      std::unique_lock<std::mutex> lock(mutex);
      ++copies_of_a;
      cv.notify_all();
      if (copies_of_a == 1 && !cv.wait_for(lock, std::chrono::seconds(10), [&] {
            return release;
          })) {
        hook_timeout = true;
      }
    });
    struct Cleanup {
      std::mutex& mutex;
      std::condition_variable& cv;
      bool& release;
      std::thread first;
      std::thread second;
      ~Cleanup() {
        {
          std::lock_guard<std::mutex> lock(mutex);
          release = true;
        }
        cv.notify_all();
        if (first.joinable()) first.join();
        if (second.joinable()) second.join();
        internal::SetInputCopyHookForTesting({});
      }
    };
    std::exception_ptr first_error, second_error;
    std::vector<uint8_t> bitmap_data[2];
    std::vector<float> map_data[2];
    auto copy = [&](int index, int output) {
      if (payload == internal::InputPayload::Bitmap) {
        bitmap_data[output] = workspace.CopyBitmap(index).RowMajorData();
      } else if (payload == internal::InputPayload::DepthMap) {
        map_data[output] = workspace.CopyDepthMap(index).GetData();
      } else {
        map_data[output] = workspace.CopyNormalMap(index).GetData();
      }
    };
    Cleanup cleanup{mutex, cv, release};
    cleanup.first = std::thread([&] {
      try {
        copy(a, 0);
      } catch (...) {
        first_error = std::current_exception();
      }
    });
    {
      std::unique_lock<std::mutex> lock(mutex);
      ASSERT_TRUE(cv.wait_for(
          lock, std::chrono::seconds(10), [&] { return copies_of_a == 1; }));
    }
    // A one-byte budget forces eviction when B is inserted/loaded.
    if (payload == internal::InputPayload::Bitmap) {
      EXPECT_FALSE(workspace.CopyBitmap(b).IsEmpty());
    } else if (payload == internal::InputPayload::DepthMap) {
      EXPECT_GT(workspace.CopyDepthMap(b).GetNumBytes(), 0);
    } else {
      EXPECT_GT(workspace.CopyNormalMap(b).GetNumBytes(), 0);
    }
    cleanup.second = std::thread([&] {
      try {
        copy(a, 1);
      } catch (...) {
        second_error = std::current_exception();
      }
    });
    {
      std::unique_lock<std::mutex> lock(mutex);
      EXPECT_TRUE(cv.wait_for(
          lock, std::chrono::seconds(10), [&] { return copies_of_a >= 2; }));
      release = true;
    }
    cv.notify_all();
    cleanup.first.join();
    cleanup.second.join();
    EXPECT_FALSE(hook_timeout);
    EXPECT_EQ(first_error, nullptr);
    EXPECT_EQ(second_error, nullptr);
    if (payload == internal::InputPayload::Bitmap) {
      EXPECT_EQ(bitmap_data[0], expected_bitmap.RowMajorData());
      EXPECT_EQ(bitmap_data[1], expected_bitmap.RowMajorData());
    } else {
      const auto& expected = payload == internal::InputPayload::DepthMap
                                 ? expected_depth.GetData()
                                 : expected_normal.GetData();
      EXPECT_EQ(map_data[0], expected);
      EXPECT_EQ(map_data[1], expected);
    }
  }
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
