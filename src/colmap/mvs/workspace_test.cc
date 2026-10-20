// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/workspace.h"

#include "colmap/mvs/bounded_workspace.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/bitmap_reader.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <array>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <new>
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

TEST(BoundedWorkspaceBase, ReservationCoalescingAndPinnedEviction) {
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "images");
  CreateDirIfNotExists(root / "stereo");
  CreateDirIfNotExists(root / "stereo" / "depth_maps");
  CreateDirIfNotExists(root / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 2;
  dataset.camera_width = 10;
  dataset.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  std::vector<std::string> names;
  for (int image_id = 1; image_id <= 2; ++image_id) {
    const auto name = reconstruction.Image(image_id).Name();
    names.push_back(name);
    DepthMap depth(10, 5, 1.0f, 10.0f);
    depth.Fill(static_cast<float>(image_id + 1));
    depth.Write(root / "stereo" / "depth_maps" / (name + ".geometric.bin"));
    if (image_id == 1) {
      // The native FP32 reader accepts trailing bytes; admission must agree.
      std::ofstream tail(
          root / "stereo" / "depth_maps" / (name + ".geometric.bin"),
          std::ios::app | std::ios::binary);
      tail.put('x');
    }
    NormalMap normal(10, 5);
    normal.Fill(1.0f);
    if (image_id == 2) normal.Set(1, 3, 2, 2.0f);
    normal.Write(root / "stereo" / "normal_maps" / (name + ".geometric.bin"));
    Bitmap bitmap(10, 5, true);
    bitmap.Fill(BitmapColor<uint8_t>(image_id, image_id, image_id));
    ASSERT_TRUE(bitmap.Write(root / "images" / name));
  }
  Workspace::Options options;
  options.workspace_path = root;
  options.workspace_format = "COLMAP";
  options.input_type = "geometric";
  options.cache_size = 951.0 / (1024.0 * 1024.0 * 1024.0);
  BoundedWorkspace workspace(options);
  const int first = workspace.GetModel().GetImageIdx(names[0]);
  const int second = workspace.GetModel().GetImageIdx(names[1]);
  EXPECT_EQ(workspace.GetLoadReservationBytes(first), 950);
  EXPECT_EQ(workspace.GetCacheStats().loads, 0);
  auto pin = workspace.AcquireImage(first);
  auto copy = pin;
  EXPECT_EQ(pin.get(), copy.get());
  EXPECT_EQ(workspace.GetCacheStats().loads, 1);
  auto waiting = std::async(std::launch::async,
                            [&] { return workspace.AcquireImage(second); });
  EXPECT_EQ(waiting.wait_for(std::chrono::milliseconds(30)),
            std::future_status::timeout);
  pin.reset();
  copy.reset();
  auto next = waiting.get();
  ASSERT_TRUE(next);
  EXPECT_EQ(next->depth_map.Get(0, 0), 3.0f);
  const auto stats = workspace.GetCacheStats();
  EXPECT_EQ(stats.loads, 2);
  EXPECT_EQ(stats.evictions, 1);
  EXPECT_EQ(stats.reserved_bytes, 0);
  EXPECT_LE(stats.peak_charged_bytes, stats.capacity_bytes);
  next.reset();
  std::vector<std::future<void>> readers;
  for (int i = 0; i < 8; ++i) {
    readers.push_back(std::async(std::launch::async, [&] {
      const auto same = workspace.AcquireImage(first);
      EXPECT_EQ(same->depth_map.Get(0, 0), 2.0f);
    }));
  }
  for (auto& reader : readers) reader.get();
  EXPECT_EQ(workspace.GetCacheStats().loads, 3);

  BoundedWorkspace components(options);
  using Component = BoundedWorkspace::Component;
  EXPECT_EQ(components.GetLoadReservationBytes(first, Component::kDepth), 200);
  EXPECT_EQ(components.GetLoadReservationBytes(first, Component::kNormal), 600);
  EXPECT_EQ(components.GetLoadReservationBytes(first, Component::kBitmap), 150);
  auto depth_pin = components.AcquireComponent(first, Component::kDepth);
  auto normal_pin = components.AcquireComponent(first, Component::kNormal);
  auto bitmap_pin = components.AcquireComponent(first, Component::kBitmap);
  ASSERT_TRUE(depth_pin && normal_pin && bitmap_pin);
  EXPECT_EQ(depth_pin->depth_map.Get(0, 0), 2.0f);
  EXPECT_EQ(normal_pin->normal_map.Get(0, 0, 2), 1.0f);
  EXPECT_EQ(bitmap_pin->bitmap.GetPixel(0, 0)->r, 1);
  auto component_wait = std::async(std::launch::async, [&] {
    return components.AcquireComponent(second, Component::kDepth);
  });
  EXPECT_EQ(component_wait.wait_for(std::chrono::milliseconds(30)),
            std::future_status::timeout);
  normal_pin.reset();
  auto second_depth = component_wait.get();
  ASSERT_TRUE(second_depth);
  EXPECT_EQ(second_depth->depth_map.Get(0, 0), 3.0f);
  EXPECT_EQ(depth_pin->depth_map.Get(0, 0), 2.0f);
  second_depth.reset();
  auto normal_again = components.AcquireComponent(first, Component::kNormal);
  ASSERT_TRUE(normal_again);
  EXPECT_EQ(normal_again->normal_map.Get(0, 0, 2), 1.0f);
  const auto component_stats = components.GetCacheStats();
  EXPECT_EQ(
      component_stats.component_loads[static_cast<size_t>(Component::kDepth)],
      2);
  EXPECT_EQ(
      component_stats.component_loads[static_cast<size_t>(Component::kNormal)],
      2);
  EXPECT_EQ(
      component_stats.component_loads[static_cast<size_t>(Component::kBitmap)],
      1);
  EXPECT_GE(component_stats.evictions, 2);
  EXPECT_LE(component_stats.peak_charged_bytes, component_stats.capacity_bytes);

  if (ImmutableMapRegion::Supported()) {
    // The decoded cache can hold only one 2x2 tile. Replacing that tile also
    // forces the one-file mapped owner to evict, while an old decoded lease
    // remains valid until its caller releases it.
    options.cache_size = 49.0 / (1024.0 * 1024.0 * 1024.0);
    const size_t mapped_file_bytes =
        ImmutableMapRegion::GetMappingReservationBytes(
            root / "stereo" / "normal_maps" / (names[0] + ".geometric.bin"));
    options.mapped_normal_bytes_limit = mapped_file_bytes - 1;
    options.mapped_normal_files_limit = 1;
    {
      BoundedWorkspace denied(options);
      for (size_t col = 0; col < 6; col += 2) {
        EXPECT_THROW(denied.AcquireNormalTile(first, 0, col, 2),
                     std::exception);
        EXPECT_EQ(denied.GetCacheStats().reserved_bytes, 0);
        EXPECT_EQ(denied.GetTileMetadataStats().live, 0);
        EXPECT_LE(denied.GetTileMetadataStats().slot_count, 1);
      }
    }
    options.mapped_normal_bytes_limit = mapped_file_bytes;
    options.mapped_normal_files_limit = 1;
    {
      BoundedWorkspace tiles(options);
      tiles.FailNextTileKeyPublicationForTesting();
      EXPECT_THROW(tiles.AcquireNormalTile(first, 1, 1, 2), std::bad_alloc);
      EXPECT_EQ(tiles.GetTileMetadataStats().slot_count, 0);
      EXPECT_EQ(tiles.GetTileMetadataStats().free, 0);
      EXPECT_EQ(tiles.GetTileMetadataStats().live, 0);
      EXPECT_EQ(tiles.GetCacheStats().reserved_bytes, 0);
      auto tile = tiles.AcquireNormalTile(first, 1, 1, 2);
      auto same_tile = tiles.AcquireNormalTile(first, 0, 0, 2);
      ASSERT_EQ(tile.get(), same_tile.get());
      EXPECT_EQ(tile->GetNormal(1, 1).z(), 1.0f);
      auto next_tile = std::async(std::launch::async, [&] {
        return tiles.AcquireNormalTile(second, 1, 1, 2);
      });
      auto coalesced_tile = std::async(std::launch::async, [&] {
        return tiles.AcquireNormalTile(second, 1, 1, 2);
      });
      const auto wait_deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (tiles.GetCacheStats().waits < 2 &&
             std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      EXPECT_GE(tiles.GetCacheStats().waits, 2);
      EXPECT_EQ(next_tile.wait_for(std::chrono::milliseconds(30)),
                std::future_status::timeout);
      EXPECT_EQ(coalesced_tile.wait_for(std::chrono::milliseconds(30)),
                std::future_status::timeout);
      tile.reset();
      same_tile.reset();
      auto second_tile = next_tile.get();
      auto second_copy = coalesced_tile.get();
      ASSERT_TRUE(second_tile);
      ASSERT_EQ(second_tile.get(), second_copy.get());
      EXPECT_EQ(second_tile->GetNormal(1, 1).z(), 1.0f);
      second_tile.reset();
      second_copy.reset();
      auto recycled_tile = tiles.AcquireNormalTile(second, 1, 3, 2);
      EXPECT_EQ(recycled_tile->GetNormal(1, 3).z(), 2.0f);
      recycled_tile.reset();
      const auto tile_stats = tiles.GetCacheStats();
      EXPECT_EQ(tile_stats.normal_tile_loads, 3);
      EXPECT_GE(tile_stats.normal_tile_hits, 1);
      EXPECT_GE(tile_stats.evictions, 1);
      EXPECT_EQ(tile_stats.reserved_bytes, 0);
      EXPECT_LE(tiles.GetTileMetadataStats().slot_count, 2);
      EXPECT_LE(tiles.GetTileMetadataStats().live, 1);
      const auto mapped_stats = tiles.GetMappedNormalStats();
      EXPECT_EQ(mapped_stats.loads, 2);
      EXPECT_EQ(mapped_stats.evictions, 1);
      EXPECT_EQ(mapped_stats.active_copies, 0);
      EXPECT_EQ(mapped_stats.reserved_bytes, 0);
      EXPECT_LE(mapped_stats.peak_charged_bytes, mapped_file_bytes);
    }
    {
      // A direct sample pins its owner through eviction. Releasing the only
      // file slot lets a waiting view progress without decoded tile entries.
      BoundedWorkspace mapped(options);
      EXPECT_EQ(mapped.GetMappedNormalReservationBytes(first),
                mapped_file_bytes);
      auto owner = mapped.AcquireMappedNormal(first);
      auto same_owner = mapped.AcquireMappedNormal(first);
      ASSERT_EQ(owner.get(), same_owner.get());
      EXPECT_EQ(owner->ReadFloat3(1, 1).z(), 1.0f);
      auto waiting = std::async(std::launch::async, [&] {
        return mapped.AcquireMappedNormal(second);
      });
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (mapped.GetMappedNormalStats().waits == 0 &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      EXPECT_GE(mapped.GetMappedNormalStats().waits, 1);
      EXPECT_EQ(waiting.wait_for(std::chrono::milliseconds(30)),
                std::future_status::timeout);
      owner.reset();
      same_owner.reset();
      auto replacement = waiting.get();
      ASSERT_TRUE(replacement);
      EXPECT_EQ(replacement->ReadFloat3(1, 3).z(), 2.0f);
      replacement.reset();
      const auto direct_stats = mapped.GetMappedNormalStats();
      EXPECT_EQ(direct_stats.loads, 2);
      EXPECT_EQ(direct_stats.evictions, 1);
      EXPECT_EQ(direct_stats.source_loads, 2);
      EXPECT_EQ(direct_stats.source_evictions, 1);
      EXPECT_LE(direct_stats.source_peak_charged_files, 1);
      EXPECT_LE(direct_stats.source_peak_live_files, 1);
      EXPECT_EQ(direct_stats.active_copies, 0);
      EXPECT_EQ(direct_stats.reserved_bytes, 0);
      EXPECT_LE(direct_stats.peak_charged_bytes, mapped_file_bytes);
      EXPECT_EQ(mapped.GetTileMetadataStats().slot_count, 0);
    }
  }

  options.mapped_normal_bytes_limit = 0;
  options.mapped_normal_files_limit = 0;

  // A tile may not reinterpret a normal map stored on a different grid.
  NormalMap wrong_grid(9, 5);
  wrong_grid.Fill(1.0f);
  wrong_grid.Write(root / "stereo" / "normal_maps" /
                   (names[1] + ".geometric.bin"));
  BoundedWorkspace mismatched(options);
  EXPECT_THROW(mismatched.GetNormalTileGrid(second), std::exception);
  EXPECT_EQ(mismatched.GetCacheStats().loads, 0);

  options.cache_size = 949.0 / (1024.0 * 1024.0 * 1024.0);
  BoundedWorkspace too_small(options);
  EXPECT_THROW(too_small.AcquireImage(first), std::runtime_error);
  EXPECT_EQ(too_small.GetCacheStats().loads, 0);
  {
    std::ofstream malformed_normal(
        root / "stereo" / "normal_maps" / (names[0] + ".geometric.bin"),
        std::ios::trunc | std::ios::binary);
    malformed_normal << std::string(129, ' ');
  }
  // Component requests only require their own input. A damaged normal must
  // not prevent an otherwise valid depth or bitmap component from loading.
  BoundedWorkspace selective(options);
  EXPECT_EQ(
      selective.AcquireComponent(first, Component::kDepth)->depth_map.Get(0, 0),
      2.0f);
  EXPECT_EQ(selective.AcquireComponent(first, Component::kBitmap)
                ->bitmap.GetPixel(0, 0)
                ->r,
            1);
  EXPECT_THROW(selective.AcquireComponent(first, Component::kNormal),
               std::exception);
  EXPECT_THROW(selective.AcquireImage(first), std::exception);
  {
    std::ofstream malformed(
        root / "stereo" / "depth_maps" / (names[0] + ".geometric.bin"),
        std::ios::trunc | std::ios::binary);
    malformed << std::string(129, ' ');
  }
  BoundedWorkspace malformed(options);
  EXPECT_THROW(malformed.GetLoadReservationBytes(first), std::exception);
  EXPECT_EQ(malformed.GetCacheStats().loads, 0);
}

TEST(BoundedWorkspaceBase, NormalTileGridValidationRetriesAndCoalesces) {
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "images");
  CreateDirIfNotExists(root / "stereo");
  CreateDirIfNotExists(root / "stereo" / "depth_maps");
  CreateDirIfNotExists(root / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 1;
  dataset.camera_width = 10;
  dataset.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  const auto name = reconstruction.Image(1).Name();
  const auto depth_path =
      root / "stereo" / "depth_maps" / (name + ".geometric.bin");
  const auto normal_path =
      root / "stereo" / "normal_maps" / (name + ".geometric.bin");
  DepthMap depth(10, 5, 1.0f, 10.0f);
  depth.Fill(2.0f);
  depth.Write(depth_path);
  {
    std::ofstream malformed(normal_path, std::ios::binary);
    malformed << std::string(129, ' ');
  }
  Workspace::Options options;
  options.workspace_path = root;
  options.workspace_format = "COLMAP";
  options.input_type = "geometric";
  {
    BoundedWorkspace rejected(options);
    const int image_idx = rejected.GetModel().GetImageIdx(name);
    // A thrown first admission must not publish an empty shape or poison the
    // once flag: both attempts still reject the malformed source.
    EXPECT_THROW(rejected.GetNormalTileGrid(image_idx), std::exception);
    EXPECT_THROW(rejected.GetNormalTileGrid(image_idx), std::exception);
    EXPECT_THROW(rejected.AcquireMappedNormal(image_idx), std::exception);
    EXPECT_EQ(rejected.GetMappedNormalStats().loads, 0);
  }

  NormalMap normal(10, 5);
  normal.Fill(1.0f);
  normal.Write(normal_path);
  {
    BoundedWorkspace valid(options);
    const int image_idx = valid.GetModel().GetImageIdx(name);
    std::promise<void> start;
    const auto ready = start.get_future().share();
    std::vector<std::future<std::pair<size_t, size_t>>> readers;
    for (int i = 0; i < 8; ++i) {
      readers.push_back(std::async(std::launch::async, [&] {
        ready.wait();
        return valid.GetNormalTileGrid(image_idx);
      }));
    }
    start.set_value();
    for (auto& reader : readers) {
      EXPECT_EQ(reader.get(), std::make_pair(size_t{10}, size_t{5}));
    }
    EXPECT_EQ(valid.GetNormalTileGrid(image_idx),
              std::make_pair(size_t{10}, size_t{5}));
    if (ImmutableMapRegion::Supported()) {
      auto mapped_options = options;
      mapped_options.mapped_normal_bytes_limit = 1 << 20;
      mapped_options.mapped_normal_files_limit = 2;
      BoundedWorkspace mapped(mapped_options);
      std::vector<std::future<BoundedWorkspace::MappedNormalLease>> owners;
      for (int i = 0; i < 8; ++i) {
        owners.push_back(std::async(std::launch::async, [&] {
          return mapped.AcquireMappedNormal(image_idx);
        }));
      }
      const auto first_owner = owners.front().get();
      for (size_t i = 1; i < owners.size(); ++i) {
        EXPECT_EQ(owners[i].get().get(), first_owner.get());
      }
      EXPECT_EQ(first_owner->ReadFloat3(0, 0).z(), 1.0f);
      EXPECT_EQ(mapped.GetMappedNormalStats().loads, 1);
      EXPECT_EQ(mapped.GetMappedNormalStats().source_loads, 1);
    }
  }

  // Input files are immutable while a Workspace owns them. A fresh instance
  // validates a changed source instead of inheriting an earlier shape.
  NormalMap wrong_grid(9, 5);
  wrong_grid.Fill(1.0f);
  wrong_grid.Write(normal_path);
  BoundedWorkspace changed(options);
  EXPECT_THROW(changed.GetNormalTileGrid(changed.GetModel().GetImageIdx(name)),
               std::exception);
}

TEST(BoundedWorkspaceBase, LeaseOutlivesWorkspace) {
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "images");
  CreateDirIfNotExists(root / "stereo");
  CreateDirIfNotExists(root / "stereo" / "depth_maps");
  CreateDirIfNotExists(root / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 1;
  dataset.camera_width = 10;
  dataset.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  const auto name = reconstruction.Image(1).Name();
  Bitmap bitmap(10, 5, true);
  bitmap.Fill(BitmapColor<uint8_t>(7, 8, 9));
  ASSERT_TRUE(bitmap.Write(root / "images" / name));
  DepthMap depth(10, 5, 1.0f, 10.0f);
  depth.Fill(2.0f);
  NormalMap normal(10, 5);
  normal.Fill(1.0f);
  const auto depth_path =
      root / "stereo" / "depth_maps" / (name + ".geometric.bin");
  const auto normal_path =
      root / "stereo" / "normal_maps" / (name + ".geometric.bin");
  depth.Write(depth_path);
  normal.Write(normal_path);
  Workspace::Options options;
  options.workspace_path = root;
  options.workspace_format = "COLMAP";
  options.input_type = "geometric";
  BoundedWorkspace::ImageLease lease;
  {
    auto workspace = std::make_unique<BoundedWorkspace>(options);
    const int image = workspace->GetModel().GetImageIdx(name);
    lease = workspace->AcquireImage(image);
    ASSERT_TRUE(lease);
  }
  EXPECT_EQ(lease->depth_map.Get(0, 0), 2.0f);
  EXPECT_EQ(lease->bitmap.GetPixel(0, 0)->g, 8);
  lease.reset();
}

TEST(BoundedWorkspaceResident, CompactRgbAdmissionAndExactSamples) {
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "images");
  CreateDirIfNotExists(root / "stereo");
  CreateDirIfNotExists(root / "stereo" / "depth_maps");
  CreateDirIfNotExists(root / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 1;
  dataset.camera_width = 10;
  dataset.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  const auto name = reconstruction.Image(1).Name();
  Workspace::Options options;
  options.workspace_path = root;
  options.workspace_format = "COLMAP";
  options.input_type = "geometric";
  options.compact_rgb_to_depth_grid = true;
  const Workspace descriptor(options);
  const auto bitmap_path = descriptor.GetBitmapPath(0);
  auto original = std::make_shared<Bitmap>(10, 5, true);
  for (int row = 0; row < 5; ++row) {
    for (int col = 0; col < 10; ++col) {
      original->SetPixel(
          col,
          row,
          BitmapColor<uint8_t>(
              col * 23 + row, row * 47 + col, col * 13 + row * 7));
    }
  }
  ASSERT_TRUE(original->Write(bitmap_path));
  auto scope = ScopedBitmapReader::Create(
      {bitmap_path},
      [owner = original](const auto&, bool rgb, bool linear, Bitmap* target) {
        EXPECT_TRUE(rgb);
        EXPECT_FALSE(linear);
        *target = *owner;
        return true;
      },
      {{10, 5}});
  const Bitmap golden = *original;
  original.reset();  // The registered callback now owns its source pixels.
  ASSERT_TRUE(std::filesystem::remove(bitmap_path));
  ASSERT_TRUE(descriptor.HasBitmap(0));
  const int image = descriptor.GetModel().GetImageIdx(name);

  for (const auto grid :
       {std::pair{10, 5}, {7, 3}, {1, 1}, {10, 1}, {1, 5}, {9, 4}}) {
    Mat<float> depth(grid.first, grid.second, 1);
    depth.Fill(2.0f);
    Mat<float> normal(grid.first, grid.second, 3);
    normal.Fill(1.0f);
    depth.Write(descriptor.GetDepthMapPath(image));
    normal.Write(descriptor.GetNormalMapPath(image));
    const bool compacted = grid != std::pair{10, 5};
    const size_t pixels = grid.first * grid.second;
    const size_t reservation = 150 + pixels * (16 + (compacted ? 3 : 0));
    options.cache_size =
        static_cast<double>(reservation + 1) / (1024.0 * 1024.0 * 1024.0);
    BoundedWorkspace workspace(options);
    EXPECT_EQ(workspace.GetLoadReservationBytes(image), reservation);
    EXPECT_EQ(workspace.GetCacheStats().loads, 0);
    auto lease = workspace.AcquireImage(image);
    ASSERT_TRUE(lease);
    EXPECT_EQ(lease->bitmap.Width(), grid.first);
    EXPECT_EQ(lease->bitmap.Height(), grid.second);
    EXPECT_EQ(lease->bitmap_on_depth_grid, compacted);
    EXPECT_EQ(lease->NumBytes(), pixels * 19);
    const float sx = static_cast<float>(grid.first) / 10;
    const float sy = static_cast<float>(grid.second) / 5;
    for (int row = 0; row < grid.second; ++row) {
      for (int col = 0; col < grid.first; ++col) {
        const auto expected =
            golden.InterpolateNearestNeighbor(col / sx, row / sy)
                .value_or(BitmapColor<uint8_t>(0));
        const auto actual = lease->bitmap.GetPixel(col, row).value();
        EXPECT_EQ(actual.r, expected.r);
        EXPECT_EQ(actual.g, expected.g);
        EXPECT_EQ(actual.b, expected.b);
      }
    }
    const auto stats = workspace.GetCacheStats();
    EXPECT_EQ(stats.peak_charged_bytes, reservation);
    EXPECT_EQ(stats.reserved_bytes, 0);
    EXPECT_EQ(stats.resident_bytes, pixels * 19);
    lease.reset();
    options.cache_size =
        static_cast<double>(reservation - 1) / (1024.0 * 1024.0 * 1024.0);
    BoundedWorkspace too_small(options);
    EXPECT_THROW(too_small.AcquireImage(image), std::runtime_error);
    EXPECT_EQ(too_small.GetCacheStats().loads, 0);
  }
  scope->Close();
  scope.reset();
  EXPECT_FALSE(descriptor.HasBitmap(image));
  auto no_header = ScopedBitmapReader::Create(
      {bitmap_path}, [](const auto&, bool, bool, Bitmap*) { return false; });
  BoundedWorkspace no_metadata(options);
  EXPECT_THROW(no_metadata.GetLoadReservationBytes(image),
               std::invalid_argument);
  EXPECT_EQ(no_metadata.GetCacheStats().loads, 0);
  no_header->Close();
  no_header.reset();

  Bitmap disk(10, 5, true);
  disk.Fill(BitmapColor<uint8_t>(7, 8, 9));
  ASSERT_TRUE(disk.Write(bitmap_path));
  auto failed = ScopedBitmapReader::Create(
      {bitmap_path},
      [](const auto&, bool, bool, Bitmap*) { return false; },
      {{10, 5}});
  options.cache_size = 1000.0 / (1024.0 * 1024.0 * 1024.0);
  BoundedWorkspace managed_failure(options);
  EXPECT_THROW(managed_failure.AcquireImage(image), std::exception);
  failed->Close();
  failed.reset();
  BoundedWorkspace ordinary_disk(options);
  EXPECT_EQ(ordinary_disk.AcquireImage(image)->bitmap.GetPixel(0, 0)->g, 8);
}

TEST(BoundedWorkspaceResident, ScopeAndPixelsSurviveConcurrentEviction) {
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "images");
  CreateDirIfNotExists(root / "stereo");
  CreateDirIfNotExists(root / "stereo" / "depth_maps");
  CreateDirIfNotExists(root / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 2;
  dataset.camera_width = 10;
  dataset.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  Workspace::Options options;
  options.workspace_path = root;
  options.workspace_format = "COLMAP";
  options.input_type = "geometric";
  options.cache_size = 951.0 / (1024.0 * 1024.0 * 1024.0);
  Workspace descriptor(options);
  std::vector<std::filesystem::path> paths;
  std::vector<int> images;
  auto owners = std::make_shared<std::vector<Bitmap>>();
  for (int image_id = 1; image_id <= 2; ++image_id) {
    const auto name = reconstruction.Image(image_id).Name();
    const int image = descriptor.GetModel().GetImageIdx(name);
    images.push_back(image);
    paths.push_back(descriptor.GetBitmapPath(image));
    DepthMap depth(10, 5, 1.0f, 10.0f);
    depth.Fill(static_cast<float>(image_id + 1));
    depth.Write(descriptor.GetDepthMapPath(image));
    NormalMap normal(10, 5);
    normal.Fill(1.0f);
    normal.Write(descriptor.GetNormalMapPath(image));
    Bitmap bitmap(10, 5, true);
    bitmap.Fill(BitmapColor<uint8_t>(image_id, image_id, image_id));
    owners->push_back(bitmap);
  }
  std::weak_ptr<std::vector<Bitmap>> weak_owner = owners;
  auto scope = ScopedBitmapReader::Create(
      paths,
      [strong = owners, paths](const auto& path, bool, bool, Bitmap* target) {
        for (size_t i = 0; i < paths.size(); ++i) {
          if (path == std::filesystem::absolute(paths[i]).lexically_normal()) {
            *target = strong->at(i);
            return true;
          }
        }
        return false;
      },
      {{10, 5}, {10, 5}});
  owners.reset();
  ASSERT_FALSE(weak_owner.expired());
  ASSERT_TRUE(descriptor.HasBitmap(images[0]));
  ASSERT_TRUE(descriptor.HasBitmap(images[1]));
  BoundedWorkspace workspace(options);
  auto held = workspace.AcquireImage(images[0]);
  auto copied_pin = held;
  auto waiting = std::async(std::launch::async,
                            [&] { return workspace.AcquireImage(images[1]); });
  EXPECT_EQ(waiting.wait_for(std::chrono::milliseconds(30)),
            std::future_status::timeout);
  held.reset();
  copied_pin.reset();
  auto second = waiting.get();  // Worker has joined before closing the scope.
  EXPECT_EQ(workspace.GetCacheStats().evictions, 1);
  scope->Close();
  scope.reset();
  EXPECT_TRUE(weak_owner.expired());
  EXPECT_FALSE(descriptor.HasBitmap(images[0]));
  EXPECT_FALSE(descriptor.HasBitmap(images[1]));
  ASSERT_TRUE(second);
  EXPECT_EQ(second->bitmap.GetPixel(0, 0)->r, 2);
  EXPECT_EQ(second->depth_map.Get(0, 0), 3.0f);
}

TEST(BoundedWorkspaceBase, MappedNormalStripeKeysPinAcrossBands) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "images");
  CreateDirIfNotExists(root / "stereo");
  CreateDirIfNotExists(root / "stereo" / "depth_maps");
  CreateDirIfNotExists(root / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 1;
  dataset.camera_width = 10;
  dataset.camera_height = 260;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  const auto name = reconstruction.Image(1).Name();
  DepthMap depth(10, 260, 1.0f, 10.0f);
  depth.Fill(2.0f);
  depth.Write(root / "stereo" / "depth_maps" / (name + ".geometric.bin"));
  const auto normal_path =
      root / "stereo" / "normal_maps" / (name + ".geometric.bin");
  NormalMap normal(10, 260);
  normal.Fill(1.0f);
  normal.Set(256, 9, 2, 2.0f);
  normal.Write(normal_path);
  Workspace::Options options;
  options.workspace_path = root;
  options.workspace_format = "COLMAP";
  options.input_type = "geometric";
  const size_t first_bytes =
      ImmutableMapRegion::GetWindowReservationBytes(normal_path, 0, 256);
  const size_t edge_bytes =
      ImmutableMapRegion::GetWindowReservationBytes(normal_path, 256, 4);
  options.mapped_normal_bytes_limit = std::max(first_bytes, edge_bytes);
  options.mapped_normal_files_limit = 1;
  {
    BoundedWorkspace workspace(options);
    const int image = workspace.GetModel().GetImageIdx(name);
    auto first = workspace.AcquireMappedNormalWindow(image, 0, 256);
    auto same = workspace.AcquireMappedNormalWindow(image, 0, 256);
    ASSERT_EQ(first.get(), same.get());
    EXPECT_EQ(first->ReadFloat3(255, 9).z(), 1.0f);
    EXPECT_THROW(first->ReadFloat3(256, 9), std::exception);
    auto waiting = std::async(std::launch::async, [&] {
      return workspace.AcquireMappedNormalWindow(image, 256, 4);
    });
    auto coalesced = std::async(std::launch::async, [&] {
      return workspace.AcquireMappedNormalWindow(image, 256, 4);
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (workspace.GetMappedNormalStats().waits < 2 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_GE(workspace.GetMappedNormalStats().waits, 2);
    first.reset();
    same.reset();
    auto edge = waiting.get();
    auto edge_copy = coalesced.get();
    ASSERT_EQ(edge.get(), edge_copy.get());
    EXPECT_EQ(edge->ReadFloat3(256, 9).z(), 2.0f);
    const auto stats = workspace.GetMappedNormalStats();
    EXPECT_EQ(stats.loads, 2);
    EXPECT_EQ(stats.source_loads, 1);
    EXPECT_GE(stats.source_hits, 1);
    EXPECT_LE(stats.source_peak_charged_files, 1);
    EXPECT_EQ(stats.source_live_files, 1);
    EXPECT_GE(stats.hits, 2);
    EXPECT_EQ(stats.evictions, 1);
    EXPECT_EQ(stats.resident_files, 1);
    EXPECT_EQ(stats.resident_regions, 3);
    EXPECT_EQ(stats.peak_charged_regions, 3);
    EXPECT_LE(stats.peak_charged_bytes, stats.capacity_bytes);
    EXPECT_EQ(workspace.GetTileMetadataStats().slot_count, 0);
  }
  // Repair is only admitted by a fresh immutable Workspace, never by a
  // pinned owner observing a file rewrite mid-lifetime.
  {
    std::ofstream bad(normal_path, std::ios::trunc | std::ios::binary);
    bad << std::string(129, ' ');
  }
  {
    BoundedWorkspace bad(options);
    const int image = bad.GetModel().GetImageIdx(name);
    EXPECT_THROW(bad.AcquireMappedNormalWindow(image, 0, 256), std::exception);
    EXPECT_EQ(bad.GetMappedNormalStats().reserved_bytes, 0);
    EXPECT_EQ(bad.GetMappedNormalStats().source_charged_files, 0);
    EXPECT_THROW(bad.AcquireMappedNormalWindow(image, 0, 256), std::exception);
    EXPECT_EQ(bad.GetMappedNormalStats().source_charged_files, 0);
  }
  normal.Write(normal_path);
  BoundedWorkspace repaired(options);
  const int image = repaired.GetModel().GetImageIdx(name);
  EXPECT_EQ(
      repaired.AcquireMappedNormalWindow(image, 256, 4)->ReadFloat3(256, 9).z(),
      2.0f);
}

TEST(BoundedWorkspaceBase, SignedNormalTileValidationAndExactFallback) {
  if (!ImmutableMapRegion::Supported())
    GTEST_SKIP() << "Immutable FP32 map owners are unsupported on this host";
  const auto root = CreateTestDir();
  CreateDirIfNotExists(root / "sparse");
  CreateDirIfNotExists(root / "stereo");
  CreateDirIfNotExists(root / "stereo" / "depth_maps");
  CreateDirIfNotExists(root / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 1;
  dataset.camera_width = 8;
  dataset.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  const auto name = reconstruction.Image(1).Name();
  Mat<float> depth(8, 5, 1);
  depth.Fill(2.0f);
  depth.Write(root / "stereo" / "depth_maps" / (name + ".geometric.bin"));
  Mat<float> normal(8, 5, 3);
  normal.Fill(0.5f);
  const auto normal_path =
      root / "stereo" / "normal_maps" / (name + ".geometric.bin");
  normal.Write(normal_path);
  Workspace::Options options;
  options.workspace_path = root;
  options.workspace_format = "COLMAP";
  options.input_type = "geometric";
  options.cache_size = 4096.0 / (1024.0 * 1024.0 * 1024.0);
  options.normal_tiles_snorm16 = true;
  options.mapped_normal_bytes_limit = 1 << 20;
  options.mapped_normal_files_limit = 2;
  {
    BoundedWorkspace workspace(options);
    const int image = workspace.GetModel().GetImageIdx(name);
    EXPECT_EQ(workspace.GetNormalTileReservationBytes(image, 4), 288);
    EXPECT_EQ(workspace.GetNormalTileResidentBytes(image, 4), 96);
    auto tile = workspace.AcquireNormalTile(image, 1, 1, 4);
    ASSERT_TRUE(tile);
    EXPECT_TRUE(tile->normal_map.GetData().empty());
    EXPECT_EQ(tile->snorm_normals.size(), 4u * 4u * 3u);
    EXPECT_NE(tile->GetNormal(1, 1).x(), 0.5f);
    EXPECT_EQ(workspace.ReadOriginalNormal(image, 1, 1).x(), 0.5f);
    EXPECT_EQ(workspace.GetMappedNormalStats().active_copies, 0);
    for (size_t row = 0; row < 5; ++row) {
      for (size_t col = 0; col < 8; ++col) {
        const auto original = workspace.ReadOriginalNormal(image, row, col);
        for (int channel = 0; channel < 3; ++channel) {
          EXPECT_EQ(original[channel], 0.5f);
        }
        EXPECT_EQ(workspace.GetMappedNormalStats().active_copies, 0);
      }
    }
    EXPECT_THROW(workspace.ReadOriginalNormal(image, 5, 0), std::exception);
    EXPECT_EQ(workspace.GetMappedNormalStats().active_copies, 0);
    EXPECT_THROW(workspace.ReadOriginalNormal(image, 0, 8), std::exception);
    EXPECT_EQ(workspace.GetMappedNormalStats().active_copies, 0);
    tile.reset();
    EXPECT_EQ(workspace.GetMappedNormalStats().active_copies, 0);
  }
  // Mutate only after all leases from the first immutable Workspace end.
  normal.Set(0, 0, 0, 2.0f);
  normal.Write(normal_path);
  BoundedWorkspace invalid(options);
  const int image = invalid.GetModel().GetImageIdx(name);
  EXPECT_THROW(invalid.AcquireNormalTile(image, 0, 0, 4), std::exception);
  EXPECT_EQ(invalid.GetCacheStats().reserved_bytes, 0);
  EXPECT_THROW(invalid.AcquireNormalTile(image, 0, 0, 4), std::exception);
  EXPECT_EQ(invalid.GetCacheStats().reserved_bytes, 0);
}

TEST_F(WorkspaceTests, ResidentGuardCertificateAndIdentityRefusal) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  ASSERT_TRUE(normal_precision_guard::SupportedBuild());
  auto options = GetOptions();
  options.num_threads = 1;
  options.full_ram_normal_storage = NormalStorage::kOct2x16;
  options.full_ram_normal_guard = true;
  options.full_ram_normal_guard_bytes = 1 << 20;
  options.full_ram_normal_guard_files = 1;
  Workspace workspace(options);
  const int image = workspace.GetModel().GetImageIdx(image_name_);
  workspace.Load({image_name_});
  const auto error = workspace.GetStoredNormalError(image);
  const auto decoded = workspace.GetStoredNormal(image, 0, 0);
  const Eigen::Vector3f original = Eigen::Vector3f::Ones();
  EXPECT_EQ(workspace.GetOriginalNormalMappingStats().loads, 0);
  for (int channel = 0; channel < 3; ++channel)
    EXPECT_GE(error(channel),
              std::abs(double(original(channel)) - decoded(channel)));
  EXPECT_EQ(workspace.GetOriginalNormalMappingStats().active_copies, 0);

  // Replacement after eager pack must refuse; same dimensions do not establish
  // identity.
  const auto path = workspace.GetNormalMapPath(image);
  const auto saved = path.string() + ".saved";
  std::filesystem::rename(path, saved);
  NormalMap replacement(10, 5);
  replacement.Fill(0.25f);
  replacement.Write(path);
  EXPECT_THROW(workspace.ReadOriginalStoredNormal(image, 0, 0), std::exception);
  EXPECT_THROW(workspace.ValidateOriginalNormalSources(), std::exception);
  workspace.ReleaseOriginalNormalSources();
  EXPECT_THROW(workspace.GetStoredNormalError(image), std::exception);
  std::filesystem::remove(path);
  std::filesystem::rename(saved, path);

  // A new generation admits the restored inode, then detects in-place payload
  // change.
  workspace.Load({image_name_});
  EXPECT_TRUE((workspace.ReadOriginalStoredNormal(image, 0, 0).array() ==
               Eigen::Vector3f::Ones().array())
                  .all());
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file);
    file.seekp(-static_cast<std::streamoff>(sizeof(float)), std::ios::end);
    const float changed = 0.125f;
    file.write(reinterpret_cast<const char*>(&changed), sizeof(changed));
    file.flush();
    ASSERT_TRUE(file);
  }
  EXPECT_THROW(workspace.ReadOriginalStoredNormal(image, 0, 0), std::exception);
}

TEST_F(WorkspaceTests, ResidentGuardGridAndByteAdmissionRefuse) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  ASSERT_TRUE(normal_precision_guard::SupportedBuild());
  auto options = GetOptions();
  options.num_threads = 1;
  options.full_ram_normal_storage = NormalStorage::kOct2x16;
  options.full_ram_normal_guard = true;
  options.full_ram_normal_guard_bytes = 1;
  options.full_ram_normal_guard_files = 1;
  Workspace too_small(options);
  EXPECT_THROW(too_small.Load({image_name_}), std::exception);
  options.full_ram_normal_guard_bytes = 1 << 20;
  options.max_image_size = 6;
  Workspace resized(options);
  EXPECT_THROW(resized.Load({image_name_}), std::exception);
  options.max_image_size = -1;
  Workspace descriptor(options);
  Mat<float> depth(9, 5, 1);
  depth.Fill(1);
  depth.Write(descriptor.GetDepthMapPath(0));
  EXPECT_THROW(descriptor.Load({image_name_}), std::exception);
  // A genuinely smaller original normal+depth grid is unchanged by native
  // Downsize.
  NormalMap normal(9, 5);
  normal.Fill(0.5f);
  normal.Write(descriptor.GetNormalMapPath(0));
  descriptor.Load({image_name_});
  EXPECT_EQ(descriptor.GetStoredNormalDimensions(0),
            (std::pair<size_t, size_t>{9, 5}));
  EXPECT_TRUE((descriptor.ReadOriginalStoredNormal(0, 0, 0).array() ==
               Eigen::Vector3f::Constant(0.5f).array())
                  .all());
}

TEST(Workspace, ResidentGuardPackRollbackAndIncompleteCertificate) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  ASSERT_TRUE(normal_precision_guard::SupportedBuild());
  const auto path = CreateTestDir() / "normal.bin";
  NormalMap original(2, 1);
  original.Fill(0.5f);
  original.Write(path);
  NormalMapStorage storage;
  const auto source = std::make_shared<ImmutableMapSource>(
      path, ImmutableMapSource::BindIdentity{});
  storage.normal_map = original;
  const auto before = storage.normal_map.GetData();
  normal_precision_guard::CodecErrorCertificate incomplete(2);
  incomplete.AddDecoded(Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
  EXPECT_THROW(incomplete.Seal(), std::exception);
  const int rounding = std::fegetround();
  ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
  EXPECT_THROW(storage.ReadAndPackOriginal(source, 2, 1), std::exception);
  EXPECT_EQ(std::fesetround(rounding), 0);
  EXPECT_EQ(storage.normal_map.GetData(), before);
  EXPECT_TRUE(storage.oct_normals.empty());
  EXPECT_FALSE(storage.normal_error_valid);
  storage.ReadAndPackOriginal(source, 2, 1);
  EXPECT_TRUE(storage.normal_map.GetData().empty());
  EXPECT_TRUE(storage.normal_error_valid);
}

// Held-FD positional read prototype: generated files only, no real-map inputs.
TEST(Workspace, ResidentPositionalReadPreservesFloatBitsAndEdges) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  const auto path = CreateTestDir() / "normal.bin";
  Mat<float> original(3, 2, 3);
  const std::array<uint32_t, 8> bits = {0x00000000u,
                                        0x80000000u,
                                        0x00000001u,
                                        0x80000001u,
                                        0x3f800000u,
                                        0xbf800000u,
                                        0x3eaaaaabu,
                                        0x7f7fffffu};
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 3; ++col) {
        const uint32_t expected =
            bits[(channel * 6 + row * 3 + col) % bits.size()];
        float value;
        std::memcpy(&value, &expected, sizeof(value));
        original.Set(row, col, channel, value);
      }
    }
  }
  original.Write(path);
  auto source = std::make_shared<ImmutableMapSource>(
      path, ImmutableMapSource::BindIdentity{});
  for (size_t row = 0; row < 2; ++row) {
    ImmutableMapRegion mapped(source, row, 1);
    for (size_t col = 0; col < 3; ++col) {
      source->ValidateUnchanged();
      const Eigen::Vector3f direct = source->ReadFloat3Sample(row, col);
      source->ValidateUnchanged();
      const Eigen::Vector3f previous = mapped.ReadFloat3(row, col);
      for (size_t channel = 0; channel < 3; ++channel) {
        const float expected = original.Get(row, col, channel);
        EXPECT_EQ(std::memcmp(&direct[channel], &expected, sizeof(float)), 0);
        EXPECT_EQ(
            std::memcmp(&direct[channel], &previous[channel], sizeof(float)),
            0);
      }
    }
  }
  EXPECT_THROW(source->ReadFloat3Sample(2, 0), std::exception);
  EXPECT_THROW(source->ReadFloat3Sample(0, 3), std::exception);
  EXPECT_THROW(source->ReadFloat3Sample(std::numeric_limits<size_t>::max(), 0),
               std::exception);
}

TEST(Workspace, ResidentPositionalReadRefusesUnboundAndWrongShape) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  const auto path = CreateTestDir() / "depth.bin";
  Mat<float> depth(3, 2, 1);
  depth.Fill(1.0f);
  depth.Write(path);
  ImmutableMapSource wrong_shape(path, ImmutableMapSource::BindIdentity{});
  EXPECT_THROW(wrong_shape.ReadFloat3Sample(0, 0), std::exception);
  ImmutableMapSource unbound(path);
  EXPECT_THROW(unbound.ReadFloat3Sample(0, 0), std::exception);
  const auto malformed = path.string() + ".malformed";
  {
    std::ofstream file(malformed, std::ios::binary);
    file << "3&2&3&";
  }
  EXPECT_THROW(
      (ImmutableMapSource(malformed, ImmutableMapSource::BindIdentity{})),
      std::exception);
}

TEST(Workspace, ResidentPositionalReadRefusesPartialSampleAndMutation) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  const auto path = CreateTestDir() / "normal.bin";
  NormalMap original(3, 2);
  original.Fill(0.5f);
  original.Write(path);
  ImmutableMapSource source(path, ImmutableMapSource::BindIdentity{});
  const auto bytes = std::filesystem::file_size(path);
  // A two-byte final component makes pread return partial progress then EOF.
  // Deliberately call the leaf method without the caller's precheck to exercise
  // its short-read refusal; production Workspace always has both identity
  // checks.
  std::filesystem::resize_file(path, bytes - 2);
  EXPECT_THROW(source.ReadFloat3Sample(1, 2), std::exception);
  EXPECT_THROW(source.ValidateUnchanged(), std::exception);
  std::filesystem::resize_file(path, bytes - sizeof(float));
  EXPECT_THROW(source.ReadFloat3Sample(1, 2), std::exception);
}

TEST_F(WorkspaceTests, ResidentPositionalReadUsesSameGuardTransaction) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  auto options = GetOptions();
  options.num_threads = 1;
  options.full_ram_normal_storage = NormalStorage::kOct2x16;
  options.full_ram_normal_guard = true;
  options.full_ram_normal_guard_bytes = 1 << 20;
  options.full_ram_normal_guard_files = 1;
  Workspace workspace(options);
  workspace.Load({image_name_});
  const int image = workspace.GetModel().GetImageIdx(image_name_);
  for (size_t row : {size_t(0), size_t(4)}) {
    for (size_t col : {size_t(0), size_t(9)}) {
      const auto value = workspace.ReadOriginalStoredNormal(image, row, col);
      const float one = 1.0f;
      for (int channel = 0; channel < 3; ++channel)
        EXPECT_EQ(std::memcmp(&value[channel], &one, sizeof(float)), 0);
    }
  }
  const auto stats = workspace.GetOriginalNormalMappingStats();
  EXPECT_EQ(stats.active_copies, 0);
  EXPECT_EQ(stats.reserved_bytes, 0);
  EXPECT_EQ(stats.loads, 0);
  EXPECT_EQ(stats.resident_bytes, 0);
  EXPECT_EQ(stats.source_live_files, 1);
  EXPECT_THROW(workspace.ReadOriginalStoredNormal(image, 5, 0), std::exception);
  const auto path = workspace.GetNormalMapPath(image);
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
  EXPECT_THROW(workspace.ReadOriginalStoredNormal(image, 0, 0), std::exception);
  EXPECT_THROW(workspace.ValidateOriginalNormalSources(), std::exception);
  workspace.ReleaseOriginalNormalSources();
  EXPECT_THROW(workspace.ReadOriginalStoredNormal(image, 0, 0), std::exception);
}

TEST(Workspace, ResidentOriginalSourceAndRowLeaseLifetime) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  const auto path = CreateTestDir() / "normal.bin";
  NormalMap original(2, 2);
  original.Fill(0.5f);
  original.Write(path);
  auto pool = std::make_shared<MappedNormalPool>(1 << 20, 1, true);
  auto source = pool->AcquireSource(0, path);
  source->ValidateUnchanged();
  // Read uses the actual held FD even before any row mapping exists.
  NormalMap eager;
  eager.Read(source->GetPinnedReadPath());
  EXPECT_EQ(eager.GetData(), original.GetData());
  auto lease = pool->Acquire(0, path, 0, 1);
  pool.reset();
  source.reset();
  EXPECT_TRUE(((*lease).ReadFloat3(0, 1).array() ==
               Eigen::Vector3f::Constant(0.5f).array())
                  .all());
}

// Full-RAM normal-storage contracts.
TEST(Workspace, FullRamNormalStorageContracts) {
  const auto root = CreateTestDir() / "storage";
  for (const auto& dir : {root / "sparse",
                          root / "images",
                          root / "stereo" / "depth_maps",
                          root / "stereo" / "normal_maps"}) {
    std::filesystem::create_directories(dir);
  }
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 3;
  dataset.camera_width = 10;
  dataset.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  for (const auto& [image_id, image] : reconstruction.Images()) {
    Mat<float> depth(10, 5, 1);
    depth.Fill(2.0f);
    depth.Write(root / "stereo" / "depth_maps" /
                (image.Name() + ".geometric.bin"));
    Bitmap bitmap(10, 5, true);
    bitmap.Fill(BitmapColor<uint8_t>(
        12 + image_id, 34 + 2 * image_id, 56 + 3 * image_id));
    ASSERT_TRUE(bitmap.Write(root / "images" / image.Name()));
  }
  Workspace::Options fixture;
  fixture.workspace_path = root;
  fixture.workspace_format = "COLMAP";
  fixture.input_type = "geometric";
  Workspace descriptor(fixture);
  std::vector<std::string> names;
  const std::vector<Eigen::Vector3f> samples{{0, 0, 0},
                                             {1, 0, 0},
                                             {0, 0, -1},
                                             {-0.4f, 0.3f, -0.2f},
                                             {1, -1, 1},
                                             {1e-8f, 0, 0},
                                             {-0.00017f, 0.71f, 0.1f}};
  std::vector<Mat<float>> originals;
  for (size_t view = 0; view < descriptor.GetModel().images.size(); ++view) {
    names.push_back(descriptor.GetModel().GetImageName(view));
    Mat<float> map(10, 5, 3);
    for (size_t row = 0; row < 5; ++row)
      for (size_t col = 0; col < 10; ++col) {
        const auto& value = samples[(row * 10 + col + view) % samples.size()];
        for (size_t channel = 0; channel < 3; ++channel)
          map.Set(row, col, channel, value[channel]);
      }
    map.Write(descriptor.GetNormalMapPath(view));
    originals.push_back(map);
  }
  if (ImmutableMapRegion::Supported() &&
      normal_precision_guard::SupportedBuild()) {
    auto guarded = fixture;
    guarded.num_threads = 1;
    guarded.full_ram_normal_storage = NormalStorage::kOct2x16;
    guarded.full_ram_normal_guard = true;
    guarded.full_ram_normal_guard_bytes = 1 << 20;
    guarded.full_ram_normal_guard_files = 1;
    Workspace insufficient_files(guarded);
    EXPECT_THROW(insufficient_files.Load(names), std::exception);
    guarded.full_ram_normal_guard_files = 3;
    Workspace admitted(guarded);
    admitted.Load(names);
    EXPECT_EQ(admitted.GetOriginalNormalMappingStats().source_live_files, 3);
    EXPECT_EQ(admitted.GetOriginalNormalMappingStats().active_copies, 0);
  }
  EXPECT_EQ(fixture.full_ram_normal_storage, NormalStorage::kFp32);
  for (const auto* name : {"snorm16x3", "oct2x16-mag16"}) {
    NormalStorage codec = NormalStorage::kFp32;
    EXPECT_FALSE(ParseFullRamNormalStorage(name, &codec));
    EXPECT_EQ(codec, NormalStorage::kFp32);
  }
  for (const int value : {-1, 1, 3, 4}) {
    auto invalid = fixture;
    invalid.full_ram_normal_storage = static_cast<NormalStorage>(value);
    EXPECT_THROW(Workspace{invalid}, std::exception);
  }
  for (const auto* name : {"oct2x16"}) {
    SCOPED_TRACE(name);
    NormalStorage codec = NormalStorage::kFp32;
    ASSERT_TRUE(ParseFullRamNormalStorage(name, &codec));
    auto options = fixture;
    options.full_ram_normal_storage = codec;
    EXPECT_THROW(CachedWorkspace{options}, std::exception);
    EXPECT_THROW(BoundedWorkspace{options}, std::exception);
    for (const auto conflict :
         {&Workspace::Options::normal_tiles_snorm16,
          &Workspace::Options::exact_depth_state,
          &Workspace::Options::compact_rgb_to_depth_grid}) {
      auto invalid = options;
      invalid.*conflict = true;
      EXPECT_THROW(Workspace{invalid}, std::exception);
    }
    auto invalid = options;
    invalid.mapped_normal_bytes_limit = 1000;
    EXPECT_THROW(Workspace{invalid}, std::exception);
    invalid = options;
    invalid.mapped_normal_files_limit = 1;
    EXPECT_THROW(Workspace{invalid}, std::exception);
    invalid = options;
    invalid.full_ram_normal_storage = static_cast<NormalStorage>(-1);
    EXPECT_THROW(Workspace{invalid}, std::exception);
    for (const int max_size : {-1, 6}) {
      options.max_image_size = max_size;
      options.num_threads = 4;
      auto raw_options = options;
      raw_options.full_ram_normal_storage = NormalStorage::kFp32;
      Workspace raw(raw_options);
      raw.Load(names);
      CachedWorkspace cached(raw_options);
      Workspace compact(options);
      compact.Load(names);
      for (size_t view = 0; view < names.size(); ++view) {
        const auto& normal = raw.GetNormalMap(view);
        EXPECT_EQ(cached.GetNormalMap(view).GetData(), normal.GetData());
        EXPECT_EQ(compact.GetStoredNormalDimensions(view),
                  std::make_pair(normal.GetWidth(), normal.GetHeight()));
        NormalMapStorage expected;
        expected.normal_map = normal;
        expected.Pack(codec);
        constexpr size_t bytes = 4;
        EXPECT_TRUE(expected.normal_map.GetData().empty());
        EXPECT_EQ(compact.GetStoredNormalNumBytes(view),
                  normal.GetWidth() * normal.GetHeight() * bytes);
        for (size_t row = 0; row < normal.GetHeight(); ++row)
          for (size_t col = 0; col < normal.GetWidth(); ++col)
            EXPECT_TRUE((compact.GetStoredNormal(view, row, col).array() ==
                         expected.GetNormal(row, col).array())
                            .all());
        EXPECT_THROW(compact.GetNormalMap(view), std::exception);
        EXPECT_THROW(compact.GetNormalMapLease(view), std::exception);
      }
      const size_t before_bytes = compact.GetStoredNormalNumBytes(0);
      const auto before_sample = compact.GetStoredNormal(0, 0, 0);
      compact.Load(names);
      EXPECT_EQ(compact.GetStoredNormalNumBytes(0), before_bytes);
      EXPECT_TRUE(
          (compact.GetStoredNormal(0, 0, 0).array() == before_sample.array())
              .all());
    }
    options.max_image_size = -1;
    Workspace reload(options);
    reload.Load(names);
    const auto bad_path = descriptor.GetNormalMapPath(0);
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            1.01f,
                            -1.01f}) {
      Mat<float> invalid = originals[0];
      invalid.Set(0, 0, 0, bad);
      invalid.Write(bad_path);
      EXPECT_THROW(reload.Load(names), std::exception);
      EXPECT_THROW(reload.GetStoredNormal(0, 0, 0), std::exception);
      originals[0].Write(bad_path);
      reload.Load(names);
      EXPECT_TRUE(reload.GetStoredNormal(0, 0, 0).isZero());
    }
  }
}

TEST(Workspace, FullRamPackPrevalidatesBeforeAllocatingOrReleasing) {
  for (const auto codec : {NormalStorage::kOct2x16}) {
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            1.01f,
                            -1.01f}) {
      NormalMapStorage storage;
      storage.normal_map = NormalMap(2, 2);
      storage.normal_map.Fill(0.5f);
      storage.normal_map.Set(1, 1, 2, bad);
      EXPECT_THROW(storage.Pack(codec), std::exception);
      EXPECT_EQ(storage.normal_map.GetData().size(), 12);
      EXPECT_TRUE(storage.oct_normals.empty());
    }
  }
  for (const int value : {-1, 1, 3, 4}) {
    NormalMapStorage storage;
    storage.normal_map = NormalMap(2, 2);
    storage.normal_map.Fill(0.5f);
    const auto original = storage.normal_map.GetData();
    EXPECT_THROW(storage.Pack(static_cast<NormalStorage>(value)),
                 std::exception);
    EXPECT_EQ(storage.normal_map.GetData(), original);
    EXPECT_TRUE(storage.oct_normals.empty());
  }
  NormalMapStorage invalid;
  EXPECT_THROW(invalid.Pack(NormalStorage::kOct2x16), std::exception);
  static_cast<Mat<float>&>(invalid.normal_map) = Mat<float>(2, 2, 1);
  EXPECT_THROW(invalid.Pack(NormalStorage::kOct2x16), std::exception);
}

TEST_F(WorkspaceTests, ResidentGuard24WorkerAdmissionAndSourceRefusal) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  ASSERT_TRUE(normal_precision_guard::SupportedBuild());
  auto options = GetOptions();
  options.num_threads = 1;  // Loader count, independent of row concurrency.
  options.full_ram_normal_storage = NormalStorage::kOct2x16;
  options.full_ram_normal_guard = true;
  options.full_ram_normal_guard_concurrency = 24;
  options.full_ram_normal_guard_files = 24;
  const auto path =
      temp_dir_ / "stereo/normal_maps" / (image_name_ + ".geometric.bin");
  auto source = std::make_shared<ImmutableMapSource>(
      path, ImmutableMapSource::BindIdentity{});
  size_t maximum = 0;
  for (size_t row = 0; row < 5; ++row)
    maximum = std::max(maximum, source->GetWindowReservationBytes(row, 1));
  options.full_ram_normal_guard_bytes = 24 * maximum;
  const std::vector<std::string> names{image_name_};
  auto insufficient = options;
  --insufficient.full_ram_normal_guard_bytes;
  Workspace low_bytes(insufficient);
  EXPECT_THROW(low_bytes.Load(names), std::exception);
  EXPECT_THROW(low_bytes.GetStoredNormal(0, 0, 0), std::exception);
  insufficient = options;
  --insufficient.full_ram_normal_guard_files;
  Workspace low_files(insufficient);
  EXPECT_THROW(low_files.Load(names), std::exception);
  insufficient = options;
  insufficient.full_ram_normal_guard_concurrency =
      std::numeric_limits<size_t>::max() / 3;
  insufficient.full_ram_normal_guard_files =
      std::numeric_limits<size_t>::max() / 3;
  Workspace overflow(insufficient);
  EXPECT_THROW(overflow.Load(names), std::exception);
  Workspace admitted(options);
  admitted.Load(names);
  std::atomic<int> passed{0};
  std::vector<std::thread> threads;
  for (size_t worker = 0; worker < 24; ++worker)
    threads.emplace_back([&, worker] {
      try {
        for (size_t i = 0; i < 20; ++i)
          if (!admitted.ReadOriginalStoredNormal(0, worker % 5, worker % 10)
                   .isApprox(Eigen::Vector3f::Ones(), 0.0f))
            return;
        ++passed;
      } catch (...) {
      }
    });
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(passed.load(), 24);
  EXPECT_EQ(admitted.GetOriginalNormalMappingStats().active_copies, 0);
  // Preserve the held inode, replace its logical name, and refuse all readers.
  std::filesystem::rename(path, path.string() + ".original");
  NormalMap replacement(10, 5);
  replacement.Fill(0.5f);
  replacement.Write(path);
  passed = 0;
  threads.clear();
  for (size_t worker = 0; worker < 24; ++worker)
    threads.emplace_back([&] {
      try {
        admitted.ReadOriginalStoredNormal(0, 0, 0);
      } catch (const std::exception&) {
        ++passed;
      }
    });
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(passed.load(), 24);
  EXPECT_THROW(admitted.ValidateOriginalNormalSources(), std::exception);
  EXPECT_EQ(admitted.GetOriginalNormalMappingStats().active_copies, 0);
  admitted.ReleaseOriginalNormalSources();
  EXPECT_EQ(admitted.GetOriginalNormalMappingStats().source_live_files, 0);
}

TEST(Workspace, ResidentRowPool24ConcurrentOwnersAndLifetime) {
  ASSERT_TRUE(ImmutableMapRegion::Supported());
  const auto path = CreateTestDir() / "normal.bin";
  NormalMap original(10, 24);
  original.Fill(0.5f);
  original.Write(path);
  auto source = std::make_shared<ImmutableMapSource>(
      path, ImmutableMapSource::BindIdentity{});
  size_t maximum = 0;
  for (size_t row = 0; row < 24; ++row)
    maximum = std::max(maximum, source->GetWindowReservationBytes(row, 1));
  source.reset();
  for (const bool same_row : {false, true}) {
    auto pool = std::make_shared<MappedNormalPool>(24 * maximum, 24, true);
    auto retained = pool->AcquireSource(0, path);
    std::mutex mutex;
    std::condition_variable changed;
    size_t acquired = 0;
    bool release = false;
    std::atomic<size_t> passed{0};
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < 24; ++worker)
      threads.emplace_back([&, worker, owned_pool = pool]() mutable {
        try {
          const size_t row = same_row ? 0 : worker;
          auto lease = owned_pool->Acquire(0, path, row, 1);
          owned_pool
              .reset();  // Lease owns the pool for the remaining lifetime.
          {
            std::unique_lock<std::mutex> lock(mutex);
            ++acquired;
            changed.notify_all();
            changed.wait(lock, [&] { return release; });
          }
          if (((*lease).ReadFloat3(row, 9).array() == 0.5f).all()) ++passed;
        } catch (...) {
        }
      });
    bool complete;
    {
      std::unique_lock<std::mutex> lock(mutex);
      complete = changed.wait_for(
          lock, std::chrono::seconds(5), [&] { return acquired == 24; });
    }
    EXPECT_TRUE(complete);
    if (complete) {
      std::lock_guard<std::mutex> lock(pool->mutex);
      EXPECT_EQ(pool->stats.active_copies, 24);
      EXPECT_EQ(pool->stats.reserved_bytes, 0);
      EXPECT_EQ(pool->stats.resident_files, same_row ? 1 : 24);
      EXPECT_LE(pool->stats.peak_charged_bytes, 24 * maximum);
      EXPECT_LE(pool->stats.peak_charged_regions, 72);
    }
    retained.reset();
    pool.reset();  // All owners still held solely by the 24 leases.
    {
      std::lock_guard<std::mutex> lock(mutex);
      release = true;
    }
    changed.notify_all();
    for (auto& thread : threads) thread.join();
    EXPECT_EQ(passed.load(), 24);
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
