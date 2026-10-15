// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/fusion.h"

#include "colmap/math/math.h"
#include "colmap/mvs/model.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/bitmap_reader.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(StereoFusion, IndependentPixelsPreserveCompleteWorkerOutput) {
  const auto root = CreateTestDir();
  for (const auto& sub : {"sparse",
                          "images",
                          "stereo",
                          "stereo/depth_maps",
                          "stereo/normal_maps"}) {
    CreateDirIfNotExists(root / sub);
  }
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 1;
  dataset.camera_width = 30;
  dataset.camera_height = 40;
  dataset.camera_params = {30.0, 15.0, 20.0, 0.0};
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(root / "sparse");
  ASSERT_EQ(reconstruction.Images().size(), 1);
  const auto name = reconstruction.Images().begin()->second.Name();
  Mat<float> depth(30, 40, 1);
  depth.Fill(5.0f);
  depth.Write(root / "stereo/depth_maps" / (name + ".geometric.bin"));
  Mat<float> normal(30, 40, 3);
  normal.Fill(0.0f);
  for (int row = 0; row < 40; ++row) {
    for (int col = 0; col < 30; ++col) normal.Set(row, col, 2, 1.0f);
  }
  normal.Write(root / "stereo/normal_maps" / (name + ".geometric.bin"));
  Bitmap image(30, 40, true);
  image.Fill(BitmapColor<uint8_t>(0, 64, 128));
  ASSERT_TRUE(image.Write(root / "images" / name));
  {
    std::ofstream config(root / "stereo/fusion.cfg");
    config << name << '\n';
  }
  Model model;
  model.Read(root, "COLMAP");
  const auto overlapping = model.GetMaxOverlappingImages(10, 0.0);
  ASSERT_EQ(model.images.size(), 1);
  ASSERT_EQ(overlapping.size(), 1);
  ASSERT_TRUE(overlapping.front().empty());

  // Include every serialized attribute and its visibility association. Bits
  // distinguish signed zero as well; task completion order is irrelevant.
  using Record = std::tuple<uint32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t,
                            uint32_t,
                            uint8_t,
                            uint8_t,
                            uint8_t,
                            std::vector<int>>;
  const auto bits = [](float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  };
  const auto capture = [&](const StereoFusion& fusion) {
    std::vector<Record> records;
    const auto& points = fusion.GetFusedPoints();
    const auto& visibility = fusion.GetFusedPointsVisibility();
    records.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      const auto& p = points[i];
      EXPECT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) &&
                  std::isfinite(p.z) && std::isfinite(p.nx) &&
                  std::isfinite(p.ny) && std::isfinite(p.nz));
      auto visible = visibility.at(i);
      EXPECT_EQ(visible, std::vector<int>{0});
      records.emplace_back(bits(p.x),
                           bits(p.y),
                           bits(p.z),
                           bits(p.nx),
                           bits(p.ny),
                           bits(p.nz),
                           p.r,
                           p.g,
                           p.b,
                           std::move(visible));
    }
    std::sort(records.begin(), records.end());
    return records;
  };
  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.num_threads = 1;
  StereoFusion serial(options, root, "COLMAP", "", "geometric");
  serial.Run();
  ASSERT_EQ(serial.GetFusedPoints().size(), 30 * 40);
  ASSERT_EQ(serial.GetFusedPointsVisibility().size(), 30 * 40);
  const auto expected = capture(serial);
  options.num_threads = 2;
  StereoFusion parallel(options, root, "COLMAP", "", "geometric");
  for (int run = 0; run < 2; ++run) {
    parallel.Run();
    ASSERT_EQ(parallel.GetFusedPoints().size(), expected.size());
    ASSERT_EQ(parallel.GetFusedPointsVisibility().size(), expected.size());
    EXPECT_EQ(capture(parallel), expected);
  }

  // All row bands execute, but every pixel is skipped before Fuse. The
  // unconditional task-entry worker lookup must remain valid in this case.
  const auto masks = root / "masks";
  CreateDirIfNotExists(masks);
  Bitmap mask(30, 40, true);
  mask.Fill(BitmapColor<uint8_t>(0, 0, 0));
  ASSERT_TRUE(mask.Write(masks / (name + ".png")));
  options.mask_path = masks;
  StereoFusion masked(options, root, "COLMAP", "", "geometric");
  masked.Run();
  EXPECT_TRUE(masked.GetFusedPoints().empty());
  EXPECT_TRUE(masked.GetFusedPointsVisibility().empty());
}

TEST(StereoFusion, Integration) {
  const auto temp_dir = CreateTestDir();
  CreateDirIfNotExists(temp_dir / "sparse");
  CreateDirIfNotExists(temp_dir / "images");
  CreateDirIfNotExists(temp_dir / "stereo");
  CreateDirIfNotExists(temp_dir / "stereo" / "depth_maps");
  CreateDirIfNotExists(temp_dir / "stereo" / "normal_maps");

  // Create synthetic reconstruction with 2 overlapping images.
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.camera_width = 30;
  synthetic_dataset_options.camera_height = 20;
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  reconstruction.Write(temp_dir / "sparse");

  // Create depth maps, normal maps, and consistency graphs for both images.
  std::vector<std::string> image_names;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    image_names.push_back(image.Name());

    // Create depth map with constant depth.
    Mat<float> depth_map(synthetic_dataset_options.camera_width,
                         synthetic_dataset_options.camera_height,
                         1);
    depth_map.Fill(5.0f);
    depth_map.Write(temp_dir / "stereo" / "depth_maps" /
                    (image.Name() + ".geometric.bin"));

    // Create normal map pointing in z direction.
    Mat<float> normal_map(synthetic_dataset_options.camera_width,
                          synthetic_dataset_options.camera_height,
                          3);
    const size_t num_pixels = normal_map.GetHeight() * normal_map.GetWidth();
    for (size_t i = 0; i < num_pixels; ++i) {
      normal_map.GetPtr()[3 * i + 0] = 0.0f;  // nx
      normal_map.GetPtr()[3 * i + 1] = 0.0f;  // ny
      normal_map.GetPtr()[3 * i + 2] = 1.0f;  // nz
    }
    normal_map.Write(temp_dir / "stereo" / "normal_maps" /
                     (image.Name() + ".geometric.bin"));

    // Create bitmap.
    Bitmap bitmap(synthetic_dataset_options.camera_width,
                  synthetic_dataset_options.camera_height,
                  true);
    bitmap.Fill(BitmapColor<uint8_t>(0, 64, 128));
    bitmap.Write(temp_dir / "images" / image.Name());
  }

  // Write fusion config
  std::ofstream fusion_cfg(temp_dir / "stereo" / "fusion.cfg");
  for (const auto& name : image_names) {
    fusion_cfg << name << "\n";
  }
  fusion_cfg.close();

  // Run fusion
  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.max_traversal_depth = 10;
  options.check_num_images = 10;
  options.use_cache = false;
  options.num_threads = 1;
  EXPECT_FALSE(options.bounded_cache);

  StereoFusion fusion(options, temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  // Verify that some points were fused
  const auto& fused_points = fusion.GetFusedPoints();
  const auto& visibility = fusion.GetFusedPointsVisibility();

  EXPECT_GT(fused_points.size(), 0);
  EXPECT_EQ(fused_points.size(), visibility.size());

  for (const auto& point : fused_points) {
    EXPECT_GT(point.x, -10.0f);
    EXPECT_LT(point.x, 10.0f);
    EXPECT_GT(point.y, -10.0f);
    EXPECT_LT(point.y, 10.0f);
    EXPECT_GT(point.z, -10.0f);
    EXPECT_LT(point.z, 10.0f);
    EXPECT_EQ(point.r, 0);
    EXPECT_EQ(point.g, 64);
    EXPECT_EQ(point.b, 128);
    EXPECT_FLOAT_EQ(
        point.nx * point.nx + point.ny * point.ny + point.nz * point.nz, 1.0f);
  }

  for (const auto& vis : visibility) {
    EXPECT_GT(vis.size(), 0);
  }

  // The opt-in whole-image cache must preserve the eager path's XYZ,
  // normals, RGB, visibility, and point order on this deterministic fixture.
  options.bounded_cache = true;
  options.cache_size = 12000.0 / (1024.0 * 1024.0 * 1024.0);
  StereoFusion bounded(options, temp_dir, "COLMAP", "", "geometric");
  bounded.Run();
  const auto& bounded_points = bounded.GetFusedPoints();
  const auto& bounded_visibility = bounded.GetFusedPointsVisibility();
  ASSERT_EQ(bounded_points.size(), fused_points.size());
  EXPECT_EQ(bounded_visibility, visibility);
  for (size_t i = 0; i < fused_points.size(); ++i) {
    const auto& a = fused_points[i];
    const auto& b = bounded_points[i];
    EXPECT_FLOAT_EQ(a.x, b.x);
    EXPECT_FLOAT_EQ(a.y, b.y);
    EXPECT_FLOAT_EQ(a.z, b.z);
    EXPECT_FLOAT_EQ(a.nx, b.nx);
    EXPECT_FLOAT_EQ(a.ny, b.ny);
    EXPECT_FLOAT_EQ(a.nz, b.nz);
    EXPECT_EQ(a.r, b.r);
    EXPECT_EQ(a.g, b.g);
    EXPECT_EQ(a.b, b.b);
  }

  bool stop_checked = false;
  StereoFusion cancelled_fusion(options, temp_dir, "COLMAP", "", "geometric");
  cancelled_fusion.SetCheckIfStoppedFunc([&stop_checked]() {
    stop_checked = true;
    return true;
  });
  cancelled_fusion.Run();
  EXPECT_TRUE(stop_checked);
  EXPECT_TRUE(cancelled_fusion.GetFusedPoints().empty());
  // Aggregation consumes worker visibility buffers. A second run must
  // reconstruct the complete ordered output after those buffers were moved.
  const auto first_points = fused_points;
  const auto first_visibility = visibility;
  fusion.Run();
  ASSERT_EQ(fusion.GetFusedPoints().size(), first_points.size());
  EXPECT_EQ(fusion.GetFusedPointsVisibility(), first_visibility);
  for (size_t i = 0; i < first_points.size(); ++i) {
    const auto& a = first_points[i];
    const auto& b = fusion.GetFusedPoints()[i];
    EXPECT_FLOAT_EQ(a.x, b.x);
    EXPECT_FLOAT_EQ(a.y, b.y);
    EXPECT_FLOAT_EQ(a.z, b.z);
    EXPECT_FLOAT_EQ(a.nx, b.nx);
    EXPECT_FLOAT_EQ(a.ny, b.ny);
    EXPECT_FLOAT_EQ(a.nz, b.nz);
    EXPECT_EQ(a.r, b.r);
    EXPECT_EQ(a.g, b.g);
    EXPECT_EQ(a.b, b.b);
  }
}

TEST(StereoFusion, CompactResidentRgbMatchesDiskFusion) {
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
  Workspace::Options workspace_options;
  workspace_options.workspace_path = root;
  workspace_options.workspace_format = "COLMAP";
  workspace_options.input_type = "geometric";
  Workspace descriptor(workspace_options);
  std::vector<std::string> names;
  std::vector<std::filesystem::path> paths;
  auto originals = std::make_shared<std::vector<Bitmap>>();
  for (int image_id = 1; image_id <= 2; ++image_id) {
    const auto name = reconstruction.Image(image_id).Name();
    names.push_back(name);
    const int image = descriptor.GetModel().GetImageIdx(name);
    paths.push_back(descriptor.GetBitmapPath(image));
    Mat<float> depth(7, 3, 1);
    depth.Fill(5.0f);
    depth.Write(descriptor.GetDepthMapPath(image));
    Mat<float> normal(7, 3, 3);
    normal.Fill(0.0f);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 7; ++col) {
        normal.Set(row, col, 2, 1.0f);
      }
    }
    normal.Write(descriptor.GetNormalMapPath(image));
    Bitmap bitmap(10, 5, true);
    for (int row = 0; row < 5; ++row) {
      for (int col = 0; col < 10; ++col) {
        bitmap.SetPixel(
            col,
            row,
            BitmapColor<uint8_t>(
                col * 19 + row, row * 31 + col, image_id * 20 + col));
      }
    }
    ASSERT_TRUE(bitmap.Write(paths.back()));
    originals->push_back(bitmap);
  }
  {
    std::ofstream config(root / "stereo" / "fusion.cfg");
    for (const auto& name : names) config << name << '\n';
  }
  StereoFusionOptions options;
  options.num_threads = 1;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.max_traversal_depth = 10;
  options.check_num_images = 10;
  StereoFusion disk(options, root, "COLMAP", "", "geometric");
  disk.Run();
  const auto& expected = disk.GetFusedPoints();
  ASSERT_FALSE(expected.empty());
  auto scope = ScopedBitmapReader::Create(
      paths,
      [strong = originals, paths](
          const auto& path, bool rgb, bool linear, Bitmap* target) {
        EXPECT_TRUE(rgb);
        EXPECT_FALSE(linear);
        for (size_t i = 0; i < paths.size(); ++i) {
          if (path == std::filesystem::absolute(paths[i]).lexically_normal()) {
            *target = strong->at(i);
            return true;
          }
        }
        return false;
      },
      {{10, 5}, {10, 5}});
  originals.reset();
  for (const auto& path : paths) ASSERT_TRUE(std::filesystem::remove(path));
  options.bounded_cache = true;
  options.bounded_cache_compact_rgb = true;
  options.cache_size = 600.0 / (1024.0 * 1024.0 * 1024.0);
  StereoFusion compact(options, root, "COLMAP", "", "geometric");
  compact.Run();  // The strong scope owner stays alive through worker join.
  const auto& actual = compact.GetFusedPoints();
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_EQ(compact.GetFusedPointsVisibility(),
            disk.GetFusedPointsVisibility());
  for (size_t i = 0; i < actual.size(); ++i) {
    const auto& a = actual[i];
    const auto& b = expected[i];
    EXPECT_FLOAT_EQ(a.x, b.x);
    EXPECT_FLOAT_EQ(a.y, b.y);
    EXPECT_FLOAT_EQ(a.z, b.z);
    EXPECT_FLOAT_EQ(a.nx, b.nx);
    EXPECT_FLOAT_EQ(a.ny, b.ny);
    EXPECT_FLOAT_EQ(a.nz, b.nz);
    EXPECT_EQ(a.r, b.r);
    EXPECT_EQ(a.g, b.g);
    EXPECT_EQ(a.b, b.b);
  }
  scope->Close();
  for (const auto& name : names) {
    EXPECT_FALSE(descriptor.HasBitmap(descriptor.GetModel().GetImageIdx(name)));
  }
}

TEST(ReadPointsVisibility, RoundTrip) {
  const auto test_dir = CreateTestDir();
  const auto vis_path = test_dir / "test.vis";

  std::vector<std::vector<int>> expected = {
      {0, 1, 2},
      {1, 3},
      {},
      {0, 2, 3, 4},
  };
  WritePointsVisibility(vis_path, expected);

  const auto actual = ReadPointsVisibility(vis_path, expected.size());

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(actual[i].size(), expected[i].size());
    for (size_t j = 0; j < expected[i].size(); ++j) {
      EXPECT_EQ(actual[i][j], expected[i][j]);
    }
  }
}

TEST(ReadPointsVisibility, SizeMismatch) {
  const auto test_dir = CreateTestDir();
  const auto vis_path = test_dir / "test.vis";

  std::vector<std::vector<int>> data = {{0, 1}, {2}};
  WritePointsVisibility(vis_path, data);

  EXPECT_THROW(ReadPointsVisibility(vis_path, 5), std::invalid_argument);
}

TEST(StereoFusion, Rotation180Invariance) {
  const size_t width = 40;
  const size_t height = 30;
  const double focal_length = 100.0;
  const double cx = 18.0;
  const double cy = 12.0;

  const int test_col = 12;
  const int test_row = 8;
  const float test_depth = 5.0f;

  auto CreateWorkspace = [&](const std::filesystem::path& dir,
                             const double cam_cx,
                             const double cam_cy,
                             const Rigid3d& cam_from_world,
                             const int col,
                             const int row) {
    CreateDirIfNotExists(dir / "sparse");
    CreateDirIfNotExists(dir / "images");
    CreateDirIfNotExists(dir / "stereo");
    CreateDirIfNotExists(dir / "stereo" / "depth_maps");
    CreateDirIfNotExists(dir / "stereo" / "normal_maps");

    Reconstruction reconstruction;
    colmap::Camera camera;
    camera.camera_id = 1;
    camera.model_id = CameraModelId::kPinhole;
    camera.width = width;
    camera.height = height;
    camera.params = {focal_length, focal_length, cam_cx, cam_cy};
    reconstruction.AddCameraWithTrivialRig(camera);

    colmap::Image image;
    image.SetImageId(1);
    image.SetName("image.png");
    image.SetCameraId(1);
    reconstruction.AddImageWithTrivialFrame(image, cam_from_world);
    reconstruction.Write(dir / "sparse");

    Mat<float> depth_map(width, height, 1);
    depth_map.Fill(0.0f);
    depth_map.Set(row, col, 0, test_depth);
    depth_map.Write(dir / "stereo" / "depth_maps" / "image.png.geometric.bin");

    Mat<float> normal_map(width, height, 3);
    normal_map.Fill(0.0f);
    normal_map.Set(row, col, 2, 1.0f);
    normal_map.Write(dir / "stereo" / "normal_maps" /
                     "image.png.geometric.bin");

    Bitmap bitmap(width, height, true);
    bitmap.Fill(BitmapColor<uint8_t>(100, 100, 100));
    bitmap.Write(dir / "images" / "image.png");

    std::ofstream fusion_cfg(dir / "stereo" / "fusion.cfg");
    fusion_cfg << "image.png\n";
    fusion_cfg.close();
  };

  const auto temp_dir1 = CreateTestDir();
  CreateWorkspace(temp_dir1, cx, cy, Rigid3d(), test_col, test_row);

  const auto temp_dir2 = CreateTestDir();
  const Eigen::Quaterniond q_z180(
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
  CreateWorkspace(temp_dir2,
                  width - cx,
                  height - cy,
                  Rigid3d(q_z180, Eigen::Vector3d::Zero()),
                  width - 1 - test_col,
                  height - 1 - test_row);

  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.use_cache = false;

  StereoFusion fusion1(options, temp_dir1, "COLMAP", "", "geometric");
  fusion1.Run();

  StereoFusion fusion2(options, temp_dir2, "COLMAP", "", "geometric");
  fusion2.Run();

  const auto& points1 = fusion1.GetFusedPoints();
  const auto& points2 = fusion2.GetFusedPoints();

  ASSERT_EQ(points1.size(), 1);
  ASSERT_EQ(points2.size(), 1);

  // Expected 3D point using pixel center at (col + 0.5, row + 0.5).
  const float expected_x = test_depth * (test_col + 0.5f - cx) / focal_length;
  const float expected_y = test_depth * (test_row + 0.5f - cy) / focal_length;
  const float expected_z = test_depth;

  EXPECT_NEAR(points1[0].x, expected_x, 1e-5f);
  EXPECT_NEAR(points1[0].y, expected_y, 1e-5f);
  EXPECT_NEAR(points1[0].z, expected_z, 1e-5f);

  EXPECT_NEAR(points2[0].x, expected_x, 1e-5f);
  EXPECT_NEAR(points2[0].y, expected_y, 1e-5f);
  EXPECT_NEAR(points2[0].z, expected_z, 1e-5f);

  EXPECT_NEAR(points1[0].x, points2[0].x, 1e-5f);
  EXPECT_NEAR(points1[0].y, points2[0].y, 1e-5f);
  EXPECT_NEAR(points1[0].z, points2[0].z, 1e-5f);
}

#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

TEST(StereoFusionOptions, CachePoliciesAreMutuallyExclusive) {
  StereoFusionOptions options;
  EXPECT_TRUE(options.Check());
  options.use_cache = true;
  EXPECT_TRUE(options.Check());
  options.bounded_cache = true;
  EXPECT_FALSE(options.Check());
  options.use_cache = false;
  EXPECT_TRUE(options.Check());
}

// Compare compact resident maps with their decoded FP32 equivalents.

}  // namespace
}  // namespace mvs
}  // namespace colmap
