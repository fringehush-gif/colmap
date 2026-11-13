// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/delaunay_meshing.h"

#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/mvs/delaunay_meshing_worker.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(DelaunayImageWorkers, WorkerFailureCancelsAndJoins) {
  using namespace std::chrono_literals;
  std::promise<void> first_entered;
  auto first_entered_future = first_entered.get_future().share();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  std::promise<std::array<std::shared_future<void>, 2>> tasks;
  auto tasks_future = tasks.get_future();
  std::atomic<int> unexpected_images{0};
  std::atomic<bool> first_finished{false};

  auto run = std::async(std::launch::async, [&]() {
    std::array<std::shared_future<void>, 2> submitted;
    size_t num_submitted = 0;
    internal::RunDelaunayImageWorkers(
        16,
        2,
        [&](const size_t index) {
          if (index == 0) {
            first_entered.set_value();
            if (release_future.wait_for(5s) != std::future_status::ready) {
              throw std::runtime_error("release timeout");
            }
            first_finished = true;
          } else if (index == 1) {
            if (first_entered_future.wait_for(5s) !=
                std::future_status::ready) {
              throw std::runtime_error("first worker timeout");
            }
            throw std::runtime_error("image failure");
          } else {
            ++unexpected_images;
          }
        },
        [&](ThreadPool& pool, const auto& worker) {
          submitted[num_submitted++] = pool.AddTask(worker);
          if (num_submitted == submitted.size()) tasks.set_value(submitted);
        });
  });

  const auto submitted = tasks_future.wait_for(5s);
  std::future_status failed = std::future_status::timeout;
  if (submitted == std::future_status::ready) {
    const auto futures = tasks_future.get();
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      if (futures[0].wait_for(5ms) == std::future_status::ready ||
          futures[1].wait_for(5ms) == std::future_status::ready) {
        failed = std::future_status::ready;
        break;
      }
    }
  }
  const auto joined_early = run.wait_for(100ms);
  release.set_value();
  const auto finished = run.wait_for(5s);
  EXPECT_EQ(submitted, std::future_status::ready);
  EXPECT_EQ(failed, std::future_status::ready);
  EXPECT_EQ(joined_early, std::future_status::timeout);
  ASSERT_EQ(finished, std::future_status::ready);
  EXPECT_TRUE(first_finished);
  EXPECT_EQ(unexpected_images, 0);
  try {
    run.get();
    FAIL() << "Expected worker exception";
  } catch (const AggregateException& error) {
    EXPECT_NE(std::string(error.what()).find("image failure"),
              std::string::npos);
  } catch (...) {
    FAIL() << "Expected aggregate worker exception";
  }
}

class DelaunaySubmissionFailureTest : public ::testing::TestWithParam<bool> {};

TEST_P(DelaunaySubmissionFailureTest, DrainsAndPreservesException) {
  using namespace std::chrono_literals;
  const bool worker_throws = GetParam();
  std::promise<void> first_entered, submission_failed, drain_entered, release;
  auto entered_future = first_entered.get_future().share();
  auto failed_future = submission_failed.get_future();
  auto drain_future = drain_entered.get_future();
  auto release_future = release.get_future().share();
  std::atomic<bool> first_finished{false};
  std::atomic<int> unexpected_images{0};

  auto run = std::async(std::launch::async, [&]() {
    int submitted = 0;
    internal::RunDelaunayImageWorkers(
        16,
        2,
        [&](const size_t index) {
          if (index != 0) {
            ++unexpected_images;
            return;
          }
          first_entered.set_value();
          if (release_future.wait_for(5s) != std::future_status::ready) {
            throw std::runtime_error("release timeout");
          }
          first_finished = true;
          if (worker_throws) throw std::runtime_error("drained worker failure");
        },
        [&](ThreadPool& pool, const auto& worker) {
          if (++submitted == 1) {
            pool.AddTask(worker);
          } else {
            if (entered_future.wait_for(5s) != std::future_status::ready) {
              throw std::runtime_error("first worker timeout");
            }
            submission_failed.set_value();
            throw std::runtime_error("submission failure");
          }
        },
        [&](ThreadPool& pool) {
          drain_entered.set_value();
          pool.Wait();
        });
  });

  const auto failed = failed_future.wait_for(5s);
  const auto draining = drain_future.wait_for(5s);
  const auto joined_early = run.wait_for(100ms);
  release.set_value();
  const auto finished = run.wait_for(5s);
  EXPECT_EQ(failed, std::future_status::ready);
  EXPECT_EQ(draining, std::future_status::ready);
  EXPECT_EQ(joined_early, std::future_status::timeout);
  ASSERT_EQ(finished, std::future_status::ready);
  EXPECT_TRUE(first_finished);
  EXPECT_EQ(unexpected_images, 0);
  try {
    run.get();
    FAIL() << "Expected submission exception";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "submission failure");
  } catch (...) {
    FAIL() << "Expected original submission exception";
  }
}

INSTANTIATE_TEST_SUITE_P(DelaunayImageWorkers,
                         DelaunaySubmissionFailureTest,
                         ::testing::Bool());

Reconstruction CreateAndWriteSyntheticReconstruction(
    const std::filesystem::path& sparse_path,
    int num_frames = 5,
    int num_points3D = 100) {
  CreateDirIfNotExists(sparse_path);
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = num_frames;
  synthetic_dataset_options.num_points3D = num_points3D;
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  reconstruction.Write(sparse_path);
  return reconstruction;
}

#if defined(COLMAP_CGAL_ENABLED)

TEST(SparseDelaunayMeshing, ScaleInvariantWeights) {
  const auto test_dir = CreateTestDir();
  SetPRNGSeed(0);
  const Reconstruction original =
      CreateAndWriteSyntheticReconstruction(test_dir / "original");

  using PointKey = std::array<double, 3>;
  using FaceKey = std::array<PointKey, 3>;
  const std::array<double, 3> scales = {0.125, 1.0, 8.0};
  std::array<std::vector<PointKey>, 3> vertices;
  std::array<std::vector<FaceKey>, 3> faces;

  for (size_t scale_idx = 0; scale_idx < scales.size(); ++scale_idx) {
    const double scale = scales[scale_idx];
    SCOPED_TRACE(scale);
    Reconstruction scaled = original;
    scaled.Transform(
        Sim3d(scale, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()));

    if (scale_idx == 0) {
      std::vector<Eigen::Vector3f> points;
      for (const auto& item : scaled.Points3D()) {
        points.push_back(item.second.xyz.cast<float>());
      }
      ASSERT_GT(points.size(), 1);
      double min_pair_distance = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
          const double distance =
              (points[i].cast<double>() - points[j].cast<double>()).norm();
          min_pair_distance = std::min(min_pair_distance, distance);
        }
      }
      std::ostringstream formatted_distance;
      formatted_distance << std::setprecision(17) << min_pair_distance;
      RecordProperty("min_pair_distance_scale_0", formatted_distance.str());
      ASSERT_GT(min_pair_distance, 1e-7);
    }

    const auto sparse_path = test_dir / ("sparse_" + std::to_string(scale_idx));
    const auto output_path =
        test_dir / ("mesh_" + std::to_string(scale_idx) + ".ply");
    CreateDirIfNotExists(sparse_path);
    scaled.Write(sparse_path);
    DelaunayMeshingOptions options;
    options.num_threads = 1;
    SetPRNGSeed(0);
    SparseDelaunayMeshing(options, sparse_path, output_path);
    const PlyMesh mesh = ReadPlyMesh(output_path).mesh;
    ASSERT_FALSE(mesh.vertices.empty());
    ASSERT_FALSE(mesh.faces.empty());

    std::vector<PointKey> ordered_vertices;
    ordered_vertices.reserve(mesh.vertices.size());
    for (const auto& vertex : mesh.vertices) {
      const PointKey point = {
          vertex.x / scale, vertex.y / scale, vertex.z / scale};
      ASSERT_TRUE(std::isfinite(point[0]) && std::isfinite(point[1]) &&
                  std::isfinite(point[2]));
      ordered_vertices.push_back(point);
    }
    vertices[scale_idx] = ordered_vertices;
    std::sort(vertices[scale_idx].begin(), vertices[scale_idx].end());
    for (const auto& face : mesh.faces) {
      ASSERT_LT(face.vertex_idx1, ordered_vertices.size());
      ASSERT_LT(face.vertex_idx2, ordered_vertices.size());
      ASSERT_LT(face.vertex_idx3, ordered_vertices.size());
      ASSERT_NE(face.vertex_idx1, face.vertex_idx2);
      ASSERT_NE(face.vertex_idx2, face.vertex_idx3);
      ASSERT_NE(face.vertex_idx3, face.vertex_idx1);
      const PointKey& a = ordered_vertices[face.vertex_idx1];
      const PointKey& b = ordered_vertices[face.vertex_idx2];
      const PointKey& c = ordered_vertices[face.vertex_idx3];
      faces[scale_idx].push_back(
          std::min({FaceKey{a, b, c}, FaceKey{b, c, a}, FaceKey{c, a, b}}));
    }
    std::sort(faces[scale_idx].begin(), faces[scale_idx].end());
    RecordProperty("valid_scale_" + std::to_string(scale_idx), "true");
  }

  for (const size_t scale_idx : {size_t{0}, size_t{2}}) {
    EXPECT_TRUE(vertices[scale_idx] == vertices[1])
        << "scale normalized vertices mismatch for scale " << scales[scale_idx];
    EXPECT_TRUE(faces[scale_idx] == faces[1])
        << "scale normalized oriented faces mismatch for scale "
        << scales[scale_idx];
  }
}

class DelaunayMeshingTest : public ::testing::TestWithParam<int> {};

// Include more workers than images to cover the bounded worker count.
INSTANTIATE_TEST_SUITE_P(NumThreads,
                         DelaunayMeshingTest,
                         ::testing::Values(1, 2, 8));

TEST_P(DelaunayMeshingTest, SparseIntegration) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  CreateAndWriteSyntheticReconstruction(sparse_path);

  DelaunayMeshingOptions options;
  options.num_threads = GetParam();
  SparseDelaunayMeshing(options, sparse_path, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST_P(DelaunayMeshingTest, NonSubsampled) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  CreateAndWriteSyntheticReconstruction(sparse_path);

  // Setting max_proj_dist=0 exercises the non-subsampled
  // CreateDelaunayTriangulation() path instead of
  // CreateSubSampledDelaunayTriangulation().
  DelaunayMeshingOptions options;
  options.max_proj_dist = 0;
  options.num_threads = GetParam();
  SparseDelaunayMeshing(options, sparse_path, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST_P(DelaunayMeshingTest, PointAtCameraCenter) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  Reconstruction reconstruction =
      CreateAndWriteSyntheticReconstruction(sparse_path);

  // Move one point onto the projection center of an image that observes it.
  // Its viewing ray from that image then has zero length, which used to make
  // the ray caster walk the triangulation forever.
  const point3D_t point3D_id = *reconstruction.Point3DIds().begin();
  Point3D& point3D = reconstruction.Point3D(point3D_id);
  const image_t image_id = point3D.track.Element(0).image_id;
  point3D.xyz = reconstruction.Image(image_id).ProjectionCenter();
  reconstruction.Write(sparse_path);

  DelaunayMeshingOptions options;
  options.num_threads = GetParam();
  SparseDelaunayMeshing(options, sparse_path, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST_P(DelaunayMeshingTest, DenseIntegration) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  const auto reconstruction =
      CreateAndWriteSyntheticReconstruction(sparse_path, 3, 50);

  // Create fused.ply from reconstruction points
  std::vector<PlyPoint> ply_points;
  ply_points.reserve(reconstruction.NumPoints3D());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    PlyPoint ply_point;
    ply_point.x = static_cast<float>(point3D.xyz(0));
    ply_point.y = static_cast<float>(point3D.xyz(1));
    ply_point.z = static_cast<float>(point3D.xyz(2));
    ply_point.nx = 0.0f;
    ply_point.ny = 0.0f;
    ply_point.nz = 1.0f;
    ply_point.r = 128;
    ply_point.g = 128;
    ply_point.b = 128;
    ply_points.push_back(ply_point);
  }
  WriteBinaryPlyPoints(test_dir / "fused.ply",
                       ply_points,
                       /*write_normal=*/true,
                       /*write_rgb=*/true);

  // Create fused.ply.vis: for each point, list visible image indices.
  // Each point is visible in all images to give sufficient multi-view
  // information for the graph-cut optimization.
  const auto vis_path = test_dir / "fused.ply.vis";
  std::fstream vis_file(vis_path, std::ios::out | std::ios::binary);
  THROW_CHECK_FILE_OPEN(vis_file, vis_path);
  const uint64_t num_points = ply_points.size();
  const uint32_t num_visible =
      static_cast<uint32_t>(reconstruction.NumRegImages());
  WriteBinaryLittleEndian<uint64_t>(&vis_file, num_points);
  for (size_t i = 0; i < num_points; ++i) {
    WriteBinaryLittleEndian<uint32_t>(&vis_file, num_visible);
    for (uint32_t j = 0; j < num_visible; ++j) {
      WriteBinaryLittleEndian<uint32_t>(&vis_file, j);
    }
  }
  vis_file.close();

  DelaunayMeshingOptions options;
  options.num_threads = GetParam();
  DenseDelaunayMeshing(options, test_dir, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

#endif  // COLMAP_CGAL_ENABLED

}  // namespace
}  // namespace mvs
}  // namespace colmap
