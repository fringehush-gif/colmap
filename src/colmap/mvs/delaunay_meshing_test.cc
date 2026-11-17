// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/delaunay_meshing.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(DelaunayMeshingProbability, StandardDeviationScaling) {
  // The old squared-input comparison against 5 * sigma saturates too early
  // for sigma > 0.2 and too late below it. Test both visibility and scene
  // scales.
  for (const double sigma : {0.125, 0.5, 1.0, 3.0, 8.0}) {
    const double normalization = -0.5 / (sigma * sigma);
    for (const double factor : {0.0, 0.25, 1.0, 2.0, 4.0, 4.999, 5.0, 8.0}) {
      SCOPED_TRACE(testing::Message()
                   << "sigma=" << sigma << " factor=" << factor);
      const double value = factor * sigma;
      const double probability = internal::ComputeDelaunayMeshingProbability(
          value * value, normalization);
      const long double normalized = factor;
      const double expected =
          factor >= 5.0 ? 1.0
                        : static_cast<double>(
                              -std::expm1(-0.5L * normalized * normalized));
      EXPECT_NEAR(probability, expected, 1e-15);
    }
  }
  // Four observations at the default sigma must not already saturate.
  EXPECT_NEAR(internal::ComputeDelaunayMeshingProbability(16.0, -0.5 / 9.0),
              0.5888877094928125,
              1e-15);
}

TEST(DelaunayMeshingProbability, SaturationBoundaryAndZero) {
  EXPECT_LT(internal::ComputeDelaunayMeshingProbability(
                std::nextafter(25.0, 0.0), -0.5),
            1.0);
  EXPECT_EQ(internal::ComputeDelaunayMeshingProbability(25.0, -0.5), 1.0);
  EXPECT_EQ(internal::ComputeDelaunayMeshingProbability(
                std::nextafter(25.0, 26.0), -0.5),
            1.0);
  // Positive subnormal sigma can produce negative infinite normalization.
  EXPECT_EQ(internal::ComputeDelaunayMeshingProbability(
                0.0, -std::numeric_limits<double>::infinity()),
            0.0);
  EXPECT_EQ(internal::ComputeDelaunayMeshingProbability(
                1.0, -std::numeric_limits<double>::infinity()),
            1.0);
}

#if defined(COLMAP_CGAL_ENABLED)

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

TEST(SparseDelaunayMeshing, VisibilityProbabilityBalancesEvidence) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  CreateDirIfNotExists(sparse_path);

  Reconstruction reconstruction;
  const Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera);
  Track track;
  for (image_t image_id = 1; image_id <= 4; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.camera_id);
    image.SetName(std::to_string(image_id) + ".png");
    image.SetPoints2D(std::vector<Eigen::Vector2d>{{0.5, 0.5}});
    const Eigen::Quaterniond rotation =
        image_id < 4 ? Eigen::Quaterniond(0, 1, 0, 0)
                     : Eigen::Quaterniond(0.5, 0.5, 0.5, -0.5);
    reconstruction.AddImageWithTrivialFrame(
        image, Rigid3d(rotation, Eigen::Vector3d(-0.25, 0.25, 2)));
    track.AddElement(image_id, 0);
  }

  // Only the interior point P is observed. The four corners retain a finite
  // tetrahedron behind every observation, without adding observation weights.
  const std::array<Eigen::Vector3d, 5> points = {
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(1, 0, 0),
      Eigen::Vector3d(0, 1, 0),
      Eigen::Vector3d(0, 0, 1),
      Eigen::Vector3d(0.25, 0.25, 0.25)};
  for (size_t i = 0; i < points.size(); ++i) {
    reconstruction.AddPoint3D(points[i], i == 4 ? track : Track());
  }
  reconstruction.Write(sparse_path);

  // Four-view confidence is 1-exp(-8/9), not 1. These three upper views and
  // one side view balance the shape prior so the side view alone cannot add
  // a second inside cell. Both distance probabilities saturate to 1.
  DelaunayMeshingOptions options;
  options.max_proj_dist = 0;
  options.num_threads = 1;
  options.visibility_sigma = 3;
  options.distance_sigma_factor = 0.01;
  options.quality_regularization = 1.75;
  options.max_side_length_percentile = 100;
  options.max_side_length_factor = 2;
  SparseDelaunayMeshing(options, sparse_path, output_path);

  const PlyMesh mesh = ReadPlyMesh(output_path).mesh;
  ASSERT_EQ(mesh.vertices.size(), 4);
  ASSERT_EQ(mesh.faces.size(), 3);

  // Compare every oriented triangle by input coordinates. Cyclic rotation and
  // serialization order may vary, but winding and multiplicity must not.
  std::vector<std::array<size_t, 3>> faces;
  for (const auto& face : mesh.faces) {
    std::array<size_t, 3> indices = {
        face.vertex_idx1, face.vertex_idx2, face.vertex_idx3};
    for (size_t& index : indices) {
      ASSERT_LT(index, mesh.vertices.size());
      const auto& vertex = mesh.vertices[index];
      const auto it = std::find(points.begin(),
                                points.end(),
                                Eigen::Vector3d(vertex.x, vertex.y, vertex.z));
      ASSERT_NE(it, points.end());
      index = static_cast<size_t>(std::distance(points.begin(), it));
    }
    std::rotate(indices.begin(),
                std::min_element(indices.begin(), indices.end()),
                indices.end());
    faces.push_back(indices);
  }
  std::sort(faces.begin(), faces.end());
  const std::vector<std::array<size_t, 3>> expected = {
      {0, 1, 4}, {0, 4, 2}, {1, 2, 4}};
  EXPECT_EQ(faces, expected);
}

TEST(SparseDelaunayMeshing, Integration) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  CreateAndWriteSyntheticReconstruction(sparse_path);

  DelaunayMeshingOptions options;
  options.num_threads = 1;
  SparseDelaunayMeshing(options, sparse_path, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST(SparseDelaunayMeshing, NonSubsampled) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  CreateAndWriteSyntheticReconstruction(sparse_path);

  // Setting max_proj_dist=0 exercises the non-subsampled
  // CreateDelaunayTriangulation() path instead of
  // CreateSubSampledDelaunayTriangulation().
  DelaunayMeshingOptions options;
  options.max_proj_dist = 0;
  options.num_threads = 1;
  SparseDelaunayMeshing(options, sparse_path, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST(SparseDelaunayMeshing, PointAtCameraCenter) {
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
  options.num_threads = 1;
  SparseDelaunayMeshing(options, sparse_path, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST(DenseDelaunayMeshing, Integration) {
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
  options.num_threads = 1;
  DenseDelaunayMeshing(options, test_dir, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

#endif  // COLMAP_CGAL_ENABLED

}  // namespace
}  // namespace mvs
}  // namespace colmap
