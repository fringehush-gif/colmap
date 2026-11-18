// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/poisson_meshing.h"

#include "colmap/math/random_eigen.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include "thirdparty/PoissonRecon/SurfaceTrimmer.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

void WriteRandomPlyPoints(const std::filesystem::path& path,
                          int num_points = 100) {
  std::vector<PlyPoint> ply_points;
  ply_points.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    const Eigen::Vector3d point3D = RandomEigenVectord<3>();
    PlyPoint ply_point;
    ply_point.x = point3D.x();
    ply_point.y = point3D.y();
    ply_point.z = point3D.z();
    ply_point.nx = 0.0f;
    ply_point.ny = 0.0f;
    ply_point.nz = 1.0f;
    ply_point.r = 0;
    ply_point.g = 64;
    ply_point.b = 128;
    ply_points.push_back(ply_point);
  }
  WriteBinaryPlyPoints(
      path, ply_points, /*write_normal=*/true, /*write_rgb=*/true);
}

void WriteCubePlyPoints(const std::filesystem::path& path) {
  std::vector<PlyPoint> points;
  points.reserve(6 * 5 * 5);
  for (int axis = 0; axis < 3; ++axis) {
    for (const float sign : {-1.0f, 1.0f}) {
      const Eigen::Vector3f normal = sign * Eigen::Vector3f::Unit(axis);
      for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
          Eigen::Vector3f point = Eigen::Vector3f::Zero();
          point[axis] = sign;
          point[(axis + 1) % 3] = i / 2.0f;
          point[(axis + 2) % 3] = j / 2.0f;
          points.push_back({point.x(),
                            point.y(),
                            point.z(),
                            normal.x(),
                            normal.y(),
                            normal.z(),
                            32,
                            64,
                            128});
        }
      }
    }
  }
  WriteBinaryPlyPoints(path, points, /*write_normal=*/true, /*write_rgb=*/true);
}

int RunSurfaceTrimmerWithArgs(std::vector<std::string> args) {
  char program[] = "SurfaceTrimmer";
  std::vector<char*> argv;
  argv.reserve(args.size() + 2);
  argv.push_back(program);
  for (auto& arg : args) argv.push_back(arg.data());
  argv.push_back(nullptr);
  return RunSurfaceTrimmer(static_cast<int>(args.size() + 1), argv.data());
}

TEST(PoissonMeshing, Integration) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "points.ply";
  const auto output_path = test_dir / "mesh.ply";
  WriteRandomPlyPoints(input_path);

  PoissonMeshingOptions options;
  options.point_weight = 1.0;
  options.depth = 3;   // Use smaller depth for faster test
  options.trim = 0.0;  // Disable trimming
  options.num_threads = 1;

  EXPECT_TRUE(PoissonMeshing(options, input_path, output_path));

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST(PoissonMeshing, WithTrimming) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "points.ply";
  const auto output_path = test_dir / "mesh.ply";
  WriteRandomPlyPoints(input_path);

  PoissonMeshingOptions options;
  options.point_weight = 1.0;
  options.depth = 3;
  options.trim = 5.0;
  options.num_threads = 1;

  EXPECT_TRUE(PoissonMeshing(options, input_path, output_path));
  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  // With random data and trimming, we can't make strong assumptions about
  // the number of vertices, but reading the file ensures valid PLY format.
  EXPECT_GE(mesh_vertices.size(), 0);
}

TEST(PoissonMeshing, SequentialCallsResetOmittedOptions) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "points.ply";
  const auto output_path = test_dir / "subsequent.ply";
  WriteCubePlyPoints(input_path);

  PoissonMeshingOptions options;
  options.point_weight = 1.0;
  options.depth = 3;
  options.trim = 0.1;
  options.color = true;
  options.num_threads = 1;
  ASSERT_TRUE(PoissonMeshing(options, input_path, test_dir / "trimmed.ply"));

  options.depth = 6;
  options.trim = 0.0;
  ASSERT_TRUE(PoissonMeshing(options, input_path, output_path));
  const auto mesh = ReadPlyMesh(output_path).mesh;
  ASSERT_GE(mesh.vertices.size(), 3);
  ASSERT_FALSE(mesh.faces.empty());

  std::ifstream input(output_path, std::ios::binary);
  ASSERT_TRUE(input.is_open());
  std::string line;
  bool has_density = false;
  while (std::getline(input, line) && line != "end_header") {
    has_density |=
        line == "property float value" || line == "property double value";
  }
  ASSERT_EQ(line, "end_header");
  EXPECT_FALSE(has_density);
}

TEST(PoissonMeshing, TrimmerDoesNotReusePreviousArguments) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "input.ply";
  const auto output_path = test_dir / "output.ply";
  {
    std::ofstream input(input_path);
    ASSERT_TRUE(input.is_open());
    input << "ply\nformat ascii 1.0\nelement vertex 3\n"
          << "property float x\nproperty float y\nproperty float z\n"
          << "property float value\nelement face 1\n"
          << "property list uchar int vertex_indices\nend_header\n"
          << "0 0 0 2\n1 0 0 2\n0 1 0 2\n3 0 1 2\n";
    input.close();
    ASSERT_TRUE(input.good());
  }

  for (const std::string omitted : {"all", "in", "trim", "out"}) {
    SCOPED_TRACE(omitted);
    ASSERT_EQ(RunSurfaceTrimmerWithArgs({"--in",
                                         input_path.string(),
                                         "--out",
                                         output_path.string(),
                                         "--trim",
                                         "1",
                                         "--aRatio",
                                         "0"}),
              EXIT_SUCCESS);
    const auto mesh = ReadPlyMesh(output_path).mesh;
    ASSERT_EQ(mesh.vertices.size(), 3);
    ASSERT_EQ(mesh.faces.size(), 1);
    std::filesystem::rename(output_path,
                            test_dir / ("first-" + omitted + ".ply"));

    std::vector<std::string> args;
    if (omitted != "all") {
      if (omitted != "in")
        args.insert(args.end(), {"--in", input_path.string()});
      if (omitted != "trim") args.insert(args.end(), {"--trim", "1"});
      if (omitted != "out")
        args.insert(args.end(), {"--out", output_path.string()});
    }
    EXPECT_EQ(RunSurfaceTrimmerWithArgs(args),
              omitted == "out" ? EXIT_SUCCESS : EXIT_FAILURE);
    EXPECT_FALSE(ExistsFile(output_path));
  }
}

TEST(PoissonMeshing, TrimmerResetsIslandOptions) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "input.ply";
  const auto output_path = test_dir / "output.ply";
  {
    std::ofstream input(input_path);
    ASSERT_TRUE(input.is_open());
    // Disjoint triangles with areas 1 and 0.0001, both above the trim
    // threshold.
    input << "ply\nformat ascii 1.0\nelement vertex 6\n"
          << "property float x\nproperty float y\nproperty float z\n"
          << "property float value\nelement face 2\n"
          << "property list uchar int vertex_indices\nend_header\n"
          << "0 0 0 2\n2 0 0 2\n0 1 0 2\n"
          << "3 0 0 2\n3.02 0 0 2\n3 0.01 0 2\n"
          << "3 0 1 2\n3 3 4 5\n";
    input.close();
    ASSERT_TRUE(input.good());
  }

  const std::vector<std::string> common = {"--in",
                                           input_path.string(),
                                           "--out",
                                           output_path.string(),
                                           "--trim",
                                           "1"};
  for (const bool reset_ratio : {true, false}) {
    SCOPED_TRACE(reset_ratio);
    auto args = common;
    args.insert(args.end(),
                {"--aRatio", reset_ratio ? "0" : "0.001", "--removeIslands"});
    ASSERT_EQ(RunSurfaceTrimmerWithArgs(args), EXIT_SUCCESS);
    auto mesh = ReadPlyMesh(output_path).mesh;
    ASSERT_EQ(mesh.faces.size(), reset_ratio ? 2 : 1);
    ASSERT_EQ(mesh.vertices.size(), reset_ratio ? 6 : 3);

    args = common;
    if (reset_ratio) args.push_back("--removeIslands");
    ASSERT_EQ(RunSurfaceTrimmerWithArgs(args), EXIT_SUCCESS);
    mesh = ReadPlyMesh(output_path).mesh;
    EXPECT_EQ(mesh.faces.size(), reset_ratio ? 1 : 2);
    EXPECT_EQ(mesh.vertices.size(), reset_ratio ? 3 : 6);
    if (reset_ratio) {
      // The remaining vertices must belong to the large triangle.
      for (const auto& vertex : mesh.vertices) EXPECT_LE(vertex.x, 2.0f);
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
