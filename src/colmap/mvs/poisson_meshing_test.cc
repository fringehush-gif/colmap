// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/poisson_meshing.h"

#include "colmap/math/random_eigen.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include "thirdparty/PoissonRecon/Geometry.h"
#include "thirdparty/PoissonRecon/RegularTree.h"

#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
#include "thirdparty/PoissonRecon/aux_gpu.h"
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

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

struct ChildInitializationData {
  int calls = 0;
  int serial = -1;
};

template <unsigned int Dim>
void CheckThreadSafeChildInitialization() {
  using Node = PoissonRecon::RegularTreeNode<Dim, ChildInitializationData, int>;
  constexpr int kChildCount = 1 << Dim;
  Node root;
  int callbacks = 0;
  auto initialize = [&](Node& child) {
    ++callbacks;
    ++child.nodeData.calls;
    child.nodeData.serial = callbacks;
  };

  ASSERT_TRUE(root.template initChildren<true>(nullptr, initialize));
  ASSERT_NE(root.children, nullptr);
  EXPECT_EQ(callbacks, kChildCount);
  std::array<int, kChildCount> published_serials{};
  for (int child_index = 0; child_index < kChildCount; ++child_index) {
    const Node& child = root.children[child_index];
    const auto position = child.depthAndOffset();
    EXPECT_EQ(child.parent, &root);
    EXPECT_EQ(child.children, nullptr);
    EXPECT_EQ(child.nodeData.calls, 1);
    EXPECT_EQ(position.depth, 1);
    for (unsigned int axis = 0; axis < Dim; ++axis) {
      EXPECT_EQ(position.offset[axis], (child_index >> axis) & 1);
    }
    published_serials[child_index] = child.nodeData.serial;
  }

  // A subsequent attempt loses the CAS and must not modify published nodes.
  Node* published = root.children;
  EXPECT_FALSE(root.template initChildren<true>(nullptr, initialize));
  EXPECT_EQ(root.children, published);
  for (int child_index = 0; child_index < kChildCount; ++child_index) {
    EXPECT_EQ(root.children[child_index].nodeData.calls, 1);
    EXPECT_EQ(root.children[child_index].nodeData.serial,
              published_serials[child_index]);
  }
  root.cleanChildren(true);
}

TEST(PoissonMeshing, ThreadSafeChildrenInitializedOnce) {
  CheckThreadSafeChildInitialization<2>();
  CheckThreadSafeChildInitialization<3>();
}

TEST(PoissonMeshing, FinalizerWorkerCountsRetainFiniteColoredMesh) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "points.ply";
  WriteRandomPlyPoints(input_path);

  PoissonMeshingOptions options;
  options.point_weight = 1.0;
  options.depth = 3;
  options.trim = 0.0;
  options.color = true;

  for (const int workers : std::array<int, 3>{1, 3, 12}) {
    SCOPED_TRACE(workers);
    options.num_threads = workers;
    const auto output_path =
        test_dir / ("mesh-workers-" + std::to_string(workers) + ".ply");
    ASSERT_TRUE(PoissonMeshing(options, input_path, output_path));
    ASSERT_TRUE(ExistsFile(output_path));

    const std::vector<PlyPoint> vertices = ReadPly(output_path);
    ASSERT_GE(vertices.size(), 3);
    for (const PlyPoint& vertex : vertices) {
      EXPECT_TRUE(std::isfinite(vertex.x));
      EXPECT_TRUE(std::isfinite(vertex.y));
      EXPECT_TRUE(std::isfinite(vertex.z));
      EXPECT_EQ(vertex.r, 0);
      EXPECT_LE(std::abs(static_cast<int>(vertex.g) - 64), 1);
      EXPECT_LE(std::abs(static_cast<int>(vertex.b) - 128), 1);
    }
  }
}

#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
TEST(PoissonMeshing, EmptyGpuAuxiliaryViewPreservesOutput) {
  const ::AuxNativeGpu::View view{};
  ::AuxNativeGpu::Four output{{-777, -777, -777, -777}};
  ::AuxNativeGpu::Stats stats;
  EXPECT_FALSE(::AuxNativeGpu::Reduce(view, &output, stats));
  EXPECT_STREQ(stats.reason, "shape");
  for (const float value : output.v) {
    EXPECT_EQ(value, -777);
  }
}

TEST(PoissonMeshing, GpuAuxiliaryReductionRetainsExactSmallTree) {
  ::AuxNativeGpu::Stats admission;
  if (!::AuxNativeGpu::Available(admission)) {
    GTEST_SKIP() << "CUDA auxiliary kernel unavailable: " << admission.reason;
  }
  const std::uint32_t level_offsets[] = {0, 1, 3};
  const std::uint32_t child_offsets[] = {0, 2, 2, 2};
  const std::uint32_t children[] = {1, 2};
  const std::uint32_t seed_offsets[] = {0, 0, 1, 2};
  const std::uint32_t seed_ids[] = {0, 1};
  const ::AuxNativeGpu::Four seeds[] = {{{1, 2, 3, 1}}, {{4, 5, 6, 1}}};
  const ::AuxNativeGpu::View view{3,
                                  2,
                                  1,
                                  level_offsets,
                                  child_offsets,
                                  children,
                                  seed_offsets,
                                  seed_ids,
                                  seeds};
  std::array<::AuxNativeGpu::Four, 3> output;
  ::AuxNativeGpu::Stats stats;
  ASSERT_TRUE(::AuxNativeGpu::Reduce(view, output.data(), stats));
  const float expected[3][4] = {{5, 7, 9, 2}, {4, 8, 12, 4}, {16, 20, 24, 4}};
  for (size_t node = 0; node < output.size(); ++node) {
    for (int channel = 0; channel < 4; ++channel) {
      EXPECT_EQ(output[node].v[channel], expected[node][channel]);
    }
  }
}
#endif

}  // namespace
}  // namespace mvs
}  // namespace colmap
