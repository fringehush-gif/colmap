// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/poisson_meshing.h"

#include "colmap/math/random_eigen.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include "thirdparty/PoissonRecon/PreProcessor.h"

// FEM declarations require the preceding vendor configuration.
#include "thirdparty/PoissonRecon/FEMTree.h"
#include "thirdparty/PoissonRecon/Geometry.h"
#include "thirdparty/PoissonRecon/PolygonIndexChain.h"
#include "thirdparty/PoissonRecon/RegularTree.h"
#include "thirdparty/PoissonRecon/trim_gpu_adapter.h"

// NestedVector requires stream declarations supplied by RegularTree.
#include "thirdparty/PoissonRecon/NestedVector.h"

#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
#include "thirdparty/PoissonRecon/aux_gpu.h"
#include "thirdparty/PoissonRecon/normal_capture.h"
#include "thirdparty/PoissonRecon/trim_gpu.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
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

std::vector<PlyPoint> MakeDeterministicColoredCubePoints() {
  std::vector<PlyPoint> points;
  points.reserve(96);
  for (int axis = 0; axis < 3; ++axis) {
    for (int sign : std::array<int, 2>{-1, 1}) {
      for (int u = 0; u < 4; ++u) {
        for (int v = 0; v < 4; ++v) {
          std::array<float, 3> position{};
          std::array<float, 3> normal{};
          position[axis] = static_cast<float>(sign);
          position[(axis + 1) % 3] = static_cast<float>(2 * u - 3) * 0.25f;
          position[(axis + 2) % 3] = static_cast<float>(2 * v - 3) * 0.25f;
          normal[axis] = static_cast<float>(sign);
          PlyPoint point;
          point.x = position[0];
          point.y = position[1];
          point.z = position[2];
          point.nx = normal[0];
          point.ny = normal[1];
          point.nz = normal[2];
          point.r = 17;
          point.g = 109;
          point.b = 233;
          points.push_back(point);
        }
      }
    }
  }
  return points;
}

bool WriteReorderedRgbAsciiPly(const std::filesystem::path& path,
                               const std::vector<PlyPoint>& points) {
  std::ofstream output(path);
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  output << "ply\nformat ascii 1.0\nelement vertex " << points.size()
         << "\nproperty float x\nproperty float y\nproperty float z"
         << "\nproperty float nx\nproperty float ny\nproperty float nz"
         << "\nproperty uchar blue\nproperty uchar red"
         << "\nproperty uchar green\nend_header\n";
  for (const PlyPoint& point : points) {
    output << point.x << ' ' << point.y << ' ' << point.z << ' ' << point.nx
           << ' ' << point.ny << ' ' << point.nz << ' '
           << static_cast<int>(point.b) << ' ' << static_cast<int>(point.r)
           << ' ' << static_cast<int>(point.g) << '\n';
  }
  output.close();
  return output.good();
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

template <unsigned int Depth>
void CheckConcurrentNestedVectorSize() {
  constexpr size_t kCapacity = size_t{1} << (3 * (Depth + 1));
  constexpr int kReaders = 4;
  for (int repeat = 0; repeat < 16; ++repeat) {
    PoissonRecon::NestedVector<int, Depth, 3> values;
    EXPECT_EQ(values.size(), 0);
    values.resize(1, -7);
    const int* first = &values[0];
    std::atomic<size_t> requested_size{1};
    std::atomic<int> ready{0};
    std::atomic<bool> done{false};
    std::atomic<bool> invalid_size{false};
    std::array<std::thread, kReaders> readers;
    for (std::thread& reader : readers) {
      reader = std::thread([&] {
        ++ready;
        do {
          const size_t observed = values.size();
          // A lower bound can lag behind growth, but cannot exceed a size
          // announced before that growth started.
          if (observed > requested_size.load()) {
            invalid_size.store(true);
          }
        } while (!done.load());
      });
    }
    while (ready.load() != kReaders) {
      std::this_thread::yield();
    }
    for (size_t count = 2; count <= kCapacity; ++count) {
      requested_size.store(count);
      values.resize(count, -7);
    }
    done.store(true);
    for (std::thread& reader : readers) {
      reader.join();
    }
    EXPECT_FALSE(invalid_size.load());
    EXPECT_EQ(values.size(), kCapacity);
    EXPECT_EQ(&values[0], first);
    for (size_t i = 0; i < kCapacity; ++i) {
      EXPECT_EQ(values[i], -7);
    }
  }
}

TEST(PoissonMeshing, ConcurrentNestedVectorSizeRemainsLowerBound) {
  // Small blocks exercise publication at two recursive nesting depths.
  CheckConcurrentNestedVectorSize<1>();
  CheckConcurrentNestedVectorSize<2>();
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

TEST(PoissonMeshing, StandardAndReorderedRgbInputsRetainOrderedGeometry) {
  const auto test_dir = CreateTestDir();
  const auto standard_input = test_dir / "standard-rgb.ply";
  const auto reordered_input = test_dir / "reordered-rgb.ply";
  const auto standard_output = test_dir / "standard-mesh.ply";
  const auto reordered_output = test_dir / "reordered-mesh.ply";
  const auto points = MakeDeterministicColoredCubePoints();
  WriteBinaryPlyPoints(standard_input, points, true, true);
  ASSERT_TRUE(WriteReorderedRgbAsciiPly(reordered_input, points));

  // The ASCII fallback must decode the same samples before comparing meshes.
  const auto standard_points = ReadPly(standard_input);
  const auto reordered_points = ReadPly(reordered_input);
  ASSERT_EQ(standard_points.size(), points.size());
  ASSERT_EQ(reordered_points.size(), points.size());
  for (size_t index = 0; index < points.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(standard_points[index].x, reordered_points[index].x);
    EXPECT_EQ(standard_points[index].y, reordered_points[index].y);
    EXPECT_EQ(standard_points[index].z, reordered_points[index].z);
    EXPECT_EQ(standard_points[index].nx, reordered_points[index].nx);
    EXPECT_EQ(standard_points[index].ny, reordered_points[index].ny);
    EXPECT_EQ(standard_points[index].nz, reordered_points[index].nz);
    EXPECT_EQ(standard_points[index].r, reordered_points[index].r);
    EXPECT_EQ(standard_points[index].g, reordered_points[index].g);
    EXPECT_EQ(standard_points[index].b, reordered_points[index].b);
  }

  PoissonMeshingOptions options;
  options.point_weight = 1.0;
  options.depth = 4;
  options.trim = 0.0;
  options.color = true;
  options.num_threads = 1;
  ASSERT_TRUE(PoissonMeshing(options, standard_input, standard_output));
  ASSERT_TRUE(PoissonMeshing(options, reordered_input, reordered_output));

  const auto standard_mesh = ReadPly(standard_output);
  const auto reordered_mesh = ReadPly(reordered_output);
  ASSERT_EQ(standard_mesh.size(), reordered_mesh.size());
  ASSERT_GE(standard_mesh.size(), 3);
  for (size_t index = 0; index < standard_mesh.size(); ++index) {
    SCOPED_TRACE(index);
    const PlyPoint& standard = standard_mesh[index];
    const PlyPoint& reordered = reordered_mesh[index];
    EXPECT_TRUE(std::isfinite(standard.x));
    EXPECT_TRUE(std::isfinite(standard.y));
    EXPECT_TRUE(std::isfinite(standard.z));
    EXPECT_TRUE(std::isfinite(reordered.x));
    EXPECT_TRUE(std::isfinite(reordered.y));
    EXPECT_TRUE(std::isfinite(reordered.z));
    EXPECT_EQ(standard.x, reordered.x);
    EXPECT_EQ(standard.y, reordered.y);
    EXPECT_EQ(standard.z, reordered.z);
    EXPECT_LE(std::abs(static_cast<int>(standard.r) - reordered.r), 1);
    EXPECT_LE(std::abs(static_cast<int>(standard.g) - reordered.g), 1);
    EXPECT_LE(std::abs(static_cast<int>(standard.b) - reordered.b), 1);
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

TEST(PoissonMeshing, GpuTrimConnectedComponentsUsesMaximumFaceRoots) {
  ::PoissonTrimGpu::Stats admission;
  if (!::PoissonTrimGpu::Available(admission)) {
    GTEST_SKIP() << "CUDA trim kernel unavailable: " << admission.reason;
  }

  // Faces 0 and 1 share an edge, face 2 is disconnected, and face 3 has
  // repeated and self edges. Representatives must be the maximum face index.
  const std::uint64_t offsets[] = {0, 3, 6, 9, 12};
  const std::uint32_t vertices[] = {0, 1, 2, 2, 1, 3, 4, 5, 6, 7, 7, 8};
  std::uint32_t roots[4] = {};
  ::PoissonTrimGpu::Stats stats;
  ASSERT_TRUE(::PoissonTrimGpu::ConnectedComponents(
      4, 12, offsets, vertices, roots, stats));
  EXPECT_STREQ(stats.reason, "complete");
  EXPECT_EQ(roots[0], 1);
  EXPECT_EQ(roots[1], 1);
  EXPECT_EQ(roots[2], 2);
  EXPECT_EQ(roots[3], 3);
}

TEST(PoissonMeshing, GpuTrimBoundaryGraphAndNonmanifoldRejection) {
  ::PoissonTrimGpu::Stats admission;
  if (!::PoissonTrimGpu::Available(admission)) {
    GTEST_SKIP() << "CUDA trim kernel unavailable: " << admission.reason;
  }

  const std::uint64_t offsets[] = {0, 3, 6, 9};
  const std::uint32_t vertices[] = {0, 1, 2, 1, 0, 3, 4, 5, 6};
  const std::uint32_t components[] = {0, 1, 1};
  std::vector<std::uint64_t> graph;
  ::PoissonTrimGpu::Stats stats;
  ASSERT_TRUE(::PoissonTrimGpu::BoundaryGraph(
      3, 9, offsets, vertices, components, graph, stats));
  ASSERT_EQ(graph.size(), 1);
  EXPECT_EQ(graph.front(), std::uint64_t(1));

  const std::uint64_t nonmanifold_offsets[] = {0, 3, 6, 9};
  const std::uint32_t nonmanifold_vertices[] = {0, 1, 2, 1, 0, 3, 0, 1, 4};
  const std::uint32_t nonmanifold_components[] = {0, 1, 2};
  graph.front() = std::uint64_t(0x0000000200000001);
  EXPECT_FALSE(::PoissonTrimGpu::BoundaryGraph(3,
                                               9,
                                               nonmanifold_offsets,
                                               nonmanifold_vertices,
                                               nonmanifold_components,
                                               graph,
                                               stats));
  EXPECT_STREQ(stats.reason, "nonmanifold");
  ASSERT_EQ(graph.size(), 1);
  EXPECT_EQ(graph.front(), std::uint64_t(0x0000000200000001));
}

PoissonNormalGpu::Record MakeNormalRecord() {
  PoissonNormalGpu::Record record{};
  std::fill(std::begin(record.node), std::end(record.node), UINT32_MAX);
  return record;
}

TEST(PoissonMeshing, NormalReplayRetainsNativeFirstTouchAndRepeatedSlots) {
  using Tree = PoissonRecon::FEMTree<3, float>;
  using Node = Tree::FEMTreeNode;
  using Value = PoissonRecon::Point<float, 3>;
  Node nodes[2];
  nodes[0].nodeData.nodeIndex = 0;
  nodes[1].nodeData.nodeIndex = 1;
  PoissonRecon::SparseNodeData<Value, PoissonRecon::UIntPack<0, 0, 0>> field;
  const Value zero;

  auto record = MakeNormalRecord();
  record.node[0] = 1;
  record.coefficient[0] = 0.5f;
  record.node[1] = 0;
  record.coefficient[1] = 1.0f;
  record.node[26] = 1;
  record.coefficient[26] = 0.25f;
  record.value[0] = 2.0f;
  record.value[1] = 4.0f;
  record.value[2] = 8.0f;
  const std::vector<std::vector<PoissonNormalGpu::Record>> blocks{{record}};
  std::vector<std::uint32_t> order;
  PoissonNormalCapture::Replay(
      blocks,
      2,
      [&](std::uint32_t index) { return &nodes[index]; },
      [&](Node* node, const float* components, float coefficient) {
        order.push_back(node->nodeData.nodeIndex);
        Value value;
        for (int channel = 0; channel < 3; ++channel) {
          value[channel] = components[channel];
        }
        PoissonRecon::Atomic<Value>::Add(field.at(node, zero),
                                         value * coefficient);
      });
  EXPECT_EQ(order, (std::vector<std::uint32_t>{1, 0, 1}));
  EXPECT_EQ(field.size(), 2);
  EXPECT_EQ(field.index(&nodes[1]), 0);
  EXPECT_EQ(field.index(&nodes[0]), 1);
  for (int channel = 0; channel < 3; ++channel) {
    EXPECT_EQ(field.at(&nodes[0], zero)[channel], record.value[channel]);
    EXPECT_EQ(field.at(&nodes[1], zero)[channel],
              record.value[channel] * 0.75f);
  }
}

TEST(PoissonMeshing, GpuNormalPrelaunchRefusalPreservesOutputs) {
  const auto record = MakeNormalRecord();
  const PoissonNormalGpu::Record* blocks[] = {&record};
  const std::size_t counts[] = {1};
  std::array<float, 3> output{7, 8, 9};
  std::uint32_t touched = 77;
  PoissonNormalGpu::Stats stats;
  EXPECT_FALSE(PoissonNormalGpu::Reduce(
      blocks, counts, 1, 128000001, output.data(), &touched, stats));
  EXPECT_STREQ(stats.reason, "shape");
  EXPECT_EQ(output, (std::array<float, 3>{7, 8, 9}));
  EXPECT_EQ(touched, 77);

  // This is rejected from metadata before reading the claimed record array.
  const std::size_t oversized_counts[] = {8000000};
  EXPECT_FALSE(PoissonNormalGpu::Reduce(
      blocks, oversized_counts, 1, 128000000, output.data(), &touched, stats));
  EXPECT_STREQ(stats.reason, "workspace-budget");
  EXPECT_EQ(output, (std::array<float, 3>{7, 8, 9}));
  EXPECT_EQ(touched, 77);
}

TEST(PoissonMeshing, GpuNormalReductionRetainsSlotsAndSubnormalBits) {
  PoissonNormalGpu::Stats stats;
  if (!PoissonNormalGpu::Available(stats)) {
    GTEST_SKIP() << stats.reason;
  }
  std::array<PoissonNormalGpu::Record, 3> records;
  for (auto& record : records) {
    record = MakeNormalRecord();
  }
  records[0].node[0] = 0;
  records[0].coefficient[0] = 0.5f;
  records[0].node[26] = 2;
  records[0].coefficient[26] = 1;
  records[0].value[0] = 1;
  records[0].value[1] = 2;
  records[0].value[2] = 3;
  records[1].node[0] = 0;
  records[1].coefficient[0] = 0.25f;
  records[1].value[0] = 4;
  records[1].value[1] = 8;
  records[1].value[2] = 12;
  records[2].node[12] = 3;
  records[2].coefficient[12] = 1;
  const std::uint32_t denormal_bits = 1;
  std::memcpy(&records[2].value[0], &denormal_bits, sizeof(denormal_bits));
  const PoissonNormalGpu::Record* blocks[] = {records.data(), nullptr};
  const std::size_t counts[] = {records.size(), 0};
  std::array<float, 12> output;
  output.fill(-7);
  std::array<std::uint32_t, 4> touched{77, 77, 77, 77};
  ASSERT_TRUE(PoissonNormalGpu::Reduce(
      blocks, counts, 2, 4, output.data(), touched.data(), stats));
  EXPECT_EQ(touched, (std::array<std::uint32_t, 4>{1, 0, 1, 1}));
  EXPECT_EQ(output[0], 1.5f);
  EXPECT_EQ(output[1], 3.0f);
  EXPECT_EQ(output[2], 4.5f);
  for (int channel = 0; channel < 3; ++channel) {
    EXPECT_EQ(output[3 + channel], 0);
    EXPECT_EQ(output[6 + channel], records[0].value[channel]);
  }
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &output[9], sizeof(result_bits));
  EXPECT_EQ(result_bits, denormal_bits);
}

TEST(PoissonMeshing, GpuNormalInvalidNodeRemainsAnError) {
  PoissonNormalGpu::Stats stats;
  if (!PoissonNormalGpu::Available(stats)) {
    GTEST_SKIP() << stats.reason;
  }
  auto record = MakeNormalRecord();
  record.node[0] = 1;
  record.coefficient[0] = 1;
  record.value[0] = 1;
  const PoissonNormalGpu::Record* blocks[] = {&record};
  const std::size_t counts[] = {1};
  std::array<float, 3> output{7, 8, 9};
  std::uint32_t touched = 77;
  EXPECT_THROW(PoissonNormalGpu::Reduce(
                   blocks, counts, 1, 1, output.data(), &touched, stats),
               std::runtime_error);
  EXPECT_EQ(output, (std::array<float, 3>{7, 8, 9}));
  EXPECT_EQ(touched, 77);
}

struct NormalSlabTestContext {
  std::vector<PoissonNormalGpu::Record>* records = nullptr;
  std::vector<float> values;
  std::vector<std::uint32_t> touched;
  std::size_t next_node = 0;
  int ready_calls = 0;
  int consume_calls = 0;
  bool released = false;
  bool throw_ready = false;
  bool throw_consume = false;
};

void NormalSlabReady(void* opaque) {
  auto& context = *static_cast<NormalSlabTestContext*>(opaque);
  ++context.ready_calls;
  if (context.throw_ready) throw std::runtime_error("ready callback");
  if (context.records) {
    std::memset(context.records->data(),
                0xa5,
                context.records->size() * sizeof(PoissonNormalGpu::Record));
    std::vector<PoissonNormalGpu::Record>().swap(*context.records);
  }
  context.released = true;
}

void NormalSlabConsume(void* opaque,
                       std::size_t begin,
                       std::size_t count,
                       const float* values,
                       const std::uint32_t* touched) {
  auto& context = *static_cast<NormalSlabTestContext*>(opaque);
  ++context.consume_calls;
  EXPECT_EQ(context.ready_calls, 1);
  EXPECT_TRUE(context.released);
  EXPECT_EQ(begin, context.next_node);
  if (context.throw_consume) throw std::runtime_error("consume callback");
  context.values.insert(context.values.end(), values, values + count * 3);
  context.touched.insert(context.touched.end(), touched, touched + count);
  context.next_node += count;
}

TEST(PoissonMeshing, GpuNormalSlabsMatchFullReadbackAfterCaptureRelease) {
  PoissonNormalGpu::Stats stats;
  if (!PoissonNormalGpu::Available(stats)) GTEST_SKIP() << stats.reason;
  for (const std::size_t slab_nodes : {1, 2, 3, 5, 10, 11, 17}) {
    SCOPED_TRACE(slab_nodes);
    std::vector<PoissonNormalGpu::Record> records(3, MakeNormalRecord());
    records[0].node[0] = 0;
    records[0].coefficient[0] = 0.5f;
    records[0].node[26] = 10;
    records[0].coefficient[26] = 1;
    records[0].value[0] = 1;
    records[0].value[1] = 2;
    records[0].value[2] = 3;
    records[1].node[0] = 0;
    records[1].coefficient[0] = 0.25f;
    records[1].value[0] = 4;
    records[1].value[1] = 8;
    records[1].value[2] = 12;
    records[2].node[12] = 3;
    records[2].coefficient[12] = 1;
    const std::uint32_t denormal_bits = 1;
    std::memcpy(&records[2].value[0], &denormal_bits, sizeof(denormal_bits));
    const PoissonNormalGpu::Record* blocks[] = {records.data(), nullptr};
    const std::size_t counts[] = {records.size(), 0};
    std::array<float, 33> full_values;
    full_values.fill(-7);
    std::array<std::uint32_t, 11> full_touched;
    full_touched.fill(77);
    ASSERT_TRUE(PoissonNormalGpu::Reduce(
        blocks, counts, 2, 11, full_values.data(), full_touched.data(), stats));

    NormalSlabTestContext context;
    context.records = &records;
    const PoissonNormalGpu::SlabCallbacks callbacks{
        &context, NormalSlabReady, NormalSlabConsume};
    ASSERT_TRUE(PoissonNormalGpu::ReduceSlabs(
        blocks, counts, 2, 11, slab_nodes, callbacks, stats));
    EXPECT_EQ(context.ready_calls, 1);
    EXPECT_TRUE(context.released);
    EXPECT_TRUE(records.empty());
    EXPECT_EQ(context.next_node, 11);
    EXPECT_EQ(context.consume_calls,
              static_cast<int>((10 + slab_nodes) / slab_nodes));
    ASSERT_EQ(context.values.size(), full_values.size());
    ASSERT_EQ(context.touched.size(), full_touched.size());
    EXPECT_EQ(
        std::memcmp(
            context.values.data(), full_values.data(), sizeof(full_values)),
        0);
    EXPECT_EQ(
        context.touched,
        (std::vector<std::uint32_t>(full_touched.begin(), full_touched.end())));
    std::uint32_t result_bits = 0;
    std::memcpy(&result_bits, &context.values[9], sizeof(result_bits));
    EXPECT_EQ(result_bits, denormal_bits);
  }
}

TEST(PoissonMeshing, GpuNormalSlabPrelaunchRefusalCallsNoCallbacks) {
  const auto record = MakeNormalRecord();
  const PoissonNormalGpu::Record* blocks[] = {&record};
  NormalSlabTestContext context;
  const PoissonNormalGpu::SlabCallbacks counting_callbacks{
      &context, NormalSlabReady, NormalSlabConsume};
  PoissonNormalGpu::Stats stats;
  const std::size_t one[] = {1};
  EXPECT_FALSE(PoissonNormalGpu::ReduceSlabs(
      blocks, one, 1, 128000001, 1, counting_callbacks, stats));
  EXPECT_STREQ(stats.reason, "shape");
  EXPECT_FALSE(PoissonNormalGpu::ReduceSlabs(
      blocks, one, 1, 11, 0, counting_callbacks, stats));
  EXPECT_STREQ(stats.reason, "shape");
  const std::size_t oversized_counts[] = {8000000};
  EXPECT_FALSE(PoissonNormalGpu::ReduceSlabs(
      blocks, oversized_counts, 1, 128000000, 1, counting_callbacks, stats));
  EXPECT_STREQ(stats.reason, "workspace-budget");
  EXPECT_EQ(context.ready_calls, 0);
  EXPECT_EQ(context.consume_calls, 0);
}

TEST(PoissonMeshing, GpuNormalSlabValidationAndCallbackErrorsPropagate) {
  PoissonNormalGpu::Stats stats;
  if (!PoissonNormalGpu::Available(stats)) GTEST_SKIP() << stats.reason;
  auto record = MakeNormalRecord();
  record.node[0] = 1;
  record.coefficient[0] = 1;
  record.value[0] = 1;
  const PoissonNormalGpu::Record* blocks[] = {&record};
  const std::size_t counts[] = {1};
  NormalSlabTestContext context;
  const PoissonNormalGpu::SlabCallbacks callbacks{
      &context, NormalSlabReady, NormalSlabConsume};
  EXPECT_THROW(
      PoissonNormalGpu::ReduceSlabs(blocks, counts, 1, 1, 1, callbacks, stats),
      std::runtime_error);
  EXPECT_EQ(context.ready_calls, 0);
  EXPECT_EQ(context.consume_calls, 0);

  record.node[0] = 0;
  record.value[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(
      PoissonNormalGpu::ReduceSlabs(blocks, counts, 1, 1, 1, callbacks, stats),
      std::runtime_error);
  EXPECT_EQ(context.ready_calls, 0);
  EXPECT_EQ(context.consume_calls, 0);

  record.value[0] = 1;
  context.throw_ready = true;
  EXPECT_THROW(
      PoissonNormalGpu::ReduceSlabs(blocks, counts, 1, 1, 1, callbacks, stats),
      std::runtime_error);
  EXPECT_EQ(context.ready_calls, 1);
  EXPECT_EQ(context.consume_calls, 0);
  context = NormalSlabTestContext{};
  context.throw_consume = true;
  EXPECT_THROW(
      PoissonNormalGpu::ReduceSlabs(blocks, counts, 1, 1, 1, callbacks, stats),
      std::runtime_error);
  EXPECT_EQ(context.ready_calls, 1);
  EXPECT_EQ(context.consume_calls, 1);
}

#endif

struct TrimReplayNode {
  double area = 0;
  std::vector<TrimReplayNode*> neighbors;
  std::vector<int> polygonIndices;
};

void ConnectTrimReplayNodes(std::vector<TrimReplayNode>& nodes,
                            std::size_t first,
                            std::size_t second) {
  nodes[first].neighbors.push_back(&nodes[second]);
  nodes[second].neighbors.push_back(&nodes[first]);
}

TEST(PoissonMeshing, TrimMergeReplayCompletesNewDegreeOneChain) {
  std::vector<TrimReplayNode> nodes(3);
  nodes[0].area = 1;
  nodes[1].area = 2;
  nodes[2].area = 10;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    nodes[i].polygonIndices.push_back(static_cast<int>(i));
  }
  ConnectTrimReplayNodes(nodes, 0, 1);
  ConnectTrimReplayNodes(nodes, 1, 2);

  std::size_t completed_merges = 99;
  EXPECT_TRUE(::PoissonTrimGpu::HasOrderIndependentMerges(
      nodes, 5.0, completed_merges));
  EXPECT_EQ(completed_merges, 2);
  EXPECT_EQ(nodes[0].area, 1);
  EXPECT_EQ(nodes[1].area, 2);
  EXPECT_EQ(nodes[2].area, 10);
  EXPECT_EQ(nodes[1].neighbors.size(), 2);
}

TEST(PoissonMeshing, TrimMergeReplayRejectsSelectedBranchingNode) {
  std::vector<TrimReplayNode> nodes(3);
  nodes[0].area = 2;
  nodes[1].area = 1;
  nodes[2].area = 3;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    nodes[i].polygonIndices.push_back(static_cast<int>(i));
  }
  ConnectTrimReplayNodes(nodes, 0, 1);
  ConnectTrimReplayNodes(nodes, 1, 2);

  std::size_t completed_merges = 99;
  EXPECT_FALSE(::PoissonTrimGpu::HasOrderIndependentMerges(
      nodes, 5.0, completed_merges));
  EXPECT_EQ(completed_merges, 0);
}

TEST(PoissonMeshing, TrimMergeReplayUsesMinimumIndexForTiesAndIgnoresIsolated) {
  std::vector<TrimReplayNode> tied_nodes(3);
  tied_nodes[0].area = 1;
  tied_nodes[1].area = 1;
  tied_nodes[2].area = 10;
  for (std::size_t i = 0; i < tied_nodes.size(); ++i) {
    tied_nodes[i].polygonIndices.push_back(static_cast<int>(i));
  }
  ConnectTrimReplayNodes(tied_nodes, 0, 1);
  ConnectTrimReplayNodes(tied_nodes, 1, 2);

  std::size_t completed_merges = 99;
  EXPECT_TRUE(::PoissonTrimGpu::HasOrderIndependentMerges(
      tied_nodes, 1.5, completed_merges));
  EXPECT_EQ(completed_merges, 1);

  std::vector<TrimReplayNode> isolated_nodes(2);
  isolated_nodes[0].area = 0.5;
  isolated_nodes[1].area = 1;
  isolated_nodes[0].polygonIndices.push_back(0);
  isolated_nodes[1].polygonIndices.push_back(1);
  EXPECT_TRUE(::PoissonTrimGpu::HasOrderIndependentMerges(
      isolated_nodes, 5.0, completed_merges));
  EXPECT_EQ(completed_merges, 0);
}

TEST(PoissonMeshing, TrimMergeReplayRejectsNonfiniteAreaAndThreshold) {
  std::vector<TrimReplayNode> nodes(1);
  nodes[0].area = 1;
  nodes[0].polygonIndices.push_back(0);
  std::size_t completed_merges = 99;

  EXPECT_FALSE(::PoissonTrimGpu::HasOrderIndependentMerges(
      nodes, std::numeric_limits<double>::quiet_NaN(), completed_merges));
  EXPECT_EQ(completed_merges, 0);
  EXPECT_FALSE(::PoissonTrimGpu::HasOrderIndependentMerges(
      nodes, std::numeric_limits<double>::infinity(), completed_merges));
  EXPECT_EQ(completed_merges, 0);

  nodes[0].area = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(::PoissonTrimGpu::HasOrderIndependentMerges(
      nodes, 5.0, completed_merges));
  EXPECT_EQ(completed_merges, 0);
  nodes[0].area = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(::PoissonTrimGpu::HasOrderIndependentMerges(
      nodes, 5.0, completed_merges));
  EXPECT_EQ(completed_merges, 0);
}

using ScratchPoint = ::PoissonRecon::Point<float>;
using ScratchData = ::PoissonRecon::DirectSum<float, ScratchPoint>;
using ScratchExtractor =
    ::PoissonRecon::_LevelSetExtractor<true, float, 3, ScratchData>;

ScratchExtractor::Key ScratchKey(unsigned int index) {
  ScratchExtractor::Key key;
  key[0] = index;
  key[1] = index + 100;
  key[2] = index + 200;
  return key;
}

ScratchPoint ScratchRgb(float value) {
  ScratchPoint result(std::size_t(3));
  result[0] = value;
  result[1] = value + 0.25f;
  result[2] = value + 0.5f;
  return result;
}

ScratchExtractor::Vertex ScratchVertex(float value) {
  ScratchExtractor::Vertex result;
  for (int axis = 0; axis < 3; ++axis) {
    result.template get<0>()[axis] = value + static_cast<float>(axis);
    result.template get<1>()[axis] = value + static_cast<float>(axis) + 0.125f;
  }
  result.template get<2>() = value + 0.75f;
  result.template get<3>().template get<0>() = ScratchRgb(value);
  return result;
}

TEST(PoissonMeshing, DynamicPointMoveRetainsOwnershipAndDimensions) {
  EXPECT_TRUE(std::is_nothrow_move_constructible_v<ScratchPoint>);
  EXPECT_TRUE(std::is_nothrow_move_constructible_v<ScratchExtractor::Vertex>);
  ScratchPoint source = ScratchRgb(1.0f);
  float* owned = &source[0];
  ScratchPoint moved(std::move(source));
  EXPECT_EQ(source.dim(), 0);
  ASSERT_EQ(moved.dim(), 3);
  EXPECT_EQ(&moved[0], owned);
  EXPECT_FLOAT_EQ(moved[2], 1.5f);
  moved = std::move(moved);
  ASSERT_EQ(moved.dim(), 3);
  EXPECT_EQ(&moved[0], owned);

  ScratchPoint target = ScratchRgb(5.0f);
  target = std::move(moved);
  EXPECT_EQ(moved.dim(), 0);
  ASSERT_EQ(target.dim(), 3);
  EXPECT_EQ(&target[0], owned);
  ScratchPoint empty;
  target = std::move(empty);
  EXPECT_EQ(target.dim(), 0);
  EXPECT_EQ(empty.dim(), 0);

  ScratchPoint three = ScratchRgb(7.0f);
  ScratchPoint two(std::size_t(2));
  two[0] = 9.0f;
  two[1] = 10.0f;
  float* three_owned = &three[0];
  float* two_owned = &two[0];
  EXPECT_THROW(three = std::move(two), std::exception);
  EXPECT_EQ(three.dim(), 3);
  EXPECT_EQ(two.dim(), 2);
  EXPECT_EQ(&three[0], three_owned);
  EXPECT_EQ(&two[0], two_owned);

  std::vector<ScratchPoint> points;
  points.reserve(1);
  points.push_back(ScratchRgb(0.0f));
  float* first_owned = &points[0][0];
  for (int i = 1; i < 128; ++i)
    points.push_back(ScratchRgb(static_cast<float>(i)));
  EXPECT_EQ(&points[0][0], first_owned);
  for (int i = 0; i < 128; ++i) {
    ASSERT_EQ(points[i].dim(), 3);
    EXPECT_FLOAT_EQ(points[i][2], static_cast<float>(i) + 0.5f);
  }
}

template <typename Values>
void CheckScratchMovesAcrossWorkers() {
  const unsigned int prior_threads = ::PoissonRecon::ThreadPool::NumThreads();
  ::PoissonRecon::ThreadPool::SetNumThreads(2);
  struct RestoreThreads {
    unsigned int count;
    ~RestoreThreads() { ::PoissonRecon::ThreadPool::SetNumThreads(count); }
  } restore{prior_threads};
  Values values;
  typename Values::Scratch scratch;
  const auto edge_key = ScratchKey(3);
  scratch.eKeyValues[0].emplace_back(edge_key,
                                     std::make_pair(10, ScratchVertex(11.0f)));
  scratch.eKeyValues[1].emplace_back(edge_key,
                                     std::make_pair(20, ScratchVertex(21.0f)));
  float* last_owned = &scratch.eKeyValues[1]
                           .back()
                           .second.second.template get<3>()
                           .template get<0>()[0];
  values.setFromScratch(scratch.eKeyValues);
  auto edge = values.edgeVertexMap.find(edge_key);
  ASSERT_NE(edge, values.edgeVertexMap.end());
  EXPECT_EQ(edge->second.first, 20);
  EXPECT_FLOAT_EQ(edge->second.second.template get<3>().template get<0>()[0],
                  21.0f);
  EXPECT_EQ(&edge->second.second.template get<3>().template get<0>()[0],
            last_owned);

  const auto face_key = ScratchKey(4);
  std::vector<ScratchExtractor::IsoEdge> first;
  first.reserve(2);
  first.emplace_back(ScratchKey(31), ScratchKey(32));
  scratch.fKeyValues[0].emplace_back(face_key, std::move(first));
  scratch.fKeyValues[1].emplace_back(
      face_key,
      std::vector<ScratchExtractor::IsoEdge>{
          ScratchExtractor::IsoEdge(ScratchKey(33), ScratchKey(34))});
  const auto* first_owned = scratch.fKeyValues[0][0].second.data();
  values.setFromScratch(scratch.fKeyValues);
  auto face = values.faceEdgeMap.find(face_key);
  ASSERT_NE(face, values.faceEdgeMap.end());
  ASSERT_EQ(face->second.size(), 2);
  EXPECT_EQ(face->second.data(), first_owned);
  EXPECT_EQ(face->second[0][0], ScratchKey(31));
  EXPECT_EQ(face->second[0][1], ScratchKey(32));
  EXPECT_EQ(face->second[1][0], ScratchKey(33));
  EXPECT_EQ(face->second[1][1], ScratchKey(34));
}

TEST(PoissonMeshing, ExtractionScratchMovesPreserveDuplicateOrder) {
  CheckScratchMovesAcrossWorkers<ScratchExtractor::SliceValues>();
  CheckScratchMovesAcrossWorkers<ScratchExtractor::XSliceValues>();
}

template <typename Index>
void CheckPolygonIndexChainSpliceOrder() {
  using Chain = PoissonRecon::PolygonIndexChain<Index>;
  static_assert(
      std::is_same_v<typename std::iterator_traits<
                         typename Chain::const_iterator>::iterator_category,
                     std::input_iterator_tag>);
  auto links = Chain::MakeArena(12);
  std::array<Chain, 6> chains;
  std::array<std::list<Index>, 6> expected;
  const std::array<std::vector<Index>, 6> components = {
      std::vector<Index>{0, 3, 6},
      std::vector<Index>{1, 4},
      std::vector<Index>{},
      std::vector<Index>{2, 5},
      std::vector<Index>{7, 8},
      std::vector<Index>{9, 10, 11}};

  const auto check = [&](size_t index) {
    const Chain& chain = chains[index];
    const std::vector<Index> actual(chain.begin(), chain.end());
    const std::vector<Index> oracle(expected[index].begin(),
                                    expected[index].end());
    EXPECT_EQ(actual, oracle) << "component " << index;
    EXPECT_EQ(chain.size(), oracle.size()) << "component " << index;
    EXPECT_EQ(chain.empty(), chain.begin() == chain.end())
        << "component " << index;
  };
  for (size_t i = 0; i < chains.size(); ++i) {
    chains[i].bind(links);
    check(i);
    chains[i].append(components[i].begin(), components[i].end());
    expected[i].insert(
        expected[i].end(), components[i].begin(), components[i].end());
    check(i);
  }

  const auto merge = [&](size_t destination, size_t source) {
    chains[destination].append(chains[source]);
    expected[destination].splice(expected[destination].end(), expected[source]);
    check(destination);
    check(source);
  };
  merge(0, 1);  // Nonempty destination and donor.
  merge(0, 2);  // Empty donor.
  merge(2, 0);  // Empty destination; donor becomes empty.
  merge(2, 3);
  merge(4, 5);
  merge(2, 4);  // Preserve the order of two prior merged chains.
  for (size_t i = 0; i < chains.size(); ++i) {
    check(i);
  }
}

TEST(PoissonMeshing, PolygonIndexChainIntPreservesSpliceOrder) {
  CheckPolygonIndexChainSpliceOrder<int>();
}

TEST(PoissonMeshing, PolygonIndexChainLongLongPreservesSpliceOrder) {
  CheckPolygonIndexChainSpliceOrder<long long>();
}

TEST(PoissonMeshing, ParallelForVisitsEachIndexOnceAcrossBackends) {
  using Pool = PoissonRecon::ThreadPool;
  std::vector<Pool::ParallelType> backends = {Pool::NONE, Pool::ASYNC};
#ifdef _OPENMP
  backends.push_back(Pool::OPEN_MP);
#endif
  for (const auto backend : backends) {
    for (const auto schedule : {Pool::STATIC, Pool::DYNAMIC}) {
      for (const size_t chunk : {size_t{1}, size_t{128}}) {
        for (const size_t end : {size_t{7}, size_t{8}, size_t{300}}) {
          std::vector<std::atomic<int>> visits(300);
          for (auto& count : visits) {
            count.store(0);
          }
          std::atomic<bool> invalid_thread{false};
          Pool::ParallelFor(
              7,
              end,
              [&](unsigned int thread, size_t index) {
                if (thread >= 4 || (backend == Pool::NONE && thread != 0)) {
                  invalid_thread.store(true);
                }
                visits.at(index).fetch_add(1);
              },
              4,
              backend,
              schedule,
              chunk);
          EXPECT_FALSE(invalid_thread.load());
          for (size_t index = 0; index < visits.size(); ++index) {
            EXPECT_EQ(visits[index].load(), index >= 7 && index < end ? 1 : 0);
          }
        }
      }
    }
  }
}

TEST(PoissonMeshing, ParallelForPreservesCallableStateSemantics) {
  using Pool = PoissonRecon::ThreadPool;
  struct Callback {
    int calls = 0;
    void operator()(unsigned int, size_t) { ++calls; }
  };
  Callback raw;
  Pool::ParallelFor(0, 3, raw, 1, Pool::NONE);
  EXPECT_EQ(raw.calls, 0);

  const std::function<void(unsigned int, size_t)> erased = Callback{};
  Pool::ParallelFor(0, 3, erased, 1, Pool::NONE);
  ASSERT_NE(erased.target<Callback>(), nullptr);
  EXPECT_EQ(erased.target<Callback>()->calls, 3);

  int last_call = 0;
  Pool::ParallelFor(
      0,
      3,
      [count = 0, &last_call](unsigned int, size_t) mutable {
        last_call = ++count;
      },
      1,
      Pool::NONE);
  EXPECT_EQ(last_call, 3);
}

template <unsigned int Dim, unsigned int LeftRadius, unsigned int RightRadius>
void CheckCachedNeighborWindows() {
  using Node = PoissonRecon::RegularTreeNode<Dim, int, int>;
  using Left = ParameterPack::IsotropicUIntPack<Dim, LeftRadius>;
  using Right = ParameterPack::IsotropicUIntPack<Dim, RightRadius>;
  using Key = typename Node::template ConstNeighborKey<Left, Right>;
  constexpr int kChildren = 1 << Dim;
  constexpr int kWidth = LeftRadius + RightRadius + 1;
  Node root;
  auto initialize = [](Node&) {};
  auto grow = [&](Node& node) {
    return node.template initChildren<false>(nullptr, initialize);
  };
  ASSERT_TRUE(grow(root));
  ASSERT_TRUE(grow(root.children[0]));
  ASSERT_TRUE(grow(root.children[kChildren - 1]));
  ASSERT_TRUE(grow(root.children[0].children[0]));
  ASSERT_TRUE(grow(root.children[kChildren - 1].children[kChildren - 1]));
  std::vector<const Node*> nodes;
  auto collect = [&](auto&& self, const Node* node) -> void {
    nodes.push_back(node);
    if (node->children != nullptr) {
      for (int child = 0; child < kChildren; ++child) {
        self(self, &node->children[child]);
      }
    }
  };
  collect(collect, &root);
  const std::vector<const Node*> queries = {
      &root,
      &root.children[0],
      &root.children[kChildren - 1],
      &root.children[0].children[0],
      &root.children[kChildren - 1].children[0],
      &root.children[0].children[0].children[kChildren - 1],
      &root.children[kChildren - 1].children[kChildren - 1].children[0],
      &root.children[0].children[kChildren - 1],
      &root.children[kChildren - 1],
      &root,
      &root.children[0].children[0],
      &root.children[0].children[0].children[0],
  };
  Key key;
  key.set(3);
  for (const Node* query : queries) {
    int depth;
    int offset[Dim];
    query->depthAndOffset(depth, offset);
    const auto& window = key.getNeighbors(query).neighbors;
    int indices[Dim] = {};
    for (unsigned int slot = 0; slot < window.Size(); ++slot) {
      const Node* expected = nullptr;
      for (const Node* node : nodes) {
        int node_depth;
        int node_offset[Dim];
        node->depthAndOffset(node_depth, node_offset);
        bool matches = node_depth == depth;
        for (unsigned int dim = 0; dim < Dim; ++dim) {
          matches &= node_offset[dim] ==
                     offset[dim] + indices[dim] - static_cast<int>(LeftRadius);
        }
        if (matches) {
          expected = node;
          break;
        }
      }
      EXPECT_EQ(window(indices), expected)
          << "dimension=" << Dim << " radii=" << LeftRadius << ","
          << RightRadius << " depth=" << depth << " slot=" << slot;
      for (unsigned int dim = 0; dim < Dim; ++dim) {
        if (++indices[dim] < kWidth) {
          break;
        }
        indices[dim] = 0;
      }
    }
  }
  root.cleanChildren(true);
}

template <unsigned int Dim>
void CheckCachedNeighborRadii() {
  CheckCachedNeighborWindows<Dim, 0, 0>();
  CheckCachedNeighborWindows<Dim, 1, 1>();
  CheckCachedNeighborWindows<Dim, 2, 2>();
  CheckCachedNeighborWindows<Dim, 2, 0>();
  CheckCachedNeighborWindows<Dim, 0, 2>();
}

TEST(PoissonMeshing, CachedNeighborWindowsMatchSparseTree1D) {
  CheckCachedNeighborRadii<1>();
}

TEST(PoissonMeshing, CachedNeighborWindowsMatchSparseTree2D) {
  CheckCachedNeighborRadii<2>();
}

TEST(PoissonMeshing, CachedNeighborWindowsMatchSparseTree3D) {
  CheckCachedNeighborRadii<3>();
}

template <unsigned int Dim, bool Mutable>
void CheckNeighborKeyCopyAssignment() {
  using Node = PoissonRecon::RegularTreeNode<Dim, int, int>;
  using Radius = ParameterPack::IsotropicUIntPack<Dim, 1>;
  using Key = std::conditional_t<
      Mutable,
      typename Node::template NeighborKey<Radius, Radius>,
      typename Node::template ConstNeighborKey<Radius, Radius>>;
  SCOPED_TRACE(Dim);

  Node root;
  auto initialize = [](Node&) {};
  ASSERT_TRUE(root.template initChildren<false>(nullptr, initialize));
  ASSERT_TRUE(
      root.children[0].template initChildren<false>(nullptr, initialize));
  Node* leaf = &root.children[0].children[0];

  auto cached_slots = [](const Key& key) {
    std::vector<const Node*> slots;
    for (int depth = 0; depth <= key.depth(); ++depth) {
      const auto& window = key.neighbors[depth].neighbors;
      for (unsigned int slot = 0; slot < window.Size(); ++slot) {
        slots.push_back(window.data[slot]);
      }
    }
    return slots;
  };

  Key source;
  source.set(2);
  if constexpr (Mutable) {
    source.template getNeighbors<false, false>(leaf, nullptr);
  } else {
    source.getNeighbors(leaf);
  }
  const std::vector<const Node*> expected = cached_slots(source);
  ASSERT_FALSE(expected.empty());
  {
    Key destination;
    destination.set(1);
    EXPECT_EQ(&(destination = source), &destination);
    EXPECT_EQ(destination.depth(), source.depth());
    EXPECT_NE(destination.neighbors, source.neighbors);
    EXPECT_EQ(cached_slots(destination), expected);

    EXPECT_EQ(&(destination = destination), &destination);
    EXPECT_EQ(cached_slots(destination), expected);
    destination.set(-1);
    EXPECT_EQ(cached_slots(source), expected);
  }
  EXPECT_EQ(cached_slots(source), expected);
  {
    Key destination;
    destination = source;
    EXPECT_EQ(cached_slots(destination), expected);
  }
  EXPECT_EQ(cached_slots(source), expected);
  EXPECT_EQ(&(source = source), &source);
  EXPECT_EQ(cached_slots(source), expected);

  Key empty;
  Key previously_allocated;
  previously_allocated.set(0);
  EXPECT_EQ(&(previously_allocated = empty), &previously_allocated);
  EXPECT_EQ(previously_allocated.depth(), -1);
  EXPECT_EQ(previously_allocated.neighbors, nullptr);
  EXPECT_EQ(&(empty = empty), &empty);
  root.cleanChildren(true);
}

TEST(PoissonMeshing, ConstNeighborKeyCopyAssignmentOwnsCachedWindows) {
  CheckNeighborKeyCopyAssignment<1, false>();
  CheckNeighborKeyCopyAssignment<2, false>();
  CheckNeighborKeyCopyAssignment<3, false>();
}

TEST(PoissonMeshing, NeighborKeyCopyAssignmentOwnsCachedWindows) {
  CheckNeighborKeyCopyAssignment<1, true>();
  CheckNeighborKeyCopyAssignment<2, true>();
  CheckNeighborKeyCopyAssignment<3, true>();
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
