// SPDX-License-Identifier: BSD-3-Clause
// The vendor configuration must precede its declarations.
// clang-format off
#include "PreProcessor.h"
#include "Reconstructors.h"
// clang-format on

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

namespace PoissonRecon {
namespace {

class SerialThreads {
 public:
  SerialThreads()
      : count_(ThreadPool::NumThreads()),
        mode_(ThreadPool::ParallelizationType) {
    ThreadPool::SetNumThreads(1);
    ThreadPool::ParallelizationType = ThreadPool::NONE;
  }
  ~SerialThreads() {
    ThreadPool::SetNumThreads(count_);
    ThreadPool::ParallelizationType = mode_;
  }

 private:
  unsigned int count_;
  ThreadPool::ParallelType mode_;
};

constexpr unsigned int kSignature =
    FEMDegreeAndBType<1, BOUNDARY_FREE>::Signature;
constexpr int kDepth = 3;
constexpr float kIsoValue = 0x1p-26f;

template <unsigned int Dim>
using Plane =
    Reconstructor::Implicit<float, Dim, IsotropicUIntPack<Dim, kSignature>>;

template <unsigned int Dim>
void InitializePlane(Plane<Dim>& implicit) {
  auto& tree = implicit.tree;
  FEMTreeInitializer<Dim, float>::Initialize(
      tree.spaceRoot(),
      kDepth,
      [](int, int*) { return true; },
      tree.nodeAllocators.front(),
      tree.initializer());
  tree.template finalizeForMultigrid<1, 1>(
      kDepth,
      [](int depth, const int*) { return depth <= kDepth; },
      [](const typename FEMTree<Dim, float>::FEMTreeNode*) { return false; },
      std::make_tuple());
  implicit.solution =
      tree.initDenseNodeData(IsotropicUIntPack<Dim, kSignature>());
  for (node_index_type i = 0;
       i < static_cast<node_index_type>(tree.nodesSize());
       ++i) {
    const auto* node = tree.node(i);
    int depth, offset[Dim];
    tree.depthAndOffset(node, depth, offset);
    if (depth != kDepth || node->nodeData.nodeIndex < 0) continue;
    // Degree-one nodal coefficients represent sum(x)-Dim/2 exactly here.
    float value = -float(Dim) / 2;
    for (unsigned int d = 0; d < Dim; ++d)
      value += std::ldexp(float(offset[d]), -kDepth);
    implicit.solution[i] = value;
  }
  implicit.isoValue = kIsoValue;
}

template <unsigned int Dim>
struct Mesh {
  std::vector<Point<double, Dim>> positions;
  std::vector<Point<float, Dim>> gradients;
  std::vector<float> densities;
  std::vector<Reconstructor::Face<Dim - 1>> faces;
};

template <typename PositionReal, unsigned int Dim, bool InCore>
Mesh<Dim> Extract(const Plane<Dim>& implicit, bool linear_fit) {
  using Info = Reconstructor::
      OutputVertexInfoWithPosition<PositionReal, float, Dim, true, true>;
  auto factory = Info::GetFactory();
  Reconstructor::OutputInputFactoryTypeStreamWithPosition<
      PositionReal,
      float,
      Dim,
      typename Info::Factory,
      InCore,
      false>
      vertices(factory, Info::Convert);
  Reconstructor::OutputInputFaceStream<Dim - 1, InCore, false> faces;
  Reconstructor::LevelSetExtractionParameters parameters;
  parameters.linearFit = linear_fit;
  parameters.outputGradients = true;
  parameters.outputDensity = true;
  parameters.forceManifold = false;
  parameters.polygonMesh = false;
  parameters.gridCoordinates = false;
  parameters.verbose = false;
  implicit.template extractLevelSet<PositionReal>(vertices, faces, parameters);

  Mesh<Dim> mesh;
  typename Info::Factory::VertexType vertex = factory();
  vertices.reset();
  while (vertices.read(vertex)) {
    mesh.positions.emplace_back(vertex.template get<0>());
    mesh.gradients.push_back(vertex.template get<1>());
    mesh.densities.push_back(vertex.template get<2>());
  }
  Reconstructor::Face<Dim - 1> face;
  faces.reset();
  while (faces.read(face)) mesh.faces.push_back(face);
  return mesh;
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <unsigned int Dim>
void ExpectInteriorRoots() {
  using DoublePoint = Point<double, Dim>;
  SerialThreads serial;
  Plane<Dim> implicit;
  InitializePlane(implicit);
  Point<float, Dim> corner;
  for (unsigned int d = 0; d < Dim; ++d) corner[d] = 0.5f;
  auto evaluator = implicit.evaluator();
  ASSERT_EQ(evaluator(corner), 0.0f);
  for (unsigned int axis = 0; axis < Dim; ++axis) {
    auto endpoint = corner;
    endpoint[axis] += 0.125f;
    ASSERT_EQ(evaluator(endpoint), 0.125f);
  }

  for (bool linear_fit : {true, false}) {
    SCOPED_TRACE(linear_fit);
    const auto legacy = Extract<float, Dim, true>(implicit, linear_fit);
    const auto wide = Extract<double, Dim, true>(implicit, linear_fit);
    const auto streamed = Extract<double, Dim, false>(implicit, linear_fit);
    ASSERT_FALSE(wide.positions.empty());
    ASSERT_EQ(wide.positions.size(), legacy.positions.size());
    ASSERT_EQ(wide.positions.size(), streamed.positions.size());
    EXPECT_EQ(wide.faces, streamed.faces);
    for (size_t i = 0; i < wide.positions.size(); ++i) {
      for (unsigned int d = 0; d < Dim; ++d) {
        EXPECT_EQ(wide.positions[i][d], streamed.positions[i][d]);
        EXPECT_EQ(float(wide.positions[i][d]), float(legacy.positions[i][d]));
        EXPECT_EQ(FloatBits(wide.gradients[i][d]),
                  FloatBits(legacy.gradients[i][d]));
        EXPECT_EQ(FloatBits(wide.gradients[i][d]),
                  FloatBits(streamed.gradients[i][d]));
      }
      EXPECT_EQ(FloatBits(wide.densities[i]), FloatBits(legacy.densities[i]));
      EXPECT_EQ(FloatBits(wide.densities[i]), FloatBits(streamed.densities[i]));
    }

    std::array<node_index_type, Dim> root_ids;
    root_ids.fill(-1);
    for (size_t i = 0; i < wide.positions.size(); ++i) {
      for (unsigned int axis = 0; axis < Dim; ++axis) {
        bool on_edge =
            wide.positions[i][axis] > 0.5 &&
            wide.positions[i][axis] < double(std::nextafter(0.5f, 1.0f));
        for (unsigned int d = 0; d < Dim; ++d)
          if (d != axis) on_edge &= wide.positions[i][d] == 0.5;
        if (on_edge) {
          EXPECT_DOUBLE_EQ(wide.positions[i][axis], 0.5 + double(kIsoValue));
          root_ids[axis] = static_cast<node_index_type>(i);
        }
      }
    }
    for (unsigned int axis = 0; axis < Dim; ++axis) {
      ASSERT_GE(root_ids[axis], 0) << "Missing interior root on axis " << axis;
      for (unsigned int d = 0; d < Dim; ++d)
        EXPECT_EQ(legacy.positions[root_ids[axis]][d], 0.5);
    }

    auto expected_ids = root_ids;
    std::sort(expected_ids.begin(), expected_ids.end());
    bool found_face = false;
    for (const auto& face : wide.faces) {
      std::array<node_index_type, Dim> ids;
      if constexpr (Dim == 2) {
        ids = {face.first, face.second};
      } else {
        if (face.size() != Dim) continue;
        std::copy(face.begin(), face.end(), ids.begin());
      }
      std::sort(ids.begin(), ids.end());
      found_face |= ids == expected_ids;
    }
    ASSERT_TRUE(found_face);
    const auto first =
        wide.positions[root_ids[1]] - wide.positions[root_ids[0]];
    if constexpr (Dim == 2) {
      EXPECT_GT(DoublePoint::SquareNorm(first), 0.0);
    } else {
      const auto second =
          wide.positions[root_ids[2]] - wide.positions[root_ids[0]];
      const auto cross = DoublePoint::CrossProduct(first, second);
      EXPECT_GT(DoublePoint::SquareNorm(cross), 0.0);
    }
  }
}

TEST(LevelSetPositionPrecision, PreservesInteriorCurveRoots) {
  ExpectInteriorRoots<2>();
}

TEST(LevelSetPositionPrecision, PreservesInteriorSurfaceRoots) {
  ExpectInteriorRoots<3>();
}

}  // namespace
}  // namespace PoissonRecon
