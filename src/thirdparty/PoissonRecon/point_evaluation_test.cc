// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

// FEMTree requires configuration declarations before its headers.
// clang-format off
#include "PreProcessor.h"
#include "FEMTree.h"
// clang-format on
#include <gtest/gtest.h>

namespace {

// Complete free-boundary fields give independent partition-of-unity and
// affine-gradient oracles, including exact knots and their two sides.
template <unsigned int Dim, unsigned int Degree>
struct CellKnotField {
  static constexpr unsigned int kSig =
      PoissonRecon::FEMDegreeAndBType<Degree, PoissonRecon::BOUNDARY_FREE>::Signature;
  using Sigs = PoissonRecon::IsotropicUIntPack<Dim, kSig>;
  using Tree = PoissonRecon::FEMTree<Dim, float>;
  using Vec = PoissonRecon::Point<float, Dim>;
  Tree tree{MEMORY_ALLOCATOR_BLOCK_SIZE};
  PoissonRecon::DenseNodeData<float, Sigs> coefficients;
  PoissonRecon::SparseNodeData<float, Sigs> sparse;

  size_t invalid_nodes = 0;

  explicit CellKnotField(const int grid_depth, const bool cell_labels = false) {
    tree.template finalizeForMultigrid<Degree, Degree>(
        grid_depth, [=](int d, const int*) { return d <= grid_depth; },
        [](const auto*) { return true; }, std::tuple<>{});
    coefficients.resize(tree.nodesSize());
    for (size_t i = 0; i < tree.nodesSize(); ++i) {
      const auto* node = tree.node(i);
      int depth;
      typename Tree::LocalOffset offset;
      tree.depthAndOffset(node, depth, offset);
      float value = 0;
      if (depth == 2) {
        if constexpr (Degree == 0) {
          value = 1;
          if (cell_labels)
            for (unsigned int axis = 0; axis < Dim; ++axis)
              value += std::ldexp(float(offset[axis]), int(2 * axis));
        } else {
          // Odd degrees are node-centered; even degrees are cell-centered.
          for (unsigned int axis = 0; axis < Dim; ++axis)
            value += std::ldexp(1.f, -int(2 * axis)) *
                     (offset[axis] + (Degree % 2 ? 0.f : 0.5f)) / 4;
        }
        if (node->nodeData.nodeIndex < 0 ||
            static_cast<size_t>(node->nodeData.nodeIndex) >= tree.nodesSize())
          ++invalid_nodes;
        else
          sparse[node] = value;
      }
      coefficients[i] = value;
    }
  }
};

template <unsigned int Dim, unsigned int Degree = 1>
void CheckAffinePointKnots() {
  using Field = CellKnotField<Dim, Degree>;
  // Depth 2 exercises current coefficients; depth 3 uses prolonged parent data.
  for (const int grid_depth : {2, 3}) {
    SCOPED_TRACE(grid_depth);
    Field field(grid_depth);
    ASSERT_EQ(field.invalid_nodes, 0);
    typename Field::Tree::template MultiThreadedEvaluator<typename Field::Sigs, 1>
        evaluator(&field.tree, field.coefficients, 1);
    typename Field::Tree::template MultiThreadedSparseEvaluator<typename Field::Sigs, float>
        sparse_evaluator(&field.tree, field.sparse, 1);
    for (unsigned int knot_axis = 0; knot_axis <= Dim; ++knot_axis) {
      SCOPED_TRACE(knot_axis);
      for (const float delta : {-1.f / 1024, 0.f, 1.f / 1024}) {
        SCOPED_TRACE(delta);
        typename Field::Vec point;
        float expected = 0;
        for (unsigned int axis = 0; axis < Dim; ++axis) {
          point[axis] = knot_axis == Dim || knot_axis == axis
                            ? 0.5f + delta : 0.375f + axis / 64.f;
          expected += std::ldexp(1.f, -int(2 * axis)) * point[axis];
        }
        const auto values = evaluator.values(point);
        EXPECT_NEAR(values[0], expected, 2e-6);
        for (unsigned int axis = 0; axis < Dim; ++axis)
          EXPECT_NEAR(values[axis + 1], std::ldexp(1.f, -int(2 * axis)), 2e-5);
        float sparse_value = 0;
        sparse_evaluator.addValue(point, sparse_value);
        EXPECT_NEAR(sparse_value, expected, 2e-6);
      }
    }
  }
}

TEST(PoissonPointEvaluation, AffineInteriorKnots2D) {
  CheckAffinePointKnots<2>();
}

TEST(PoissonPointEvaluation, AffineInteriorKnots3D) {
  CheckAffinePointKnots<3>();
}

TEST(PoissonPointEvaluation, ConstantDegreeZeroInteriorKnots) {
  using Field = CellKnotField<3, 0>;
  for (const int grid_depth : {2, 3}) {
    SCOPED_TRACE(grid_depth);
    Field field(grid_depth);
    ASSERT_EQ(field.invalid_nodes, 0);
    Field::Tree::MultiThreadedEvaluator<Field::Sigs, 0>
        evaluator(&field.tree, field.coefficients, 1);
    Field::Tree::MultiThreadedSparseEvaluator<Field::Sigs, float>
        sparse_evaluator(&field.tree, field.sparse, 1);
    for (const float delta : {-1.f / 1024, 0.f, 1.f / 1024}) {
      SCOPED_TRACE(delta);
      Field::Vec point;
      for (unsigned int axis = 0; axis < 3; ++axis) point[axis] = 0.5f + delta;
      EXPECT_NEAR(evaluator.values(point)[0], 1.f, 2e-6);
      float sparse_value = 0;
      sparse_evaluator.addValue(point, sparse_value);
      EXPECT_NEAR(sparse_value, 1.f, 2e-6);
    }
  }
}


TEST(PoissonPointEvaluation, QuadraticInteriorKnots) {
  CheckAffinePointKnots<3, 2>();
}

TEST(PoissonPointEvaluation, UnequalDegreeZeroUsesSelectedSide) {
  using Field = CellKnotField<3, 0>;
  const std::array<std::pair<float, float>, 5> cases = {{{0.5f - 1.f / 1024, 22},
      {0.5f, 22}, {0.5f + 1.f / 1024, 23}, {0, 21}, {1, 24}}};
  for (const int depth : {2, 3}) {
    Field field(depth, true);
    ASSERT_EQ(field.invalid_nodes, 0);
    Field::Tree::MultiThreadedEvaluator<Field::Sigs, 0>
        evaluator(&field.tree, field.coefficients, 1);
    Field::Tree::MultiThreadedSparseEvaluator<Field::Sigs, float>
        sparse_evaluator(&field.tree, field.sparse, 1);
    for (const auto& item : cases) {
      SCOPED_TRACE(depth);
      SCOPED_TRACE(item.first);
      const Field::Vec point(item.first, 7.f / 16, 3.f / 8);
      // The selected cell owns a tie on the left, not an average.
      EXPECT_FLOAT_EQ(evaluator.values(point)[0], item.second);
      float value = 0;
      sparse_evaluator.addValue(point, value);
      EXPECT_FLOAT_EQ(value, item.second);
    }
  }
}


template <unsigned int Degree>
void CheckPointEndpoints() {
  using Field = CellKnotField<3, Degree>;
  for (const int depth : {2, 3}) {
    Field field(depth);
    ASSERT_EQ(field.invalid_nodes, 0);
    typename Field::Tree::template MultiThreadedEvaluator<typename Field::Sigs, 1>
        evaluator(&field.tree, field.coefficients, 1);
    typename Field::Tree::template MultiThreadedSparseEvaluator<typename Field::Sigs, float>
        sparse_evaluator(&field.tree, field.sparse, 1);
    for (const float end : {0.f, 1.f}) {
      SCOPED_TRACE(Degree);
      SCOPED_TRACE(depth);
      SCOPED_TRACE(end);
      const typename Field::Vec point(end, 7.f / 16, 3.f / 8);
      const float used = end == 0 ? float(1e-6) : float(1. - 1e-6);
      const float sparse_expected = end + 0.25f * point[1] + 0.0625f * point[2];
      const float dense_expected = used + 0.25f * point[1] + 0.0625f * point[2];
      ASSERT_GT(std::abs(dense_expected - sparse_expected), 7e-7);
      const auto values = evaluator.values(point);
      EXPECT_NEAR(values[0], dense_expected, 5e-7);
      EXPECT_NEAR(values[1], 1.f, 2e-5);
      EXPECT_NEAR(values[2], 0.25f, 2e-5);
      EXPECT_NEAR(values[3], 0.0625f, 2e-5);
      float value = 0;
      sparse_evaluator.addValue(point, value);
      EXPECT_NEAR(value, sparse_expected, 5e-7);
    }
  }
}

TEST(PoissonPointEvaluation, EndpointsKeepDenseNudgeDistinctFromSparse) {
  CheckPointEndpoints<1>();
  CheckPointEndpoints<2>();
}


TEST(PoissonPointEvaluation, AdaptiveFinerInteriorKnots) {
  using Field = CellKnotField<3, 1>;
  Field::Tree tree(MEMORY_ALLOCATOR_BLOCK_SIZE);
  tree.finalizeForMultigrid<1, 1>(
      2, [](int d, const int* off) {
        return d <= 2 || (d == 3 && (off[0] == 4 || off[0] == 5) &&
                          off[1] == 3 && off[2] == 3);
      }, [](const auto*) { return true; }, std::tuple<>{});
  ASSERT_EQ(tree.maxDepth(), 3);
  PoissonRecon::DenseNodeData<float, Field::Sigs> coefficients;
  coefficients.resize(tree.nodesSize());
  struct Term { std::array<int, 3> offset; double coefficient; };
  std::vector<Term> terms;
  for (size_t i = 0; i < tree.nodesSize(); ++i) {
    const auto* node = tree.node(i);
    int depth;
    Field::Tree::LocalOffset off;
    tree.depthAndOffset(node, depth, off);
    coefficients[i] = 0;
    if (depth == 3 && tree.isValidFEMNode(Field::Sigs(), node)) {
      coefficients[i] = 1.f + off[0] / 8.f + off[1] / 16.f + off[2] / 32.f;
      terms.push_back({{off[0], off[1], off[2]}, coefficients[i]});
    }
  }
  ASSERT_FALSE(terms.empty());
  Field::Tree::MultiThreadedEvaluator<Field::Sigs, 1>
      evaluator(&tree, coefficients, 1);
  ASSERT_LE(evaluator._coarseCoefficients.size(), tree.nodesSize());
  for (size_t i = 0; i < evaluator._coarseCoefficients.size(); ++i) {
    int depth;
    Field::Tree::LocalOffset off;
    tree.depthAndOffset(tree.node(i), depth, off);
    if (depth <= 1) ASSERT_EQ(evaluator._coarseCoefficients[i], 0);
  }
  for (const float delta : {-1.f / 1024, 0.f, 1.f / 1024}) {
    SCOPED_TRACE(delta);
    const Field::Vec point(0.5f + delta, 3.f / 8, 3.f / 8);
    const auto* node = tree.leaf(point);
    ASSERT_NE(node, nullptr);
    while (PoissonRecon::GetGhostFlag<3>(node)) node = node->parent;
    int depth;
    Field::Tree::LocalOffset off;
    tree.depthAndOffset(node, depth, off);
    if (delta <= 0) ASSERT_EQ(depth, 2);  // Nonzero output needs finer neighbors.
    std::array<double, 4> expected{};
    for (const auto& term : terms) {
      std::array<double, 3> basis{}, derivative{};
      for (int axis = 0; axis < 3; ++axis) {
        const double x = double(point[axis]) * 8 - term.offset[axis];
        basis[axis] = std::max(0., 1. - std::abs(x));
        // Independent linear hat polynomial, with its left piece at a knot.
        derivative[axis] = x > -1 && x <= 0 ? 8 : x > 0 && x <= 1 ? -8 : 0;
      }
      expected[0] += term.coefficient * basis[0] * basis[1] * basis[2];
      for (int axis = 0; axis < 3; ++axis)
        expected[axis + 1] += term.coefficient * derivative[axis] *
                             basis[(axis + 1) % 3] * basis[(axis + 2) % 3];
    }
    ASSERT_GT(expected[0], 0);
    const auto values = evaluator.values(point);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(values[i], expected[i], 1e-4);
  }
}

}  // namespace
