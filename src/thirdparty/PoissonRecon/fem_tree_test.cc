// SPDX-License-Identifier: BSD-3-Clause

#include <atomic>
#include <cstddef>
#include <tuple>
#include <vector>

// FEMTree requires the configuration declarations before its headers.
// clang-format off
#include "PreProcessor.h"
#include "FEMTree.h"
// clang-format on
#include <gtest/gtest.h>

namespace {

TEST(PoissonFEMTreeTest, NeighborSelectionCanBeEmpty) {
  PoissonRecon::FEMTree<3, float> tree(1024);
  size_t visited = 0;
  const auto visit = [&](const auto*) { ++visited; };
  tree.processNeighbors<0, 0, false>(
      [](const auto*) { return false; }, std::tuple<>{}, visit);
  EXPECT_EQ(visited, 0);
  tree.processNeighbors<0, 0, false>(
      [](const auto*) { return true; }, std::tuple<>{}, visit);
  EXPECT_GT(visited, 0);
}

TEST(PoissonFEMTreeTest, NeighboringLeafSelectionCanBeEmpty) {
  PoissonRecon::FEMTree<3, float> tree(1024);
  std::atomic<size_t> visited{0};
  const auto visit = [&](const auto*) { ++visited; };
  tree.processNeighboringLeaves<0, 0>(
      [](const auto*) { return false; }, visit, false);
  EXPECT_EQ(visited.load(), 0);
  tree.processNeighboringLeaves<0, 0>(
      [](const auto*) { return true; }, visit, false);
  EXPECT_GT(visited.load(), 0);
}

TEST(PoissonFEMTreeTest, InitializesTriangleInsideOneGridCell) {
  using namespace PoissonRecon;
  FEMTree<3, float> tree(1024);
  const std::vector<Point<float, 3>> vertices = {
      Point<float, 3>(0.10f, 0.10f, 0.10f),
      Point<float, 3>(0.20f, 0.10f, 0.10f),
      Point<float, 3>(0.10f, 0.20f, 0.10f)};
  const std::vector<SimplexIndex<2, node_index_type>> triangles = {{0, 1, 2}};
  std::vector<NodeSimplices<3, float>> node_simplices;
  FEMTreeInitializer<3, float>::Initialize(tree.spaceRoot(),
                                           vertices,
                                           triangles,
                                           1,
                                           2,
                                           node_simplices,
                                           tree.nodeAllocators,
                                           tree.initializer());
  ASSERT_FALSE(node_simplices.empty());
  size_t count = 0;
  for (const auto& node : node_simplices) {
    ASSERT_NE(node.node, nullptr);
    for (const auto& simplex : node.data) {
      EXPECT_EQ(simplex.first, 0);
      EXPECT_GT(simplex.second.measure(), 0);
      ++count;
    }
  }
  EXPECT_GT(count, 0);
}

}  // namespace
