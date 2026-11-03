// SPDX-License-Identifier: BSD-3-Clause
// Include the owner to exercise its private component graph invariant.
#include <array>
#include <cstring>
#include <vector>

#include "SurfaceTrimmer.cpp"
#include <gtest/gtest.h>

namespace {
template <typename Index>
std::vector<Index> Indices(const typename ComponentGraph<Index>::Node& node) {
  return {node.polygonIndices.begin(), node.polygonIndices.end()};
}

template <typename Index, size_t Count>
void Sanity(
    const std::array<typename ComponentGraph<Index>::Node, Count>& nodes) {
  ComponentGraph<Index>::SanityCheck(nodes.size(),
                                     [&](size_t i) { return &nodes[i]; });
}

template <typename Index>
void CheckLeaf() {
  using Node = typename ComponentGraph<Index>::Node;
  auto arena = PoissonRecon::PolygonIndexChain<Index>::MakeArena(4);
  const auto* const data = arena.data();
  const auto capacity = arena.capacity();
  std::array<Node, 2> nodes;
  const std::array<Index, 2> first{0, 1}, second{2, 3};
  nodes[0].polygonIndices.bind(arena);
  nodes[1].polygonIndices.bind(arena);
  nodes[0].polygonIndices.append(first.begin(), first.end());
  nodes[1].polygonIndices.append(second.begin(), second.end());
  nodes[0].area = 1;
  nodes[1].area = 3;
  nodes[0].neighbors = {&nodes[1]};
  nodes[1].neighbors = {&nodes[0]};
  ASSERT_NO_THROW((Sanity<Index, 2>(nodes)));
  nodes[0].merge();
  EXPECT_EQ(nodes[0].area, 0);
  EXPECT_TRUE(nodes[0].polygonIndices.empty());
  EXPECT_TRUE(nodes[0].neighbors.empty());
  EXPECT_EQ(nodes[1].area, 4);
  EXPECT_EQ(Indices<Index>(nodes[1]), (std::vector<Index>{2, 3, 0, 1}));
  EXPECT_TRUE(nodes[1].neighbors.empty());
  const std::vector<Index> expected{
      1, PoissonRecon::PolygonIndexChain<Index>::Missing, 3, 0};
  EXPECT_EQ(arena, expected);
  EXPECT_EQ(
      std::memcmp(arena.data(), expected.data(), arena.size() * sizeof(Index)),
      0);
  EXPECT_EQ(arena.data(), data);
  EXPECT_EQ(arena.size(), 4);
  EXPECT_EQ(arena.capacity(), capacity);
  testing::Test::RecordProperty("arena_storage_bytes",
                                static_cast<int>(arena.size() * sizeof(Index)));
  EXPECT_NO_THROW((Sanity<Index, 2>(nodes)));
}

template <typename Index>
void CheckFourCycle() {
  using Node = typename ComponentGraph<Index>::Node;
  auto arena = PoissonRecon::PolygonIndexChain<Index>::MakeArena(4);
  const auto* const data = arena.data();
  const auto capacity = arena.capacity();
  std::array<Node, 4> nodes;
  for (Index i = 0; i < 4; ++i) {
    nodes[i].area = double(i + 1);
    nodes[i].polygonIndices.bind(arena);
    nodes[i].polygonIndices.append(&i, &i + 1);
  }
  nodes[0].neighbors = {&nodes[1], &nodes[2]};
  nodes[1].neighbors = {&nodes[0], &nodes[3]};
  nodes[2].neighbors = {&nodes[0], &nodes[3]};
  nodes[3].neighbors = {&nodes[1], &nodes[2]};
  ASSERT_NO_THROW((Sanity<Index, 4>(nodes)));
  nodes[0].merge();
  for (const size_t retired : {size_t{0}, size_t{2}}) {
    EXPECT_EQ(nodes[retired].area, 0);
    EXPECT_TRUE(nodes[retired].polygonIndices.empty());
    EXPECT_TRUE(nodes[retired].neighbors.empty());
  }
  EXPECT_EQ(nodes[1].area, 6);
  EXPECT_EQ(nodes[3].area, 4);
  EXPECT_EQ(Indices<Index>(nodes[1]), (std::vector<Index>{1, 0, 2}));
  EXPECT_EQ(Indices<Index>(nodes[3]), (std::vector<Index>{3}));
  EXPECT_EQ(nodes[1].neighbors, (std::vector<Node*>{&nodes[3]}));
  EXPECT_EQ(nodes[3].neighbors, (std::vector<Node*>{&nodes[1]}));
  const auto missing = PoissonRecon::PolygonIndexChain<Index>::Missing;
  const std::vector<Index> expected{2, 0, missing, missing};
  EXPECT_EQ(arena, expected);
  EXPECT_EQ(
      std::memcmp(arena.data(), expected.data(), arena.size() * sizeof(Index)),
      0);
  EXPECT_EQ(arena.data(), data);
  EXPECT_EQ(arena.size(), 4);
  EXPECT_EQ(arena.capacity(), capacity);
  testing::Test::RecordProperty("arena_storage_bytes",
                                static_cast<int>(arena.size() * sizeof(Index)));
  EXPECT_NO_THROW((Sanity<Index, 4>(nodes)));
}

TEST(SurfaceTrimmerComponentGraph, LeafInt32) { CheckLeaf<int>(); }
TEST(SurfaceTrimmerComponentGraph, LeafInt64) { CheckLeaf<long long>(); }
TEST(SurfaceTrimmerComponentGraph, FourCycleInt32) { CheckFourCycle<int>(); }
TEST(SurfaceTrimmerComponentGraph, FourCycleInt64) {
  CheckFourCycle<long long>();
}

}  // namespace
