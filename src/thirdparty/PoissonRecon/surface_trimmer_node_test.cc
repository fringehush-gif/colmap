// SPDX-License-Identifier: BSD-3-Clause
// Include the owner to exercise its private component graph invariant.
#include "colmap/util/testing.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <string>
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

TEST(SurfaceTrimmerArea, ThinTriangleIsIndependentOfIndexOrder) {
  using Point = PoissonRecon::Point<float, 3>;
  using Vertex = PoissonRecon::DirectSum<float, Point>;
  constexpr float height = 1.0f / 16384.0f;
  const std::vector<Vertex> vertices{Vertex(Point(0, 0, 0)),
                                     Vertex(Point(1, 0, 0)),
                                     Vertex(Point(1, height, 0))};
  std::vector<int> polygon{0, 1, 2};
  do {
    EXPECT_DOUBLE_EQ((PolygonArea<float, 3, int>(vertices, polygon)),
                     double(height) / 2);
  } while (std::next_permutation(polygon.begin(), polygon.end()));
}

TEST(SurfaceTrimmerArea, ThinPolygonIsIndependentOfWindingAndStart) {
  using Point = PoissonRecon::Point<float, 3>;
  using Vertex = PoissonRecon::DirectSum<float, Point>;
  constexpr float height = 1.0f / 16384.0f;
  const std::vector<Vertex> vertices{Vertex(Point(2, 4, -3)),
                                     Vertex(Point(3, 4, -3)),
                                     Vertex(Point(3, 4 + height, -3)),
                                     Vertex(Point(2, 4 + height, -3))};
  std::vector<int> polygon{0, 1, 2, 3};
  for (int winding = 0; winding < 2; ++winding) {
    for (int rotation = 0; rotation < 4; ++rotation) {
      EXPECT_DOUBLE_EQ((PolygonArea<float, 3, int>(vertices, polygon)),
                       double(height));
      std::rotate(polygon.begin(), polygon.begin() + 1, polygon.end());
    }
    std::reverse(polygon.begin(), polygon.end());
  }
}

TEST(SurfaceTrimmerArea, DefaultIslandThresholdRetainsThinTriangle) {
  using Factory = PoissonRecon::VertexFactory::Factory<
      float,
      PoissonRecon::VertexFactory::PositionFactory<float, 3>,
      PoissonRecon::VertexFactory::ValueFactory<float>>;
  const Factory factory;
  const std::array<std::array<float, 3>, 6> expected{{{0, 0, 0},
                                                      {1, 0, 0},
                                                      {1, 1.0f / 16384.0f, 0},
                                                      {2, 0, 0},
                                                      {2.25f, 0, 0},
                                                      {2, 0.125f, 0}}};
  const auto directory = colmap::CreateTestDir();
  const auto input_path = directory / "input.ply";
  const auto output_path = directory / "output.ply";
  std::vector<Factory::VertexType> input_vertices(expected.size());
  for (size_t i = 0; i < input_vertices.size(); ++i) {
    for (size_t axis = 0; axis < 3; ++axis) {
      input_vertices[i].template get<0>()[axis] = expected[i][axis];
    }
    input_vertices[i].template get<1>() = 2;
  }
  const std::vector<std::vector<int>> input_faces{{0, 1, 2}, {3, 4, 5}};
  // Preserve the exact input coordinates through the round trip.
  PoissonRecon::PLY::WritePolygons<Factory, int>(input_path.string(),
                                                 factory,
                                                 input_vertices,
                                                 input_faces,
                                                 PLY_BINARY_NATIVE,
                                                 {});
  // The thin component's true area is 2^-15, above the default 0.001 ratio
  // against the other component's area of 1/64. Enable actual island removal.
  std::vector<std::string> args{"SurfaceTrimmer",
                                "--in",
                                input_path.string(),
                                "--out",
                                output_path.string(),
                                "--trim",
                                "1",
                                "--removeIslands"};
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  ASSERT_EQ(RunSurfaceTrimmer(static_cast<int>(argv.size()), argv.data()), 0);

  std::vector<Factory::VertexType> vertices;
  std::vector<std::vector<int>> polygons;
  std::vector<std::string> comments;
  int file_type = 0;
  PoissonRecon::PLY::ReadPolygons<Factory, int>(
      output_path.string(), factory, vertices, polygons, file_type, comments);
  ASSERT_EQ(polygons.size(), 2);
  ASSERT_EQ(vertices.size(), 6);
  for (size_t i = 0; i < vertices.size(); ++i) {
    for (size_t axis = 0; axis < 3; ++axis) {
      EXPECT_EQ(vertices[i].template get<0>()[axis], expected[i][axis]);
    }
    EXPECT_EQ(vertices[i].template get<1>(), 2);
  }
  std::sort(polygons.begin(), polygons.end());
  EXPECT_EQ(polygons, (std::vector<std::vector<int>>{{0, 1, 2}, {3, 4, 5}}));
  std::filesystem::remove(input_path);
  std::filesystem::remove(output_path);
  std::filesystem::remove(directory);
}

TEST(SurfaceTrimmerTriangulation,
     QuadRoundTripMinimizesOrdinaryAreaAndPreservesWinding) {
  using Factory = PoissonRecon::VertexFactory::Factory<
      float,
      PoissonRecon::VertexFactory::PositionFactory<float, 3>,
      PoissonRecon::VertexFactory::ValueFactory<float>>;
  const Factory factory;
  // Diagonal (0,2) uniquely minimizes the sum of ordinary triangle areas.
  // The old overwrite and a sum of squared areas instead choose (1,3).
  const std::array<std::array<float, 3>, 4> expected{
      {{0, 0, 0}, {1, 0, 0}, {2, 1, 0}, {0, 2, 4}}};
  std::vector<Factory::VertexType> input_vertices(expected.size());
  for (size_t i = 0; i < input_vertices.size(); ++i) {
    for (size_t axis = 0; axis < 3; ++axis) {
      input_vertices[i].template get<0>()[axis] = expected[i][axis];
    }
    input_vertices[i].template get<1>() = 2;
  }
  const auto directory = colmap::CreateTestDir();
  const auto input_path = directory / "quad-input.ply";
  const auto output_path = directory / "quad-output.ply";
  for (int winding = 0; winding < 2; ++winding) {
    SCOPED_TRACE(winding);
    std::vector<int> polygon{0, 1, 2, 3};
    if (winding != 0) {
      std::reverse(polygon.begin(), polygon.end());
    }
    PoissonRecon::PLY::WritePolygons<Factory, int>(input_path.string(),
                                                   factory,
                                                   input_vertices,
                                                   {polygon},
                                                   PLY_BINARY_NATIVE,
                                                   {});
    // Keep the complete quad and select triangle output. Disable component
    // filtering so this witness isolates the triangulation objective.
    std::vector<std::string> args{"SurfaceTrimmer",
                                  "--in",
                                  input_path.string(),
                                  "--out",
                                  output_path.string(),
                                  "--trim",
                                  "1",
                                  "--aRatio",
                                  "0"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    ASSERT_EQ(RunSurfaceTrimmer(static_cast<int>(argv.size()), argv.data()), 0);

    std::vector<Factory::VertexType> vertices;
    std::vector<std::vector<int>> polygons;
    std::vector<std::string> comments;
    int file_type = 0;
    PoissonRecon::PLY::ReadPolygons<Factory, int>(
        output_path.string(), factory, vertices, polygons, file_type, comments);
    ASSERT_TRUE(file_type == PLY_BINARY_LE || file_type == PLY_BINARY_BE);
    ASSERT_EQ(vertices.size(), 4);
    ASSERT_EQ(polygons.size(), 2);
    for (size_t i = 0; i < vertices.size(); ++i) {
      for (size_t axis = 0; axis < 3; ++axis) {
        EXPECT_EQ(vertices[i].template get<0>()[axis], expected[i][axis]);
      }
      EXPECT_EQ(vertices[i].template get<1>(), 2);
    }
    for (auto& triangle : polygons) {
      ASSERT_EQ(triangle.size(), 3);
      for (const int index : triangle) {
        ASSERT_GE(index, 0);
        ASSERT_LT(index, 4);
      }
      // Cyclic rotation preserves winding; sorting each face would not.
      std::rotate(triangle.begin(),
                  std::min_element(triangle.begin(), triangle.end()),
                  triangle.end());
    }
    std::sort(polygons.begin(), polygons.end());
    const std::vector<std::vector<int>> expected_faces =
        winding == 0 ? std::vector<std::vector<int>>{{0, 1, 2}, {0, 2, 3}}
                     : std::vector<std::vector<int>>{{0, 2, 1}, {0, 3, 2}};
    EXPECT_EQ(polygons, expected_faces);
    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);
  }
  std::filesystem::remove(directory);
}

}  // namespace
