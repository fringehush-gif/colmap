// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/testing.h"
#include "SurfaceTrimmer.cpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

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

TEST(SurfaceTrimmerArea, ConfiguredIslandThresholdRetainsThinTriangle) {
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
  // against the other component's area of 1/64. Set the ratio explicitly:
  // the vendor parser retains values between calls.
  std::vector<std::string> args{"SurfaceTrimmer",
                                "--in",
                                input_path.string(),
                                "--out",
                                output_path.string(),
                                "--trim",
                                "1",
                                "--aRatio",
                                "0.001",
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
