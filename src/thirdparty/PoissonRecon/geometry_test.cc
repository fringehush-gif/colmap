// SPDX-License-Identifier: BSD-3-Clause

#include "Geometry.h"
#include "MAT.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace {

TEST(PoissonTriangleArea, RetainsThinFloatGeometryAcrossPermutations) {
  using Point = PoissonRecon::Point<float, 3>;
  constexpr float height = 1e-10f;
  const std::array<Point, 3> points{
      Point(0, 0, 0), Point(1, 0, 0), Point(1, height, 0)};
  const float expected = float(double(height) * double(height) / 4);
  std::array<int, 3> order{0, 1, 2};
  do {
    EXPECT_FLOAT_EQ(PoissonRecon::SquareArea(
                        points[order[0]], points[order[1]], points[order[2]]),
                    expected);
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST(PoissonTriangleArea, PromotesFloatCoordinatesBeforeSubtracting) {
  using Point = PoissonRecon::Point<float, 3>;
  constexpr float coordinate = 16777216.0f;  // 2^24, whose float ULP is 2.
  const std::array<Point, 3> points{Point(-coordinate, -coordinate, 0),
                                    Point(coordinate, coordinate, 0),
                                    Point(coordinate, coordinate + 2, 0)};
  std::array<int, 3> order{0, 1, 2};
  do {
    // The determinant is 2^26, so the squared triangle area is 2^50.
    EXPECT_FLOAT_EQ(PoissonRecon::SquareArea(
                        points[order[0]], points[order[1]], points[order[2]]),
                    1125899906842624.0f);
  } while (std::next_permutation(order.begin(), order.end()));
}

template <typename Real>
class PoissonTriangleAreaTypeTest : public ::testing::Test {};
using AreaTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(PoissonTriangleAreaTypeTest, AreaTypes);

TYPED_TEST(PoissonTriangleAreaTypeTest, RetainsScalarTypeAndPrecision) {
  using Point = PoissonRecon::Point<TypeParam, 4>;
  constexpr TypeParam height = 4 * std::numeric_limits<TypeParam>::epsilon();
  const Point first(0, 0, 0, 0);
  const Point second(1, 1, 0, 0);
  const Point third(2, TypeParam{2} + height, 0, 0);
  const auto square_area = PoissonRecon::SquareArea(first, second, third);
  static_assert(
      std::is_same_v<decltype(PoissonRecon::SquareArea(first, second, third)),
                     TypeParam>);
  // In this plane the determinant is exactly height. In particular, do not
  // narrow long-double coordinates to double on hosts with wider precision.
  EXPECT_EQ(square_area, height * height / 4);
}

TYPED_TEST(PoissonTriangleAreaTypeTest, HandlesTwoDimensionsAndDegeneracy) {
  using Point = PoissonRecon::Point<TypeParam, 2>;
  const Point origin(0, 0);
  const Point first(3, 0);
  const Point second(0, 4);
  EXPECT_EQ(PoissonRecon::SquareArea(origin, first, second), TypeParam{36});
  EXPECT_EQ(PoissonRecon::Area(origin, first, second), TypeParam{6});
  EXPECT_EQ(PoissonRecon::SquareArea(origin, first, Point(6, 0)), TypeParam{0});
  EXPECT_EQ(PoissonRecon::Area(origin, first, first), TypeParam{0});
  using Point4 = PoissonRecon::Point<TypeParam, 4>;
  EXPECT_EQ(PoissonRecon::SquareArea(
                Point4(0, 0, 0, 0), Point4(0, 0, 3, 0), Point4(0, 0, 0, 4)),
            TypeParam{36});
}

template <typename Real>
class PoissonQuadTriangulationTest : public ::testing::Test {};
TYPED_TEST_SUITE(PoissonQuadTriangulationTest, AreaTypes);

TYPED_TEST(PoissonQuadTriangulationTest,
           MinimizesOrdinaryAreaAndPreservesWinding) {
  using Point = PoissonRecon::Point<TypeParam, 3>;
  const std::array<std::array<Point, 4>, 3> fixtures{{
      {Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 1)},
      {Point(0, 0, 0), Point(1, 0, 0), Point(2, 1, 0), Point(0, 2, 4)},
      {Point(TypeParam{5} / 16, TypeParam{3} / 16, TypeParam{9} / 16),
       Point(TypeParam{11} / 16, TypeParam{1} / 16, TypeParam{6} / 16),
       Point(TypeParam{9} / 16, TypeParam{4} / 16, 0),
       Point(0, 0, TypeParam{1} / 16)},
  }};
  for (size_t fixture_idx = 0; fixture_idx < fixtures.size(); ++fixture_idx) {
    std::array<int, 4> order{0, 1, 2, 3};
    for (int winding = 0; winding < 2; ++winding) {
      for (int rotation = 0; rotation < 4; ++rotation) {
        SCOPED_TRACE(testing::Message()
                     << fixture_idx << ":" << winding << ":" << rotation);
        std::vector<Point> vertices;
        for (const int index : order) {
          vertices.push_back(fixtures[fixture_idx][index]);
        }
        const auto triangles =
            PoissonRecon::MinimalAreaTriangulation<int, TypeParam, 3>(
                GetPointer(vertices), vertices.size());
        ASSERT_EQ(triangles.size(), 2);
        std::array<std::array<int, 4>, 4> directed_edges{};
        for (const auto& triangle : triangles) {
          bool has_zero = false;
          bool has_two = false;
          for (const int index : triangle.idx) {
            ASSERT_GE(index, 0);
            ASSERT_LT(index, 4);
            has_zero |= order[index] == 0;
            has_two |= order[index] == 2;
          }
          // Both fixtures have a unique ordinary-area minimum at diagonal
          // (0,2); fixture 1 instead favors (1,3) under a squared-area sum.
          // Fixture 2 is a scaled orthogonal transform of fixture 0.
          EXPECT_TRUE(has_zero && has_two);
          for (int edge = 0; edge < 3; ++edge) {
            const int from = triangle.idx[edge];
            const int to = triangle.idx[(edge + 1) % 3];
            EXPECT_NE(from, to);
            ++directed_edges[from][to];
          }
        }
        for (int index = 0; index < 4; ++index) {
          const int next = (index + 1) % 4;
          EXPECT_EQ(directed_edges[index][next], 1);
          EXPECT_EQ(directed_edges[next][index], 0);
        }
        EXPECT_EQ(directed_edges[0][2], directed_edges[2][0]);
        EXPECT_EQ(directed_edges[1][3], directed_edges[3][1]);
        EXPECT_EQ(directed_edges[0][2] + directed_edges[2][0] +
                      directed_edges[1][3] + directed_edges[3][1],
                  2);
        std::rotate(order.begin(), order.begin() + 1, order.end());
      }
      std::reverse(order.begin(), order.end());
    }
  }
}

TYPED_TEST(PoissonQuadTriangulationTest, RetainsTheExistingEqualAreaTie) {
  using Point = PoissonRecon::Point<TypeParam, 3>;
  const std::vector<Point> square{
      Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0)};
  const auto triangles =
      PoissonRecon::MinimalAreaTriangulation<int, TypeParam, 3>(
          GetPointer(square), square.size());
  ASSERT_EQ(triangles.size(), 2);
  const std::array<std::array<int, 3>, 2> expected{{{0, 1, 2}, {2, 3, 0}}};
  for (size_t i = 0; i < triangles.size(); ++i) {
    for (size_t vertex = 0; vertex < 3; ++vertex) {
      EXPECT_EQ(triangles[i].idx[vertex], expected[i][vertex]);
    }
  }
}

}  // namespace
