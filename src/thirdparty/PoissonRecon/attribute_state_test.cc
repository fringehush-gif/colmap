// SPDX-License-Identifier: BSD-3-Clause
#include "Geometry.h"
#include <gtest/gtest.h>

namespace {
template <typename Real>
class PoissonAttributeCopyTest : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(PoissonAttributeCopyTest, ScalarTypes);

TYPED_TEST(PoissonAttributeCopyTest, EmptyCopiesAndClears) {
  using Point = PoissonRecon::Point<TypeParam>;
  // Retain a runtime dimension so the old null/zero-byte copy is instrumented.
  volatile size_t dimension = 0;
  const Point empty(dimension);
  Point destination;
  destination = empty;
  EXPECT_EQ(destination.dim(), 0);
  destination = Point(3);
  destination[0] = TypeParam(7);
  destination = empty;
  EXPECT_EQ(destination.dim(), 0);
}

TYPED_TEST(PoissonAttributeCopyTest, CopiesValuesAndPreservesSource) {
  using Point = PoissonRecon::Point<TypeParam>;
  Point source(3);
  source[0] = TypeParam(-2);
  source[1] = TypeParam(3);
  source[2] = TypeParam(0.5);
  Point destination;
  destination = source;
  ASSERT_EQ(destination.dim(), 3);
  for (size_t i = 0; i < 3; ++i) EXPECT_EQ(destination[i], source[i]);
  destination[0] = TypeParam(9);
  EXPECT_EQ(source[0], TypeParam(-2));
  destination = source;
  EXPECT_EQ(destination[0], source[0]);
}

TYPED_TEST(PoissonAttributeCopyTest, SelfAssignmentPreservesState) {
  using Point = PoissonRecon::Point<TypeParam>;
  for (const size_t dimension : {size_t(0), size_t(3)}) {
    Point point(dimension);
    if (dimension) point[0] = TypeParam(17);
    const Point& alias = point;
    point = alias;
    EXPECT_EQ(point.dim(), dimension);
    if (dimension) EXPECT_EQ(point[0], TypeParam(17));
  }
}

TYPED_TEST(PoissonAttributeCopyTest, NonemptyDimensionMismatchPreservesState) {
  using Point = PoissonRecon::Point<TypeParam>;
  Point destination(2);
  destination[0] = TypeParam(11);
  destination[1] = TypeParam(-5);
  const Point source(3);
  EXPECT_THROW(destination = source, PoissonRecon::Exception);
  ASSERT_EQ(destination.dim(), 2);
  EXPECT_EQ(destination[0], TypeParam(11));
  EXPECT_EQ(destination[1], TypeParam(-5));
  EXPECT_EQ(source.dim(), 3);
}
}  // namespace
