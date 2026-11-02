// SPDX-License-Identifier: BSD-3-Clause

#include "Geometry.h"

#include <cstddef>
#include <initializer_list>

#include <gtest/gtest.h>

namespace {

template <typename Real>
PoissonRecon::Point<Real> MakePoint(std::initializer_list<Real> values) {
  PoissonRecon::Point<Real> point(values.size());
  size_t i = 0;
  for (const Real value : values)
    point[i++] = value;
  return point;
}

template <typename Real>
class PoissonDynamicPointTest : public ::testing::Test {};
using PointTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(PoissonDynamicPointTest, PointTypes);

TYPED_TEST(PoissonDynamicPointTest, EmptyAndZeroVectors) {
  using Point = PoissonRecon::Point<TypeParam>;
  const Point empty;
  EXPECT_EQ(Point::Dot(empty, empty), TypeParam{0});
  EXPECT_EQ(Point::SquareNorm(empty), TypeParam{0});
  const Point zero(3);
  EXPECT_EQ(Point::Dot(zero, zero), TypeParam{0});
  EXPECT_EQ(Point::SquareNorm(zero), TypeParam{0});
}

TYPED_TEST(PoissonDynamicPointTest, SignedProductsNormAndOrthogonality) {
  using Point = PoissonRecon::Point<TypeParam>;
  const auto first = MakePoint<TypeParam>({-2, 3, -4});
  const auto second = MakePoint<TypeParam>({5, 6, 7});
  EXPECT_EQ(Point::Dot(first, second), TypeParam{-20});
  EXPECT_EQ(Point::Dot(second, first), TypeParam{-20});
  EXPECT_EQ(Point::SquareNorm(first), TypeParam{29});
  const auto orthogonal = MakePoint<TypeParam>({-3, -2, 0});
  EXPECT_EQ(Point::Dot(first, orthogonal), TypeParam{0});
}

TYPED_TEST(PoissonDynamicPointTest, ExactFractionalProducts) {
  using Point = PoissonRecon::Point<TypeParam>;
  const auto first = MakePoint<TypeParam>({0.5, -1.5, 2});
  const auto second = MakePoint<TypeParam>({-2, 3, 0.25});
  EXPECT_EQ(Point::Dot(first, second), TypeParam{-5});
  EXPECT_EQ(Point::SquareNorm(first), TypeParam{6.5});
}

TYPED_TEST(PoissonDynamicPointTest, DimensionMismatchPreservesException) {
  using Point = PoissonRecon::Point<TypeParam>;
  const Point empty;
  const Point one(1);
  const Point two(2);
  EXPECT_THROW(Point::Dot(empty, one), PoissonRecon::Exception);
  EXPECT_THROW(Point::Dot(two, one), PoissonRecon::Exception);
}

} // namespace
