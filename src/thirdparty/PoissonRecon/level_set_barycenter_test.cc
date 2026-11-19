// SPDX-License-Identifier: BSD-3-Clause

// FEMTree.h requires the vendor configuration before its declarations.
// clang-format off
#include "PreProcessor.h"
#include "FEMTree.h"
// clang-format on

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStream.imp.h"
#include <gtest/gtest.h>

namespace {

// The extractor accepts a scalar type. Make its default state a defined NaN so
// the old default-construct-and-multiply-by-zero accumulator fails without
// reading arbitrary stack bytes. Every input vertex is initialized explicitly.
struct NaNDefaultFloat {
  NaNDefaultFloat() : value(std::numeric_limits<float>::quiet_NaN()) {}
  // Match the implicit numeric conversions used by the generic scalar API.
  // NOLINTNEXTLINE(google-explicit-constructor)
  NaNDefaultFloat(float value) : value(value) {}
  explicit operator double() const { return value; }

  NaNDefaultFloat& operator+=(NaNDefaultFloat other) {
    value += other.value;
    return *this;
  }
  NaNDefaultFloat& operator-=(NaNDefaultFloat other) {
    value -= other.value;
    return *this;
  }
  NaNDefaultFloat& operator*=(NaNDefaultFloat other) {
    value *= other.value;
    return *this;
  }
  NaNDefaultFloat& operator/=(NaNDefaultFloat other) {
    value /= other.value;
    return *this;
  }

  friend NaNDefaultFloat operator+(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs += rhs;
  }
  friend NaNDefaultFloat operator-(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs -= rhs;
  }
  friend NaNDefaultFloat operator-(NaNDefaultFloat value) {
    return -value.value;
  }
  friend NaNDefaultFloat operator*(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs *= rhs;
  }
  friend NaNDefaultFloat operator/(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs /= rhs;
  }
  friend bool operator==(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs.value == rhs.value;
  }
  friend bool operator!=(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs.value != rhs.value;
  }
  friend bool operator<(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs.value < rhs.value;
  }
  friend bool operator>(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs.value > rhs.value;
  }
  friend bool operator<=(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs.value <= rhs.value;
  }
  friend bool operator>=(NaNDefaultFloat lhs, NaNDefaultFloat rhs) {
    return lhs.value >= rhs.value;
  }
  friend NaNDefaultFloat sqrt(NaNDefaultFloat value) {
    return std::sqrt(value.value);
  }

  float value;
};

static_assert(std::is_trivially_copyable_v<NaNDefaultFloat>);

template <typename Real>
class LevelSetBarycenterTest : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double, NaNDefaultFloat>;
TYPED_TEST_SUITE(LevelSetBarycenterTest, ScalarTypes);

TYPED_TEST(LevelSetBarycenterTest, AveragesInitializedVertexAttributes) {
  using Real = TypeParam;
  using Point = PoissonRecon::Point<Real, 3>;
  using Extractor = PoissonRecon::_LevelSetExtractor<true, Real, 3, Point>;
  using Vertex = typename Extractor::Vertex;
  using Index = PoissonRecon::node_index_type;

  const std::vector<Vertex> input{
      Vertex(Point(0, 0, 0), Point(1, 0, 0), Real(2), Point(12, 24, 36)),
      Vertex(Point(2, 0, 0), Point(0, 2, 0), Real(4), Point(36, 48, 60)),
      Vertex(Point(2, 4, 0), Point(0, 0, 3), Real(6), Point(60, 72, 84)),
      Vertex(Point(0, 4, 0), Point(3, 2, 1), Real(8), Point(84, 96, 108))};
  std::vector<std::pair<Index, Vertex>> polygon;
  for (Index i = 0; i < static_cast<Index>(input.size()); ++i) {
    polygon.emplace_back(i, input[i]);
  }
  std::vector<Vertex> vertices = input;
  std::vector<std::vector<Index>> faces;
  PoissonRecon::VectorBackedOutputDataStream<Vertex> vertex_stream(vertices);
  PoissonRecon::OutputDataStream<Vertex>& vertex_output = vertex_stream;
  PoissonRecon::VectorBackedOutputDataStream<std::vector<Index>> face_stream(
      faces);

  // Opposite corners share z=0, selecting the actual manifold fan path.
  EXPECT_EQ(Extractor::AddIsoPolygons(
                0, vertex_output, face_stream, polygon, false, true),
            4);
  ASSERT_EQ(vertices.size(), 5);
  EXPECT_EQ(faces,
            (std::vector<std::vector<Index>>{
                {0, 4, 1}, {1, 4, 2}, {2, 4, 3}, {3, 4, 0}}));
  const Vertex& center = vertices.back();
  const std::array<double, 3> position{1, 2, 0};
  const std::array<double, 3> gradient{1, 1, 1};
  const std::array<double, 3> color{48, 60, 72};
  for (size_t axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(static_cast<double>(center.template get<0>()[axis]),
              position[axis]);
    EXPECT_EQ(static_cast<double>(center.template get<1>()[axis]),
              gradient[axis]);
    EXPECT_EQ(static_cast<double>(center.template get<3>()[axis]), color[axis]);
  }
  const double density = static_cast<double>(center.template get<2>());
  EXPECT_TRUE(std::isfinite(density));
  EXPECT_EQ(density, 5);
}

TEST(LevelSetBarycenter, TriangleDoesNotInsertVertex) {
  using Point = PoissonRecon::Point<float, 3>;
  using Extractor = PoissonRecon::_LevelSetExtractor<false, float, 3, char>;
  using Vertex = Extractor::Vertex;
  using Index = PoissonRecon::node_index_type;

  const std::vector<std::pair<Index, Vertex>> input{
      {0, Vertex(Point(0, 0, 0), Point(0, 0, 1), 2.0f)},
      {1, Vertex(Point(1, 0, 0), Point(0, 0, 1), 4.0f)},
      {2, Vertex(Point(0, 1, 0), Point(0, 0, 1), 6.0f)}};
  auto polygon = input;
  std::vector<Vertex> vertices;
  std::vector<std::vector<Index>> faces;
  PoissonRecon::VectorBackedOutputDataStream<Vertex> vertex_stream(vertices);
  PoissonRecon::OutputDataStream<Vertex>& vertex_output = vertex_stream;
  PoissonRecon::VectorBackedOutputDataStream<std::vector<Index>> face_stream(
      faces);
  EXPECT_EQ(Extractor::AddIsoPolygons(
                0, vertex_output, face_stream, polygon, false, true),
            1);
  EXPECT_TRUE(vertices.empty());
  EXPECT_EQ(faces, (std::vector<std::vector<Index>>{{2, 1, 0}}));
}

}  // namespace
