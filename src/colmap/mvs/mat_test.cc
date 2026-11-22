// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include <array>
#include <limits>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(Mat, Empty) {
  Mat<int> mat;
  EXPECT_EQ(mat.GetWidth(), 0);
  EXPECT_EQ(mat.GetHeight(), 0);
  EXPECT_EQ(mat.GetDepth(), 0);
  EXPECT_EQ(mat.GetNumBytes(), 0);
}

TEST(Mat, EmptyDimensionsPreserveShape) {
  const size_t max_dimension = std::numeric_limits<size_t>::max();
  const std::array<std::array<size_t, 3>, 3> shapes = {{
      {0, max_dimension, max_dimension},
      {max_dimension, 0, max_dimension},
      {max_dimension, max_dimension, 0},
  }};
  for (const auto& shape : shapes) {
    const Mat<float> mat(shape[0], shape[1], shape[2]);
    EXPECT_EQ(mat.GetWidth(), shape[0]);
    EXPECT_EQ(mat.GetHeight(), shape[1]);
    EXPECT_EQ(mat.GetDepth(), shape[2]);
    EXPECT_TRUE(mat.GetData().empty());
    EXPECT_EQ(mat.GetNumBytes(), 0);
  }
}

TEST(Mat, ConstructorRejectsUnrepresentableDimensions) {
  const size_t side = size_t{1} << (std::numeric_limits<size_t>::digits / 2);
  const size_t max_elements = std::vector<float>().max_size();
  const std::array<std::array<size_t, 3>, 8> shapes = {{
      {side, side, 1},
      {side, 1, side},
      {1, side, side},
      {max_elements + 1, 1, 1},
      {1, max_elements + 1, 1},
      {1, 1, max_elements + 1},
      {max_elements, 2, 1},
      {max_elements, 1, 2},
  }};
  for (const auto& shape : shapes) {
    SCOPED_TRACE(::testing::Message()
                 << shape[0] << "x" << shape[1] << "x" << shape[2]);
    EXPECT_THROW(Mat<float>(shape[0], shape[1], shape[2]), std::length_error);
  }
}

TEST(Mat, NonEmpty) {
  Mat<int> mat(1, 2, 3);
  EXPECT_EQ(mat.GetWidth(), 1);
  EXPECT_EQ(mat.GetHeight(), 2);
  EXPECT_EQ(mat.GetDepth(), 3);
  EXPECT_EQ(mat.GetNumBytes(), 24);
}

TEST(Mat, GetSet) {
  Mat<int> mat(1, 2, 3);

  EXPECT_EQ(mat.GetNumBytes(), 24);

  mat.Set(0, 0, 0, 1);
  mat.Set(0, 0, 1, 2);
  mat.Set(0, 0, 2, 3);
  mat.Set(1, 0, 0, 4);
  mat.Set(1, 0, 1, 5);
  mat.Set(1, 0, 2, 6);

  EXPECT_EQ(mat.Get(0, 0, 0), 1);
  EXPECT_EQ(mat.Get(0, 0, 1), 2);
  EXPECT_EQ(mat.Get(0, 0, 2), 3);
  EXPECT_EQ(mat.Get(1, 0, 0), 4);
  EXPECT_EQ(mat.Get(1, 0, 1), 5);
  EXPECT_EQ(mat.Get(1, 0, 2), 6);

  int slice[3];
  mat.GetSlice(0, 0, slice);
  EXPECT_EQ(slice[0], 1);
  EXPECT_EQ(slice[1], 2);
  EXPECT_EQ(slice[2], 3);
  mat.GetSlice(1, 0, slice);
  EXPECT_EQ(slice[0], 4);
  EXPECT_EQ(slice[1], 5);
  EXPECT_EQ(slice[2], 6);
}

TEST(Mat, Fill) {
  Mat<int> mat(1, 2, 3);

  EXPECT_EQ(mat.GetNumBytes(), 24);

  mat.Fill(10);
  mat.Set(0, 0, 0, 10);
  mat.Set(0, 0, 1, 10);
  mat.Set(0, 0, 2, 10);
  mat.Set(1, 0, 0, 10);
  mat.Set(1, 0, 1, 10);
  mat.Set(1, 0, 2, 10);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
