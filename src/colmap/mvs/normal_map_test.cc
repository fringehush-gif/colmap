// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/normal_map.h"

#include "colmap/util/testing.h"

#include <stdexcept>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(NormalMap, Empty) {
  NormalMap normal_map;
  EXPECT_EQ(normal_map.GetWidth(), 0);
  EXPECT_EQ(normal_map.GetHeight(), 0);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, NonEmpty) {
  NormalMap normal_map(1, 2);
  EXPECT_EQ(normal_map.GetWidth(), 1);
  EXPECT_EQ(normal_map.GetHeight(), 2);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, ReadWrite) {
  const auto path = CreateTestDir() / "normal.bin";
  NormalMap expected(2, 3);
  for (size_t i = 0; i < expected.GetData().size(); ++i) {
    expected.GetPtr()[i] = static_cast<float>(i) / 16.0f;
  }
  expected.Write(path);

  NormalMap actual(1, 1);
  actual.Fill(-1.0f);
  ASSERT_NO_THROW(actual.Read(path));
  EXPECT_EQ(actual.GetWidth(), expected.GetWidth());
  EXPECT_EQ(actual.GetHeight(), expected.GetHeight());
  EXPECT_EQ(actual.GetDepth(), 3);
  EXPECT_EQ(actual.GetData(), expected.GetData());
}

TEST(NormalMap, ReadRejectsWrongChannels) {
  const auto path = CreateTestDir() / "normal.bin";
  for (const size_t depth : {1, 2, 4}) {
    SCOPED_TRACE(depth);
    Mat<float> input(4, 6, depth);
    input.Fill(0.5f);
    input.Write(path);

    Mat<float> generic;
    ASSERT_NO_THROW(generic.Read(path));
    EXPECT_EQ(generic.GetDepth(), depth);
    EXPECT_EQ(generic.GetData(), input.GetData());
    NormalMap actual(2, 3);
    actual.Fill(-0.25f);
    const NormalMap previous = actual;
    EXPECT_THAT([&] { actual.Read(path); },
                testing::ThrowsMessage<std::invalid_argument>(
                    testing::HasSubstr("mat.GetDepth() == 3")));
    EXPECT_EQ(actual.GetWidth(), previous.GetWidth());
    EXPECT_EQ(actual.GetHeight(), previous.GetHeight());
    EXPECT_EQ(actual.GetDepth(), previous.GetDepth());
    EXPECT_EQ(actual.GetData(), previous.GetData());

    NormalMap replacement(4, 6);
    replacement.Fill(0.75f);
    replacement.Write(path);
    ASSERT_NO_THROW(actual.Read(path));
    EXPECT_EQ(actual.GetWidth(), replacement.GetWidth());
    EXPECT_EQ(actual.GetHeight(), replacement.GetHeight());
    EXPECT_EQ(actual.GetDepth(), 3);
    EXPECT_EQ(actual.GetData(), replacement.GetData());
  }
}

TEST(NormalMap, ReadFailurePreservesState) {
  NormalMap actual(2, 3);
  actual.Fill(0.5f);
  const NormalMap previous = actual;
  EXPECT_THROW(actual.Read(CreateTestDir() / "missing.bin"),
               std::invalid_argument);
  EXPECT_EQ(actual.GetWidth(), previous.GetWidth());
  EXPECT_EQ(actual.GetHeight(), previous.GetHeight());
  EXPECT_EQ(actual.GetDepth(), previous.GetDepth());
  EXPECT_EQ(actual.GetData(), previous.GetData());
}

TEST(NormalMap, MatOwnershipAndConsumers) {
  Mat<float> mat(5, 3, 3);
  for (size_t row = 0; row < mat.GetHeight(); ++row) {
    for (size_t col = 0; col < mat.GetWidth(); ++col) {
      const size_t axis = (row + 2 * col + 2) % 3;
      mat.Set(row, col, axis, axis == 2 ? -1.0f : 1.0f);
    }
  }
  const auto expected = mat.GetData();
  const float* const data = mat.GetPtr();
  const NormalMap copied(mat);
  EXPECT_NE(copied.GetPtr(), data);
  EXPECT_EQ(copied.GetData(), expected);
  const NormalMap moved(std::move(mat));
  EXPECT_EQ(moved.GetPtr(), data);
  // Do not depend on the moved-from dimensions when reusing the source.
  mat = Mat<float>(1, 1, 3);
  mat.Fill(99.0f);
  EXPECT_EQ(copied.GetData(), expected);
  EXPECT_EQ(moved.GetData(), expected);

  const auto path = CreateTestDir() / "map.bin";
  moved.Write(path);
  NormalMap loaded;
  loaded.Read(path);
  EXPECT_EQ(loaded.GetWidth(), 5);
  EXPECT_EQ(loaded.GetHeight(), 3);
  EXPECT_EQ(loaded.GetDepth(), 3);
  EXPECT_EQ(loaded.GetData(), expected);
  const Bitmap bitmap = loaded.ToBitmap();
  EXPECT_EQ(bitmap.Width(), 5);
  EXPECT_EQ(bitmap.Height(), 3);
  EXPECT_EQ(bitmap.GetPixel(0, 0).value(), BitmapColor<uint8_t>(128, 128, 255));
}

TEST(NormalMap, RejectedMatRemainsReusable) {
  Mat<float> mat(5, 3, 1);
  mat.Fill(0.5f);
  const float* const data = mat.GetPtr();
  EXPECT_THROW((NormalMap{mat}), std::invalid_argument);
  EXPECT_THROW((NormalMap{std::move(mat)}), std::invalid_argument);
  // An invalid channel count is rejected before storage is moved.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(mat.GetPtr(), data);
  EXPECT_EQ(mat.GetWidth(), 5);
  EXPECT_EQ(mat.GetHeight(), 3);
  EXPECT_EQ(mat.GetDepth(), 1);
  EXPECT_EQ(mat.GetData(), std::vector<float>(15 * 1, 0.5f));
  mat = Mat<float>(5, 3, 3);
  mat.Fill(0.75f);
  const NormalMap accepted(std::move(mat));
  EXPECT_EQ(accepted.GetData(), std::vector<float>(15 * 3, 0.75f));
}

TEST(NormalMap, Rescale) {
  NormalMap normal_map(6, 7);
  normal_map.Rescale(0.5);
  EXPECT_EQ(normal_map.GetWidth(), 3);
  EXPECT_EQ(normal_map.GetHeight(), 4);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, Downsize) {
  NormalMap normal_map(6, 7);
  normal_map.Downsize(2, 4);
  EXPECT_EQ(normal_map.GetWidth(), 2);
  EXPECT_EQ(normal_map.GetHeight(), 2);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, ToBitmap) {
  NormalMap normal_map(2, 2);
  normal_map.Set(0, 0, 0, 0);
  normal_map.Set(0, 0, 1, 0);
  normal_map.Set(0, 0, 2, 1);
  normal_map.Set(0, 1, 0, 0);
  normal_map.Set(0, 1, 1, 1);
  normal_map.Set(0, 1, 2, 0);
  normal_map.Set(1, 0, 0, 1);
  normal_map.Set(1, 0, 1, 0);
  normal_map.Set(1, 0, 2, 0);
  normal_map.Set(1, 1, 0, 1 / std::sqrt(2.0f));
  normal_map.Set(1, 1, 1, 1 / std::sqrt(2.0f));
  normal_map.Set(1, 1, 2, 0);
  const Bitmap bitmap = normal_map.ToBitmap();
  EXPECT_EQ(bitmap.Width(), normal_map.GetWidth());
  EXPECT_EQ(bitmap.Height(), normal_map.GetHeight());
  EXPECT_TRUE(bitmap.IsRGB());
  EXPECT_EQ(bitmap.GetPixel(0, 0).value(), BitmapColor<uint8_t>(128, 128, 0));
  EXPECT_EQ(bitmap.GetPixel(0, 1).value(), BitmapColor<uint8_t>(0, 128, 0));
  EXPECT_EQ(bitmap.GetPixel(1, 0).value(), BitmapColor<uint8_t>(128, 0, 0));
  EXPECT_EQ(bitmap.GetPixel(1, 1).value(), BitmapColor<uint8_t>(37, 37, 0));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
