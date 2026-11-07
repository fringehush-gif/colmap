// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/normal_map.h"

#include "colmap/image/warp.h"
#include "colmap/util/testing.h"

#include <utility>

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

TEST(NormalMap, RescaleValuesAndReuse) {
  for (const bool zero : {false, true}) {
    NormalMap map(7, 5);
    for (size_t r = 0; r < map.GetHeight(); ++r) {
      for (size_t c = 0; c < map.GetWidth(); ++c) {
        map.Set(r, c, 0, zero ? 0.0f : static_cast<float>(c + 1));
        map.Set(r, c, 1, zero ? 0.0f : -static_cast<float>(r + 1));
        map.Set(r, c, 2, zero ? 0.0f : 2.0f);
      }
    }
    for (const float factor : {0.5f, 1.0f, 0.75f, 0.5f}) {
      const size_t width = std::round(map.GetWidth() * factor);
      const size_t height = std::round(map.GetHeight() * factor);
      Mat<float> expected(width, height, 3);
      for (size_t d = 0; d < 3; ++d) {
        DownsampleImage(map.GetPtr() + d * map.GetWidth() * map.GetHeight(),
                        map.GetHeight(),
                        map.GetWidth(),
                        height,
                        width,
                        expected.GetPtr() + d * width * height);
      }
      for (size_t r = 0; r < height; ++r) {
        for (size_t c = 0; c < width; ++c) {
          Eigen::Vector3f n(expected.Get(r, c, 0),
                            expected.Get(r, c, 1),
                            expected.Get(r, c, 2));
          if (n.squaredNorm() > 0) n /= std::sqrt(n.squaredNorm());
          for (size_t d = 0; d < 3; ++d) expected.Set(r, c, d, n(d));
        }
      }
      map.Rescale(factor);
      EXPECT_EQ(map.GetWidth(), width);
      EXPECT_EQ(map.GetHeight(), height);
      EXPECT_EQ(map.GetDepth(), 3);
      EXPECT_EQ(map.GetData(), expected.GetData());
    }
    const auto path = CreateTestDir() / "rescaled.bin";
    map.Write(path);
    NormalMap loaded;
    loaded.Read(path);
    EXPECT_EQ(loaded.GetData(), map.GetData());
    loaded.Downsize(1, 1);
    EXPECT_EQ(loaded.GetWidth(), 1);
    EXPECT_EQ(loaded.GetHeight(), 1);
  }
  NormalMap empty;
  empty.Rescale(0.5f);
  EXPECT_TRUE(empty.GetData().empty());
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
