// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/depth_map.h"

#include "colmap/util/testing.h"

#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(DepthMap, Empty) {
  DepthMap depth_map;
  EXPECT_EQ(depth_map.GetWidth(), 0);
  EXPECT_EQ(depth_map.GetHeight(), 0);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), -1);
  EXPECT_EQ(depth_map.GetDepthMax(), -1);
}

TEST(DepthMap, NonEmpty) {
  DepthMap depth_map(1, 2, 0, 1);
  EXPECT_EQ(depth_map.GetWidth(), 1);
  EXPECT_EQ(depth_map.GetHeight(), 2);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0);
  EXPECT_EQ(depth_map.GetDepthMax(), 1);
}

TEST(DepthMap, ReadWrite) {
  const auto path = CreateTestDir() / "depth.bin";
  DepthMap expected(2, 3, 1, 8);
  for (size_t i = 0; i < expected.GetData().size(); ++i) {
    expected.GetPtr()[i] = static_cast<float>(i + 1);
  }
  expected.Write(path);

  DepthMap actual(1, 1, 0.25f, 100.0f);
  actual.Fill(-1.0f);
  ASSERT_NO_THROW(actual.Read(path));
  EXPECT_EQ(actual.GetWidth(), expected.GetWidth());
  EXPECT_EQ(actual.GetHeight(), expected.GetHeight());
  EXPECT_EQ(actual.GetDepth(), 1);
  EXPECT_EQ(actual.GetData(), expected.GetData());
  EXPECT_EQ(actual.GetDepthMin(), 0.25f);
  EXPECT_EQ(actual.GetDepthMax(), 100.0f);
}

TEST(DepthMap, ReadRejectsWrongChannels) {
  const auto path = CreateTestDir() / "depth.bin";
  for (const size_t depth : {2, 3, 4}) {
    SCOPED_TRACE(depth);
    Mat<float> input(4, 6, depth);
    input.Fill(0.5f);
    input.Write(path);

    Mat<float> generic;
    ASSERT_NO_THROW(generic.Read(path));
    EXPECT_EQ(generic.GetDepth(), depth);
    EXPECT_EQ(generic.GetData(), input.GetData());

    DepthMap actual(2, 3, 0.25f, 100.0f);
    actual.Fill(7.0f);
    const DepthMap previous = actual;
    EXPECT_THAT([&] { actual.Read(path); },
                testing::ThrowsMessage<std::invalid_argument>(
                    testing::HasSubstr("mat.GetDepth() == 1")));
    EXPECT_EQ(actual.GetWidth(), previous.GetWidth());
    EXPECT_EQ(actual.GetHeight(), previous.GetHeight());
    EXPECT_EQ(actual.GetDepth(), previous.GetDepth());
    EXPECT_EQ(actual.GetData(), previous.GetData());
    EXPECT_EQ(actual.GetDepthMin(), previous.GetDepthMin());
    EXPECT_EQ(actual.GetDepthMax(), previous.GetDepthMax());

    DepthMap replacement(4, 6, 1, 3);
    replacement.Fill(2.0f);
    replacement.Write(path);
    ASSERT_NO_THROW(actual.Read(path));
    EXPECT_EQ(actual.GetWidth(), replacement.GetWidth());
    EXPECT_EQ(actual.GetHeight(), replacement.GetHeight());
    EXPECT_EQ(actual.GetDepth(), 1);
    EXPECT_EQ(actual.GetData(), replacement.GetData());
    EXPECT_EQ(actual.GetDepthMin(), previous.GetDepthMin());
    EXPECT_EQ(actual.GetDepthMax(), previous.GetDepthMax());
  }
}

TEST(DepthMap, ReadFailurePreservesState) {
  DepthMap actual(2, 3, 0.25f, 100.0f);
  actual.Fill(7.0f);
  const DepthMap previous = actual;
  EXPECT_THROW(actual.Read(CreateTestDir() / "missing.bin"),
               std::invalid_argument);
  EXPECT_EQ(actual.GetWidth(), previous.GetWidth());
  EXPECT_EQ(actual.GetHeight(), previous.GetHeight());
  EXPECT_EQ(actual.GetDepth(), previous.GetDepth());
  EXPECT_EQ(actual.GetData(), previous.GetData());
  EXPECT_EQ(actual.GetDepthMin(), previous.GetDepthMin());
  EXPECT_EQ(actual.GetDepthMax(), previous.GetDepthMax());
}

TEST(DepthMap, Rescale) {
  DepthMap depth_map(6, 7, 0, 1);
  depth_map.Rescale(0.5);
  EXPECT_EQ(depth_map.GetWidth(), 3);
  EXPECT_EQ(depth_map.GetHeight(), 4);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0);
  EXPECT_EQ(depth_map.GetDepthMax(), 1);
}

TEST(DepthMap, Downsize) {
  DepthMap depth_map(6, 7, 0, 1);
  depth_map.Downsize(2, 4);
  EXPECT_EQ(depth_map.GetWidth(), 2);
  EXPECT_EQ(depth_map.GetHeight(), 2);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0);
  EXPECT_EQ(depth_map.GetDepthMax(), 1);
}

TEST(DepthMap, ToBitmap) {
  DepthMap depth_map(2, 2, 0.1, 0.9);
  depth_map.Fill(0.9);
  depth_map.Set(0, 0, 0, 0.1);
  depth_map.Set(0, 1, 0, 0.5);
  const Bitmap bitmap = depth_map.ToBitmap(0, 100);
  EXPECT_EQ(bitmap.Width(), depth_map.GetWidth());
  EXPECT_EQ(bitmap.Height(), depth_map.GetHeight());
  EXPECT_TRUE(bitmap.IsRGB());
  EXPECT_EQ(bitmap.GetPixel(0, 0).value(), BitmapColor<uint8_t>(0, 0, 128));
  EXPECT_EQ(bitmap.GetPixel(0, 1).value(), BitmapColor<uint8_t>(128, 0, 0));
  EXPECT_EQ(bitmap.GetPixel(1, 0).value(), BitmapColor<uint8_t>(128, 255, 127));
  EXPECT_EQ(bitmap.GetPixel(1, 1).value(), BitmapColor<uint8_t>(128, 0, 0));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
