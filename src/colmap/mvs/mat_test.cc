// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/util/testing.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>

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

TEST(Mat, ReadWriteFloatBits) {
  const auto path = CreateTestDir() / "mat.bin";
  const auto output_path = CreateTestDir() / "output.bin";
  const std::array<uint32_t, 12> patterns = {0x00000000,
                                             0x80000000,
                                             0x00000001,
                                             0x007fffff,
                                             0x00800000,
                                             0x3f800000,
                                             0xbf800000,
                                             0x7f7fffff,
                                             0x7f800000,
                                             0xff800000,
                                             0x7fc12345,
                                             0x7f812345};
  Mat<float> mat(4, 2, 3);
  mat.Fill(-2.5f);
  for (const size_t depth : {1, 3, 4, 1}) {
    SCOPED_TRACE(depth);
    const size_t width = 2;
    const size_t height = 3;
    std::string bytes = "2&3&" + std::to_string(depth) + "&";
    for (size_t i = 0; i < width * height * depth; ++i) {
      const uint32_t bits = patterns[i % patterns.size()];
      for (size_t j = 0; j < sizeof(float); ++j) {
        bytes.push_back(static_cast<char>((bits >> (8 * j)) & 0xff));
      }
    }
    std::ofstream(path, std::ios::binary).write(bytes.data(), bytes.size());

    mat.Read(path);
    EXPECT_EQ(mat.GetWidth(), width);
    EXPECT_EQ(mat.GetHeight(), height);
    EXPECT_EQ(mat.GetDepth(), depth);
    ASSERT_EQ(mat.GetNumBytes(), width * height * depth * sizeof(float));
    for (size_t slice = 0; slice < depth; ++slice) {
      for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
          const size_t i = slice * width * height + row * width + col;
          const float value = mat.Get(row, col, slice);
          uint32_t bits;
          std::memcpy(&bits, &value, sizeof(bits));
          EXPECT_EQ(bits, patterns[i % patterns.size()]);
        }
      }
    }

    mat.Write(output_path);
    std::ifstream output(output_path, std::ios::binary);
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(output), {}), bytes);
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
