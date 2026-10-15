// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

std::filesystem::path MatFilePath() {
  static size_t next_id = 0;
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("colmap_mat_test_" + std::to_string(stamp) + "_" +
          std::to_string(next_id++) + ".bin");
}

void WriteMatFile(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file.write(bytes.data(), bytes.size());
  ASSERT_TRUE(file.good());
}

std::string LittleEndianFloats(const std::vector<uint32_t>& bits) {
  std::string bytes;
  for (const uint32_t value : bits) {
    for (size_t i = 0; i < sizeof(value); ++i) {
      bytes.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
    }
  }
  return bytes;
}

uint32_t FloatBits(const float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

class MatFileTest : public ::testing::Test {
 protected:
  void TearDown() override {
    if (!path_.empty()) {
      std::error_code error;
      std::filesystem::remove(path_, error);
    }
  }

  std::filesystem::path path_ = MatFilePath();
};

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

TEST_F(MatFileTest, ReadPreservesFloatBits) {
  const std::vector<uint32_t> expected = {
      0x3f800000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345};
  WriteMatFile(path_, "5&1&1&" + LittleEndianFloats(expected));

  Mat<float> mat;
  mat.Read(path_);

  ASSERT_EQ(mat.GetWidth(), expected.size());
  ASSERT_EQ(mat.GetHeight(), 1);
  ASSERT_EQ(mat.GetDepth(), 1);
  ASSERT_EQ(mat.GetData().size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(FloatBits(mat.GetData()[i]), expected[i]);
  }
}

TEST_F(MatFileTest, ReadAcceptsTrailingBytes) {
  const uint32_t expected = 0x3f800000;
  WriteMatFile(path_, "1&1&1&" + LittleEndianFloats({expected}) + "trailing");

  Mat<float> mat;
  mat.Read(path_);

  ASSERT_EQ(mat.GetData().size(), 1);
  EXPECT_EQ(FloatBits(mat.GetData()[0]), expected);
}

TEST_F(MatFileTest, FailedReadPreservesExistingMat) {
  Mat<float> mat(2, 1, 1);
  mat.Set(0, 0, 0, -0.0f);
  mat.Set(0, 1, 0, 42.0f);
  const std::vector<uint32_t> original_bits = {FloatBits(mat.Get(0, 0)),
                                               FloatBits(mat.Get(0, 1))};

  WriteMatFile(path_, "2&1&1&" + LittleEndianFloats({0x3f800000}));
  EXPECT_THROW(mat.Read(path_), std::exception);

  EXPECT_EQ(mat.GetWidth(), 2);
  EXPECT_EQ(mat.GetHeight(), 1);
  EXPECT_EQ(mat.GetDepth(), 1);
  ASSERT_EQ(mat.GetData().size(), original_bits.size());
  for (size_t i = 0; i < original_bits.size(); ++i) {
    EXPECT_EQ(FloatBits(mat.GetData()[i]), original_bits[i]);
  }
}

TEST_F(MatFileTest, ReadRejectsInvalidHeadersAndOverlongPrefix) {
  const std::vector<std::string> invalid_headers = {
      "1x1&1&",
      "1&1x1&",
      "1&1&1x",
      "184467440737095516160&1&1&",
      std::string(128, '9') + "&1&1&"};
  for (const std::string& header : invalid_headers) {
    WriteMatFile(path_, header);
    Mat<float> mat;
    EXPECT_THROW(mat.Read(path_), std::exception) << header.substr(0, 32);
  }
}

TEST_F(MatFileTest, ReadRejectsDimensionOverflowBeforePayloadAllocation) {
  WriteMatFile(path_, "18446744073709551615&2&1&");
  Mat<float> mat;
  EXPECT_THROW(mat.Read(path_), std::exception);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
