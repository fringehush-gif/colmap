// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/util/endian.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>
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

TEST(Mat, ReadWrite) {
  const auto path = CreateTestDir() / "depth.bin";
  for (size_t depth = 1; depth <= 4; ++depth) {
    SCOPED_TRACE(depth);
    Mat<float> original(2, 3, depth);
    for (size_t i = 0; i < original.GetData().size(); ++i) {
      original.GetPtr()[i] = static_cast<float>(i) / 2 - 3;
    }
    original.Write(path);
    Mat<float> loaded;
    loaded.Read(path);
    EXPECT_EQ(loaded.GetWidth(), original.GetWidth());
    EXPECT_EQ(loaded.GetHeight(), original.GetHeight());
    EXPECT_EQ(loaded.GetDepth(), original.GetDepth());
    EXPECT_EQ(loaded.GetData(), original.GetData());
  }
}

TEST(Mat, ReadRejectsTruncatedPayload) {
  const auto path = CreateTestDir() / "depth.bin";
  Mat<float> original(2, 3, 2);
  original.Fill(1.5f);
  Mat<float> loaded(3, 1, 1);
  loaded.Fill(-2.5f);
  const Mat<float> previous = loaded;
  for (size_t num_bytes = 0; num_bytes < original.GetNumBytes(); ++num_bytes) {
    SCOPED_TRACE(num_bytes);
    original.Write(path);
    const auto header_size =
        std::filesystem::file_size(path) - original.GetNumBytes();
    std::filesystem::resize_file(path, header_size + num_bytes);
    EXPECT_THROW(loaded.Read(path), std::invalid_argument);
    EXPECT_EQ(loaded.GetWidth(), previous.GetWidth());
    EXPECT_EQ(loaded.GetHeight(), previous.GetHeight());
    EXPECT_EQ(loaded.GetDepth(), previous.GetDepth());
    EXPECT_EQ(loaded.GetData(), previous.GetData());
  }
}

TEST(Mat, ReadRejectsInvalidHeader) {
  const auto path = CreateTestDir() / "depth.bin";
  const std::string max_dimension =
      std::to_string(std::numeric_limits<size_t>::max());
  const std::vector<std::string> headers = {
      "1x1&1&",
      "1&1x1&",
      "1&1&1x",
      "0&1&1&",
      "1&0&1&",
      "1&1&0&",
      "-1&1&1&",
      "1&-1&1&",
      "1&1&-1&",
      "1&",
      max_dimension + "0&1&1&",
      max_dimension + "&" + max_dimension + "&1&",
      "-" + max_dimension + "&1&1&",
      "1&-" + max_dimension + "&1&",
      "1&1&-" + max_dimension + "&"};
  Mat<float> loaded(3, 1, 1);
  loaded.Fill(-2.5f);
  const Mat<float> previous = loaded;
  for (const auto& header : headers) {
    SCOPED_TRACE(header);
    std::ofstream file(path, std::ios::binary);
    file << header;
    WriteBinaryLittleEndian<float>(&file, 1.5f);
    file.close();
    EXPECT_THROW(loaded.Read(path), std::invalid_argument);
    EXPECT_EQ(loaded.GetWidth(), previous.GetWidth());
    EXPECT_EQ(loaded.GetHeight(), previous.GetHeight());
    EXPECT_EQ(loaded.GetDepth(), previous.GetDepth());
    EXPECT_EQ(loaded.GetData(), previous.GetData());
    loaded = previous;
  }
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

#ifdef __linux__
TEST(Mat, WriteReportsIncompleteOutput) {
  if (!std::filesystem::exists("/dev/full")) {
    GTEST_SKIP() << "Linux failure sink unavailable";
  }
  Mat<float> mat(2, 3, 1);
  mat.Fill(4.0f);
  EXPECT_THROW(mat.Write("/dev/full"), std::exception);
  Mat<float> larger(8192, 1, 1);
  EXPECT_THROW(larger.Write("/dev/full"), std::exception);
}

TEST(Mat, WritePreservesExistingHardLink) {
  const auto root = CreateTestDir();
  const auto path = root / "map.bin";
  const auto alias = root / "alias.bin";
  Mat<float> mat(2, 3, 1);
  mat.Fill(4.0f);
  mat.Write(path);
  std::filesystem::create_hard_link(path, alias);
  mat.Fill(7.0f);
  mat.Write(path);
  EXPECT_TRUE(std::filesystem::equivalent(path, alias));
  Mat<float> readback;
  readback.Read(alias);
  EXPECT_EQ(readback.Get(0, 0, 0), 7.0f);
}
#endif

}  // namespace
}  // namespace mvs
}  // namespace colmap
