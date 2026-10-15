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

TEST_F(MatFileTest, MappedNormalRegionKeepsPlanarBitsAtOddEdge) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  const std::vector<uint32_t> bits = {0x3f800001,
                                      0x3f800002,
                                      0x3f800003,
                                      0x3f800004,
                                      0x3f800005,
                                      0x3f800006,
                                      0x40000001,
                                      0x40000002,
                                      0x40000003,
                                      0x40000004,
                                      0x40000005,
                                      0x40000006,
                                      0x40400001,
                                      0x40400002,
                                      0x40400003,
                                      0x40400004,
                                      0x40400005,
                                      0x40400006};
  WriteMatFile(path_, "3&2&3&" + LittleEndianFloats(bits) + "trailing");
  ImmutableMapRegion source(path_);
  EXPECT_GE(source.GetMappingBytes(), std::filesystem::file_size(path_));
  for (size_t row = 0; row < 2; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      const auto sample = source.ReadFloat3(row, col);
      for (size_t channel = 0; channel < 3; ++channel) {
        EXPECT_EQ(FloatBits(sample[static_cast<Eigen::Index>(channel)]),
                  bits[channel * 6 + row * 3 + col]);
      }
    }
  }
  EXPECT_THROW(source.ReadFloat3(2, 0), std::exception);
  EXPECT_THROW(source.ReadFloat3(0, 3), std::exception);
  Mat<float> tile;
  tile.ReadRegion(source, 1, 0, 2, 2);
  EXPECT_EQ(tile.GetWidth(), 2);
  EXPECT_EQ(tile.GetHeight(), 2);
  EXPECT_EQ(tile.GetDepth(), 3);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 2; ++col) {
        EXPECT_EQ(FloatBits(tile.Get(row, col, channel)),
                  bits[channel * 6 + row * 3 + col + 1]);
      }
    }
  }
}

TEST_F(MatFileTest, FailedMappedNormalRegionPreservesReceiver) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  WriteMatFile(path_, "1&1&3&" + LittleEndianFloats({1, 2, 3}));
  Mat<float> receiver(1, 1, 1);
  receiver.Set(0, 0, 0, -0.0f);
  {
    ImmutableMapRegion source(path_);
    EXPECT_THROW(receiver.ReadRegion(source, 1, 0, 1, 1), std::exception);
  }
  EXPECT_EQ(receiver.GetWidth(), 1);
  EXPECT_EQ(receiver.GetDepth(), 1);
  EXPECT_EQ(FloatBits(receiver.Get(0, 0)), 0x80000000);
  WriteMatFile(path_, "1&1&3&" + LittleEndianFloats({1, 2}));
  EXPECT_THROW(ImmutableMapRegion truncated(path_), std::exception);
  EXPECT_EQ(FloatBits(receiver.Get(0, 0)), 0x80000000);
  WriteMatFile(path_, std::string(128, '9') + "&1&3&");
  EXPECT_THROW(ImmutableMapRegion invalid(path_), std::exception);
  EXPECT_EQ(FloatBits(receiver.Get(0, 0)), 0x80000000);
}

TEST_F(MatFileTest, MappedNormalStripeCrossesPlanarPageAndRowEdges) {
  if (!ImmutableMapRegion::Supported()) GTEST_SKIP();
  constexpr size_t width = 7, height = 260;
  std::vector<uint32_t> bits(3 * width * height);
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        bits[(channel * height + row) * width + col] =
            0x3f800000u +
            static_cast<uint32_t>(channel * width * height + row * width + col);
      }
    }
  }
  WriteMatFile(path_, "7&260&3&" + LittleEndianFloats(bits));
  const size_t first_charge =
      ImmutableMapRegion::GetWindowReservationBytes(path_, 0, 256);
  const size_t edge_charge =
      ImmutableMapRegion::GetWindowReservationBytes(path_, 256, 4);
  {
    auto shared_source = std::make_shared<ImmutableMapSource>(path_);
    EXPECT_EQ(shared_source->GetWindowReservationBytes(0, 256), first_charge);
    EXPECT_EQ(shared_source->GetWindowReservationBytes(256, 4), edge_charge);
    ImmutableMapRegion first(shared_source, 0, 256);
    ImmutableMapRegion edge(shared_source, 256, 4);
    shared_source.reset();  // Both owners keep the one validated file source.
    EXPECT_EQ(first.GetMappingBytes(), first_charge);
    EXPECT_EQ(edge.GetMappingBytes(), edge_charge);
    EXPECT_EQ(first.GetMappingRegionCount(), 3);
    EXPECT_EQ(edge.GetMappingRegionCount(), 3);
    const std::array<std::pair<const ImmutableMapRegion*, size_t>, 4> samples =
        {{{&first, 0}, {&first, 255}, {&edge, 256}, {&edge, 259}}};
    for (const auto& [source, row] : samples) {
      for (size_t col : {size_t{0}, width - 1}) {
        const auto normal = source->ReadFloat3(row, col);
        for (size_t channel = 0; channel < 3; ++channel) {
          EXPECT_EQ(FloatBits(normal[static_cast<Eigen::Index>(channel)]),
                    bits[(channel * height + row) * width + col]);
        }
      }
    }
    EXPECT_THROW(first.ReadFloat3(256, 0), std::exception);
    EXPECT_THROW(edge.ReadFloat3(255, 0), std::exception);
    auto retry_source = std::make_shared<ImmutableMapSource>(path_);
    EXPECT_THROW(ImmutableMapRegion(retry_source, 256, 5), std::exception);
    EXPECT_EQ(
        FloatBits(
            ImmutableMapRegion(retry_source, 256, 4).ReadFloat3(256, 0).x()),
        bits[256 * width]);
    EXPECT_THROW(ImmutableMapRegion(path_, 256, 5), std::exception);
    EXPECT_THROW(ImmutableMapRegion::GetWindowReservationBytes(path_, 260, 1),
                 std::exception);
    Mat<float> decoded;
    EXPECT_THROW(decoded.ReadRegion(edge, 0, 256, 1, 1), std::exception);
  }
  // Fresh owners reject truncation and an unbounded header before mapping.
  WriteMatFile(path_,
               "7&260&3&" + LittleEndianFloats(std::vector<uint32_t>(
                                bits.begin(), bits.end() - 1)));
  EXPECT_THROW(ImmutableMapRegion(path_, 0, 256), std::exception);
  WriteMatFile(path_, std::string(129, '9') + "&260&3&");
  EXPECT_THROW(ImmutableMapRegion(path_, 0, 1), std::exception);
}

TEST(Mat, SignedNormalCodecPreservesValidComponentPolicy) {
  for (const float value : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}) {
    const auto encoded = EncodeNormalSnorm16Unchecked(value);
    const float decoded = DecodeNormalSnorm16(encoded);
    EXPECT_LE(std::abs(decoded - value), 1.0f / 32767.0f);
    EXPECT_GE(decoded, -1.0f);
    EXPECT_LE(decoded, 1.0f);
  }
  EXPECT_EQ(DecodeNormalSnorm16(EncodeNormalSnorm16Unchecked(1.0f)), 1.0f);
  EXPECT_EQ(DecodeNormalSnorm16(EncodeNormalSnorm16Unchecked(-1.0f)), -1.0f);
}

TEST(Mat, OctNormalDirectionAndMagnitudeContracts) {
  OctNormalPair encoded{};
  EncodeNormalOct2x16Unchecked(Eigen::Vector3f::Zero(), &encoded);
  EXPECT_EQ(encoded.x, std::numeric_limits<int16_t>::min());
  EXPECT_EQ(encoded.y, std::numeric_limits<int16_t>::min());
  EXPECT_TRUE(DecodeNormalOct2x16(encoded.x, encoded.y).isZero());
  const Eigen::Vector3f folded(-0.3f, 0.2f, -0.93f);
  EncodeNormalOct2x16Unchecked(folded, &encoded);
  EXPECT_GT(DecodeNormalOct2x16(encoded.x, encoded.y).dot(folded.normalized()),
            0.99999f);
  const Eigen::Vector3f nonunit(0.99f, 0.0f, 0.0f);
  EXPECT_LT(std::abs(DecodeNormalMagnitudeUnorm16(
                         EncodeNormalMagnitudeUnorm16Unchecked(nonunit)) -
                     0.99f),
            1.4e-5f);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
