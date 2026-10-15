// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {
namespace {
constexpr size_t kMaxHeaderBytes = 128;
}  // namespace

template <>
void Mat<float>::Read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);

  std::string header;
  header.reserve(kMaxHeaderBytes);
  size_t num_delimiters = 0;
  while (header.size() < kMaxHeaderBytes && num_delimiters < 3) {
    char c = 0;
    file.get(c);
    THROW_CHECK(file) << path;
    header.push_back(c);
    if (c == '&') {
      ++num_delimiters;
    }
  }
  THROW_CHECK_EQ(num_delimiters, 3) << path;

  std::istringstream header_stream(header);
  size_t width = 0, height = 0, depth = 0;
  char sx = 0, sy = 0, sz = 0;
  header_stream >> width >> sx >> height >> sy >> depth >> sz;
  THROW_CHECK(header_stream && sx == '&' && sy == '&' && sz == '&') << path;
  THROW_CHECK_EQ(header_stream.peek(), std::char_traits<char>::eof()) << path;
  THROW_CHECK_GT(width, 0) << path;
  THROW_CHECK_GT(height, 0) << path;
  THROW_CHECK_GT(depth, 0) << path;
  THROW_CHECK_LE(width, std::numeric_limits<size_t>::max() / height) << path;
  const size_t pixels = width * height;
  THROW_CHECK_LE(pixels, std::numeric_limits<size_t>::max() / depth) << path;
  const size_t samples = pixels * depth;
  THROW_CHECK_LE(
      samples,
      static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
          sizeof(float))
      << path;
  const auto bytes = static_cast<std::streamsize>(samples * sizeof(float));
  const auto begin = file.tellg();
  THROW_CHECK_GE(begin, std::streampos(0)) << path;
  file.seekg(0, std::ios::end);
  const auto end = file.tellg();
  THROW_CHECK_GE(end, begin) << path;
  THROW_CHECK_GE(end - begin, bytes) << path;
  file.seekg(begin);
  THROW_CHECK(file) << path;

  std::vector<float> data(samples);
  if (IsLittleEndian()) {
    file.read(reinterpret_cast<char*>(data.data()), bytes);
    THROW_CHECK_EQ(file.gcount(), bytes) << path;
  } else {
    ReadBinaryLittleEndian<float>(&file, &data);
  }
  THROW_CHECK(file) << path;

  width_ = width;
  height_ = height;
  depth_ = depth;
  data_.swap(data);
}

template <>
void Mat<float>::Write(const std::filesystem::path& path) const {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);
  file << width_ << "&" << height_ << "&" << depth_ << "&";
  WriteBinaryLittleEndian<float>(&file, {data_.data(), data_.size()});
  file.close();
}

}  // namespace mvs
}  // namespace colmap
