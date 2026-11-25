// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

namespace colmap {
namespace mvs {

template <>
void Mat<float>::Read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);

  char unused_char;
  file >> width_ >> unused_char >> height_ >> unused_char >> depth_ >>
      unused_char;
  THROW_CHECK_GT(width_, 0) << path;
  THROW_CHECK_GT(height_, 0) << path;
  THROW_CHECK_GT(depth_, 0) << path;
  data_.resize(width_ * height_ * depth_);

  ReadBinaryLittleEndian<float>(&file, &data_);
  file.close();
}

template <>
void Mat<float>::Write(const std::filesystem::path& path) const {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);
  file << width_ << "&" << height_ << "&" << depth_ << "&";
  THROW_CHECK(file) << path;
  if (IsLittleEndian()) {
    // Bound both the byte multiplication and the streamsize conversion.
    constexpr size_t kMaxChunkSize =
        static_cast<size_t>(std::min<std::uintmax_t>(
            std::numeric_limits<size_t>::max(),
            std::numeric_limits<std::streamsize>::max())) /
        sizeof(float);
    for (size_t offset = 0; offset < data_.size();) {
      const size_t chunk_size = std::min(data_.size() - offset, kMaxChunkSize);
      file.write(reinterpret_cast<const char*>(data_.data() + offset),
                 static_cast<std::streamsize>(chunk_size * sizeof(float)));
      THROW_CHECK(file) << path;
      offset += chunk_size;
    }
  } else {
    WriteBinaryLittleEndian<float>(&file, {data_.data(), data_.size()});
  }
  THROW_CHECK(file) << path;
  file.close();
  THROW_CHECK(file) << path;
}

}  // namespace mvs
}  // namespace colmap
