// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__linux__)
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#endif

namespace colmap {
namespace mvs {
namespace {

constexpr size_t kMaxHeaderBytes = 128;

struct MappedFloatShape {
  size_t width, height, depth, bytes, payload_start;
};

MappedFloatShape ReadMappedFloatShape(std::ifstream& file,
                                      const std::filesystem::path& path) {
  std::string header;
  header.reserve(kMaxHeaderBytes);
  size_t delimiters = 0;
  while (header.size() < kMaxHeaderBytes && delimiters < 3) {
    char value = 0;
    file.get(value);
    THROW_CHECK(file) << path;
    header.push_back(value);
    if (value == '&') ++delimiters;
  }
  THROW_CHECK_EQ(delimiters, 3) << path;
  std::istringstream stream(header);
  SetFullPrecTextStream(stream);
  size_t width = 0, height = 0, depth = 0;
  char sx = 0, sy = 0, sz = 0;
  stream >> width >> sx >> height >> sy >> depth >> sz;
  THROW_CHECK(stream && sx == '&' && sy == '&' && sz == '&') << path;
  THROW_CHECK_EQ(stream.peek(), std::char_traits<char>::eof()) << path;
  THROW_CHECK_GT(width, 0) << path;
  THROW_CHECK_GT(height, 0) << path;
  THROW_CHECK(depth == 1 || depth == 3)
      << "FP32 depth/normal map channels: " << path;
  THROW_CHECK_LE(width, std::numeric_limits<size_t>::max() / height) << path;
  THROW_CHECK_LE(width * height,
                 std::numeric_limits<size_t>::max() / (depth * sizeof(float)))
      << path;
  const size_t bytes = width * height * depth * sizeof(float);
  const auto start = file.tellg();
  THROW_CHECK_GE(start, std::streampos(0)) << path;
  const auto start_off = static_cast<std::streamoff>(start);
  THROW_CHECK_GE(start_off, 0) << path;
  THROW_CHECK_LE(bytes,
                 static_cast<size_t>(
                     std::numeric_limits<std::streamoff>::max() - start_off))
      << path;
  file.seekg(0, std::ios::end);
  const auto end = file.tellg();
  THROW_CHECK_GE(end, start) << path;
  THROW_CHECK_GE(end - start, static_cast<std::streamoff>(bytes)) << path;
  return {width, height, depth, bytes, static_cast<size_t>(start_off)};
}

}  // namespace

struct ImmutableMapSource::Impl {
  MappedFloatShape shape;
  size_t file_bytes = 0;
#if defined(__linux__)
  std::unique_ptr<boost::interprocess::file_mapping> file;
#endif
};

struct ImmutableMapRegion::Impl {
  std::shared_ptr<const ImmutableMapSource> source;
  MappedFloatShape shape;
  size_t mapped_bytes = 0;
  size_t row_begin = 0, row_count = 0;
  size_t region_count = 1;
#if defined(__linux__)
  std::unique_ptr<boost::interprocess::mapped_region> region;
  std::array<std::unique_ptr<boost::interprocess::mapped_region>, 3> planes;
#endif
  const char* payload = nullptr;
  std::array<const char*, 3> plane_payload{};
};

bool ImmutableMapRegion::Supported() {
#if defined(__linux__)
  return IsLittleEndian();
#else
  return false;
#endif
}

namespace {
size_t MappingPageCharge(const std::uintmax_t bytes) {
  THROW_CHECK(ImmutableMapRegion::Supported());
#if defined(__linux__)
  const size_t page = boost::interprocess::mapped_region::get_page_size();
  THROW_CHECK_GT(page, 0);
  THROW_CHECK_LE(bytes, std::numeric_limits<size_t>::max() - (page - 1));
  return ((static_cast<size_t>(bytes) + page - 1) / page) * page;
#else
  return 0;
#endif
}

struct WindowPlan {
  std::array<size_t, 3> offsets{}, lengths{};
  size_t charged_bytes = 0;
};

WindowPlan PlanWindow(const MappedFloatShape& shape,
                      const size_t row_begin,
                      const size_t row_count) {
  THROW_CHECK_EQ(shape.depth, 3) << "Normal windows require three channels";
  THROW_CHECK_GT(row_count, 0);
  THROW_CHECK_LT(row_begin, shape.height);
  THROW_CHECK_LE(row_count, shape.height - row_begin);
#if defined(__linux__)
  const size_t page = boost::interprocess::mapped_region::get_page_size();
  THROW_CHECK_GT(page, 0);
  THROW_CHECK_LE(shape.width,
                 std::numeric_limits<size_t>::max() / sizeof(float));
  const size_t row_bytes = shape.width * sizeof(float);
  THROW_CHECK_LE(shape.height, std::numeric_limits<size_t>::max() / row_bytes);
  const size_t plane_bytes = shape.height * row_bytes;
  THROW_CHECK_LE(row_count, std::numeric_limits<size_t>::max() / row_bytes);
  const size_t payload_bytes = row_count * row_bytes;
  WindowPlan plan;
  for (size_t channel = 0; channel < 3; ++channel) {
    THROW_CHECK_LE(channel * plane_bytes,
                   std::numeric_limits<size_t>::max() - shape.payload_start);
    const size_t base = shape.payload_start + channel * plane_bytes;
    THROW_CHECK_LE(row_begin * row_bytes,
                   std::numeric_limits<size_t>::max() - base);
    const size_t begin = base + row_begin * row_bytes;
    const size_t aligned = begin / page * page;
    const size_t prefix = begin - aligned;
    THROW_CHECK_LE(payload_bytes, std::numeric_limits<size_t>::max() - prefix);
    const size_t length = prefix + payload_bytes;
    THROW_CHECK_LE(
        aligned,
        static_cast<size_t>(
            std::numeric_limits<boost::interprocess::offset_t>::max()));
    THROW_CHECK_LE(
        length,
        static_cast<size_t>(
            std::numeric_limits<boost::interprocess::offset_t>::max()) -
            aligned);
    const size_t charge = MappingPageCharge(length);
    THROW_CHECK_LE(charge,
                   std::numeric_limits<size_t>::max() - plan.charged_bytes);
    plan.offsets[channel] = aligned;
    plan.lengths[channel] = length;
    plan.charged_bytes += charge;
  }
  return plan;
#else
  return {};
#endif
}
}  // namespace

size_t ImmutableMapRegion::GetMappingReservationBytes(
    const std::filesystem::path& path) {
  return MappingPageCharge(std::filesystem::file_size(path));
}

size_t ImmutableMapRegion::GetWindowReservationBytes(
    const std::filesystem::path& path,
    const size_t row_begin,
    const size_t row_count) {
  THROW_CHECK(Supported());
  std::ifstream input(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(input, path);
  return PlanWindow(ReadMappedFloatShape(input, path), row_begin, row_count)
      .charged_bytes;
}

ImmutableMapSource::ImmutableMapSource(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>()) {
  THROW_CHECK(ImmutableMapRegion::Supported());
  std::ifstream input(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(input, path);
  impl_->shape = ReadMappedFloatShape(input, path);
  input.close();
  THROW_CHECK(input);
#if defined(__linux__)
  namespace bip = boost::interprocess;
  impl_->file =
      std::make_unique<bip::file_mapping>(path.c_str(), bip::read_only);
  bip::offset_t file_bytes = 0;
  THROW_CHECK(impl_->file->get_size(file_bytes));
  THROW_CHECK_GE(file_bytes, 0);
  const size_t start = impl_->shape.payload_start;
  THROW_CHECK_LE(impl_->shape.bytes,
                 std::numeric_limits<size_t>::max() - start);
  THROW_CHECK_GE(static_cast<size_t>(file_bytes), start + impl_->shape.bytes);
  impl_->file_bytes = static_cast<size_t>(file_bytes);
#endif
}

std::array<size_t, 3> ImmutableMapSource::GetShape() const {
  return {impl_->shape.width, impl_->shape.height, impl_->shape.depth};
}
ImmutableMapSource::~ImmutableMapSource() = default;
size_t ImmutableMapSource::GetMappingReservationBytes() const {
  return MappingPageCharge(impl_->file_bytes);
}
size_t ImmutableMapSource::GetWindowReservationBytes(
    const size_t row_begin, const size_t row_count) const {
  return PlanWindow(impl_->shape, row_begin, row_count).charged_bytes;
}

ImmutableMapRegion::ImmutableMapRegion(const std::filesystem::path& path)
    : ImmutableMapRegion(std::make_shared<ImmutableMapSource>(path)) {}

ImmutableMapRegion::ImmutableMapRegion(
    std::shared_ptr<const ImmutableMapSource> source)
    : impl_(std::make_unique<Impl>()) {
  THROW_CHECK(source != nullptr);
  impl_->source = std::move(source);
  impl_->shape = impl_->source->impl_->shape;
  impl_->row_count = impl_->shape.height;
  const size_t start = impl_->shape.payload_start;
  THROW_CHECK_LE(impl_->shape.bytes,
                 std::numeric_limits<size_t>::max() - start);
  const size_t mapped_bytes = start + impl_->shape.bytes;
  impl_->mapped_bytes = MappingPageCharge(mapped_bytes);
#if defined(__linux__)
  namespace bip = boost::interprocess;
  THROW_CHECK_LE(
      mapped_bytes,
      static_cast<size_t>(std::numeric_limits<bip::offset_t>::max()));
  THROW_CHECK_GE(impl_->source->impl_->file_bytes, mapped_bytes);
  impl_->region = std::make_unique<bip::mapped_region>(
      *impl_->source->impl_->file, bip::read_only, 0, mapped_bytes);
  impl_->payload =
      static_cast<const char*>(impl_->region->get_address()) + start;
#endif
}

ImmutableMapRegion::ImmutableMapRegion(const std::filesystem::path& path,
                                       const size_t row_begin,
                                       const size_t row_count)
    : ImmutableMapRegion(
          std::make_shared<ImmutableMapSource>(path), row_begin, row_count) {}

ImmutableMapRegion::ImmutableMapRegion(
    std::shared_ptr<const ImmutableMapSource> source,
    const size_t row_begin,
    const size_t row_count)
    : impl_(std::make_unique<Impl>()) {
  THROW_CHECK(source != nullptr);
  impl_->source = std::move(source);
  impl_->shape = impl_->source->impl_->shape;
  const WindowPlan plan = PlanWindow(impl_->shape, row_begin, row_count);
  impl_->mapped_bytes = plan.charged_bytes;
  impl_->row_begin = row_begin;
  impl_->row_count = row_count;
  impl_->region_count = 3;
#if defined(__linux__)
  namespace bip = boost::interprocess;
  for (size_t channel = 0; channel < 3; ++channel) {
    THROW_CHECK_GE(impl_->source->impl_->file_bytes,
                   plan.offsets[channel] + plan.lengths[channel]);
    impl_->planes[channel] = std::make_unique<bip::mapped_region>(
        *impl_->source->impl_->file,
        bip::read_only,
        static_cast<bip::offset_t>(plan.offsets[channel]),
        plan.lengths[channel]);
    const size_t row_bytes = impl_->shape.width * sizeof(float);
    const size_t begin =
        impl_->shape.payload_start +
        (channel * impl_->shape.height + row_begin) * row_bytes;
    impl_->plane_payload[channel] =
        static_cast<const char*>(impl_->planes[channel]->get_address()) +
        (begin - plan.offsets[channel]);
  }
#endif
}

ImmutableMapRegion::~ImmutableMapRegion() = default;
size_t ImmutableMapRegion::GetMappingBytes() const {
  return impl_->mapped_bytes;
}
size_t ImmutableMapRegion::GetMappingRegionCount() const {
  return impl_->region_count;
}

float ImmutableMapRegion::ReadFloat1Sample(const size_t row,
                                           const size_t col,
                                           const size_t expected_width,
                                           const size_t expected_height) const {
  const auto& owner = *impl_;
  const auto& shape = owner.shape;
  THROW_CHECK_EQ(owner.region_count, size_t{1})
      << "Exact depth access requires the complete map owner";
  THROW_CHECK_EQ(shape.depth, size_t{1})
      << "Exact depth access requires one channel";
  THROW_CHECK_EQ(shape.width, expected_width)
      << "Original depth width differs from consumed fusion grid";
  THROW_CHECK_EQ(shape.height, expected_height)
      << "Original depth height differs from consumed fusion grid";
  THROW_CHECK_LT(col, shape.width);
  THROW_CHECK_LT(row, shape.height);
  THROW_CHECK(owner.payload != nullptr);
  const size_t pixel_index = row * shape.width + col;
  float sample;
  std::memcpy(
      &sample, owner.payload + pixel_index * sizeof(float), sizeof(float));
  return sample;
}

Eigen::Vector3f ImmutableMapRegion::ReadFloat3(const size_t row,
                                               const size_t col) const {
  const auto& shape = impl_->shape;
  THROW_CHECK_EQ(shape.depth, 3);
  THROW_CHECK_LT(row, shape.height);
  THROW_CHECK_GE(row, impl_->row_begin);
  THROW_CHECK_LT(row - impl_->row_begin, impl_->row_count);
  THROW_CHECK_LT(col, shape.width);
  Eigen::Vector3f normal;
  for (size_t channel = 0; channel < 3; ++channel) {
    const char* source = nullptr;
    if (impl_->region_count == 3) {
      const size_t index = (row - impl_->row_begin) * shape.width + col;
      source = impl_->plane_payload[channel] + index * sizeof(float);
    } else {
      const size_t index = (channel * shape.height + row) * shape.width + col;
      source = impl_->payload + index * sizeof(float);
    }
    std::memcpy(
        &normal[static_cast<Eigen::Index>(channel)], source, sizeof(float));
  }
  return normal;
}

template <typename T>
void Mat<T>::ReadRegion(const ImmutableMapRegion& source,
                        const size_t x,
                        const size_t y,
                        const size_t width,
                        const size_t height) {
  static_assert(std::is_same_v<T, float>);
  THROW_CHECK_EQ(source.impl_->region_count, 1)
      << "Decoded region reads require the complete map owner";
  const auto& shape = source.impl_->shape;
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  THROW_CHECK_LT(x, shape.width);
  THROW_CHECK_LT(y, shape.height);
  THROW_CHECK_LE(width, shape.width - x);
  THROW_CHECK_LE(height, shape.height - y);
  THROW_CHECK_LE(width, std::numeric_limits<size_t>::max() / height);
  THROW_CHECK_LE(width * height,
                 std::numeric_limits<size_t>::max() / shape.depth);
  std::vector<T> values(width * height * shape.depth);
  for (size_t plane = 0; plane < shape.depth; ++plane) {
    for (size_t row = 0; row < height; ++row) {
      const size_t source_index =
          (plane * shape.height + y + row) * shape.width + x;
      T* destination = values.data() + (plane * height + row) * width;
      std::memcpy(destination,
                  source.impl_->payload + source_index * sizeof(float),
                  width * sizeof(float));
    }
  }
  data_.swap(values);
  width_ = width;
  height_ = height;
  depth_ = shape.depth;
}

template void Mat<float>::ReadRegion(
    const ImmutableMapRegion&, size_t, size_t, size_t, size_t);

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
