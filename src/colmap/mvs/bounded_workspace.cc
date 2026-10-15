// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/bounded_workspace.h"

#include "colmap/sensor/bitmap_reader.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/oiio_utils.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <OpenImageIO/imageio.h>

namespace colmap {
namespace mvs {
namespace {

size_t CheckedAddBytes(const size_t lhs, const size_t rhs) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    throw std::overflow_error("Workspace decoded byte count overflow");
  }
  return lhs + rhs;
}

size_t CheckedRasterBytes(const size_t width,
                          const size_t height,
                          const size_t bytes_per_pixel) {
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  if (width > std::numeric_limits<size_t>::max() / height ||
      width * height > std::numeric_limits<size_t>::max() / bytes_per_pixel) {
    throw std::overflow_error("Workspace raster byte count overflow");
  }
  return width * height * bytes_per_pixel;
}

struct RasterShape {
  size_t width;
  size_t height;
  size_t bytes;
};

RasterShape ReadMapShape(const std::filesystem::path& path,
                         const size_t expected_channels) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  // Keep the same bounded three-delimiter FP32 contract as Mat<float>::Read.
  // Ordinary map readers accept trailing bytes.
  std::string header;
  size_t separators = 0;
  while (header.size() < 128 && separators < 3) {
    char byte = 0;
    file.get(byte);
    THROW_CHECK(file) << path;
    header.push_back(byte);
    if (byte == '&') ++separators;
  }
  THROW_CHECK_EQ(separators, 3) << path;
  std::istringstream shape(header);
  size_t width = 0, height = 0, channels = 0;
  char sep1 = 0, sep2 = 0, sep3 = 0;
  shape >> width >> sep1 >> height >> sep2 >> channels >> sep3;
  THROW_CHECK(shape && sep1 == '&' && sep2 == '&' && sep3 == '&') << path;
  THROW_CHECK_EQ(shape.peek(), std::char_traits<char>::eof()) << path;
  THROW_CHECK_EQ(channels, expected_channels) << path;
  const size_t bytes =
      CheckedRasterBytes(width, height, channels * sizeof(float));
  THROW_CHECK_LE(
      bytes, static_cast<size_t>(std::numeric_limits<std::streamoff>::max()))
      << path;
  const auto begin = file.tellg();
  THROW_CHECK_GE(begin, std::streampos(0)) << path;
  file.seekg(0, std::ios::end);
  const auto end = file.tellg();
  THROW_CHECK_GE(end, begin) << path;
  THROW_CHECK_GE(end - begin, static_cast<std::streamoff>(bytes)) << path;
  return {width, height, bytes};
}

}  // namespace

size_t BoundedWorkspace::ImageData::NumBytes() const {
  return CheckedAddBytes(
      bitmap.NumBytes(),
      CheckedAddBytes(depth_map.GetData().capacity() * sizeof(float),
                      normal_map.GetData().capacity() * sizeof(float)));
}

struct BoundedWorkspace::CacheEntry {
  std::shared_ptr<ImageData> image;
  std::exception_ptr error;
  size_t pins = 0;
  uint64_t last_access = 0;
  bool loading = true;
  size_t active_index = 0;
};

struct BoundedWorkspace::CacheState
    : std::enable_shared_from_this<BoundedWorkspace::CacheState> {
  CacheState(const size_t capacity, const size_t num_images)
      : entries(num_images), load_bytes(num_images, 0) {
    stats.capacity_bytes = capacity;
  }

  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::shared_ptr<CacheEntry>> entries;
  std::vector<size_t> load_bytes;
  std::vector<int> active_keys;
  CacheStats stats;
  uint64_t clock = 0;

  void Retire(const int key) {
    const auto entry = entries.at(key);
    const size_t index = entry->active_index;
    const int last = active_keys.back();
    active_keys[index] = last;
    entries[last]->active_index = index;
    active_keys.pop_back();
    entries[key].reset();
  }

  bool MakeRoom(const size_t requested) {
    while (requested >
           stats.capacity_bytes - stats.resident_bytes - stats.reserved_bytes) {
      int victim = -1;
      uint64_t oldest = std::numeric_limits<uint64_t>::max();
      for (const int key : active_keys) {
        const auto& entry = entries[key];
        if (!entry->loading && entry->pins == 0 &&
            entry->last_access < oldest) {
          oldest = entry->last_access;
          victim = key;
        }
      }
      if (victim < 0) return false;
      stats.resident_bytes -= entries[victim]->image->NumBytes();
      Retire(victim);
      ++stats.evictions;
    }
    return true;
  }

  ImageLease MakeLease(const std::shared_ptr<CacheEntry>& entry) {
    try {
      return ImageLease(shared_from_this(), entry);
    } catch (...) {
      --entry->pins;
      changed.notify_all();
      throw;
    }
  }
};

BoundedWorkspace::ImageLease::ImageLease(
    std::shared_ptr<CacheState> state,
    std::shared_ptr<CacheEntry> entry) noexcept
    : state_(std::move(state)), entry_(std::move(entry)) {}

BoundedWorkspace::ImageLease::ImageLease(const ImageLease& other)
    : state_(other.state_), entry_(other.entry_) {
  if (entry_) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ++entry_->pins;
  }
}

BoundedWorkspace::ImageLease& BoundedWorkspace::ImageLease::operator=(
    const ImageLease& other) {
  if (this != &other) {
    ImageLease replacement(other);
    *this = std::move(replacement);
  }
  return *this;
}

BoundedWorkspace::ImageLease& BoundedWorkspace::ImageLease::operator=(
    ImageLease&& other) noexcept {
  if (this != &other) {
    reset();
    state_ = std::move(other.state_);
    entry_ = std::move(other.entry_);
  }
  return *this;
}

BoundedWorkspace::ImageLease::~ImageLease() { reset(); }

const BoundedWorkspace::ImageData* BoundedWorkspace::ImageLease::get()
    const noexcept {
  return entry_ ? entry_->image.get() : nullptr;
}

void BoundedWorkspace::ImageLease::reset() noexcept {
  const auto state = std::move(state_);
  if (entry_) {
    std::lock_guard<std::mutex> lock(state->mutex);
    const bool unpinned = --entry_->pins == 0;
    entry_.reset();
    if (unpinned) state->changed.notify_all();
  }
}

BoundedWorkspace::BoundedWorkspace(const Options& options)
    : Workspace(options) {
  THROW_CHECK(!options.compact_rgb_to_depth_grid || options.image_as_rgb);
  const double bytes = options.cache_size * 1024.0 * 1024.0 * 1024.0;
  THROW_CHECK(std::isfinite(bytes));
  THROW_CHECK_GE(bytes, 1.0);
  THROW_CHECK_LT(bytes,
                 static_cast<double>(std::numeric_limits<size_t>::max()));
  THROW_CHECK_LE(model_.images.size(),
                 static_cast<size_t>(std::numeric_limits<int>::max()));
  cache_state_ = std::make_shared<CacheState>(static_cast<size_t>(bytes),
                                              model_.images.size());
}

size_t BoundedWorkspace::RequiredLoadBytes(const int image_idx) const {
  // Header-only planning. Fusion inputs must remain immutable while loading.
  const auto bitmap_path = GetBitmapPath(image_idx);
  OIIO::ImageSpec spec;
  if (const auto reader = ScopedBitmapReader::Find(bitmap_path)) {
    const auto dimensions = reader->ImageSize(bitmap_path);
    if (!dimensions) {
      throw std::invalid_argument(
          "Resident fusion image needs declared RGB8 dimensions before "
          "admission");
    }
    spec.width = dimensions->first;
    spec.height = dimensions->second;
    spec.nchannels = 3;
  } else {
    EnsureOpenImageIOInitialized();
    OIIO::ImageSpec config;
    config["oiio:reorient"] = 0;
    const auto utf8_path = bitmap_path.u8string();
    const std::string path(reinterpret_cast<const char*>(utf8_path.data()),
                           utf8_path.size());
    const auto input = OIIO::ImageInput::open(path, &config);
    THROW_CHECK_NOTNULL(input.get());
    spec = input->spec();
    input->close();
  }
  THROW_CHECK_GT(spec.width, 0);
  THROW_CHECK_GT(spec.height, 0);
  THROW_CHECK_GE(spec.nchannels, 1);
  THROW_CHECK_LE(spec.nchannels, 4);

  const auto depth = ReadMapShape(GetDepthMapPath(image_idx), 1);
  const auto normal = ReadMapShape(GetNormalMapPath(image_idx), 3);
  THROW_CHECK_EQ(depth.width, normal.width);
  THROW_CHECK_EQ(depth.height, normal.height);
  const auto& image = model_.images.at(image_idx);
  const size_t width = image.GetWidth();
  const size_t height = image.GetHeight();
  const size_t channels = options_.image_as_rgb ? 3 : 1;
  const size_t source_bitmap_bytes =
      CheckedRasterBytes(spec.width, spec.height, 3);
  size_t bytes = CheckedAddBytes(source_bitmap_bytes,
                                 CheckedAddBytes(depth.bytes, normal.bytes));
  const int decoded_channels = spec.nchannels <= 2 ? 1 : 3;
  if (decoded_channels != static_cast<int>(channels)) {
    bytes = CheckedAddBytes(bytes, source_bitmap_bytes);
  }
  if (spec.width != static_cast<int>(width) ||
      spec.height != static_cast<int>(height)) {
    bytes = CheckedAddBytes(bytes, CheckedRasterBytes(width, height, channels));
  }
  for (const auto& shape : {depth, normal}) {
    if (shape.width > width || shape.height > height) {
      const size_t scratch = CheckedAddBytes(shape.bytes, shape.bytes);
      bytes = CheckedAddBytes(bytes, CheckedAddBytes(scratch, scratch));
    }
  }
  if (options_.compact_rgb_to_depth_grid) {
    const auto grid = GetDepthMapDimensions(image_idx);
    if (grid.first != width || grid.second != height) {
      // Source pixels remain live until the compact buffer is filled.
      bytes = CheckedAddBytes(bytes,
                              CheckedRasterBytes(grid.first, grid.second, 3));
    }
  }
  return bytes;
}

size_t BoundedWorkspace::GetLoadReservationBytes(const int image_idx) const {
  model_.images.at(image_idx);
  const auto state = cache_state_;
  std::unique_lock<std::mutex> lock(state->mutex);
  const size_t cached = state->load_bytes.at(image_idx);
  if (cached != 0) return cached;
  lock.unlock();
  const size_t bytes = RequiredLoadBytes(image_idx);
  lock.lock();
  state->load_bytes[image_idx] = bytes;
  return bytes;
}

std::pair<size_t, size_t> BoundedWorkspace::GetDepthMapDimensions(
    const int image_idx) const {
  const auto shape = ReadMapShape(GetDepthMapPath(image_idx), 1);
  const auto& image = model_.images.at(image_idx);
  if (shape.width <= image.GetWidth() && shape.height <= image.GetHeight()) {
    return {shape.width, shape.height};
  }
  const float factor =
      std::min(static_cast<float>(image.GetWidth()) / shape.width,
               static_cast<float>(image.GetHeight()) / shape.height);
  const size_t width = std::round(shape.width * factor);
  const size_t height = std::round(shape.height * factor);
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  return {width, height};
}

size_t BoundedWorkspace::GetResidentBytesEstimate(const int image_idx) const {
  const auto grid = GetDepthMapDimensions(image_idx);
  const auto& image = model_.images.at(image_idx);
  const size_t rgb_bytes =
      options_.compact_rgb_to_depth_grid
          ? CheckedRasterBytes(grid.first, grid.second, 3)
          : CheckedRasterBytes(image.GetWidth(),
                               image.GetHeight(),
                               options_.image_as_rgb ? 3 : 1);
  return CheckedAddBytes(rgb_bytes,
                         CheckedRasterBytes(grid.first, grid.second, 16));
}

BoundedWorkspace::ImageLease BoundedWorkspace::AcquireImage(
    const int image_idx) {
  model_.images.at(image_idx);
  const auto state = cache_state_;
  std::unique_lock<std::mutex> lock(state->mutex);
  size_t required_bytes = 0;
  std::shared_ptr<CacheEntry> entry;
  for (;;) {
    entry = state->entries[image_idx];
    if (entry) {
      ++entry->pins;
      ++state->stats.hits;
      entry->last_access = ++state->clock;
      if (entry->loading) {
        ++state->stats.waits;
        state->changed.wait(lock, [&] { return !entry->loading; });
      }
      if (entry->error) {
        --entry->pins;
        state->changed.notify_all();
        std::rethrow_exception(entry->error);
      }
      return state->MakeLease(entry);
    }
    if (required_bytes == 0) {
      required_bytes = state->load_bytes[image_idx];
      if (required_bytes == 0) {
        lock.unlock();
        required_bytes = GetLoadReservationBytes(image_idx);
        lock.lock();
      }
      continue;
    }
    if (required_bytes > state->stats.capacity_bytes) {
      throw std::runtime_error("StereoFusion bounded_cache: image " +
                               model_.GetImageName(image_idx) + " needs " +
                               std::to_string(required_bytes) +
                               " reserved bytes, exceeding cache_size (" +
                               std::to_string(state->stats.capacity_bytes) +
                               ")");
    }
    if (!state->MakeRoom(required_bytes)) {
      ++state->stats.waits;
      state->changed.wait(lock);
      continue;
    }
    entry = std::make_shared<CacheEntry>();
    entry->active_index = state->active_keys.size();
    state->active_keys.push_back(image_idx);
    state->entries[image_idx] = entry;
    entry->pins = 1;
    entry->last_access = ++state->clock;
    state->stats.reserved_bytes += required_bytes;
    state->stats.peak_charged_bytes =
        std::max(state->stats.peak_charged_bytes,
                 state->stats.resident_bytes + state->stats.reserved_bytes);
    ++state->stats.loads;
    break;
  }

  lock.unlock();
  std::shared_ptr<ImageData> data;
  try {
    data = std::make_shared<ImageData>();
    const auto& image = model_.images.at(image_idx);
    const size_t width = image.GetWidth();
    const size_t height = image.GetHeight();
    THROW_CHECK(
        data->bitmap.Read(GetBitmapPath(image_idx), options_.image_as_rgb));
    if (data->bitmap.Width() != static_cast<int>(width) ||
        data->bitmap.Height() != static_cast<int>(height)) {
      data->bitmap.Rescale(static_cast<int>(width), static_cast<int>(height));
    }
    data->depth_map.Read(GetDepthMapPath(image_idx));
    data->normal_map.Read(GetNormalMapPath(image_idx));
    data->depth_map.Downsize(width, height);
    data->normal_map.Downsize(width, height);
    THROW_CHECK_EQ(data->depth_map.GetWidth(), data->normal_map.GetWidth());
    THROW_CHECK_EQ(data->depth_map.GetHeight(), data->normal_map.GetHeight());
    if (options_.compact_rgb_to_depth_grid) {
      const auto grid = GetDepthMapDimensions(image_idx);
      if (grid.first != width || grid.second != height) {
        Bitmap compact(
            static_cast<int>(grid.first), static_cast<int>(grid.second), true);
        // Match fusion float division, round and out-of-range black.
        const float scale_x = static_cast<float>(grid.first) / width;
        const float scale_y = static_cast<float>(grid.second) / height;
        for (size_t row = 0; row < grid.second; ++row) {
          for (size_t col = 0; col < grid.first; ++col) {
            compact.SetPixel(
                col,
                row,
                data->bitmap
                    .InterpolateNearestNeighbor(static_cast<int>(col) / scale_x,
                                                static_cast<int>(row) / scale_y)
                    .value_or(BitmapColor<uint8_t>(0)));
          }
        }
        data->bitmap = std::move(compact);
        data->bitmap_on_depth_grid = true;
      }
    }
    THROW_CHECK_LE(data->NumBytes(), required_bytes);
    THROW_CHECK_LE(data->NumBytes(), GetResidentBytesEstimate(image_idx));
  } catch (...) {
    data.reset();
    lock.lock();
    entry->error = std::current_exception();
    entry->loading = false;
    --entry->pins;
    state->stats.reserved_bytes -= required_bytes;
    state->Retire(image_idx);
    state->changed.notify_all();
    throw;
  }
  lock.lock();
  entry->image = std::move(data);
  entry->loading = false;
  state->stats.reserved_bytes -= required_bytes;
  state->stats.resident_bytes += entry->image->NumBytes();
  state->changed.notify_all();
  return state->MakeLease(entry);
}

BoundedWorkspace::CacheStats BoundedWorkspace::GetCacheStats() const {
  std::lock_guard<std::mutex> lock(cache_state_->mutex);
  return cache_state_->stats;
}

const Bitmap& BoundedWorkspace::GetBitmap(const int) {
  throw std::logic_error("BoundedWorkspace requires AcquireImage lease");
}
const DepthMap& BoundedWorkspace::GetDepthMap(const int) {
  throw std::logic_error("BoundedWorkspace requires AcquireImage lease");
}
const NormalMap& BoundedWorkspace::GetNormalMap(const int) {
  throw std::logic_error("BoundedWorkspace requires AcquireImage lease");
}

}  // namespace mvs
}  // namespace colmap
