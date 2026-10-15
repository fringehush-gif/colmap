// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/bounded_workspace.h"

#include "colmap/sensor/bitmap_reader.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/oiio_utils.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
  SetFullPrecTextStream(shape);
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

Eigen::Vector3f BoundedWorkspace::ImageData::GetNormal(const size_t row,
                                                       const size_t col) const {
  THROW_CHECK_GE(row, normal_origin_y);
  THROW_CHECK_GE(col, normal_origin_x);
  const size_t local_row = row - normal_origin_y;
  const size_t local_col = col - normal_origin_x;
  THROW_CHECK_LT(local_row, normal_map.GetHeight());
  THROW_CHECK_LT(local_col, normal_map.GetWidth());
  Eigen::Vector3f normal;
  normal_map.GetSlice(local_row, local_col, normal.data());
  return normal;
}

struct BoundedWorkspace::CacheEntry {
  std::shared_ptr<ImageData> image;
  std::exception_ptr error;
  size_t pins = 0;
  uint64_t last_access = 0;
  bool loading = true;
  size_t active_index = 0;
  std::optional<std::tuple<int, size_t, size_t, size_t>> tile_key;
};

struct BoundedWorkspace::CacheState
    : std::enable_shared_from_this<BoundedWorkspace::CacheState> {
  CacheState(const size_t capacity, const size_t num_images)
      : entries(num_images), load_bytes(num_images, 0), base_keys(num_images) {
    stats.capacity_bytes = capacity;
  }

  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::shared_ptr<CacheEntry>> entries;
  std::vector<size_t> load_bytes;
  std::vector<int> active_keys;
  std::map<std::tuple<int, size_t, size_t, size_t>, int> tile_keys;
  std::vector<int> free_tile_keys;
  size_t base_keys;
  CacheStats stats;
  uint64_t clock = 0;
  bool fail_next_tile_key_publication_for_testing = false;

  void PublishTileKey(const std::tuple<int, size_t, size_t, size_t>& tile_key,
                      const int key) {
    if (std::exchange(fail_next_tile_key_publication_for_testing, false)) {
      throw std::bad_alloc();
    }
    const auto [_, inserted] = tile_keys.emplace(tile_key, key);
    THROW_CHECK(inserted);
  }

  void Retire(const int key) {
    const auto entry = entries.at(key);
    const size_t index = entry->active_index;
    const int last = active_keys.back();
    active_keys[index] = last;
    entries[last]->active_index = index;
    active_keys.pop_back();
    if (entry->tile_key) {
      tile_keys.erase(*entry->tile_key);
      free_tile_keys.push_back(key);
    }
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

BoundedWorkspace::MappedNormalPool::Lease::Lease(
    std::shared_ptr<MappedNormalPool> pool, Entry* entry)
    : pool(std::move(pool)), entry(entry) {}
BoundedWorkspace::MappedNormalPool::Lease::Lease(Lease&& other) noexcept
    : pool(std::move(other.pool)), entry(std::exchange(other.entry, nullptr)) {}
BoundedWorkspace::MappedNormalPool::Lease::~Lease() {
  if (!entry) return;
  std::lock_guard<std::mutex> lock(pool->mutex);
  --entry->pins;
  --pool->stats.active_copies;
  pool->changed.notify_all();
}

BoundedWorkspace::MappedNormalPool::MappedNormalPool(const size_t bytes,
                                                     const size_t files) {
  THROW_CHECK_LE(files, std::numeric_limits<size_t>::max() / 3);
  stats.capacity_bytes = bytes;
  stats.capacity_files = files;
  stats.capacity_regions = files * 3;
}

BoundedWorkspace::MappedNormalPool::Lease
BoundedWorkspace::MappedNormalPool::Acquire(const int image_idx,
                                            const std::filesystem::path& path) {
  return AcquireImpl(Key{image_idx, 0, 0}, path);
}

BoundedWorkspace::MappedNormalPool::Lease
BoundedWorkspace::MappedNormalPool::Acquire(const int image_idx,
                                            const std::filesystem::path& path,
                                            const size_t row_begin,
                                            const size_t row_count) {
  THROW_CHECK_GT(row_count, 0);
  return AcquireImpl(Key{image_idx, row_begin, row_count}, path);
}

std::shared_ptr<const ImmutableMapSource>
BoundedWorkspace::MappedNormalPool::AcquireSource(
    const int image_idx, const std::filesystem::path& path) {
  std::unique_lock<std::mutex> lock(mutex);
  for (;;) {
    auto found = sources.find(image_idx);
    if (found != sources.end()) {
      THROW_CHECK_EQ(found->second.path, path);
      if (found->second.loading) {
        ++stats.waits;
        changed.wait(lock);
        continue;
      }
      found->second.last_access = ++clock;
      ++stats.source_hits;
      return found->second.source;
    }
    if (sources.size() + sources_closing >= stats.capacity_files) {
      auto victim = sources.end();
      for (auto it = sources.begin(); it != sources.end(); ++it) {
        if (it->second.loading) continue;
        bool pinned = false;
        for (const auto& [key, entry] : entries) {
          if (std::get<0>(key) == it->first &&
              (entry.loading || entry.pins != 0)) {
            pinned = true;
            break;
          }
        }
        if (!pinned && (victim == sources.end() ||
                        it->second.last_access < victim->second.last_access)) {
          victim = it;
        }
      }
      if (victim == sources.end()) {
        ++stats.waits;
        changed.wait(lock);
        continue;
      }
      // A cached, unpinned stripe still owns its source. Retire those stripes
      // first so a single-FD pool can make forward progress for a new image.
      for (auto it = entries.begin(); it != entries.end();) {
        if (std::get<0>(it->first) != victim->first) {
          ++it;
          continue;
        }
        stats.resident_bytes -= it->second.owner->GetMappingBytes();
        --stats.resident_files;
        stats.resident_regions -= it->second.regions;
        it = entries.erase(it);
        ++stats.evictions;
      }
      // A caller may temporarily hold a source while waiting on stripe
      // capacity; it releases that reference before sleeping below.
      if (victim->second.source.use_count() != 1) {
        ++stats.waits;
        changed.wait_for(lock, std::chrono::milliseconds(10));
        continue;
      }
      auto retired = std::move(victim->second.source);
      sources.erase(victim);
      ++sources_closing;
      ++stats.source_evictions;
      lock.unlock();
      retired
          .reset();  // Close outside the pool mutex before reusing its FD slot.
      lock.lock();
      --sources_closing;
      --stats.source_live_files;
      stats.source_charged_files = sources.size() + sources_closing;
      changed.notify_all();
      continue;
    }
    SourceEntry pending;
    pending.path = path;
    const bool inserted = sources.emplace(image_idx, std::move(pending)).second;
    THROW_CHECK(inserted);
    stats.source_charged_files = sources.size() + sources_closing;
    stats.source_peak_charged_files =
        std::max(stats.source_peak_charged_files, stats.source_charged_files);
    lock.unlock();
    std::shared_ptr<const ImmutableMapSource> source;
    try {
      source = std::make_shared<ImmutableMapSource>(path);
    } catch (...) {
      lock.lock();
      sources.erase(image_idx);
      stats.source_charged_files = sources.size() + sources_closing;
      changed.notify_all();
      throw;
    }
    lock.lock();
    auto& ready = sources.at(image_idx);
    ready.source = source;
    ready.loading = false;
    ready.last_access = ++clock;
    ++stats.source_loads;
    ++stats.source_live_files;
    stats.source_peak_live_files =
        std::max(stats.source_peak_live_files, stats.source_live_files);
    changed.notify_all();
    return source;
  }
}

BoundedWorkspace::MappedNormalPool::Lease
BoundedWorkspace::MappedNormalPool::AcquireImpl(
    const Key key, const std::filesystem::path& path) {
  // The third key component distinguishes a full owner from every stripe.
  const bool window = std::get<2>(key) != 0;
  const size_t regions = window ? 3 : 1;
  std::unique_lock<std::mutex> lock(mutex);
  size_t requested = 0;
  bool reservation_known = false;
  std::shared_ptr<const ImmutableMapSource> source;
  for (;;) {
    auto found = entries.find(key);
    if (found != entries.end()) {
      if (found->second.loading) {
        ++stats.waits;
        changed.wait(lock);
        continue;
      }
      ++found->second.pins;
      found->second.last_access = ++clock;
      ++stats.hits;
      ++stats.active_copies;
      return Lease(shared_from_this(), &found->second);
    }
    if (!reservation_known) {
      // The Workspace input is immutable. A resident owner already pins this
      // exact image's mapped file, so only a new mapping needs filesystem
      // reservation metadata. Recheck the key after dropping the lock.
      lock.unlock();
      source = AcquireSource(std::get<0>(key), path);
      requested = window ? source->GetWindowReservationBytes(std::get<1>(key),
                                                             std::get<2>(key))
                         : source->GetMappingReservationBytes();
      lock.lock();
      reservation_known = true;
      continue;
    }
    THROW_CHECK_LE(requested, stats.capacity_bytes);
    THROW_CHECK_LE(regions, stats.capacity_regions);
    while (requested > stats.capacity_bytes - stats.resident_bytes -
                           stats.reserved_bytes ||
           stats.resident_files + stats.reserved_files >=
               stats.capacity_files ||
           regions > stats.capacity_regions - stats.resident_regions -
                         stats.reserved_regions) {
      auto victim = entries.end();
      for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (!it->second.loading && it->second.pins == 0 &&
            (victim == entries.end() ||
             it->second.last_access < victim->second.last_access)) {
          victim = it;
        }
      }
      if (victim == entries.end()) break;
      stats.resident_bytes -= victim->second.owner->GetMappingBytes();
      --stats.resident_files;
      stats.resident_regions -= victim->second.regions;
      entries.erase(victim);  // Unmap/close before freeing the charge.
      ++stats.evictions;
    }
    if (requested > stats.capacity_bytes - stats.resident_bytes -
                        stats.reserved_bytes ||
        stats.resident_files + stats.reserved_files >= stats.capacity_files ||
        regions > stats.capacity_regions - stats.resident_regions -
                      stats.reserved_regions) {
      source.reset();
      reservation_known = false;
      ++stats.waits;
      changed.wait(lock);
      continue;
    }
    auto [inserted, fresh] = entries.try_emplace(key);
    THROW_CHECK(fresh);
    inserted->second.pins = 1;
    inserted->second.reserved_bytes = requested;
    inserted->second.regions = regions;
    stats.reserved_bytes += requested;
    ++stats.reserved_files;
    stats.reserved_regions += regions;
    stats.peak_charged_bytes = std::max(
        stats.peak_charged_bytes, stats.resident_bytes + stats.reserved_bytes);
    stats.peak_charged_files = std::max(
        stats.peak_charged_files, stats.resident_files + stats.reserved_files);
    stats.peak_charged_regions =
        std::max(stats.peak_charged_regions,
                 stats.resident_regions + stats.reserved_regions);
    lock.unlock();
    std::unique_ptr<ImmutableMapRegion> owner;
    try {
      owner = window ? std::make_unique<ImmutableMapRegion>(
                           source, std::get<1>(key), std::get<2>(key))
                     : std::make_unique<ImmutableMapRegion>(source);
      THROW_CHECK_LE(owner->GetMappingBytes(), requested);
      THROW_CHECK_EQ(owner->GetMappingRegionCount(), regions);
    } catch (...) {
      lock.lock();
      entries.erase(key);
      stats.reserved_bytes -= requested;
      --stats.reserved_files;
      stats.reserved_regions -= regions;
      changed.notify_all();
      throw;
    }
    lock.lock();
    auto& entry = entries.at(key);
    entry.owner = std::move(owner);
    entry.loading = false;
    entry.last_access = ++clock;
    stats.reserved_bytes -= requested;
    --stats.reserved_files;
    stats.reserved_regions -= regions;
    stats.resident_bytes += entry.owner->GetMappingBytes();
    ++stats.resident_files;
    stats.resident_regions += regions;
    ++stats.loads;
    ++stats.active_copies;
    changed.notify_all();
    return Lease(shared_from_this(), &entry);
  }
}

BoundedWorkspace::BoundedWorkspace(const Options& options)
    : Workspace(options),
      normal_tile_grid_once_(model_.images.size()),
      normal_tile_grids_(model_.images.size()) {
  THROW_CHECK(!options.compact_rgb_to_depth_grid || options.image_as_rgb);
  const double bytes = options.cache_size * 1024.0 * 1024.0 * 1024.0;
  THROW_CHECK(std::isfinite(bytes));
  THROW_CHECK_GE(bytes, 1.0);
  THROW_CHECK_LT(bytes,
                 static_cast<double>(std::numeric_limits<size_t>::max()));
  THROW_CHECK_LE(model_.images.size(),
                 static_cast<size_t>(std::numeric_limits<int>::max() / 4));
  THROW_CHECK_EQ(options.mapped_normal_bytes_limit == 0,
                 options.mapped_normal_files_limit == 0);
  if (options.mapped_normal_bytes_limit) {
    THROW_CHECK(ImmutableMapRegion::Supported());
    mapped_normal_pool_ = std::make_shared<MappedNormalPool>(
        options.mapped_normal_bytes_limit, options.mapped_normal_files_limit);
  }
  cache_state_ = std::make_shared<CacheState>(static_cast<size_t>(bytes),
                                              model_.images.size() * 4);
}

int BoundedWorkspace::ComponentKey(const int image_idx,
                                   const Component component) const {
  model_.images.at(image_idx);
  const int part = static_cast<int>(component);
  THROW_CHECK_GE(part, 0);
  THROW_CHECK_LT(part, 4);
  return image_idx * 4 + part;
}

size_t BoundedWorkspace::RequiredLoadBytes(const int image_idx,
                                           const Component component) const {
  // Header-only planning. Fusion inputs must remain immutable while loading.
  const bool read_bitmap =
      component == Component::kImage || component == Component::kBitmap;
  const bool read_depth =
      component == Component::kImage || component == Component::kDepth;
  const bool read_normal =
      component == Component::kImage || component == Component::kNormal;
  const auto bitmap_path = GetBitmapPath(image_idx);
  OIIO::ImageSpec spec;
  if (!read_bitmap) {
    const auto& image = model_.images.at(image_idx);
    spec.width = image.GetWidth();
    spec.height = image.GetHeight();
    spec.nchannels = 3;
  } else if (const auto reader = ScopedBitmapReader::Find(bitmap_path)) {
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

  const bool need_depth_shape =
      read_depth || read_normal ||
      (read_bitmap && options_.compact_rgb_to_depth_grid);
  RasterShape depth = need_depth_shape
                          ? ReadMapShape(GetDepthMapPath(image_idx), 1)
                          : RasterShape{0, 0, 0};
  RasterShape normal = read_normal
                           ? ReadMapShape(GetNormalMapPath(image_idx), 3)
                           : RasterShape{0, 0, 0};
  if (read_normal) {
    THROW_CHECK_EQ(depth.width, normal.width);
    THROW_CHECK_EQ(depth.height, normal.height);
  }
  if (!read_depth) depth.bytes = 0;
  const auto& image = model_.images.at(image_idx);
  const size_t width = image.GetWidth();
  const size_t height = image.GetHeight();
  const size_t channels = options_.image_as_rgb ? 3 : 1;
  const size_t source_bitmap_bytes =
      CheckedRasterBytes(spec.width, spec.height, 3);
  size_t bytes = CheckedAddBytes(read_bitmap ? source_bitmap_bytes : 0,
                                 CheckedAddBytes(depth.bytes, normal.bytes));
  const int decoded_channels = spec.nchannels <= 2 ? 1 : 3;
  if (read_bitmap && decoded_channels != static_cast<int>(channels)) {
    bytes = CheckedAddBytes(bytes, source_bitmap_bytes);
  }
  if (read_bitmap && (spec.width != static_cast<int>(width) ||
                      spec.height != static_cast<int>(height))) {
    bytes = CheckedAddBytes(bytes, CheckedRasterBytes(width, height, channels));
  }
  for (const auto& shape : {depth, normal}) {
    if (shape.width > width || shape.height > height) {
      const size_t scratch = CheckedAddBytes(shape.bytes, shape.bytes);
      bytes = CheckedAddBytes(bytes, CheckedAddBytes(scratch, scratch));
    }
  }
  if (read_bitmap && options_.compact_rgb_to_depth_grid) {
    const auto grid = GetDepthMapDimensions(image_idx);
    if (grid.first != width || grid.second != height) {
      // Source pixels remain live until the compact buffer is filled.
      bytes = CheckedAddBytes(bytes,
                              CheckedRasterBytes(grid.first, grid.second, 3));
    }
  }
  return bytes;
}

size_t BoundedWorkspace::GetLoadReservationBytes(
    const int image_idx, const Component component) const {
  const int key = ComponentKey(image_idx, component);
  const auto state = cache_state_;
  std::unique_lock<std::mutex> lock(state->mutex);
  const size_t cached = state->load_bytes.at(key);
  if (cached != 0) return cached;
  lock.unlock();
  const size_t bytes = RequiredLoadBytes(image_idx, component);
  lock.lock();
  state->load_bytes[key] = bytes;
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

size_t BoundedWorkspace::GetResidentBytesEstimate(
    const int image_idx, const Component component) const {
  const auto& image = model_.images.at(image_idx);
  const auto grid =
      component == Component::kBitmap && !options_.compact_rgb_to_depth_grid
          ? std::pair<size_t, size_t>{image.GetWidth(), image.GetHeight()}
          : GetDepthMapDimensions(image_idx);
  const size_t rgb_bytes =
      options_.compact_rgb_to_depth_grid
          ? CheckedRasterBytes(grid.first, grid.second, 3)
          : CheckedRasterBytes(image.GetWidth(),
                               image.GetHeight(),
                               options_.image_as_rgb ? 3 : 1);
  const size_t depth_bytes = CheckedRasterBytes(grid.first, grid.second, 4);
  const size_t normal_bytes = CheckedRasterBytes(grid.first, grid.second, 12);
  if (component == Component::kBitmap) return rgb_bytes;
  if (component == Component::kDepth) return depth_bytes;
  if (component == Component::kNormal) return normal_bytes;
  return CheckedAddBytes(rgb_bytes, CheckedAddBytes(depth_bytes, normal_bytes));
}

std::pair<size_t, size_t> BoundedWorkspace::GetNormalTileGrid(
    const int image_idx) const {
  const auto& image = model_.images.at(image_idx);
  std::call_once(normal_tile_grid_once_.at(image_idx), [&] {
    const auto depth = ReadMapShape(GetDepthMapPath(image_idx), 1);
    const auto normal = ReadMapShape(GetNormalMapPath(image_idx), 3);
    THROW_CHECK_EQ(depth.width, normal.width);
    THROW_CHECK_EQ(depth.height, normal.height);
    THROW_CHECK_LE(depth.width, image.GetWidth())
        << "FP32 normal tiles cannot resample a stored grid";
    THROW_CHECK_LE(depth.height, image.GetHeight())
        << "FP32 normal tiles cannot resample a stored grid";
    normal_tile_grids_.at(image_idx) = {depth.width, depth.height};
  });
  return normal_tile_grids_.at(image_idx);
}

size_t BoundedWorkspace::GetNormalTileReservationBytes(
    const int image_idx, const size_t tile_size) const {
  THROW_CHECK_GT(tile_size, 0);
  const auto grid = GetNormalTileGrid(image_idx);
  return CheckedRasterBytes(
      std::min(tile_size, grid.first), std::min(tile_size, grid.second), 12);
}

size_t BoundedWorkspace::GetNormalTileResidentBytes(
    const int image_idx, const size_t tile_size) const {
  return GetNormalTileReservationBytes(image_idx, tile_size);
}

BoundedWorkspace::ImageLease BoundedWorkspace::AcquireNormalTile(
    const int image_idx,
    const size_t row,
    const size_t col,
    const size_t tile_size) {
  THROW_CHECK_GT(tile_size, 0);
  THROW_CHECK(mapped_normal_pool_ != nullptr);
  const auto grid = GetNormalTileGrid(image_idx);
  THROW_CHECK_LT(row, grid.second);
  THROW_CHECK_LT(col, grid.first);
  const size_t x = col - col % tile_size;
  const size_t y = row - row % tile_size;
  return AcquireImpl(image_idx,
                     Component::kNormal,
                     NormalRegion{x,
                                  y,
                                  std::min(tile_size, grid.first - x),
                                  std::min(tile_size, grid.second - y),
                                  tile_size});
}

size_t BoundedWorkspace::GetMappedNormalReservationBytes(
    const int image_idx) const {
  THROW_CHECK(mapped_normal_pool_ != nullptr);
  GetNormalTileGrid(image_idx);  // Validate both stored grids and payloads.
  return mapped_normal_pool_
      ->AcquireSource(image_idx, GetNormalMapPath(image_idx))
      ->GetMappingReservationBytes();
}

size_t BoundedWorkspace::GetMappedNormalWindowReservationBytes(
    const int image_idx, const size_t row_begin, const size_t row_count) const {
  THROW_CHECK(mapped_normal_pool_ != nullptr);
  const auto grid = GetNormalTileGrid(image_idx);
  THROW_CHECK_LT(row_begin, grid.second);
  THROW_CHECK_GT(row_count, 0);
  THROW_CHECK_LE(row_count, grid.second - row_begin);
  return mapped_normal_pool_
      ->AcquireSource(image_idx, GetNormalMapPath(image_idx))
      ->GetWindowReservationBytes(row_begin, row_count);
}

BoundedWorkspace::MappedNormalLease BoundedWorkspace::AcquireMappedNormal(
    const int image_idx) {
  THROW_CHECK(mapped_normal_pool_ != nullptr);
  GetNormalTileGrid(image_idx);  // Failed admission remains retryable.
  auto owner = std::make_shared<MappedNormalPool::Lease>(
      mapped_normal_pool_->Acquire(image_idx, GetNormalMapPath(image_idx)));
  return MappedNormalLease(owner, &**owner);
}

BoundedWorkspace::MappedNormalLease BoundedWorkspace::AcquireMappedNormalWindow(
    const int image_idx, const size_t row_begin, const size_t row_count) {
  THROW_CHECK(mapped_normal_pool_ != nullptr);
  const auto grid = GetNormalTileGrid(image_idx);
  THROW_CHECK_LT(row_begin, grid.second);
  THROW_CHECK_GT(row_count, 0);
  THROW_CHECK_LE(row_count, grid.second - row_begin);
  auto owner =
      std::make_shared<MappedNormalPool::Lease>(mapped_normal_pool_->Acquire(
          image_idx, GetNormalMapPath(image_idx), row_begin, row_count));
  return MappedNormalLease(owner, &**owner);
}

BoundedWorkspace::ImageLease BoundedWorkspace::AcquireImage(
    const int image_idx) {
  return AcquireComponent(image_idx, Component::kImage);
}

BoundedWorkspace::ImageLease BoundedWorkspace::AcquireComponent(
    const int image_idx, const Component component) {
  return AcquireImpl(image_idx, component, std::nullopt);
}

BoundedWorkspace::ImageLease BoundedWorkspace::AcquireImpl(
    const int image_idx,
    const Component component,
    const std::optional<NormalRegion> region) {
  model_.images.at(image_idx);
  const auto state = cache_state_;
  size_t required_bytes =
      region ? CheckedRasterBytes(region->width, region->height, 12) : 0;
  if (region && required_bytes > state->stats.capacity_bytes) {
    throw std::runtime_error("Normal tile exceeds bounded decoded cache");
  }
  std::unique_lock<std::mutex> lock(state->mutex);
  int key = ComponentKey(image_idx, component);
  std::optional<std::tuple<int, size_t, size_t, size_t>> tile_key;
  if (region) {
    tile_key =
        std::make_tuple(image_idx, region->x, region->y, region->tile_size);
    auto found = state->tile_keys.find(*tile_key);
    if (found != state->tile_keys.end()) {
      key = found->second;
    } else {
      key = -1;  // Publish a dynamic key only after capacity admission.
    }
  }
  std::shared_ptr<CacheEntry> entry;
  for (;;) {
    // A waiter can lose the admission race to another loader of this tile.
    // Re-resolve after every wake so all callers coalesce on its entry.
    if (tile_key) {
      const auto found = state->tile_keys.find(*tile_key);
      key = found == state->tile_keys.end() ? -1 : found->second;
    }
    entry = key < 0 ? nullptr : state->entries[key];
    if (entry) {
      ++entry->pins;
      ++state->stats.hits;
      if (region) ++state->stats.normal_tile_hits;
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
      required_bytes = state->load_bytes[key];
      if (required_bytes == 0) {
        lock.unlock();
        required_bytes = GetLoadReservationBytes(image_idx, component);
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
    entry->tile_key = tile_key;
    // All reserves can fail before logical mutation. Retire may later return
    // any dynamic key without allocating because free_tile_keys is reserved
    // for every dynamic slot created here.
    state->active_keys.reserve(state->active_keys.size() + 1);
    if (key < 0) {
      if (state->free_tile_keys.empty()) {
        THROW_CHECK_LT(state->entries.size(),
                       static_cast<size_t>(std::numeric_limits<int>::max()));
        state->entries.reserve(state->entries.size() + 1);
        state->load_bytes.reserve(state->load_bytes.size() + 1);
        state->free_tile_keys.reserve(state->entries.size() + 1 -
                                      state->base_keys);
        key = static_cast<int>(state->entries.size());
        state->entries.emplace_back();
        state->load_bytes.push_back(0);
        try {
          state->PublishTileKey(*tile_key, key);
        } catch (...) {
          state->load_bytes.pop_back();
          state->entries.pop_back();
          throw;
        }
      } else {
        key = state->free_tile_keys.back();
        state->PublishTileKey(*tile_key, key);
        state->free_tile_keys.pop_back();
      }
    }
    entry->active_index = state->active_keys.size();
    state->active_keys.push_back(key);
    state->entries[key] = entry;
    entry->pins = 1;
    entry->last_access = ++state->clock;
    state->stats.reserved_bytes += required_bytes;
    state->stats.peak_charged_bytes =
        std::max(state->stats.peak_charged_bytes,
                 state->stats.resident_bytes + state->stats.reserved_bytes);
    ++state->stats.loads;
    ++state->stats.component_loads[static_cast<size_t>(component)];
    if (region) ++state->stats.normal_tile_loads;
    break;
  }

  lock.unlock();
  std::shared_ptr<ImageData> data;
  try {
    data = std::make_shared<ImageData>();
    const auto& image = model_.images.at(image_idx);
    const size_t width = image.GetWidth();
    const size_t height = image.GetHeight();
    if (component == Component::kImage || component == Component::kBitmap) {
      THROW_CHECK(
          data->bitmap.Read(GetBitmapPath(image_idx), options_.image_as_rgb));
      if (data->bitmap.Width() != static_cast<int>(width) ||
          data->bitmap.Height() != static_cast<int>(height)) {
        data->bitmap.Rescale(static_cast<int>(width), static_cast<int>(height));
      }
    }
    if (component == Component::kImage || component == Component::kDepth) {
      data->depth_map.Read(GetDepthMapPath(image_idx));
      data->depth_map.Downsize(width, height);
    }
    if (component == Component::kImage || component == Component::kNormal) {
      if (region) {
        auto mapped = mapped_normal_pool_->Acquire(image_idx,
                                                   GetNormalMapPath(image_idx));
        data->normal_map.ReadRegion(
            *mapped, region->x, region->y, region->width, region->height);
        data->normal_origin_x = region->x;
        data->normal_origin_y = region->y;
      } else {
        data->normal_map.Read(GetNormalMapPath(image_idx));
      }
      if (!region) {
        data->normal_map.Downsize(width, height);
        const auto grid = GetDepthMapDimensions(image_idx);
        THROW_CHECK_EQ(data->normal_map.GetWidth(), grid.first);
        THROW_CHECK_EQ(data->normal_map.GetHeight(), grid.second);
      }
    }
    if (component == Component::kImage) {
      THROW_CHECK_EQ(data->depth_map.GetWidth(), data->normal_map.GetWidth());
      THROW_CHECK_EQ(data->depth_map.GetHeight(), data->normal_map.GetHeight());
    }
    if (options_.compact_rgb_to_depth_grid &&
        (component == Component::kImage || component == Component::kBitmap)) {
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
    THROW_CHECK_LE(data->NumBytes(),
                   region
                       ? CheckedRasterBytes(region->width, region->height, 12)
                       : GetResidentBytesEstimate(image_idx, component));
  } catch (...) {
    data.reset();
    lock.lock();
    entry->error = std::current_exception();
    entry->loading = false;
    --entry->pins;
    state->stats.reserved_bytes -= required_bytes;
    state->Retire(key);
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

BoundedWorkspace::MappedNormalStats BoundedWorkspace::GetMappedNormalStats()
    const {
  if (!mapped_normal_pool_) return {};
  std::lock_guard<std::mutex> lock(mapped_normal_pool_->mutex);
  return mapped_normal_pool_->stats;
}

BoundedWorkspace::TileMetadataStats BoundedWorkspace::GetTileMetadataStats()
    const {
  std::lock_guard<std::mutex> lock(cache_state_->mutex);
  return {cache_state_->entries.size() - cache_state_->base_keys,
          cache_state_->free_tile_keys.size(),
          cache_state_->tile_keys.size()};
}

void BoundedWorkspace::FailNextTileKeyPublicationForTesting() {
  std::lock_guard<std::mutex> lock(cache_state_->mutex);
  cache_state_->fail_next_tile_key_publication_for_testing = true;
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
