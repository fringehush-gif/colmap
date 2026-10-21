// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/workspace.h"

#include "colmap/sensor/bitmap_reader.h"
#include "colmap/util/file.h"
#include "colmap/util/threading.h"

#include <array>
#include <charconv>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <system_error>

namespace colmap {
namespace mvs {
namespace {

// A conservative FP32 map admission check. The ordinary reader remains the
// fallback for every uncertain header, missing payload, or non-normal map.
// Trailing bytes are accepted, as by Mat<float>::Read.
bool IsCompleteThreeChannelFloatMap(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  // The native writer emits decimal dimensions separated by '&'. Cap header
  // work; unusual or locale-dependent headers retain the ordinary read path.
  std::string header;
  size_t delimiters = 0;
  while (header.size() < 128 && delimiters < 3) {
    char byte = 0;
    if (!file.get(byte)) return false;
    header.push_back(byte);
    delimiters += byte == '&';
  }
  if (delimiters != 3) return false;
  const size_t first = header.find('&');
  const size_t second = header.find('&', first + 1);
  const size_t third = header.size() - 1;
  std::array<size_t, 3> dimensions{};
  const auto parse_dimension =
      [&](const size_t begin, const size_t end, size_t* value) {
        if (begin == end) return false;
        const auto result =
            std::from_chars(header.data() + begin, header.data() + end, *value);
        return result.ec == std::errc() && result.ptr == header.data() + end &&
               *value > 0;
      };
  if (!parse_dimension(0, first, &dimensions[0]) ||
      !parse_dimension(first + 1, second, &dimensions[1]) ||
      !parse_dimension(second + 1, third, &dimensions[2]) ||
      dimensions[2] != 3) {
    return false;
  }
  const size_t width = dimensions[0];
  const size_t height = dimensions[1];
  const size_t depth = dimensions[2];
  const size_t max_size = std::numeric_limits<size_t>::max();
  if (width > max_size / height || width * height > max_size / depth ||
      width * height * depth > max_size / sizeof(float)) {
    return false;
  }
  std::error_code error;
  const auto file_bytes = std::filesystem::file_size(path, error);
  const auto payload_bytes = static_cast<std::uintmax_t>(header.size());
  const auto required_bytes =
      static_cast<std::uintmax_t>(width * height * depth * sizeof(float));
  return !error && file_bytes >= payload_bytes &&
         file_bytes - payload_bytes >= required_bytes;
}

}  // namespace

Workspace::Workspace(const Options& options) : options_(options) {
  StringToLower(&options_.input_type);
  model_.Read(options_.workspace_path, options_.workspace_format);
  if (options_.max_image_size > 0) {
    for (auto& image : model_.images) {
      image.Downsize(options_.max_image_size, options_.max_image_size);
    }
  }

  depth_map_path_ =
      options_.workspace_path / options_.stereo_folder / "depth_maps";
  normal_map_path_ =
      options_.workspace_path / options_.stereo_folder / "normal_maps";
}

std::string Workspace::GetFileName(const int image_idx) const {
  const auto& image_name = model_.GetImageName(image_idx);
  return StringPrintf(
      "%s.%s.bin", image_name.c_str(), options_.input_type.c_str());
}

void Workspace::Load(const std::vector<std::string>& image_names) {
  const size_t num_images = model_.images.size();
  bitmaps_.resize(num_images);
  depth_maps_.resize(num_images);
  normal_maps_.resize(num_images);
  auto LoadWorkspaceData = [&, this](const int image_idx) {
    const size_t width = model_.images.at(image_idx).GetWidth();
    const size_t height = model_.images.at(image_idx).GetHeight();

    // Read and rescale bitmap
    auto bitmap = std::make_unique<Bitmap>();
    bitmap->Read(GetBitmapPath(image_idx), options_.image_as_rgb);
    if (bitmap->Width() != static_cast<int>(width) ||
        bitmap->Height() != static_cast<int>(height)) {
      bitmap->Rescale(static_cast<int>(width), static_cast<int>(height));
    }
    bitmaps_[image_idx] = std::move(bitmap);

    // Read and rescale depth map
    auto depth_map = std::make_unique<DepthMap>();
    depth_map->Read(GetDepthMapPath(image_idx));
    if (depth_map->GetWidth() != width || depth_map->GetHeight() != height) {
      depth_map->Downsize(width, height);
    }
    depth_maps_[image_idx] = std::move(depth_map);

    auto normal = std::make_unique<NormalMap>();
    normal->Read(GetNormalMapPath(image_idx));
    if (normal->GetWidth() != width || normal->GetHeight() != height) {
      normal->Downsize(width, height);
    }
    normal_maps_[image_idx] = std::move(normal);
  };

  // Each model slot is published by at most one task in this Load call.
  std::vector<uint8_t> enqueued(num_images, 0);
  const int num_threads = GetEffectiveNumThreads(options_.num_threads);
  ThreadPool thread_pool(num_threads);
  Timer timer;
  timer.Start();

  LOG(INFO) << StringPrintf("Loading workspace data with %d threads...",
                            num_threads);
  for (size_t i = 0; i < image_names.size(); ++i) {
    const int image_idx = model_.GetImageIdx(image_names[i]);
    if (enqueued[image_idx]) continue;
    if (HasBitmap(image_idx) && HasDepthMap(image_idx)) {
      thread_pool.AddTask(LoadWorkspaceData, image_idx);
      enqueued[image_idx] = 1;
    } else {
      LOG(WARNING) << StringPrintf(
          "Ignoring image %s, because input does not exist.",
          image_names[i].c_str());
    }
  }
  try {
    thread_pool.Wait();
  } catch (...) {
    const auto failure = std::current_exception();
    try {
      thread_pool.Wait();
    } catch (...) {
    }
    std::rethrow_exception(failure);
  }
  timer.PrintMinutes();
}

void Workspace::ReleaseFullRamImageInputs(const int image_idx) {
  // Cached workspaces do not initialize the eager owners with Load.
  THROW_CHECK_EQ(bitmaps_.size(), model_.images.size());
  THROW_CHECK_EQ(depth_maps_.size(), model_.images.size());
  THROW_CHECK_EQ(normal_maps_.size(), model_.images.size());
  auto& bitmap = bitmaps_.at(image_idx);
  auto& depth = depth_maps_.at(image_idx);
  auto& normal = normal_maps_.at(image_idx);
  bitmap.reset();
  depth.reset();
  normal.reset();
}

const Bitmap& Workspace::GetBitmap(const int image_idx) {
  return *THROW_CHECK_NOTNULL(bitmaps_.at(image_idx));
}

const DepthMap& Workspace::GetDepthMap(const int image_idx) {
  return *THROW_CHECK_NOTNULL(depth_maps_.at(image_idx));
}

const NormalMap& Workspace::GetNormalMap(const int image_idx) {
  return *THROW_CHECK_NOTNULL(normal_maps_.at(image_idx));
}

std::shared_ptr<const Bitmap> Workspace::GetBitmapLease(const int image_idx) {
  return {std::shared_ptr<const Bitmap>{}, &GetBitmap(image_idx)};
}

std::shared_ptr<const DepthMap> Workspace::GetDepthMapLease(
    const int image_idx) {
  return {std::shared_ptr<const DepthMap>{}, &GetDepthMap(image_idx)};
}

std::shared_ptr<const NormalMap> Workspace::GetNormalMapLease(
    const int image_idx) {
  return {std::shared_ptr<const NormalMap>{}, &GetNormalMap(image_idx)};
}

std::filesystem::path Workspace::GetBitmapPath(const int image_idx) const {
  return model_.images.at(image_idx).GetPath();
}

std::filesystem::path Workspace::GetDepthMapPath(const int image_idx) const {
  return depth_map_path_ / GetFileName(image_idx);
}

std::filesystem::path Workspace::GetNormalMapPath(const int image_idx) const {
  return normal_map_path_ / GetFileName(image_idx);
}

bool Workspace::HasBitmap(const int image_idx) const {
  const auto path = GetBitmapPath(image_idx);
  return ScopedBitmapReader::Find(path) || ExistsFile(path);
}

bool Workspace::HasDepthMap(const int image_idx) const {
  return ExistsFile(GetDepthMapPath(image_idx));
}

bool Workspace::HasNormalMap(const int image_idx) const {
  return ExistsFile(GetNormalMapPath(image_idx));
}

CachedWorkspace::CachedImage::CachedImage(CachedImage&& other) noexcept {
  num_bytes = other.num_bytes;
  bitmap = std::move(other.bitmap);
  depth_map = std::move(other.depth_map);
  normal_map = std::move(other.normal_map);
}

CachedWorkspace::CachedImage& CachedWorkspace::CachedImage::operator=(
    CachedImage&& other) noexcept {
  if (this != &other) {
    num_bytes = other.num_bytes;
    bitmap = std::move(other.bitmap);
    depth_map = std::move(other.depth_map);
    normal_map = std::move(other.normal_map);
  }
  return *this;
}

CachedWorkspace::CachedWorkspace(const Options& options)
    : Workspace(options),
      cache_((size_t)(1024.0 * 1024.0 * 1024.0 * options.cache_size),
             [](const int) { return std::make_shared<CachedImage>(); }) {}

const Bitmap& CachedWorkspace::GetBitmap(const int image_idx) {
  return *GetBitmapLease(image_idx);
}

std::shared_ptr<const Bitmap> CachedWorkspace::GetBitmapLease(
    const int image_idx) {
  std::shared_ptr<CachedImage> cached_image;
  {
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    cached_image = cache_.Get(image_idx);
  }
  std::lock_guard<std::mutex> lock(cached_image->mutex);
  if (!cached_image->bitmap) {
    cached_image->bitmap = std::make_unique<Bitmap>();
    cached_image->bitmap->Read(GetBitmapPath(image_idx), options_.image_as_rgb);
    if (options_.max_image_size > 0) {
      cached_image->bitmap->Rescale(model_.images.at(image_idx).GetWidth(),
                                    model_.images.at(image_idx).GetHeight());
    }
    cached_image->num_bytes += cached_image->bitmap->NumBytes();
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    // Handle the case where another thread has already evicted the image.
    if (cache_.Exists(image_idx)) {
      cache_.UpdateNumBytes(image_idx);
    }
  }
  return {cached_image, cached_image->bitmap.get()};
}

const DepthMap& CachedWorkspace::GetDepthMap(const int image_idx) {
  return *GetDepthMapLease(image_idx);
}

std::shared_ptr<const DepthMap> CachedWorkspace::GetDepthMapLease(
    const int image_idx) {
  std::shared_ptr<CachedImage> cached_image;
  {
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    cached_image = cache_.Get(image_idx);
  }
  std::lock_guard<std::mutex> lock(cached_image->mutex);
  if (!cached_image->depth_map) {
    cached_image->depth_map = std::make_unique<DepthMap>();
    cached_image->depth_map->Read(GetDepthMapPath(image_idx));
    if (options_.max_image_size > 0) {
      cached_image->depth_map->Downsize(
          model_.images.at(image_idx).GetWidth(),
          model_.images.at(image_idx).GetHeight());
    }
    cached_image->num_bytes += cached_image->depth_map->GetNumBytes();
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    // Handle the case where another thread has already evicted the image.
    if (cache_.Exists(image_idx)) {
      cache_.UpdateNumBytes(image_idx);
    }
  }
  return {cached_image, cached_image->depth_map.get()};
}

const NormalMap& CachedWorkspace::GetNormalMap(const int image_idx) {
  return *GetNormalMapLease(image_idx);
}

std::shared_ptr<const NormalMap> CachedWorkspace::GetNormalMapLease(
    const int image_idx) {
  std::shared_ptr<CachedImage> cached_image;
  {
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    cached_image = cache_.Get(image_idx);
  }
  std::lock_guard<std::mutex> lock(cached_image->mutex);
  if (!cached_image->normal_map) {
    cached_image->normal_map = std::make_unique<NormalMap>();
    cached_image->normal_map->Read(GetNormalMapPath(image_idx));
    if (options_.max_image_size > 0) {
      cached_image->normal_map->Downsize(
          model_.images.at(image_idx).GetWidth(),
          model_.images.at(image_idx).GetHeight());
    }
    cached_image->num_bytes += cached_image->normal_map->GetNumBytes();
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    // Handle the case where another thread has already evicted the image.
    if (cache_.Exists(image_idx)) {
      cache_.UpdateNumBytes(image_idx);
    }
  }
  return {cached_image, cached_image->normal_map.get()};
}

bool CachedWorkspace::ValidateSourceNormalWithoutMaterialization(
    const int image_idx) {
  std::shared_ptr<CachedImage> cached_image;
  {
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    cached_image = cache_.Get(image_idx);
  }
  std::lock_guard<std::mutex> lock(cached_image->mutex);
  if (cached_image->normal_map) {
    return cached_image->normal_map->GetDepth() == 3;
  }
  return IsCompleteThreeChannelFloatMap(GetNormalMapPath(image_idx));
}

void ImportPMVSWorkspace(const Workspace& workspace,
                         const std::string& option_name) {
  const auto& workspace_path = workspace.GetOptions().workspace_path;
  const auto& stereo_folder = workspace.GetOptions().stereo_folder;

  CreateDirIfNotExists(workspace_path / stereo_folder);
  CreateDirIfNotExists(workspace_path / stereo_folder / "depth_maps");
  CreateDirIfNotExists(workspace_path / stereo_folder / "normal_maps");
  CreateDirIfNotExists(workspace_path / stereo_folder / "consistency_graphs");

  const auto option_lines = ReadTextFileLines(workspace_path / option_name);
  for (const auto& line : option_lines) {
    if (!StringStartsWith(line, "timages")) {
      continue;
    }

    const auto elems = StringSplit(line, " ");
    int num_images = std::stoull(elems[1]);

    std::vector<int> image_idxs;
    if (num_images == -1) {
      THROW_CHECK_EQ(elems.size(), 4);
      const int range_lower = std::stoull(elems[2]);
      const int range_upper = std::stoull(elems[3]);
      THROW_CHECK_LT(range_lower, range_upper);
      num_images = range_upper - range_lower;
      image_idxs.resize(num_images);
      std::iota(image_idxs.begin(), image_idxs.end(), range_lower);
    } else {
      THROW_CHECK_EQ(num_images + 2, elems.size());
      image_idxs.reserve(num_images);
      for (size_t i = 2; i < elems.size(); ++i) {
        const int image_idx = std::stoull(elems[i]);
        image_idxs.push_back(image_idx);
      }
    }

    std::vector<std::string> image_names;
    image_names.reserve(num_images);
    for (const auto image_idx : image_idxs) {
      const std::string image_name =
          workspace.GetModel().GetImageName(image_idx);
      image_names.push_back(image_name);
    }

    const auto& overlapping_images =
        workspace.GetModel().GetMaxOverlappingImagesFromPMVS();

    const auto patch_match_path =
        workspace_path / stereo_folder / "patch-match.cfg";
    const auto fusion_path = workspace_path / stereo_folder / "fusion.cfg";
    std::ofstream patch_match_file(patch_match_path, std::ios::trunc);
    std::ofstream fusion_file(fusion_path, std::ios::trunc);
    THROW_CHECK_FILE_OPEN(patch_match_file, patch_match_path);
    THROW_CHECK_FILE_OPEN(fusion_file, fusion_path);
    for (size_t i = 0; i < image_names.size(); ++i) {
      const auto& ref_image_name = image_names[i];
      patch_match_file << ref_image_name << '\n';
      if (overlapping_images.empty()) {
        patch_match_file << "__auto__, 20\n";
      } else {
        for (const int image_idx : overlapping_images[i]) {
          patch_match_file << workspace.GetModel().GetImageName(image_idx)
                           << ", ";
        }
        patch_match_file << '\n';
      }
      fusion_file << ref_image_name << '\n';
    }
  }
}

}  // namespace mvs
}  // namespace colmap
