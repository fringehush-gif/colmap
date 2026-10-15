// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/util/hash_containers.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace colmap {

class Bitmap;

// Optional, exact-path ingress at Bitmap::Read. Registration does not own a
// second pixel cache. The callback must fill an owned Bitmap; failures must not
// silently fall back to a different raster on disk. Keep the scope open until
// native consumers join. Reads admitted before Close retain callback lifetime.
class ScopedBitmapReader {
 public:
  using Callback =
      std::function<bool(const std::filesystem::path&, bool, bool, Bitmap*)>;
  static std::shared_ptr<ScopedBitmapReader> Create(
      const std::vector<std::filesystem::path>& paths,
      Callback callback,
      const std::vector<std::pair<int, int>>& image_sizes_wh = {});
  static std::shared_ptr<ScopedBitmapReader> Find(
      const std::filesystem::path& path);
  ~ScopedBitmapReader();
  void Close();
  bool Read(const std::filesystem::path& path,
            bool as_rgb,
            bool linearize_colorspace,
            Bitmap* target) const;
  // Immutable RGB8 dimensions permit metadata-only workspace admission.
  // Absence means the scope has no declared header, not permission to decode.
  std::optional<std::pair<int, int>> ImageSize(
      const std::filesystem::path& path) const;

 private:
  ScopedBitmapReader(std::vector<std::string> paths,
                     Callback callback,
                     FlatHashMap<std::string, std::pair<int, int>> image_sizes);
  const std::vector<std::string> paths_;
  const Callback callback_;
  const FlatHashMap<std::string, std::pair<int, int>> image_sizes_;
};

}  // namespace colmap
