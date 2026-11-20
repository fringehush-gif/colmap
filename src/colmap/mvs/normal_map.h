// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/mat.h"
#include "colmap/sensor/bitmap.h"

namespace colmap {
namespace mvs {

// Normal map class that stores per-pixel normals as a MxNx3 image.
class NormalMap : public Mat<float> {
 public:
  NormalMap();
  NormalMap(size_t width, size_t height);
  explicit NormalMap(const Mat<float>& mat);

  // Intentionally hide Mat::Read to validate channels before replacing data.
  // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
  void Read(const std::filesystem::path& path);

  void Rescale(float factor);
  void Downsize(size_t max_width, size_t max_height);

  Bitmap ToBitmap() const;
};

}  // namespace mvs
}  // namespace colmap
