// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/patch_match.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace colmap {
namespace mvs {

// Batch only: one workspace and immutable source-map generation. Inputs stay
// unchanged through Create; source IDs from different workspaces never alias.
// No cross-batch cache. Attach views before Run and join consumers before
// release. No GPU LUT or changed shader sampling interface.
class PatchMatchSparseSources {
 public:
  // Build/device/driver-entrypoint support, independent of batch profitability.
  static bool IsSupported();
  static size_t Budget(const std::vector<const PatchMatch::Problem*>& problems,
                       bool geometric);
  static std::shared_ptr<const PatchMatchSparseSources> Create(
      const std::vector<const PatchMatch::Problem*>& problems, bool geometric);
  ~PatchMatchSparseSources();
  bool Matches(const PatchMatch::Problem& problem, bool geometric) const;

 private:
  template <typename T>
  friend class PatchMatchSourceArray;
  struct Impl;
  explicit PatchMatchSparseSources(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Owns either the original dense helper or a texture view plus a bank token.
// Its raw GetObj ABI and logical dimensions match the original source owner.
template <typename T>
class PatchMatchSourceArray {
 public:
  using HostSlice = typename CudaArrayLayeredTexture<T>::HostSlice;
  static std::unique_ptr<PatchMatchSourceArray> FromHostArray(
      const cudaTextureDesc& desc,
      size_t width,
      size_t height,
      size_t depth,
      const T* data);
  static std::unique_ptr<PatchMatchSourceArray> FromHostSlices(
      const cudaTextureDesc& desc,
      size_t width,
      size_t height,
      const std::vector<HostSlice>& slices);
  static std::unique_ptr<PatchMatchSourceArray> FromSparse(
      std::shared_ptr<const PatchMatchSparseSources> bank, int reference);
  ~PatchMatchSourceArray();
  cudaTextureObject_t GetObj() const;
  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;
  bool IsSparse() const;

 private:
  struct Impl;
  explicit PatchMatchSourceArray(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

extern template class PatchMatchSourceArray<uint8_t>;
extern template class PatchMatchSourceArray<float>;

}  // namespace mvs
}  // namespace colmap
