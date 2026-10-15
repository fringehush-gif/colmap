// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/gpu_mat.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <limits>
#include <memory>
#include <vector>

namespace colmap {
namespace mvs {

template <typename T>
class CudaArrayLayeredTexture {
 public:
  struct HostSlice {
    const T* data;
    size_t width;
    size_t height;
    size_t pitch_bytes;
  };

  static std::unique_ptr<CudaArrayLayeredTexture<T>> FromHostSlices(
      const cudaTextureDesc& texture_desc,
      size_t width,
      size_t height,
      const std::vector<HostSlice>& slices);

  static std::unique_ptr<CudaArrayLayeredTexture<T>> FromGpuMat(
      const cudaTextureDesc& texture_desc, const GpuMat<T>& mat);
  static std::unique_ptr<CudaArrayLayeredTexture<T>> FromHostArray(
      const cudaTextureDesc& texture_desc,
      const size_t width,
      const size_t height,
      const size_t depth,
      const T* data);

  cudaTextureObject_t GetObj() const;

  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;

  CudaArrayLayeredTexture(const cudaTextureDesc& texture_desc,
                          const size_t width,
                          const size_t height,
                          const size_t depth);
  ~CudaArrayLayeredTexture();

 private:
  // Define class as non-copyable and non-movable.
  CudaArrayLayeredTexture(CudaArrayLayeredTexture const&) = delete;
  void operator=(CudaArrayLayeredTexture const& obj) = delete;
  CudaArrayLayeredTexture(CudaArrayLayeredTexture&&) = delete;

  const size_t width_;
  const size_t height_;
  const size_t depth_;

  cudaArray_t array_;
  const cudaTextureDesc texture_desc_;
  cudaResourceDesc resource_desc_;
  cudaTextureObject_t texture_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::unique_ptr<CudaArrayLayeredTexture<T>>
CudaArrayLayeredTexture<T>::FromGpuMat(const cudaTextureDesc& texture_desc,
                                       const GpuMat<T>& mat) {
  auto array = std::make_unique<CudaArrayLayeredTexture<T>>(
      texture_desc, mat.GetWidth(), mat.GetHeight(), mat.GetDepth());

  cudaMemcpy3DParms params;
  memset(&params, 0, sizeof(params));
  params.extent =
      make_cudaExtent(mat.GetWidth(), mat.GetHeight(), mat.GetDepth());
  params.kind = cudaMemcpyDeviceToDevice;
  params.srcPtr = make_cudaPitchedPtr(
      (void*)mat.GetPtr(), mat.GetPitch(), mat.GetWidth(), mat.GetHeight());
  params.dstArray = array->array_;
  CUDA_SAFE_CALL(cudaMemcpy3D(&params));

  return array;
}

template <typename T>
std::unique_ptr<CudaArrayLayeredTexture<T>>
CudaArrayLayeredTexture<T>::FromHostArray(const cudaTextureDesc& texture_desc,
                                          const size_t width,
                                          const size_t height,
                                          const size_t depth,
                                          const T* data) {
  auto array = std::make_unique<CudaArrayLayeredTexture<T>>(
      texture_desc, width, height, depth);

  cudaMemcpy3DParms params;
  memset(&params, 0, sizeof(params));
  params.extent = make_cudaExtent(width, height, depth);
  params.kind = cudaMemcpyHostToDevice;
  params.srcPtr =
      make_cudaPitchedPtr((void*)data, width * sizeof(T), width, height);
  params.dstArray = array->array_;
  CUDA_SAFE_CALL(cudaMemcpy3D(&params));

  return array;
}

template <typename T>
std::unique_ptr<CudaArrayLayeredTexture<T>>
CudaArrayLayeredTexture<T>::FromHostSlices(
    const cudaTextureDesc& texture_desc,
    const size_t width,
    const size_t height,
    const std::vector<HostSlice>& slices) {
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  THROW_CHECK(!slices.empty());
  const size_t max_bytes = std::numeric_limits<size_t>::max();
  THROW_CHECK_LE(width, max_bytes / sizeof(T));
  const size_t row_bytes = width * sizeof(T);
  THROW_CHECK_LE(height, max_bytes / row_bytes);
  for (const auto& slice : slices) {
    THROW_CHECK(slice.data != nullptr);
    THROW_CHECK_GT(slice.width, 0);
    THROW_CHECK_GT(slice.height, 0);
    THROW_CHECK_LE(slice.width, width);
    THROW_CHECK_LE(slice.height, height);
    const size_t slice_row_bytes = slice.width * sizeof(T);
    THROW_CHECK_GE(slice.pitch_bytes, slice_row_bytes);
    // Refuse a wrapped last-row offset before any CUDA allocation or copy.
    THROW_CHECK_LE(slice.height - 1,
                   (max_bytes - slice_row_bytes) / slice.pitch_bytes);
  }

  auto array = std::make_unique<CudaArrayLayeredTexture<T>>(
      texture_desc, width, height, slices.size());
  // Equal-sized slices need no padding or additional device allocation.
  // Mixed sizes reuse one device-zero plane rather than a full host stack.
  std::unique_ptr<GpuMat<T>> zero_plane;
  for (size_t layer = 0; layer < slices.size(); ++layer) {
    const auto& slice = slices[layer];
    if (slice.width != width || slice.height != height) {
      if (!zero_plane) {
        zero_plane = std::make_unique<GpuMat<T>>(width, height);
        THROW_CHECK_GT(zero_plane->GetPitch(), 0);
        THROW_CHECK_LE(height, max_bytes / zero_plane->GetPitch());
        // Clear the complete allocated pitched plane, including row padding.
        // cudaMemset is also provided by the existing CUDA-to-HIP shim.
        CUDA_SAFE_CALL(cudaMemset(
            zero_plane->GetPtr(), 0, zero_plane->GetPitch() * height));
      }
      cudaMemcpy3DParms clear = {};
      clear.srcPtr = make_cudaPitchedPtr(
          zero_plane->GetPtr(), zero_plane->GetPitch(), row_bytes, height);
      clear.dstArray = array->array_;
      clear.dstPos = make_cudaPos(0, 0, layer);
      clear.extent = make_cudaExtent(width, height, 1);
      clear.kind = cudaMemcpyDeviceToDevice;
      CUDA_SAFE_CALL(cudaMemcpy3D(&clear));
    }

    cudaMemcpy3DParms copy = {};
    copy.srcPtr = make_cudaPitchedPtr(const_cast<T*>(slice.data),
                                      slice.pitch_bytes,
                                      slice.width * sizeof(T),
                                      slice.height);
    copy.dstArray = array->array_;
    copy.dstPos = make_cudaPos(0, 0, layer);
    copy.extent = make_cudaExtent(slice.width, slice.height, 1);
    copy.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL(cudaMemcpy3D(&copy));
  }
  return array;
}

template <typename T>
CudaArrayLayeredTexture<T>::CudaArrayLayeredTexture(
    const cudaTextureDesc& texture_desc,
    const size_t width,
    const size_t height,
    const size_t depth)
    : texture_desc_(texture_desc),
      width_(width),
      height_(height),
      depth_(depth) {
  THROW_CHECK_GT(width_, 0);
  THROW_CHECK_GT(height_, 0);
  THROW_CHECK_GT(depth_, 0);

  cudaExtent extent = make_cudaExtent(width_, height_, depth_);
  const cudaChannelFormatDesc fmt = cudaCreateChannelDesc<T>();
  CUDA_SAFE_CALL(cudaMalloc3DArray(&array_, &fmt, extent, cudaArrayLayered));

  memset(&resource_desc_, 0, sizeof(resource_desc_));
  resource_desc_.resType = cudaResourceTypeArray;
  resource_desc_.res.array.array = array_;

  try {
    CUDA_SAFE_CALL(cudaCreateTextureObject(
        &texture_, &resource_desc_, &texture_desc_, nullptr));
  } catch (...) {
    // A failed constructor does not run the destructor. Preserve the original
    // texture-creation exception even if this best-effort cleanup fails.
    (void)cudaFreeArray(array_);
    throw;
  }
}

template <typename T>
CudaArrayLayeredTexture<T>::~CudaArrayLayeredTexture() {
  CUDA_SAFE_CALL(cudaDestroyTextureObject(texture_));
  CUDA_SAFE_CALL(cudaFreeArray(array_));
}

template <typename T>
cudaTextureObject_t CudaArrayLayeredTexture<T>::GetObj() const {
  return texture_;
}

template <typename T>
size_t CudaArrayLayeredTexture<T>::GetWidth() const {
  return width_;
}

template <typename T>
size_t CudaArrayLayeredTexture<T>::GetHeight() const {
  return height_;
}

template <typename T>
size_t CudaArrayLayeredTexture<T>::GetDepth() const {
  return depth_;
}

}  // namespace mvs
}  // namespace colmap
