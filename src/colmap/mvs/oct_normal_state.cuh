// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/cuda_rotate.h"
#include "colmap/mvs/gpu_mat.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

#if defined(__CUDACC__)
#include "colmap/mvs/oct_magnitude.cuh"
#endif

namespace colmap {
namespace mvs {

static_assert(sizeof(unsigned int) == 4,
              "oct40 sticky error must occupy four bytes");

struct OctNormalView;

#if defined(__CUDACC__)
using NormalCode = oct_magnitude::Encoded;

// A thread owns both fields of a pixel. Callers provide the same ordering and
// lifetime guarantees as GpuMatView; the two planes are not an atomic tuple.
struct OctNormalView {
  oct_revisit::Code* direction;
  uint8_t* magnitude;
  size_t pixel_pitch;
  size_t width;
  size_t height;
  unsigned int* error;
  unsigned int frame;

  __host__ __device__ size_t GetWidth() const { return width; }
  __host__ __device__ size_t GetHeight() const { return height; }
  __host__ __device__ size_t GetDepth() const { return 3; }

  __device__ void Signal(unsigned int bits) const { atomicOr(error, bits); }
  __device__ bool InBounds(size_t row, size_t col) const {
    if (row < height && col < width) return true;
    Signal(1u << 8);
    return false;
  }
  __device__ static bool WellFormed(oct_magnitude::Encoded code) {
    const bool x = code.direction.x == oct_revisit::kInvalid;
    const bool y = code.direction.y == oct_revisit::kInvalid;
    return code.magnitude == oct_magnitude::kInvalidMagnitude ? x && y
                                                              : !x && !y;
  }
  __device__ oct_magnitude::Encoded ReadCode(size_t row, size_t col) const {
    if (!InBounds(row, col)) return oct_magnitude::Invalid();
    const size_t index = row * pixel_pitch + col;
    const oct_magnitude::Encoded code{direction[index], magnitude[index]};
    if (WellFormed(code)) return code;
    Signal(1u << 9);
    return oct_magnitude::Invalid();
  }
  __device__ void StoreCode(size_t row,
                            size_t col,
                            oct_magnitude::Encoded code) const {
    if (!InBounds(row, col)) return;
    if (!WellFormed(code)) {
      Signal(1u << 9);
      code = oct_magnitude::Invalid();
    }
    const size_t index = row * pixel_pitch + col;
    direction[index] = code.direction;
    magnitude[index] = code.magnitude;
  }
  __device__ void DecodeCode(oct_magnitude::Encoded code, float* normal) const {
    float3 value;
    if (!oct_magnitude::Decode(code, value)) Signal(1u << 9);
    // Decode canonically first. Reordering decoder operations by rotating
    // oct coordinates would change the associated FP32 normal/cached cost.
    switch (frame & 3u) {
      case 0:
        normal[0] = value.x;
        normal[1] = value.y;
        break;
      case 1:
        normal[0] = value.y;
        normal[1] = -value.x;
        break;
      case 2:
        normal[0] = -value.x;
        normal[1] = -value.y;
        break;
      default:
        normal[0] = -value.y;
        normal[1] = value.x;
        break;
    }
    normal[2] = value.z;
  }
  __device__ void GetSlice(size_t row, size_t col, float* normal) const {
    DecodeCode(ReadCode(row, col), normal);
  }
  __device__ float Get(size_t row, size_t col, size_t axis = 0) const {
    if (axis >= 3) {
      Signal(1u << 10);
      return 0;
    }
    float normal[3];
    GetSlice(row, col, normal);
    return normal[axis];
  }
  __device__ oct_magnitude::Encoded Canonicalize(float* normal) const {
    float3 canonical;
    switch (frame & 3u) {
      case 0:
        canonical = make_float3(normal[0], normal[1], normal[2]);
        break;
      case 1:
        canonical = make_float3(-normal[1], normal[0], normal[2]);
        break;
      case 2:
        canonical = make_float3(-normal[0], -normal[1], normal[2]);
        break;
      default:
        canonical = make_float3(normal[1], -normal[0], normal[2]);
        break;
    }
    oct_magnitude::Encoded code;
    // This owner admits a two-step final norm
    // error. Keep the reusable codec's stricter default policy unchanged.
    const auto status = oct_magnitude::Encode<2>(canonical, code, 0x1p-23f);
    if (status != oct_magnitude::Status::kOk &&
        status != oct_magnitude::Status::kZero)
      Signal(1u << static_cast<unsigned int>(status));
    // Failures leave canonical zero and set the sticky refusal. The host must
    // CheckDomain before consuming results; there is no silent FP32 fallback.
    DecodeCode(code, normal);
    return code;
  }
  // Initialization only. A selected hypothesis must instead retain and store
  // its existing code, not encode a decoded winner for a second time.
  __device__ void SetSlice(size_t row, size_t col, const float* normal) const {
    float values[3] = {normal[0], normal[1], normal[2]};
    StoreCode(row, col, Canonicalize(values));
  }
  // The native component setter is used only for filter invalidation. It is
  // deliberately not a general component update for a coupled codec value.
  __device__ void Set(size_t row, size_t col, size_t axis, float value) const {
    if (!InBounds(row, col)) return;
    if (axis >= 3 || value != 0.0f) {
      Signal(1u << 10);
      return;
    }
    if (axis == 0) StoreCode(row, col, oct_magnitude::Invalid());
  }
};

namespace internal {

static __global__ void OctNormalUpload(GpuMatView<float> input,
                                       OctNormalView output,
                                       size_t row_offset) {
  const size_t count = input.width * input.height;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += stride) {
    const size_t row = index / output.width, col = index % output.width;
    float normal[3];
    input.GetSlice(row, col, normal);
    output.SetSlice(row_offset + row, col, normal);
  }
}

static __global__ void OctNormalDownload(OctNormalView input,
                                         GpuMatView<float> output,
                                         size_t row_offset) {
  const size_t count = output.width * output.height;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += stride) {
    const size_t row = index / input.width, col = index % input.width;
    float normal[3];
    input.GetSlice(row_offset + row, col, normal);
    output.SetSlice(row, col, normal);
  }
}

}  // namespace internal
#endif  // __CUDACC__

// Separate direction/magnitude planes share
// one allocation and one pixel pitch rounded to 128 pixels. No retained FP32
// owner or hidden normalization/fallback is kept here.
class OctNormalMap {
 public:
  OctNormalMap(size_t width, size_t height, bool reserve_rotation = false)
      : width_(width), height_(height) {
    THROW_CHECK_GT(width, 0);
    THROW_CHECK_GT(height, 0);
    THROW_CHECK_LE(width, static_cast<size_t>(std::numeric_limits<int>::max()));
    THROW_CHECK_LE(height,
                   static_cast<size_t>(std::numeric_limits<int>::max()));
    const size_t max = std::numeric_limits<size_t>::max();
    THROW_CHECK_LE(width, max - 127);
    pixel_pitch_ = (width + 127) / 128 * 128;
    // Production CudaRotate accepts signed-int byte pitches.
    THROW_CHECK_LE(pixel_pitch_,
                   static_cast<size_t>(std::numeric_limits<int>::max()) / 4);
    THROW_CHECK_LE(height, (max - sizeof(unsigned int)) / 5 / pixel_pitch_);
    plane_pixels_ = pixel_pitch_ * height;
    if (reserve_rotation) {
      THROW_CHECK_LE(height, max - 127);
      const size_t rotated_pitch = (height + 127) / 128 * 128;
      THROW_CHECK_LE(rotated_pitch,
                     static_cast<size_t>(std::numeric_limits<int>::max()) / 4);
      THROW_CHECK_LE(width, (max - sizeof(unsigned int)) / 5 / rotated_pitch);
      plane_pixels_ = std::max(plane_pixels_, rotated_pitch * width);
    }
    payload_bytes_ = 5 * plane_pixels_;
    try {
      CUDA_SAFE_CALL(cudaMalloc(&data_, payload_bytes_));
      CUDA_SAFE_CALL(
          cudaMalloc(reinterpret_cast<void**>(&error_), sizeof(unsigned int)));
      CUDA_SAFE_CALL(cudaMemset(error_, 0, sizeof(unsigned int)));
    } catch (...) {
      Release();
      throw;
    }
    // Like GpuMat, pixels must be initialized before their first read.
  }
  ~OctNormalMap() { Release(); }
  OctNormalMap(const OctNormalMap&) = delete;
  OctNormalMap& operator=(const OctNormalMap&) = delete;
  OctNormalMap(OctNormalMap&&) = delete;
  OctNormalMap& operator=(OctNormalMap&&) = delete;

  size_t GetWidth() const { return width_; }
  size_t GetHeight() const { return height_; }
  size_t GetDepth() const { return 3; }
  size_t GetPixelPitch() const { return pixel_pitch_; }
  size_t GetCapacityBytes() const {
    return payload_bytes_ + sizeof(unsigned int);
  }
  unsigned int GetFrame() const { return frame_; }

  bool CanStore(size_t width, size_t height) const {
    if (width == 0 || height == 0 ||
        width > size_t(std::numeric_limits<int>::max()) ||
        height > size_t(std::numeric_limits<int>::max()))
      return false;
    const size_t pitch = (width + 127) / 128 * 128;
    return pitch <= size_t(std::numeric_limits<int>::max()) / 4 &&
           pitch <= plane_pixels_ / height;
  }
  void SetShape(size_t width, size_t height) {
    THROW_CHECK(CanStore(width, height));
    width_ = width;
    height_ = height;
    pixel_pitch_ = (width + 127) / 128 * 128;
  }

  static size_t TransferRows(size_t width, size_t height) {
    THROW_CHECK_GT(width, 0);
    THROW_CHECK_GT(height, 0);
    THROW_CHECK_LE(width,
                   std::numeric_limits<size_t>::max() / sizeof(float) / 3);
    constexpr size_t kTransferBudget = 4 * 1024 * 1024;
    const size_t row_bytes = width * sizeof(float) * 3;
    return std::min(height, std::max(size_t{1}, kTransferBudget / row_bytes));
  }
  static size_t TransferCapacityBytes(size_t width, size_t height) {
    return TransferRows(width, height) * width * sizeof(float) * 3;
  }

  void CheckDomain() const {
    unsigned int failure = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(&failure, error_, sizeof(failure), cudaMemcpyDeviceToHost));
    THROW_CHECK_EQ(failure, 0u) << "Oct40 normal domain refusal (sticky bits)";
  }

#if defined(__CUDACC__)
  OctNormalView View() const {
    return {static_cast<oct_revisit::Code*>(data_),
            static_cast<uint8_t*>(data_) + 4 * plane_pixels_,
            pixel_pitch_,
            width_,
            height_,
            error_,
            frame_};
  }
  // CPU input follows GpuMat's planar xyz layout, with byte pitch per row.
  // The temporary upload allocation is released before returning.
  void CopyFromCPU(const float* data, size_t byte_pitch) {
    CheckDomain();
    THROW_CHECK(data != nullptr);
    THROW_CHECK_GE(byte_pitch, width_ * sizeof(float));
    THROW_CHECK_LE(byte_pitch,
                   std::numeric_limits<size_t>::max() / height_ / 3);
    const size_t capacity_rows = TransferRows(width_, height_);
    auto temporary = AllocateTransfer(TransferCapacityBytes(width_, height_));
    for (size_t offset = 0; offset < height_; offset += capacity_rows) {
      const size_t rows = std::min(capacity_rows, height_ - offset);
      const GpuMatView<float> tile{
          temporary.get(), width_ * sizeof(float), width_, rows, 3};
      for (size_t axis = 0; axis < 3; ++axis) {
        const auto* source = reinterpret_cast<const char*>(data) +
                             (axis * height_ + offset) * byte_pitch;
        CUDA_SAFE_CALL(cudaMemcpy2D(temporary.get() + axis * rows * width_,
                                    tile.pitch,
                                    source,
                                    byte_pitch,
                                    tile.pitch,
                                    rows,
                                    cudaMemcpyHostToDevice));
      }
      internal::OctNormalUpload<<<TransferBlocks(rows), 128>>>(
          tile, View(), offset);
      CUDA_SYNC_AND_CHECK();
    }
    CheckDomain();
  }
  Mat<float> CopyToMat() const {
    CheckDomain();
    THROW_CHECK_LE(
        width_,
        std::numeric_limits<size_t>::max() / height_ / 3 / sizeof(float));
    Mat<float> result(width_, height_, 3);
    const size_t capacity_rows = TransferRows(width_, height_);
    auto temporary = AllocateTransfer(TransferCapacityBytes(width_, height_));
    for (size_t offset = 0; offset < height_; offset += capacity_rows) {
      const size_t rows = std::min(capacity_rows, height_ - offset);
      const GpuMatView<float> tile{
          temporary.get(), width_ * sizeof(float), width_, rows, 3};
      internal::OctNormalDownload<<<TransferBlocks(rows), 128>>>(
          View(), tile, offset);
      CUDA_SYNC_AND_CHECK();
      for (size_t axis = 0; axis < 3; ++axis) {
        float* target = result.GetPtr() + (axis * height_ + offset) * width_;
        CUDA_SAFE_CALL(cudaMemcpy2D(target,
                                    tile.pitch,
                                    temporary.get() + axis * rows * width_,
                                    tile.pitch,
                                    tile.pitch,
                                    rows,
                                    cudaMemcpyDeviceToHost));
      }
    }
    CheckDomain();
    return result;
  }
  void RotateTo(OctNormalMap& output) const {
    THROW_CHECK_NE(this, &output);
    CheckDomain();
    output.CheckDomain();
    // The pinned ROT2 CudaRotate uses a two-dimensional 32-pixel tile grid.
    // Check its actual device
    // limits instead of accepting an invalid launch for oversized shapes.
    int device = 0;
    cudaDeviceProp properties;
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device));
    THROW_CHECK_LE((height_ + 31) / 32,
                   static_cast<size_t>(properties.maxGridSize[1]));
    THROW_CHECK_LE((width_ + 31) / 32,
                   static_cast<size_t>(properties.maxGridSize[0]));
    output.SetShape(height_, width_);
    CudaRotate(static_cast<const oct_revisit::Code*>(data_),
               static_cast<oct_revisit::Code*>(output.data_),
               static_cast<int>(width_),
               static_cast<int>(height_),
               static_cast<int>(4 * pixel_pitch_),
               static_cast<int>(4 * output.pixel_pitch_));
    CudaRotate(static_cast<const uint8_t*>(data_) + 4 * plane_pixels_,
               static_cast<uint8_t*>(output.data_) + 4 * output.plane_pixels_,
               static_cast<int>(width_),
               static_cast<int>(height_),
               static_cast<int>(pixel_pitch_),
               static_cast<int>(output.pixel_pitch_));
    CUDA_SYNC_AND_CHECK();
    output.frame_ = (frame_ + 1) & 3u;
    // Raw plane copies cannot change either prechecked domain status.
  }
#else
  OctNormalView View() const;
  void CopyFromCPU(const float* data, size_t byte_pitch);
  Mat<float> CopyToMat() const;
  void RotateTo(OctNormalMap& output) const;
#endif

 private:
  struct TransferDeleter {
    void operator()(float* pointer) const noexcept { (void)cudaFree(pointer); }
  };
  using TransferBuffer = std::unique_ptr<float, TransferDeleter>;
  static TransferBuffer AllocateTransfer(size_t bytes) {
    float* pointer = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes));
    return TransferBuffer(pointer);
  }
  unsigned int TransferBlocks(size_t rows) const {
    return static_cast<unsigned int>(
        std::min(size_t(65535), (width_ * rows - 1) / 128 + 1));
  }
  void Release() noexcept {
    if (error_ != nullptr) {
      const auto status = cudaFree(error_);
      (void)status;
      error_ = nullptr;
    }
    if (data_ != nullptr) {
      const auto status = cudaFree(data_);
      (void)status;
      data_ = nullptr;
    }
  }
  size_t width_, height_;
  size_t pixel_pitch_ = 0, plane_pixels_ = 0, payload_bytes_ = 0;
  void* data_ = nullptr;
  unsigned int* error_ = nullptr;
  unsigned int frame_ = 0;
};

#if defined(__CUDACC__)
// The representation is uniform across the entire launch.  The code and
// frame of a packed winner travel together; FP32 paths retain the input value.
struct ActiveNormalView {
  GpuMatView<float> fp32;
  OctNormalView oct;
  bool packed;

  __device__ void GetSlice(size_t row, size_t col, float* normal) const {
    if (packed)
      oct.GetSlice(row, col, normal);
    else
      fp32.GetSlice(row, col, normal);
  }
  __device__ float Get(size_t row, size_t col, size_t axis) const {
    return packed ? oct.Get(row, col, axis) : fp32.Get(row, col, axis);
  }
  __device__ void SetSlice(size_t row, size_t col, const float* normal) {
    if (packed)
      oct.SetSlice(row, col, normal);
    else
      fp32.SetSlice(row, col, normal);
  }
  __device__ void Set(size_t row, size_t col, size_t axis, float value) {
    if (packed)
      oct.Set(row, col, axis, value);
    else
      fp32.Set(row, col, axis, value);
  }
};

__device__ inline void LoadNormal(ActiveNormalView view,
                                  size_t row,
                                  size_t col,
                                  float* normal,
                                  NormalCode& code) {
  if (view.packed) {
    code = view.oct.ReadCode(row, col);
    view.oct.DecodeCode(code, normal);
  } else {
    view.fp32.GetSlice(row, col, normal);
  }
}
__device__ inline void PrepareNormal(ActiveNormalView view,
                                     float* normal,
                                     NormalCode& code) {
  if (view.packed) code = view.oct.Canonicalize(normal);
}
__device__ inline void StoreNormal(ActiveNormalView view,
                                   size_t row,
                                   size_t col,
                                   const float* normal,
                                   NormalCode code) {
  if (view.packed)
    view.oct.StoreCode(row, col, code);
  else
    view.fp32.SetSlice(row, col, normal);
}

#endif

}  // namespace mvs
}  // namespace colmap
