// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/cuda_rotate.h"
#include "colmap/mvs/gpu_mat.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace colmap {
namespace mvs {

// Experimental representation of NCC cost, whose native writer bounds values
// to [0, 2]. Only the checkpoint-HMM path can use this owner: the ordinary
// probability-map rotation swaps float storage with cost storage.
struct PackedNccCostView {
  static constexpr int kStorageBits = 16;
  GpuMatView<uint16_t> codes;
  uint32_t* domain_error;

  __host__ __device__ size_t GetWidth() const { return codes.GetWidth(); }
  __host__ __device__ size_t GetHeight() const { return codes.GetHeight(); }
  __host__ __device__ size_t GetDepth() const { return codes.GetDepth(); }

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ float Get(size_t row, size_t col, size_t source) const {
    return static_cast<float>(codes.Get(row, col, source)) * (2.0f / 65535.0f);
  }

  __device__ void Set(size_t row, size_t col, size_t source, float cost) {
    if (!isfinite(cost) || cost < 0.0f || cost > 2.0f) {
      atomicExch(domain_error, 1u);
      return;
    }
    codes.Set(row,
              col,
              source,
              static_cast<uint16_t>(nearbyintf(cost * (65535.0f / 2.0f))));
  }
#endif
};

class PackedNccCostMap {
 public:
  PackedNccCostMap(size_t width, size_t height, size_t sources)
      : original_width_(width), original_height_(height), sources_(sources) {
    THROW_CHECK_GT(width, 0);
    THROW_CHECK_GT(height, 0);
    THROW_CHECK_GT(sources, 0);
    THROW_CHECK_LE(sources, std::numeric_limits<int>::max());
    THROW_CHECK_LE(width, std::numeric_limits<int>::max());
    THROW_CHECK_LE(height, std::numeric_limits<int>::max());
    capacity_ =
        std::max(Bytes(width, height, sources), Bytes(height, width, sources));
    THROW_CHECK_LE(capacity_,
                   std::numeric_limits<size_t>::max() - sizeof(uint32_t));
    try {
      CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ptr_), capacity_));
      CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&domain_error_),
                                sizeof(uint32_t)));
      CUDA_SAFE_CALL(cudaMemset(domain_error_, 0, sizeof(uint32_t)));
    } catch (...) {
      (void)cudaFree(domain_error_);
      (void)cudaFree(ptr_);
      throw;
    }
    Reorient(width, height);
  }

  ~PackedNccCostMap() {
    (void)cudaFree(domain_error_);
    (void)cudaFree(ptr_);
  }
  PackedNccCostMap(const PackedNccCostMap&) = delete;
  PackedNccCostMap& operator=(const PackedNccCostMap&) = delete;

  size_t GetWidth() const { return width_; }
  size_t GetHeight() const { return height_; }
  size_t GetDepth() const { return sources_; }
  size_t GetCapacityBytes() const { return capacity_ + sizeof(uint32_t); }
  PackedNccCostView View() const {
    return {{ptr_, pitch_, width_, height_, sources_}, domain_error_};
  }

  void CheckDomain() const {
    uint32_t domain_status = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&domain_status,
                              domain_error_,
                              sizeof(domain_status),
                              cudaMemcpyDeviceToHost));
    THROW_CHECK_EQ(domain_status, 0u) << "NCC cost outside finite [0,2]";
  }

  void Reorient(size_t width, size_t height) {
    THROW_CHECK((width == original_width_ && height == original_height_) ||
                (width == original_height_ && height == original_width_));
    const size_t pitch = Pitch(width);
    THROW_CHECK_LE(Bytes(width, height, sources_), capacity_);
    width_ = width;
    height_ = height;
    pitch_ = pitch;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // Old and new cost planes may overlap when pitches differ. Stage each
  // source slice into already-dead HMM checkpoint memory before replacing it.
  void RotateWithPlaneScratch(void* scratch, size_t scratch_bytes) {
    THROW_CHECK(scratch != nullptr);
    THROW_CHECK_NE(scratch, ptr_);
    const size_t new_width = height_, new_height = width_;
    const size_t new_pitch = Pitch(new_width);
    const size_t old_plane = CheckedMultiply(pitch_, height_);
    const size_t new_plane = CheckedMultiply(new_pitch, new_height);
    THROW_CHECK_LE(new_plane, scratch_bytes);
    // Reject interior aliases too: writing a rotated slice must not clobber
    // any live source plane or the sticky domain flag.
    const uintptr_t begin = reinterpret_cast<uintptr_t>(scratch);
    THROW_CHECK_LE(new_plane, std::numeric_limits<uintptr_t>::max() - begin);
    const uintptr_t end = begin + new_plane;
    const uintptr_t data = reinterpret_cast<uintptr_t>(ptr_);
    const uintptr_t flag = reinterpret_cast<uintptr_t>(domain_error_);
    THROW_CHECK_LE(capacity_, std::numeric_limits<uintptr_t>::max() - data);
    THROW_CHECK_LE(sizeof(uint32_t),
                   std::numeric_limits<uintptr_t>::max() - flag);
    THROW_CHECK(end <= data || begin >= data + capacity_);
    THROW_CHECK(end <= flag || begin >= flag + sizeof(uint32_t));
    THROW_CHECK_LE(Bytes(new_width, new_height, sources_), capacity_);
    for (size_t ordinal = 0; ordinal < sources_; ++ordinal) {
      const size_t source =
          new_plane > old_plane ? sources_ - 1 - ordinal : ordinal;
      const auto* input = reinterpret_cast<const uint16_t*>(
          reinterpret_cast<const char*>(ptr_) + source * old_plane);
      auto* output = reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(ptr_) +
                                                 source * new_plane);
      auto* temp = reinterpret_cast<uint16_t*>(scratch);
      CudaRotate(input, temp, width_, height_, pitch_, new_pitch);
      CUDA_SAFE_CALL(cudaMemcpy2D(output,
                                  new_pitch,
                                  temp,
                                  new_pitch,
                                  new_width * sizeof(uint16_t),
                                  new_height,
                                  cudaMemcpyDeviceToDevice));
    }
    CUDA_SYNC_AND_CHECK();
    Reorient(new_width, new_height);
  }
#endif

 private:
  static size_t CheckedMultiply(size_t a, size_t b) {
    THROW_CHECK(b == 0 || a <= std::numeric_limits<size_t>::max() / b);
    return a * b;
  }
  static size_t Pitch(size_t width) {
    THROW_CHECK_LE(
        width, (std::numeric_limits<size_t>::max() - 511) / sizeof(uint16_t));
    const size_t pitch = (width * sizeof(uint16_t) + 511) / 512 * 512;
    THROW_CHECK_LE(pitch, std::numeric_limits<int>::max());
    return pitch;
  }
  static size_t Bytes(size_t width, size_t height, size_t sources) {
    return CheckedMultiply(CheckedMultiply(Pitch(width), height), sources);
  }

  const size_t original_width_, original_height_, sources_;
  uint16_t* ptr_ = nullptr;
  uint32_t* domain_error_ = nullptr;
  size_t capacity_ = 0;
  size_t width_ = 0, height_ = 0, pitch_ = 0;
};

// Fixed24 representation of NCC cost, whose native writer bounds values to
// [0, 2].
// Only checkpoint HMM can use it: ordinary rotation swaps float cost storage.
struct Fixed24NccCostView {
  static constexpr int kStorageBits = 24;
  uint16_t* codes;
  uint8_t* low_codes;
  size_t pixel_pitch, width, height, depth;
  uint32_t* domain_error;

  __host__ __device__ size_t GetWidth() const { return width; }
  __host__ __device__ size_t GetHeight() const { return height; }
  __host__ __device__ size_t GetDepth() const { return depth; }

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ float Get(size_t row, size_t col, size_t source) const {
    const size_t index = (source * height + row) * pixel_pitch + col;
    const uint32_t code =
        (static_cast<uint32_t>(codes[index]) << 8) | low_codes[index];
    // Promote the reserved top code to 2^24; other codes use the Q23 grid.
    return static_cast<float>(code + static_cast<uint32_t>(code == 0xffffffu)) *
           0x1p-23f;
  }

  __device__ void Set(size_t row, size_t col, size_t source, float cost) {
    const uint32_t bits = __float_as_uint(cost);
    // Accept either zero, but reject negative subnormals even under FTZ.
    if (bits > 0x40000000u && bits != 0x80000000u) {
      atomicExch(domain_error, 1u);
      return;
    }
    const uint32_t code =
        min(static_cast<uint32_t>(nearbyintf(cost * 0x1p23f)), 0xffffffu);
    const size_t index = (source * height + row) * pixel_pitch + col;
    codes[index] = static_cast<uint16_t>(code >> 8);
    low_codes[index] = static_cast<uint8_t>(code);
  }
#endif
};

class Fixed24NccCostMap {
 public:
  Fixed24NccCostMap(size_t width, size_t height, size_t sources)
      : original_width_(width), original_height_(height), sources_(sources) {
    THROW_CHECK_GT(width, 0);
    THROW_CHECK_GT(height, 0);
    THROW_CHECK_GT(sources, 0);
    THROW_CHECK_LE(sources, std::numeric_limits<int>::max());
    THROW_CHECK_LE(width, std::numeric_limits<int>::max());
    THROW_CHECK_LE(height, std::numeric_limits<int>::max());
    // Shared pixel strides keep high16/low8 padding proportional.
    capacity_pixels_ = std::max(Pixels(width, height, sources),
                                Pixels(height, width, sources));
    const size_t capacity = CheckedMultiply(capacity_pixels_, 3);
    THROW_CHECK_LE(capacity,
                   std::numeric_limits<size_t>::max() - sizeof(uint32_t));
    try {
      CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ptr_), capacity));
      CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&domain_error_),
                                sizeof(uint32_t)));
      CUDA_SAFE_CALL(cudaMemset(domain_error_, 0, sizeof(uint32_t)));
      Reorient(width, height);
    } catch (...) {
      (void)cudaFree(domain_error_);
      (void)cudaFree(ptr_);
      throw;
    }
  }

  ~Fixed24NccCostMap() {
    (void)cudaFree(domain_error_);
    (void)cudaFree(ptr_);
  }
  Fixed24NccCostMap(const Fixed24NccCostMap&) = delete;
  Fixed24NccCostMap& operator=(const Fixed24NccCostMap&) = delete;

  size_t GetWidth() const { return width_; }
  size_t GetHeight() const { return height_; }
  size_t GetDepth() const { return sources_; }
  size_t GetCapacityBytes() const {
    return 3 * capacity_pixels_ + sizeof(uint32_t);
  }
  Fixed24NccCostView View() const {
    return {
        ptr_, LowPtr(), pixel_pitch_, width_, height_, sources_, domain_error_};
  }

  void CheckDomain() const {
    uint32_t domain_status = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&domain_status,
                              domain_error_,
                              sizeof(domain_status),
                              cudaMemcpyDeviceToHost));
    THROW_CHECK_EQ(domain_status, 0u) << "NCC cost outside finite [0,2]";
  }

  void Reorient(size_t width, size_t height) {
    THROW_CHECK((width == original_width_ && height == original_height_) ||
                (width == original_height_ && height == original_width_));
    const size_t pixel_pitch = PixelPitch(width);
    THROW_CHECK_LE(Pixels(width, height, sources_), capacity_pixels_);
    width_ = width;
    height_ = height;
    pixel_pitch_ = pixel_pitch;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // Old and new cost planes may overlap when pitches differ. Stage each
  // source slice into already-dead HMM checkpoint memory before replacing it.
  void RotateWithPlaneScratch(void* scratch, size_t scratch_bytes) {
    THROW_CHECK(scratch != nullptr);
    const size_t new_width = height_, new_height = width_;
    const size_t new_pixel_pitch = PixelPitch(new_width);
    const size_t new_pitch = 2 * new_pixel_pitch;
    const size_t scratch_required = CheckedMultiply(new_pitch, new_height);
    THROW_CHECK_LE(scratch_required, scratch_bytes);
    THROW_CHECK_LE(Pixels(new_width, new_height, sources_), capacity_pixels_);
    // Reject interior aliases too: writing a rotated slice must not clobber
    // any live source plane or the sticky domain flag.
    const uintptr_t begin = reinterpret_cast<uintptr_t>(scratch);
    THROW_CHECK_EQ(begin % alignof(uint16_t), 0);
    THROW_CHECK_LE(scratch_required,
                   std::numeric_limits<uintptr_t>::max() - begin);
    const uintptr_t end = begin + scratch_required;
    const uintptr_t data = reinterpret_cast<uintptr_t>(ptr_);
    const uintptr_t flag = reinterpret_cast<uintptr_t>(domain_error_);
    const size_t capacity = 3 * capacity_pixels_;
    THROW_CHECK_LE(capacity, std::numeric_limits<uintptr_t>::max() - data);
    THROW_CHECK_LE(sizeof(uint32_t),
                   std::numeric_limits<uintptr_t>::max() - flag);
    THROW_CHECK(end <= data || begin >= data + capacity);
    THROW_CHECK(end <= flag || begin >= flag + sizeof(uint32_t));
    // Validate both regions before either is written. The common pixel stride
    // gives both planes the same traversal direction and distinct byte pitches.
    RotatePlane(ptr_, 2 * pixel_pitch_, new_pitch, scratch);
    RotatePlane(LowPtr(), pixel_pitch_, new_pixel_pitch, scratch);
    CUDA_SYNC_AND_CHECK();
    Reorient(new_width, new_height);
  }
#endif

 private:
  uint8_t* LowPtr() const {
    return reinterpret_cast<uint8_t*>(ptr_) + 2 * capacity_pixels_;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename T>
  void RotatePlane(T* data, size_t old_pitch, size_t new_pitch, void* scratch) {
    const size_t old_plane = CheckedMultiply(old_pitch, height_);
    const size_t new_plane = CheckedMultiply(new_pitch, width_);
    for (size_t ordinal = 0; ordinal < sources_; ++ordinal) {
      const size_t source =
          new_plane > old_plane ? sources_ - 1 - ordinal : ordinal;
      const auto* input = reinterpret_cast<const T*>(
          reinterpret_cast<const char*>(data) + source * old_plane);
      auto* output = reinterpret_cast<T*>(reinterpret_cast<char*>(data) +
                                          source * new_plane);
      auto* temp = reinterpret_cast<T*>(scratch);
      CudaRotate(input, temp, width_, height_, old_pitch, new_pitch);
      CUDA_SAFE_CALL(cudaMemcpy2D(output,
                                  new_pitch,
                                  temp,
                                  new_pitch,
                                  height_ * sizeof(T),
                                  width_,
                                  cudaMemcpyDeviceToDevice));
    }
  }
#endif

  static size_t CheckedMultiply(size_t a, size_t b) {
    THROW_CHECK(b == 0 || a <= std::numeric_limits<size_t>::max() / b);
    return a * b;
  }
  static size_t PixelPitch(size_t width) {
    THROW_CHECK_LE(width, std::numeric_limits<size_t>::max() - 127);
    const size_t pitch = (width + 127) / 128 * 128;
    THROW_CHECK_LE(pitch, std::numeric_limits<int>::max() / 2);
    return pitch;
  }
  static size_t Pixels(size_t width, size_t height, size_t sources) {
    return CheckedMultiply(CheckedMultiply(PixelPitch(width), height), sources);
  }

  const size_t original_width_, original_height_, sources_;
  uint16_t* ptr_ = nullptr;
  uint32_t* domain_error_ = nullptr;
  size_t capacity_pixels_ = 0;
  size_t width_ = 0, height_ = 0, pixel_pitch_ = 0;
};

}  // namespace mvs
}  // namespace colmap
