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

}  // namespace mvs
}  // namespace colmap
