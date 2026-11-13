// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/gpu_mat.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace colmap {
namespace mvs {

// Sweep-local beta checkpoints/stripe; the same allocation is dead scratch
// during state rotation. No checkpoint survives a sweep or changes a draw.
class PatchMatchHmmCheckpoint {
 public:
  PatchMatchHmmCheckpoint(size_t width,
                          size_t height,
                          size_t sources,
                          size_t rotation_element_bytes = sizeof(float))
      : original_width_(width), original_height_(height), sources_(sources) {
    capacity_ =
        RequiredCapacity(width, height, sources, rotation_element_bytes);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ptr_), capacity_));
  }
  ~PatchMatchHmmCheckpoint() {
    const auto status = cudaFree(ptr_);
    (void)CudaCleanupScope::ObserveCleanup("cudaFree",
                                           static_cast<int>(status));
  }
  PatchMatchHmmCheckpoint(const PatchMatchHmmCheckpoint&) = delete;
  PatchMatchHmmCheckpoint& operator=(const PatchMatchHmmCheckpoint&) = delete;

  static size_t RequiredCapacity(size_t width,
                                 size_t height,
                                 size_t sources,
                                 size_t rotation_element_bytes) {
    THROW_CHECK_GT(width, 0);
    THROW_CHECK_GT(height, 0);
    THROW_CHECK_LE(width, std::numeric_limits<int>::max());
    THROW_CHECK_LE(height, std::numeric_limits<int>::max());
    THROW_CHECK(rotation_element_bytes == sizeof(uint16_t) ||
                rotation_element_bytes == sizeof(float));
    return std::max({WorkspaceBytes(width, height, sources),
                     WorkspaceBytes(height, width, sources),
                     Multiply(Pitch(width, rotation_element_bytes), height),
                     Multiply(Pitch(height, rotation_element_bytes), width)});
  }

  float* GetPtr() const { return ptr_; }
  size_t GetCapacityBytes() const { return capacity_; }

  static size_t StripeRows(size_t height) {
    THROW_CHECK_GT(height, 0);
    size_t lo = 1, hi = height;
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      if (mid >= height / mid + (height % mid != 0))
        hi = mid;
      else
        lo = mid + 1;
    }
    return lo;
  }
  GpuMatView<float> Checkpoints(size_t width, size_t height) const {
    CheckOrientation(width, height);
    const size_t rows = StripeRows(height);
    return {ptr_, Pitch(width), width, (height - 1) / rows + 1, sources_};
  }
  GpuMatView<float> Stripe(size_t width, size_t height) const {
    const auto checkpoints = Checkpoints(width, height);
    const size_t offset = Multiply(
        Multiply(checkpoints.GetPitch(), checkpoints.GetHeight()), sources_);
    return {reinterpret_cast<float*>(reinterpret_cast<char*>(ptr_) + offset),
            Pitch(width),
            width,
            StripeRows(height),
            sources_};
  }

 private:
  static size_t Multiply(size_t a, size_t b) {
    THROW_CHECK(b == 0 || a <= std::numeric_limits<size_t>::max() / b);
    return a * b;
  }
  static size_t Pitch(size_t width, size_t element_bytes = sizeof(float)) {
    THROW_CHECK_LE(width,
                   (std::numeric_limits<size_t>::max() - 511) / element_bytes);
    const size_t pitch = (width * element_bytes + 511) / 512 * 512;
    THROW_CHECK_LE(pitch, std::numeric_limits<int>::max());
    return pitch;
  }
  static size_t WorkspaceBytes(size_t width, size_t height, size_t sources) {
    const size_t stripe = StripeRows(height);
    const size_t checkpoints = (height - 1) / stripe + 1;
    THROW_CHECK_LE(stripe, std::numeric_limits<size_t>::max() - checkpoints);
    return Multiply(Multiply(Pitch(width), stripe + checkpoints), sources);
  }
  void CheckOrientation(size_t width, size_t height) const {
    THROW_CHECK((width == original_width_ && height == original_height_) ||
                (width == original_height_ && height == original_width_));
  }
  size_t original_width_, original_height_, sources_, capacity_;
  float* ptr_ = nullptr;
};

}  // namespace mvs
}  // namespace colmap
