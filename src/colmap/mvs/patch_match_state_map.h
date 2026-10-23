// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/gpu_mat.h"

#include <algorithm>
#include <limits>

namespace colmap {
namespace mvs {

// Private PatchMatch cost/probability owner, not a general tensor allocator.
// Its two legal spatial views share one allocation; only DEAD roles reorient.
class PatchMatchStateMap {
 public:
  PatchMatchStateMap(size_t width, size_t height, size_t sources)
      : original_width_(width), original_height_(height), depth_(sources) {
    THROW_CHECK_GT(width, 0);
    THROW_CHECK_GT(height, 0);
    THROW_CHECK_LE(width, std::numeric_limits<int>::max());
    THROW_CHECK_LE(height, std::numeric_limits<int>::max());
    const size_t a = CheckedBytes(Pitch(width), height, sources);
    const size_t b = CheckedBytes(Pitch(height), width, sources);
    capacity_ = std::max(a, b);
    if (capacity_ != 0)
      CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ptr_), capacity_));
    Reorient(width, height);
  }
  ~PatchMatchStateMap() { (void)cudaFree(ptr_); }
  PatchMatchStateMap(const PatchMatchStateMap&) = delete;
  PatchMatchStateMap& operator=(const PatchMatchStateMap&) = delete;

  size_t GetWidth() const { return width_; }
  size_t GetHeight() const { return height_; }
  size_t GetDepth() const { return depth_; }
  size_t GetPitch() const { return pitch_; }
  size_t GetCapacityBytes() const { return capacity_; }
  float* GetPtr() const { return ptr_; }
  GpuMatView<float> View() const {
    return {ptr_, pitch_, width_, height_, depth_};
  }

  void Reorient(size_t width, size_t height) {
    THROW_CHECK((width == original_width_ && height == original_height_) ||
                (width == original_height_ && height == original_width_));
    const size_t pitch = Pitch(width);
    THROW_CHECK_LE(CheckedBytes(pitch, height, depth_), capacity_);
    width_ = width;
    height_ = height;
    pitch_ = pitch;
  }

  Mat<float> CopyToMat() const {
    Mat<float> result(width_, height_, depth_);
    if (depth_ != 0)
      CUDA_SAFE_CALL(cudaMemcpy2D(result.GetPtr(),
                                  width_ * sizeof(float),
                                  ptr_,
                                  pitch_,
                                  width_ * sizeof(float),
                                  height_ * depth_,
                                  cudaMemcpyDeviceToHost));
    return result;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  void FillWithScalar(float value) {
    const dim3 block(32, 16);
    const dim3 grid((width_ - 1) / 32 + 1, (height_ - 1) / 16 + 1);
    internal::FillWithScalarKernel<float><<<grid, block>>>(View(), value);
    CUDA_SYNC_AND_CHECK();
  }

  void Rotate(PatchMatchStateMap* output) const {
    THROW_CHECK_NE(this, output);
    THROW_CHECK_EQ(output->width_, height_);
    THROW_CHECK_EQ(output->height_, width_);
    THROW_CHECK_EQ(output->depth_, depth_);
    for (size_t slice = 0; slice < depth_; ++slice) {
      CudaRotate(
          reinterpret_cast<float*>(reinterpret_cast<char*>(ptr_) +
                                   slice * pitch_ * height_),
          reinterpret_cast<float*>(reinterpret_cast<char*>(output->ptr_) +
                                   slice * output->pitch_ * output->height_),
          width_,
          height_,
          pitch_,
          output->pitch_);
    }
    CUDA_SYNC_AND_CHECK();
  }
#endif

 private:
  static size_t Pitch(size_t width) {
    const size_t max = std::numeric_limits<size_t>::max();
    THROW_CHECK_LE(width, (max - 511) / sizeof(float));
    const size_t pitch = (width * sizeof(float) + 511) / 512 * 512;
    // Production CudaRotate takes signed-int pitches.
    THROW_CHECK_LE(pitch, std::numeric_limits<int>::max());
    return pitch;
  }
  static size_t CheckedBytes(size_t pitch, size_t height, size_t depth) {
    const size_t max = std::numeric_limits<size_t>::max();
    THROW_CHECK_LE(height, max / pitch);
    const size_t plane = pitch * height;
    THROW_CHECK(depth == 0 || plane <= max / depth);
    return plane * depth;
  }
  const size_t original_width_, original_height_, depth_;
  float* ptr_ = nullptr;
  size_t capacity_ = 0, width_ = 0, height_ = 0, pitch_ = 0;
};

}  // namespace mvs
}  // namespace colmap
