// SPDX-License-Identifier: BSD-3-Clause
// Persistent PatchMatch selection-probability storage for CUDA checkpoint mode.
#pragma once
#include "colmap/mvs/gpu_mat.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace colmap {
namespace mvs {

// Kernel math stays float. Only a committed persistent probability is rounded.
struct ProbabilityUnorm16View {
  // Fixed absolute roundoff budget at unit probability scale (4*FLT_EPSILON).
  // This is not permission to clamp material/nonfinite invalid states.
  static constexpr float kEndpointRoundoff = 0x1p-21f;
  uint16_t* ptr;
  size_t pitch, width, height, depth;
  uint32_t* invalid;
  __host__ __device__ size_t GetWidth() const { return width; }
  __host__ __device__ size_t GetHeight() const { return height; }
  __host__ __device__ size_t GetDepth() const { return depth; }
#if defined(__CUDACC__)
  __device__ float Get(size_t row, size_t col, size_t layer = 0) const {
    const auto* line = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const char*>(ptr) + (layer * height + row) * pitch);
    return __fmul_rn(static_cast<float>(line[col]), 0x1.0001p-16f);
  }
  __device__ void Set(size_t row, size_t col, size_t layer, float value) const {
    auto* line = reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(ptr) +
                                             (layer * height + row) * pitch);
    if (!(value >= -kEndpointRoundoff && value <= 1.0f + kEndpointRoundoff)) {
      // Includes NaN/Inf and excursions beyond the fixed roundoff budget.
      atomicOr(invalid, 1u);
      line[col] = 0;
      return;
    }
    // Clamp only the accepted endpoint roundoff band. Endpoints decode exactly,
    // so this error is <=2^-21, below the unchanged interior quantization
    // bound.
    const float bounded = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    line[col] =
        static_cast<uint16_t>(__float2uint_rn(__fmul_rn(bounded, 65535.0f)));
  }
#endif
};

#if defined(__CUDACC__)
namespace probability_unorm16 {
static __global__ void Fill(ProbabilityUnorm16View view, float value) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= view.height || col >= view.width) return;
  for (size_t layer = 0; layer < view.depth; ++layer)
    view.Set(row, col, layer, value);
}

}  // namespace probability_unorm16
#endif

class PatchMatchProbabilityUnorm16 {
 public:
  PatchMatchProbabilityUnorm16(size_t width, size_t height, size_t sources)
      : original_width_(width), original_height_(height), depth_(sources) {
    THROW_CHECK_GT(width, 0);
    THROW_CHECK_GT(height, 0);
    THROW_CHECK_LE(width, std::numeric_limits<int>::max());
    THROW_CHECK_LE(height, std::numeric_limits<int>::max());
    data_bytes_ = std::max(Bytes(Pitch(width), height, sources),
                           Bytes(Pitch(height), width, sources));
    THROW_CHECK_LE(data_bytes_,
                   std::numeric_limits<size_t>::max() - sizeof(uint32_t));
    capacity_ = data_bytes_ + sizeof(uint32_t);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ptr_), capacity_));
    invalid_ = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_) +
                                           data_bytes_);
    try {
      CUDA_SAFE_CALL(cudaMemset(invalid_, 0, sizeof(uint32_t)));
      Reorient(width, height);
    } catch (...) {
      (void)cudaFree(ptr_);
      ptr_ = nullptr;
      throw;
    }
  }
  ~PatchMatchProbabilityUnorm16() { (void)cudaFree(ptr_); }
  PatchMatchProbabilityUnorm16(const PatchMatchProbabilityUnorm16&) = delete;
  PatchMatchProbabilityUnorm16& operator=(const PatchMatchProbabilityUnorm16&) =
      delete;
  size_t GetWidth() const { return width_; }
  size_t GetHeight() const { return height_; }
  size_t GetDepth() const { return depth_; }
  size_t GetPitch() const { return pitch_; }
  size_t GetCapacityBytes() const { return capacity_; }
  ProbabilityUnorm16View View() const {
    return {ptr_, pitch_, width_, height_, depth_, invalid_};
  }
  void ValidateStoredRange() const {
    uint32_t invalid = 0;
    CUDA_SAFE_CALL(cudaMemcpy(
        &invalid, invalid_, sizeof(invalid), cudaMemcpyDeviceToHost));
    THROW_CHECK_EQ(invalid, 0u)
        << "UNORM16 probability saw nonfinite/out-of-range state";
  }
  Mat<float> CopyToMat() const {
    ValidateStoredRange();
    Mat<float> result(width_, height_, depth_);
    std::vector<uint16_t> codes(result.GetData().size());
    if (depth_ != 0)
      CUDA_SAFE_CALL(cudaMemcpy2D(codes.data(),
                                  width_ * sizeof(uint16_t),
                                  ptr_,
                                  pitch_,
                                  width_ * sizeof(uint16_t),
                                  height_ * depth_,
                                  cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < codes.size(); ++i)
      result.GetPtr()[i] = static_cast<float>(codes[i]) * 0x1.0001p-16f;
    return result;
  }

#if defined(__CUDACC__)
  void FillWithScalar(float value) {
    THROW_CHECK(value >= -ProbabilityUnorm16View::kEndpointRoundoff &&
                value <= 1.0f + ProbabilityUnorm16View::kEndpointRoundoff);
    const dim3 block(32, 8);
    const dim3 grid((width_ - 1) / 32 + 1, (height_ - 1) / 8 + 1);
    probability_unorm16::Fill<<<grid, block>>>(View(), value);
    CUDA_SYNC_AND_CHECK();
  }
  void RotateWithPlaneScratch(float* scratch, size_t scratch_bytes) {
    THROW_CHECK(scratch != nullptr);
    THROW_CHECK_NE(static_cast<void*>(scratch), static_cast<void*>(ptr_));
    const size_t nw = height_, nh = width_, np = Pitch(nw);
    const size_t old_plane = Bytes(pitch_, height_, 1);
    const size_t new_plane = Bytes(np, nh, 1);
    THROW_CHECK_LE(new_plane, scratch_bytes);
    THROW_CHECK_LE(Bytes(np, nh, depth_), data_bytes_);
    const auto own_begin = reinterpret_cast<uintptr_t>(ptr_);
    const auto scratch_begin = reinterpret_cast<uintptr_t>(scratch);
    const auto max_address = std::numeric_limits<uintptr_t>::max();
    THROW_CHECK_LE(capacity_, max_address - own_begin);
    THROW_CHECK_LE(new_plane, max_address - scratch_begin);
    THROW_CHECK(scratch_begin + new_plane <= own_begin ||
                own_begin + capacity_ <= scratch_begin);
    for (size_t ordinal = 0; ordinal < depth_; ++ordinal) {
      const size_t layer =
          new_plane > old_plane ? depth_ - 1 - ordinal : ordinal;
      auto* input = reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(ptr_) +
                                                layer * old_plane);
      auto* output = reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(ptr_) +
                                                 layer * new_plane);
      // Actual current native cuda_rotate.h defines the generic T primitive.
      CudaRotate<uint16_t>(input,
                           reinterpret_cast<uint16_t*>(scratch),
                           static_cast<int>(width_),
                           static_cast<int>(height_),
                           static_cast<int>(pitch_),
                           static_cast<int>(np));
      CUDA_SAFE_CALL(cudaMemcpy2D(output,
                                  np,
                                  scratch,
                                  np,
                                  nw * sizeof(uint16_t),
                                  nh,
                                  cudaMemcpyDeviceToDevice));
    }
    CUDA_SYNC_AND_CHECK();
    Reorient(nw, nh);
  }
#endif

 private:
  static size_t Pitch(size_t width) {
    THROW_CHECK_LE(width, (std::numeric_limits<size_t>::max() - 511) / 2);
    const size_t pitch = (width * 2 + 511) / 512 * 512;
    THROW_CHECK_LE(pitch, std::numeric_limits<int>::max());
    return pitch;
  }
  static size_t Bytes(size_t pitch, size_t height, size_t depth) {
    THROW_CHECK_LE(height, std::numeric_limits<size_t>::max() / pitch);
    const size_t plane = pitch * height;
    THROW_CHECK(depth == 0 ||
                plane <= std::numeric_limits<size_t>::max() / depth);
    return plane * depth;
  }
  void Reorient(size_t width, size_t height) {
    THROW_CHECK((width == original_width_ && height == original_height_) ||
                (width == original_height_ && height == original_width_));
    THROW_CHECK_LE(Bytes(Pitch(width), height, depth_), data_bytes_);
    width_ = width;
    height_ = height;
    pitch_ = Pitch(width);
  }
  const size_t original_width_, original_height_, depth_;
  uint16_t* ptr_ = nullptr;
  uint32_t* invalid_ = nullptr;
  size_t data_bytes_ = 0, capacity_ = 0, width_ = 0, height_ = 0, pitch_ = 0;
};

}  // namespace mvs
}  // namespace colmap
