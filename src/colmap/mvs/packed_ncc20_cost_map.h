// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/cuda_rotate.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace colmap {
namespace mvs {

namespace ncc20 {
constexpr unsigned kMaxCode = 0xfffffu;
constexpr float kStep = 0x1p-19f;

#if defined(__CUDACC__) || defined(__HIPCC__)
// Codes 0..Max-1 use the Q19 grid; Max denotes the exact endpoint two.
// The missing grid point is the midpoint between the last two reconstructions.
__device__ inline float Decode(unsigned code) {
  return static_cast<float>(code + unsigned(code == kMaxCode)) * kStep;
}

__device__ inline bool Encode(float value, unsigned& code) {
  const unsigned bits = __float_as_uint(value);
  if (bits > 0x40000000u && bits != 0x80000000u) return false;
  unsigned rounded = __float2uint_rn(__fmul_rn(value, 0x1p19f));
  rounded = min(rounded, kMaxCode);
  // Select the nearest endpoint reconstruction; tie chooses the even lower
  // code.
  if (rounded == kMaxCode && value <= 2.0f - kStep) --rounded;
  code = rounded;
  return true;
}
#endif
}  // namespace ncc20

namespace ncc20_column {
// Narrow scheduling contract: each source/column has one word writer per row;
// successive rows are ordered by the existing cooperative row barriers or one
// serial column thread. Initial stripes must start/end on eight-row boundaries.
// All cost readers (including future-row HMM reads) precede refresh barriers.
// Concurrent arbitrary-row Set calls are NOT supported. Initialize low padding.
// Row-block-major words keep adjacent columns contiguous without sharing words:
// word=(source*(low_height/8)+row/8)*width+col, nibble=row%8.
struct View {
  uint16_t* high;
  uint32_t* low;
  size_t pitch, width, height, sources, low_height;
  unsigned* error;
#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ unsigned GetCode(size_t row, size_t col, size_t source) const {
    const size_t hi = (source * height + row) * pitch + col;
    const size_t word = (source * (low_height / 8) + row / 8) * width + col;
    return (unsigned(high[hi]) << 4) | ((low[word] >> (4 * (row & 7))) & 15u);
  }
  __device__ void StoreCode(size_t row,
                            size_t col,
                            size_t source,
                            unsigned code) const {
    if (code > ncc20::kMaxCode) {
      atomicExch(error, 1u);
      return;
    }
    const size_t hi = (source * height + row) * pitch + col;
    const size_t word = (source * (low_height / 8) + row / 8) * width + col;
    high[hi] = uint16_t(code >> 4);
    const unsigned shift = 4 * unsigned(row & 7), mask = 15u << shift;
    // No payload atomics: the caller contract orders all accesses to this word.
    const unsigned previous = low[word];
    const unsigned next = (previous & ~mask) | ((code & 15u) << shift);
    if (next != previous) low[word] = next;
  }
  __device__ float Get(size_t row, size_t col, size_t source) const {
    return ncc20::Decode(GetCode(row, col, source));
  }
  __device__ void Set(size_t row,
                      size_t col,
                      size_t source,
                      float value) const {
    unsigned code;
    if (!ncc20::Encode(value, code)) {
      atomicExch(error, 1u);
      return;
    }
    StoreCode(row, col, source, code);
  }
#endif
};

#if defined(__CUDACC__) || defined(__HIPCC__)
// Rotation has exclusive ownership. Expand only logical destination rows into
// row-major scratch; padded low rows are supplied by PackRotated, not
// allocated.
static __global__ void GatherRotated(const uint32_t* input,
                                     uint8_t* scratch,
                                     size_t width,
                                     size_t height,
                                     size_t new_pitch) {
  const size_t count = new_pitch * width;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    const size_t row = i / new_pitch, col = i % new_pitch;
    unsigned nibble = 0;
    if (col < height) {
      const size_t old_row = col, old_col = width - 1 - row;
      const size_t word = (old_row / 8) * width + old_col;
      nibble = (input[word] >> (4 * (old_row & 7))) & 15u;
    }
    scratch[i] = uint8_t(nibble);
  }
}
static __global__ void PackRotated(const uint8_t* scratch,
                                   uint32_t* output,
                                   size_t width,
                                   size_t height,
                                   size_t pitch,
                                   size_t low_height) {
  const size_t words_per_column = low_height / 8;
  const size_t count = width * words_per_column;
  for (size_t word = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       word < count;
       word += size_t(blockDim.x) * gridDim.x) {
    const size_t col = word % width, row = 8 * (word / width);
    unsigned value = 0;
#pragma unroll
    for (unsigned k = 0; k < 8; ++k)
      if (row + k < height)
        value |= unsigned(scratch[(row + k) * pitch + col]) << (4 * k);
    output[word] = value;
  }
}
#endif
}  // namespace ncc20_column

// Q19 NCC storage. Each low word belongs to one source/column/eight-row block.
struct Fixed20NccCostView : ncc20_column::View {
  static constexpr int kStorageBits = 20;
  __host__ __device__ size_t GetWidth() const { return width; }
  __host__ __device__ size_t GetHeight() const { return height; }
  __host__ __device__ size_t GetDepth() const { return sources; }
};

class Fixed20NccCostMap {
 public:
  Fixed20NccCostMap(size_t width, size_t height, size_t sources)
      : original_width_(width), original_height_(height), sources_(sources) {
    THROW_CHECK_GT(width, 0);
    THROW_CHECK_GT(height, 0);
    THROW_CHECK_GT(sources, 0);
    THROW_CHECK_LE(width, std::numeric_limits<int>::max());
    THROW_CHECK_LE(height, std::numeric_limits<int>::max());
    THROW_CHECK_LE(sources, std::numeric_limits<int>::max());
    capacity_pixels_ = std::max(Pixels(width, height, sources),
                                Pixels(height, width, sources));
    low_capacity_ = std::max(LowPixels(width, height, sources),
                             LowPixels(height, width, sources));
    THROW_CHECK_LE(low_capacity_, capacity_pixels_);
    const size_t high_bytes = Multiply(capacity_pixels_, 2);
    THROW_CHECK_EQ(low_capacity_ % 8, 0);
    THROW_CHECK_LE(high_bytes,
                   std::numeric_limits<size_t>::max() - low_capacity_ / 2 -
                       sizeof(unsigned));
    capacity_bytes_ = high_bytes + low_capacity_ / 2;
    try {
      CUDA_SAFE_CALL(
          cudaMalloc(reinterpret_cast<void**>(&ptr_), capacity_bytes_));
      THROW_CHECK_EQ(
          reinterpret_cast<uintptr_t>(LowWords()) % alignof(uint32_t), 0);
      CUDA_SAFE_CALL(cudaMemset(LowWords(), 0, low_capacity_ / 2));
      CUDA_SAFE_CALL(
          cudaMalloc(reinterpret_cast<void**>(&error_), sizeof(unsigned)));
      CUDA_SAFE_CALL(cudaMemset(error_, 0, sizeof(unsigned)));
    } catch (...) {
      Release();
      throw;
    }
    Reorient(width, height);
  }
  ~Fixed20NccCostMap() { Release(); }
  Fixed20NccCostMap(const Fixed20NccCostMap&) = delete;
  Fixed20NccCostMap& operator=(const Fixed20NccCostMap&) = delete;
  size_t GetWidth() const { return width_; }
  size_t GetHeight() const { return height_; }
  size_t GetDepth() const { return sources_; }
  size_t GetCapacityBytes() const { return capacity_bytes_ + sizeof(unsigned); }
  Fixed20NccCostView View() const {
    Fixed20NccCostView view{};
    view.high = ptr_;
    view.low = LowWords();
    view.pitch = pitch_;
    view.width = width_;
    view.height = height_;
    view.sources = sources_;
    view.low_height = Round8(height_);
    view.error = error_;
    return view;
  }
  void CheckDomain() const {
    unsigned status = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(&status, error_, sizeof(status), cudaMemcpyDeviceToHost));
    THROW_CHECK_EQ(status, 0u) << "NCC20 cost outside finite [0,2]";
  }
  void Reorient(size_t width, size_t height) {
    THROW_CHECK((width == original_width_ && height == original_height_) ||
                (width == original_height_ && height == original_width_));
    THROW_CHECK_LE(Pixels(width, height, sources_), capacity_pixels_);
    THROW_CHECK_LE(LowPixels(width, height, sources_), low_capacity_);
    width_ = width;
    height_ = height;
    pitch_ = Pitch(width);
  }
#if defined(__CUDACC__) || defined(__HIPCC__)
  void RotateWithPlaneScratch(void* scratch, size_t scratch_bytes) {
    THROW_CHECK(scratch != nullptr);
    const size_t new_pitch = Pitch(height_);
    const size_t old_plane = Multiply(pitch_, height_),
                 new_plane = Multiply(new_pitch, width_);
    const size_t old_low_plane = Multiply(width_, Round8(height_));
    const size_t new_low_plane = Multiply(height_, Round8(width_));
    const size_t high_stage = Multiply(new_plane, 2);
    // Only active destination rows are expanded, even for thin shapes.
    const size_t low_stage = new_plane;
    const size_t required = std::max(high_stage, low_stage);
    THROW_CHECK_LE(required, scratch_bytes);
    THROW_CHECK_LE(Pixels(height_, width_, sources_), capacity_pixels_);
    THROW_CHECK_LE(LowPixels(height_, width_, sources_), low_capacity_);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(scratch);
    THROW_CHECK_EQ(begin % alignof(uint16_t), 0);
    THROW_CHECK_LE(required, std::numeric_limits<uintptr_t>::max() - begin);
    const uintptr_t end = begin + required,
                    data = reinterpret_cast<uintptr_t>(ptr_),
                    flag = reinterpret_cast<uintptr_t>(error_);
    THROW_CHECK_LE(capacity_bytes_,
                   std::numeric_limits<uintptr_t>::max() - data);
    THROW_CHECK_LE(sizeof(unsigned),
                   std::numeric_limits<uintptr_t>::max() - flag);
    THROW_CHECK(end <= data || begin >= data + capacity_bytes_);
    THROW_CHECK(end <= flag || begin >= flag + sizeof(unsigned));
    // Each source is fully staged before overwriting; order protects other
    // slices.
    for (size_t ordinal = 0; ordinal < sources_; ++ordinal) {
      const size_t source =
          new_plane > old_plane ? sources_ - 1 - ordinal : ordinal;
      auto* temp = static_cast<uint16_t*>(scratch);
      CudaRotate(ptr_ + source * old_plane,
                 temp,
                 width_,
                 height_,
                 2 * pitch_,
                 2 * new_pitch);
      CUDA_SAFE_CALL(cudaMemcpy2D(ptr_ + source * new_plane,
                                  2 * new_pitch,
                                  temp,
                                  2 * new_pitch,
                                  2 * height_,
                                  width_,
                                  cudaMemcpyDeviceToDevice));
    }
    // Column-low source planes can grow while high planes shrink (129x127).
    // Choose this traversal independently; never reuse high-plane ordering.
    for (size_t ordinal = 0; ordinal < sources_; ++ordinal) {
      const size_t source =
          new_low_plane > old_low_plane ? sources_ - 1 - ordinal : ordinal;
      ncc20_column::GatherRotated<<<Blocks(low_stage), 128>>>(
          LowWords() + source * (old_low_plane / 8),
          static_cast<uint8_t*>(scratch),
          width_,
          height_,
          new_pitch);
      CUDA_SAFE_CALL(cudaGetLastError());
      ncc20_column::PackRotated<<<Blocks(new_low_plane / 8), 128>>>(
          static_cast<uint8_t*>(scratch),
          LowWords() + source * (new_low_plane / 8),
          height_,
          width_,
          new_pitch,
          Round8(width_));
      CUDA_SAFE_CALL(cudaGetLastError());
    }
    CUDA_SYNC_AND_CHECK();
    Reorient(height_, width_);
  }
#else
  void RotateWithPlaneScratch(void* scratch, size_t scratch_bytes);
#endif
 private:
  void Release() noexcept {
    (void)cudaFree(error_);
    (void)cudaFree(ptr_);
    error_ = nullptr;
    ptr_ = nullptr;
  }
  uint32_t* LowWords() const {
    return reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(ptr_) +
                                       2 * capacity_pixels_);
  }
  static size_t Multiply(size_t a, size_t b) {
    THROW_CHECK(b == 0 || a <= std::numeric_limits<size_t>::max() / b);
    return a * b;
  }
  static size_t Pitch(size_t width) {
    THROW_CHECK_LE(width, std::numeric_limits<size_t>::max() - 127);
    const size_t pitch = (width + 127) / 128 * 128;
    THROW_CHECK_LE(pitch, std::numeric_limits<int>::max() / 2);
    return pitch;
  }
  static size_t Pixels(size_t w, size_t h, size_t s) {
    return Multiply(Multiply(Pitch(w), h), s);
  }
  static size_t Round8(size_t height) {
    THROW_CHECK_LE(height, std::numeric_limits<size_t>::max() - 7);
    return (height + 7) / 8 * 8;
  }
  static size_t LowPixels(size_t w, size_t h, size_t s) {
    return Multiply(Multiply(w, Round8(h)), s);
  }
  static unsigned Blocks(size_t n) {
    return static_cast<unsigned>(std::min(size_t(65535), (n - 1) / 128 + 1));
  }
  const size_t original_width_, original_height_, sources_;
  size_t capacity_pixels_ = 0, low_capacity_ = 0, capacity_bytes_ = 0,
         width_ = 0, height_ = 0, pitch_ = 0;
  uint16_t* ptr_ = nullptr;
  unsigned* error_ = nullptr;
};

}  // namespace mvs
}  // namespace colmap
