// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <boost/core/bit.hpp>

namespace colmap::mvs::internal {

// NaNs and subnormals retain the worker's floating-point comparison semantics.
inline bool IsDefinitelyNonpositiveDepth(const uint32_t bits) {
  const uint32_t magnitude = bits & 0x7fffffffu;
  return magnitude == 0 ||
         ((bits & 0x80000000u) != 0 && magnitude >= 0x00800000u &&
          magnitude <= 0x7f800000u);
}

// Immutable random-access FP32 depth. Only the exact +0 word is implicit.
class SparseDepth {
 public:
  // Returns nullptr unless the complete representation saves at least 10%.
  static std::unique_ptr<const SparseDepth> Pack(
      const std::vector<float>& source, size_t width, size_t height);

  size_t NumBytes() const;
  size_t Width() const { return width_; }
  size_t Size() const { return width_ * height_; }

  float Get(const size_t row, const size_t col) const {
    if (row >= height_ || col >= width_) {
      throw std::out_of_range("Depth coordinate outside raster");
    }
    const size_t index = row * width_ + col;
    const size_t word = index / 64;
    const uint64_t bit = uint64_t{1} << (index % 64);
    const uint64_t occupied = occupancy_[word];
    const uint32_t bits =
        occupied & bit ? values_[prefix_[word] +
                                 boost::core::popcount(occupied & (bit - 1))]
                       : 0;
    float value = 0.0f;
    if constexpr (sizeof(value) == sizeof(bits)) {
      std::memcpy(&value, &bits, sizeof(value));
    }
    return value;
  }

  // Classifies one 64-sample group for Fusion's pre-dispatch exclusion mask.
  uint64_t NonpositiveWord(size_t word) const;

 private:
  SparseDepth() = default;
  size_t width_ = 0;
  size_t height_ = 0;
  size_t word_count_ = 0;
  size_t value_count_ = 0;
  // Exact-count arrays keep the accepted byte count and the load reservation
  // independent of std::vector's implementation-defined capacity growth.
  std::unique_ptr<uint64_t[]> occupancy_;
  std::unique_ptr<uint32_t[]> prefix_;
  std::unique_ptr<uint32_t[]> values_;
};

}  // namespace colmap::mvs::internal
