// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/gpu_mat.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace colmap {
namespace mvs {

inline size_t PackedConsistencyMaskDepth(const size_t num_sources) {
  return (num_sources + 7) / 8;
}

__host__ __device__ inline uint8_t PackedConsistencyMaskBit(
    const size_t source) {
  return static_cast<uint8_t>(1u << (source % 8));
}

// Each column owner writes its source flags serially, so updating a byte does
// not require an atomic operation. Source-parallel writers must not use this
// view.
struct PackedConsistencyMaskView {
  GpuMatView<uint8_t> bytes;

  __device__ void Set(const size_t row,
                      const size_t col,
                      const size_t source,
                      const uint8_t value) {
    const size_t plane = source / 8;
    const uint8_t bit = PackedConsistencyMaskBit(source);
    const uint8_t packed = bytes.Get(row, col, plane);
    bytes.Set(row,
              col,
              plane,
              value ? static_cast<uint8_t>(packed | bit)
                    : static_cast<uint8_t>(packed & ~bit));
  }
};

inline std::vector<int> DecodePackedConsistencyMask(
    const Mat<uint8_t>& mask, const std::vector<int>& source_image_idxs) {
  std::vector<int> consistent_image_idxs;
  std::vector<int> pixel_consistent_image_idxs;
  pixel_consistent_image_idxs.reserve(source_image_idxs.size());
  for (size_t r = 0; r < mask.GetHeight(); ++r) {
    for (size_t c = 0; c < mask.GetWidth(); ++c) {
      pixel_consistent_image_idxs.clear();
      for (size_t source = 0; source < source_image_idxs.size(); ++source) {
        if (mask.Get(r, c, source / 8) & PackedConsistencyMaskBit(source)) {
          pixel_consistent_image_idxs.push_back(source_image_idxs[source]);
        }
      }
      if (!pixel_consistent_image_idxs.empty()) {
        consistent_image_idxs.push_back(c);
        consistent_image_idxs.push_back(r);
        consistent_image_idxs.push_back(pixel_consistent_image_idxs.size());
        consistent_image_idxs.insert(consistent_image_idxs.end(),
                                     pixel_consistent_image_idxs.begin(),
                                     pixel_consistent_image_idxs.end());
      }
    }
  }
  return consistent_image_idxs;
}

}  // namespace mvs
}  // namespace colmap
