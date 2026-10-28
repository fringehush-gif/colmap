// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

namespace colmap {
namespace mvs {

// Leave ample capacity for another client and CUDA's unmodeled allocations.
// This applies only to the optional coarse PHOTO2 batch admission.
inline size_t NativeCoarseBatchReserveBytes(const size_t total_vram_bytes) {
  return std::max(size_t(512) << 20, total_vram_bytes / 12);
}

inline bool NativeCoarseBatchFits(const size_t free_bytes,
                                  const size_t owner_peak_bytes,
                                  const size_t charged_owner_count,
                                  const size_t reserve_bytes) {
  if (charged_owner_count == 0 || free_bytes < reserve_bytes ||
      owner_peak_bytes > (std::numeric_limits<size_t>::max() - reserve_bytes) /
                             charged_owner_count) {
    return false;
  }
  return free_bytes >= charged_owner_count * owner_peak_bytes + reserve_bytes;
}

// eligible[n - 3] is the existing resident-wave decision for prefix n.
inline size_t LargestNativeCoarseBatchPrefix(
    const size_t maximum_count,
    const std::array<bool, 3>& eligible,
    const size_t free_after_first_bytes,
    const size_t owner_peak_bytes,
    const size_t reserve_bytes) {
  for (size_t count = std::min(maximum_count, size_t(5)); count >= 3; --count) {
    if (eligible[count - 3] &&
        NativeCoarseBatchFits(
            free_after_first_bytes, owner_peak_bytes, count, reserve_bytes)) {
      return count;
    }
  }
  return 0;
}

}  // namespace mvs
}  // namespace colmap
