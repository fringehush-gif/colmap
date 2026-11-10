// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
namespace colmap {
namespace mvs {
namespace internal {

template <int kSampleWindowSize, int kBlockWidth>
__device__ __forceinline__ int AppendSampledSource(
    const int source,
    const bool use_source_mask,
    uint32_t& sampled_sources,
    int& num_unique_sources,
    int (&sources)[kSampleWindowSize][kBlockWidth],
    const int lane) {
  int job_slot = -1;
  if (use_source_mask) {
    job_slot = source;
    if (job_slot >= 0)
      sampled_sources |= uint32_t{1} << static_cast<uint32_t>(job_slot);
  } else if (source >= 0) {
    job_slot = 0;
    while (job_slot < num_unique_sources && sources[job_slot][lane] != source) {
      ++job_slot;
    }
    if (job_slot == num_unique_sources) {
      sources[job_slot][lane] = source;
      ++num_unique_sources;
    }
  }
  return job_slot;
}

template <int kSampleWindowSize, int kBlockWidth>
__device__ __forceinline__ void PackSampledSources(
    const bool use_source_mask,
    const uint32_t sampled_sources,
    int num_unique_sources,
    const int sample_count,
    int (&sources)[kSampleWindowSize][kBlockWidth],
    int (&aliases)[kSampleWindowSize][kBlockWidth],
    const int lane) {
  if (use_source_mask) {
    uint32_t remaining_sources = sampled_sources;
    while (remaining_sources != 0) {
      const int source = __ffs(remaining_sources) - 1;
      sources[num_unique_sources++][lane] = source;
      remaining_sources &= remaining_sources - 1;
    }
    for (int slot = 0; slot < sample_count; ++slot) {
      const int source = aliases[slot][lane];
      if (source >= 0) {
        aliases[slot][lane] =
            __popc(sampled_sources &
                   ((uint32_t{1} << static_cast<uint32_t>(source)) - 1));
      }
    }
  }
  for (int slot = num_unique_sources; slot < kSampleWindowSize; ++slot) {
    sources[slot][lane] = -1;
  }
}

}  // namespace internal

}  // namespace mvs
}  // namespace colmap
