// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>

namespace colmap {
namespace mvs {
namespace internal {

// Host-only plan for the private, tightly packed look-ahead allocation. Keep
// admission, allocation size, and view offsets together so they cannot
// disagree. An empty plan selects the existing non-banded sweep.
struct PatchMatchLookAheadPlan {
  static constexpr size_t kMaxBytes = 64 * 1024 * 1024;
  static constexpr int kMaxRows = 128;

  size_t width = 0;
  size_t sources = 0;
  int samples = 0;
  int rows = 0;
  bool geometry = false;
  // Offsets count floats, while num_bytes includes the four carry planes.
  size_t uniforms_offset = 0;
  size_t costs_offset = 0;
  size_t geometry_offset = 0;
  size_t previous_offset = 0;
  size_t num_bytes = 0;

  static PatchMatchLookAheadPlan Make(const size_t width,
                                      const size_t sources,
                                      const int samples,
                                      const bool geometry) {
    // This bounds speculative work and memory, not a measured speed crossover.
    if (width == 0 || sources == 0 || samples <= 0 || sources > 32 ||
        sources > static_cast<size_t>(samples)) {
      return {};
    }
    constexpr size_t kFloatCapacity = kMaxBytes / sizeof(float);
    if (width > kFloatCapacity / 4) return {};
    const size_t planes =
        (geometry ? 5 : 3) * sources + static_cast<size_t>(samples) + 4;
    // Divide before multiplying, so all offsets below fit the byte cap.
    const size_t available_rows = (kFloatCapacity - 4 * width) / width / planes;
    const int rows =
        static_cast<int>(std::min<size_t>(kMaxRows, available_rows));
    if (rows == 0) return {};

    PatchMatchLookAheadPlan plan;
    plan.width = width;
    plan.sources = sources;
    plan.samples = samples;
    plan.rows = rows;
    plan.geometry = geometry;
    const size_t band_pixels = width * rows;
    plan.uniforms_offset = 4 * band_pixels;
    plan.costs_offset = plan.uniforms_offset + samples * band_pixels;
    plan.geometry_offset = plan.costs_offset + 3 * sources * band_pixels;
    plan.previous_offset =
        plan.geometry_offset + (geometry ? 2 * sources * band_pixels : 0);
    plan.num_bytes = (plan.previous_offset + 4 * width) * sizeof(float);
    return plan;
  }
};

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
