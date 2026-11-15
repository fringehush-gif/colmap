// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>

namespace colmap {
namespace mvs {
namespace internal {

// Host-only plan for the private look-ahead allocation. Keep
// admission, allocation size, and view offsets together so they cannot
// disagree. An empty plan selects the existing non-banded sweep.
struct PatchMatchLookAheadPlan {
  static constexpr size_t kMaxBytes = 64 * 1024 * 1024;
  static constexpr int kMaxRows = 128;

  struct Layout {
    // Capacity/pitch in floats, possibly wider than the current image.
    size_t width = 0;
    size_t plane_rows = 0;
    // Offsets count floats; num_bytes includes the four carry planes.
    size_t uniforms_offset = 0;
    size_t costs_offset = 0;
    size_t geometry_offset = 0;
    size_t previous_offset = 0;
    size_t num_bytes = 0;
  };

  size_t sources = 0;
  int samples = 0;
  // Scheduling is independent of the storage extent of a short orientation.
  int rows = 0;
  bool geometry = false;
  // Original and quarter-turned layouts; each remains fixed across its bands.
  Layout layouts[2];
  size_t num_bytes = 0;

  static PatchMatchLookAheadPlan Make(const size_t width,
                                      const size_t height,
                                      const size_t sources,
                                      const int samples,
                                      const bool geometry) {
    // This bounds speculative work and memory, not a measured speed crossover.
    if (width == 0 || height == 0 || sources == 0 || samples <= 0 ||
        sources > 32 || sources > static_cast<size_t>(samples)) {
      return {};
    }
    constexpr size_t kFloatCapacity = kMaxBytes / sizeof(float);
    const size_t max_dimension = std::max(width, height);
    if (max_dimension > kFloatCapacity / 4) return {};
    const size_t planes =
        (geometry ? 5 : 3) * sources + static_cast<size_t>(samples) + 4;
    // Preserve the original admission and band schedule in both orientations.
    const size_t available_rows =
        (kFloatCapacity - 4 * max_dimension) / max_dimension / planes;
    const int rows =
        static_cast<int>(std::min<size_t>(kMaxRows, available_rows));
    if (rows == 0) return {};

    const auto make_layout = [&](const size_t pitch,
                                 const size_t plane_rows,
                                 const size_t capacity) {
      Layout layout;
      // Divide before multiplying. Even a rejected padded layout is checked
      // against the old allocation before any offset/product is formed.
      if (pitch > capacity / 4 ||
          plane_rows > (capacity - 4 * pitch) / pitch / planes) {
        return layout;
      }
      layout.width = pitch;
      layout.plane_rows = plane_rows;
      const size_t band_pixels = pitch * plane_rows;
      layout.uniforms_offset = 4 * band_pixels;
      layout.costs_offset = layout.uniforms_offset + samples * band_pixels;
      layout.geometry_offset = layout.costs_offset + 3 * sources * band_pixels;
      layout.previous_offset =
          layout.geometry_offset + (geometry ? 2 * sources * band_pixels : 0);
      layout.num_bytes = (layout.previous_offset + 4 * pitch) * sizeof(float);
      return layout;
    };

    PatchMatchLookAheadPlan plan;
    plan.sources = sources;
    plan.samples = samples;
    plan.rows = rows;
    plan.geometry = geometry;
    const Layout legacy = make_layout(max_dimension, rows, kFloatCapacity);
    plan.layouts[0] = legacy;
    plan.layouts[1] = legacy;
    plan.num_bytes = legacy.num_bytes;
    if (std::min(width, height) >= static_cast<size_t>(rows)) return plan;

    // Compact only when the owner allocation shrinks. Padding short rows to
    // 32 floats keeps row/plane starts aligned without changing kernel shapes.
    // If padding consumes the saving, preserve the original layouts exactly.
    const size_t capacity = legacy.num_bytes / sizeof(float);
    const Layout original = make_layout(
        (width + 31) / 32 * 32, std::min<size_t>(rows, height), capacity);
    const Layout rotated = make_layout(
        (height + 31) / 32 * 32, std::min<size_t>(rows, width), capacity);
    if (original.num_bytes != 0 && rotated.num_bytes != 0 &&
        std::max(original.num_bytes, rotated.num_bytes) < legacy.num_bytes) {
      plan.layouts[0] = original;
      plan.layouts[1] = rotated;
      plan.num_bytes = std::max(original.num_bytes, rotated.num_bytes);
    }
    return plan;
  }
};

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
