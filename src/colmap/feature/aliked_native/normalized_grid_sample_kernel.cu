// SPDX-License-Identifier: MIT
// See LICENSE-MIT and NOTICE for permission text and arithmetic references.
// Independent optional implementation of normalized, bilinear, zeros sampling.
// Arithmetic ordering follows the documented ORT 1.31.1 scalar expressions.
#include "normalized_grid_sample_kernel.h"
#include <cuda_runtime.h>

namespace {

__device__ float NormalizedCorner(const float* raw,
                                  const float* norm,
                                  uint64_t channel_offset,
                                  int64_t y,
                                  int64_t x,
                                  int64_t h,
                                  int64_t w) {
  if (x < 0 || x >= w || y < 0 || y >= h) return 0.0f;
  const uint64_t pixel = static_cast<uint64_t>(y) * static_cast<uint64_t>(w) +
                         static_cast<uint64_t>(x);
  // Division occurs BEFORE interpolation, exactly where the dense Div was.
  return raw[channel_offset + pixel] / norm[pixel];
}

__global__ void NormalizedGridSample(const float* raw,
                                     const float* norm,
                                     const float* grid,
                                     float* output,
                                     int64_t h,
                                     int64_t w,
                                     uint64_t samples) {
  const uint64_t channel = blockIdx.y;
  const uint64_t channel_offset =
      channel * static_cast<uint64_t>(h) * static_cast<uint64_t>(w);
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t sample =
           static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       sample < samples;
       sample += step) {
    // ORT align_corners=true denormalization; retain its operation order.
    float x = (grid[2 * sample] + 1.0f) / 2.0f * (w - 1);
    float y = (grid[2 * sample + 1] + 1.0f) / 2.0f * (h - 1);
    constexpr float safe_bound = static_cast<float>(int64_t{1} << 62);
    // Matched ORT 1.31.1 guards conversion then substitutes lower border zero.
    if (!isfinite(x) || x < -safe_bound || x > safe_bound) x = 0.0f;
    if (!isfinite(y) || y < -safe_bound || y > safe_bound) y = 0.0f;
    const int64_t x1 = static_cast<int64_t>(floorf(x));
    const int64_t y1 = static_cast<int64_t>(floorf(y));
    const int64_t x2 = x1 + 1, y2 = y1 + 1;
    const float right = x - x1, left = 1.0f - right;
    const float bottom = y - y1, top = 1.0f - bottom;
    const float w_lt = top * left, w_rt = top * right;
    const float w_lb = bottom * left, w_rb = bottom * right;
    const float lt = NormalizedCorner(raw, norm, channel_offset, y1, x1, h, w);
    const float rt = NormalizedCorner(raw, norm, channel_offset, y1, x2, h, w);
    const float lb = NormalizedCorner(raw, norm, channel_offset, y2, x1, h, w);
    const float rb = NormalizedCorner(raw, norm, channel_offset, y2, x2, h, w);
    output[channel * samples + sample] =
        w_lt * lt + w_rt * rt + w_lb * lb + w_rb * rb;
  }
}

}  // namespace

cudaError_t LaunchNormalizedGridSample(const float* raw,
                                       const float* norm,
                                       const float* grid,
                                       float* output,
                                       int64_t h,
                                       int64_t w,
                                       uint64_t samples,
                                       unsigned grid_x,
                                       cudaStream_t stream) {
  if (samples == 0) return cudaSuccess;
  if (!raw || !norm || !grid || !output || h <= 0 || w <= 0 || grid_x == 0) {
    return cudaErrorInvalidValue;
  }
  NormalizedGridSample<<<dim3(grid_x, 128, 1), 256, 0, stream>>>(
      raw, norm, grid, output, h, w, samples);
  return cudaGetLastError();
}
