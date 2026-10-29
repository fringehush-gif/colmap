// SPDX-License-Identifier: MIT
// See LICENSE-MIT and NOTICE for source and permission notices.
#include <cstddef>

#include "fused_concat_kernel.h"
namespace {
__device__ float ReadBilinear(const float* input,
                              int channel,
                              int y,
                              int x,
                              int h,
                              int w,
                              int ih,
                              int iw) {
  // Resize-derived arithmetic: ONNX Runtime v1.31.1 resize_impl.{h,cu}.
  // Copyright (c) Microsoft Corporation. All rights reserved.
  // MIT; see LICENSE-ONNXRUNTIME. Preserve multiply before divide.
  float fy = static_cast<float>(y) * static_cast<float>(ih - 1) /
             static_cast<float>(h - 1);
  float fx = static_cast<float>(x) * static_cast<float>(iw - 1) /
             static_cast<float>(w - 1);
  fy = max(0.0f, min(fy, static_cast<float>(ih - 1)));
  fx = max(0.0f, min(fx, static_cast<float>(iw - 1)));
  const int y0 = static_cast<int>(fy);
  const int x0 = static_cast<int>(fx);
  const int y1 = y0 + 1 < ih ? y0 + 1 : ih - 1;
  const int x1 = x0 + 1 < iw ? x0 + 1 : iw - 1;
  const float wy = y0 >= ih - 1 ? 0.0f : fy - y0;
  const float wx = x0 >= iw - 1 ? 0.0f : fx - x0;
  const float one_minus_y = 1.0f - wy;
  const float one_minus_x = 1.0f - wx;
  const size_t base = static_cast<size_t>(channel) * ih * iw;
  const float v00 = input[base + static_cast<size_t>(y0) * iw + x0];
  const float v01 = input[base + static_cast<size_t>(y0) * iw + x1];
  const float v10 = input[base + static_cast<size_t>(y1) * iw + x0];
  const float v11 = input[base + static_cast<size_t>(y1) * iw + x1];
  return v00 * (one_minus_y * one_minus_x) + v10 * (wy * one_minus_x) +
         v01 * (one_minus_y * wx) + v11 * (wy * wx);
}

__global__ void FusedKernel(const float* a,
                            const float* b,
                            const float* c,
                            const float* d,
                            float* output,
                            int h,
                            int w) {
  const int channel =
      blockIdx.y;  // One 32-channel branch region per 32 grid rows.
  const int branch = channel / 32;
  const int source_channel = channel % 32;
  const int scale = branch == 0 ? 1 : branch == 1 ? 2 : branch == 2 ? 8 : 32;
  const int ih = h / scale;
  const int iw = w / scale;
  const float* source = branch == 0 ? a : branch == 1 ? b : branch == 2 ? c : d;
  const size_t plane = static_cast<size_t>(h) * w;
  const size_t tile =
      static_cast<size_t>(blockIdx.x) * blockDim.x * 4 + threadIdx.x;
  for (int j = 0; j < 4; ++j) {
    const size_t pixel = tile + static_cast<size_t>(j) * blockDim.x;
    if (pixel >= plane) continue;
    const int y = static_cast<int>(pixel / w);
    const int x = static_cast<int>(pixel % w);
    const float value =
        branch == 0 ? a[static_cast<size_t>(source_channel) * plane + pixel]
                    : ReadBilinear(source, source_channel, y, x, h, w, ih, iw);
    output[static_cast<size_t>(channel) * plane + pixel] = value;
  }
}

}  // namespace
cudaError_t LaunchFusedConcat(const float* a,
                              const float* b,
                              const float* c,
                              const float* d,
                              float* output,
                              int h,
                              int w,
                              unsigned grid_x,
                              cudaStream_t stream) {
  FusedKernel<<<dim3(grid_x, 128), 256, 0, stream>>>(a, b, c, d, output, h, w);
  return cudaGetLastError();
}
