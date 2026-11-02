// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/cuda_to_hip.h"

namespace colmap {
namespace mvs {

// Rotate the input matrix by 90 degrees in counter-clockwise direction.
template <typename T>
void CudaRotate(const T* input,
                T* output,
                const int width,
                const int height,
                const int pitch_input,
                const int pitch_output);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDACC__) || defined(__HIPCC__)

#define TILE_DIM_ROTATE 32

namespace internal {

template <typename T>
__global__ void CudaRotateKernel(T* output_data,
                                 const T* input_data,
                                 const int width,
                                 const int height,
                                 const int input_pitch,
                                 const int output_pitch) {
  __shared__ T tile[TILE_DIM_ROTATE][TILE_DIM_ROTATE + 1];
  const int input_x = TILE_DIM_ROTATE * blockIdx.x + threadIdx.x;
  const int input_y = TILE_DIM_ROTATE * blockIdx.y + threadIdx.y;
  for (int row = 0; row < TILE_DIM_ROTATE; row += blockDim.y) {
    if (input_x < width && input_y + row < height) {
      const auto* input_row = reinterpret_cast<const T*>(
          reinterpret_cast<const char*>(input_data) +
          static_cast<size_t>(input_y + row) * input_pitch);
      tile[threadIdx.y + row][threadIdx.x] = input_row[input_x];
    }
  }
  __syncthreads();

  // Transpose in shared memory so both global reads and writes follow rows.
  const int output_x = TILE_DIM_ROTATE * blockIdx.y + threadIdx.x;
  const int source_x = TILE_DIM_ROTATE * blockIdx.x + threadIdx.y;
  for (int row = 0; row < TILE_DIM_ROTATE; row += blockDim.y) {
    if (output_x < height && source_x + row < width) {
      auto* output_row = reinterpret_cast<T*>(
          reinterpret_cast<char*>(output_data) +
          static_cast<size_t>(width - 1 - source_x - row) * output_pitch);
      output_row[output_x] = tile[threadIdx.x][threadIdx.y + row];
    }
  }
}

}  // namespace internal

template <typename T>
void CudaRotate(const T* input,
                T* output,
                const int width,
                const int height,
                const int pitch_input,
                const int pitch_output) {
  dim3 block_dim(TILE_DIM_ROTATE, 8, 1);
  dim3 grid_dim;
  grid_dim.x = (width - 1) / TILE_DIM_ROTATE + 1;
  grid_dim.y = (height - 1) / TILE_DIM_ROTATE + 1;

  internal::CudaRotateKernel<<<grid_dim, block_dim>>>(
      output, input, width, height, pitch_input, pitch_output);
}

#undef TILE_DIM_ROTATE

#endif  // defined(__CUDACC__) || defined(__HIPCC__)

}  // namespace mvs
}  // namespace colmap
