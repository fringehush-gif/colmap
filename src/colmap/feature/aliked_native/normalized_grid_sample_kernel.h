// SPDX-License-Identifier: MIT
// See LICENSE-MIT and NOTICE for permission text and arithmetic references.
#pragma once
#include <cstdint>

#include <cuda_runtime_api.h>

cudaError_t LaunchNormalizedGridSample(const float* raw,
                                       const float* norm,
                                       const float* grid,
                                       float* output,
                                       int64_t h,
                                       int64_t w,
                                       uint64_t samples,
                                       unsigned grid_x,
                                       cudaStream_t stream);
