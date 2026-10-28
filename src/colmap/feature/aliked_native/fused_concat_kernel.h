// SPDX-License-Identifier: MIT
// See LICENSE-MIT for permission text.
#pragma once
#include <cuda_runtime_api.h>

cudaError_t LaunchFusedConcat(const float* a,
                              const float* b,
                              const float* c,
                              const float* d,
                              float* output,
                              int h,
                              int w,
                              unsigned grid_x,
                              cudaStream_t stream);
