// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

namespace colmap {

int GetNumCudaDevices();

int FindBestCudaDevice();

void SetBestCudaDevice(int gpu_index);

// Use blocking host synchronization for the selected primary CUDA contexts.
// A sole -1 selects every visible device; repeated ordinals are configured
// once. Call before starting CUDA workers. Requires CUDA >= 12; preserves other
// device flags and the calling thread's current device. Does not reset any
// context.
void SetCudaDevicesBlockingSync(const std::vector<int>& gpu_indices);

// Returns whether the CUDA device has compute capability `major`.0 or newer,
// e.g., 8 for native bf16 arithmetic (Ampere) or 7 for fast fp16 arithmetic
// (Volta, Turing). Returns false if the device cannot be queried, e.g., for an
// invalid index or a missing driver. Always returns false for HIP builds.
bool CudaDeviceHasComputeCapability(int gpu_index, int major);

}  // namespace colmap
