// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
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

// Returns whether the CUDA device natively supports bf16 arithmetic, i.e., has
// compute capability 8.0 (Ampere) or newer. Returns false if the device cannot
// be queried, e.g., for an invalid index or a missing driver. Always returns
// false for HIP builds.
bool CudaDeviceSupportsBf16(int gpu_index);

}  // namespace colmap
