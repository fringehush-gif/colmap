// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cuda.h"

#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <iostream>
#include <string>

namespace colmap {
namespace {

// Check whether the first Cuda device is better than the second.
bool CompareCudaDevice(const cudaDeviceProp& d1, const cudaDeviceProp& d2) {
  bool result = (d1.major > d2.major) ||
                ((d1.major == d2.major) && (d1.minor > d2.minor)) ||
                ((d1.major == d2.major) && (d1.minor == d2.minor) &&
                 (d1.multiProcessorCount > d2.multiProcessorCount));
  return result;
}

}  // namespace

int GetNumCudaDevices() {
  int num_cuda_devices = 0;
  const cudaError_t error = cudaGetDeviceCount(&num_cuda_devices);
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
  if (error == cudaErrorInsufficientDriver) {
    // Preserve the zero-device result for callers that can fall back to CPU,
    // but do not discard the reason why GPU initialization failed.
    LOG(WARNING) << "Cannot enumerate GPU devices: "
                 << GetCudaErrorMessage(error);
    return 0;
  }
  if (error == cudaErrorNoDevice) {
    return 0;
  }
#endif
  CUDA_SAFE_CALL(error);
  return num_cuda_devices;
}

int FindBestCudaDevice() {
  const int num_devices = GetNumCudaDevices();
  THROW_CHECK_GT(num_devices, 0) << "No CUDA devices available";
  std::vector<cudaDeviceProp> all_devices(num_devices);
  std::vector<int> indices(num_devices);
  for (int id = 0; id < num_devices; ++id) {
    indices[id] = id;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&all_devices[id], id));
  }
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return CompareCudaDevice(all_devices[a], all_devices[b]);
  });
  const int selected = indices.front();
  VLOG(2) << "Found " << num_devices << " CUDA device(s), "
          << "selected device " << selected << " with name "
          << all_devices[selected].name;
  return selected;
}

void SetBestCudaDevice(const int gpu_index) {
  const int num_cuda_devices = GetNumCudaDevices();
  THROW_CHECK_GT(num_cuda_devices, 0) << "No CUDA devices available";
  const int selected = (gpu_index >= 0) ? gpu_index : FindBestCudaDevice();
  THROW_CHECK_LT(selected, num_cuda_devices) << "Invalid CUDA GPU selected";
  CUDA_SAFE_CALL(cudaSetDevice(selected));
}

void SetCudaDevicesBlockingSync(const std::vector<int>& gpu_indices) {
#if defined(COLMAP_CUDA_ENABLED) && CUDART_VERSION >= 12000
  const int num_devices = GetNumCudaDevices();
  THROW_CHECK_GT(num_devices, 0) << "No CUDA devices available";
  THROW_CHECK(!gpu_indices.empty()) << "No CUDA GPU selected";
  std::vector<int> selected = gpu_indices;
  if (selected.size() == 1 && selected.front() == -1) {
    selected.resize(num_devices);
    for (int device = 0; device < num_devices; ++device) {
      selected[device] = device;
    }
  }
  for (const int device : selected) {
    THROW_CHECK_GE(device, 0) << "Invalid CUDA GPU selected";
    THROW_CHECK_LT(device, num_devices) << "Invalid CUDA GPU selected";
  }
  std::sort(selected.begin(), selected.end());
  selected.erase(std::unique(selected.begin(), selected.end()), selected.end());

  int previous_device = 0;
  CUDA_SAFE_CALL(cudaGetDevice(&previous_device));
  try {
    for (const int device : selected) {
      CUDA_SAFE_CALL(cudaSetDevice(device));
      unsigned int previous_flags = 0;
      CUDA_SAFE_CALL(cudaGetDeviceFlags(&previous_flags));
      constexpr unsigned int kOtherFlags =
          cudaDeviceLmemResizeToMax | cudaDeviceSyncMemops;
      constexpr unsigned int kKnownFlags =
          cudaDeviceScheduleMask | cudaDeviceMapHost | kOtherFlags;
      THROW_CHECK_EQ(previous_flags & ~kKnownFlags, 0)
          << "Unsupported CUDA device flags";
      // MapHost can be implicit in readback and is not a setter input.
      CUDA_SAFE_CALL(cudaSetDeviceFlags((previous_flags & kOtherFlags) |
                                        cudaDeviceScheduleBlockingSync));
      unsigned int actual_flags = 0;
      CUDA_SAFE_CALL(cudaGetDeviceFlags(&actual_flags));
      THROW_CHECK_EQ(actual_flags & cudaDeviceScheduleMask,
                     cudaDeviceScheduleBlockingSync);
      THROW_CHECK_EQ(actual_flags & ~cudaDeviceScheduleMask,
                     previous_flags & ~cudaDeviceScheduleMask);
      LOG(INFO) << "CUDA device " << device
                << " uses blocking host synchronization (flags " << actual_flags
                << ")";
    }
  } catch (...) {
    // Preserve the original error if restoring the current device also fails.
    cudaSetDevice(previous_device);
    throw;
  }
  CUDA_SAFE_CALL(cudaSetDevice(previous_device));
#else
  LOG(FATAL_THROW) << "Blocking CUDA host synchronization requires CUDA >= 12";
#endif
}

bool CudaDeviceSupportsBf16(const int gpu_index) {
#ifdef COLMAP_HIP_ENABLED
  // The major version of HIP devices is the GFX version, not a compute
  // capability.
  return false;
#else
  cudaDeviceProp properties;
  if (cudaGetDeviceProperties(&properties, gpu_index) != cudaSuccess) {
    // Reset the error for an invalid device so that later CUDA_CHECK() calls
    // do not report it. Initialization errors (e.g., no device) are sticky.
    cudaGetLastError();
    return false;
  }
  return properties.major >= 8;
#endif
}

}  // namespace colmap
