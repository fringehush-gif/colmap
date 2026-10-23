// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <chrono>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "trim_gpu.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace PoissonTrimGpu {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kWorkspaceLimit = 3ull * 1024 * 1024 * 1024;

double ElapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

void Check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("Poisson trim GPU: ") +
                             cudaGetErrorString(error));
}

void ClearExpected(cudaError_t expected) {
  const cudaError_t pending = cudaPeekAtLastError();
  if (pending == expected) {
    const cudaError_t cleared = cudaGetLastError();
    if (cleared != expected) Check(cleared);
  } else {
    Check(pending);
  }
}

bool DeviceUnavailable(cudaError_t error) {
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

bool KernelUnavailable(cudaError_t error) {
  return error == cudaErrorInvalidDeviceFunction ||
         error == cudaErrorNoKernelImageForDevice ||
         error == cudaErrorUnsupportedPtxVersion ||
         error == cudaErrorJitCompilerNotFound ||
         error == cudaErrorJitCompilationDisabled;
}

struct PrelaunchAllocationFailure {};
struct PrelaunchUnavailable {
  cudaError_t error;
  const char* reason;
};

void CheckPrelaunch(cudaError_t error) {
  if (error == cudaErrorMemoryAllocation) throw PrelaunchAllocationFailure{};
  if (DeviceUnavailable(error))
    throw PrelaunchUnavailable{error, "device-unavailable"};
  if (KernelUnavailable(error))
    throw PrelaunchUnavailable{error, "kernel-unavailable"};
  Check(error);
}

struct Stream {
  cudaStream_t value = nullptr;
  Stream() {
    CheckPrelaunch(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
  }
  ~Stream() {
    if (value) {
      cudaStreamSynchronize(value);
      cudaStreamDestroy(value);
    }
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};

struct Event {
  cudaEvent_t value = nullptr;
  Event() { CheckPrelaunch(cudaEventCreate(&value)); }
  ~Event() {
    if (value) cudaEventDestroy(value);
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

template <class T>
struct Device {
  T* value = nullptr;
  std::size_t bytes = 0;
  explicit Device(std::size_t count) : bytes(count * sizeof(T)) {
    if (bytes) CheckPrelaunch(cudaMalloc(&value, bytes));
  }
  ~Device() {
    if (value) cudaFree(value);
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

double Duration(cudaEvent_t first, cudaEvent_t last) {
  float ms = 0;
  Check(cudaEventElapsedTime(&ms, first, last));
  return ms;
}

bool HasCapacity(std::uint64_t bytes, Stats& stats) {
  if (bytes > kWorkspaceLimit) {
    stats.reason = "workspace-budget";
    return false;
  }
  std::size_t free = 0, total = 0;
  const cudaError_t error = cudaMemGetInfo(&free, &total);
  if (DeviceUnavailable(error)) {
    ClearExpected(error);
    stats.reason = "device-unavailable";
    return false;
  }
  Check(error);
  if (bytes > (free / 4) * 3 + (free % 4) * 3 / 4) {
    stats.reason = "device-capacity";
    return false;
  }
  return true;
}

__global__ void Edges(const std::uint64_t* offsets,
                      const std::uint32_t* vertices,
                      std::size_t count,
                      std::uint64_t* keys,
                      std::uint32_t* faces,
                      std::uint32_t* parents) {
  const std::size_t face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= count) return;
  parents[face] = std::uint32_t(face);
  const std::uint64_t begin = offsets[face], end = offsets[face + 1];
  for (std::uint64_t j = begin; j < end; ++j) {
    const std::uint32_t a = vertices[j];
    const std::uint32_t b = vertices[j + 1 == end ? begin : j + 1];
    keys[j] = (std::uint64_t(min(a, b)) << 32) | max(a, b);
    faces[j] = std::uint32_t(face);
  }
}

__device__ std::uint32_t Root(std::uint32_t* parents, std::uint32_t node) {
  for (;;) {
    const std::uint32_t parent = atomicAdd(parents + node, 0u);
    if (parent == node) return node;
    const std::uint32_t grandparent = atomicAdd(parents + parent, 0u);
    if (grandparent != parent) atomicMax(parents + node, grandparent);
    node = parent;
  }
}

__global__ void Unite(const std::uint64_t* keys,
                      const std::uint32_t* faces,
                      std::size_t entries,
                      std::uint32_t* parents) {
  const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j == 0 || j >= entries || keys[j] != keys[j - 1]) return;
  std::uint32_t a = faces[j], b = faces[j - 1];
  for (;;) {
    a = Root(parents, a);
    b = Root(parents, b);
    if (a == b) return;
    const std::uint32_t low = min(a, b), high = max(a, b);
    // Every edge points strictly toward a larger polygon index.
    if (atomicCAS(parents + low, low, high) == low) return;
  }
}

__global__ void Flatten(std::uint32_t* parents,
                        std::uint32_t* roots,
                        std::size_t count) {
  const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < count) roots[j] = Root(parents, std::uint32_t(j));
}

template <class Kernel>
bool PreflightKernel(Kernel function, Stats& stats) {
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(&attributes, function);
  if (DeviceUnavailable(error)) {
    ClearExpected(error);
    stats.reason = "device-unavailable";
    return false;
  }
  if (KernelUnavailable(error)) {
    ClearExpected(error);
    stats.reason = "kernel-unavailable";
    return false;
  }
  Check(error);
  if (attributes.maxThreadsPerBlock < 256) {
    stats.reason = "kernel-block-size";
    return false;
  }
  return true;
}

void ValidateShape(std::size_t polygons,
                   std::size_t entries,
                   const std::uint64_t* offsets,
                   const std::uint32_t* vertices) {
  if (!polygons) {
    if (entries) throw std::runtime_error("GPU polygon entries without faces");
    return;
  }
  if (!offsets || (entries && !vertices))
    throw std::runtime_error("GPU polygon null input");
  if (polygons > std::size_t(INT32_MAX) || entries > std::size_t(INT32_MAX))
    throw std::runtime_error("GPU polygon count exceeds int32 ABI");
  if (offsets[0] != 0 || offsets[polygons] != entries)
    throw std::runtime_error("GPU polygon offsets");
  for (std::size_t i = 0; i < polygons; ++i)
    if (offsets[i] > offsets[i + 1])
      throw std::runtime_error("GPU polygon offset order");
}

}  // namespace

bool Available(Stats& stats) {
  const auto began = Clock::now();
  const cudaError_t pending = cudaPeekAtLastError();
  if (DeviceUnavailable(pending)) {
    ClearExpected(pending);
    stats.reason = "device-unavailable";
    stats.total_ms = ElapsedMs(began);
    return false;
  }
  Check(pending);
  if (!PreflightKernel(Edges, stats) || !PreflightKernel(Unite, stats) ||
      !PreflightKernel(Flatten, stats)) {
    stats.total_ms = ElapsedMs(began);
    return false;
  }
  stats.reason = "available";
  stats.total_ms = ElapsedMs(began);
  return true;
}

bool ConnectedComponents(std::size_t polygons,
                         std::size_t entries,
                         const std::uint64_t* offsets,
                         const std::uint32_t* vertices,
                         std::uint32_t* roots,
                         Stats& stats) {
  stats = Stats{};
  const auto began = Clock::now();
  auto reject = [&](const char* reason) {
    stats.reason = reason;
    stats.total_ms = ElapsedMs(began);
    return false;
  };
  ValidateShape(polygons, entries, offsets, vertices);
  if (polygons && !roots) throw std::runtime_error("GPU component null output");
  if (!polygons) {
    stats.reason = "complete";
    stats.total_ms = ElapsedMs(began);
    return true;
  }
  if (!Available(stats)) return reject(stats.reason);

  // A null-temp CUB query computes workspace size without allocating device
  // buffers or launching work. The typed dummy addresses are never accessed.
  std::size_t sort_bytes = 0;
  if (entries) {
    std::uint64_t key_in = 0, key_out = 0;
    std::uint32_t face_in = 0, face_out = 0;
    try {
      CheckPrelaunch(cub::DeviceRadixSort::SortPairs(nullptr,
                                                     sort_bytes,
                                                     &key_in,
                                                     &key_out,
                                                     &face_in,
                                                     &face_out,
                                                     int(entries),
                                                     0,
                                                     64));
    } catch (const PrelaunchAllocationFailure&) {
      ClearExpected(cudaErrorMemoryAllocation);
      return reject("device-allocation");
    } catch (const PrelaunchUnavailable& unavailable) {
      ClearExpected(unavailable.error);
      return reject(unavailable.reason);
    }
  }
  const std::uint64_t fixed_bytes = (polygons + 1ull) * sizeof(std::uint64_t) +
                                    2ull * entries * sizeof(std::uint64_t) +
                                    entries * sizeof(std::uint32_t) +
                                    2ull * entries * sizeof(std::uint32_t) +
                                    2ull * polygons * sizeof(std::uint32_t);
  stats.device_bytes = fixed_bytes + sort_bytes;
  if (!HasCapacity(stats.device_bytes, stats)) return reject(stats.reason);

  try {
    Stream stream;
    Event begin, uploaded, sorted, joined, downloaded;
    Device<std::uint64_t> d_offsets(polygons + 1), keys_a(entries),
        keys_b(entries);
    Device<std::uint32_t> d_vertices(entries), faces_a(entries),
        faces_b(entries), parents(polygons), d_roots(polygons);
    Device<unsigned char> scratch(sort_bytes);
    std::vector<std::uint32_t> result(polygons);

    // From here, all CUDA failures are execution failures and remain visible.
    Check(cudaEventRecord(begin.value, stream.value));
    Check(cudaMemcpyAsync(d_offsets.value,
                          offsets,
                          d_offsets.bytes,
                          cudaMemcpyHostToDevice,
                          stream.value));
    if (entries)
      Check(cudaMemcpyAsync(d_vertices.value,
                            vertices,
                            d_vertices.bytes,
                            cudaMemcpyHostToDevice,
                            stream.value));
    Edges<<<(polygons + 255) / 256, 256, 0, stream.value>>>(d_offsets.value,
                                                            d_vertices.value,
                                                            polygons,
                                                            keys_a.value,
                                                            faces_a.value,
                                                            parents.value);
    Check(cudaGetLastError());
    Check(cudaEventRecord(uploaded.value, stream.value));
    if (entries)
      Check(cub::DeviceRadixSort::SortPairs(scratch.value,
                                            sort_bytes,
                                            keys_a.value,
                                            keys_b.value,
                                            faces_a.value,
                                            faces_b.value,
                                            int(entries),
                                            0,
                                            64,
                                            stream.value));
    Check(cudaEventRecord(sorted.value, stream.value));
    if (entries) {
      Unite<<<(entries + 255) / 256, 256, 0, stream.value>>>(
          keys_b.value, faces_b.value, entries, parents.value);
      Check(cudaGetLastError());
    }
    Flatten<<<(polygons + 255) / 256, 256, 0, stream.value>>>(
        parents.value, d_roots.value, polygons);
    Check(cudaGetLastError());
    Check(cudaEventRecord(joined.value, stream.value));
    Check(cudaMemcpyAsync(result.data(),
                          d_roots.value,
                          d_roots.bytes,
                          cudaMemcpyDeviceToHost,
                          stream.value));
    Check(cudaEventRecord(downloaded.value, stream.value));
    Check(cudaStreamSynchronize(stream.value));
    stats.upload_ms = Duration(begin.value, uploaded.value);
    stats.sort_ms = Duration(uploaded.value, sorted.value);
    stats.union_ms = Duration(sorted.value, joined.value);
    stats.download_ms = Duration(joined.value, downloaded.value);
    std::copy(result.begin(), result.end(), roots);
  } catch (const PrelaunchAllocationFailure&) {
    ClearExpected(cudaErrorMemoryAllocation);
    return reject("device-allocation");
  } catch (const PrelaunchUnavailable& unavailable) {
    ClearExpected(unavailable.error);
    return reject(unavailable.reason);
  }
  stats.reason = "complete";
  stats.total_ms = ElapsedMs(began);
  return true;
}

}  // namespace PoissonTrimGpu
