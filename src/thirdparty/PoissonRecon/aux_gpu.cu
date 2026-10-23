// SPDX-License-Identifier: BSD-3-Clause
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "aux_gpu.h"
#include <cuda_runtime.h>

namespace AuxNativeGpu {
namespace {
using Clock = std::chrono::steady_clock;
double Elapsed(Clock::time_point a) {
  return std::chrono::duration<double>(Clock::now() - a).count();
}
void Check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("aux GPU: ") + cudaGetErrorString(e));
}
void ClearExpected(cudaError_t expected) {
  const cudaError_t pending = cudaPeekAtLastError();
  if (pending == expected) {
    const cudaError_t cleared = cudaGetLastError();
    if (cleared != expected) Check(cleared);
  } else
    Check(pending);
}
bool DeviceUnavailable(cudaError_t e) {
  return e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver;
}
bool KernelUnavailable(cudaError_t e) {
  return e == cudaErrorInvalidDeviceFunction ||
         e == cudaErrorNoKernelImageForDevice ||
         e == cudaErrorUnsupportedPtxVersion ||
         e == cudaErrorJitCompilerNotFound ||
         e == cudaErrorJitCompilationDisabled;
}
struct AllocationFailure {};
template <class T>
struct Buffer {
  T* p_ = nullptr;
  explicit Buffer(std::size_t n) {
    if (n) {
      const cudaError_t e = cudaMalloc(&p_, n * sizeof(T));
      if (e == cudaErrorMemoryAllocation) throw AllocationFailure{};
      Check(e);
    }
  }
  ~Buffer() {
    if (p_) cudaFree(p_);
  }
  Buffer(const Buffer&) = delete;
  void Put(const T* src, std::size_t n) {
    if (n) Check(cudaMemcpy(p_, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }
};
struct Events {
  cudaEvent_t a_ = nullptr, b_ = nullptr;
  Events() {
    Check(cudaEventCreate(&a_));
    try {
      Check(cudaEventCreate(&b_));
    } catch (...) {
      cudaEventDestroy(a_);
      throw;
    }
  }
  ~Events() {
    cudaEventDestroy(a_);
    cudaEventDestroy(b_);
  }
};
__global__ void Gather(std::uint32_t begin,
                       std::uint32_t end,
                       int depth,
                       const std::uint32_t* co,
                       const std::uint32_t* children,
                       const std::uint32_t* so,
                       const std::uint32_t* si,
                       const Four* seeds,
                       double* sums,
                       Four* output) {
  const std::uint32_t n = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= end) return;
  double value[4] = {};
  for (std::uint32_t j = so[n]; j < so[n + 1]; ++j) {
    const Four seed = seeds[si[j]];
#pragma unroll
    for (int c = 0; c < 4; ++c) value[c] += double(seed.v[c]);
  }
  for (std::uint32_t j = co[n]; j < co[n + 1]; ++j) {
    const std::uint32_t child = children[j];
#pragma unroll
    for (int c = 0; c < 4; ++c) value[c] += sums[4ull * child + c];
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    sums[4ull * n + c] = value[c];
    output[n].v[c] = float(ldexp(value[c], 2 * depth));
  }
}
}  // namespace
bool Available(Stats& s) {
  const auto started = Clock::now();
  auto reject = [&](const char* reason) {
    s.reason = reason;
    s.total_seconds = Elapsed(started);
    return false;
  };
  // Do not reinterpret an existing CUDA failure as missing kernel support.
  const cudaError_t pending = cudaPeekAtLastError();
  if (DeviceUnavailable(pending)) {
    ClearExpected(pending);
    return reject("device-unavailable");
  }
  Check(pending);
  cudaFuncAttributes attributes{};
  const cudaError_t e = cudaFuncGetAttributes(&attributes, Gather);
  if (DeviceUnavailable(e)) {
    ClearExpected(e);
    return reject("device-unavailable");
  }
  if (KernelUnavailable(e)) {
    ClearExpected(e);
    return reject("kernel-unavailable");
  }
  Check(e);
  if (attributes.maxThreadsPerBlock < 256) return reject("kernel-block-size");
  s.reason = "available";
  s.total_seconds = Elapsed(started);
  return true;
}
bool Reduce(const View& v, Four* output, Stats& s) {
  const auto started = Clock::now();
  auto reject = [&](const char* reason) {
    s.reason = reason;
    s.total_seconds = Elapsed(started);
    return false;
  };
  if (!v.nodes || !v.seeds || v.max_depth > 18 || !output)
    return reject("shape");
  if (!v.level_offsets || !v.child_offsets || !v.seed_offsets || !v.seed_ids ||
      !v.raw_seeds)
    return reject("null-view");
  if (v.level_offsets[0] || v.level_offsets[v.max_depth + 1] != v.nodes ||
      v.level_offsets[1] != 1 || v.child_offsets[0] ||
      v.child_offsets[v.nodes] != v.nodes - 1 || v.seed_offsets[0] ||
      v.seed_offsets[v.nodes] != v.seeds)
    return reject("CSR-endpoints");
  if (v.nodes > 1 && !v.children) return reject("null-children");
  for (std::uint32_t d = 0; d <= v.max_depth; ++d)
    if (v.level_offsets[d] > v.level_offsets[d + 1]) return reject("levels");
  for (std::uint32_t n = 0; n < v.nodes; ++n) {
    if (v.child_offsets[n] > v.child_offsets[n + 1] ||
        v.child_offsets[n + 1] > v.nodes - 1 ||
        v.child_offsets[n + 1] - v.child_offsets[n] > 8 ||
        v.seed_offsets[n] > v.seed_offsets[n + 1] ||
        v.seed_offsets[n + 1] > v.seeds)
      return reject("CSR-range");
    if (v.child_offsets[n] == v.child_offsets[n + 1] &&
        v.seed_offsets[n] == v.seed_offsets[n + 1])
      return reject("empty-compact-node");
    for (auto j = v.child_offsets[n]; j < v.child_offsets[n + 1]; ++j)
      if (v.children[j] <= n || v.children[j] >= v.nodes)
        return reject("child-order");
  }
  for (std::uint32_t i = 0; i < v.seeds; ++i) {
    if (v.seed_ids[i] >= v.seeds) return reject("seed-range");
    for (int c = 0; c < 4; ++c) {
      const double x = v.raw_seeds[i].v[c];
      if (!std::isfinite(x) || x < 0 ||
          (x != 0 && x < std::numeric_limits<float>::min()))
        return reject("nonnegative-normal-seed-gate");
      s.global_sums[c] += x;
      s.integral_channels += std::trunc(x) == x;
    }
  }
  // Protect every native FP32 partial sum and both native scaling stages, even
  // when the generic baseline accumulates in a different worker order.
  for (double x : s.global_sums)
    if (2 * std::ldexp(x, 7 * int(v.max_depth)) >
        double(std::numeric_limits<float>::max()))
      return reject("native-finite-bound");
  s.device_bytes = 4ull * (2ull * (v.nodes + 1) + (v.nodes - 1) + v.seeds) +
                   16ull * v.seeds + 48ull * v.nodes;
  if (s.device_bytes > 1536ull * 1024 * 1024) return reject("workspace-budget");
  if (!Available(s)) return reject(s.reason);
  std::size_t free = 0, total = 0;
  const cudaError_t memory_status = cudaMemGetInfo(&free, &total);
  if (DeviceUnavailable(memory_status)) {
    ClearExpected(memory_status);
    return reject("device-unavailable");
  }
  Check(memory_status);
  if (s.device_bytes > free * 3 / 4) return reject("device-capacity");
  try {
    Buffer<std::uint32_t> co(v.nodes + 1), children(v.nodes - 1),
        so(v.nodes + 1), si(v.seeds);
    Buffer<Four> seeds(v.seeds), out(v.nodes);
    Buffer<double> sums(4ull * v.nodes);
    Events events;
    const auto upload = Clock::now();
    co.Put(v.child_offsets, v.nodes + 1);
    children.Put(v.children, v.nodes - 1);
    so.Put(v.seed_offsets, v.nodes + 1);
    si.Put(v.seed_ids, v.seeds);
    seeds.Put(v.raw_seeds, v.seeds);
    s.upload_seconds = Elapsed(upload);
    Check(cudaEventRecord(events.a_));
    for (int d = int(v.max_depth); d >= 0; --d) {
      const auto begin = v.level_offsets[d], end = v.level_offsets[d + 1];
      if (end > begin)
        Gather<<<(end - begin + 255) / 256, 256>>>(begin,
                                                   end,
                                                   d,
                                                   co.p_,
                                                   children.p_,
                                                   so.p_,
                                                   si.p_,
                                                   seeds.p_,
                                                   sums.p_,
                                                   out.p_);
    }
    Check(cudaGetLastError());
    Check(cudaEventRecord(events.b_));
    Check(cudaEventSynchronize(events.b_));
    float ms = 0;
    Check(cudaEventElapsedTime(&ms, events.a_, events.b_));
    s.kernel_seconds = ms * .001;
    const auto download = Clock::now();
    Check(cudaMemcpy(output, out.p_, 16ull * v.nodes, cudaMemcpyDeviceToHost));
    s.download_seconds = Elapsed(download);
  } catch (const AllocationFailure&) {
    ClearExpected(cudaErrorMemoryAllocation);
    return reject("device-allocation");
  }
  s.reason = "complete";
  s.total_seconds = Elapsed(started);
  return true;
}
}  // namespace AuxNativeGpu
