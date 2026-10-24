// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "normal_gpu.h"
#include <cuda_runtime.h>

namespace PoissonNormalGpu {
namespace {

void Check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("normal GPU: ") +
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

struct AllocationFailure {};

template <class T>
struct Buffer {
  T* ptr = nullptr;
  explicit Buffer(std::size_t count) {
    if (!count) return;
    const cudaError_t error = cudaMalloc(&ptr, count * sizeof(T));
    if (error == cudaErrorMemoryAllocation) throw AllocationFailure{};
    Check(error);
  }
  ~Buffer() {
    if (ptr) cudaFree(ptr);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

__device__ bool FiniteBits(float value) {
  return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ void AddIeee(float* address, float value, unsigned* invalid) {
  // Integer CAS preserves subnormals unlike a native global float atomicAdd.
  unsigned* bits = reinterpret_cast<unsigned*>(address);
  unsigned old = atomicCAS(bits, 0u, 0u), assumed;
  do {
    assumed = old;
    const float sum = __fadd_rn(__uint_as_float(assumed), value);
    if (!FiniteBits(sum)) atomicOr(invalid, 2u);
    old = atomicCAS(bits, assumed, __float_as_uint(sum));
  } while (old != assumed);
}

__global__ void Scatter(const Record* records,
                        std::size_t record_count,
                        std::size_t node_count,
                        float* output,
                        unsigned* touched,
                        unsigned* invalid) {
  const std::size_t record =
      (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const unsigned lane = threadIdx.x & 31;
  if (record >= record_count || lane >= 27) return;
  const Record& item = records[record];
  const unsigned node = item.node[lane];
  if (node == UINT32_MAX) return;
  if (node >= node_count) {
    atomicOr(invalid, 1u);
    return;
  }
  const float coefficient = item.coefficient[lane];
  if (!FiniteBits(coefficient)) {
    atomicOr(invalid, 2u);
    return;
  }
  atomicExch(touched + node, 1u);
  for (int channel = 0; channel < 3; ++channel) {
    const float contribution = __fmul_rn(item.value[channel], coefficient);
    if (!FiniteBits(item.value[channel]) || !FiniteBits(contribution)) {
      atomicOr(invalid, 2u);
      continue;
    }
    AddIeee(output + std::size_t(node) * 3 + channel, contribution, invalid);
  }
}

}  // namespace

bool Available(Stats& stats) {
  auto reject = [&](const char* reason) {
    stats.reason = reason;
    return false;
  };
  const cudaError_t pending = cudaPeekAtLastError();
  if (DeviceUnavailable(pending)) {
    ClearExpected(pending);
    return reject("device-unavailable");
  }
  Check(pending);
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(&attributes, Scatter);
  if (DeviceUnavailable(error)) {
    ClearExpected(error);
    return reject("device-unavailable");
  }
  if (KernelUnavailable(error)) {
    ClearExpected(error);
    return reject("kernel-unavailable");
  }
  Check(error);
  if (attributes.maxThreadsPerBlock < 256) return reject("kernel-block-size");
  stats.reason = "available";
  return true;
}

bool Reduce(const Record* const* blocks,
            const std::size_t* counts,
            std::size_t block_count,
            std::size_t node_count,
            float* output,
            std::uint32_t* touched,
            Stats& stats) {
  auto reject = [&](const char* reason) {
    stats.reason = reason;
    return false;
  };
  if (!blocks || !counts || !output || !touched || !node_count ||
      node_count > 128000000)
    return reject("shape");
  std::size_t record_count = 0;
  for (std::size_t i = 0; i < block_count; ++i) {
    if (counts[i] > 8000000 - record_count || (counts[i] && !blocks[i]))
      return reject("records");
    record_count += counts[i];
  }
  if (!record_count) return reject("empty-records");

  const std::size_t record_bytes = record_count * sizeof(Record);
  const std::size_t field_bytes = node_count * 3 * sizeof(float);
  const std::size_t flag_bytes = node_count * sizeof(unsigned);
  const std::size_t required =
      record_bytes + field_bytes + flag_bytes + sizeof(unsigned);
  stats.device_bytes = required;
  if (required > (std::size_t(3) << 30)) return reject("workspace-budget");
  if (!Available(stats)) return false;
  std::size_t free_bytes = 0, total_bytes = 0;
  const cudaError_t memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (DeviceUnavailable(memory_status)) {
    ClearExpected(memory_status);
    return reject("device-unavailable");
  }
  Check(memory_status);
  if (required > free_bytes * 3 / 4) return reject("device-capacity");

  try {
    Buffer<Record> device_records(record_count);
    Buffer<float> device_output(node_count * 3);
    Buffer<unsigned> device_touched(node_count);
    Buffer<unsigned> device_invalid(1);
    Check(cudaMemset(device_output.ptr, 0, field_bytes));
    Check(cudaMemset(device_touched.ptr, 0, flag_bytes));
    Check(cudaMemset(device_invalid.ptr, 0, sizeof(unsigned)));
    std::size_t offset = 0;
    for (std::size_t i = 0; i < block_count; ++i) {
      if (counts[i])
        Check(cudaMemcpy(device_records.ptr + offset,
                         blocks[i],
                         counts[i] * sizeof(Record),
                         cudaMemcpyHostToDevice));
      offset += counts[i];
    }
    Scatter<<<unsigned((record_count * 32 + 255) / 256), 256>>>(
        device_records.ptr,
        record_count,
        node_count,
        device_output.ptr,
        device_touched.ptr,
        device_invalid.ptr);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    unsigned invalid = 0;
    Check(cudaMemcpy(&invalid,
                     device_invalid.ptr,
                     sizeof(unsigned),
                     cudaMemcpyDeviceToHost));
    if (invalid)
      throw std::runtime_error("normal GPU: invalid captured node or value");
    Check(cudaMemcpy(
        output, device_output.ptr, field_bytes, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(
        touched, device_touched.ptr, flag_bytes, cudaMemcpyDeviceToHost));
  } catch (const AllocationFailure&) {
    ClearExpected(cudaErrorMemoryAllocation);
    return reject("device-allocation");
  }
  stats.reason = "complete";
  return true;
}

bool ReduceSlabs(const Record* const* blocks,
                 const std::size_t* counts,
                 std::size_t block_count,
                 std::size_t node_count,
                 std::size_t slab_nodes,
                 const SlabCallbacks& callbacks,
                 Stats& stats) {
  auto reject = [&](const char* reason) {
    stats.reason = reason;
    return false;
  };
  if (!blocks || !counts || !callbacks.ready || !callbacks.consume ||
      !slab_nodes || !node_count || node_count > 128000000)
    return reject("shape");
  std::size_t record_count = 0;
  for (std::size_t i = 0; i < block_count; ++i) {
    if (counts[i] > 8000000 - record_count || (counts[i] && !blocks[i]))
      return reject("records");
    record_count += counts[i];
  }
  if (!record_count) return reject("empty-records");

  slab_nodes = std::min(slab_nodes, node_count);
  std::vector<float> host_field;
  std::vector<std::uint32_t> host_touched;
  try {
    host_field.resize(slab_nodes * 3);
    host_touched.resize(slab_nodes);
  } catch (const std::bad_alloc&) {
    return reject("host-readback-allocation");
  }
  const std::size_t record_bytes = record_count * sizeof(Record);
  const std::size_t field_bytes = node_count * 3 * sizeof(float);
  const std::size_t flag_bytes = node_count * sizeof(unsigned);
  const std::size_t required =
      record_bytes + field_bytes + flag_bytes + sizeof(unsigned);
  stats.device_bytes = required;
  if (required > (std::size_t(3) << 30)) return reject("workspace-budget");
  if (!Available(stats)) return false;
  std::size_t free_bytes = 0, total_bytes = 0;
  const cudaError_t memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (DeviceUnavailable(memory_status)) {
    ClearExpected(memory_status);
    return reject("device-unavailable");
  }
  Check(memory_status);
  if (required > free_bytes * 3 / 4) return reject("device-capacity");

  try {
    Buffer<Record> device_records(record_count);
    Buffer<float> device_output(node_count * 3);
    Buffer<unsigned> device_touched(node_count);
    Buffer<unsigned> device_invalid(1);
    Check(cudaMemset(device_output.ptr, 0, field_bytes));
    Check(cudaMemset(device_touched.ptr, 0, flag_bytes));
    Check(cudaMemset(device_invalid.ptr, 0, sizeof(unsigned)));
    std::size_t offset = 0;
    for (std::size_t i = 0; i < block_count; ++i) {
      if (counts[i])
        Check(cudaMemcpy(device_records.ptr + offset,
                         blocks[i],
                         counts[i] * sizeof(Record),
                         cudaMemcpyHostToDevice));
      offset += counts[i];
    }
    Scatter<<<unsigned((record_count * 32 + 255) / 256), 256>>>(
        device_records.ptr,
        record_count,
        node_count,
        device_output.ptr,
        device_touched.ptr,
        device_invalid.ptr);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    unsigned invalid = 0;
    Check(cudaMemcpy(&invalid,
                     device_invalid.ptr,
                     sizeof(unsigned),
                     cudaMemcpyDeviceToHost));
    if (invalid)
      throw std::runtime_error("normal GPU: invalid captured node or value");
    // Input blocks are no longer read below this point. All CUDA work and
    // invalid-value checks completed; any subsequent error must propagate.
    callbacks.ready(callbacks.context);
    for (std::size_t begin = 0; begin < node_count; begin += slab_nodes) {
      const std::size_t count = std::min(slab_nodes, node_count - begin);
      Check(cudaMemcpy(host_field.data(),
                       device_output.ptr + begin * 3,
                       count * 3 * sizeof(float),
                       cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(host_touched.data(),
                       device_touched.ptr + begin,
                       count * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost));
      callbacks.consume(callbacks.context,
                        begin,
                        count,
                        host_field.data(),
                        host_touched.data());
    }
  } catch (const AllocationFailure&) {
    ClearExpected(cudaErrorMemoryAllocation);
    return reject("device-allocation");
  }
  stats.reason = "complete";
  return true;
}

}  // namespace PoissonNormalGpu
