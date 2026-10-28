// SPDX-License-Identifier: MIT
// ORT 1.31.1 / CUDA 13.2 descriptor factory.
#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "normalized_grid_sample_kernel.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "ort_status_boundary.h"

namespace {
struct SampleBoundaryTag;

Ort::Status CudaStatus(cudaError_t status) {
  return status == cudaSuccess
             ? Ort::Status{}
             : Ort::Status(cudaGetErrorString(status), ORT_FAIL);
}

Ort::Status DevicePointer(const void* ptr, int device) {
  if (!ptr)
    return Ort::Status("NormalizedGridSample null pointer",
                       ORT_INVALID_ARGUMENT);
  cudaPointerAttributes attrs{};
  const auto status = cudaPointerGetAttributes(&attrs, ptr);
  if (status != cudaSuccess) return CudaStatus(status);
  if (attrs.type != cudaMemoryTypeDevice || attrs.device != device) {
    return Ort::Status("NormalizedGridSample requires same-device CUDA tensors",
                       ORT_INVALID_ARGUMENT);
  }
  return Ort::Status{};
}

Ort::Status ComputeSample(OrtKernelContext& raw_context,
                          const Ort::Custom::Tensor<float>& raw,
                          const Ort::Custom::Tensor<float>& norm,
                          const Ort::Custom::Tensor<float>& grid,
                          Ort::Custom::Tensor<float>& output) {
  try {
    const auto& shape = raw.Shape();
    const auto& gs = grid.Shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 128 ||
        shape[2] <= 0 || shape[3] <= 0 ||
        norm.Shape() != std::vector<int64_t>{1, 1, shape[2], shape[3]} ||
        gs.size() != 4 || gs[0] != 1 || gs[1] < 0 || gs[2] != 1 || gs[3] != 2) {
      return Ort::Status("NormalizedGridSample input shape mismatch",
                         ORT_INVALID_ARGUMENT);
    }
    const uint64_t h = static_cast<uint64_t>(shape[2]),
                   w = static_cast<uint64_t>(shape[3]);
    const uint64_t samples = static_cast<uint64_t>(gs[1]);
    constexpr uint64_t max_count =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 128;
    constexpr uint64_t max_bytes =
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
        (128 * sizeof(float));
    if (h > std::numeric_limits<uint64_t>::max() / w ||
        h * w > std::min(max_count, max_bytes) ||
        samples > std::min(max_count, max_bytes)) {
      return Ort::Status("NormalizedGridSample element/byte count overflow",
                         ORT_INVALID_ARGUMENT);
    }
    Ort::KernelContext context(&raw_context);
    for (size_t input = 0; input < 3; ++input) {
      if (context.GetInput(input)
              .GetTensorTypeAndShapeInfo()
              .GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Ort::Status("NormalizedGridSample requires FP32 inputs",
                           ORT_INVALID_ARGUMENT);
      }
    }
    float* dst = output.Allocate({1, 128, gs[1], 1});
    // Empty grids need a valid empty OrtValue, but no device pointer or launch.
    if (samples == 0) return Ort::Status{};
    int device = -1, stream_device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return CudaStatus(status);
    const auto stream =
        static_cast<cudaStream_t>(context.GetGPUComputeStream());
    status = cudaStreamGetDevice(stream, &stream_device);
    if (status != cudaSuccess) return CudaStatus(status);
    if (stream_device != device) {
      return Ort::Status(
          "NormalizedGridSample ORT stream/current device mismatch",
          ORT_INVALID_ARGUMENT);
    }
    for (const float* ptr : {raw.Data(),
                             norm.Data(),
                             grid.Data(),
                             static_cast<const float*>(dst)}) {
      auto pointer_status = DevicePointer(ptr, device);
      if (!pointer_status.IsOK()) return pointer_status;
    }
    if (dst == raw.Data() || dst == norm.Data() || dst == grid.Data()) {
      return Ort::Status("NormalizedGridSample requires separate output",
                         ORT_INVALID_ARGUMENT);
    }
    int max_x = 0, max_y = 0, max_threads = 0;
    status = cudaDeviceGetAttribute(&max_x, cudaDevAttrMaxGridDimX, device);
    if (status != cudaSuccess) return CudaStatus(status);
    status = cudaDeviceGetAttribute(&max_y, cudaDevAttrMaxGridDimY, device);
    if (status != cudaSuccess) return CudaStatus(status);
    status = cudaDeviceGetAttribute(
        &max_threads, cudaDevAttrMaxThreadsPerBlock, device);
    if (status != cudaSuccess) return CudaStatus(status);
    if (max_x <= 0 || max_y < 128 || max_threads < 256) {
      return Ort::Status("NormalizedGridSample launch exceeds device limits",
                         ORT_INVALID_ARGUMENT);
    }
    const auto blocks = static_cast<unsigned>(std::min<uint64_t>(
        (samples - 1) / 256 + 1,
        std::min<uint64_t>(static_cast<uint64_t>(max_x), 65535)));
    return CudaStatus(LaunchNormalizedGridSample(raw.Data(),
                                                 norm.Data(),
                                                 grid.Data(),
                                                 dst,
                                                 shape[2],
                                                 shape[3],
                                                 samples,
                                                 blocks,
                                                 stream));
  } catch (const std::exception& e) {
    return Ort::Status(e.what(), ORT_FAIL);
  }
}

}  // namespace

std::unique_ptr<Ort::Custom::OrtLiteCustomOp>
CreateNormalizedGridSampleCustomOp() {
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> op{
      Ort::Custom::CreateLiteCustomOp("NormalizedGridSample",
                                      "CUDAExecutionProvider",
                                      ComputeSample,
                                      nullptr,
                                      1,
                                      1)};
  colmap::aliked_native::StatusBoundary<SampleBoundaryTag>::Apply(
      *op, Ort::GetApi());
  return op;
}
