// SPDX-License-Identifier: BSD-3-Clause

// ORT 1.31.1 CUDA custom op for the pinned ALIKED N16ROT model.
// Inputs are the four post-Selu FP32 NCHW tensors. No Torch/Triton dependency.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "fused_concat_kernel.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "ort_status_boundary.h"

namespace {
struct FusedBoundaryTag;

Ort::Status Compute(OrtKernelContext& raw_context,
                    const Ort::Custom::Tensor<float>& a,
                    const Ort::Custom::Tensor<float>& b,
                    const Ort::Custom::Tensor<float>& c,
                    const Ort::Custom::Tensor<float>& d,
                    Ort::Custom::Tensor<float>& output) {
  try {
    const auto& shape = a.Shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 32 || shape[2] < 32 ||
        shape[3] < 32 || shape[2] % 32 || shape[3] % 32 ||
        shape[2] > std::numeric_limits<int>::max() ||
        shape[3] > std::numeric_limits<int>::max()) {
      return Ort::Status(
          "FusedMultiScaleConcat unsupported full-resolution shape",
          ORT_INVALID_ARGUMENT);
    }
    const int h = static_cast<int>(shape[2]);
    const int w = static_cast<int>(shape[3]);
    if (b.Shape() != std::vector<int64_t>{1, 32, h / 2, w / 2} ||
        c.Shape() != std::vector<int64_t>{1, 32, h / 8, w / 8} ||
        d.Shape() != std::vector<int64_t>{1, 32, h / 32, w / 32}) {
      return Ort::Status("FusedMultiScaleConcat branch shape mismatch",
                         ORT_INVALID_ARGUMENT);
    }
    const uint64_t uh = static_cast<uint64_t>(h);
    const uint64_t uw = static_cast<uint64_t>(w);
    constexpr uint64_t kChannels = 128;
    if (uh > std::numeric_limits<uint64_t>::max() / uw) {
      return Ort::Status("FusedMultiScaleConcat pixel count overflow",
                         ORT_INVALID_ARGUMENT);
    }
    const uint64_t pixels = uh * uw;
    if (pixels > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                     kChannels ||
        pixels > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                     (kChannels * sizeof(float))) {
      return Ort::Status("FusedMultiScaleConcat output byte count overflow",
                         ORT_INVALID_ARGUMENT);
    }
    constexpr uint64_t kPixelsPerGridX = 1024;
    const uint64_t grid_x = (pixels - 1) / kPixelsPerGridX + 1;
    if (grid_x > std::numeric_limits<unsigned>::max()) {
      return Ort::Status("FusedMultiScaleConcat grid dimension overflow",
                         ORT_INVALID_ARGUMENT);
    }
    int device = -1;
    cudaError_t cuda_status = cudaGetDevice(&device);
    if (cuda_status != cudaSuccess)
      return Ort::Status(cudaGetErrorString(cuda_status), ORT_FAIL);
    int max_grid_x = 0, max_grid_y = 0, max_threads = 0;
    cuda_status =
        cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, device);
    if (cuda_status != cudaSuccess)
      return Ort::Status(cudaGetErrorString(cuda_status), ORT_FAIL);
    cuda_status =
        cudaDeviceGetAttribute(&max_grid_y, cudaDevAttrMaxGridDimY, device);
    if (cuda_status != cudaSuccess)
      return Ort::Status(cudaGetErrorString(cuda_status), ORT_FAIL);
    cuda_status = cudaDeviceGetAttribute(
        &max_threads, cudaDevAttrMaxThreadsPerBlock, device);
    if (cuda_status != cudaSuccess)
      return Ort::Status(cudaGetErrorString(cuda_status), ORT_FAIL);
    if (grid_x > static_cast<uint64_t>(max_grid_x) || max_grid_y < 128 ||
        max_threads < 256) {
      return Ort::Status("FusedMultiScaleConcat launch exceeds device limits",
                         ORT_INVALID_ARGUMENT);
    }
    // Ort output and all input OrtValues remain owned by this execution frame.
    float* dst = output.Allocate({1, 128, h, w});
    Ort::KernelContext context(&raw_context);
    auto stream = static_cast<cudaStream_t>(context.GetGPUComputeStream());
    // A null CUDA stream is the valid default stream; the op is CUDA-EP-only.
    const cudaError_t launch_status =
        LaunchFusedConcat(a.Data(),
                          b.Data(),
                          c.Data(),
                          d.Data(),
                          dst,
                          h,
                          w,
                          static_cast<unsigned>(grid_x),
                          stream);  // No device synchronization.
    if (launch_status != cudaSuccess)
      return Ort::Status(cudaGetErrorString(launch_status), ORT_FAIL);
    return Ort::Status{};
  } catch (const std::exception& e) {
    return Ort::Status(e.what(), ORT_FAIL);
  }
}

}  // namespace

std::unique_ptr<Ort::Custom::OrtLiteCustomOp> CreateFusedConcatCustomOp() {
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> op{
      Ort::Custom::CreateLiteCustomOp("FusedMultiScaleConcat",
                                      "CUDAExecutionProvider",
                                      Compute,
                                      nullptr,
                                      1,
                                      1)};
  colmap::aliked_native::StatusBoundary<FusedBoundaryTag>::Apply(*op,
                                                                 Ort::GetApi());
  return op;
}
