// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include "colmap/util/logging.h"
#include "colmap/util/string.h"

namespace colmap {

CudaTimer::CudaTimer(cudaStream_t stream) : stream_(stream) {
  CUDA_SAFE_CALL(cudaEventCreate(&start_));
  CUDA_SAFE_CALL(cudaEventCreate(&stop_));
  CUDA_SAFE_CALL(cudaEventRecord(start_, stream_));
}

CudaTimer::~CudaTimer() {
  CUDA_SAFE_CALL(cudaEventDestroy(start_));
  CUDA_SAFE_CALL(cudaEventDestroy(stop_));
}

void CudaTimer::Print(const std::string& message) {
  CUDA_SAFE_CALL(cudaEventRecord(stop_, stream_));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop_));
  CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time_, start_, stop_));
  LOG(INFO) << StringPrintf(
      "%s: %.4fs", message.c_str(), elapsed_time_ / 1000.0f);
}

std::string FormatCudaError(cudaError_t error, const char* file, int line) {
  const char* name = cudaGetErrorName(error);
  const char* description = cudaGetErrorString(error);
  std::string message = StringPrintf(
      "CUDA error at %s:%i - %s (%i): %s",
      file,
      line,
      name != nullptr ? name : "unknown error",
      static_cast<int>(error),
      description != nullptr ? description : "No error description available");
  if (error == cudaErrorInsufficientDriver) {
    message +=
        "\nThe GPU driver is insufficient for this build. Update the "
        "graphics driver before using GPU features.";
#ifdef COLMAP_CUDA_ENABLED
    message += StringPrintf(" This build uses CUDA %i.%i.",
                            CUDART_VERSION / 1000,
                            (CUDART_VERSION % 1000) / 10);
#endif
  }
  return message;
}

void CudaSafeCall(const cudaError_t error,
                  const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
    LOG(FATAL_THROW) << FormatCudaError(error, file.c_str(), line);
  }
}

void CudaCheck(const char* file, const int line) {
  CudaSafeCall(cudaGetLastError(), file, line);
}

void CudaSyncAndCheck(const char* file, const int line, cudaStream_t stream) {
  const cudaError_t error = cudaStreamSynchronize(stream);
  if (error == cudaErrorInsufficientDriver) {
    CudaSafeCall(error, file, line);
  }
  if (cudaSuccess != error) {
    LOG(FATAL_THROW)
        << FormatCudaError(error, file, line)
        << "\nThis error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer "
           "to the FAQ in the documentation on how to solve this "
           "problem.";
  }
}

}  // namespace colmap
