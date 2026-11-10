// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/cuda_to_hip.h"

#include <string>

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK() CudaCheck(__FILE__, __LINE__)
#define CUDA_SYNC_AND_CHECK() CudaSyncAndCheck(__FILE__, __LINE__)

namespace colmap {

// Describe a runtime error, including its numeric code and driver guidance.
// Always returns a message, even if the runtime provides no description.
std::string GetCudaErrorMessage(cudaError_t error);

class CudaTimer {
 public:
  CudaTimer();
  ~CudaTimer();

  void Print(const std::string& message);

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  float elapsed_time_;
};

void CudaSafeCall(const cudaError_t error,
                  const std::string& file,
                  const int line);

namespace internal {
// Opt in at internal call sites that must not construct a filename on success.
inline void CudaSafeCallWithLocation(cudaError_t error,
                                     const char* file,
                                     int line) {
  if (error != cudaSuccess) ::colmap::CudaSafeCall(error, file, line);
}
}  // namespace internal

void CudaCheck(const char* file, const int line);
void CudaSyncAndCheck(const char* file, const int line);

}  // namespace colmap
