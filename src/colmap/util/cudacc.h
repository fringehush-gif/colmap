// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/cuda_to_hip.h"

#include <string>

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK() CudaCheck(__FILE__, __LINE__)
#define CUDA_SYNC_AND_CHECK() CudaSyncAndCheck(__FILE__, __LINE__)

namespace colmap {

// COLMAP CUDA translation units use per-thread default streams. Host-only
// translation units must select that stream explicitly. HIP retains stream 0.
inline cudaStream_t CudaDefaultStream() {
#ifdef COLMAP_HIP_ENABLED
  return nullptr;
#else
  return cudaStreamPerThread;
#endif
}

class CudaTimer {
 public:
  // Pass nullptr for the legacy stream, or an explicitly managed stream.
  explicit CudaTimer(cudaStream_t stream = CudaDefaultStream());
  ~CudaTimer();

  void Print(const std::string& message);

 private:
  cudaStream_t stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
  float elapsed_time_;
};

// Include the numeric code even when the runtime cannot describe an error.
// Driver incompatibility additionally includes an actionable upgrade message.
std::string FormatCudaError(cudaError_t error, const char* file, int line);

void CudaSafeCall(const cudaError_t error,
                  const std::string& file,
                  const int line);

void CudaCheck(const char* file, const int line);
void CudaSyncAndCheck(const char* file,
                      const int line,
                      cudaStream_t stream = CudaDefaultStream());

}  // namespace colmap
