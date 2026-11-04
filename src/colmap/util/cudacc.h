// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/cuda_allocation.h"
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

void CudaSafeCall(const cudaError_t error,
                  const std::string& file,
                  const int line);

// Stack-bound observer. It owns and frees nothing. Operation names must be
// static literals and all construction/destruction must occur on one thread.
class CudaCleanupScope {
 public:
  CudaCleanupScope() noexcept;
  ~CudaCleanupScope() noexcept;
  CudaCleanupScope(const CudaCleanupScope&) = delete;
  CudaCleanupScope& operator=(const CudaCleanupScope&) = delete;
  static bool ObserveCleanup(const char* operation,
                             int code,
                             bool driver = false) noexcept;
  void Check() const;

 private:
  static thread_local CudaCleanupScope* current_;
  CudaCleanupScope* previous_;
  const char* operation_ = nullptr;
  int code_ = 0;
  bool driver_ = false;
};

void CudaCheck(const char* file, const int line);
void CudaSyncAndCheck(const char* file,
                      const int line,
                      cudaStream_t stream = CudaDefaultStream());

}  // namespace colmap
