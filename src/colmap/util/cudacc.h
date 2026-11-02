// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/cuda_allocation.h"
#include "colmap/util/cuda_to_hip.h"

#include <stdexcept>
#include <string>
#include <utility>

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
// out must be an initialized pointer/handle whose zero value means no owner.
#define CUDA_SAFE_ALLOCATE(operation, out, call) \
  ::colmap::CudaCheckedAllocation(               \
      operation,                                 \
      [&]() { return (call); },                  \
      [&]() { return (out) == 0; },              \
      __FILE__,                                  \
      __LINE__)
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

// Describe a runtime error, including its numeric code and driver guidance.
// Always returns a message, even if the runtime provides no description.
std::string GetCudaErrorMessage(cudaError_t error);

class CudaTimer {
 public:
  // Pass an explicit stream to time work on that stream.
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

// A refusal identifies the allocation call only; it does not certify cleanup.
class CudaAllocationRefused : public std::invalid_argument {
 public:
  CudaAllocationRefused(const char* operation, cudaError_t code);
  const std::string& Operation() const noexcept { return operation_; }
  cudaError_t Code() const noexcept { return code_; }

 private:
  std::string operation_;
  cudaError_t code_;
};

bool CudaIsAllocationRefusal(cudaError_t status, bool allow_unsupported);
[[noreturn]] void CudaAllocationFailure(const char* operation,
                                        CudaAllocationResult result,
                                        const char* file,
                                        int line);

template <typename Allocate, typename CleanFailure>
void CudaCheckedAllocation(const char* operation,
                           Allocate&& allocate,
                           CleanFailure&& clean_failure,
                           const char* file,
                           int line,
                           bool allow_unsupported = false) {
  const auto result =
      CudaTryAllocation(std::forward<Allocate>(allocate),
                        std::forward<CleanFailure>(clean_failure),
                        allow_unsupported);
  if (result.origin == CudaAllocationOrigin::kSuccess) return;
  if (result.origin == CudaAllocationOrigin::kRefused)
    throw CudaAllocationRefused(operation, result.code);
  CudaAllocationFailure(operation, result, file, line);
}

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
