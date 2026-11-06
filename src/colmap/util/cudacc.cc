// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include "colmap/util/logging.h"
#include "colmap/util/string.h"

#include <exception>

namespace colmap {

std::string GetCudaErrorMessage(const cudaError_t error) {
  const char* description = cudaGetErrorString(error);
  std::string message =
      StringPrintf("%s (error %d)",
                   description != nullptr && description[0] != '\0'
                       ? description
                       : "Unknown GPU runtime error",
                   static_cast<int>(error));
  if (error == cudaErrorInsufficientDriver) {
#if defined(COLMAP_HIP_ENABLED)
    message +=
        ". The GPU driver is missing or insufficient for this HIP "
        "runtime. Install a compatible AMD driver or use a compatible "
        "COLMAP build.";
#else
    message +=
        ". The NVIDIA driver is missing or insufficient for this CUDA "
        "runtime. Update the NVIDIA driver or use a COLMAP build "
        "compatible with the installed driver.";
#endif
  }
  return message;
}

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

void CudaSafeCall(const cudaError_t error,
                  const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
    LOG(FATAL_THROW) << StringPrintf("CUDA error at %s:%i - %s",
                                     file.c_str(),
                                     line,
                                     GetCudaErrorMessage(error).c_str());
  }
}

CudaAllocationRefused::CudaAllocationRefused(const char* operation,
                                             cudaError_t code)
    : std::invalid_argument(
          StringPrintf("%s: %s", operation, GetCudaErrorMessage(code).c_str())),
      operation_(operation),
      code_(code) {}

bool CudaIsAllocationRefusal(cudaError_t status, bool allow_unsupported) {
  return cuda_allocation_internal::IsRefusal(status, allow_unsupported);
}

void CudaAllocationFailure(const char* operation,
                           CudaAllocationResult result,
                           const char* file,
                           int line) {
  const char* reason = "runtime failure";
  switch (result.origin) {
    case CudaAllocationOrigin::kPriorError:
      reason = "preexisting error";
      break;
    case CudaAllocationOrigin::kLastErrorMismatch:
      reason = "last-error mismatch";
      break;
    case CudaAllocationOrigin::kUnresolvedOutput:
      reason = "unresolved output";
      break;
    case CudaAllocationOrigin::kRuntimeFailure:
      break;
    case CudaAllocationOrigin::kSuccess:
    case CudaAllocationOrigin::kRefused:
      reason = "invalid failure origin";
      break;
  }
  LOG(FATAL_THROW) << StringPrintf("CUDA allocation %s: %s at %s:%d - %s",
                                   operation,
                                   reason,
                                   file,
                                   line,
                                   GetCudaErrorMessage(result.code).c_str());
  std::terminate();
}

thread_local CudaCleanupScope* CudaCleanupScope::current_ = nullptr;

CudaCleanupScope::CudaCleanupScope() noexcept : previous_(current_) {
  current_ = this;
}
CudaCleanupScope::~CudaCleanupScope() noexcept {
  if (current_ != this) std::terminate();
  current_ = previous_;
}
bool CudaCleanupScope::ObserveCleanup(const char* operation,
                                      int code,
                                      bool driver) noexcept {
  const bool active = current_ != nullptr;
  if (code != 0) {
    for (auto* scope = current_; scope != nullptr; scope = scope->previous_) {
      if (scope->operation_ == nullptr) {
        scope->operation_ =
            operation != nullptr ? operation : "unknown operation";
        scope->code_ = code;
        scope->driver_ = driver;
      }
    }
  }
  return active;
}
void CudaCleanupScope::Check() const {
  if (operation_ != nullptr) {
    const std::runtime_error error(
        StringPrintf("CUDA %s cleanup %s failed (code %d)",
                      driver_ ? "driver" : "runtime",
                      operation_,
                      code_));
    if (std::current_exception()) {
      std::throw_with_nested(error);
    }
    throw error;
  }
}

void CudaCheck(const char* file, const int line) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    LOG(FATAL_THROW) << StringPrintf("CUDA error at %s:%i - %s",
                                     file,
                                     line,
                                     GetCudaErrorMessage(error).c_str());
  }
}

void CudaSyncAndCheck(const char* file, const int line, cudaStream_t stream) {
  const cudaError_t error = cudaStreamSynchronize(stream);
  if (cudaSuccess != error) {
    LOG(FATAL_THROW)
        << StringPrintf("CUDA error at %s:%i - %s",
                        file,
                        line,
                        GetCudaErrorMessage(error).c_str())
        << "\nThis error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer "
           "to the FAQ in the documentation on how to solve this "
           "problem.";
  }
}

}  // namespace colmap
