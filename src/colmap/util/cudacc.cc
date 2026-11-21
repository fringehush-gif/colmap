// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include "colmap/util/logging.h"
#include "colmap/util/string.h"

#include <exception>
#include <stdexcept>

namespace colmap {

CudaTimer::CudaTimer(cudaStream_t stream)
    : stream_(stream), start_(nullptr), stop_(nullptr) {
  bool start_created = false;
  bool stop_created = false;
  try {
    const auto start_status = cudaEventCreate(&start_);
    start_created = start_status == cudaSuccess;
    CUDA_SAFE_CALL(start_status);
    const auto stop_status = cudaEventCreate(&stop_);
    stop_created = stop_status == cudaSuccess;
    CUDA_SAFE_CALL(stop_status);
    CUDA_SAFE_CALL(cudaEventRecord(start_, stream_));
  } catch (...) {
    if (start_created) {
      CudaCleanupScope::ObserveCleanup("cudaEventDestroy CudaTimer",
                                       cudaEventDestroy(start_));
    }
    if (stop_created) {
      CudaCleanupScope::ObserveCleanup("cudaEventDestroy CudaTimer",
                                       cudaEventDestroy(stop_));
    }
    throw;
  }
}

CudaTimer::~CudaTimer() {
  // Record release failures without throwing or skipping the second event.
  CudaCleanupScope::ObserveCleanup("cudaEventDestroy CudaTimer",
                                   cudaEventDestroy(start_));
  CudaCleanupScope::ObserveCleanup("cudaEventDestroy CudaTimer",
                                   cudaEventDestroy(stop_));
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
                                     cudaGetErrorString(error));
  }
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
    std::throw_with_nested(
        std::runtime_error(StringPrintf("CUDA %s cleanup %s failed (code %d)",
                                        driver_ ? "driver" : "runtime",
                                        operation_,
                                        code_)));
  }
}

void CudaCheck(const char* file, const int line) {
  const cudaError_t error = cudaGetLastError();
  while (error != cudaSuccess) {
    LOG(FATAL_THROW) << StringPrintf(
        "CUDA error at %s:%i - %s", file, line, cudaGetErrorString(error));
  }
}

void CudaSyncAndCheck(const char* file, const int line, cudaStream_t stream) {
  const cudaError_t error = cudaStreamSynchronize(stream);
  if (cudaSuccess != error) {
    LOG(FATAL_THROW)
        << StringPrintf("CUDA error at %s:%i - %s",
                        file,
                        line,
                        cudaGetErrorString(error))
        << "\nThis error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer "
           "to the FAQ in the documentation on how to solve this "
           "problem.";
  }
}

}  // namespace colmap
