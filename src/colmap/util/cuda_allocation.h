// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef COLMAP_HIP_ENABLED
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime_api.h>
#endif

#include <utility>

namespace colmap {

namespace cuda_allocation_internal {
#ifdef COLMAP_HIP_ENABLED
using Error = hipError_t;
inline constexpr Error kSuccess = hipSuccess;
inline Error Peek() { return hipPeekAtLastError(); }
inline Error Take() { return hipGetLastError(); }
inline bool IsRefusal(Error status, bool allow_unsupported) {
  return status == hipErrorOutOfMemory ||
         (allow_unsupported && status == hipErrorNotSupported);
}
#else
using Error = cudaError_t;
inline constexpr Error kSuccess = cudaSuccess;
inline Error Peek() { return cudaPeekAtLastError(); }
inline Error Take() { return cudaGetLastError(); }
inline bool IsRefusal(Error status, bool allow_unsupported) {
  return status == cudaErrorMemoryAllocation ||
         (allow_unsupported && status == cudaErrorNotSupported);
}
#endif
}  // namespace cuda_allocation_internal

enum class CudaAllocationOrigin {
  kSuccess,
  kRefused,
  kPriorError,
  kLastErrorMismatch,
  kUnresolvedOutput,
  kRuntimeFailure,
};
struct CudaAllocationResult {
  cuda_allocation_internal::Error code;
  CudaAllocationOrigin origin;
};

// Calls allocate at most once. Exceptions from the supplied call/validator
// propagate unchanged, so caller-specific cleanup remains responsible for them.
template <typename Allocate, typename CleanFailure>
CudaAllocationResult CudaTryAllocation(Allocate&& allocate,
                                       CleanFailure&& clean_failure,
                                       bool allow_unsupported = false) {
  const auto prior = cuda_allocation_internal::Peek();
  if (prior != cuda_allocation_internal::kSuccess)
    return {prior, CudaAllocationOrigin::kPriorError};
  const auto status = std::forward<Allocate>(allocate)();
  if (status == cuda_allocation_internal::kSuccess)
    return {status, CudaAllocationOrigin::kSuccess};
  if (!std::forward<CleanFailure>(clean_failure)())
    return {status, CudaAllocationOrigin::kUnresolvedOutput};
  if (!cuda_allocation_internal::IsRefusal(status, allow_unsupported))
    return {status, CudaAllocationOrigin::kRuntimeFailure};
  const auto recorded = cuda_allocation_internal::Take();
  if (recorded != cuda_allocation_internal::kSuccess && recorded != status)
    return {recorded, CudaAllocationOrigin::kLastErrorMismatch};
  return {status, CudaAllocationOrigin::kRefused};
}

}  // namespace colmap
