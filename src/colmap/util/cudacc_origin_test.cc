// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace {
cudaError_t pending = cudaSuccess;
cudaError_t consumed = cudaSuccess;
int allocation_calls = 0;
int take_calls = 0;
}  // namespace

extern "C" cudaError_t __wrap_cudaPeekAtLastError() { return pending; }
extern "C" cudaError_t __wrap_cudaGetLastError() {
  ++take_calls;
  return consumed;
}

namespace colmap {
namespace {
class CudaAllocationOriginTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pending = consumed = cudaSuccess;
    allocation_calls = take_calls = 0;
  }
  CudaAllocationResult Allocate(cudaError_t result,
                                bool clean = true,
                                bool allow_unsupported = false) {
    return CudaTryAllocation(
        [&] {
          ++allocation_calls;
          return result;
        },
        [=] { return clean; },
        allow_unsupported);
  }
};

TEST_F(CudaAllocationOriginTest, DistinguishesSuccessAndOwnRefusal) {
  EXPECT_EQ(Allocate(cudaSuccess).origin, CudaAllocationOrigin::kSuccess);
  EXPECT_EQ(Allocate(cudaErrorMemoryAllocation).origin,
            CudaAllocationOrigin::kRefused);
  EXPECT_EQ(Allocate(cudaErrorNotSupported).origin,
            CudaAllocationOrigin::kRuntimeFailure);
  EXPECT_EQ(Allocate(cudaErrorNotSupported, true, true).origin,
            CudaAllocationOrigin::kRefused);
  EXPECT_EQ(allocation_calls, 4);
  EXPECT_EQ(take_calls, 2);
}

TEST_F(CudaAllocationOriginTest, PreexistingErrorDoesNotCallAllocation) {
  for (const auto error : {cudaErrorInvalidValue, cudaErrorMemoryAllocation}) {
    pending = error;
    const auto result = Allocate(cudaSuccess);
    EXPECT_EQ(result.origin, CudaAllocationOrigin::kPriorError);
    EXPECT_EQ(result.code, error);
  }
  EXPECT_EQ(allocation_calls, 0);
  EXPECT_EQ(take_calls, 0);
}

TEST_F(CudaAllocationOriginTest, UnresolvedOutputCannotPermitReadmission) {
  const auto result = Allocate(cudaErrorMemoryAllocation, false);
  EXPECT_EQ(result.origin, CudaAllocationOrigin::kUnresolvedOutput);
  EXPECT_EQ(result.code, cudaErrorMemoryAllocation);
  EXPECT_EQ(allocation_calls, 1);
  EXPECT_EQ(take_calls, 0);
}

TEST_F(CudaAllocationOriginTest, MismatchedLastErrorCannotPermitReadmission) {
  consumed = cudaErrorInvalidValue;
  const auto result = Allocate(cudaErrorMemoryAllocation);
  EXPECT_EQ(result.origin, CudaAllocationOrigin::kLastErrorMismatch);
  EXPECT_EQ(result.code, cudaErrorInvalidValue);
  EXPECT_EQ(allocation_calls, 1);
  EXPECT_EQ(take_calls, 1);
}

TEST_F(CudaAllocationOriginTest, NonResourceFailureDoesNotConsumeLastError) {
  EXPECT_EQ(Allocate(cudaErrorInvalidValue).origin,
            CudaAllocationOrigin::kRuntimeFailure);
  EXPECT_EQ(take_calls, 0);
}

TEST(CudaCleanupScopeTest, NestedObserversRetainFirstFailure) {
  EXPECT_FALSE(CudaCleanupScope::ObserveCleanup("outside", 1));
  CudaCleanupScope outer;
  {
    CudaCleanupScope inner;
    EXPECT_TRUE(CudaCleanupScope::ObserveCleanup(nullptr, 42, true));
    CudaCleanupScope::ObserveCleanup("later", 43);
    try {
      inner.Check();
      FAIL() << "Expected cleanup failure";
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find("unknown operation"), std::string::npos);
      EXPECT_NE(message.find("42"), std::string::npos);
      EXPECT_NE(message.find("driver"), std::string::npos);
      EXPECT_EQ(message.find("later"), std::string::npos);
    }
  }
  EXPECT_THROW(outer.Check(), std::runtime_error);
}

TEST(CudaCleanupScopeTest, CleanupFailureNestsOriginalAllocationFailure) {
  CudaCleanupScope scope;
  bool found_original = false;
  try {
    try {
      throw std::bad_alloc();
    } catch (...) {
      CudaCleanupScope::ObserveCleanup("failed free", 42);
      scope.Check();
      throw;
    }
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("failed free"), std::string::npos);
    try {
      std::rethrow_if_nested(error);
    } catch (const std::bad_alloc&) {
      found_original = true;
    }
  }
  EXPECT_TRUE(found_original);
}
}  // namespace
}  // namespace colmap
