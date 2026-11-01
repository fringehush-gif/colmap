// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include <string>

#include <gtest/gtest.h>

static cudaError_t pending = cudaSuccess;
static cudaError_t consumed = cudaSuccess;
static int calls = 0;
extern "C" cudaError_t __wrap_cudaPeekAtLastError() { return pending; }
extern "C" cudaError_t __wrap_cudaGetLastError() { return consumed; }
namespace {
class CudaAllocationOrigin : public ::testing::Test {
 protected:
  void SetUp() override {
    pending = consumed = cudaSuccess;
    calls = 0;
  }
  void Allocate(cudaError_t returned,
                bool clean = true,
                bool allow_unsupported = false) {
    colmap::CudaCheckedAllocation(
        "fixture allocation",
        [&] {
          ++calls;
          return returned;
        },
        [&] { return clean; },
        __FILE__,
        __LINE__,
        allow_unsupported);
  }
};
TEST_F(CudaAllocationOrigin, OomIsTypedAndCarriesOrigin) {
  consumed = cudaErrorMemoryAllocation;
  try {
    Allocate(cudaErrorMemoryAllocation);
    FAIL() << "Expected typed refusal";
  } catch (const colmap::CudaAllocationRefused& error) {
    EXPECT_EQ(error.Operation(), "fixture allocation");
    EXPECT_EQ(error.Code(), cudaErrorMemoryAllocation);
  }
  EXPECT_EQ(calls, 1);
}
TEST_F(CudaAllocationOrigin, PreexistingErrorDoesNotCallAllocation) {
  for (const auto error : {cudaErrorInvalidValue, cudaErrorMemoryAllocation}) {
    pending = error;
    try {
      Allocate(cudaSuccess);
      FAIL() << "Expected preexisting error";
    } catch (const colmap::CudaAllocationRefused&) {
      FAIL() << "An earlier error must not permit retry";
    } catch (const std::invalid_argument&) {
      SUCCEED();
    }
  }
  EXPECT_EQ(calls, 0);
}
TEST_F(CudaAllocationOrigin, UnsupportedRequiresExplicitPermission) {
  try {
    Allocate(cudaErrorNotSupported);
    FAIL() << "Expected unsupported allocation error";
  } catch (const colmap::CudaAllocationRefused&) {
    FAIL() << "Ordinary allocations must not permit unsupported retry";
  } catch (const std::invalid_argument&) {
    SUCCEED();
  }
  EXPECT_THROW(Allocate(cudaErrorNotSupported, true, true),
               colmap::CudaAllocationRefused);
  EXPECT_EQ(calls, 2);
}
TEST_F(CudaAllocationOrigin, NonResourceAndUnresolvedAreNotTyped) {
  for (const auto item : {0, 1}) {
    try {
      if (item == 0)
        Allocate(cudaErrorInvalidValue);
      else
        Allocate(cudaErrorMemoryAllocation, false);
      FAIL() << "Expected generic failure";
    } catch (const colmap::CudaAllocationRefused&) {
      FAIL() << "Must not allow retry";
    } catch (const std::exception&) {
      SUCCEED();
    }
  }
}
TEST_F(CudaAllocationOrigin, ClearedLastErrorStillAllowsTypedOom) {
  consumed = cudaSuccess;
  EXPECT_THROW(Allocate(cudaErrorMemoryAllocation),
               colmap::CudaAllocationRefused);
}
TEST_F(CudaAllocationOrigin, MismatchedLastErrorIsNotTyped) {
  consumed = cudaErrorInvalidValue;
  try {
    Allocate(cudaErrorMemoryAllocation);
    FAIL() << "Expected generic failure";
  } catch (const colmap::CudaAllocationRefused&) {
    FAIL() << "Must not allow retry";
  } catch (const std::exception&) {
    SUCCEED();
  }
}
TEST_F(CudaAllocationOrigin, StatusDistinguishesAllAllocationOutcomes) {
  using O = colmap::CudaAllocationOrigin;
  auto run = [](cudaError_t status, bool clean = true) {
    return colmap::CudaTryAllocation([=] { return status; },
                                     [=] { return clean; });
  };
  EXPECT_EQ(run(cudaSuccess).origin, O::kSuccess);
  EXPECT_EQ(run(cudaErrorMemoryAllocation).origin, O::kRefused);
  EXPECT_EQ(run(cudaErrorMemoryAllocation, false).origin, O::kUnresolvedOutput);
  EXPECT_EQ(run(cudaErrorInvalidValue).origin, O::kRuntimeFailure);
  pending = cudaErrorInvalidValue;
  auto prior = run(cudaSuccess);
  EXPECT_EQ(prior.origin, O::kPriorError);
  EXPECT_EQ(prior.code, cudaErrorInvalidValue);
  pending = cudaSuccess;
  consumed = cudaErrorInvalidValue;
  auto mismatch = run(cudaErrorMemoryAllocation);
  EXPECT_EQ(mismatch.origin, O::kLastErrorMismatch);
  EXPECT_EQ(mismatch.code, cudaErrorInvalidValue);
}
TEST_F(CudaAllocationOrigin, CleanupFailureNestsOriginalOom) {
  colmap::CudaCleanupScope scope;
  try {
    try {
      throw colmap::CudaAllocationRefused("fixture allocation",
                                          cudaErrorMemoryAllocation);
    } catch (...) {
      colmap::CudaCleanupScope::ObserveCleanup("failed free", 42);
      scope.Check();
      throw;
    }
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("failed free"), std::string::npos);
    try {
      std::rethrow_if_nested(error);
      FAIL() << "Expected original refusal";
    } catch (const colmap::CudaAllocationRefused& nested) {
      EXPECT_EQ(nested.Operation(), "fixture allocation");
      EXPECT_EQ(nested.Code(), cudaErrorMemoryAllocation);
    }
    return;
  }
  FAIL() << "Expected cleanup failure";
}
TEST_F(CudaAllocationOrigin, NullOperationStillRecordsFirstFailure) {
  colmap::CudaCleanupScope outer;
  {
    colmap::CudaCleanupScope inner;
    EXPECT_TRUE(colmap::CudaCleanupScope::ObserveCleanup(nullptr, 42, true));
    colmap::CudaCleanupScope::ObserveCleanup("later", 43);
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
TEST_F(CudaAllocationOrigin, NestedObserverRetainsFirstForAllScopes) {
  EXPECT_FALSE(colmap::CudaCleanupScope::ObserveCleanup("outside", 1));
  colmap::CudaCleanupScope outer;
  {
    colmap::CudaCleanupScope inner;
    EXPECT_TRUE(colmap::CudaCleanupScope::ObserveCleanup("first", 42, true));
    colmap::CudaCleanupScope::ObserveCleanup("second", 43);
    try {
      inner.Check();
      FAIL() << "Expected failure";
    } catch (const std::exception& e) {
      EXPECT_NE(std::string(e.what()).find("first"), std::string::npos);
      EXPECT_NE(std::string(e.what()).find("42"), std::string::npos);
    }
  }
  try {
    outer.Check();
    FAIL() << "Expected failure";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("first"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("42"), std::string::npos);
  }
}
}  // namespace
