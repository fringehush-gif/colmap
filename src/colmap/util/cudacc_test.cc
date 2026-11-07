// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include <stdexcept>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CudaSafeCall, Success) {
  EXPECT_NO_THROW(CudaSafeCall(cudaSuccess, "probe.cc", 42));
}

TEST(CudaSafeCall, InsufficientDriver) {
  try {
    CudaSafeCall(cudaErrorInsufficientDriver, "probe.cc", 42);
    FAIL() << "Expected a runtime error";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_THAT(message, testing::HasSubstr("probe.cc:42"));
    EXPECT_THAT(
        message,
        testing::HasSubstr("error " + std::to_string(static_cast<int>(
                                          cudaErrorInsufficientDriver))));
    EXPECT_THAT(message, testing::HasSubstr("driver"));
    EXPECT_THAT(message, testing::HasSubstr("COLMAP build"));
    EXPECT_THAT(message, testing::Not(testing::HasSubstr("(null)")));
  }
}

TEST(CudaSafeCall, UnknownErrorPreservesCode) {
  try {
    CudaSafeCall(static_cast<cudaError_t>(123456), "probe.cc", 7);
    FAIL() << "Expected a runtime error";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_THAT(message, testing::HasSubstr("123456"));
    EXPECT_THAT(message, testing::HasSubstr("probe.cc:7"));
    EXPECT_THAT(message, testing::Not(testing::HasSubstr("(null)")));
    EXPECT_THAT(message,
                testing::Not(testing::HasSubstr("Update the NVIDIA driver")));
  }
}

class CudaCheckTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int num_devices = 0;
    const auto status = cudaGetDeviceCount(&num_devices);
    cudaGetLastError();
    if (status != cudaSuccess || num_devices == 0) {
      GTEST_SKIP() << "No CUDA/HIP device available";
    }
  }

  void TearDown() override { cudaGetLastError(); }

  void SetPendingError() {
    ASSERT_NE(cudaSetDevice(-1), cudaSuccess);
    ASSERT_NE(cudaPeekAtLastError(), cudaSuccess);
  }
};

TEST_F(CudaCheckTest, Success) {
  EXPECT_NO_THROW(CudaCheck("probe.cc", 42));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(CudaCheckTest, ErrorIsThrownAndConsumed) {
  ASSERT_NO_FATAL_FAILURE(SetPendingError());
  try {
    CudaCheck("probe.cc", 42);
    FAIL() << "Expected a CUDA runtime error";
  } catch (const std::invalid_argument& error) {
    EXPECT_THAT(error.what(), testing::HasSubstr("probe.cc:42"));
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(CudaCheckTest, UnwindingPreservesOriginalException) {
  ASSERT_NO_FATAL_FAILURE(SetPendingError());
  struct OriginalError {};
  struct CheckOnExit {
    bool* checked;
    ~CheckOnExit() {
      CudaCheck("probe.cc", 42);
      *checked = true;
    }
  };
  bool checked = false;
  try {
    const CheckOnExit check_on_exit{&checked};
    throw OriginalError{};
  } catch (const OriginalError&) {
    EXPECT_TRUE(checked);
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace
}  // namespace colmap
