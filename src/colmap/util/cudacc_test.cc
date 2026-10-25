// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include <exception>
#include <string>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CudaErrorTest, IncludesCodeNameAndLocation) {
  const std::string message =
      FormatCudaError(cudaErrorInvalidValue, "test.cc", 42);
  EXPECT_NE(message.find("test.cc:42"), std::string::npos);
#ifdef COLMAP_HIP_ENABLED
  EXPECT_NE(message.find("hipErrorInvalidValue"), std::string::npos);
#else
  EXPECT_NE(message.find("cudaErrorInvalidValue"), std::string::npos);
#endif
  EXPECT_NE(message.find("(1)"), std::string::npos);
}

TEST(CudaErrorTest, ExplainsDriverUpgrade) {
  const std::string message =
      FormatCudaError(cudaErrorInsufficientDriver, "test.cc", 1);
  EXPECT_NE(message.find("driver is insufficient"), std::string::npos);
  EXPECT_NE(message.find("Update the graphics driver"), std::string::npos);
#ifdef COLMAP_CUDA_ENABLED
  EXPECT_NE(message.find("This build uses CUDA"), std::string::npos);
#endif
}

TEST(CudaErrorTest, RetainsUnknownNumericCode) {
  const std::string message =
      FormatCudaError(static_cast<cudaError_t>(123456), "test.cc", 1);
  EXPECT_NE(message.find("123456"), std::string::npos);
  EXPECT_EQ(message.find("(null)"), std::string::npos);
}

TEST(CudaErrorTest, SuccessDoesNotThrow) {
  EXPECT_NO_THROW(CudaSafeCall(cudaSuccess, "test.cc", 1));
}

TEST(CudaErrorTest, SafeCallReportsDriverUpgrade) {
  try {
    CudaSafeCall(cudaErrorInsufficientDriver, "test.cc", 1);
    FAIL() << "Expected a CUDA exception";
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find("Update the graphics driver"),
              std::string::npos);
  }
}

}  // namespace
}  // namespace colmap
