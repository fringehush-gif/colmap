// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/cuda_texture.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#if !defined(__linux__) || !defined(__GNUC__) || defined(__clang__)
#error "CUDA API wrapping requires Linux and GNU C++"
#endif

// GNU --wrap replaces external CUDA calls only. The texture constructor and
// CudaSafeCall use the actual COLMAP implementation; no GPU calls are
// forwarded.
namespace {

enum class Failure { kNone, kAllocation, kCreation, kCleanup };
Failure failure = Failure::kNone;
std::vector<std::string> calls;
std::vector<cudaError_t> reported_errors;
int array_token;
const cudaArray_t array_id = reinterpret_cast<cudaArray_t>(&array_token);
constexpr cudaTextureObject_t kTextureId = 42;
int free_attempts = 0;
int free_successes = 0;

}  // namespace

extern "C" cudaChannelFormatDesc __wrap_cudaCreateChannelDesc(
    int x, int y, int z, int w, cudaChannelFormatKind kind) {
  calls.emplace_back("channel");
  EXPECT_EQ(x, 8);
  EXPECT_EQ(y, 0);
  EXPECT_EQ(z, 0);
  EXPECT_EQ(w, 0);
  EXPECT_EQ(kind, cudaChannelFormatKindUnsigned);
  return {x, y, z, w, kind};
}

extern "C" cudaError_t __wrap_cudaMalloc3DArray(
    cudaArray_t* array,
    const cudaChannelFormatDesc* format,
    cudaExtent extent,
    unsigned int flags) {
  calls.emplace_back("allocate");
  EXPECT_EQ(format->x, 8);
  EXPECT_EQ(format->f, cudaChannelFormatKindUnsigned);
  EXPECT_EQ(extent.width, 3);
  EXPECT_EQ(extent.height, 2);
  EXPECT_EQ(extent.depth, 4);
  EXPECT_EQ(flags, cudaArrayLayered);
  if (failure == Failure::kAllocation) {
    return cudaErrorMemoryAllocation;
  }
  *array = array_id;
  return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaCreateTextureObject(
    cudaTextureObject_t* texture,
    const cudaResourceDesc* resource,
    const cudaTextureDesc* description,
    const cudaResourceViewDesc* view) {
  calls.emplace_back("create");
  EXPECT_EQ(resource->resType, cudaResourceTypeArray);
  EXPECT_EQ(resource->res.array.array, array_id);
  EXPECT_EQ(description->filterMode, cudaFilterModePoint);
  EXPECT_EQ(description->readMode, cudaReadModeElementType);
  EXPECT_EQ(description->addressMode[0], cudaAddressModeBorder);
  EXPECT_EQ(description->normalizedCoords, 0);
  EXPECT_EQ(view, nullptr);
  if (failure == Failure::kCreation || failure == Failure::kCleanup) {
    return cudaErrorInvalidValue;
  }
  *texture = kTextureId;
  return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaFreeArray(cudaArray_t array) {
  calls.emplace_back("free");
  EXPECT_EQ(array, array_id);
  ++free_attempts;
  if (failure == Failure::kCleanup) {
    return cudaErrorUnknown;
  }
  ++free_successes;
  return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaDestroyTextureObject(
    cudaTextureObject_t texture) {
  calls.emplace_back("destroy");
  EXPECT_EQ(texture, kTextureId);
  return cudaSuccess;
}

extern "C" const char* __wrap_cudaGetErrorString(cudaError_t error) {
  calls.emplace_back("error-string");
  reported_errors.push_back(error);
  switch (error) {
    case cudaErrorMemoryAllocation:
      return "injected allocation failure";
    case cudaErrorInvalidValue:
      return "injected texture creation failure";
    case cudaErrorUnknown:
      return "injected cleanup failure";
    default:
      ADD_FAILURE() << "Unexpected CUDA error: " << error;
      return "unexpected error";
  }
}

namespace colmap {
namespace mvs {
namespace {

class CudaTextureConstructor : public testing::Test {
 protected:
  void SetUp() override {
    failure = Failure::kNone;
    calls.clear();
    reported_errors.clear();
    free_attempts = 0;
    free_successes = 0;
    description_.addressMode[0] = cudaAddressModeBorder;
    description_.filterMode = cudaFilterModePoint;
    description_.readMode = cudaReadModeElementType;
  }

  void ExpectFailure(const std::string& original_message) {
    try {
      CudaArrayLayeredTexture<uint8_t> texture(description_, 3, 2, 4);
      FAIL() << "Expected CUDA constructor failure";
    } catch (const std::invalid_argument& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find(original_message), std::string::npos);
      EXPECT_NE(message.find("CUDA error at "), std::string::npos);
      EXPECT_EQ(message.find("injected cleanup failure"), std::string::npos);
    }
  }

  cudaTextureDesc description_ = {};
};

TEST_F(CudaTextureConstructor, DestroysTextureAndFreesArray) {
  {
    CudaArrayLayeredTexture<uint8_t> texture(description_, 3, 2, 4);
    EXPECT_EQ(texture.GetObj(), kTextureId);
    EXPECT_EQ(texture.GetWidth(), 3);
    EXPECT_EQ(texture.GetHeight(), 2);
    EXPECT_EQ(texture.GetDepth(), 4);
  }
  EXPECT_EQ(calls,
            (std::vector<std::string>{
                "channel", "allocate", "create", "destroy", "free"}));
  EXPECT_TRUE(reported_errors.empty());
  EXPECT_EQ(free_attempts, 1);
  EXPECT_EQ(free_successes, 1);
}

TEST_F(CudaTextureConstructor, AllocationFailureDoesNotFreeArray) {
  failure = Failure::kAllocation;
  ExpectFailure("injected allocation failure");
  EXPECT_EQ(calls,
            (std::vector<std::string>{"channel", "allocate", "error-string"}));
  EXPECT_EQ(reported_errors,
            (std::vector<cudaError_t>{cudaErrorMemoryAllocation}));
  EXPECT_EQ(free_attempts, 0);
  EXPECT_EQ(free_successes, 0);
}

TEST_F(CudaTextureConstructor, CreationFailureFreesArray) {
  failure = Failure::kCreation;
  ExpectFailure("injected texture creation failure");
  EXPECT_EQ(calls,
            (std::vector<std::string>{
                "channel", "allocate", "create", "error-string", "free"}));
  EXPECT_EQ(reported_errors, (std::vector<cudaError_t>{cudaErrorInvalidValue}));
  EXPECT_EQ(free_attempts, 1);
  EXPECT_EQ(free_successes, 1);
}

TEST_F(CudaTextureConstructor, CleanupFailurePreservesCreationError) {
  failure = Failure::kCleanup;
  ExpectFailure("injected texture creation failure");
  EXPECT_EQ(calls,
            (std::vector<std::string>{
                "channel", "allocate", "create", "error-string", "free"}));
  EXPECT_EQ(reported_errors, (std::vector<cudaError_t>{cudaErrorInvalidValue}));
  EXPECT_EQ(free_attempts, 1);
  EXPECT_EQ(free_successes, 0);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
