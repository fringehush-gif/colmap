// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/patch_match_hmm_checkpoint.h"
#include "colmap/mvs/patch_match_state_map.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#if !defined(__linux__) || !defined(__GNUC__)
#error "CUDA API wrapping requires Linux and GNU-compatible C++"
#endif

// GNU --wrap replaces external CUDA calls only. The texture constructor and
// CudaSafeCall use the actual COLMAP implementation; no GPU calls are
// forwarded.
namespace {

enum class Failure {
  kNone,
  kAllocation,
  kCreation,
  kCleanup,
  kPitch,
  kMalloc,
  kFree,
  kMallocAndFree,
  kDestroy
};
Failure failure = Failure::kNone;
std::vector<std::string> calls;
std::vector<cudaError_t> reported_errors;
int array_token;
const cudaArray_t array_id = reinterpret_cast<cudaArray_t>(&array_token);
constexpr cudaTextureObject_t kTextureId = 42;
int free_attempts = 0;
int free_successes = 0;
int malloc_attempts = 0;
int pitched_attempts = 0;
int device_free_attempts = 0;
int allocation_token;
void* allocation_id = &allocation_token;
cudaError_t peek_error = cudaSuccess;
cudaError_t last_error = cudaSuccess;
int copy_3d_attempts = 0;
int copy_2d_attempts = 0;
int memset_attempts = 0;
int first_copy_pitched_attempts = -1;

}  // namespace

extern "C" cudaError_t __wrap_cudaPeekAtLastError() { return peek_error; }
extern "C" cudaError_t __wrap_cudaGetLastError() { return last_error; }
extern "C" cudaError_t __wrap_cudaMemcpy3D(const cudaMemcpy3DParms*) {
  if (copy_3d_attempts++ == 0) first_copy_pitched_attempts = pitched_attempts;
  return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemcpy2D(
    void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind) {
  ++copy_2d_attempts;
  return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemset(void*, int, size_t) {
  ++memset_attempts;
  return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMallocPitch(void** ptr,
                                              size_t* pitch,
                                              size_t width,
                                              size_t) {
  ++pitched_attempts;
  if (failure == Failure::kPitch) return cudaErrorMemoryAllocation;
  *ptr = allocation_id;
  *pitch = width;
  return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMalloc(void** ptr, size_t) {
  ++malloc_attempts;
  if (failure == Failure::kMalloc || failure == Failure::kMallocAndFree)
    return cudaErrorMemoryAllocation;
  *ptr = allocation_id;
  return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaFree(void* ptr) {
  EXPECT_EQ(ptr, allocation_id);
  ++device_free_attempts;
  return failure == Failure::kFree || failure == Failure::kMallocAndFree
             ? cudaErrorUnknown
             : cudaSuccess;
}

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
  return failure == Failure::kDestroy ? cudaErrorUnknown : cudaSuccess;
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
    malloc_attempts = pitched_attempts = device_free_attempts = 0;
    peek_error = last_error = cudaSuccess;
    copy_3d_attempts = copy_2d_attempts = memset_attempts = 0;
    first_copy_pitched_attempts = -1;
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
      const bool allocation_failure = failure == Failure::kAllocation;
      const char* operation =
          allocation_failure ? "cudaMalloc3DArray" : "cudaCreateTextureObject";
      const auto code = allocation_failure ? cudaErrorMemoryAllocation
                                           : cudaErrorInvalidValue;
      EXPECT_NE(message.find(operation), std::string::npos);
      EXPECT_NE(message.find("error " + std::to_string(static_cast<int>(code))),
                std::string::npos);
      if (!allocation_failure) {
        EXPECT_NE(message.find("CUDA allocation cudaCreateTextureObject: "
                               "runtime failure at "),
                  std::string::npos);
      }
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

TEST_F(CudaTextureConstructor, PaddingAllocationPrecedesSliceCopies) {
  using Texture = CudaArrayLayeredTexture<uint8_t>;
  const uint8_t pixels[6] = {};
  std::vector<Texture::HostSlice> slices(4, {pixels, 3, 2, 3});
  slices[1] = {pixels, 2, 1, 3};
  failure = Failure::kPitch;
  EXPECT_THROW(Texture::FromHostSlices(description_, 3, 2, slices),
               CudaAllocationRefused);
  EXPECT_EQ(copy_3d_attempts, 0);
}

TEST_F(CudaTextureConstructor, MixedSlicesPreparePaddingOnce) {
  using Texture = CudaArrayLayeredTexture<uint8_t>;
  const uint8_t pixels[6] = {};
  std::vector<Texture::HostSlice> slices(4, {pixels, 3, 2, 3});
  slices[1] = {pixels, 2, 1, 3};
  auto texture = Texture::FromHostSlices(description_, 3, 2, slices);
  EXPECT_EQ(pitched_attempts, 1);
  EXPECT_EQ(memset_attempts, 1);
  EXPECT_EQ(copy_3d_attempts, 5);
  EXPECT_EQ(first_copy_pitched_attempts, 1);
}

TEST_F(CudaTextureConstructor, EqualSlicesNeedNoPadding) {
  using Texture = CudaArrayLayeredTexture<uint8_t>;
  const uint8_t pixels[6] = {};
  std::vector<Texture::HostSlice> slices(4, {pixels, 3, 2, 3});
  auto texture = Texture::FromHostSlices(description_, 3, 2, slices);
  EXPECT_EQ(pitched_attempts, 0);
  EXPECT_EQ(memset_attempts, 0);
  EXPECT_EQ(copy_3d_attempts, 4);
}

TEST_F(CudaTextureConstructor, DeviceArrayAndGpuMatCopyConsumers) {
  const uint8_t pixels[24] = {};
  uint8_t output[24] = {};
  GpuMat<uint8_t> mat(3, 2, 4);
  mat.CopyToDevice(pixels, 3);
  mat.CopyToHost(output, 3);
  auto texture =
      CudaArrayLayeredTexture<uint8_t>::FromGpuMat(description_, mat);
  EXPECT_EQ(copy_2d_attempts, 2);
  EXPECT_EQ(copy_3d_attempts, 1);
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

TEST_F(CudaTextureConstructor, RefusedPitchedAndContiguousAllocations) {
  failure = Failure::kPitch;
  EXPECT_THROW((GpuMat<float>(3, 2)), CudaAllocationRefused);
  EXPECT_EQ(pitched_attempts, 1);
  EXPECT_EQ(device_free_attempts, 0);
  failure = Failure::kMalloc;
  EXPECT_THROW((PatchMatchStateMap(3, 2, 1)), CudaAllocationRefused);
  EXPECT_THROW((PatchMatchHmmCheckpoint(3, 2, 1)), CudaAllocationRefused);
  EXPECT_EQ(malloc_attempts, 2);
  EXPECT_EQ(device_free_attempts, 0);
}

TEST_F(CudaTextureConstructor, ObserverRejectsFailedCleanupBeforeRetry) {
  bool retried = false;
  CudaCleanupScope outer;
  try {
    CudaCleanupScope inner;
    try {
      GpuMat<float> temporary(3, 2);
      failure = Failure::kFree;
      throw CudaAllocationRefused("injected", cudaErrorMemoryAllocation);
    } catch (const CudaAllocationRefused&) {
      inner.Check();
      retried = true;
    }
  } catch (const std::runtime_error&) {
    SUCCEED();
  }
  EXPECT_FALSE(retried);
  EXPECT_EQ(device_free_attempts, 1);
  EXPECT_THROW(outer.Check(), std::runtime_error);
}

TEST_F(CudaTextureConstructor, FailedAggregateConstructionReportsRelease) {
  struct Aggregate {
    GpuMat<float> first{3, 2};
    PatchMatchStateMap second{3, 2, 1};
  };
  failure = Failure::kMallocAndFree;
  CudaCleanupScope cleanup;
  bool retry = false;
  try {
    Aggregate aggregate;
    FAIL() << "Expected second member allocation refusal";
  } catch (const CudaAllocationRefused&) {
    try {
      cleanup.Check();
      retry = true;
    } catch (const std::runtime_error&) {
      SUCCEED();
    }
  }
  EXPECT_EQ(pitched_attempts, 1);
  EXPECT_EQ(malloc_attempts, 1);
  EXPECT_EQ(device_free_attempts, 1);
  EXPECT_FALSE(retry);
}

TEST_F(CudaTextureConstructor, PriorAndMismatchedErrorsAreFatal) {
  peek_error = cudaErrorUnknown;
  auto prior = CudaTryAllocation([] { return cudaErrorMemoryAllocation; },
                                 [] { return true; });
  EXPECT_EQ(prior.origin, CudaAllocationOrigin::kPriorError);
  EXPECT_EQ(prior.code, cudaErrorUnknown);
  EXPECT_THROW((GpuMat<float>(3, 2)), std::invalid_argument);
  EXPECT_EQ(pitched_attempts, 0);
  peek_error = cudaSuccess;
  last_error = cudaErrorUnknown;
  failure = Failure::kPitch;
  auto mismatch = CudaTryAllocation([] { return cudaErrorMemoryAllocation; },
                                    [] { return true; });
  EXPECT_EQ(mismatch.origin, CudaAllocationOrigin::kLastErrorMismatch);
  EXPECT_EQ(mismatch.code, cudaErrorUnknown);
  try {
    GpuMat<float> mat(3, 2);
    FAIL() << "Expected fatal mismatched CUDA error";
  } catch (const CudaAllocationRefused&) {
    FAIL() << "Mismatch must not become a retryable refusal";
  } catch (const std::invalid_argument&) {
    SUCCEED();
  }
  EXPECT_EQ(pitched_attempts, 1);
}

TEST_F(CudaTextureConstructor, ObserverAttemptsArrayFreeAfterDestroyFailure) {
  CudaCleanupScope cleanup;
  {
    CudaArrayLayeredTexture<uint8_t> texture(description_, 3, 2, 4);
    failure = Failure::kDestroy;
  }
  EXPECT_EQ(free_attempts, 1);
  EXPECT_EQ(free_successes, 1);
  EXPECT_THROW(cleanup.Check(), std::runtime_error);
}

TEST_F(CudaTextureConstructor, ObserverReportsTextureConstructorCleanup) {
  failure = Failure::kCleanup;
  CudaCleanupScope cleanup;
  EXPECT_THROW((CudaArrayLayeredTexture<uint8_t>(description_, 3, 2, 4)),
               std::invalid_argument);
  EXPECT_EQ(free_attempts, 1);
  EXPECT_THROW(cleanup.Check(), std::runtime_error);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
