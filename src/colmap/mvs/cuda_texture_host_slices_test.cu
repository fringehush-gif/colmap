// SPDX-License-Identifier: BSD-3-Clause

#if defined(__CUDACC__)
#define BOOST_PP_VARIADICS 0
#endif

#include "colmap/mvs/cuda_texture.h"
#include "colmap/util/cuda.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

__global__ void ReadSamples(cudaTextureObject_t texture,
                            float* output,
                            int width,
                            int height,
                            int layers,
                            float offset) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = width * height * layers;
  if (index >= count) return;
  const int x = index % width;
  const int y = (index / width) % height;
  const int layer = index / (width * height);
  output[index] = tex2DLayered<float>(texture, x + offset, y + offset, layer);
}

std::vector<float> Read(cudaTextureObject_t texture,
                        int width,
                        int height,
                        int layers,
                        float offset) {
  const size_t count = static_cast<size_t>(width) * height * layers;
  float* device = nullptr;
  CUDA_SAFE_CALL(cudaMalloc(&device, count * sizeof(float)));
  ReadSamples<<<(count + 127) / 128, 128>>>(
      texture, device, width, height, layers, offset);
  CUDA_SAFE_CALL(cudaGetLastError());
  std::vector<float> output(count);
  CUDA_SAFE_CALL(cudaMemcpy(
      output.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(device));
  return output;
}

template <typename T>
void RunTextureSliceFixture(const bool mixed,
                            const cudaTextureDesc& descriptor) {
  constexpr int width = 13, height = 9, layers = 3;
  using Texture = CudaArrayLayeredTexture<T>;
  std::vector<std::vector<T>> storage(layers);
  std::vector<typename Texture::HostSlice> slices;
  std::vector<T> padded(width * height * layers, T{});
  for (int layer = 0; layer < layers; ++layer) {
    const int w = mixed ? width - layer * 3 : width;
    const int h = mixed ? height - layer * 2 : height;
    const int stride = w + 5;
    // Sentinel row padding must never reach the texture.
    storage[layer].assign(stride * h, static_cast<T>(193));
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        T value = static_cast<T>((x * 17 + y * 31 + layer * 43) % 256);
        storage[layer][y * stride + x] = value;
        padded[(layer * height + y) * width + x] = value;
      }
    }
    slices.push_back({storage[layer].data(),
                      static_cast<size_t>(w),
                      static_cast<size_t>(h),
                      stride * sizeof(T)});
  }
  if (sizeof(T) == sizeof(float)) {
    // Preserve signed zero, infinities, subnormals and a NaN payload in FP32.
    const uint32_t values[] = {
        0x80000000, 0x7f800000, 0xff800000, 0x00000001, 0x7fc01234};
    for (size_t i = 0; i < 5; ++i) {
      std::memcpy(&storage[0][i], &values[i], sizeof(T));
      std::memcpy(&padded[i], &values[i], sizeof(T));
    }
  }
  auto control =
      Texture::FromHostArray(descriptor, width, height, layers, padded.data());
  auto candidate = Texture::FromHostSlices(descriptor, width, height, slices);
  for (const float offset : {0.5f, 0.75f, -0.25f}) {
    const auto expected =
        Read(control->GetObj(), width, height, layers, offset);
    const auto observed =
        Read(candidate->GetObj(), width, height, layers, offset);
    if (std::memcmp(expected.data(),
                    observed.data(),
                    expected.size() * sizeof(float)) != 0) {
      throw std::runtime_error("Texture sample bits differ");
    }
  }
}

// All these checks precede CUDA allocation, so they need no GPU or driver.
TEST(CudaTextureHostSlices, RefusesInvalidSlicesBeforeAllocation) {
  using Texture = CudaArrayLayeredTexture<float>;
  float value = 1;
  cudaTextureDesc descriptor = {};
  const std::vector<std::vector<Texture::HostSlice>> invalid = {
      {},
      {{nullptr, 1, 1, 4}},
      {{&value, 0, 1, 4}},
      {{&value, 1, 0, 4}},
      {{&value, 2, 1, 8}},
      {{&value, 1, 2, 4}},
      {{&value, 1, 1, 3}}};
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_THROW(Texture::FromHostSlices(descriptor, 1, 1, invalid[i]),
                 std::exception);
  }
}

TEST(CudaTextureHostSlices, RefusesInvalidExtentsBeforeAllocation) {
  using Texture = CudaArrayLayeredTexture<float>;
  const size_t max_bytes = std::numeric_limits<size_t>::max();
  float value = 1;
  cudaTextureDesc descriptor = {};
  struct InvalidCase {
    size_t width;
    size_t height;
    std::vector<Texture::HostSlice> slices;
  };
  const std::vector<InvalidCase> invalid = {
      {0, 1, {{&value, 1, 1, sizeof(float)}}},
      {1, 0, {{&value, 1, 1, sizeof(float)}}},
      {max_bytes / sizeof(float) + 1, 1, {{&value, 1, 1, sizeof(float)}}},
      {1, max_bytes / sizeof(float) + 1, {{&value, 1, 1, sizeof(float)}}},
      {1, 2, {{&value, 1, 2, max_bytes}}}};
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    const auto& test = invalid[i];
    EXPECT_THROW(Texture::FromHostSlices(
                     descriptor, test.width, test.height, test.slices),
                 std::exception);
  }
}

class CudaTextureHostSlicesGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    // Visible device zero respects the launcher's external GPU UUID mask.
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }

  cudaTextureDesc ImageDescriptor() const {
    cudaTextureDesc descriptor = {};
    descriptor.addressMode[0] = cudaAddressModeBorder;
    descriptor.addressMode[1] = cudaAddressModeBorder;
    descriptor.filterMode = cudaFilterModeLinear;
    descriptor.readMode = cudaReadModeNormalizedFloat;
    return descriptor;
  }

  cudaTextureDesc DepthDescriptor() const {
    cudaTextureDesc descriptor = {};
    descriptor.addressMode[0] = cudaAddressModeBorder;
    descriptor.addressMode[1] = cudaAddressModeBorder;
    descriptor.filterMode = cudaFilterModePoint;
    descriptor.readMode = cudaReadModeElementType;
    return descriptor;
  }
};

TEST_F(CudaTextureHostSlicesGpu, EqualPitchedUint8LinearSamplesMatch) {
  ASSERT_NO_THROW(RunTextureSliceFixture<uint8_t>(false, ImageDescriptor()));
}

TEST_F(CudaTextureHostSlicesGpu, MixedPitchedUint8LinearSamplesMatch) {
  ASSERT_NO_THROW(RunTextureSliceFixture<uint8_t>(true, ImageDescriptor()));
}

TEST_F(CudaTextureHostSlicesGpu, EqualPitchedFloat32PointSamplesMatch) {
  ASSERT_NO_THROW(RunTextureSliceFixture<float>(false, DepthDescriptor()));
}

TEST_F(CudaTextureHostSlicesGpu, MixedPitchedFloat32PointSamplesMatch) {
  ASSERT_NO_THROW(RunTextureSliceFixture<float>(true, DepthDescriptor()));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
