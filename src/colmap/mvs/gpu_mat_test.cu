// SPDX-License-Identifier: BSD-3-Clause

#if defined(__CUDACC__) || defined(__HIPCC__)
#define BOOST_PP_VARIADICS 0
#endif

#include "colmap/math/math.h"
#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/gpu_mat_prng.h"
#include "colmap/mvs/packed_ncc_cost_map.h"
#include "colmap/mvs/patch_match_hmm_checkpoint.h"
#include "colmap/mvs/patch_match_probability_unorm16.h"
#include "colmap/mvs/patch_match_sampling.cuh"
#include "colmap/mvs/patch_match_state_map.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {

namespace {
__global__ void CheckSourceSampling(int* errors) {
  int failures = 0;
  for (int i = 0; i < 3; ++i) {
    const int stride = i == 0 ? 1 : (i == 1 ? 8 : 32);
    float probs[96] = {};
    failures += TransformPDFToCDF(probs, 3, stride);
    failures += SampleSource(probs, 3, stride, FLT_EPSILON / 2) != -1;
    failures += SampleSource(probs, 3, stride, 1.0f) != -1;
    probs[stride] = 1.0f;
    probs[2 * stride] = 3.0f;
    failures += !TransformPDFToCDF(probs, 3, stride);
    failures +=
        probs[0] != 0.0f || probs[stride] != 0.25f || probs[2 * stride] != 1.0f;
    failures += SampleSource(probs, 3, stride, FLT_EPSILON / 2) != 1;
    failures += SampleSource(probs, 3, stride, 0.5f) != 2;
    failures += SampleSource(probs, 3, stride, 1.0f) != 2;
  }
  float invalid[2] = {-1.0f, 1.0f};
  failures += !TransformPDFToCDF(invalid, 2, 1);
  failures += SampleSource(invalid, 2, 1, 0.5f) != -1;
  invalid[0] = __int_as_float(0x7fc00000);
  invalid[1] = 0.0f;
  failures += !TransformPDFToCDF(invalid, 2, 1);
  failures += SampleSource(invalid, 2, 1, 0.5f) != -1;
  *errors = failures;
}
}  // namespace

TEST(PatchMatchSampling, DeviceSupportAndLowerEndpoint) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "No CUDA or HIP device available";
  }
  GpuMat<int> errors(1, 1);
  CheckSourceSampling<<<1, 1>>>(errors.GetPtr());
  CUDA_SYNC_AND_CHECK();
  int actual = -1;
  errors.CopyToHost(&actual, sizeof(actual));
  EXPECT_EQ(actual, 0);
}

TEST(GpuMat, FillWithVector) {
  GpuMat<float> array(100, 100, 2);
  const std::vector<float> vector = {1.0f, 2.0f};
  array.FillWithVector(vector.data());

  std::vector<float> array_host(100 * 100 * 2, 0.0f);
  array.CopyToHost(array_host.data(), 100 * sizeof(float));

  for (size_t r = 0; r < 100; ++r) {
    for (size_t c = 0; c < 100; ++c) {
      EXPECT_EQ(array_host[0 * 100 * 100 + r * 100 + c], 1.0f);
      EXPECT_EQ(array_host[1 * 100 * 100 + r * 100 + c], 2.0f);
    }
  }
}

template <typename T>
void TestTransposeImage(const size_t width,
                        const size_t height,
                        const size_t depth) {
  GpuMat<T> array(width, height, depth);

  GpuMatPRNG prng_array(width, height);
  array.FillWithRandomNumbers(T(0.0), T(100.0), prng_array);

  GpuMat<T> array_transposed(height, width, depth);
  array.Transpose(&array_transposed);

  std::vector<T> array_host(width * height * depth, T(0.0));
  array.CopyToHost(array_host.data(), width * sizeof(T));

  std::vector<T> array_transposed_host(width * height * depth, 0);
  array_transposed.CopyToHost(array_transposed_host.data(), height * sizeof(T));

  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      for (size_t d = 0; d < depth; ++d) {
        EXPECT_EQ(array_host[d * width * height + r * width + c],
                  array_transposed_host[d * width * height + c * height + r]);
      }
    }
  }
}

TEST(GpuMat, Transpose) {
  for (size_t w = 1; w <= 5; ++w) {
    for (size_t h = 1; h <= 5; ++h) {
      for (size_t d = 1; d <= 3; ++d) {
        const size_t width = 20 * w;
        const size_t height = 20 * h;
        TestTransposeImage<int8_t>(width, height, d);
        TestTransposeImage<int16_t>(width, height, d);
        TestTransposeImage<int32_t>(width, height, d);
        TestTransposeImage<int64_t>(width, height, d);
        TestTransposeImage<float>(width, height, d);
        TestTransposeImage<double>(width, height, d);
      }
    }
  }
}

template <typename T>
void TestFlipHorizontalImage(const size_t width,
                             const size_t height,
                             const size_t depth) {
  GpuMat<T> array(width, height, depth);

  GpuMatPRNG prng_array(width, height);
  array.FillWithRandomNumbers(T(0.0), T(100.0), prng_array);

  GpuMat<T> array_flipped(width, height, depth);
  array.FlipHorizontal(&array_flipped);

  std::vector<T> array_host(width * height * depth, T(0.0));
  array.CopyToHost(array_host.data(), width * sizeof(T));

  std::vector<T> array_flipped_host(width * height * depth, 0);
  array_flipped.CopyToHost(array_flipped_host.data(), width * sizeof(T));

  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      for (size_t d = 0; d < depth; ++d) {
        EXPECT_EQ(
            array_host[d * width * height + r * width + c],
            array_flipped_host[d * width * height + r * width + width - 1 - c]);
      }
    }
  }
}

TEST(GpuMat, FlipHorizontal) {
  for (size_t w = 1; w <= 5; ++w) {
    for (size_t h = 1; h <= 5; ++h) {
      for (size_t d = 1; d <= 3; ++d) {
        const size_t width = 20 * w;
        const size_t height = 20 * h;
        TestFlipHorizontalImage<int8_t>(width, height, d);
        TestFlipHorizontalImage<int16_t>(width, height, d);
        TestFlipHorizontalImage<int32_t>(width, height, d);
        TestFlipHorizontalImage<int64_t>(width, height, d);
        TestFlipHorizontalImage<float>(width, height, d);
        TestFlipHorizontalImage<double>(width, height, d);
      }
    }
  }
}

template <typename T>
void TestRotateImage(const size_t width,
                     const size_t height,
                     const size_t depth) {
  GpuMat<T> array(width, height, depth);

  GpuMatPRNG prng_array(width, height);
  array.FillWithRandomNumbers(T(0.0), T(100.0), prng_array);

  GpuMat<T> array_rotated(height, width, depth);
  array.Rotate(&array_rotated);

  std::vector<T> array_host(width * height * depth, T(0.0));
  array.CopyToHost(array_host.data(), width * sizeof(T));

  std::vector<T> array_rotated_host(width * height * depth, 0);
  array_rotated.CopyToHost(array_rotated_host.data(), height * sizeof(T));

  const double arrayCenterH = width / 2.0 - 0.5;
  const double arrayCenterV = height / 2.0 - 0.5;
  const double angle = -M_PI / 2;
  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      for (size_t d = 0; d < depth; ++d) {
        const size_t rotc =
            std::round(std::cos(angle) * (c - arrayCenterH) -
                       std::sin(angle) * (r - arrayCenterV) + arrayCenterV);
        const size_t rotr =
            std::round(std::sin(angle) * (c - arrayCenterH) +
                       std::cos(angle) * (r - arrayCenterV) + arrayCenterH);
        EXPECT_EQ(
            array_host[d * width * height + r * width + c],
            array_rotated_host[d * width * height + rotr * height + rotc]);
      }
    }
  }
}

TEST(GpuMat, Rotate) {
  for (size_t w = 1; w <= 5; ++w) {
    for (size_t h = 1; h <= 5; ++h) {
      for (size_t d = 1; d <= 3; ++d) {
        const size_t width = 20 * w;
        const size_t height = 20 * h;
        TestRotateImage<int8_t>(width, height, d);
        TestRotateImage<int16_t>(width, height, d);
        TestRotateImage<int32_t>(width, height, d);
        TestRotateImage<int64_t>(width, height, d);
        TestRotateImage<float>(width, height, d);
        TestRotateImage<double>(width, height, d);
      }
    }
  }
}

TEST(GpuMat, RotateTileEdges) {
  for (const size_t width : {1, 7, 31, 32, 33, 63, 65}) {
    for (const size_t height : {1, 5, 31, 32, 33, 63, 65}) {
      for (const size_t depth : {1, 3}) {
        TestRotateImage<int8_t>(width, height, depth);
        TestRotateImage<int16_t>(width, height, depth);
        TestRotateImage<int32_t>(width, height, depth);
        TestRotateImage<int64_t>(width, height, depth);
        TestRotateImage<float>(width, height, depth);
        TestRotateImage<double>(width, height, depth);
      }
    }
  }
}

namespace {

float FloatFromBits(const std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint32_t FloatBits(const float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<float> StatePattern(const size_t width,
                                const size_t height,
                                const size_t sources,
                                const size_t generation) {
  const std::array<std::uint32_t, 8> bits = {0x00000000u,
                                             0x80000000u,
                                             0x7fc12345u,
                                             0x00000001u,
                                             0x807fffffu,
                                             0x7f800000u,
                                             0xff800000u,
                                             0x3f000000u};
  std::vector<float> values(width * height * sources);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = FloatFromBits(bits[(i + generation * 3) % bits.size()]);
  }
  return values;
}

std::vector<float> RotateStateOnHost(const std::vector<float>& source,
                                     const size_t width,
                                     const size_t height,
                                     const size_t sources) {
  std::vector<float> output(source.size());
  for (size_t s = 0; s < sources; ++s) {
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        output[s * width * height + (width - 1 - col) * height + row] =
            source[s * width * height + row * width + col];
      }
    }
  }
  return output;
}

void UploadState(PatchMatchStateMap* output, const std::vector<float>& values) {
  ASSERT_EQ(values.size(),
            output->GetWidth() * output->GetHeight() * output->GetDepth());
  ASSERT_EQ(cudaMemcpy2D(output->GetPtr(),
                         output->GetPitch(),
                         values.data(),
                         output->GetWidth() * sizeof(float),
                         output->GetWidth() * sizeof(float),
                         output->GetHeight() * output->GetDepth(),
                         cudaMemcpyHostToDevice),
            cudaSuccess);
}

void ExpectStateBits(const PatchMatchStateMap& actual,
                     const std::vector<float>& expected) {
  const Mat<float> copied = actual.CopyToMat();
  ASSERT_EQ(expected.size(),
            actual.GetWidth() * actual.GetHeight() * actual.GetDepth());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(FloatBits(copied.GetPtr()[i]), FloatBits(expected[i])) << i;
  }
}

}  // namespace

TEST(PatchMatchStateMap, RotatesRawBitsWithStableCapacity) {
  const std::vector<std::array<size_t, 3>> shapes = {
      {1, 1, 1}, {1, 7, 3}, {9, 1, 2}, {3, 5, 4}, {129, 2, 3}};
  for (const auto& shape : shapes) {
    const size_t width = shape[0], height = shape[1], sources = shape[2];
    PatchMatchStateMap first(width, height, sources);
    PatchMatchStateMap second(width, height, sources);
    PatchMatchStateMap* input = &first;
    PatchMatchStateMap* output = &second;
    const float* const first_ptr = first.GetPtr();
    const float* const second_ptr = second.GetPtr();
    const size_t capacity = first.GetCapacityBytes();
    ASSERT_NE(first_ptr, second_ptr);
    ASSERT_EQ(capacity, second.GetCapacityBytes());

    std::vector<float> expected = StatePattern(width, height, sources, 0);
    UploadState(input, expected);
    size_t current_width = width, current_height = height;
    for (int turn = 0; turn < 8; ++turn) {
      SCOPED_TRACE(::testing::Message() << width << "x" << height << "x"
                                        << sources << " turn=" << turn);
      output->Reorient(current_height, current_width);
      input->Rotate(output);
      expected =
          RotateStateOnHost(expected, current_width, current_height, sources);
      ExpectStateBits(*output, expected);
      std::swap(input, output);
      std::swap(current_width, current_height);
      EXPECT_EQ(first.GetPtr(), first_ptr);
      EXPECT_EQ(second.GetPtr(), second_ptr);
      EXPECT_EQ(first.GetCapacityBytes(), capacity);
      EXPECT_EQ(second.GetCapacityBytes(), capacity);
    }
  }
}

TEST(PatchMatchStateMap, ThreeOwnerDeadRoleCycle) {
  constexpr size_t kWidth = 5, kHeight = 3, kSources = 4;
  PatchMatchStateMap owner_a(kWidth, kHeight, kSources);
  PatchMatchStateMap owner_b(kWidth, kHeight, kSources);
  PatchMatchStateMap owner_c(kWidth, kHeight, kSources);
  PatchMatchStateMap* selected = &owner_a;
  PatchMatchStateMap* previous = &owner_b;
  PatchMatchStateMap* cost = &owner_c;
  const std::array<float*, 3> original_ptrs = {
      owner_a.GetPtr(), owner_b.GetPtr(), owner_c.GetPtr()};
  ASSERT_NE(original_ptrs[0], original_ptrs[1]);
  ASSERT_NE(original_ptrs[0], original_ptrs[2]);
  ASSERT_NE(original_ptrs[1], original_ptrs[2]);
  previous->FillWithScalar(0.5f);
  size_t width = kWidth, height = kHeight;
  for (size_t turn = 0; turn < 8; ++turn) {
    SCOPED_TRACE(turn);
    const std::vector<float> selected_values =
        StatePattern(width, height, kSources, turn);
    const std::vector<float> cost_values =
        StatePattern(width, height, kSources, turn + 1);
    UploadState(selected, selected_values);
    UploadState(cost, cost_values);
    previous->Reorient(height, width);
    selected->Rotate(previous);
    selected->Reorient(height, width);
    cost->Rotate(selected);
    std::swap(cost, selected);
    selected->Reorient(height, width);

    ExpectStateBits(
        *previous, RotateStateOnHost(selected_values, width, height, kSources));
    ExpectStateBits(*cost,
                    RotateStateOnHost(cost_values, width, height, kSources));
    EXPECT_EQ(owner_a.GetPtr(), original_ptrs[0]);
    EXPECT_EQ(owner_b.GetPtr(), original_ptrs[1]);
    EXPECT_EQ(owner_c.GetPtr(), original_ptrs[2]);
    std::swap(width, height);
  }
}

// Test persisted words, not a second implementation of the device quantizer.
#ifndef COLMAP_HIP_ENABLED
namespace {
__global__ void RoundTripNccCodes(PackedNccCostView view) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 65536) return;
  const size_t row = i / 256, col = i % 256;
  view.codes.Set(row, col, 0, static_cast<uint16_t>(i));
  view.Set(row, col, 0, view.Get(row, col, 0));
}
__global__ void WriteNccCost(PackedNccCostView view, float value) {
  view.Set(0, 0, 0, value);
}
__global__ void ReadNccCost(PackedNccCostView view, float* output) {
  output[0] = view.Get(0, 0, 0);
}
std::vector<uint16_t> NccCodes(const PackedNccCostMap& map) {
  const auto view = map.View().codes;
  std::vector<uint16_t> result(view.width * view.height * view.depth);
  CUDA_SAFE_CALL(cudaMemcpy2D(result.data(),
                              view.width * sizeof(uint16_t),
                              view.ptr,
                              view.pitch,
                              view.width * sizeof(uint16_t),
                              view.height * view.depth,
                              cudaMemcpyDeviceToHost));
  return result;
}
}  // namespace

TEST(PackedNccCostMap, ScalarErrorBoundAndExactEndpoints) {
  PackedNccCostMap map(1, 1, 1);
  GpuMat<float> output(1, 1);
  for (const float cost : {0.0f, 2.0f, 0.5f, 1.0f, 1.5f, 0.1234567f}) {
    WriteNccCost<<<1, 1>>>(map.View(), cost);
    ReadNccCost<<<1, 1>>>(map.View(), output.GetPtr());
    CUDA_SYNC_AND_CHECK();
    EXPECT_NO_THROW(map.CheckDomain());
    float actual = 0;
    output.CopyToHost(&actual, sizeof(float));
    if (cost == 0.0f || cost == 2.0f) {
      EXPECT_EQ(actual, cost);
    } else {
      EXPECT_LE(std::abs(actual - cost), 1.54e-5f);
    }
  }
}
TEST(PackedNccCostMap, AllCodesRoundTripAndEndpoints) {
  PackedNccCostMap map(256, 256, 1);
  RoundTripNccCodes<<<256, 256>>>(map.View());
  CUDA_SYNC_AND_CHECK();
  EXPECT_NO_THROW(map.CheckDomain());
  const auto codes = NccCodes(map);
  for (size_t i = 0; i < codes.size(); ++i) ASSERT_EQ(codes[i], i) << i;
}

TEST(PackedNccCostMap, RejectsEachInvalidDomain) {
  for (const float invalid : {-0.1f,
                              2.1f,
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
    PackedNccCostMap map(1, 1, 1);
    WriteNccCost<<<1, 1>>>(map.View(), invalid);
    CUDA_SYNC_AND_CHECK();
    EXPECT_ANY_THROW(map.CheckDomain());
  }
}

TEST(PatchMatchProbabilityUnorm16, RotatesWithCompactHmmScratch) {
  constexpr size_t width = 385, height = 513, sources = 3;
  PatchMatchProbabilityUnorm16 map(width, height, sources);
  map.FillWithScalar(0.5f);
  std::vector<uint16_t> codes(width * height * sources);
  for (size_t i = 0; i < codes.size(); ++i)
    codes[i] = static_cast<uint16_t>(i % 65536);
  const auto view = map.View();
  CUDA_SAFE_CALL(cudaMemcpy2D(view.ptr,
                              view.pitch,
                              codes.data(),
                              width * sizeof(uint16_t),
                              width * sizeof(uint16_t),
                              height * sources,
                              cudaMemcpyHostToDevice));
  std::vector<float> expected = map.CopyToMat().GetData();
  PatchMatchHmmCheckpoint compact(width, height, sources, sizeof(uint16_t));
  PatchMatchHmmCheckpoint full(width, height, sources);
  ASSERT_LT(compact.GetCapacityBytes(), full.GetCapacityBytes());
  size_t w = width, h = height;
  for (int turn = 0; turn < 4; ++turn) {
    std::vector<float> rotated(expected.size());
    for (size_t source = 0; source < sources; ++source)
      for (size_t row = 0; row < h; ++row)
        for (size_t col = 0; col < w; ++col)
          rotated[(source * w + (w - 1 - col)) * h + row] =
              expected[(source * h + row) * w + col];
    map.RotateWithPlaneScratch(compact.GetPtr(), compact.GetCapacityBytes());
    std::swap(w, h);
    expected.swap(rotated);
    EXPECT_EQ(map.GetWidth(), w);
    EXPECT_EQ(map.GetHeight(), h);
    ASSERT_EQ(map.CopyToMat().GetData(), expected);
  }
}

TEST(PackedNccCostMap, RotatesOddPlanesWithBoundedScratch) {
  for (const auto [width, height] : {std::pair<size_t, size_t>{7, 5},
                                     {257, 3},
                                     {3, 257},
                                     {385, 513},
                                     {513, 385}}) {
    for (const size_t kSources : {size_t{1}, size_t{2}, size_t{3}}) {
      PackedNccCostMap map(width, height, kSources);
      const size_t capacity = map.GetCapacityBytes();
      std::vector<uint16_t> expected(width * height * kSources);
      for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = static_cast<uint16_t>(i % 65536);
      const auto view = map.View().codes;
      CUDA_SAFE_CALL(cudaMemcpy2D(view.ptr,
                                  view.pitch,
                                  expected.data(),
                                  width * sizeof(uint16_t),
                                  width * sizeof(uint16_t),
                                  height * kSources,
                                  cudaMemcpyHostToDevice));
      PatchMatchHmmCheckpoint scratch(
          width, height, kSources, sizeof(uint16_t));
      const size_t scratch_bytes = scratch.GetCapacityBytes();
      if ((width == 385 && height == 513) || (width == 513 && height == 385)) {
        PatchMatchHmmCheckpoint full(width, height, kSources);
        EXPECT_LT(scratch_bytes, full.GetCapacityBytes());
      }
      const size_t next_plane =
          ((height * sizeof(uint16_t) + 511) / 512 * 512) * width;
      EXPECT_ANY_THROW(
          map.RotateWithPlaneScratch(scratch.GetPtr(), next_plane - 1));
      EXPECT_ANY_THROW(map.RotateWithPlaneScratch(view.ptr + 1, scratch_bytes));
      EXPECT_EQ(map.GetWidth(), width);
      EXPECT_EQ(map.GetHeight(), height);
      size_t w = width, h = height;
      for (size_t turn = 0; turn < 4; ++turn) {
        std::vector<uint16_t> rotated(expected.size());
        for (size_t source = 0; source < kSources; ++source)
          for (size_t row = 0; row < h; ++row)
            for (size_t col = 0; col < w; ++col)
              rotated[(source * w + (w - 1 - col)) * h + row] =
                  expected[(source * h + row) * w + col];
        map.RotateWithPlaneScratch(scratch.GetPtr(), scratch_bytes);
        std::swap(w, h);
        expected.swap(rotated);
        ASSERT_EQ(NccCodes(map), expected);
        EXPECT_EQ(map.GetCapacityBytes(), capacity);
        EXPECT_EQ(map.GetWidth(), w);
        EXPECT_EQ(map.GetHeight(), h);
      }
      EXPECT_NO_THROW(map.CheckDomain());
      EXPECT_ANY_THROW(map.Reorient(width + 1, height));
    }
  }
}

TEST(PackedNccCostMap, RejectsInvalidOrOverflowingDimensions) {
  EXPECT_ANY_THROW(PackedNccCostMap(0, 1, 1));
  EXPECT_ANY_THROW(PackedNccCostMap(1, 0, 1));
  EXPECT_ANY_THROW(PackedNccCostMap(1, 1, 0));
  EXPECT_ANY_THROW(PackedNccCostMap(std::numeric_limits<size_t>::max(), 1, 1));
  EXPECT_ANY_THROW(PackedNccCostMap(1, 1, std::numeric_limits<size_t>::max()));
}
#endif

// Test persisted words, not a second implementation of the device quantizer.
#ifndef COLMAP_HIP_ENABLED
namespace {
__host__ __device__ uint32_t Fixed24CodePattern(size_t i, size_t source) {
  const uint32_t index = static_cast<uint32_t>(i);
  const uint32_t code = ((index & 0xffffu) << 8) |
                        ((index ^ (index >> 8) ^ (index >> 16)) & 0xffu);
  return code ^ ((static_cast<uint32_t>(source) * 0x5a39c7u) & 0xffffffu);
}
__global__ void RoundTripFixed24Codes(Fixed24NccCostView view) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t plane = view.GetWidth() * view.GetHeight();
  if (i >= plane * view.GetDepth()) return;
  const size_t source = i / plane, pixel = i % plane;
  const size_t row = pixel / view.GetWidth(), col = pixel % view.GetWidth();
  const uint32_t code = Fixed24CodePattern(pixel, source);
  const size_t index = (source * view.height + row) * view.pixel_pitch + col;
  view.codes[index] = static_cast<uint16_t>(code >> 8);
  view.low_codes[index] = static_cast<uint8_t>(code);
  view.Set(row, col, source, view.Get(row, col, source));
}
__global__ void WriteFixed24Cost(Fixed24NccCostView view, float value) {
  view.Set(0, 0, 0, value);
}
__global__ void ReadFixed24Cost(Fixed24NccCostView view, float* output) {
  output[0] = view.Get(0, 0, 0);
}
std::vector<uint32_t> Fixed24Codes(const Fixed24NccCostMap& map) {
  const auto view = map.View();
  const size_t width = map.GetWidth(), rows = map.GetHeight() * map.GetDepth();
  std::vector<uint16_t> high(width * rows);
  std::vector<uint8_t> low(width * rows);
  CUDA_SAFE_CALL(cudaMemcpy2D(high.data(),
                              width * sizeof(uint16_t),
                              view.codes,
                              2 * view.pixel_pitch,
                              width * sizeof(uint16_t),
                              rows,
                              cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(low.data(),
                              width,
                              view.low_codes,
                              view.pixel_pitch,
                              width,
                              rows,
                              cudaMemcpyDeviceToHost));
  std::vector<uint32_t> result(high.size());
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = (static_cast<uint32_t>(high[i]) << 8) | low[i];
  }
  return result;
}
}  // namespace

TEST(Fixed24NccCostMap, ScalarErrorBoundAndExactEndpoints) {
  Fixed24NccCostMap map(1, 1, 1);
  GpuMat<float> output(1, 1);
  for (const float cost : {0.0f,
                           -0.0f,
                           2.0f,
                           0.5f,
                           1.0f,
                           1.5f,
                           0.1234567f,
                           0x1p-24f,
                           3.0f * 0x1p-24f,
                           std::numeric_limits<float>::denorm_min(),
                           std::nextafter(1.0f, 0.0f),
                           std::nextafter(1.0f, 2.0f),
                           std::nextafter(2.0f, 0.0f)}) {
    WriteFixed24Cost<<<1, 1>>>(map.View(), cost);
    ReadFixed24Cost<<<1, 1>>>(map.View(), output.GetPtr());
    CUDA_SYNC_AND_CHECK();
    EXPECT_NO_THROW(map.CheckDomain());
    float actual = 0;
    output.CopyToHost(&actual, sizeof(float));
    if (cost == std::nextafter(2.0f, 0.0f)) {
      EXPECT_EQ(actual, 2.0f);
    } else if (cost >= 1.0f || cost == 0.0f) {
      EXPECT_EQ(actual, cost);
    } else {
      EXPECT_LE(std::abs(actual - cost), 0x1p-24f);
    }
  }
}
TEST(Fixed24NccCostMap, CodeWordsRoundTripAcrossSources) {
  static_assert(Fixed24NccCostView::kStorageBits == 24);
  constexpr size_t kPlane = 256 * 256, kSources = 3;
  Fixed24NccCostMap map(256, 256, kSources);
  RoundTripFixed24Codes<<<kPlane * kSources / 256, 256>>>(map.View());
  CUDA_SYNC_AND_CHECK();
  EXPECT_NO_THROW(map.CheckDomain());
  const auto codes = Fixed24Codes(map);
  for (size_t i = 0; i < codes.size(); ++i) {
    ASSERT_EQ(codes[i], Fixed24CodePattern(i % kPlane, i / kPlane)) << i;
  }
}

TEST(Fixed24NccCostMap, RejectsEachInvalidDomain) {
  for (const float invalid : {-0.1f,
                              -std::numeric_limits<float>::denorm_min(),
                              2.1f,
                              std::nextafter(2.0f, 3.0f),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
    Fixed24NccCostMap map(1, 1, 1);
    WriteFixed24Cost<<<1, 1>>>(map.View(), FloatFromBits(0x3f923456u));
    CUDA_SYNC_AND_CHECK();
    const auto original = Fixed24Codes(map);
    WriteFixed24Cost<<<1, 1>>>(map.View(), invalid);
    CUDA_SYNC_AND_CHECK();
    EXPECT_ANY_THROW(map.CheckDomain());
    EXPECT_EQ(Fixed24Codes(map), original);
    WriteFixed24Cost<<<1, 1>>>(map.View(), 0.0f);
    CUDA_SYNC_AND_CHECK();
    EXPECT_ANY_THROW(map.CheckDomain());
  }
}

TEST(Fixed24NccCostMap, RotatesOddPlanesWithBoundedScratch) {
  for (const auto [width, height] : {std::pair<size_t, size_t>{1, 1},
                                     {7, 5},
                                     {257, 3},
                                     {513, 257},
                                     {385, 513},
                                     {513, 385}}) {
    for (const size_t kSources : {size_t{1}, size_t{2}, size_t{3}}) {
      Fixed24NccCostMap map(width, height, kSources);
      const size_t capacity = map.GetCapacityBytes();
      std::vector<uint32_t> expected(width * height * kSources);
      std::vector<uint16_t> high(expected.size());
      std::vector<uint8_t> low(expected.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] =
            Fixed24CodePattern(i % (width * height), i / (width * height));
        high[i] = static_cast<uint16_t>(expected[i] >> 8);
        low[i] = static_cast<uint8_t>(expected[i]);
      }
      const auto view = map.View();
      CUDA_SAFE_CALL(cudaMemcpy2D(view.codes,
                                  2 * view.pixel_pitch,
                                  high.data(),
                                  width * sizeof(uint16_t),
                                  width * sizeof(uint16_t),
                                  height * kSources,
                                  cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy2D(view.low_codes,
                                  view.pixel_pitch,
                                  low.data(),
                                  width,
                                  width,
                                  height * kSources,
                                  cudaMemcpyHostToDevice));
      EXPECT_EQ(view.pixel_pitch, (width + 127) / 128 * 128);
      PatchMatchStateMap original_float(width, height, kSources);
      EXPECT_EQ((capacity - sizeof(uint32_t)) * 4,
                original_float.GetCapacityBytes() * 3);
      PatchMatchHmmCheckpoint scratch(
          width, height, kSources, sizeof(uint16_t));
      const size_t scratch_bytes = scratch.GetCapacityBytes();
      if ((width == 385 && height == 513) || (width == 513 && height == 385)) {
        PatchMatchHmmCheckpoint full(width, height, kSources);
        EXPECT_LT(scratch_bytes, full.GetCapacityBytes());
      }
      const size_t next_plane = 2 * ((height + 127) / 128 * 128) * width;
      EXPECT_ANY_THROW(
          map.RotateWithPlaneScratch(scratch.GetPtr(), next_plane - 1));
      EXPECT_ANY_THROW(map.RotateWithPlaneScratch(nullptr, scratch_bytes));
      EXPECT_ANY_THROW(
          map.RotateWithPlaneScratch(view.codes + 1, scratch_bytes));
      EXPECT_ANY_THROW(
          map.RotateWithPlaneScratch(view.low_codes, scratch_bytes));
      EXPECT_ANY_THROW(
          map.RotateWithPlaneScratch(view.domain_error, scratch_bytes));
      EXPECT_ANY_THROW(map.RotateWithPlaneScratch(
          reinterpret_cast<char*>(scratch.GetPtr()) + 1, scratch_bytes - 1));
      EXPECT_EQ(map.GetWidth(), width);
      EXPECT_EQ(map.GetHeight(), height);
      EXPECT_EQ(Fixed24Codes(map), expected);
      size_t w = width, h = height;
      for (size_t turn = 0; turn < 4; ++turn) {
        std::vector<uint32_t> rotated(expected.size());
        for (size_t source = 0; source < kSources; ++source)
          for (size_t row = 0; row < h; ++row)
            for (size_t col = 0; col < w; ++col)
              rotated[(source * w + (w - 1 - col)) * h + row] =
                  expected[(source * h + row) * w + col];
        map.RotateWithPlaneScratch(scratch.GetPtr(), scratch_bytes);
        std::swap(w, h);
        expected.swap(rotated);
        ASSERT_EQ(Fixed24Codes(map), expected);
        EXPECT_EQ(map.GetCapacityBytes(), capacity);
        EXPECT_EQ(map.GetWidth(), w);
        EXPECT_EQ(map.GetHeight(), h);
        EXPECT_EQ(map.View().codes, view.codes);
        EXPECT_EQ(map.View().low_codes, view.low_codes);
      }
      EXPECT_NO_THROW(map.CheckDomain());
      EXPECT_ANY_THROW(map.Reorient(width + 1, height));
      EXPECT_EQ(map.GetWidth(), width);
      EXPECT_EQ(map.GetHeight(), height);
      EXPECT_EQ(Fixed24Codes(map), expected);
    }
  }
}

TEST(Fixed24NccCostMap, RejectsInvalidOrOverflowingDimensions) {
  EXPECT_ANY_THROW(Fixed24NccCostMap(0, 1, 1));
  EXPECT_ANY_THROW(Fixed24NccCostMap(1, 0, 1));
  EXPECT_ANY_THROW(Fixed24NccCostMap(1, 1, 0));
  EXPECT_ANY_THROW(Fixed24NccCostMap(std::numeric_limits<size_t>::max(), 1, 1));
  EXPECT_ANY_THROW(Fixed24NccCostMap(1, std::numeric_limits<size_t>::max(), 1));
  EXPECT_ANY_THROW(Fixed24NccCostMap(1, 1, std::numeric_limits<size_t>::max()));
  EXPECT_ANY_THROW(Fixed24NccCostMap(
      static_cast<size_t>(std::numeric_limits<int>::max()), 1, 1));
  EXPECT_ANY_THROW(Fixed24NccCostMap(
      65536, 65536, static_cast<size_t>(std::numeric_limits<int>::max())));
  EXPECT_ANY_THROW(Fixed24NccCostMap(
      1048576, 1048576, static_cast<size_t>(std::numeric_limits<int>::max())));
}
#endif

}  // namespace mvs
}  // namespace colmap

namespace colmap {
namespace mvs {
namespace ncc20_test {
void Check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void Need(bool p, const char* m) {
  if (!p) throw std::runtime_error(m);
}
size_t Pitch(size_t w) { return (w + 127) / 128 * 128; }
size_t Round8(size_t h) { return (h + 7) / 8 * 8; }
__host__ __device__ unsigned Pattern(size_t i) {
  return unsigned((i * 7919 + 104729) & 0xfffff);
}
__global__ void Fill(Fixed20NccCostView v, unsigned* bad) {
  // Same ownership as native initialization: one thread per column, serial
  // rows/sources within distinct aligned64-row stripes. Never pixel-parallel.
  constexpr size_t stripe = 64;
  static_assert(stripe % 8 == 0);
  const size_t c = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= v.width) return;
  const size_t stripe_end = size_t(blockIdx.y + 1) * stripe;
  const size_t end = stripe_end < v.height ? stripe_end : v.height;
  for (size_t r = size_t(blockIdx.y) * stripe; r < end; ++r)
    for (size_t s = 0; s < v.sources; ++s) {
      const size_t i = (s * v.height + r) * v.width + c;
      v.StoreCode(r, c, s, Pattern(i));
      v.StoreCode(r, c, s, Pattern(i));
      if (v.GetCode(r, c, s) != Pattern(i)) atomicAdd(bad, 1u);
    }
}
__global__ void CooperativeRows(Fixed20NccCostView v, unsigned* bad) {
  const size_t c = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (size_t r = 0; r < v.height; ++r) {
    // Source ownership changes with row, including within one packed word.
    // Read current/future fields before any writer. Prior-row writes completed
    // at the preceding end-row barrier; this models HMM stripe replay too.
    const size_t first = (threadIdx.y + r) % blockDim.y;
    if (c < v.width)
      for (size_t s = first; s < v.sources; s += blockDim.y)
        for (size_t rr = r; rr < v.height && rr < r + 9; ++rr) {
          const size_t i = (s * v.height + rr) * v.width + c;
          if (v.GetCode(rr, c, s) != Pattern(i)) atomicAdd(bad, 1u);
        }
    __syncthreads();
    if (c < v.width)
      for (size_t s = first; s < v.sources; s += blockDim.y) {
        const size_t i = (s * v.height + r) * v.width + c;
        const unsigned value = Pattern(i) ^ 0x12345u;
        v.StoreCode(r, c, s, value);
        v.StoreCode(r, c, s, value);
        if (v.GetCode(r, c, s) != value) atomicAdd(bad, 1u);
      }
    __syncthreads();
  }
}
__global__ void Verify(Fixed20NccCostView v,
                       const unsigned* expected,
                       unsigned* bad) {
  const size_t n = v.width * v.height * v.sources;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x) {
    const size_t c = i % v.width, r = (i / v.width) % v.height,
                 s = i / (v.width * v.height);
    if (v.GetCode(r, c, s) != expected[i]) atomicAdd(bad, 1u);
  }
}
__global__ void Invalid(Fixed20NccCostView v) { v.Set(0, 0, 0, -1.0f); }
__global__ void InvalidRaw(Fixed20NccCostView v) {
  v.StoreCode(0, 0, 0, ncc20::kMaxCode + 1);
}
std::vector<unsigned char> Snapshot(Fixed20NccCostMap& map) {
  std::vector<unsigned char> b(map.GetCapacityBytes() - sizeof(unsigned));
  Check(
      cudaMemcpy(b.data(), map.View().high, b.size(), cudaMemcpyDeviceToHost));
  return b;
}
template <class F>
void Refuses(F operation) {
  bool refused = false;
  try {
    operation();
  } catch (const std::exception&) {
    refused = true;
  }
  Need(refused, "expected refusal");
}
TEST(NccFixed20, ColumnOwnershipAndRotation) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();

  size_t cases = 0, rotations = 0, alias_checks = 0;
  const size_t shapes[][3] = {{1, 1, 1},
                              {1, 3072, 3},
                              {3072, 1, 3},
                              {2, 257, 2},
                              {257, 2, 2},
                              {129, 127, 3},
                              {127, 129, 3},
                              {385, 513, 4},
                              {513, 385, 9}};
  for (const auto& shape : shapes) {
    size_t w = shape[0], h = shape[1], s = shape[2];
    Fixed20NccCostMap map(w, h, s);
    const size_t cap = std::max(Pitch(w) * h, Pitch(h) * w) * s;
    const size_t lowcap = std::max(w * Round8(h), h * Round8(w)) * s;
    Need(map.GetCapacityBytes() == 2 * cap + lowcap / 2 + sizeof(unsigned),
         "capacity charge");
    Need(reinterpret_cast<uintptr_t>(map.View().low) % 4 == 0, "low alignment");
    Need(map.View().low_height == Round8(h), "low stride");
    std::vector<unsigned> low(lowcap / 8, 1);
    Check(cudaMemcpy(
        low.data(), map.View().low, lowcap / 2, cudaMemcpyDeviceToHost));
    for (unsigned x : low) Need(x == 0, "initial low word");
    Check(cudaMemset(map.View().high, 0x5a, 2 * cap));
    GpuMat<unsigned> errors_owner(1, 1), expected_owner(w * h * s, 1);
    unsigned* bad = errors_owner.GetPtr();
    unsigned* expected_device = expected_owner.GetPtr();
    Check(cudaMemset(bad, 0, 4));
    // Qualify the real retained scratch provider, including thin dimensions.
    colmap::mvs::PatchMatchHmmCheckpoint hmm(w, h, s, sizeof(uint16_t));
    if ((w == 385 && h == 513) || (w == 513 && h == 385)) {
      colmap::mvs::PatchMatchHmmCheckpoint full(w, h, s);
      EXPECT_LT(hmm.GetCapacityBytes(), full.GetCapacityBytes());
    }
    void* scratch = hmm.GetPtr();
    const size_t scratch_bytes = hmm.GetCapacityBytes();
    Need(scratch_bytes >= 2 * cap / s, "actual HMM scratch reserve");
    Fill<<<dim3((w + 127) / 128, (h + 63) / 64), 128>>>(map.View(), bad);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    CooperativeRows<<<dim3((w + 7) / 8), dim3(8, 4)>>>(map.View(), bad);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    std::vector<unsigned> expected(w * h * s);
    for (size_t i = 0; i < expected.size(); ++i)
      expected[i] = Pattern(i) ^ 0x12345u;
    Check(cudaMemcpy(
        low.data(), map.View().low, lowcap / 2, cudaMemcpyDeviceToHost));
    for (size_t src = 0; src < s; ++src)
      for (size_t c = 0; c < w; ++c)
        for (size_t r = h; r < Round8(h); ++r) {
          const size_t word = (src * (Round8(h) / 8) + r / 8) * w + c;
          Need(((low[word] >> (4 * (r % 8))) & 15u) == 0,
               "initial padded rows preserved");
        }
    const auto before = Snapshot(map);
    Refuses(
        [&] { map.RotateWithPlaneScratch(map.View().high, scratch_bytes); });
    ++alias_checks;
    Refuses([&] {
      map.RotateWithPlaneScratch(map.View().high + 1, scratch_bytes);
    });
    ++alias_checks;
    Refuses([&] { map.RotateWithPlaneScratch(map.View().low, scratch_bytes); });
    ++alias_checks;
    Refuses(
        [&] { map.RotateWithPlaneScratch(map.View().error, scratch_bytes); });
    ++alias_checks;
    Refuses([&] {
      map.RotateWithPlaneScratch(static_cast<unsigned char*>(scratch) + 1,
                                 scratch_bytes);
    });
    ++alias_checks;
    Refuses([&] { map.RotateWithPlaneScratch(scratch, 2 * Pitch(h) * w - 1); });
    ++alias_checks;
    Refuses([&] { map.RotateWithPlaneScratch(nullptr, scratch_bytes); });
    ++alias_checks;
    Need(before == Snapshot(map), "refused rotation changed payload");
    Need(map.GetWidth() == w && map.GetHeight() == h,
         "refused rotation changed shape");
    map.CheckDomain();
    for (unsigned turn = 0; turn < 4; ++turn) {
      std::vector<unsigned> next(expected.size());
      for (size_t src = 0; src < s; ++src)
        for (size_t r = 0; r < w; ++r)
          for (size_t c = 0; c < h; ++c)
            next[(src * w + r) * h + c] =
                expected[(src * h + c) * w + (w - 1 - r)];
      map.RotateWithPlaneScratch(scratch, scratch_bytes);
      std::swap(w, h);
      expected.swap(next);
      Need(map.GetWidth() == w && map.GetHeight() == h, "rotated shape");
      Need(map.View().low_height == Round8(h), "rotated low stride");
      Check(cudaMemcpy(expected_device,
                       expected.data(),
                       expected.size() * sizeof(unsigned),
                       cudaMemcpyHostToDevice));
      Verify<<<256, 128>>>(map.View(), expected_device, bad);
      Check(cudaGetLastError());
      Check(cudaDeviceSynchronize());
      map.CheckDomain();
      ++rotations;
      Check(cudaMemcpy(
          low.data(), map.View().low, lowcap / 2, cudaMemcpyDeviceToHost));
      for (size_t src = 0; src < s; ++src)
        for (size_t c = 0; c < w; ++c)
          for (size_t r = h; r < Round8(h); ++r) {
            const size_t word = (src * (Round8(h) / 8) + r / 8) * w + c;
            Need(((low[word] >> (4 * (r % 8))) & 15u) == 0,
                 "rotated padded rows");
          }
    }
    unsigned errors = 0;
    Check(cudaMemcpy(&errors, bad, 4, cudaMemcpyDeviceToHost));
    Need(errors == 0, "owner code parity");
    const auto valid = Snapshot(map);
    Invalid<<<1, 1>>>(map.View());
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    Refuses([&] { map.CheckDomain(); });
    Need(valid == Snapshot(map), "invalid scalar changed payload");
    Check(cudaMemset(map.View().error, 0, sizeof(unsigned)));
    InvalidRaw<<<1, 1>>>(map.View());
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    Refuses([&] { map.CheckDomain(); });
    Need(valid == Snapshot(map), "invalid raw code changed payload");
    ++cases;
  }
  Refuses([&] { Fixed20NccCostMap m(0, 1, 1); });
  Refuses([&] { Fixed20NccCostMap m(1, 0, 1); });
  Refuses([&] { Fixed20NccCostMap m(1, 1, 0); });
  Refuses([&] { Fixed20NccCostMap m(size_t(-1), 1, 1); });
  EXPECT_EQ(cases, 9);
  EXPECT_EQ(rotations, 36);
  EXPECT_EQ(alias_checks, 63);
}

__global__ void CheckCodebook(unsigned* errors) {
  for (unsigned code = blockIdx.x * blockDim.x + threadIdx.x;
       code <= ncc20::kMaxCode;
       code += blockDim.x * gridDim.x) {
    unsigned restored = 0;
    if (!ncc20::Encode(ncc20::Decode(code), restored) || restored != code)
      atomicAdd(errors, 1u);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned code = 123;
    if (ncc20::Decode(ncc20::kMaxCode) != 2.0f) atomicAdd(errors, 1u);
    if (!ncc20::Encode(2.0f - ncc20::kStep, code) ||
        code != ncc20::kMaxCode - 1)
      atomicAdd(errors, 1u);
    if (!ncc20::Encode(nextafterf(2.0f - ncc20::kStep, 2.0f), code) ||
        code != ncc20::kMaxCode)
      atomicAdd(errors, 1u);
    const unsigned invalid[] = {0x80000001u,
                                0xbf800000u,
                                0x40000001u,
                                0x7f800000u,
                                0xff800000u,
                                0x7fc00000u};
    for (unsigned bits : invalid) {
      code = 123;
      if (ncc20::Encode(__uint_as_float(bits), code) || code != 123)
        atomicAdd(errors, 1u);
    }
    const unsigned zeros[] = {0u, 0x80000000u, 1u};
    for (unsigned bits : zeros)
      if (!ncc20::Encode(__uint_as_float(bits), code) || code != 0)
        atomicAdd(errors, 1u);
  }
}
TEST(NccFixed20, CodebookAndEndpoints) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  GpuMat<unsigned> errors(1, 1);
  errors.FillWithScalar(0u);
  CheckCodebook<<<256, 128>>>(errors.GetPtr());
  CUDA_CHECK();
  CUDA_SYNC_AND_CHECK();
  unsigned count = 0;
  errors.CopyToHost(&count, sizeof(count));
  EXPECT_EQ(count, 0u);
}

}  // namespace ncc20_test
}  // namespace mvs
}  // namespace colmap
