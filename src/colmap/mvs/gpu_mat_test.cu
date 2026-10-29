// SPDX-License-Identifier: BSD-3-Clause

#if defined(__CUDACC__) || defined(__HIPCC__)
#define BOOST_PP_VARIADICS 0
#endif

#include "colmap/math/math.h"
#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/gpu_mat_prng.h"
#include "colmap/mvs/packed_ncc_cost_map.h"
#include "colmap/mvs/patch_match_hmm_checkpoint.h"
#include "colmap/mvs/patch_match_state_map.h"

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

TEST(PackedNccCostMap, RotatesOddPlanesWithBoundedScratch) {
  for (const auto [width, height] :
       {std::pair<size_t, size_t>{7, 5}, {257, 3}, {3, 257}}) {
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
      PatchMatchHmmCheckpoint scratch(width, height, kSources);
      const size_t scratch_bytes = scratch.GetCapacityBytes();
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

}  // namespace mvs
}  // namespace colmap
