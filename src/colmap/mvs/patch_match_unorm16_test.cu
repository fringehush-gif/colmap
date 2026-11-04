// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/packed_ncc_cost_map.h"
#include "colmap/mvs/patch_match_probability_unorm16.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

__global__ void StoreUnorm16(GpuMatView<float> values,
                             ProbabilityUnorm16View probability,
                             PackedNccCostView cost,
                             bool probability_mode) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= values.GetWidth()) return;
  if (probability_mode) {
    probability.Set(0, col, 0, values.Get(0, col, 0));
  } else {
    cost.Set(0, col, 0, values.Get(0, col, 0));
  }
}

__global__ void DecodeAndAdd(PackedNccCostView cost,
                             GpuMatView<float> accumulators,
                             GpuMatView<uint32_t> result) {
  result.Set(
      0, 0, 0, __float_as_uint(cost.Get(0, 0, 0) + accumulators.Get(0, 0, 0)));
}

// Both inputs have at most 24 significant bits, so this product is exact in
// double. Implement ties-even directly rather than depending on host fenv.
uint16_t ReferenceCode(float value, float scale) {
  const double exact = static_cast<double>(value) * scale;
  unsigned lower = static_cast<unsigned>(std::floor(exact));
  const double fraction = exact - lower;
  return static_cast<uint16_t>(
      lower + (fraction > 0.5 || (fraction == 0.5 && (lower & 1u))));
}

TEST(PatchMatchUnorm16, RoundTripsAndHalfIntegerNeighbors) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "CUDA device required";
  }
  for (const bool probability_mode : {true, false}) {
    const float scale = probability_mode ? 65535.0f : 32767.5f;
    std::vector<float> values;
    values.reserve(65536 + 3 * 65535);
    for (unsigned code = 0; code <= 65535; ++code) {
      values.push_back(static_cast<float>(code) *
                       (probability_mode ? 0x1.0001p-16f : 0x1.0001p-15f));
    }
    for (unsigned code = 0; code < 65535; ++code) {
      const float midpoint = static_cast<float>((code + 0.5) / scale);
      values.push_back(std::nextafter(midpoint, 0.0f));
      values.push_back(midpoint);
      values.push_back(
          std::nextafter(midpoint, std::numeric_limits<float>::infinity()));
    }
    GpuMat<float> input(values.size(), 1, 1);
    input.CopyToDevice(values.data(), values.size() * sizeof(float));
    PatchMatchProbabilityUnorm16 probability(values.size(), 1, 1);
    PackedNccCostMap cost(values.size(), 1, 1);
    StoreUnorm16<<<(values.size() + 255) / 256, 256>>>(
        input.View(), probability.View(), cost.View(), probability_mode);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::vector<uint16_t> codes(values.size());
    if (probability_mode) {
      probability.ValidateStoredRange();
      const auto view = probability.View();
      CUDA_SAFE_CALL(cudaMemcpy(codes.data(),
                                view.ptr,
                                codes.size() * sizeof(uint16_t),
                                cudaMemcpyDeviceToHost));
    } else {
      cost.CheckDomain();
      const auto view = cost.View();
      CUDA_SAFE_CALL(cudaMemcpy(codes.data(),
                                view.codes.ptr,
                                codes.size() * sizeof(uint16_t),
                                cudaMemcpyDeviceToHost));
    }
    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(codes[i], ReferenceCode(values[i], scale))
          << "probability=" << probability_mode << " index=" << i;
      if (i <= 65535) ASSERT_EQ(codes[i], i);
    }
  }
}

TEST(PatchMatchUnorm16, DecoderMaterializesBeforeConsumerArithmetic) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "CUDA device required";
  }
  PackedNccCostMap cost(1, 1, 1);
  const uint16_t code = 263;
  CUDA_SAFE_CALL(cudaMemcpy(
      cost.View().codes.ptr, &code, sizeof(code), cudaMemcpyHostToDevice));
  GpuMat<float> accumulators(1, 1, 1);
  const float accumulator = 0.1f;
  accumulators.CopyToDevice(&accumulator, sizeof(accumulator));
  GpuMat<uint32_t> result(1, 1, 1);
  DecodeAndAdd<<<1, 1>>>(cost.View(), accumulators.View(), result.View());
  CUDA_SAFE_CALL(cudaGetLastError());
  uint32_t bits = 0;
  result.CopyToHost(&bits, sizeof(bits));
  EXPECT_EQ(bits, 0x3ddd3cdeu);
}

TEST(PatchMatchUnorm16, CostDomainAndStickyRefusal) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "CUDA device required";
  }
  // Raw binary32 inputs preserve subnormals and signed NaNs under host FTZ.
  const uint32_t cases[] = {0x80000000u,
                            0x00000000u,
                            0x40000000u,
                            0x3fffffffu,
                            0x40000001u,
                            0x80000001u,
                            0xbf800000u,
                            0x7f800000u,
                            0xff800000u,
                            0x7fc00001u,
                            0xffc00001u};
  for (const uint32_t bits : cases) {
    const bool accepted = bits <= 0x40000000u || bits == 0x80000000u;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    GpuMat<float> input(1, 1, 1);
    input.CopyToDevice(&value, sizeof(value));
    PatchMatchProbabilityUnorm16 probability(1, 1, 1);
    PackedNccCostMap cost(1, 1, 1);
    const uint16_t sentinel = 12345;
    CUDA_SAFE_CALL(cudaMemcpy(cost.View().codes.ptr,
                              &sentinel,
                              sizeof(sentinel),
                              cudaMemcpyHostToDevice));
    StoreUnorm16<<<1, 1>>>(
        input.View(), probability.View(), cost.View(), false);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    uint16_t code = 0;
    CUDA_SAFE_CALL(cudaMemcpy(
        &code, cost.View().codes.ptr, sizeof(code), cudaMemcpyDeviceToHost));
    if (accepted) {
      EXPECT_NO_THROW(cost.CheckDomain());
      EXPECT_EQ(code, ReferenceCode(value, 32767.5f));
    } else {
      EXPECT_ANY_THROW(cost.CheckDomain());
      EXPECT_EQ(code, sentinel);
      // A later valid store must not erase a prior rejection.
      const float valid = 1.0f;
      input.CopyToDevice(&valid, sizeof(valid));
      StoreUnorm16<<<1, 1>>>(
          input.View(), probability.View(), cost.View(), false);
      CUDA_SAFE_CALL(cudaGetLastError());
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      EXPECT_ANY_THROW(cost.CheckDomain());
    }
  }
}

TEST(PatchMatchUnorm16, ProbabilityEndpointBudget) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "CUDA device required";
  }
  const float epsilon = ProbabilityUnorm16View::kEndpointRoundoff;
  const float infinity = std::numeric_limits<float>::infinity();
  const float values[] = {-0.0f,
                          0.0f,
                          1.0f,
                          -epsilon,
                          1.0f + epsilon,
                          std::nextafter(-epsilon, -infinity),
                          std::nextafter(1.0f + epsilon, infinity),
                          infinity,
                          -infinity,
                          std::numeric_limits<float>::quiet_NaN()};
  for (const float value : values) {
    const bool accepted = value >= -epsilon && value <= 1.0f + epsilon;
    GpuMat<float> input(1, 1, 1);
    input.CopyToDevice(&value, sizeof(value));
    PatchMatchProbabilityUnorm16 probability(1, 1, 1);
    PackedNccCostMap cost(1, 1, 1);
    StoreUnorm16<<<1, 1>>>(input.View(), probability.View(), cost.View(), true);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if (accepted) {
      EXPECT_NO_THROW(probability.ValidateStoredRange());
      const float bounded = std::max(0.0f, std::min(1.0f, value));
      uint16_t code = 0;
      CUDA_SAFE_CALL(cudaMemcpy(
          &code, probability.View().ptr, sizeof(code), cudaMemcpyDeviceToHost));
      EXPECT_EQ(code, ReferenceCode(bounded, 65535.0f));
    } else {
      EXPECT_ANY_THROW(probability.ValidateStoredRange());
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
