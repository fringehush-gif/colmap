// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/packed_ncc_cost_map.h"
#include "colmap/util/testing_cuda.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

using PatchMatchCostUnorm16 = CudaTest;

__global__ void StoreValues(GpuMatView<float> values,
                            PackedNccCostView output) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= values.GetWidth()) return;
  output.Set(0, col, 0, values.Get(0, col, 0));
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

TEST_F(PatchMatchCostUnorm16, RoundTripsAndHalfIntegerNeighbors) {
  const float scale = 32767.5f;
  std::vector<float> values;
  values.reserve(65536 + 3 * 65535);
  for (unsigned code = 0; code <= 65535; ++code) {
    values.push_back(static_cast<float>(code) * 0x1.0001p-15f);
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
  PackedNccCostMap output(values.size(), 1, 1);
  StoreValues<<<(values.size() + 255) / 256, 256>>>(input.View(),
                                                    output.View());
  CUDA_SYNC_AND_CHECK();
  output.CheckDomain();
  std::vector<uint16_t> codes(values.size());
  CUDA_SAFE_CALL(cudaMemcpy(codes.data(),
                            output.View().codes.ptr,
                            codes.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(codes[i], ReferenceCode(values[i], scale)) << "index=" << i;
    if (i <= 65535) ASSERT_EQ(codes[i], i);
  }
}

TEST_F(PatchMatchCostUnorm16, DecoderMaterializesBeforeConsumerArithmetic) {
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

TEST_F(PatchMatchCostUnorm16, CostDomainAndStickyRefusal) {
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
    PackedNccCostMap cost(1, 1, 1);
    const uint16_t sentinel = 12345;
    CUDA_SAFE_CALL(cudaMemcpy(cost.View().codes.ptr,
                              &sentinel,
                              sizeof(sentinel),
                              cudaMemcpyHostToDevice));
    StoreValues<<<1, 1>>>(input.View(), cost.View());
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
      StoreValues<<<1, 1>>>(input.View(), cost.View());
      CUDA_SAFE_CALL(cudaGetLastError());
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      EXPECT_ANY_THROW(cost.CheckDomain());
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
