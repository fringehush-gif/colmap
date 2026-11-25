// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/patch_match_probability_unorm16.h"
#include "colmap/util/testing_cuda.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

using PatchMatchProbabilityUnorm16 = CudaTest;

__global__ void StoreValues(GpuMatView<float> values,
                            ProbabilityUnorm16View output) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= values.GetWidth()) return;
  output.Set(0, col, 0, values.Get(0, col, 0));
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

TEST_F(PatchMatchProbabilityUnorm16, RoundTripsAndHalfIntegerNeighbors) {
  const float scale = 65535.0f;
  std::vector<float> values;
  values.reserve(65536 + 3 * 65535);
  for (unsigned code = 0; code <= 65535; ++code) {
    values.push_back(static_cast<float>(code) * 0x1.0001p-16f);
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
  mvs::PatchMatchProbabilityUnorm16 output(values.size(), 1, 1);
  StoreValues<<<(values.size() + 255) / 256, 256>>>(input.View(),
                                                    output.View());
  CUDA_SYNC_AND_CHECK();
  output.ValidateStoredRange();
  std::vector<uint16_t> codes(values.size());
  CUDA_SAFE_CALL(cudaMemcpy(codes.data(),
                            output.View().ptr,
                            codes.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(codes[i], ReferenceCode(values[i], scale)) << "index=" << i;
    if (i <= 65535) ASSERT_EQ(codes[i], i);
  }
}

TEST_F(PatchMatchProbabilityUnorm16, ProbabilityEndpointBudget) {
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
    mvs::PatchMatchProbabilityUnorm16 probability(1, 1, 1);
    StoreValues<<<1, 1>>>(input.View(), probability.View());
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

TEST_F(PatchMatchProbabilityUnorm16, RotationChecksScratchAlignment) {
  mvs::PatchMatchProbabilityUnorm16 probability(5, 3, 2);
  probability.FillWithScalar(0.5f);
  const auto expected = probability.CopyToMat().GetData();
  GpuMat<uint8_t> scratch(5 * 512 + 2, 1, 1);
  try {
    probability.RotateWithPlaneScratch(
        reinterpret_cast<float*>(scratch.GetPtr() + 1), scratch.GetPitch() - 1);
    FAIL() << "Unaligned scratch reached rotation";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(
        std::string(error.what()).find("scratch_begin % alignof(uint16_t)"),
        std::string::npos);
  }
  EXPECT_EQ(probability.GetWidth(), 5);
  EXPECT_EQ(probability.GetHeight(), 3);
  EXPECT_EQ(probability.CopyToMat().GetData(), expected);

  // The scratch carrier is float*, but the rotation stores uint16_t values.
  probability.RotateWithPlaneScratch(
      reinterpret_cast<float*>(scratch.GetPtr() + 2), scratch.GetPitch() - 2);
  EXPECT_EQ(probability.GetWidth(), 3);
  EXPECT_EQ(probability.GetHeight(), 5);
  EXPECT_EQ(probability.CopyToMat().GetData(), expected);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
