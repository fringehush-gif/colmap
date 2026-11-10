// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/gpu_mat.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(GpuMatCopy, PitchedBitsRoundTrip) {
  int devices = 0;
  const auto status = cudaGetDeviceCount(&devices);
  if (status != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  constexpr size_t width = 17, height = 9, depth = 2;
  std::vector<uint32_t> input(width * height * depth);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<uint32_t>(i) * 2654435761U;
  }
  const uint32_t special[] = {
      0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x00000001};
  std::copy(std::begin(special), std::end(special), input.begin());
  GpuMat<uint32_t> device(width, height, depth);
  device.CopyToDevice(input.data(), width * sizeof(uint32_t));
  std::vector<uint32_t> output(input.size());
  device.CopyToHost(output.data(), width * sizeof(uint32_t));
  EXPECT_EQ(output, input);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
