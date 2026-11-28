// SPDX-License-Identifier: BSD-3-Clause

#if defined(__CUDACC__) || defined(__HIPCC__)
#define BOOST_PP_VARIADICS 0
#endif

#include "colmap/mvs/packed_consistency_mask.h"
#include "colmap/util/cuda.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

__global__ void PackConsistencyMask(PackedConsistencyMaskView mask) {
  const size_t col = threadIdx.x;
  const uint16_t flags[2][3] = {{0x181, 0x002, 0x080}, {0x200, 0x3ff, 0x201}};
  for (size_t row = 0; row < 2; ++row) {
    for (size_t source = 0; source < 10; ++source) {
      mask.Set(row, col, source, (flags[row][col] >> source) & 1);
    }
  }
  if (col != 1) {
    mask.Set(0, col, 7, 0);
  }
}

TEST(PackedConsistencyMask, PackRotateDecodeSourceOrder) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  const std::vector<int> source_image_idxs = {
      90, 4, 71, 5, 62, 6, 53, 7, 44, 8};
  GpuMat<uint8_t> mask(
      3, 2, PackedConsistencyMaskDepth(source_image_idxs.size()));
  mask.FillWithScalar(0);
  PackConsistencyMask<<<1, 3>>>(PackedConsistencyMaskView{mask.View()});
  CUDA_SYNC_AND_CHECK();

  GpuMat<uint8_t> rotated(2, 3, mask.GetDepth());
  mask.Rotate(&rotated);
  const std::vector<int> expected = {1,  0,  2, 90, 8,  0,  1, 1,  4, 1,  1,
                                     10, 90, 4, 71, 5,  62, 6, 53, 7, 44, 8,
                                     0,  2,  2, 90, 44, 1,  2, 1,  8};
  EXPECT_EQ(DecodePackedConsistencyMask(rotated.CopyToMat(), source_image_idxs),
            expected);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
