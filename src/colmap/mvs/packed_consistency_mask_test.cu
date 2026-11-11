// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/packed_consistency_mask.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

class PackedConsistencyMaskTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
};

// Exactly one thread owns each pixel, as required by the production view.
__global__ void SetAll(PackedConsistencyMaskView mask, size_t sources) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= mask.bytes.width || row >= mask.bytes.height) return;
  for (size_t source = 0; source < sources; ++source) {
    mask.Set(row, col, source, 1);
  }
}

__global__ void ClearOne(PackedConsistencyMaskView mask, size_t sources) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= mask.bytes.width || row >= mask.bytes.height) return;
  mask.Set(row, col, (row * 7 + col * 11) % sources, 0);
}

// Reference is a dense boolean matrix per pixel, not a byte-packed formula.
std::vector<uint8_t> RotateReference(const std::vector<uint8_t>& dense,
                                     size_t width,
                                     size_t height,
                                     size_t sources) {
  std::vector<uint8_t> out(dense.size());
  for (size_t r = 0; r < width; ++r) {
    for (size_t c = 0; c < height; ++c) {
      for (size_t s = 0; s < sources; ++s) {
        out[(r * height + c) * sources + s] =
            dense[(c * width + (width - 1 - r)) * sources + s];
      }
    }
  }
  return out;
}

void CheckDecoded(const GpuMat<uint8_t>& mask,
                  const std::vector<uint8_t>& dense,
                  size_t sources) {
  const size_t width = mask.GetWidth(), height = mask.GetHeight();
  const size_t planes = PackedConsistencyMaskDepth(sources);
  ASSERT_EQ(mask.GetDepth(), planes);
  ASSERT_GE(mask.GetPitch(), width * sizeof(uint8_t));
  std::vector<uint8_t> bytes(width * height * planes);
  mask.CopyToHost(bytes.data(), width * sizeof(uint8_t));
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      std::vector<int> actual, expected;
      for (size_t source = 0; source < sources; ++source) {
        // Deliberately descending IDs verify source-order decode, not sorting.
        const int id = 1000 + static_cast<int>((sources - source) * 17);
        if (bytes[(source / 8) * width * height + row * width + col] &
            (1u << (source % 8)))
          actual.push_back(id);
        if (dense[(row * width + col) * sources + source])
          expected.push_back(id);
      }
      EXPECT_EQ(actual, expected) << "pixel " << row << "," << col;
      if (sources % 8 != 0) {
        const auto tail =
            bytes[(planes - 1) * width * height + row * width + col];
        EXPECT_EQ(tail >> (sources % 8), 0)
            << "tail bits at " << row << "," << col;
      }
    }
  }
}

TEST_F(PackedConsistencyMaskTest, DeviceWriteClearDecodeAndFourRotations) {
  constexpr size_t width = 35, height = 19;
  for (const size_t sources : {size_t{1},
                               size_t{7},
                               size_t{8},
                               size_t{9},
                               size_t{17},
                               size_t{31},
                               size_t{32},
                               size_t{33}}) {
    SCOPED_TRACE(::testing::Message() << "sources=" << sources);
    const size_t planes = PackedConsistencyMaskDepth(sources);
    ASSERT_EQ(planes, (sources + 7) / 8);
    GpuMat<uint8_t> a(width, height, planes), b(height, width, planes);
    a.FillWithScalar(0);
    const dim3 block(16, 8);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    SetAll<<<grid, block>>>(PackedConsistencyMaskView{a.View()}, sources);
    CUDA_SYNC_AND_CHECK();
    std::vector<uint8_t> dense(width * height * sources, 1);
    CheckDecoded(a, dense, sources);
    ClearOne<<<grid, block>>>(PackedConsistencyMaskView{a.View()}, sources);
    CUDA_SYNC_AND_CHECK();
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        dense[(row * width + col) * sources + (row * 7 + col * 11) % sources] =
            0;
      }
    }
    // The complete decoded vectors prove clearing did not alter neighbors.
    CheckDecoded(a, dense, sources);
    auto* current = &a;
    auto* next = &b;
    size_t current_width = width, current_height = height;
    for (int rotation = 0; rotation < 4; ++rotation) {
      current->Rotate(next);
      dense = RotateReference(dense, current_width, current_height, sources);
      std::swap(current_width, current_height);
      std::swap(current, next);
      CheckDecoded(*current, dense, sources);
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
