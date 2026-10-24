// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cuda.h"

#include "colmap/util/cudacc.h"

#include <thread>

#include <gtest/gtest.h>

namespace colmap {
namespace {

#if CUDART_VERSION >= 12000
class CudaBlockingSync : public ::testing::Test {
 protected:
  void SetUp() override {
    if (GetNumCudaDevices() == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    ASSERT_EQ(cudaGetDevice(&device_), cudaSuccess);
    original_flags_.resize(GetNumCudaDevices());
    for (int device = 0; device < original_flags_.size(); ++device) {
      ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
      ASSERT_EQ(cudaGetDeviceFlags(&original_flags_[device]), cudaSuccess);
    }
    ASSERT_EQ(cudaSetDevice(device_), cudaSuccess);
    flags_ = original_flags_[device_];
  }

  void TearDown() override {
    if (!IsSkipped()) {
      for (int device = 0; device < original_flags_.size(); ++device) {
        EXPECT_EQ(cudaSetDevice(device), cudaSuccess);
        EXPECT_EQ(
            cudaSetDeviceFlags(original_flags_[device] & ~cudaDeviceMapHost),
            cudaSuccess);
      }
      EXPECT_EQ(cudaSetDevice(device_), cudaSuccess);
    }
  }

  int device_ = 0;
  unsigned int flags_ = 0;
  std::vector<unsigned int> original_flags_;
};

TEST_F(CudaBlockingSync, PreservesOtherFlagsAndSharesPolicyWithWorkers) {
  ASSERT_EQ(cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceSyncMemops),
            cudaSuccess);
  unsigned int before_flags = 0;
  ASSERT_EQ(cudaGetDeviceFlags(&before_flags), cudaSuccess);
  SetCudaDevicesBlockingSync({device_, device_});
  SetCudaDevicesBlockingSync({device_});
  unsigned int worker_flags = 0;
  cudaError_t worker_error = cudaSuccess;
  std::thread worker([&]() {
    worker_error = cudaSetDevice(device_);
    if (worker_error == cudaSuccess) {
      worker_error = cudaGetDeviceFlags(&worker_flags);
    }
  });
  worker.join();
  ASSERT_EQ(worker_error, cudaSuccess);
  EXPECT_EQ(worker_flags & cudaDeviceScheduleMask,
            cudaDeviceScheduleBlockingSync);
  EXPECT_EQ(worker_flags & ~cudaDeviceScheduleMask,
            before_flags & ~cudaDeviceScheduleMask);
  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, device_);
}

TEST_F(CudaBlockingSync, RejectsInvalidSelectionBeforeChangingFlags) {
  EXPECT_ANY_THROW(SetCudaDevicesBlockingSync({}));
  EXPECT_ANY_THROW(SetCudaDevicesBlockingSync({-2}));
  EXPECT_ANY_THROW(SetCudaDevicesBlockingSync({device_, -1}));
  EXPECT_ANY_THROW(SetCudaDevicesBlockingSync({GetNumCudaDevices()}));
  unsigned int after_flags = 0;
  ASSERT_EQ(cudaGetDeviceFlags(&after_flags), cudaSuccess);
  EXPECT_EQ(after_flags, flags_);
}

TEST_F(CudaBlockingSync, AllVisibleDevicesAndCurrentDeviceRestoration) {
  const int count = GetNumCudaDevices();
  std::vector<unsigned int> original_flags(count);
  for (int device = 0; device < count; ++device) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    ASSERT_EQ(cudaGetDeviceFlags(&original_flags[device]), cudaSuccess);
  }
  ASSERT_EQ(cudaSetDevice(device_), cudaSuccess);
  SetCudaDevicesBlockingSync({-1});
  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, device_);
  for (int device = 0; device < count; ++device) {
    EXPECT_EQ(cudaSetDevice(device), cudaSuccess);
    unsigned int actual = 0;
    EXPECT_EQ(cudaGetDeviceFlags(&actual), cudaSuccess);
    EXPECT_EQ(actual & cudaDeviceScheduleMask, cudaDeviceScheduleBlockingSync);
    EXPECT_EQ(actual & ~cudaDeviceScheduleMask,
              original_flags[device] & ~cudaDeviceScheduleMask);
    EXPECT_EQ(cudaSetDeviceFlags(original_flags[device] & ~cudaDeviceMapHost),
              cudaSuccess);
  }
}
#endif  // CUDART_VERSION >= 12000

}  // namespace
}  // namespace colmap
