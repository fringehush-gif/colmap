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
    device_captured_ = true;
    const int num_devices = GetNumCudaDevices();
    original_flags_.resize(num_devices);
    for (int device = 0; device < num_devices; ++device) {
      ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
      ASSERT_EQ(cudaGetDeviceFlags(&original_flags_[device]), cudaSuccess);
      ++captured_flags_;
    }
    ASSERT_EQ(cudaSetDevice(device_), cudaSuccess);
    flags_ = original_flags_[device_];
  }

  void TearDown() override {
    if (!IsSkipped()) {
      for (int device = 0; device < captured_flags_; ++device) {
        const cudaError_t status = cudaSetDevice(device);
        EXPECT_EQ(status, cudaSuccess);
        if (status == cudaSuccess) {
          EXPECT_EQ(
              cudaSetDeviceFlags(original_flags_[device] & ~cudaDeviceMapHost),
              cudaSuccess);
        }
      }
      if (device_captured_) {
        EXPECT_EQ(cudaSetDevice(device_), cudaSuccess);
      }
    }
  }

  int device_ = 0;
  int captured_flags_ = 0;
  bool device_captured_ = false;
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

TEST(CudaDeviceSupportsBf16Test, RejectsInvalidGpuIndex) {
  // No GPU needed: invalid input fails closed without querying a device.
  EXPECT_FALSE(CudaDeviceSupportsBf16(""));
  EXPECT_FALSE(CudaDeviceSupportsBf16("abc"));
  EXPECT_FALSE(CudaDeviceSupportsBf16("0,1"));
  EXPECT_FALSE(CudaDeviceSupportsBf16("0;1"));
}

TEST(CudaDeviceSupportsBf16Test, MatchesDeviceProperties) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "CUDA device required";
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  EXPECT_EQ(CudaDeviceSupportsBf16("0"), properties.major >= 8);
  // A negative index selects device 0, matching the ONNX provider behavior.
  EXPECT_EQ(CudaDeviceSupportsBf16("-1"), properties.major >= 8);
}

}  // namespace
}  // namespace colmap
