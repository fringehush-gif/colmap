// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef COLMAP_TEST_CUDA_TIMER_WRAPPING
namespace {
int timer_event_handles[2] = {};
struct TimerEvents {
  int creates = 0, destroys = 0, live = 0, fail_stage = 0;
  unsigned destroy_mask = 0;
  bool failed_output = false;
  cudaStream_t stream = nullptr;
} timer_events;
}  // namespace

extern "C" cudaError_t __wrap_cudaEventCreate(cudaEvent_t* event) {
  const int ordinal = ++timer_events.creates;
  const bool fail = ordinal == timer_events.fail_stage;
  *event = nullptr;
  if (!fail || timer_events.failed_output) {
    *event = reinterpret_cast<cudaEvent_t>(&timer_event_handles[ordinal - 1]);
    ++timer_events.live;
  }
  return fail ? cudaErrorMemoryAllocation : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t,
                                              cudaStream_t stream) {
  timer_events.stream = stream;
  return timer_events.fail_stage == 3 ? cudaErrorInvalidResourceHandle
                                      : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t event) {
  ++timer_events.destroys;
  --timer_events.live;
  const unsigned mask =
      event == reinterpret_cast<cudaEvent_t>(&timer_event_handles[0]) ? 1u : 2u;
  return (timer_events.destroy_mask & mask) != 0 ? cudaErrorUnknown
                                                 : cudaSuccess;
}
#endif

namespace colmap {
namespace {

#ifdef COLMAP_TEST_CUDA_TIMER_WRAPPING
TEST(CudaTimer, ConstructionAndUnwind) {
  // Event calls are synthetic; these ownership regressions need no GPU.
  for (const auto [stage, failed_output] : {std::pair{0, false},
                                            {1, false},
                                            {1, true},
                                            {2, false},
                                            {2, true},
                                            {3, false}}) {
    SCOPED_TRACE(::testing::Message() << stage << ":" << failed_output);
    timer_events = {};
    timer_events.fail_stage = stage;
    timer_events.failed_output = failed_output;
    const CudaCleanupScope cleanup;
    if (stage == 0)
      EXPECT_NO_THROW((CudaTimer{}));
    else
      EXPECT_THROW((CudaTimer{}), std::invalid_argument);
    EXPECT_EQ(timer_events.live, 0);
    EXPECT_EQ(timer_events.destroys,
              stage == 1 ? int(failed_output)
                         : (stage == 2 ? 1 + int(failed_output) : 2));
    if (stage == 0 || stage == 3)
      EXPECT_EQ(timer_events.stream, CudaDefaultStream());
    EXPECT_NO_THROW(cleanup.Check());
  }
}

TEST(CudaTimer, CleanupFailuresRemainObservable) {
  struct PrimaryError : std::exception {};
  for (const bool primary : {false, true}) {
    for (const unsigned mask : {1u, 2u, 3u}) {
      SCOPED_TRACE(::testing::Message() << primary << ":" << mask);
      timer_events = {};
      timer_events.destroy_mask = mask;
      const CudaCleanupScope cleanup;
      try {
        try {
          const CudaTimer timer;
          if (primary) throw PrimaryError{};
        } catch (...) {
          cleanup.Check();
          throw;
        }
        cleanup.Check();
        FAIL() << "Expected an observed event cleanup failure";
      } catch (const std::runtime_error& error) {
        EXPECT_THAT(error.what(), testing::HasSubstr("cudaEventDestroy"));
        if (primary) EXPECT_THROW(std::rethrow_if_nested(error), PrimaryError);
      }
      EXPECT_EQ(timer_events.destroys, 2);
      EXPECT_EQ(timer_events.live, 0);
    }
  }
}
#endif

TEST(CudaSafeCall, Success) {
  EXPECT_NO_THROW(CudaSafeCall(cudaSuccess, "probe.cc", 42));
}

TEST(CudaSafeCall, InsufficientDriver) {
  try {
    CudaSafeCall(cudaErrorInsufficientDriver, "probe.cc", 42);
    FAIL() << "Expected a runtime error";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_THAT(message, testing::HasSubstr("probe.cc:42"));
    EXPECT_THAT(
        message,
        testing::HasSubstr("error " + std::to_string(static_cast<int>(
                                          cudaErrorInsufficientDriver))));
    EXPECT_THAT(message, testing::HasSubstr("driver"));
    EXPECT_THAT(message, testing::HasSubstr("COLMAP build"));
    EXPECT_THAT(message, testing::Not(testing::HasSubstr("(null)")));
  }
}

TEST(CudaSafeCall, UnknownErrorPreservesCode) {
  try {
    CudaSafeCall(static_cast<cudaError_t>(123456), "probe.cc", 7);
    FAIL() << "Expected a runtime error";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_THAT(message, testing::HasSubstr("123456"));
    EXPECT_THAT(message, testing::HasSubstr("probe.cc:7"));
    EXPECT_THAT(message, testing::Not(testing::HasSubstr("(null)")));
    EXPECT_THAT(message,
                testing::Not(testing::HasSubstr("Update the NVIDIA driver")));
  }
}

class CudaCheckTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int num_devices = 0;
    const auto status = cudaGetDeviceCount(&num_devices);
    cudaGetLastError();
    if (status != cudaSuccess || num_devices == 0) {
      GTEST_SKIP() << "No CUDA/HIP device available";
    }
  }

  void TearDown() override { cudaGetLastError(); }

  void SetPendingError() {
    ASSERT_NE(cudaSetDevice(-1), cudaSuccess);
    ASSERT_NE(cudaPeekAtLastError(), cudaSuccess);
  }
};

TEST_F(CudaCheckTest, Success) {
  EXPECT_NO_THROW(CudaCheck("probe.cc", 42));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(CudaCheckTest, ErrorIsThrownAndConsumed) {
  ASSERT_NO_FATAL_FAILURE(SetPendingError());
  try {
    CudaCheck("probe.cc", 42);
    FAIL() << "Expected a CUDA runtime error";
  } catch (const std::invalid_argument& error) {
    EXPECT_THAT(error.what(), testing::HasSubstr("probe.cc:42"));
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(CudaCheckTest, UnwindingPreservesOriginalException) {
  ASSERT_NO_FATAL_FAILURE(SetPendingError());
  struct OriginalError {};
  struct CheckOnExit {
    bool* checked;
    ~CheckOnExit() {
      CudaCheck("probe.cc", 42);
      *checked = true;
    }
  };
  bool checked = false;
  try {
    const CheckOnExit check_on_exit{&checked};
    throw OriginalError{};
  } catch (const OriginalError&) {
    EXPECT_TRUE(checked);
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace
}  // namespace colmap
