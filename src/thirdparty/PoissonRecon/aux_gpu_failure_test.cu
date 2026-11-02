// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <array>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {
struct FaultState {
  int allocations = 0;
  int events = 0;
  int copies = 0;
  int fail_allocation = 0;
  int fail_event = 0;
  bool failed_output = false;
  bool null_success = false;
  bool destroy_error = false;
  bool free_error = false;
  bool mismatch = false;
  bool fake_pending = false;
  bool tracking_error = false;
  cudaError_t pending = cudaSuccess;
  cudaError_t returned = cudaErrorMemoryAllocation;
  std::vector<void*> memory;
  std::vector<cudaEvent_t> handles;
};
FaultState state;

void Pending(cudaError_t error) {
  state.fake_pending = true;
  state.pending = error;
}

cudaError_t FaultPeek() {
  return state.fake_pending ? state.pending : cudaPeekAtLastError();
}
cudaError_t FaultTake() {
  if (!state.fake_pending) return cudaGetLastError();
  const auto result = state.pending;
  state.fake_pending = false;
  state.pending = cudaSuccess;
  return result;
}
cudaError_t FaultMalloc(void** output, size_t bytes) {
  ++state.allocations;
  if (state.allocations == state.fail_allocation) {
    *output = nullptr;
    if (state.failed_output) {
      const auto status = cudaMalloc(output, bytes);
      if (status != cudaSuccess) return status;
      state.memory.push_back(*output);
    }
    if (state.null_success) return cudaSuccess;
    Pending(state.mismatch ? cudaErrorUnknown : state.returned);
    return state.returned;
  }
  const auto status = cudaMalloc(output, bytes);
  if (status == cudaSuccess) state.memory.push_back(*output);
  return status;
}
template <class T>
cudaError_t FaultMalloc(T** output, size_t bytes) {
  return FaultMalloc(reinterpret_cast<void**>(output), bytes);
}
cudaError_t FaultFree(void* pointer) {
  const auto status = cudaFree(pointer);
  if (status != cudaSuccess) return status;
  const auto it = std::find(state.memory.begin(), state.memory.end(), pointer);
  if (it == state.memory.end()) {
    state.tracking_error = true;
  } else {
    state.memory.erase(it);
  }
  if (state.free_error) {
    Pending(cudaErrorUnknown);
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}
cudaError_t FaultCreate(cudaEvent_t* output) {
  ++state.events;
  if (state.events == state.fail_event) {
    *output = nullptr;
    if (state.failed_output) {
      const auto status = cudaEventCreate(output);
      if (status != cudaSuccess) return status;
      state.handles.push_back(*output);
    }
    if (state.null_success) return cudaSuccess;
    Pending(state.mismatch ? cudaErrorUnknown : state.returned);
    return state.returned;
  }
  const auto status = cudaEventCreate(output);
  if (status == cudaSuccess) state.handles.push_back(*output);
  return status;
}
cudaError_t FaultDestroy(cudaEvent_t event) {
  const auto status = cudaEventDestroy(event);
  if (status != cudaSuccess) return status;
  const auto it = std::find(state.handles.begin(), state.handles.end(), event);
  if (it == state.handles.end()) {
    state.tracking_error = true;
  } else {
    state.handles.erase(it);
  }
  if (state.destroy_error) {
    Pending(cudaErrorUnknown);
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}
cudaError_t FaultCopy(void* to,
                      const void* from,
                      size_t bytes,
                      cudaMemcpyKind kind) {
  ++state.copies;
  return cudaMemcpy(to, from, bytes, kind);
}
}  // namespace

// Compile the real implementation in isolated namespaces so the injected
// inline allocation helper cannot share definitions with production objects.
#define colmap AuxFailureColmap
#define AuxNativeGpu AuxNativeGpuFaultTest
#define cudaMalloc FaultMalloc
#define cudaFree FaultFree
#define cudaEventCreate FaultCreate
#define cudaEventDestroy FaultDestroy
#define cudaPeekAtLastError FaultPeek
#define cudaGetLastError FaultTake
#define cudaMemcpy FaultCopy
#include "aux_gpu.cu"
#undef cudaMemcpy
#undef cudaGetLastError
#undef cudaPeekAtLastError
#undef cudaEventDestroy
#undef cudaEventCreate
#undef cudaFree
#undef cudaMalloc
#undef AuxNativeGpu
#undef colmap

namespace {
using AuxNativeGpuFaultTest::Available;
using AuxNativeGpuFaultTest::Four;
using AuxNativeGpuFaultTest::Reduce;
using AuxNativeGpuFaultTest::Stats;
using AuxNativeGpuFaultTest::View;

class AuxGpuFailureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count == 0)) {
      cudaGetLastError();
      GTEST_SKIP() << "CUDA device unavailable";
    }
    ASSERT_EQ(status, cudaSuccess);
    Reset();
    Stats admission;
    if (!Available(admission)) {
      GTEST_SKIP() << "Auxiliary GPU kernel unavailable: " << admission.reason;
    }
  }
  void TearDown() override {
    EXPECT_TRUE(state.memory.empty());
    EXPECT_TRUE(state.handles.empty());
    EXPECT_FALSE(state.tracking_error);
    cudaGetLastError();
  }
  void Reset() {
    ASSERT_TRUE(state.memory.empty());
    ASSERT_TRUE(state.handles.empty());
    EXPECT_FALSE(state.tracking_error);
    state = FaultState{};
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }
  bool Run(Four& output, Stats& stats) {
    const uint32_t levels[] = {0, 1}, children[] = {0, 0};
    const uint32_t offsets[] = {0, 1}, ids[] = {0};
    const Four seeds[] = {{{1, 2, 3, 1}}};
    const View view{1, 1, 0, levels, children, nullptr, offsets, ids, seeds};
    return Reduce(view, &output, stats);
  }
  void CleanCall() {
    Reset();
    Four output{};
    Stats stats;
    ASSERT_TRUE(Run(output, stats));
    EXPECT_EQ(output.v[0], 1);
    EXPECT_EQ(output.v[1], 2);
    EXPECT_EQ(output.v[2], 3);
    EXPECT_EQ(output.v[3], 1);
    EXPECT_TRUE(state.memory.empty());
    EXPECT_TRUE(state.handles.empty());
  }
  void ExpectFailure(bool fallback, bool expect_nested = false) {
    Four output{{-1, -1, -1, -1}};
    Stats stats;
    bool refused = false, threw = false, nested = false;
    try {
      refused = !Run(output, stats);
    } catch (const std::exception& error) {
      threw = true;
      if (expect_nested) {
        EXPECT_NE(std::string(error.what()).find("aux GPU cleanup failure"),
                  std::string::npos);
        try {
          std::rethrow_if_nested(error);
        } catch (const std::exception& cause) {
          nested =
              std::string(cause.what()).find("terminal allocation state") !=
              std::string::npos;
        }
      }
    }
    EXPECT_EQ(refused, fallback);
    EXPECT_EQ(threw, !fallback);
    if (expect_nested) EXPECT_TRUE(nested);
    EXPECT_TRUE(state.memory.empty());
    EXPECT_TRUE(state.handles.empty());
    EXPECT_FALSE(state.tracking_error);
    if (state.fail_event) EXPECT_EQ(state.events, state.fail_event);
    if (state.fail_allocation)
      EXPECT_EQ(state.allocations, state.fail_allocation);
    if (fallback) {
      EXPECT_STREQ(stats.reason, "device-allocation");
      EXPECT_EQ(state.copies, 0);
      for (const float value : output.v) EXPECT_EQ(value, -1);
    }
  }
};

TEST(AuxGpuAdmission, PriorErrorIsTerminal) {
  for (const auto error :
       {cudaErrorMemoryAllocation, cudaErrorUnknown, cudaErrorInvalidValue}) {
    SCOPED_TRACE(error);
    state = FaultState{};
    Pending(error);
    Stats admission;
    EXPECT_THROW(Available(admission), std::runtime_error);
    EXPECT_EQ(state.allocations, 0);
    EXPECT_EQ(state.events, 0);
  }
  state = FaultState{};
}

TEST_F(AuxGpuFailureTest, EventAllocationRefusal) {
  for (const int position : {1, 2}) {
    SCOPED_TRACE(position);
    Reset();
    state.fail_event = position;
    ExpectFailure(true);
    CleanCall();
  }
}

TEST_F(AuxGpuFailureTest, InvalidAcquisitionAndCleanupFailures) {
  CleanCall();
  for (const int position : {1, 2}) {
    SCOPED_TRACE(position);
    Reset();
    state.fail_event = position;
    state.failed_output = true;
    ExpectFailure(false);
    CleanCall();
    Reset();
    state.fail_event = position;
    state.null_success = true;
    ExpectFailure(false);
    CleanCall();
    Reset();
    state.fail_event = position;
    state.mismatch = true;
    ExpectFailure(false);
    CleanCall();
    for (const auto code :
         {cudaErrorInvalidValue, cudaErrorContextIsDestroyed}) {
      Reset();
      state.fail_event = position;
      state.returned = code;
      ExpectFailure(false);
      CleanCall();
    }
  }
  for (const int position : {1, 3}) {
    SCOPED_TRACE(position);
    Reset();
    state.fail_allocation = position;
    ExpectFailure(true);
    CleanCall();
    Reset();
    state.fail_allocation = position;
    state.failed_output = true;
    ExpectFailure(false);
    CleanCall();
    Reset();
    state.fail_allocation = position;
    state.null_success = true;
    ExpectFailure(false);
    CleanCall();
    Reset();
    state.fail_allocation = position;
    state.mismatch = true;
    ExpectFailure(false);
    CleanCall();
  }
  for (const auto code : {cudaErrorMemoryAllocation,
                          cudaErrorUnknown,
                          cudaErrorInsufficientDriver}) {
    Reset();
    Pending(code);
    ExpectFailure(false);
    EXPECT_EQ(state.allocations, 0);
    EXPECT_EQ(state.events, 0);
    CleanCall();
  }
  Reset();
  state.fail_event = 2;
  state.destroy_error = true;
  ExpectFailure(false);
  CleanCall();
  Reset();
  state.fail_event = 1;
  state.free_error = true;
  ExpectFailure(false);
  CleanCall();
  Reset();
  state.destroy_error = true;
  ExpectFailure(false);
  CleanCall();
  Reset();
  state.free_error = true;
  ExpectFailure(false);
  CleanCall();
  Reset();
  state.fail_event = 2;
  state.returned = cudaErrorInvalidValue;
  state.destroy_error = true;
  ExpectFailure(false, true);
  CleanCall();
}
}  // namespace
