// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace PoissonGpuCleanupTest {
extern thread_local bool fail_next_host_allocation;
}
using PoissonGpuCleanupTest::fail_next_host_allocation;

namespace {
struct FaultState {
  int allocations = 0, events = 0, streams = 0, frees = 0, destroys = 0,
      stream_destroys = 0, syncs = 0, copies = 0;
  int fail_allocation = 0, fail_event = 0, fail_stream = 0, fail_free = 0,
      fail_destroy = 0, fail_stream_destroy = 0, fail_sync = 0, fail_copy = 0;
  bool fail_boundary_host_allocation = false;
  bool fail_after_readback = false;
  void* readback_destination = nullptr;
  std::size_t readback_bytes = 0;
  std::uint64_t allocated_bytes = 0;
  std::array<const void*, 4> sort_ranges{};
  std::array<const void*, 3> select_ranges{};
  std::size_t sort_bytes = 0, select_bytes = 0;
  cudaStream_t sort_stream = nullptr, select_stream = nullptr;
  bool failed_output = false, null_success = false, mismatch = false,
       fake_pending = false, tracking_error = false;
  cudaError_t pending = cudaSuccess, returned = cudaErrorMemoryAllocation;
  std::vector<void*> memory;
  std::vector<cudaEvent_t> handles;
  std::vector<cudaStream_t> stream_handles;
  std::vector<char> order;
};
FaultState state;
void Pending(cudaError_t e) {
  state.fake_pending = true;
  state.pending = e;
}
cudaError_t FaultPeek() {
  return state.fake_pending ? state.pending : cudaPeekAtLastError();
}
cudaError_t FaultTake() {
  if (!state.fake_pending) return cudaGetLastError();
  auto e = state.pending;
  state.fake_pending = false;
  state.pending = cudaSuccess;
  return e;
}
cudaError_t FaultMalloc(void** out, std::size_t bytes) {
  bool fail = ++state.allocations == state.fail_allocation;
  *out = nullptr;
  if (!fail || state.failed_output) {
    auto e = cudaMalloc(out, bytes);
    if (e != cudaSuccess) return e;
    state.memory.push_back(*out);
    state.allocated_bytes += bytes;
  }
  if (!fail) return cudaSuccess;
  if (state.null_success) return cudaSuccess;
  Pending(state.mismatch ? cudaErrorUnknown : state.returned);
  return state.returned;
}
template <class T>
cudaError_t FaultMalloc(T** out, std::size_t bytes) {
  return FaultMalloc(reinterpret_cast<void**>(out), bytes);
}
template <class T>
void Erase(std::vector<T>& items, T value) {
  auto it = std::find(items.begin(), items.end(), value);
  if (it == items.end())
    state.tracking_error = true;
  else
    items.erase(it);
}
cudaError_t FaultFree(void* p) {
  state.order.push_back('F');
  ++state.frees;
  auto e = cudaFree(p);
  if (e != cudaSuccess) return e;
  Erase(state.memory, p);
  return state.frees == state.fail_free ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t FaultCreate(cudaEvent_t* out) {
  bool fail = ++state.events == state.fail_event;
  *out = nullptr;
  if (!fail || state.failed_output) {
    auto e = cudaEventCreate(out);
    if (e != cudaSuccess) return e;
    state.handles.push_back(*out);
  }
  if (!fail) return cudaSuccess;
  if (state.null_success) return cudaSuccess;
  Pending(state.mismatch ? cudaErrorUnknown : state.returned);
  return state.returned;
}
cudaError_t FaultDestroy(cudaEvent_t h) {
  state.order.push_back('E');
  ++state.destroys;
  auto e = cudaEventDestroy(h);
  if (e != cudaSuccess) return e;
  Erase(state.handles, h);
  return state.destroys == state.fail_destroy ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t FaultStreamCreate(cudaStream_t* out, unsigned flags) {
  bool fail = ++state.streams == state.fail_stream;
  *out = nullptr;
  if (!fail || state.failed_output) {
    auto e = cudaStreamCreateWithFlags(out, flags);
    if (e != cudaSuccess) return e;
    state.stream_handles.push_back(*out);
  }
  if (!fail) return cudaSuccess;
  if (state.null_success) return cudaSuccess;
  Pending(state.mismatch ? cudaErrorUnknown : state.returned);
  return state.returned;
}
cudaError_t FaultStreamDestroy(cudaStream_t h) {
  state.order.push_back('D');
  ++state.stream_destroys;
  auto e = cudaStreamDestroy(h);
  if (e != cudaSuccess) return e;
  Erase(state.stream_handles, h);
  return state.stream_destroys == state.fail_stream_destroy ? cudaErrorUnknown
                                                            : cudaSuccess;
}
cudaError_t FaultStreamSync(cudaStream_t h) {
  state.order.push_back('S');
  ++state.syncs;
  auto e = cudaStreamSynchronize(h);
  if (e != cudaSuccess) return e;
  return state.syncs == state.fail_sync ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t FaultDeviceSync() {
  state.order.push_back('S');
  ++state.syncs;
  auto e = cudaDeviceSynchronize();
  if (e != cudaSuccess) return e;
  return state.syncs == state.fail_sync ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t FaultCopy(void* to,
                      const void* from,
                      std::size_t bytes,
                      cudaMemcpyKind kind) {
  if (++state.copies == state.fail_copy) return cudaErrorInvalidValue;
  return cudaMemcpy(to, from, bytes, kind);
}
cudaError_t FaultCopyAsync(void* to,
                           const void* from,
                           std::size_t bytes,
                           cudaMemcpyKind kind,
                           cudaStream_t stream) {
  if (++state.copies == state.fail_copy) return cudaErrorInvalidValue;
  const auto result = cudaMemcpyAsync(to, from, bytes, kind, stream);
  if (result == cudaSuccess && kind == cudaMemcpyDeviceToHost) {
    state.readback_destination = to;
    state.readback_bytes = bytes;
    if (state.fail_after_readback) return cudaErrorUnknown;
  }
  if (result == cudaSuccess && state.fail_boundary_host_allocation &&
      state.copies == 5) {
    fail_next_host_allocation = true;
  }
  return result;
}
}  // namespace

namespace CleanupFailureCub {
struct DeviceRadixSort {
  template <class Key, class Value>
  static cudaError_t SortPairs(void* temporary,
                               std::size_t& bytes,
                               const Key* keys_in,
                               Key* keys_out,
                               const Value* values_in,
                               Value* values_out,
                               int count,
                               int begin_bit = 0,
                               int end_bit = sizeof(Key) * 8,
                               cudaStream_t stream = nullptr) {
    const auto result = cub::DeviceRadixSort::SortPairs(temporary,
                                                        bytes,
                                                        keys_in,
                                                        keys_out,
                                                        values_in,
                                                        values_out,
                                                        count,
                                                        begin_bit,
                                                        end_bit,
                                                        stream);
    if (temporary && result == cudaSuccess) {
      state.sort_ranges = {keys_in, keys_out, values_in, values_out};
      state.sort_bytes = bytes;
      state.sort_stream = stream;
      state.order.push_back('R');
    }
    return result;
  }
};
struct DeviceSelect {
  template <class Input, class Output, class Count, class Predicate>
  static cudaError_t If(void* temporary,
                        std::size_t& bytes,
                        Input input,
                        Output output,
                        Count count,
                        int items,
                        Predicate predicate,
                        cudaStream_t stream = nullptr) {
    const auto result = cub::DeviceSelect::If(
        temporary, bytes, input, output, count, items, predicate, stream);
    if (temporary && result == cudaSuccess) {
      state.select_ranges = {input, output, count};
      state.select_bytes = bytes;
      state.select_stream = stream;
      state.order.push_back('L');
    }
    return result;
  }
};
}  // namespace CleanupFailureCub

// Use the actual implementations with isolated symbols and allocation-helper
// namespace. Faults return errors after real releases, without poisoning CUDA.
#define colmap CleanupFailureColmap
#define cub CleanupFailureCub
#define PoissonNormalGpu CleanupFailureNormal
#define PoissonTrimGpu CleanupFailureTrim
#define cudaMalloc FaultMalloc
#define cudaFree FaultFree
#define cudaEventCreate FaultCreate
#define cudaEventDestroy FaultDestroy
#define cudaStreamCreateWithFlags FaultStreamCreate
#define cudaStreamDestroy FaultStreamDestroy
#define cudaStreamSynchronize FaultStreamSync
#define cudaDeviceSynchronize FaultDeviceSync
#define cudaPeekAtLastError FaultPeek
#define cudaGetLastError FaultTake
#define cudaMemcpy FaultCopy
#define cudaMemcpyAsync FaultCopyAsync
#include "normal_gpu.cu"
#include "trim_gpu.cu"
#undef cudaMemcpyAsync
#undef cudaMemcpy
#undef cudaGetLastError
#undef cudaPeekAtLastError
#undef cudaDeviceSynchronize
#undef cudaStreamSynchronize
#undef cudaStreamDestroy
#undef cudaStreamCreateWithFlags
#undef cudaEventDestroy
#undef cudaEventCreate
#undef cudaFree
#undef cudaMalloc
#undef PoissonTrimGpu
#undef PoissonNormalGpu
#undef colmap
#undef cub

namespace {
class GpuCleanupFaultTest : public ::testing::TestWithParam<int> {
 protected:
  std::array<float, 6> output;
  std::array<std::uint32_t, 2> touched, roots;
  std::vector<std::uint64_t> graph;
  int ready_count = 0, consume_count = 0;
  bool throw_ready = false, nonmanifold = false;
  void SetUp() override {
    int count = 0;
    auto e = cudaGetDeviceCount(&count);
    if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver ||
        (e == cudaSuccess && !count)) {
      cudaGetLastError();
      GTEST_SKIP() << "CUDA device unavailable";
    }
    ASSERT_EQ(e, cudaSuccess);
    Reset();
    CleanupFailureNormal::Stats n;
    CleanupFailureTrim::Stats t;
    if (!CleanupFailureNormal::Available(n) ||
        !CleanupFailureTrim::Available(t))
      GTEST_SKIP() << "Required native kernel unavailable";
  }
  void TearDown() override {
    EXPECT_TRUE(state.memory.empty());
    EXPECT_TRUE(state.handles.empty());
    EXPECT_TRUE(state.stream_handles.empty());
    EXPECT_FALSE(state.tracking_error);
    // A failing test still releases real resources; forced recovery is not a
    // pass.
    for (auto p : state.memory) cudaFree(p);
    for (auto h : state.handles) cudaEventDestroy(h);
    for (auto h : state.stream_handles) {
      cudaStreamSynchronize(h);
      cudaStreamDestroy(h);
    }
    fail_next_host_allocation = false;
    state = FaultState{};
    cudaGetLastError();
  }
  void Reset() {
    ASSERT_TRUE(state.memory.empty());
    ASSERT_TRUE(state.handles.empty());
    ASSERT_TRUE(state.stream_handles.empty());
    EXPECT_FALSE(state.tracking_error);
    state = FaultState{};
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    output.fill(-77);
    touched.fill(77);
    roots.fill(77);
    graph = {999};
    ready_count = consume_count = 0;
    throw_ready = nonmanifold = false;
  }
  bool Invoke() {
    if (GetParam() < 2) {
      CleanupFailureNormal::Record record{};
      std::fill(std::begin(record.node), std::end(record.node), UINT32_MAX);
      record.node[0] = 0;
      record.coefficient[0] = 1;
      record.value[0] = 1;
      record.value[1] = 2;
      record.value[2] = 3;
      const auto* block = &record;
      const std::size_t count = 1;
      CleanupFailureNormal::Stats stats;
      if (GetParam() == 0)
        return CleanupFailureNormal::Reduce(
            &block, &count, 1, 2, output.data(), touched.data(), stats);
      CleanupFailureNormal::SlabCallbacks callbacks{
          this,
          [](void* p) {
            auto& self = *static_cast<GpuCleanupFaultTest*>(p);
            ++self.ready_count;
            if (self.throw_ready)
              throw std::logic_error("ready callback failure");
          },
          [](void* p,
             std::size_t first,
             std::size_t count,
             const float* field,
             const std::uint32_t* flags) {
            auto& self = *static_cast<GpuCleanupFaultTest*>(p);
            ++self.consume_count;
            std::copy(
                field, field + count * 3, self.output.begin() + first * 3);
            std::copy(flags, flags + count, self.touched.begin() + first);
          }};
      return CleanupFailureNormal::ReduceSlabs(
          &block, &count, 1, 2, 1, callbacks, stats);
    }
    CleanupFailureTrim::Stats stats;
    if (nonmanifold) {
      const std::uint64_t offsets[] = {0, 3, 6, 9};
      const std::uint32_t vertices[] = {0, 1, 2, 1, 0, 3, 0, 1, 4},
                          components[] = {0, 1, 2};
      return CleanupFailureTrim::BoundaryGraph(
          3, 9, offsets, vertices, components, graph, stats);
    }
    const std::uint64_t offsets[] = {0, 3, 6};
    const std::uint32_t vertices[] = {0, 1, 2, 2, 1, 3}, components[] = {0, 1};
    if (GetParam() == 2)
      return CleanupFailureTrim::ConnectedComponents(
          2, 6, offsets, vertices, roots.data(), stats);
    return CleanupFailureTrim::BoundaryGraph(
        2, 6, offsets, vertices, components, graph, stats);
  }
  void Untouched() {
    for (auto v : output) EXPECT_EQ(v, -77);
    for (auto v : touched) EXPECT_EQ(v, 77);
    for (auto v : roots) EXPECT_EQ(v, 77);
    EXPECT_EQ(graph, (std::vector<std::uint64_t>{999}));
    EXPECT_EQ(ready_count, 0);
    EXPECT_EQ(consume_count, 0);
  }
  void Released() {
    EXPECT_TRUE(state.memory.empty());
    EXPECT_TRUE(state.handles.empty());
    EXPECT_TRUE(state.stream_handles.empty());
    EXPECT_FALSE(state.tracking_error);
  }
  void Success() {
    ASSERT_TRUE(Invoke());
    Released();
    if (GetParam() < 2) {
      EXPECT_EQ(output, (std::array<float, 6>{1, 2, 3, 0, 0, 0}));
      EXPECT_EQ(touched, (std::array<std::uint32_t, 2>{1, 0}));
    } else if (GetParam() == 2)
      EXPECT_EQ(roots, (std::array<std::uint32_t, 2>{1, 1}));
    else
      EXPECT_EQ(graph, (std::vector<std::uint64_t>{1}));
  }
};

TEST_P(GpuCleanupFaultTest, CleanRealExecution) {
  Success();
  if (GetParam() == 1) {
    EXPECT_EQ(ready_count, 1);
    EXPECT_EQ(consume_count, 2);
  }
}
TEST_P(GpuCleanupFaultTest, AcquisitionRefusalAndInvalidOutputs) {
  Success();
  const std::array<int, 3> counts{
      state.allocations, state.events, state.streams};
  for (int domain = 0; domain < 3; ++domain)
    for (int position = 1; position <= counts[domain]; ++position)
      for (int fault = 0; fault < 4; ++fault) {
        SCOPED_TRACE(::testing::Message()
                     << domain << ":" << position << ":" << fault);
        Reset();
        if (domain == 0)
          state.fail_allocation = position;
        else if (domain == 1)
          state.fail_event = position;
        else
          state.fail_stream = position;
        state.failed_output = fault == 1;
        state.null_success = fault == 2;
        state.mismatch = fault == 3;
        if (fault == 0) {
          EXPECT_FALSE(Invoke());
          Untouched();
          EXPECT_EQ(state.copies, 0);
        } else
          EXPECT_ANY_THROW(Invoke());
        Released();
        Reset();
        Success();
      }
}
TEST_P(GpuCleanupFaultTest, CleanupAfterSuccessfulWorkIsTerminal) {
  Success();
  const std::array<int, 3> counts{
      state.frees, state.destroys, state.stream_destroys};
  for (int domain = 0; domain < 3; ++domain)
    for (int position = 1; position <= counts[domain]; ++position) {
      SCOPED_TRACE(::testing::Message() << domain << ":" << position);
      Reset();
      if (domain == 0)
        state.fail_free = position;
      else if (domain == 1)
        state.fail_destroy = position;
      else
        state.fail_stream_destroy = position;
      EXPECT_ANY_THROW(Invoke());
      Released();
      Reset();
      Success();
    }
}
TEST_P(GpuCleanupFaultTest, ExecutionFailureQuiescesBeforeDeviceRelease) {
  Reset();
  state.fail_copy = 1;
  EXPECT_THROW(Invoke(), std::runtime_error);
  Released();
  if (GetParam() >= 2) {
    auto sync = std::find(state.order.begin(), state.order.end(), 'S'),
         free = std::find(state.order.begin(), state.order.end(), 'F');
    EXPECT_NE(sync, state.order.end());
    EXPECT_LT(sync, free);
  }
  Reset();
  Success();
  const int syncs = state.syncs;
  for (int position = 1; position <= syncs; ++position) {
    Reset();
    state.fail_sync = position;
    EXPECT_THROW(Invoke(), std::runtime_error);
    Released();
    Reset();
    Success();
  }
}
TEST_P(GpuCleanupFaultTest, OriginalCauseSurvivesCleanupFailure) {
  Reset();
  state.fail_copy = 1;
  state.fail_free = 1;
  bool nested = false;
  try {
    Invoke();
    FAIL() << "expected combined failure";
  } catch (const std::exception& e) {
    try {
      std::rethrow_if_nested(e);
    } catch (const std::exception& cause) {
      nested = std::string(cause.what()).find("cleanup failure") ==
               std::string::npos;
    }
  }
  EXPECT_TRUE(nested);
  Released();
  Reset();
  Success();
}
TEST_P(GpuCleanupFaultTest, AcquisitionFailureWithCleanupErrorIsTerminal) {
  Reset();
  state.fail_allocation = 2;
  state.fail_free = 1;
  EXPECT_ANY_THROW(Invoke());
  EXPECT_EQ(FaultPeek(), cudaSuccess);
  Released();
  Reset();
  Success();
  if (GetParam() >= 2) {
    Reset();
    state.fail_event = 2;
    state.fail_destroy = 1;
    EXPECT_ANY_THROW(Invoke());
    Released();
    Reset();
    Success();
    Reset();
    state.fail_stream = 1;
    state.failed_output = true;
    state.fail_stream_destroy = 1;
    EXPECT_THROW(Invoke(), std::runtime_error);
    Released();
    Reset();
    Success();
  }
}
TEST_P(GpuCleanupFaultTest, CallbackAndRefusalBoundaries) {
  if (GetParam() == 0) {
    Reset();
    CleanupFailureNormal::Stats stats;
    const std::size_t count = 1;
    EXPECT_FALSE(CleanupFailureNormal::Reduce(
        nullptr, &count, 1, 2, output.data(), touched.data(), stats));
    Untouched();
    EXPECT_EQ(state.allocations, 0);
    Reset();
    Success();
  }
  if (GetParam() == 2) {
    Reset();
    CleanupFailureTrim::Stats stats;
    EXPECT_TRUE(CleanupFailureTrim::ConnectedComponents(
        0, 0, nullptr, nullptr, roots.data(), stats));
    Untouched();
    EXPECT_EQ(state.allocations, 0);
    Reset();
    Success();
  }
  if (GetParam() == 1) {
    Reset();
    throw_ready = true;
    state.fail_free = 1;
    bool cause = false;
    try {
      Invoke();
      FAIL() << "expected callback plus cleanup error";
    } catch (const std::exception& e) {
      try {
        std::rethrow_if_nested(e);
      } catch (const std::logic_error&) {
        cause = true;
      }
    }
    EXPECT_TRUE(cause);
    EXPECT_EQ(ready_count, 1);
    EXPECT_EQ(consume_count, 0);
    Released();
    Reset();
    state.fail_copy = 3;
    EXPECT_THROW(Invoke(), std::runtime_error);
    EXPECT_EQ(ready_count, 1);
    EXPECT_EQ(consume_count, 0);
    Released();
    Reset();
    Success();
  }
  if (GetParam() == 3) {
    Reset();
    nonmanifold = true;
    EXPECT_FALSE(Invoke());
    EXPECT_EQ(graph, (std::vector<std::uint64_t>{999}));
    Released();
    Reset();
    nonmanifold = true;
    state.fail_free = 1;
    EXPECT_THROW(Invoke(), std::runtime_error);
    EXPECT_EQ(graph, (std::vector<std::uint64_t>{999}));
    Released();
    Reset();
    Success();
  }
}
class GpuHostReadbackFaultTest : public GpuCleanupFaultTest {};
TEST_P(GpuHostReadbackFaultTest, HostAllocationTimingControlsRefusal) {
  Reset();
  // Fault instrumentation must not allocate between the intended seam and the
  // backend readback allocation; all tracking capacity is prepared first.
  state.memory.reserve(32);
  state.handles.reserve(8);
  state.stream_handles.reserve(2);
  state.order.reserve(64);
  state.fail_boundary_host_allocation = GetParam() == 3;
  bool result = true, threw = false;
  {
    struct ResetAllocationFault {
      ~ResetAllocationFault() { fail_next_host_allocation = false; }
    } reset_fault;
    if (GetParam() != 3) fail_next_host_allocation = true;
    try {
      result = Invoke();
    } catch (const std::bad_alloc&) {
      threw = true;
    }
  }
  if (GetParam() == 3) {
    EXPECT_TRUE(threw);
    EXPECT_EQ(graph, (std::vector<std::uint64_t>{999}));
  } else {
    EXPECT_FALSE(threw);
    EXPECT_FALSE(result);
    EXPECT_EQ(state.allocations, 0);
    Untouched();
  }
  Released();
  Reset();
  Success();
}
INSTANTIATE_TEST_SUITE_P(HostReadbackApis,
                         GpuHostReadbackFaultTest,
                         ::testing::Values(1, 3));
class GpuConnectedReadbackTest : public GpuCleanupFaultTest {};
TEST_P(GpuConnectedReadbackTest, UsesCallerStorageWithoutStagingAllocation) {
  Reset();
  // Reserve fault-tracking storage before arming the application allocator.
  state.memory.reserve(32);
  state.handles.reserve(8);
  state.stream_handles.reserve(2);
  state.order.reserve(64);
  bool result = false, unused_allocation_fault = false;
  {
    struct ResetAllocationFault {
      ~ResetAllocationFault() { fail_next_host_allocation = false; }
    } reset_fault;
    fail_next_host_allocation = true;
    result = Invoke();
    unused_allocation_fault = fail_next_host_allocation;
  }
  EXPECT_TRUE(result);
  EXPECT_TRUE(unused_allocation_fault);
  EXPECT_EQ(state.readback_destination, roots.data());
  EXPECT_EQ(state.readback_bytes, sizeof(roots));
  EXPECT_EQ(roots, (std::array<std::uint32_t, 2>{1, 1}));
  Released();
}
TEST_P(GpuConnectedReadbackTest,
       QueuedReadbackFailureSynchronizesCallerStorage) {
  Reset();
  state.fail_after_readback = true;
  EXPECT_THROW(Invoke(), std::runtime_error);
  EXPECT_EQ(state.readback_destination, roots.data());
  EXPECT_EQ(state.readback_bytes, sizeof(roots));
  const auto sync = std::find(state.order.begin(), state.order.end(), 'S');
  const auto free = std::find(state.order.begin(), state.order.end(), 'F');
  EXPECT_NE(sync, state.order.end());
  EXPECT_LT(sync, free);
  // The real synchronization succeeds before the injected error propagates.
  // Throwing does not promise that caller outputs remain unchanged.
  EXPECT_EQ(roots, (std::array<std::uint32_t, 2>{1, 1}));
  Released();
  Reset();
  Success();
}
INSTANTIATE_TEST_SUITE_P(ConnectedReadbackApi,
                         GpuConnectedReadbackTest,
                         ::testing::Values(2));
class GpuBoundaryStorageTest : public GpuCleanupFaultTest {};
TEST_P(GpuBoundaryStorageTest, RetiredSortInputsOwnSelectionStorage) {
  Reset();
  Success();
  RecordProperty("entries", 6);
  RecordProperty("device_allocation_requests", state.allocations);
  RecordProperty("device_requested_bytes",
                 std::to_string(state.allocated_bytes));
  for (std::size_t i = 0; i < state.sort_ranges.size(); ++i) {
    ASSERT_NE(state.sort_ranges[i], nullptr);
    for (std::size_t j = 0; j < i; ++j) {
      const auto a = reinterpret_cast<std::uintptr_t>(state.sort_ranges[i]);
      const auto b = reinterpret_cast<std::uintptr_t>(state.sort_ranges[j]);
      EXPECT_TRUE(a + 6 * sizeof(std::uint64_t) <= b ||
                  b + 6 * sizeof(std::uint64_t) <= a);
    }
  }
  EXPECT_EQ(state.select_ranges[0], state.sort_ranges[0]);
  EXPECT_EQ(state.select_ranges[1], state.sort_ranges[2]);
  EXPECT_NE(state.select_ranges[0], state.select_ranges[1]);
  EXPECT_NE(state.select_ranges[2], state.select_ranges[0]);
  EXPECT_NE(state.select_ranges[2], state.select_ranges[1]);
  const auto count = reinterpret_cast<std::uintptr_t>(state.select_ranges[2]);
  for (int i = 0; i < 2; ++i) {
    const auto range = reinterpret_cast<std::uintptr_t>(state.select_ranges[i]);
    EXPECT_TRUE(count + sizeof(int) <= range ||
                range + 6 * sizeof(std::uint64_t) <= count);
  }
  EXPECT_EQ(state.sort_stream, state.select_stream);
  EXPECT_NE(state.sort_stream, nullptr);
  EXPECT_LT(std::find(state.order.begin(), state.order.end(), 'R'),
            std::find(state.order.begin(), state.order.end(), 'L'));
}
TEST_P(GpuBoundaryStorageTest, VariedGraphsMatchIndependentEdgeIncidence) {
  using Face = std::vector<std::uint32_t>;
  struct Input {
    std::vector<Face> faces;
    std::vector<std::uint32_t> labels;
  };
  std::vector<Input> cases{
      {{}, {}},
      {{{}}, {0}},
      {{{0}}, {0}},
      {{{0, 1, 2}}, {0}},
      {{{0, 1, 2}, {1, 0, 3}}, {0, 0}},
      {{{0, 1, 2}, {1, 0, 3}}, {7, UINT32_MAX}},
      {{{0, 1, 2}, {0, 1, 3}}, {0, 1}},
      {{{0, 0, 1}, {0, 0, 2}}, {0, 1}},
      {{{0, 1, 2}, {1, 0, 3}, {0, 1, 4}}, {0, 1, 2}},
      {{{0, 1, 2}, {1, 0, 3}, {4, 5, 6}, {5, 4, 7}}, {0, 1, 0, 1}}};
  for (std::uint32_t side : {2u, 17u, 257u}) {
    Input input;
    for (std::uint32_t y = 0; y < side; ++y)
      for (std::uint32_t x = 0; x < side; ++x) {
        const auto a = y * (side + 1) + x, b = a + 1, c = a + side + 1,
                   d = c + 1;
        input.faces.push_back({a, b, c});
        input.faces.push_back({b, d, c});
        input.labels.push_back((x + 3 * y) % 5);
        input.labels.push_back((2 * x + y) % 5);
      }
    cases.push_back(std::move(input));
  }
  for (std::size_t test = 0; test < cases.size(); ++test) {
    SCOPED_TRACE(test);
    Reset();
    const auto& input = cases[test];
    std::map<std::pair<std::uint32_t, std::uint32_t>,
             std::vector<std::pair<std::uint32_t, bool>>>
        incidence;
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::uint32_t> vertices;
    for (std::size_t f = 0; f < input.faces.size(); ++f) {
      const auto& face = input.faces[f];
      for (std::size_t i = 0; i < face.size(); ++i) {
        const auto a = face[i], b = face[(i + 1) % face.size()];
        vertices.push_back(a);
        if (a != b)
          incidence[{std::min(a, b), std::max(a, b)}].push_back(
              {input.labels[f], a > b});
      }
      offsets.push_back(vertices.size());
    }
    bool manifold = true;
    std::set<std::pair<std::uint32_t, std::uint32_t>> adjacent;
    for (const auto& item : incidence) {
      const auto& edges = item.second;
      if (edges.size() > 2) manifold = false;
      if (edges.size() == 2 && edges[0].first != edges[1].first &&
          edges[0].second != edges[1].second)
        adjacent.emplace(std::min(edges[0].first, edges[1].first),
                         std::max(edges[0].first, edges[1].first));
    }
    std::vector<std::uint64_t> expected;
    for (const auto& edge : adjacent)
      expected.push_back((std::uint64_t(edge.first) << 32) | edge.second);
    CleanupFailureTrim::Stats stats;
    const auto result = CleanupFailureTrim::BoundaryGraph(input.faces.size(),
                                                          vertices.size(),
                                                          offsets.data(),
                                                          vertices.data(),
                                                          input.labels.data(),
                                                          graph,
                                                          stats);
    EXPECT_EQ(result, manifold);
    if (manifold)
      EXPECT_EQ(graph, expected);
    else
      EXPECT_EQ(graph, (std::vector<std::uint64_t>{999}));
    if (!input.faces.empty() && !vertices.empty()) {
      const auto fixed = (input.faces.size() + 1ull) * sizeof(std::uint64_t) +
                         4ull * vertices.size() * sizeof(std::uint64_t) +
                         vertices.size() * sizeof(std::uint32_t) +
                         input.faces.size() * sizeof(std::uint32_t) +
                         sizeof(std::uint32_t) + sizeof(int);
      EXPECT_EQ(stats.device_bytes,
                fixed + std::max(state.sort_bytes, state.select_bytes));
      EXPECT_EQ(state.allocated_bytes, stats.device_bytes);
    } else
      EXPECT_EQ(state.allocated_bytes, 0);
    Released();
  }
}
INSTANTIATE_TEST_SUITE_P(BoundaryStorageApi,
                         GpuBoundaryStorageTest,
                         ::testing::Values(3));
INSTANTIATE_TEST_SUITE_P(ActualImplementations,
                         GpuCleanupFaultTest,
                         ::testing::Values(0, 1, 2, 3));
}  // namespace
