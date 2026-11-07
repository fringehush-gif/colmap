// SPDX-License-Identifier: BSD-3-Clause
#include <cstdlib>
#include <new>
#include <optional>
#include <string>

#include "trim_gpu.h"
#include <gtest/gtest.h>

namespace {
thread_local bool fail_next_allocation = false;
int backend_mode = 0, backend_calls = 0;
}  // namespace
// Isolated test target: fail one caller-thread allocation only inside host
// staging.
void* operator new(std::size_t bytes) {
  if (fail_next_allocation) {
    fail_next_allocation = false;
    throw std::bad_alloc();
  }
  if (void* result = std::malloc(bytes ? bytes : 1)) return result;
  throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

namespace PoissonTrimGpu {
bool Available(Stats&) { return true; }
bool ConnectedComponents(std::size_t count,
                         std::size_t,
                         const std::uint64_t*,
                         const std::uint32_t*,
                         std::uint32_t* roots,
                         Stats&) {
  ++backend_calls;
  if (backend_mode == 1) throw std::bad_alloc();
  if (backend_mode == 2) return false;
  for (std::size_t i = 0; i < count; ++i) roots[i] = std::uint32_t(i);
  return true;
}
bool BoundaryGraph(std::size_t,
                   std::size_t,
                   const std::uint64_t*,
                   const std::uint32_t*,
                   const std::uint32_t*,
                   std::vector<std::uint64_t>& graph,
                   Stats&) {
  ++backend_calls;
  if (backend_mode == 1) throw std::bad_alloc();
  if (backend_mode == 2) return false;
  graph.clear();
  return true;
}
}  // namespace PoissonTrimGpu
// The isolated host test target provides backend faults and does not link the
// production GPU backend. The real adapter and public declarations stay intact.
#include "trim_gpu_adapter.h"

namespace {
class TrimAdapterFailureTest : public ::testing::TestWithParam<bool> {
 protected:
  const std::vector<std::vector<int>> polygons{{0, 1, 2}}, partition{{0}};
  std::vector<std::vector<int>> components{{123}};
  std::vector<std::uint64_t> graph{999};
  PoissonTrimGpu::Stats stats;
  std::optional<std::string> previous;
  // This standalone test runs serially, creates no workers, and restores the
  // process environment after each case.
  // NOLINTBEGIN(concurrency-mt-unsafe)
  void SetMode(const char* value) {
#ifdef _WIN32
    ASSERT_EQ(_putenv_s("POISSON_GPU_TRIM", value ? value : ""), 0);
#else
    ASSERT_EQ(value ? setenv("POISSON_GPU_TRIM", value, 1)
                    : unsetenv("POISSON_GPU_TRIM"),
              0);
#endif
  }
  // NOLINTEND(concurrency-mt-unsafe)
  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (const char* value = std::getenv("POISSON_GPU_TRIM")) previous = value;
    SetMode("verify");
    backend_mode = backend_calls = 0;
  }
  void TearDown() override {
    fail_next_allocation = false;
    SetMode(previous ? previous->c_str() : nullptr);
  }
  bool Invoke() {
    return GetParam() ? PoissonTrimGpu::TryBoundaryGraph(
                            polygons, partition, graph, stats)
                      : PoissonTrimGpu::TryConnectedComponents(
                            polygons, components, stats);
  }
  void Untouched() {
    EXPECT_EQ(components, (std::vector<std::vector<int>>{{123}}));
    EXPECT_EQ(graph, (std::vector<std::uint64_t>{999}));
  }
};
TEST_P(TrimAdapterFailureTest, BackendAllocationErrorIsTerminal) {
  backend_mode = 1;
  EXPECT_THROW(Invoke(), std::bad_alloc);
  EXPECT_EQ(backend_calls, 1);
  Untouched();
}
TEST_P(TrimAdapterFailureTest, HostStagingAllocationMayRefuse) {
  struct AllocationFailureScope {
    AllocationFailureScope() { fail_next_allocation = true; }
    ~AllocationFailureScope() { fail_next_allocation = false; }
  };
  bool result = true;
  {
    AllocationFailureScope scope;
    result = Invoke();
  }
  EXPECT_FALSE(result);
  EXPECT_EQ(backend_calls, 0);
  EXPECT_STREQ(stats.reason, "host-allocation");
  Untouched();
}
TEST_P(TrimAdapterFailureTest, BackendRefusalLeavesOutputsUntouched) {
  backend_mode = 2;
  EXPECT_FALSE(Invoke());
  EXPECT_EQ(backend_calls, 1);
  Untouched();
}
TEST_P(TrimAdapterFailureTest, SuccessPublishesOutputs) {
  ASSERT_TRUE(Invoke());
  EXPECT_EQ(backend_calls, 1);
  if (GetParam())
    EXPECT_TRUE(graph.empty());
  else
    EXPECT_EQ(components, (std::vector<std::vector<int>>{{0}}));
}
INSTANTIATE_TEST_SUITE_P(BothAdapters,
                         TrimAdapterFailureTest,
                         ::testing::Values(false, true));
}  // namespace
