// SPDX-License-Identifier: BSD-3-Clause
#include <cstdlib>
#include <new>

namespace PoissonGpuCleanupTest {
thread_local bool fail_next_host_allocation = false;
}
using PoissonGpuCleanupTest::fail_next_host_allocation;
// Isolated fault-test target; only explicit host-readback cases enable failure.
void* operator new(std::size_t bytes) {
  if (fail_next_host_allocation) {
    fail_next_host_allocation = false;
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
