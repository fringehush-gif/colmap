// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/threading.h"

#include <atomic>
#include <system_error>

#include <gtest/gtest.h>

#if defined(COLMAP_TEST_PTHREAD_INTERPOSITION)
#if !defined(__linux__) || !defined(__GNUC__) || defined(__clang__)
#error "pthread failure injection requires Linux and GNU C++"
#endif
#include <cerrno>
#include <cstdlib>

#include <dlfcn.h>
#include <pthread.h>

namespace {

std::atomic<bool> record_calls{false};
std::atomic<int> create_calls{0};
std::atomic<int> created{0};
std::atomic<int> joined{0};
int fail_after = -1;

class ScopedPthreadFailure {
 public:
  explicit ScopedPthreadFailure(const int failure_index) {
    create_calls = 0;
    created = 0;
    joined = 0;
    fail_after = failure_index;
    record_calls = true;
  }
  ~ScopedPthreadFailure() { record_calls = false; }
};

}  // namespace

// Interposition also reaches pthread calls made inside shared libstdc++.
// Resolve the real OS functions; only the selected creation returns EAGAIN.
extern "C" int pthread_create(pthread_t* thread,
                              const pthread_attr_t* attributes,
                              void* (*entry)(void*),
                              void* argument) {
  using Create =
      int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
  static const auto real_create =
      reinterpret_cast<Create>(dlsym(RTLD_NEXT, "pthread_create"));
  if (real_create == nullptr) {
    std::abort();
  }
  const bool record = record_calls.load();
  if (record && create_calls.fetch_add(1) == fail_after) {
    return EAGAIN;
  }
  const int status = real_create(thread, attributes, entry, argument);
  if (record && status == 0) {
    created.fetch_add(1);
  }
  return status;
}

extern "C" int pthread_join(pthread_t thread, void** result) {
  using Join = int (*)(pthread_t, void**);
  static const auto real_join =
      reinterpret_cast<Join>(dlsym(RTLD_NEXT, "pthread_join"));
  if (real_join == nullptr) {
    std::abort();
  }
  const int status = real_join(thread, result);
  if (record_calls.load() && status == 0) {
    joined.fetch_add(1);
  }
  return status;
}
#endif

namespace colmap {
namespace {

TEST(ThreadPoolConstructor, ExecutesTasksAndJoinsWorkers) {
#if defined(COLMAP_TEST_PTHREAD_INTERPOSITION)
  const ScopedPthreadFailure injection(-1);
#endif
  std::atomic<int> completed{0};
  std::atomic<int> sum{0};
  std::atomic<int> invalid_worker_indices{0};
  {
    ThreadPool pool(4);
    EXPECT_EQ(pool.NumThreads(), 4);
    for (int value = 0; value < 32; ++value) {
      pool.AddTask([&pool, &completed, &sum, &invalid_worker_indices, value]() {
        const int index = pool.GetThreadIndex();
        if (index < 0 || index >= 4) {
          invalid_worker_indices.fetch_add(1);
        }
        sum.fetch_add(value);
        completed.fetch_add(1);
      });
    }
    pool.Wait();
    EXPECT_EQ(completed.load(), 32);
    EXPECT_EQ(sum.load(), 496);
    EXPECT_EQ(invalid_worker_indices.load(), 0);
  }
#if defined(COLMAP_TEST_PTHREAD_INTERPOSITION)
  EXPECT_EQ(create_calls.load(), 4);
  EXPECT_EQ(created.load(), 4);
  EXPECT_EQ(joined.load(), 4);
#endif
}

#if defined(COLMAP_TEST_PTHREAD_INTERPOSITION)
class ThreadPoolConstructorFailure : public testing::TestWithParam<int> {};

TEST_P(ThreadPoolConstructorFailure, JoinsWorkersAndPreservesError) {
  const int successful_creations = GetParam();
  const ScopedPthreadFailure injection(successful_creations);
  try {
    ThreadPool pool(4);
    FAIL() << "Expected thread creation failure";
  } catch (const std::system_error& error) {
    EXPECT_EQ(error.code(),
              std::make_error_code(std::errc::resource_unavailable_try_again));
  }
  EXPECT_EQ(create_calls.load(), successful_creations + 1);
  EXPECT_EQ(created.load(), successful_creations);
  EXPECT_EQ(joined.load(), successful_creations);
}

INSTANTIATE_TEST_SUITE_P(PartialCreation,
                         ThreadPoolConstructorFailure,
                         testing::Values(0, 1, 3));
#endif

}  // namespace
}  // namespace colmap
