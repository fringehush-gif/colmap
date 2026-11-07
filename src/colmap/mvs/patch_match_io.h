// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/threading.h"

#include <exception>
#include <future>
#include <utility>
#include <vector>

namespace colmap {
namespace mvs {

// Accepted tasks may borrow local inputs or controller state. Drain them before
// a failed submission or worker can unwind that state. Input readers retain
// futures to preserve individual read errors; the scheduler leaves them empty.
// The caller serializes submissions to this pool.
template <typename Submit>
void RunPatchMatchTasks(ThreadPool& pool, Submit&& submit) {
  std::vector<std::shared_future<void>> futures;
  std::exception_ptr first_failure;
  try {
    std::forward<Submit>(submit)(futures);
  } catch (...) {
    first_failure = std::current_exception();
  }

  for (auto& future : futures) {
    try {
      future.get();
    } catch (...) {
      if (!first_failure) {
        first_failure = std::current_exception();
      }
    }
  }

  // Wait also drains ThreadPool's failure checkers. Otherwise pool destruction
  // can rediscover a handled worker error and terminate the process.
  try {
    pool.Wait();
  } catch (...) {
    if (!first_failure) {
      first_failure = std::current_exception();
    }
  }
  if (first_failure) {
    std::rethrow_exception(first_failure);
  }
}

}  // namespace mvs
}  // namespace colmap
