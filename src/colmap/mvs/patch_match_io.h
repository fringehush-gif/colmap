// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/threading.h"

#include <exception>
#include <future>
#include <utility>
#include <vector>

namespace colmap {
namespace mvs {

// All accepted tasks may borrow the caller's local images and maps. A failed
// submission or read must not unwind those locals before sibling tasks finish.
// The caller serializes submissions to this shared pool.
template <typename Submit>
void RunPatchMatchInputReads(ThreadPool& pool, Submit&& submit) {
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
  // can rediscover a handled read error and terminate the process.
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
