// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <limits>

namespace colmap::mvs::internal {

// Entry metadata must outlive every caller. Closing is permanent: an evicted
// cache key receives a new entry rather than reopening an old generation.
class TransientPins {
 public:
  bool TryAcquire() noexcept {
    size_t readers = readers_.load(std::memory_order_relaxed);
    while (readers < kClosed - 1) {
      if (readers_.compare_exchange_weak(readers,
                                         readers + 1,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }

  // The caller owns one pin and has finished accessing the payload. A true
  // result requires the containing cache's last-reader wakeup protocol.
  bool Release() noexcept {
    // The cache's waiter arm/recheck and last-reader flag check must also be
    // sequentially consistent; release/acquire across two atomics is not a
    // sufficient lost-wakeup protocol on every supported CPU.
    const size_t previous = readers_.fetch_sub(1, std::memory_order_seq_cst);
    assert(previous > 0 && previous < kClosed);
    return previous == 1;
  }

  // The cache mutex separately excludes ordinary pins and pending loads.
  // Success excludes every subsequent transient payload reader.
  bool TryClose() noexcept {
    size_t expected = 0;
    return readers_.compare_exchange_strong(expected,
                                            kClosed,
                                            std::memory_order_seq_cst,
                                            std::memory_order_seq_cst);
  }

  // SC because an armed capacity recheck must not miss a completed release.
  bool IsIdle() const noexcept {
    return readers_.load(std::memory_order_seq_cst) == 0;
  }
  bool IsClosed() const noexcept {
    return readers_.load(std::memory_order_seq_cst) == kClosed;
  }

 private:
  static constexpr size_t kClosed = std::numeric_limits<size_t>::max();
  std::atomic<size_t> readers_{0};
};

}  // namespace colmap::mvs::internal
