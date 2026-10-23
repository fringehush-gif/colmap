// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "normal_gpu.h"

namespace PoissonNormalCapture {

// Only the owning Poisson::Solver normal-field call sets this admission token.
inline thread_local bool admitted = false;
inline thread_local std::vector<PoissonNormalGpu::Record>* target = nullptr;

struct Scope {
  bool previous;
  explicit Scope(bool enabled) : previous(admitted) { admitted = enabled; }
  ~Scope() { admitted = previous; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};

struct ThreadScope {
  std::vector<PoissonNormalGpu::Record>* previous;
  explicit ThreadScope(std::vector<PoissonNormalGpu::Record>* records)
      : previous(target) {
    target = records;
  }
  ~ThreadScope() { target = previous; }
  ThreadScope(const ThreadScope&) = delete;
  ThreadScope& operator=(const ThreadScope&) = delete;
};

// Replay the already evaluated native splats. The caller supplies its real
// tree index resolver and SparseNodeData insertion; no sample callback runs.
template <class Records, class Resolve, class Add>
void Replay(const Records& blocks,
            std::size_t node_count,
            Resolve&& resolve,
            Add&& add) {
  for (const auto& records : blocks)
    for (const auto& record : records)
      for (unsigned slot = 0; slot < 27; ++slot) {
        const std::uint32_t index = record.node[slot];
        if (index == UINT32_MAX) continue;
        if (index >= node_count)
          throw std::runtime_error("normal GPU replay missing node");
        auto* node = resolve(index);
        if (!node) throw std::runtime_error("normal GPU replay missing node");
        add(node, record.value, record.coefficient[slot]);
      }
}

}  // namespace PoissonNormalCapture
