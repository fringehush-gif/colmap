// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace PoissonTrimGpu {

struct Stats {
  const char* reason = nullptr;
  double upload_ms = 0;
  double sort_ms = 0;
  double union_ms = 0;
  double download_ms = 0;
  double total_ms = 0;
  std::uint64_t device_bytes = 0;
};

// False means the current CUDA device cannot run the kernels. Other CUDA
// failures throw so they cannot be confused with an ordinary CPU fallback.
bool Available(Stats& stats);

// On false, caller-owned outputs remain untouched. Unexpected execution
// failures throw; only known prelaunch resource failures return false.
bool ConnectedComponents(std::size_t polygons,
                         std::size_t entries,
                         const std::uint64_t* offsets,
                         const std::uint32_t* vertices,
                         std::uint32_t* roots,
                         Stats& stats);

// A nonmanifold edge incidence returns false and leaves graph untouched.
bool BoundaryGraph(std::size_t polygons,
                   std::size_t entries,
                   const std::uint64_t* offsets,
                   const std::uint32_t* vertices,
                   const std::uint32_t* components,
                   std::vector<std::uint64_t>& graph,
                   Stats& stats);

}  // namespace PoissonTrimGpu
