// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>

namespace PoissonNormalGpu {

// One native splat: slots retain WindowLoop order, including inactive slots.
struct Record {
  std::uint32_t node[27];
  float coefficient[27];
  float value[3];
};
static_assert(sizeof(Record) == 228, "normal record ABI");

struct Stats {
  const char* reason = nullptr;
  std::uint64_t device_bytes = 0;
};

bool Available(Stats& stats);
// False means only a known unavailable device/kernel or a prelaunch resource
// refusal. Once execution begins, CUDA and validation errors throw.
bool Reduce(const Record* const* blocks,
            const std::size_t* counts,
            std::size_t block_count,
            std::size_t node_count,
            float* output,
            std::uint32_t* touched,
            Stats& stats);

}  // namespace PoissonNormalGpu
