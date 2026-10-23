// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace AuxNativeGpu {
inline thread_local bool native_reference = false;
struct Four {
  float v[4];
};
static_assert(sizeof(Four) == 16, "Aux float4 ABI");
// Nodes are depth-major; child lists retain native octant order. Every node
// belongs to the union of admitted sample ancestor paths, including zero seeds.
struct View {
  std::uint32_t nodes, seeds, max_depth;
  const std::uint32_t *level_offsets, *child_offsets, *children, *seed_offsets,
      *seed_ids;
  const Four* raw_seeds;
};
struct Stats {
  double total_seconds = 0, upload_seconds = 0, kernel_seconds = 0,
         download_seconds = 0;
  std::uint64_t device_bytes = 0, integral_channels = 0;
  double global_sums[4] = {};
  const char* reason = "not-run";
};
// Output is scaled by 4^localDepth. The existing native 32^depth pass remains.
// False means inadmissible input/capacity, before any reconstruction mutation.
// A CUDA execution failure throws; it must not masquerade as successful work.
// Preflight the actual gather kernel on the current device before host CSR
// packing. False reports a known unavailable device/kernel in stats.reason;
// other errors throw.
bool Available(Stats& stats);
bool Reduce(const View& input, Four* output, Stats& stats);
}  // namespace AuxNativeGpu
