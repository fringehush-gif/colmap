// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace colmap::mvs {
class PatchMatchCuda;
class PatchMatchCudaTestPeer {
 public:
  static void SetCompactDomainError(PatchMatchCuda& owner, int cost_bits);
  static void* ConsistencyMaskPointer(const PatchMatchCuda& owner);
  static void* AlternateDepthPointer(const PatchMatchCuda& owner);
};
}  // namespace colmap::mvs
