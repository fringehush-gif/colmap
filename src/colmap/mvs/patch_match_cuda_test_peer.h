// SPDX-License-Identifier: BSD-3-Clause

#pragma once
namespace colmap::mvs {
class PatchMatchCuda;
class PatchMatchCudaTestPeer {
 public:
  static void PoisonOctNormal(PatchMatchCuda& owner);
};
}  // namespace colmap::mvs
