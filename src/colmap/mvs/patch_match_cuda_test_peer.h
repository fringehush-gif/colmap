// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <utility>

namespace colmap::mvs {
struct PatchMatchTestGeometry {
  std::pair<std::size_t, std::size_t> depth, normal, cost, reference,
      probability;
  unsigned int normal_frame;
  int rotation;
};
class PatchMatchCuda;
class PatchMatchCudaTestPeer {
 public:
  static PatchMatchTestGeometry Geometry(const PatchMatchCuda& owner);
  static void RotateThree(PatchMatchCuda& owner);
  static void CheckUsable(const PatchMatchCuda& owner);
  static void PoisonOctNormal(PatchMatchCuda& owner);
  static void PoisonProbability(PatchMatchCuda& owner);
};
}  // namespace colmap::mvs
