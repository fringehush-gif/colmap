// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/mat.h"

#include <array>
#include <cstdint>
#include <vector>

namespace colmap {
namespace mvs {
class PatchMatchCuda;
namespace internal {

struct PhotoInitializationResult {
  Mat<float> depth;
  Mat<float> normal;
  Mat<uint32_t> continuation;
  Mat<int> normal_pairs;
};

// Independent retained 020c three-stage initialization, implemented in the test
// CUDA translation unit. It does not call the new fused production helpers.
PhotoInitializationResult ProbeLegacyPhotoInitialization(
    int width,
    int height,
    const std::array<float, 4>& inv_K,
    float depth_min,
    float depth_max);

// Read-only state-copy draws in canonical perimeter order, before any reseed.
Mat<uint32_t> ProbeConstructorContinuation(PatchMatchCuda& patch_match);

struct SweepTraceResult {
  Mat<int> draws;
  Mat<int> winners;
  Mat<int> support;
  // Four slices per hypothesis: depth followed by three normal components.
  Mat<float> hypotheses;
  Mat<float> hypothesis_costs;
  Mat<float> forward;
  Mat<float> initial_cost;
  Mat<float> cost;
  Mat<float> depth;
  Mat<float> normal;
  Mat<float> selection;
  Mat<uint32_t> continuation;
  // Raw draws from a state copy before this sweep, for shared-corner checks.
  Mat<uint32_t> continuation_before;
};

SweepTraceResult ProbeSweep(PatchMatchCuda& patch_match,
                            bool cooperative,
                            unsigned long long seed);

// Four real sweeps and rotations, with one seed operation at sequence start.
// Uses the existing GEO trace contract and requires filtering to be disabled.
std::vector<SweepTraceResult> ProbeSweepSequence(
    PatchMatchCuda& patch_match, bool cooperative, unsigned long long seed);

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
