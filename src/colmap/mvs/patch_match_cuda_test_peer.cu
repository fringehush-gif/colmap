// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/oct_normal_state.cuh"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/patch_match_cuda_test_peer.h"
#include "colmap/mvs/patch_match_probability_unorm16.h"
#include "colmap/util/cuda.h"
namespace colmap::mvs {
namespace {
__global__ void PoisonProbabilityValue(ProbabilityUnorm16View view) {
  view.Set(0, 0, 0, __int_as_float(0x7fc00000));
}
}  // namespace

void PatchMatchCudaTestPeer::PoisonProbability(PatchMatchCuda& owner) {
  THROW_CHECK(owner.prev_sel_prob_unorm16_map_);
  PoisonProbabilityValue<<<1, 1>>>(owner.prev_sel_prob_unorm16_map_->View());
  CUDA_SYNC_AND_CHECK();
}

void PatchMatchCudaTestPeer::PoisonOctNormal(PatchMatchCuda& owner) {
  THROW_CHECK(owner.oct_normal_map_);
  const unsigned int failure = 1u << 9;
  CUDA_SAFE_CALL(cudaMemcpy(owner.oct_normal_map_->View().error,
                            &failure,
                            sizeof(failure),
                            cudaMemcpyHostToDevice));
}
}  // namespace colmap::mvs
