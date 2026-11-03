// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/oct_normal_state.cuh"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/patch_match_cuda_test_peer.h"
#include "colmap/util/cuda.h"
namespace colmap::mvs {
void PatchMatchCudaTestPeer::PoisonOctNormal(PatchMatchCuda& owner) {
  THROW_CHECK(owner.oct_normal_map_);
  const unsigned int failure = 1u << 9;
  CUDA_SAFE_CALL(cudaMemcpy(owner.oct_normal_map_->View().error,
                            &failure,
                            sizeof(failure),
                            cudaMemcpyHostToDevice));
}
}  // namespace colmap::mvs
