// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/packed_ncc_cost_map.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/patch_match_cuda_test_peer.h"

namespace colmap::mvs {
void PatchMatchCudaTestPeer::SetCompactDomainError(PatchMatchCuda& owner,
                                                   int cost_bits) {
  uint32_t* error;
  switch (cost_bits) {
    case 0:
      error = owner.prev_sel_prob_unorm16_map_->View().invalid;
      break;
    case 16:
      error = owner.packed_cost_map_->View().domain_error;
      break;
    case 24:
      error = owner.fixed24_cost_map_->View().domain_error;
      break;
    default:
      THROW_CHECK(false) << "Unknown test storage format";
      return;
  }
  const uint32_t invalid = 1u;
  CUDA_SAFE_CALL(
      cudaMemcpy(error, &invalid, sizeof(invalid), cudaMemcpyHostToDevice));
}

}  // namespace colmap::mvs
