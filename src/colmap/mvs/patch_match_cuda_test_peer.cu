// SPDX-License-Identifier: BSD-3-Clause

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
    default:
      THROW_CHECK(false) << "Unknown test storage format";
      return;
  }
  const uint32_t invalid = 1u;
  CUDA_SAFE_CALL(
      cudaMemcpy(error, &invalid, sizeof(invalid), cudaMemcpyHostToDevice));
}

void* PatchMatchCudaTestPeer::ConsistencyMaskPointer(
    const PatchMatchCuda& owner) {
  return owner.consistency_mask_->GetWidth() ? owner.consistency_mask_->GetPtr()
                                             : nullptr;
}

void* PatchMatchCudaTestPeer::AlternateDepthPointer(
    const PatchMatchCuda& owner) {
  THROW_CHECK(owner.alternate_depth_map_);
  return owner.alternate_depth_map_->GetPtr();
}

}  // namespace colmap::mvs
