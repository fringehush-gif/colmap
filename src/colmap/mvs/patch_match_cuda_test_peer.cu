// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/oct_normal_state.cuh"
#include "colmap/mvs/packed_ncc20_cost_map.h"
#include "colmap/mvs/packed_ncc_cost_map.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/patch_match_cuda_test_peer.h"
#include "colmap/mvs/patch_match_probability_unorm16.h"
#include "colmap/util/cuda.h"
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
    case 20:
      error = owner.fixed20_cost_map_->View().error;
      break;
    default:
      THROW_CHECK(false) << "Unknown test storage format";
      return;
  }
  const uint32_t invalid = 1u;
  CUDA_SAFE_CALL(
      cudaMemcpy(error, &invalid, sizeof(invalid), cudaMemcpyHostToDevice));
}

PatchMatchTestGeometry PatchMatchCudaTestPeer::Geometry(
    const PatchMatchCuda& owner) {
  return {{owner.depth_map_->GetWidth(), owner.depth_map_->GetHeight()},
          {owner.oct_normal_map_ ? owner.oct_normal_map_->GetWidth()
                                 : owner.normal_map_->GetWidth(),
           owner.oct_normal_map_ ? owner.oct_normal_map_->GetHeight()
                                 : owner.normal_map_->GetHeight()},
          {owner.CostMapWidth(), owner.CostMapHeight()},
          {owner.ref_image_->GetWidth(), owner.ref_image_->GetHeight()},
          {owner.prev_sel_prob_map_->GetWidth(),
           owner.prev_sel_prob_map_->GetHeight()},
          owner.oct_normal_map_ ? owner.oct_normal_map_->GetFrame() : 0u,
          owner.rotation_in_half_pi_};
}

void PatchMatchCudaTestPeer::RotateThree(PatchMatchCuda& owner) {
  for (int i = 0; i < 3; ++i) owner.Rotate();
}

void PatchMatchCudaTestPeer::CheckUsable(const PatchMatchCuda& owner) {
  owner.CheckUsable();
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

void* PatchMatchCudaTestPeer::OctNormalAllocationPointer(
    const PatchMatchCuda& owner) {
  THROW_CHECK(owner.oct_normal_map_);
  return owner.oct_normal_map_->View().direction;
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
