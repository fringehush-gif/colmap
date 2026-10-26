// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/gpu_mat_prng.h"
#include "colmap/mvs/gpu_mat_ref_image.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_hmm_checkpoint.h"
#include "colmap/mvs/patch_match_state_map.h"
#include "colmap/util/cuda_to_hip.h"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace colmap {
namespace mvs {

template <typename T>
class PatchMatchSourceArray;
class PatchMatchSparseSources;

class PatchMatchCuda {
 public:
  // Deferral is internal to same-shape batches after a full first owner has
  // qualified both capacities and the controller retained the parent budget.
  PatchMatchCuda(const PatchMatchOptions& options,
                 const PatchMatch::Problem& problem,
                 std::shared_ptr<const PatchMatchSparseSources> shared = {},
                 const PatchMatchCuda* geometry_donor = nullptr);
  ~PatchMatchCuda();

  void Run();

  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;
  std::vector<int> GetConsistentImageIdxs() const;
#ifndef COLMAP_HIP_ENABLED
  // Conservative owner-local estimate for admitting a second live instance.
  size_t PairPeakBytes() const;
  // Requires same-shape, serial rotations on the current CUDA device.
  bool CanShareGeometryRotation() const;
  void UseSparseSources(std::shared_ptr<const PatchMatchSparseSources> shared);
#endif

 private:
#ifndef COLMAP_HIP_ENABLED
  friend void RunPatchMatchPair(PatchMatchCuda& first,
                                PatchMatchCuda& second,
                                cudaStream_t first_stream,
                                cudaStream_t second_stream);
  friend void RunPatchMatchBatch(const std::vector<PatchMatchCuda*>& views,
                                 const std::vector<cudaStream_t>& streams);
#endif
  template <int kWindowSize, int kWindowStep, bool kCheckpointHmm>
  void RunWithWindowSizeAndStep();

  GpuMatView<float> GeometryView(const GpuMat<float>& owner) const;
  Mat<float> GeometryCopyToMat(const GpuMat<float>& owner) const;

  void ComputeCudaConfig();

  void BindRefImageTexture();

  void InitRefImage();
  GpuMatView<float> ReferenceStatisticsView(bool squared) const;
  cudaTextureObject_t ReferenceStatisticsTexture() const;
  void InitSourceImages();
  void InitTransforms();
  void InitWorkspaceMemory();
  void RotateStateMaps(size_t width, size_t height);
#ifndef COLMAP_HIP_ENABLED
  void InitAlternateOrientationMemory(const PatchMatchCuda* geometry_donor);
  void EnsureGeometryRotationMemory();
  static size_t GeometryBytes(const GpuMat<float>& depth,
                              const GpuMat<float>& normal);
  void BindReusableRefImageTexture();
  void RotateWithBufferReuse(bool pack_geometry);
#endif

  // Rotate reference image by 90 degrees in counter-clockwise direction.
  void Rotate(bool pack_geometry = false);

  // Packed logical planes can use padded physical owners of sufficient
  // capacity. GpuMat dimensions remain allocation metadata after a shared
  // rotation.
  bool dense_geometry_ = false;
  size_t geometry_alternate_bound_ = 0;
  size_t geometry_peak_floor_ = 0;

  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  // Dimensions for sweeping from top to bottom, i.e. one thread per column.
  dim3 sweep_block_size_;
  dim3 sweep_grid_size_;
  // Dimensions for element-wise operations, i.e. one thread per pixel.
  dim3 elem_wise_block_size_;
  dim3 elem_wise_grid_size_;

  // Original (not rotated) dimension of reference image.
  size_t ref_width_;
  size_t ref_height_;

  // Rotation of reference image in pi/2. This is equivalent to the number of
  // calls to `rotate` mod 4.
  int rotation_in_half_pi_;

  // Default-window source jobs per wave, chosen once for the bound device.
  int cooperative_sample_batch_size_ = 4;

  // Reference and source image input data.
  std::unique_ptr<CudaArrayLayeredTexture<uint8_t>> ref_image_texture_;
  std::unique_ptr<PatchMatchSourceArray<uint8_t>> src_images_texture_;
  std::unique_ptr<PatchMatchSourceArray<float>> src_depth_maps_texture_;

  // Relative poses from rotated versions of reference image to source images
  // corresponding to _rotationInHalfPi:
  //
  //    [S(1), S(2), S(3), ..., S(n)]
  //
  // where n is the number of source images and:
  //
  //    S(i) = [K_i(0, 0), K_i(0, 2), K_i(1, 1), K_i(1, 2), R_i(:), T_i(:)
  //            C_i(:), P(:), P^-1(:)]
  //
  // where i denotes the index of the source image and K is its calibration.
  // R, T, C, P, P^-1 denote the relative rotation, translation, camera
  // center, projection, and inverse projection from there reference to the
  // i-th source image.
  std::unique_ptr<CudaArrayLayeredTexture<float>> poses_texture_[4];

  // Calibration matrix for rotated versions of reference image
  // as {K[0, 0], K[0, 2], K[1, 1], K[1, 2]} corresponding to _rotationInHalfPi.
  float ref_K_host_[4][4];
  float ref_inv_K_host_[4][4];

  // Data for reference image.
  std::unique_ptr<GpuMat<uint8_t>> ref_image_;
  // Original-image FP32 statistics stored as two unconverted integer channels.
  std::unique_ptr<CudaArrayLayeredTexture<uint2>> ref_statistics_texture_;
  size_t ref_statistics_init_peak_bytes_ = 0;
  // Nondefault windows keep the original rotated FP32 planes.
  std::unique_ptr<GpuMat<float>> ref_sum_image_;
  std::unique_ptr<GpuMat<float>> ref_squared_sum_image_;
  std::unique_ptr<GpuMat<float>> depth_map_;
  std::unique_ptr<GpuMat<float>> normal_map_;
  std::unique_ptr<PatchMatchStateMap> sel_prob_map_;
  std::unique_ptr<PatchMatchStateMap> prev_sel_prob_map_;
  std::unique_ptr<PatchMatchStateMap> cost_map_;
  std::unique_ptr<PatchMatchHmmCheckpoint> hmm_checkpoint_;
  // Canonical original-image perimeter; corners retain one shared state.
  std::unique_ptr<GpuMat<curandState>> rand_state_map_;
  // Packed source-membership bits, one byte plane per eight source slots.
  // Tail bits stay zero; source identities/order still live in problem_.
  std::unique_ptr<GpuMat<uint8_t>> consistency_mask_;
  bool consistency_mask_valid_ = false;

  // Shared memory is too small to hold local state for each thread,
  // so this is workspace memory in global memory.
  std::unique_ptr<GpuMat<float>> global_workspace_;

#ifndef COLMAP_HIP_ENABLED
  // Per-problem storage for the opposite reference-image orientation.
  std::unique_ptr<GpuMat<float>> alternate_depth_map_;
  std::unique_ptr<GpuMat<float>> alternate_normal_map_;
  std::unique_ptr<GpuMat<uint8_t>> alternate_ref_image_;
  std::unique_ptr<GpuMat<float>> alternate_ref_sum_image_;
  std::unique_ptr<GpuMat<float>> alternate_ref_squared_sum_image_;
  std::unique_ptr<GpuMat<uint8_t>> alternate_consistency_mask_;
  std::unique_ptr<CudaArrayLayeredTexture<uint8_t>>
      alternate_ref_image_texture_;
  // Non-owning array caches; the two owning textures outlive every use. Swap
  // each cache with its matching owner. Never cudaFreeArray these handles.
  cudaArray_t ref_image_array_ = nullptr;
  cudaArray_t alternate_ref_image_array_ = nullptr;
#endif
};

#ifndef COLMAP_HIP_ENABLED
// Query the native default Sweep's device residency for both orientations.
bool CanPairPatchMatchViews(size_t first_width,
                            size_t first_height,
                            size_t second_width,
                            size_t second_height,
                            bool geometric);
// Explicit checkpoint mode uses its own compiled kernel resource query.
bool CanPairPatchMatchViews(size_t first_width,
                            size_t first_height,
                            size_t second_width,
                            size_t second_height,
                            bool geometric,
                            bool checkpoint_hmm);

// Both instances own prepared, independent real inputs on one CUDA device.
void RunPatchMatchPair(PatchMatchCuda& first,
                       PatchMatchCuda& second,
                       cudaStream_t first_stream,
                       cudaStream_t second_stream);
// Same native kernels and per-view state; controller admits groups of 3 to 5.
void RunPatchMatchBatch(const std::vector<PatchMatchCuda*>& views,
                        const std::vector<cudaStream_t>& streams);
// The resource query is device-specific and preserves the pair admission.
bool CanBatchPatchMatchViews(
    const std::vector<std::pair<size_t, size_t>>& view_shapes, bool geometric);
bool CanBatchPatchMatchViews(
    const std::vector<std::pair<size_t, size_t>>& view_shapes,
    bool geometric,
    bool checkpoint_hmm);
#endif

}  // namespace mvs
}  // namespace colmap
