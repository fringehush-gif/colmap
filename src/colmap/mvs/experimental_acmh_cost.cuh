// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Included in patch_match_cuda.cu after SweepOptions.
// The step-2 experimental backend computes bilateral weights inside NCC.
namespace experimental_acmh {

template <int kWindowSize, int kWindowStep, int kBlockWidth = THREADS_PER_BLOCK>
struct CompactPhotoConsistencyCostComputer {
  const static int kWindowRadius = kWindowSize / 2;
  static constexpr int kNumTapsPerAxis = (kWindowSize - 1) / kWindowStep + 1;
  static constexpr int kNumTaps = kNumTapsPerAxis * kNumTapsPerAxis;
  static constexpr int kMaxCachedNumTaps = 121;
  // At 32 threads, the 121-tap cache uses 15,488 shared bytes per block.
  static constexpr bool kCacheWeights =
      kWindowStep == 1 && kNumTaps <= kMaxCachedNumTaps;
  static constexpr int kWeightCacheSize =
      kCacheWeights ? kNumTaps * kBlockWidth : 1;
  float* local_weights = nullptr;

  __device__ CompactPhotoConsistencyCostComputer(
      const RefCamera& camera,
      const cudaTextureObject_t ref_image_texture,
      const cudaTextureObject_t src_images_texture,
      const cudaTextureObject_t poses_texture,
      const float sigma_spatial,
      const float sigma_color)
      : local_ref_image(ref_image_texture),
        camera_(camera),
        src_images_texture_(src_images_texture),
        poses_texture_(poses_texture),
        bilateral_weight_computer_(sigma_spatial, sigma_color) {}

  // Maximum photo consistency cost as 1 - min(NCC).
  const float kMaxCost = 2.0f;

  // Every lane owns a distinct compact checkerboard pixel. Explicit indices
  // decouple native NCC arithmetic from the native contiguous-lane layout.
  static constexpr int kTileWidth = 32 + 2 * kWindowRadius;
  static constexpr int kTileRows = 2 + 2 * kWindowRadius;
  struct TileView {
    float* data = nullptr;
    __device__ explicit TileView(cudaTextureObject_t) {}
  };
  TileView local_ref_image;
  int ref_patch_start = 0;
  int ref_center_index = 0;

  // Precomputed sum of raw and squared image intensities.
  float local_ref_sum = 0.0f;
  float local_ref_squared_sum = 0.0f;

  // Index of source image.
  int src_image_idx = -1;

  // Center position of patch in reference image.
  int row = -1;
  int col = -1;

  // Depth and normal for which to warp patch.
  float depth = 0.0f;
  const float* normal = nullptr;

  __device__ inline void ReadWeights(const int first_row = 0,
                                     const int row_stride = 1) {
    if constexpr (kCacheWeights) {
      const int thread_id = threadIdx.x;
      int ref_image_idx = ref_patch_start;
      const float ref_center_color = local_ref_image.data[ref_center_index];
      ref_image_idx += first_row * kWindowStep * kTileWidth;
      int tap = first_row * kNumTapsPerAxis;
      for (int dy = -kWindowRadius + first_row * kWindowStep;
           dy <= kWindowRadius;
           dy += row_stride * kWindowStep) {
        int idx = ref_image_idx;
        for (int dx = -kWindowRadius; dx <= kWindowRadius; dx += kWindowStep) {
          local_weights[tap * kBlockWidth + thread_id] =
              bilateral_weight_computer_.Compute(
                  dy, dx, ref_center_color, local_ref_image.data[idx]);
          ++tap;
          idx += kWindowStep;
        }
        tap += (row_stride - 1) * kNumTapsPerAxis;
        ref_image_idx += row_stride * kWindowStep * kTileWidth;
      }
    }
  }

  __device__ inline float Compute() const {
    float tform[9];
    ComposeHomography(
        camera_, poses_texture_, src_image_idx, row, col, depth, normal, tform);

    float tform_step[8];
    for (int i = 0; i < 8; ++i) {
      tform_step[i] = kWindowStep * tform[i];
    }

    const int thread_id = threadIdx.x;
    const int row_start = row - kWindowRadius;
    const int col_start = col - kWindowRadius;

    float col_src = tform[0] * col_start + tform[1] * row_start + tform[2];
    float row_src = tform[3] * col_start + tform[4] * row_start + tform[5];
    float z = tform[6] * col_start + tform[7] * row_start + tform[8];
    float base_col_src = col_src;
    float base_row_src = row_src;
    float base_z = z;

    int ref_image_idx = ref_patch_start;
    int ref_image_base_idx = ref_image_idx;

    const float ref_center_color = local_ref_image.data[ref_center_index];
    const float ref_color_sum = local_ref_sum;
    const float ref_color_squared_sum = local_ref_squared_sum;
    float src_color_sum = 0.0f;
    float src_color_squared_sum = 0.0f;
    float src_ref_color_sum = 0.0f;
    float bilateral_weight_sum = 0.0f;
    int weight_tap = 0;

    for (int row = -kWindowRadius; row <= kWindowRadius; row += kWindowStep) {
      for (int col = -kWindowRadius; col <= kWindowRadius; col += kWindowStep) {
        const float inv_z = 1.0f / z;
        const float norm_col_src = inv_z * col_src + 0.5f;
        const float norm_row_src = inv_z * row_src + 0.5f;
        const float ref_color = local_ref_image.data[ref_image_idx];
#if defined(__GFX9__)
        const float src_color = SampleLayeredBilinear(
            src_images_texture_, norm_col_src, norm_row_src, src_image_idx);
#else
        const float src_color = tex2DLayered<float>(
            src_images_texture_, norm_col_src, norm_row_src, src_image_idx);
#endif

        float bilateral_weight;
        if constexpr (kCacheWeights) {
          bilateral_weight =
              local_weights[weight_tap * kBlockWidth + thread_id];
          ++weight_tap;
        } else {
          bilateral_weight = bilateral_weight_computer_.Compute(
              row, col, ref_center_color, ref_color);
        }

        const float bilateral_weight_src = bilateral_weight * src_color;

        src_color_sum += bilateral_weight_src;
        src_color_squared_sum += bilateral_weight_src * src_color;
        src_ref_color_sum += bilateral_weight_src * ref_color;
        bilateral_weight_sum += bilateral_weight;

        ref_image_idx += kWindowStep;

        // Accumulate warped source coordinates per row to reduce numerical
        // errors. Note that this is necessary since coordinates usually are in
        // the order of 1000s as opposed to the color values which are
        // normalized to the range [0, 1].
        col_src += tform_step[0];
        row_src += tform_step[3];
        z += tform_step[6];
      }

      ref_image_base_idx += kWindowStep * kTileWidth;
      ref_image_idx = ref_image_base_idx;

      base_col_src += tform_step[1];
      base_row_src += tform_step[4];
      base_z += tform_step[7];

      col_src = base_col_src;
      row_src = base_row_src;
      z = base_z;
    }

    const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
    src_color_sum *= inv_bilateral_weight_sum;
    src_color_squared_sum *= inv_bilateral_weight_sum;
    src_ref_color_sum *= inv_bilateral_weight_sum;

    const float ref_color_var =
        ref_color_squared_sum - ref_color_sum * ref_color_sum;
    const float src_color_var =
        src_color_squared_sum - src_color_sum * src_color_sum;

    // Based on Jensen's Inequality for convex functions, the variance
    // should always be larger than 0. Do not make this threshold smaller.
    constexpr float kMinVar = 1e-5f;
    if (ref_color_var < kMinVar || src_color_var < kMinVar) {
      return kMaxCost;
    } else {
      const float src_ref_color_covar =
          src_ref_color_sum - ref_color_sum * src_color_sum;
      const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
      return max(0.0f,
                 min(kMaxCost, 1.0f - src_ref_color_covar / src_ref_color_var));
    }
  }

 private:
  const RefCamera& camera_;
  const cudaTextureObject_t src_images_texture_;
  const cudaTextureObject_t poses_texture_;
  const BilateralWeightComputer bilateral_weight_computer_;
};

}  // namespace experimental_acmh
