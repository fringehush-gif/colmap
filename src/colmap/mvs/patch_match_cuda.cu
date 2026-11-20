// SPDX-License-Identifier: BSD-3-Clause

// NOTE: On Blackwell GPUs (CUDA compute >= 100), nvcc may miscompile the
// SweepFromTopToBottom kernel at -O2/-O3 optimization levels, producing
// all-zero normal maps and corrupted depth maps. The CMake build system
// excludes sm_100+ architectures for this target and instead emits sm_90
// PTX, which Blackwell GPUs JIT-compile at first launch.
// See: https://github.com/colmap/colmap/issues/3514

#define _USE_MATH_DEFINES

#include "colmap/mvs/packed_consistency_mask.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/patch_match_cuda_workspace.h"
#include "colmap/mvs/patch_match_geometry.cuh"
#include "colmap/mvs/patch_match_rng_perimeter.h"
#include "colmap/mvs/patch_match_sampling.cuh"
#ifdef COLMAP_PATCH_MATCH_TESTING
#include "colmap/mvs/patch_match_cuda_test.h"
#endif
#include "colmap/util/cuda.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>

// The number of threads per Cuda thread block. Warning: Do not change this
// value, since the templated window sizes rely on this value.
#define THREADS_PER_BLOCK 32

// We must not include "util/math.h" to avoid any Eigen includes here,
// since Visual Studio cannot compile some of the Eigen/Boost expressions.
#ifndef DEG2RAD
#define DEG2RAD(deg) deg * 0.0174532925199432
#endif

namespace colmap {
namespace mvs {

namespace internal {
// A launch-local reader. The texture owns unchanged original-orientation bits;
// fallback planes already have the current orientation. Rows passed here are
// global image rows, including reads performed by a band producer.
struct ReferenceStatisticsView {
  GpuMatView<float> sum;
  GpuMatView<float> squared_sum;
  cudaTextureObject_t texture;
  PatchMatchRngPerimeter original;
  unsigned int rotation;

  __device__ float2 Get(const size_t row, const size_t col) const {
#ifndef COLMAP_HIP_ENABLED
    if (texture != 0) {
      size_t original_row, original_col;
      original.OriginalCoordinates(
          rotation, row, col, &original_row, &original_col);
      const uint2 bits = tex2DLayered<uint2>(
          texture, original_col + 0.5f, original_row + 0.5f, 0);
      return make_float2(__uint_as_float(bits.x), __uint_as_float(bits.y));
    }
#endif
    return make_float2(sum.Get(row, col), squared_sum.Get(row, col));
  }
};
}  // namespace internal

namespace {

// An immutable value copied into each kernel launch's parameter state.
struct RefCamera {
  float K[4];      // fx, cx, fy, cy
  float inv_K[4];  // 1/fx, -cx/fx, 1/fy, -cy/fy
};
static_assert(sizeof(RefCamera) == 8 * sizeof(float));

RefCamera MakeRefCamera(const float K[4], const float inv_K[4]) {
  RefCamera camera;
  std::copy_n(K, 4, camera.K);
  std::copy_n(inv_K, 4, camera.inv_K);
  return camera;
}

}  // namespace

__device__ inline void Mat33DotVec3(const float mat[9],
                                    const float vec[3],
                                    float result[3]) {
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

__device__ inline void Mat33DotVec3Homogeneous(const float mat[9],
                                               const float vec[2],
                                               float result[2]) {
  const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
  result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
  result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
}

__device__ inline float DotProduct3(const float vec1[3], const float vec2[3]) {
  return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

__device__ inline float GenerateRandomDepth(const float depth_min,
                                            const float depth_max,
                                            curandState* rand_state) {
  return internal::DrawUniformInterval(depth_min, depth_max, rand_state);
}

__device__ inline void GenerateRandomNormal(const RefCamera& camera,
                                            const int row,
                                            const int col,
                                            curandState* rand_state,
                                            float normal[3]) {
  // Unbiased sampling of normal, according to George Marsaglia, "Choosing a
  // Point from the Surface of a Sphere", 1972.
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  while (s >= 1.0f) {
    v1 = 2.0f * curand_uniform(rand_state) - 1.0f;
    v2 = 2.0f * curand_uniform(rand_state) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }

  const float s_norm = sqrt(1.0f - s);
  normal[0] = 2.0f * v1 * s_norm;
  normal[1] = 2.0f * v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;

  // Make sure normal is looking away from camera.
  const float view_ray[3] = {camera.inv_K[0] * col + camera.inv_K[1],
                             camera.inv_K[2] * row + camera.inv_K[3],
                             1.0f};
  if (DotProduct3(normal, view_ray) > 0) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
}

__device__ inline float PerturbDepth(const float perturbation,
                                     const float depth,
                                     curandState* rand_state) {
  const float depth_min = (1.0f - perturbation) * depth;
  const float depth_max = (1.0f + perturbation) * depth;
  return GenerateRandomDepth(depth_min, depth_max, rand_state);
}

__device__ inline void PerturbNormal(const RefCamera& camera,
                                     const int row,
                                     const int col,
                                     const float perturbation,
                                     const float normal[3],
                                     curandState* rand_state,
                                     float perturbed_normal[3],
                                     const int num_trials = 0) {
  // Perturbation rotation angles.
  const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
  const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
  const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

  const float sin_a1 = sin(a1);
  const float sin_a2 = sin(a2);
  const float sin_a3 = sin(a3);
  const float cos_a1 = cos(a1);
  const float cos_a2 = cos(a2);
  const float cos_a3 = cos(a3);

  // R = Rx * Ry * Rz
  float R[9];
  R[0] = cos_a2 * cos_a3;
  R[1] = -cos_a2 * sin_a3;
  R[2] = sin_a2;
  R[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
  R[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
  R[5] = -cos_a2 * sin_a1;
  R[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
  R[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
  R[8] = cos_a1 * cos_a2;

  // Perturb the normal vector.
  Mat33DotVec3(R, normal, perturbed_normal);

  // Make sure the perturbed normal is still looking in the same direction as
  // the viewing direction, otherwise try again but with smaller perturbation.
  const float view_ray[3] = {camera.inv_K[0] * col + camera.inv_K[1],
                             camera.inv_K[2] * row + camera.inv_K[3],
                             1.0f};
  if (DotProduct3(perturbed_normal, view_ray) >= 0.0f) {
    const int kMaxNumTrials = 3;
    if (num_trials < kMaxNumTrials) {
      PerturbNormal(camera,
                    row,
                    col,
                    0.5f * perturbation,
                    normal,
                    rand_state,
                    perturbed_normal,
                    num_trials + 1);
      return;
    } else {
      perturbed_normal[0] = normal[0];
      perturbed_normal[1] = normal[1];
      perturbed_normal[2] = normal[2];
      return;
    }
  }

  // Make sure normal has unit norm.
  const float inv_norm = rsqrt(DotProduct3(perturbed_normal, perturbed_normal));
  perturbed_normal[0] *= inv_norm;
  perturbed_normal[1] *= inv_norm;
  perturbed_normal[2] *= inv_norm;
}

__device__ inline void ComputePointAtDepth(const RefCamera& camera,
                                           const float row,
                                           const float col,
                                           const float depth,
                                           float point[3]) {
  point[0] = depth * (camera.inv_K[0] * col + camera.inv_K[1]);
  point[1] = depth * (camera.inv_K[2] * row + camera.inv_K[3]);
  point[2] = depth;
}

// First, compute triangulation angle between reference and source image for 3D
// point. Second, compute incident angle between viewing direction of source
// image and normal direction of 3D point. Both angles are cosine distances.
__device__ inline void ComputeViewingAngles(
    const cudaTextureObject_t poses_texture,
    const float point[3],
    const float normal[3],
    const int image_idx,
    float* cos_triangulation_angle,
    float* cos_incident_angle) {
  *cos_triangulation_angle = 0.0f;
  *cos_incident_angle = 0.0f;

  // Projection center of source image.
  float C[3];
  for (int i = 0; i < 3; ++i) {
    C[i] = tex2D<float>(poses_texture, i + 16, image_idx);
  }

  // Ray from point to camera.
  const float SX[3] = {C[0] - point[0], C[1] - point[1], C[2] - point[2]};

  // Length of ray from reference image to point.
  const float RX_inv_norm = rsqrt(DotProduct3(point, point));

  // Length of ray from point to source image.
  const float SX_inv_norm = rsqrt(DotProduct3(SX, SX));

  *cos_incident_angle = DotProduct3(SX, normal) * SX_inv_norm;
  *cos_triangulation_angle =
      -DotProduct3(SX, point) * RX_inv_norm * SX_inv_norm;
}

__device__ inline void ComposeHomography(
    const RefCamera& camera,
    const cudaTextureObject_t poses_texture,
    const int image_idx,
    const int row,
    const int col,
    const float depth,
    const float normal[3],
    float H[9]) {
  // Calibration of source image.
  float K[4];
  for (int i = 0; i < 4; ++i) {
    K[i] = tex2D<float>(poses_texture, i, image_idx);
  }

  // Relative rotation between reference and source image.
  float R[9];
  for (int i = 0; i < 9; ++i) {
    R[i] = tex2D<float>(poses_texture, i + 4, image_idx);
  }

  // Relative translation between reference and source image.
  float T[3];
  for (int i = 0; i < 3; ++i) {
    T[i] = tex2D<float>(poses_texture, i + 13, image_idx);
  }

  // Distance to the plane.
  const float dist =
      depth *
      (normal[0] * (camera.inv_K[0] * col + camera.inv_K[1]) +
       normal[1] * (camera.inv_K[2] * row + camera.inv_K[3]) + normal[2]);
  const float inv_dist = 1.0f / dist;

  const float inv_dist_N0 = inv_dist * normal[0];
  const float inv_dist_N1 = inv_dist * normal[1];
  const float inv_dist_N2 = inv_dist * normal[2];

  // Homography as H = K * (R - T * n' / d) * Kref^-1.
  H[0] = camera.inv_K[0] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
                            K[1] * (R[6] + inv_dist_N0 * T[2]));
  H[1] = camera.inv_K[2] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
                            K[1] * (R[7] + inv_dist_N1 * T[2]));
  H[2] = K[0] * (R[2] + inv_dist_N2 * T[0]) +
         K[1] * (R[8] + inv_dist_N2 * T[2]) +
         camera.inv_K[1] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
                            K[1] * (R[6] + inv_dist_N0 * T[2])) +
         camera.inv_K[3] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
                            K[1] * (R[7] + inv_dist_N1 * T[2]));
  H[3] = camera.inv_K[0] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
                            K[3] * (R[6] + inv_dist_N0 * T[2]));
  H[4] = camera.inv_K[2] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
                            K[3] * (R[7] + inv_dist_N1 * T[2]));
  H[5] = K[2] * (R[5] + inv_dist_N2 * T[1]) +
         K[3] * (R[8] + inv_dist_N2 * T[2]) +
         camera.inv_K[1] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
                            K[3] * (R[6] + inv_dist_N0 * T[2])) +
         camera.inv_K[3] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
                            K[3] * (R[7] + inv_dist_N1 * T[2]));
  H[6] = camera.inv_K[0] * (R[6] + inv_dist_N0 * T[2]);
  H[7] = camera.inv_K[2] * (R[7] + inv_dist_N1 * T[2]);
  H[8] = R[8] + camera.inv_K[1] * (R[6] + inv_dist_N0 * T[2]) +
         camera.inv_K[3] * (R[7] + inv_dist_N1 * T[2]) + inv_dist_N2 * T[2];
}

// Each thread in the current warp / thread block reads in 3 columns of the
// reference image. The shared memory holds 3 * kBlockWidth columns and
// kWindowSize rows of the reference image. Each thread copies every
// kBlockWidth-th column from global to shared memory offset by its ID.
// For example, if kBlockWidth = 32, then thread 0 reads columns 0, 32, 64
// and thread 1 columns 1, 33, 65. When computing the photoconsistency, which is
// shared among each thread block, each thread can then read the reference image
// colors from shared memory. Note that this limits the window radius to a
// maximum of kBlockWidth.
template <int kWindowSize, int kBlockWidth = THREADS_PER_BLOCK>
struct LocalRefImage {
  const static int kWindowRadius = kWindowSize / 2;
  static_assert(kWindowRadius <= kBlockWidth);
  const static int kThreadBlockRadius = 1;
  const static int kThreadBlockSize = 2 * kThreadBlockRadius + 1;
  const static int kNumRows = kWindowSize;
  const static int kNumColumns = kThreadBlockSize * kBlockWidth;
  const static int kDataSize = kNumRows * kNumColumns;

  __device__ explicit LocalRefImage(const cudaTextureObject_t ref_image_texture)
      : ref_image_texture_(ref_image_texture) {}

  float* data = nullptr;

  __device__ inline void Read(const int row, const bool stripe_start = false) {
    // For the first row, read the entire block into shared memory. For all
    // consecutive rows, it is only necessary to shift the rows in shared memory
    // up by one element and then read in a new row at the bottom of the shared
    // memory. Note that this assumes that the calling loop starts with the
    // first row and then consecutively reads in the next row.

    const int thread_id = threadIdx.x;
    const int thread_block_first_id = blockDim.x * blockIdx.x;

    const int local_col_start = thread_id;
    const int global_col_start =
        thread_block_first_id - kThreadBlockRadius * kBlockWidth + thread_id;

    if (row == 0 || stripe_start) {
      int global_row = row - kWindowRadius;
      for (int local_row = 0; local_row < kNumRows; ++local_row, ++global_row) {
        int local_col = local_col_start;
        int global_col = global_col_start;
#pragma unroll
        for (int block = 0; block < kThreadBlockSize; ++block) {
          data[local_row * kNumColumns + local_col] =
              tex2D<float>(ref_image_texture_, global_col, global_row);
          local_col += kBlockWidth;
          global_col += kBlockWidth;
        }
      }
    } else {
      // Move rows in shared memory up by one row.
      for (int local_row = 1; local_row < kNumRows; ++local_row) {
        int local_col = local_col_start;
#pragma unroll
        for (int block = 0; block < kThreadBlockSize; ++block) {
          data[(local_row - 1) * kNumColumns + local_col] =
              data[local_row * kNumColumns + local_col];
          local_col += kBlockWidth;
        }
      }

      // Read next row into the last row of shared memory.
      const int local_row = kNumRows - 1;
      const int global_row = row + kWindowRadius;
      int local_col = local_col_start;
      int global_col = global_col_start;
#pragma unroll
      for (int block = 0; block < kThreadBlockSize; ++block) {
        data[local_row * kNumColumns + local_col] =
            tex2D<float>(ref_image_texture_, global_col, global_row);
        local_col += kBlockWidth;
        global_col += kBlockWidth;
      }
    }
  }

 private:
  const cudaTextureObject_t ref_image_texture_;
};

#if defined(__GFX9__)
// AMD CDNA/gfx9 hardware cannot create a linear-filtered layered texture
// (cudaCreateTextureObject returns hipErrorNotSupported), so on gfx9 device
// code the source image texture is bound with cudaFilterModePoint (see the
// runtime branch in InitSourceImages) and we emulate the hardware bilinear
// fetch with four point samples. Non-normalized coordinates place texel
// centers at integer+0.5; cudaAddressModeBorder returns 0 outside the image,
// so out-of-range point samples reproduce the border blend. __GFX9__ is a
// per-architecture device macro, so a fat binary still uses the hardware path
// for its non-gfx9 (e.g. RDNA) device code.
__device__ inline float SampleLayeredBilinear(const cudaTextureObject_t texture,
                                              const float x,
                                              const float y,
                                              const int layer) {
  const float px = x - 0.5f;
  const float py = y - 0.5f;
  const float fx = floorf(px);
  const float fy = floorf(py);
  const float wx = px - fx;
  const float wy = py - fy;
  const float c00 = tex2DLayered<float>(texture, fx + 0.5f, fy + 0.5f, layer);
  const float c10 = tex2DLayered<float>(texture, fx + 1.5f, fy + 0.5f, layer);
  const float c01 = tex2DLayered<float>(texture, fx + 0.5f, fy + 1.5f, layer);
  const float c11 = tex2DLayered<float>(texture, fx + 1.5f, fy + 1.5f, layer);
  return (c00 * (1.0f - wx) + c10 * wx) * (1.0f - wy) +
         (c01 * (1.0f - wx) + c11 * wx) * wy;
}
#endif

// The return values is 1 - NCC, so the range is [0, 2], the smaller the
// value, the better the color consistency.
template <int kWindowSize,
          int kWindowStep,
          int kBlockWidth = THREADS_PER_BLOCK,
          bool kReuseNormalization = false>
struct PhotoConsistencyCostComputer {
  const static int kWindowRadius = kWindowSize / 2;
  static constexpr int kNumTapsPerAxis = (kWindowSize - 1) / kWindowStep + 1;
  static constexpr int kNumTaps = kNumTapsPerAxis * kNumTapsPerAxis;
  // Restrict cached weights to the retained default specialization. Other
  // windows preserve native inline FP32 evaluation rather than widen caching.
  static constexpr bool kCacheWeights = kWindowSize == 11 && kWindowStep == 1;
  static constexpr bool kPrepareNormalization =
      kCacheWeights && kReuseNormalization;
  static constexpr int kWeightCacheSize =
      kCacheWeights ? kNumTaps * kBlockWidth : 1;
  float* local_weights = nullptr;

  __device__ PhotoConsistencyCostComputer(
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

  // Thread warp local reference image data around current patch.
  typedef LocalRefImage<kWindowSize, kBlockWidth> LocalRefImageType;
  LocalRefImageType local_ref_image;

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

  __device__ inline void ReadRef(const int row,
                                 const bool stripe_start = false) {
    if constexpr (kPrepareNormalization) normalization_is_prepared_ = false;
    local_ref_image.Read(row, stripe_start);
  }

  __device__ inline void ReadWeights(const int first_row = 0,
                                     const int row_stride = 1) {
    if constexpr (kCacheWeights) {
      // Every worker must invalidate, including workers whose assigned weight
      // row is beyond the patch. Such workers still evaluate source jobs.
      if constexpr (kPrepareNormalization) normalization_is_prepared_ = false;
      const int thread_id = threadIdx.x;
      int ref_image_idx = kBlockWidth - kWindowRadius + thread_id;
      const float ref_center_color =
          local_ref_image.data[ref_image_idx + kWindowRadius * 3 * kBlockWidth +
                               kWindowRadius];
      ref_image_idx += first_row * kWindowStep * 3 * kBlockWidth;
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
        ref_image_idx += row_stride * kWindowStep * 3 * kBlockWidth;
      }
    }
  }

  __device__ inline void Read(const int row, const bool stripe_start = false) {
    ReadRef(row, stripe_start);
    __syncthreads();
    ReadWeights();
  }

  __device__ inline float Compute() const {
    if constexpr (kPrepareNormalization) {
      if (normalization_is_prepared_) return ComputeImpl<true>();
    }
    return ComputeImpl<false>();
  }

#ifdef COLMAP_PATCH_MATCH_TESTING
  // Literal PR59 arithmetic, including late reference-variance rejection and
  // the per-call weight fold. Keep the native oracle independent of
  // ComputeImpl.
  __device__ inline float ComputeReferenceForTesting() const {
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

    int ref_image_idx = kBlockWidth - kWindowRadius + thread_id;
    int ref_image_base_idx = ref_image_idx;

    const float ref_center_color =
        local_ref_image.data[ref_image_idx + kWindowRadius * 3 * kBlockWidth +
                             kWindowRadius];
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

      ref_image_base_idx += kWindowStep * 3 * kBlockWidth;
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
#endif

 private:
  template <bool kPreparedNormalization>
  __device__ inline float ComputeImpl() const {
    static_assert(!kPreparedNormalization || kPrepareNormalization);
    const float ref_color_sum = local_ref_sum;
    const float ref_color_squared_sum = local_ref_squared_sum;
    const float ref_color_var =
        ref_color_squared_sum - ref_color_sum * ref_color_sum;
    // Keep the original expression and comparison. In particular, NaN does
    // not satisfy this rejection and follows the remaining arithmetic.
    constexpr float kMinVar = 1e-5f;
    if (ref_color_var < kMinVar) {
      if constexpr (kPrepareNormalization) normalization_is_prepared_ = false;
      return kMaxCost;
    }

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

    int ref_image_idx = kBlockWidth - kWindowRadius + thread_id;
    int ref_image_base_idx = ref_image_idx;

    const float ref_center_color =
        local_ref_image.data[ref_image_idx + kWindowRadius * 3 * kBlockWidth +
                             kWindowRadius];
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
        if constexpr (!kPreparedNormalization) {
          bilateral_weight_sum += bilateral_weight;
        }

        ref_image_idx += kWindowStep;

        // Accumulate warped source coordinates per row to reduce numerical
        // errors. Note that this is necessary since coordinates usually are in
        // the order of 1000s as opposed to the color values which are
        // normalized to the range [0, 1].
        col_src += tform_step[0];
        row_src += tform_step[3];
        z += tform_step[6];
      }

      ref_image_base_idx += kWindowStep * 3 * kBlockWidth;
      ref_image_idx = ref_image_base_idx;

      base_col_src += tform_step[1];
      base_row_src += tform_step[4];
      base_z += tform_step[7];

      col_src = base_col_src;
      row_src = base_row_src;
      z = base_z;
    }

    float inv_bilateral_weight_sum;
    if constexpr (kPreparedNormalization) {
      inv_bilateral_weight_sum = prepared_inverse_weight_sum_;
    } else {
      // The first actual NCC evaluation retains the original ordered fold.
      // Reuse its reciprocal only while this thread owns the same weight row.
      inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
      if constexpr (kPrepareNormalization) {
        prepared_inverse_weight_sum_ = inv_bilateral_weight_sum;
        normalization_is_prepared_ = true;
      }
    }
    src_color_sum *= inv_bilateral_weight_sum;
    src_color_squared_sum *= inv_bilateral_weight_sum;
    src_ref_color_sum *= inv_bilateral_weight_sum;

    const float src_color_var =
        src_color_squared_sum - src_color_sum * src_color_sum;

    // The reference variance was checked before any warped source tap.
    // Do not make the retained variance threshold smaller.
    if (src_color_var < kMinVar) {
      return kMaxCost;
    } else {
      const float src_ref_color_covar =
          src_ref_color_sum - ref_color_sum * src_color_sum;
      const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
      return max(0.0f,
                 min(kMaxCost, 1.0f - src_ref_color_covar / src_ref_color_var));
    }
  }

  // Only phases with repeated source jobs opt in to this cache. It belongs to
  // one thread's current reference weights, independent of source/hypothesis.
  // ReadRef and every ReadWeights call invalidate it, including workers that
  // load no weight taps. It never spans rows, stripes or rotations.
  mutable float prepared_inverse_weight_sum_ = 0.0f;
  mutable bool normalization_is_prepared_ = false;

  const RefCamera& camera_;
  const cudaTextureObject_t src_images_texture_;
  const cudaTextureObject_t poses_texture_;
  const BilateralWeightComputer bilateral_weight_computer_;
};

__device__ inline float ComputeGeomConsistencyCost(
    const RefCamera& camera,
    const cudaTextureObject_t poses_texture,
    const cudaTextureObject_t src_depth_maps_texture,
    const float row,
    const float col,
    const float depth,
    const int image_idx,
    const float max_cost) {
  // Extract projection matrices for source image.
  float P[12];
  for (int i = 0; i < 12; ++i) {
    P[i] = tex2D<float>(poses_texture, i + 19, image_idx);
  }
  float inv_P[12];
  for (int i = 0; i < 12; ++i) {
    inv_P[i] = tex2D<float>(poses_texture, i + 31, image_idx);
  }

  // Project point in reference image to world.
  float forward_point[3];
  ComputePointAtDepth(camera, row, col, depth, forward_point);

  // Project world point to source image.
  const float inv_forward_z =
      1.0f / (P[8] * forward_point[0] + P[9] * forward_point[1] +
              P[10] * forward_point[2] + P[11]);
  float src_col =
      inv_forward_z * (P[0] * forward_point[0] + P[1] * forward_point[1] +
                       P[2] * forward_point[2] + P[3]);
  float src_row =
      inv_forward_z * (P[4] * forward_point[0] + P[5] * forward_point[1] +
                       P[6] * forward_point[2] + P[7]);

  // Extract depth in source image.
  const float src_depth = tex2DLayered<float>(
      src_depth_maps_texture, src_col + 0.5f, src_row + 0.5f, image_idx);

  // Projection outside of source image.
  if (src_depth == 0.0f) {
    return max_cost;
  }

  // Project point in source image to world.
  src_col *= src_depth;
  src_row *= src_depth;
  const float backward_point_x =
      inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
  const float backward_point_y =
      inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
  const float backward_point_z = inv_P[8] * src_col + inv_P[9] * src_row +
                                 inv_P[10] * src_depth + inv_P[11];
  const float inv_backward_point_z = 1.0f / backward_point_z;

  // Project world point back to reference image.
  const float backward_col =
      inv_backward_point_z *
      (camera.K[0] * backward_point_x + camera.K[1] * backward_point_z);
  const float backward_row =
      inv_backward_point_z *
      (camera.K[2] * backward_point_y + camera.K[3] * backward_point_z);

  // Return truncated reprojection error between original observation and
  // the forward-backward projected observation.
  const float diff_col = col - backward_col;
  const float diff_row = row - backward_row;
  return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

// Find index of minimum in given values.
template <int kNumCosts>
__device__ inline int FindMinCost(const float costs[kNumCosts]) {
  float min_cost = costs[0];
  int min_cost_idx = 0;
  for (int idx = 1; idx < kNumCosts; ++idx) {
    if (costs[idx] <= min_cost) {
      min_cost = costs[idx];
      min_cost_idx = idx;
    }
  }
  return min_cost_idx;
}

class LikelihoodComputer {
 public:
  __device__ LikelihoodComputer(const float ncc_sigma,
                                const float min_triangulation_angle,
                                const float incident_angle_sigma)
      : cos_min_triangulation_angle_(cos(min_triangulation_angle)),
        inv_incident_angle_sigma_square_(
            -0.5f / (incident_angle_sigma * incident_angle_sigma)),
        inv_ncc_sigma_square_(-0.5f / (ncc_sigma * ncc_sigma)),
        ncc_norm_factor_(ComputeNCCCostNormFactor(ncc_sigma)) {}

  // Compute forward message from current cost and forward message of
  // previous / neighboring pixel.
  __device__ float ComputeForwardMessage(const float cost,
                                         const float prev) const {
    return ComputeMessage<true>(cost, prev);
  }

  // Compute backward message from current cost and backward message of
  // previous / neighboring pixel.
  __device__ float ComputeBackwardMessage(const float cost,
                                          const float prev) const {
    return ComputeMessage<false>(cost, prev);
  }

  // Compute the selection probability from the forward and backward message.
  __device__ inline float ComputeSelProb(const float alpha,
                                         const float beta,
                                         const float prev,
                                         const float prev_weight) const {
    const float zn0 = (1.0f - alpha) * (1.0f - beta);
    const float zn1 = alpha * beta;
    const float curr = zn1 / (zn0 + zn1);
    return prev_weight * prev + (1.0f - prev_weight) * curr;
  }

  // Compute NCC probability. Note that cost = 1 - NCC.
  __device__ inline float ComputeNCCProb(const float cost) const {
    return exp(cost * cost * inv_ncc_sigma_square_) * ncc_norm_factor_;
  }

  // Compute the triangulation angle probability.
  __device__ inline float ComputeTriProb(
      const float cos_triangulation_angle) const {
    if (cos_triangulation_angle > cos_min_triangulation_angle_) {
      const float scaled = 1.0f - (1.0f - cos_triangulation_angle) /
                                      (1.0f - cos_min_triangulation_angle_);
      const float likelihood = 1.0f - scaled * scaled;
      return min(1.0f, max(0.0f, likelihood));
    } else {
      return 1.0f;
    }
  }

  // Compute the incident angle probability.
  __device__ inline float ComputeIncProb(const float cos_incident_angle) const {
    const float x = 1.0f - max(0.0f, cos_incident_angle);
    return exp(x * x * inv_incident_angle_sigma_square_);
  }

  // Compute the warping/resolution prior probability.
  template <int kWindowSize>
  __device__ inline float ComputeResolutionProb(const float H[9],
                                                const float row,
                                                const float col) const {
    const int kWindowRadius = kWindowSize / 2;

    // Warp corners of patch in reference image to source image.
    float src1[2];
    const float ref1[2] = {col - kWindowRadius, row - kWindowRadius};
    Mat33DotVec3Homogeneous(H, ref1, src1);
    float src2[2];
    const float ref2[2] = {col - kWindowRadius, row + kWindowRadius};
    Mat33DotVec3Homogeneous(H, ref2, src2);
    float src3[2];
    const float ref3[2] = {col + kWindowRadius, row + kWindowRadius};
    Mat33DotVec3Homogeneous(H, ref3, src3);
    float src4[2];
    const float ref4[2] = {col + kWindowRadius, row - kWindowRadius};
    Mat33DotVec3Homogeneous(H, ref4, src4);

    // Compute area of patches in reference and source image.
    const float ref_area = kWindowSize * kWindowSize;
    const float src_area =
        abs(0.5f * (src1[0] * src2[1] - src2[0] * src1[1] - src1[0] * src4[1] +
                    src2[0] * src3[1] - src3[0] * src2[1] + src4[0] * src1[1] +
                    src3[0] * src4[1] - src4[0] * src3[1]));

    if (ref_area > src_area) {
      return src_area / ref_area;
    } else {
      return ref_area / src_area;
    }
  }

 private:
  // The normalization for the likelihood function, i.e. the normalization for
  // the prior on the matching cost.
  __device__ static inline float ComputeNCCCostNormFactor(
      const float ncc_sigma) {
    // A = sqrt(2pi)*sigma/2*erf(sqrt(2)/sigma)
    // erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t^2) dt
    return 2.0f / (sqrt(2.0f * M_PI) * ncc_sigma *
                   erff(2.0f / (ncc_sigma * 1.414213562f)));
  }

  // Compute the forward or backward message.
  template <bool kForward>
  __device__ inline float ComputeMessage(const float cost,
                                         const float prev) const {
    constexpr float kUniformProb = 0.5f;
    constexpr float kNoChangeProb = 0.99999f;
    const float kChangeProb = 1.0f - kNoChangeProb;
    const float emission = ComputeNCCProb(cost);

    float zn0;  // Message for selection probability = 0.
    float zn1;  // Message for selection probability = 1.
    if (kForward) {
      zn0 = (prev * kChangeProb + (1.0f - prev) * kNoChangeProb) * kUniformProb;
      zn1 = (prev * kNoChangeProb + (1.0f - prev) * kChangeProb) * emission;
    } else {
      zn0 = prev * emission * kChangeProb +
            (1.0f - prev) * kUniformProb * kNoChangeProb;
      zn1 = prev * emission * kNoChangeProb +
            (1.0f - prev) * kUniformProb * kChangeProb;
    }

    return zn1 / (zn0 + zn1);
  }

  const float cos_min_triangulation_angle_;
  const float inv_incident_angle_sigma_square_;
  const float inv_ncc_sigma_square_;
  const float ncc_norm_factor_;
};

// PHOTO consumes each original pixel's seed, depth and normal draws locally.
__global__ void InitDepthNormalMaps(const RefCamera camera,
                                    GpuMatView<float> depth_map,
                                    GpuMatView<float> normal_map,
                                    GpuMatView<curandState> perimeter,
                                    PatchMatchRngPerimeter layout,
                                    float depth_min,
                                    float depth_max) {
  const size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= layout.height || col >= layout.width) return;
  curandState state;
  curand_init(internal::GpuMatPRNGSeed(row, col, layout.width), 0, 0, &state);
  depth_map.Set(row, col, GenerateRandomDepth(depth_min, depth_max, &state));
  float normal[3];
  GenerateRandomNormal(
      camera, static_cast<int>(row), static_cast<int>(col), &state, normal);
  normal_map.SetSlice(row, col, normal);
  if (row == 0 || col == 0 || row + 1 == layout.height ||
      col + 1 == layout.width)
    perimeter.Set(0, layout.Index(row, col), state);
}

// GEO copies supplied maps and retains freshly seeded states without draws.
__global__ void InitPerimeter(GpuMatView<curandState> perimeter,
                              PatchMatchRngPerimeter layout) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= layout.Count()) return;
  size_t row, col;
  layout.Coordinates(index, &row, &col);
  curand_init(internal::GpuMatPRNGSeed(row, col, layout.width),
              0,
              0,
              &perimeter.GetRef(0, index));
}

// Rotate normals by 90deg around z-axis in counter-clockwise direction.
__global__ void RotateNormalMap(GpuMatView<float> normal_map) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < normal_map.GetWidth() && row < normal_map.GetHeight()) {
    float normal[3];
    normal_map.GetSlice(row, col, normal);
    float rotated_normal[3];
    rotated_normal[0] = normal[1];
    rotated_normal[1] = -normal[0];
    rotated_normal[2] = normal[2];
    normal_map.SetSlice(row, col, rotated_normal);
  }
}

#ifndef COLMAP_HIP_ENABLED
// Store float representations in integer channels: texture reads perform no
// floating-point normalization, filtering, or denormal conversion.
__global__ void PackReferenceStatistics(const GpuMatView<float> sum,
                                        const GpuMatView<float> squared_sum,
                                        GpuMatView<uint2> packed) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < packed.height && col < packed.width) {
    packed.Set(row,
               col,
               make_uint2(__float_as_uint(sum.Get(row, col)),
                          __float_as_uint(squared_sum.Get(row, col))));
  }
}

#endif

constexpr int kInitialCostStripeRows = 64;

template <int kWindowSize,
          int kWindowStep
#ifdef COLMAP_PATCH_MATCH_TESTING
          ,
          bool kReferenceArithmetic = false
#endif
          >
__global__ void ComputeInitialCost(
    const RefCamera camera,
    GpuMatView<float> cost_map,
    const GpuMatView<float> depth_map,
    const GpuMatView<float> normal_map,
    const cudaTextureObject_t ref_image_texture,
    const internal::ReferenceStatisticsView ref_statistics,
    const cudaTextureObject_t src_images_texture,
    const cudaTextureObject_t poses_texture,
    const float sigma_spatial,
    const float sigma_color) {
  const int col = blockDim.x * blockIdx.x + threadIdx.x;

  typedef PhotoConsistencyCostComputer<kWindowSize,
                                       kWindowStep,
                                       THREADS_PER_BLOCK,
                                       true>
      PhotoConsistencyCostComputerType;
  PhotoConsistencyCostComputerType pcc_computer(camera,
                                                ref_image_texture,
                                                src_images_texture,
                                                poses_texture,
                                                sigma_spatial,
                                                sigma_color);
  pcc_computer.col = col;

  __shared__ float local_ref_image_data
      [PhotoConsistencyCostComputerType::LocalRefImageType::kDataSize];
  pcc_computer.local_ref_image.data = &local_ref_image_data[0];
  __shared__ float
      local_weight_data[PhotoConsistencyCostComputerType::kWeightCacheSize];
  pcc_computer.local_weights = &local_weight_data[0];

  float normal[3] = {0};
  pcc_computer.normal = normal;

  const int first_row = blockIdx.y * kInitialCostStripeRows;
  const int last_row = min(first_row + kInitialCostStripeRows,
                           static_cast<int>(cost_map.GetHeight()));
  for (int row = first_row; row < last_row; ++row) {
    // Finish every column's previous-row reads before reusing the shared tile.
    __syncthreads();
    // Note that this must be executed even for pixels outside the borders,
    // since pixels are used in the local neighborhood of the current pixel.
    pcc_computer.Read(row, row == first_row);

    if (col < cost_map.GetWidth()) {
      pcc_computer.depth = depth_map.Get(row, col);
      normal_map.GetSlice(row, col, normal);

      pcc_computer.row = row;
      const float2 statistics = ref_statistics.Get(row, col);
      pcc_computer.local_ref_sum = statistics.x;
      pcc_computer.local_ref_squared_sum = statistics.y;

      for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
        pcc_computer.src_image_idx = image_idx;
#ifdef COLMAP_PATCH_MATCH_TESTING
        if constexpr (kReferenceArithmetic) {
          cost_map.Set(
              row, col, image_idx, pcc_computer.ComputeReferenceForTesting());
        } else
#endif
        {
          cost_map.Set(row, col, image_idx, pcc_computer.Compute());
        }
      }
    }
  }
}

struct SweepOptions {
  float perturbation = 1.0f;
  float depth_min = 0.0f;
  float depth_max = 1.0f;
  int num_samples = 15;
  float sigma_spatial = 3.0f;
  float sigma_color = 0.3f;
  float ncc_sigma = 0.6f;
  float min_triangulation_angle = 0.5f;
  float incident_angle_sigma = 0.9f;
  float prev_sel_prob_weight = 0.0f;
  float geom_consistency_regularizer = 0.1f;
  float geom_consistency_max_cost = 5.0f;
  float filter_min_ncc = 0.1f;
  float filter_min_triangulation_angle = 3.0f;
  int filter_min_num_consistent = 2;
  float filter_geom_consistency_max_cost = 1.0f;
};

// Bounded look-ahead state. All views use one instance-local
// allocation; first_row/num_rows describe the active band in global coordinates.
struct LookAheadViews {
  GpuMatView<float> random_parameters;
  // Producer writes raw uniforms. Sweep keeps its first sixteen selected IDs
  // in shared storage, then replaces each consumed tail uniform with an ID
  // (-1..31), exactly representable as float.
  GpuMatView<float> uniforms;
  GpuMatView<float> costs;
  GpuMatView<float> geometric_costs;
  GpuMatView<float> previous_parameters;
  int first_row = 0;
  int num_rows = 0;
};

class LookAheadWorkspace {
 public:
  explicit LookAheadWorkspace(const internal::PatchMatchLookAheadPlan& plan)
      : plan_(plan) {
    CHECK_GT(plan_.rows, 0);
    CUDA_SAFE_CALL(
        cudaMalloc(reinterpret_cast<void**>(&data_), plan_.num_bytes));
  }

  ~LookAheadWorkspace() { (void)cudaFree(data_); }
  LookAheadWorkspace(const LookAheadWorkspace&) = delete;
  LookAheadWorkspace& operator=(const LookAheadWorkspace&) = delete;

  int Rows() const { return plan_.rows; }

  LookAheadViews Views(const unsigned int rotation,
                       const int first_row,
                       const int num_rows) const {
    const auto& layout = plan_.layouts[rotation & 1];
    const size_t width = layout.width;
    const size_t rows = layout.plane_rows;
    const size_t sources = plan_.sources;
    const size_t pitch = width * sizeof(float);
    const GpuMatView<float> geometric_costs =
        plan_.geometry ? GpuMatView<float>{data_ + layout.geometry_offset,
                                           pitch,
                                           width,
                                           rows,
                                           2 * sources}
                       : GpuMatView<float>{};
    return {{data_, pitch, width, rows, 4},
            {data_ + layout.uniforms_offset,
             pitch,
             width,
             rows,
             static_cast<size_t>(plan_.samples)},
            {data_ + layout.costs_offset, pitch, width, rows, 3 * sources},
            geometric_costs,
            {data_ + layout.previous_offset, pitch, width, 1, 4},
            first_row,
            num_rows};
  }

 private:
  float* data_ = nullptr;
  const internal::PatchMatchLookAheadPlan plan_;
};

std::unique_ptr<LookAheadWorkspace> MakeLookAheadWorkspace(
    const size_t width,
    const size_t height,
    const size_t sources,
    const int samples,
    const bool geom_consistency_term) {
#ifdef COLMAP_HIP_ENABLED
  return nullptr;
#else
  const auto plan = internal::PatchMatchLookAheadPlan::Make(
      width, height, sources, samples, geom_consistency_term);
  if (plan.rows == 0) return nullptr;
  // Allocation errors follow CUDA_SAFE_CALL; do not fall back after mutation.
  return std::make_unique<LookAheadWorkspace>(plan);
#endif
}

__device__ inline void InitializeSourceMessages(
    LikelihoodComputer& likelihood_computer,
    const GpuMatView<float> cost_map,
    GpuMatView<float> sel_prob_map,
    float* forward_message,
    const int source_stride,
    const int col,
    const int image_idx) {
  float beta = 0.5f;
  for (int row = cost_map.GetHeight() - 1; row >= 0; --row) {
    const float cost = cost_map.Get(row, col, image_idx);
    // The current emission belongs to alpha; beta contains only the suffix.
    sel_prob_map.Set(row, col, image_idx, beta);
    beta = likelihood_computer.ComputeBackwardMessage(cost, beta);
  }
  forward_message[image_idx * source_stride] = 0.5f;
}

__global__ void InitializeSweepMessages(
    GpuMatView<float> global_workspace,
    const GpuMatView<float> cost_map,
    GpuMatView<float> sel_prob_map,
    const SweepOptions options) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int source = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= cost_map.GetWidth() || source >= cost_map.GetDepth()) return;
  LikelihoodComputer likelihood_computer(options.ncc_sigma,
                                         options.min_triangulation_angle,
                                         options.incident_angle_sigma);
  InitializeSourceMessages(likelihood_computer,
                           cost_map,
                           sel_prob_map,
                           global_workspace.GetPtr() + col,
                           global_workspace.GetWidth(),
                           col,
                           source);
}

__global__ void GenerateBandHypotheses(
    const RefCamera camera,
    const GpuMatView<float> depth_map,
    const GpuMatView<float> normal_map,
    GpuMatView<curandState> rand_state_map,
    const PatchMatchRngPerimeter rand_state_layout,
    const unsigned int rand_state_rotation,
    const SweepOptions options,
    LookAheadViews band) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= depth_map.GetWidth()) return;
  const size_t state_index =
      rand_state_layout.SweepIndex(rand_state_rotation, col);
  curandState rand_state = rand_state_map.Get(0, state_index);
  for (int local_row = 0; local_row < band.num_rows; ++local_row) {
    const int row = band.first_row + local_row;
    const float depth = depth_map.Get(row, col);
    float normal[3];
    normal_map.GetSlice(row, col, normal);
    const float random_depth =
        PerturbDepth(options.perturbation, depth, &rand_state);
    float random_normal[3];
    PerturbNormal(camera,
                  row,
                  col,
                  options.perturbation * M_PI,
                  normal,
                  &rand_state,
                  random_normal);
    band.random_parameters.Set(local_row, col, 0, random_depth);
    for (int axis = 0; axis < 3; ++axis) {
      band.random_parameters.Set(local_row, col, axis + 1, random_normal[axis]);
    }
    // These are raw draws, not sources: CDF selection still belongs to Sweep.
    for (int sample = 0; sample < options.num_samples; ++sample) {
      band.uniforms.Set(local_row, col, sample, curand_uniform(&rand_state));
    }
  }
  rand_state_map.Set(0, state_index, rand_state);
}

constexpr int kLookAheadStripeRows = 8;

template <bool kGeomConsistencyTerm>
__global__ void PrecomputeBandCosts(
    const RefCamera camera,
    const GpuMatView<float> depth_map,
    const GpuMatView<float> normal_map,
    const cudaTextureObject_t ref_image_texture,
    const internal::ReferenceStatisticsView ref_statistics,
    const cudaTextureObject_t src_images_texture,
    const cudaTextureObject_t src_depth_maps_texture,
    const cudaTextureObject_t poses_texture,
    const SweepOptions options,
    LookAheadViews band) {
  const int col = blockIdx.x * 8 + threadIdx.x;
  const int hypothesis = threadIdx.y % 3;
  const int first_source = threadIdx.y / 3;
  const int num_sources = band.costs.GetDepth() / 3;
  const int first_local_row = blockIdx.y * kLookAheadStripeRows;
  const int last_local_row =
      min(first_local_row + kLookAheadStripeRows, band.num_rows);
  using CostComputer = PhotoConsistencyCostComputer<11, 1, 8, true>;
  CostComputer computer(camera,
                        ref_image_texture,
                        src_images_texture,
                        poses_texture,
                        options.sigma_spatial,
                        options.sigma_color);
  computer.col = col;
  __shared__ float reference[CostComputer::LocalRefImageType::kDataSize];
  __shared__ float weights[CostComputer::kWeightCacheSize];
  __shared__ bool same_depth[8];
  __shared__ bool same_normal[8];
  computer.local_ref_image.data = reference;
  computer.local_weights = weights;
  for (int local_row = first_local_row; local_row < last_local_row; ++local_row) {
    const int row = band.first_row + local_row;
    __syncthreads();
    if (threadIdx.y == 0) {
      computer.ReadRef(row, local_row == first_local_row);
      same_depth[threadIdx.x] = false;
      same_normal[threadIdx.x] = false;
#ifndef COLMAP_HIP_ENABLED
      if (col < depth_map.GetWidth()) {
        // Compare once per column and publish through the existing tile
        // barrier. Bit equality preserves signed zero and NaN distinctions.
        same_depth[threadIdx.x] =
            __float_as_uint(depth_map.Get(row, col)) ==
            __float_as_uint(band.random_parameters.Get(local_row, col, 0));
        float current_normal[3];
        normal_map.GetSlice(row, col, current_normal);
        bool equal_normal = true;
        for (int axis = 0; axis < 3; ++axis) {
          equal_normal =
              equal_normal &&
              __float_as_uint(current_normal[axis]) == __float_as_uint(
                  band.random_parameters.Get(local_row, col, axis + 1));
        }
        same_normal[threadIdx.x] = equal_normal;
      }
#endif
    }
    __syncthreads();
    computer.ReadWeights(threadIdx.y, 12);
    __syncthreads();
    if (col >= depth_map.GetWidth()) continue;
    // h0 writes each aliased destination. Its skipped worker writes nothing,
    // so no extra synchronization or producer-to-producer read is needed.
    if ((hypothesis == 1 && same_depth[threadIdx.x]) ||
        (hypothesis == 2 && same_normal[threadIdx.x])) {
      continue;
    }

    float normal[3];
    if (hypothesis == 2) {
      normal_map.GetSlice(row, col, normal);
    } else {
      for (int axis = 0; axis < 3; ++axis) {
        normal[axis] = band.random_parameters.Get(local_row, col, axis + 1);
      }
    }
    computer.row = row;
    computer.normal = normal;
    computer.depth = hypothesis == 1
                         ? depth_map.Get(row, col)
                         : band.random_parameters.Get(local_row, col, 0);
    const float2 statistics = ref_statistics.Get(row, col);
    computer.local_ref_sum = statistics.x;
    computer.local_ref_squared_sum = statistics.y;
    for (int source = first_source; source < num_sources; source += 4) {
      computer.src_image_idx = source;
      const float cost = computer.Compute();
      band.costs.Set(
          local_row, col, hypothesis * num_sources + source, cost);
      if (hypothesis == 0) {
        if (same_depth[threadIdx.x]) {
          band.costs.Set(local_row, col, num_sources + source, cost);
        }
        if (same_normal[threadIdx.x]) {
          band.costs.Set(local_row, col, 2 * num_sources + source, cost);
        }
      }
      if constexpr (kGeomConsistencyTerm) {
        if (hypothesis < 2) {
          // Geometry depends on depth, not normal. h0 has random depth and
          // h1 current depth; h2 repeats random depth and needs no extra job.
          const int depth_kind = hypothesis == 0 ? 1 : 0;
          const float geometric_cost =
              ComputeGeomConsistencyCost(camera,
                                         poses_texture,
                                         src_depth_maps_texture,
                                         row,
                                         col,
                                         computer.depth,
                                         source,
                                         options.geom_consistency_max_cost);
          band.geometric_costs.Set(
              local_row, col, depth_kind * num_sources + source, geometric_cost);
          if (hypothesis == 0 && same_depth[threadIdx.x]) {
            band.geometric_costs.Set(local_row, col, source, geometric_cost);
          }
        }
      }
    }
  }
}

constexpr int kCooperativeSampleBatchSize = 4;
constexpr int kCooperativeSampleWindowSize = 16;

template <bool kCooperative,
          int kBlockWidth = THREADS_PER_BLOCK,
          int kSampleWindowSize = 1,
          int kLookAheadCapacity = 0,
          bool kGeomConsistencyTerm = false>
struct CooperativeNccJob {};

template <int kBlockWidth,
          int kSampleWindowSize,
          int kLookAheadCapacity,
          bool kGeomConsistencyTerm>
struct CooperativeNccJob<true,
                         kBlockWidth,
                         kSampleWindowSize,
                         kLookAheadCapacity,
                         kGeomConsistencyTerm> {
  static constexpr int kNumSlots =
      kLookAheadCapacity ? kLookAheadCapacity : kSampleWindowSize;
  static constexpr int kNumParameters = kLookAheadCapacity ? 1 : 4;
  static constexpr int kNumGeometryKinds = kGeomConsistencyTerm ? 3 : 1;
  // One padded rank shifts each band hypothesis plane by eight banks. This
  // avoids identical plane offsets when the first warp folds four hypotheses.
  static constexpr int kResultStride = kNumSlots + (kLookAheadCapacity ? 1 : 0);
  float depths[kNumParameters][kBlockWidth];
  float normals[kNumParameters][3][kBlockWidth];
  float results[4 * kResultStride][kBlockWidth];
  float geometric_costs[kNumGeometryKinds * kResultStride][kBlockWidth];
  float ref_sum[kBlockWidth];
  float ref_squared_sum[kBlockWidth];
  int source[kNumSlots][kBlockWidth];
  float best_depth[kBlockWidth];
  float best_normal[3][kBlockWidth];
  int best_idx[kBlockWidth];

  // Legacy slots contain four hypothesis lanes. Band workers each evaluate one
  // source, so transpose that layout to keep their stores in distinct banks.
  __device__ float& Result(const int slot,
                          const int hypothesis,
                          const int column) {
    return results[kLookAheadCapacity ? hypothesis * kResultStride + slot
                                     : slot * 4 + hypothesis][column];
  }

  __device__ float& GeometricCost(const int slot,
                                 const int kind,
                                 const int column) {
    return geometric_costs[kLookAheadCapacity ? kind * kResultStride + slot
                                            : slot * kNumGeometryKinds + kind]
                          [column];
  }
};

#ifndef COLMAP_HIP_ENABLED
// This helper is called after cache publication. The first warp executes every
// shuffle, including lanes for columns past valid_width; other warps return.
// All accumulation stays serial within each hypothesis, never a tree reduction.
template <int kCapacity, bool kGeomConsistencyTerm>
__device__ void FoldLookAheadSamples(
    const GpuMatView<float> sampled_sources,
    const int local_row,
    const int col,
    const int valid_width,
    const int num_samples,
    const float regularizer,
    const uint32_t source_mask,
    const int (&first_sources)[kCooperativeSampleWindowSize][8],
    const float (&incumbent_cost)[kCapacity][8],
    CooperativeNccJob<true,
                      8,
                      kCooperativeSampleWindowSize,
                      kCapacity,
                      kGeomConsistencyTerm>& cache,
    float (&costs)[5]) {
  const bool owner = threadIdx.y == 0;
  // x spans eight columns; y=0..3 is exactly the first warp. Each lane
  // folds one hypothesis in the original sample order, and y=0 also folds
  // hypothesis4. Shuffles deliver sums without another block barrier.
  if (threadIdx.y < 4) {
    constexpr uint32_t kFirstWarpMask = 0xffffffffu;
    const int hypothesis = threadIdx.y;
    float hypothesis_cost = 0.0f;
    float fourth_cost = 0.0f;
    const auto accumulate_cost = [&](const int index,
                                      const int source,
                                      const int slot,
                                      float& sum) {
      sum += index == 0 ? incumbent_cost[source][threadIdx.x]
                        : cache.Result(slot, index - 1, threadIdx.x);
      if constexpr (kGeomConsistencyTerm) {
        const int kind = index == 3 ? 0 : (index == 4 ? 2 : index);
        sum += regularizer * cache.GeometricCost(slot, kind, threadIdx.x);
      }
    };
    for (int sample = 0; sample < num_samples; ++sample) {
      int source = -1;
      int slot = -1;
      if (owner && col < valid_width) {
        source = sample < kCooperativeSampleWindowSize
                     ? first_sources[sample][threadIdx.x]
                     : static_cast<int>(
                           sampled_sources.Get(local_row, col, sample));
        if (source >= 0) {
          slot = __popc(source_mask & ((uint32_t{1} << source) - 1));
        }
      }
      // Invalid columns still execute every shuffle with the full mask.
      source = __shfl_sync(kFirstWarpMask, source, threadIdx.x);
      slot = __shfl_sync(kFirstWarpMask, slot, threadIdx.x);
      if (source < 0) continue;
      accumulate_cost(hypothesis, source, slot, hypothesis_cost);
      if (owner) accumulate_cost(4, source, slot, fourth_cost);
    }
    const float first_cost =
        __shfl_sync(kFirstWarpMask, hypothesis_cost, threadIdx.x + 8);
    const float second_cost =
        __shfl_sync(kFirstWarpMask, hypothesis_cost, threadIdx.x + 16);
    const float third_cost =
        __shfl_sync(kFirstWarpMask, hypothesis_cost, threadIdx.x + 24);
    if (owner && col < valid_width) {
      costs[0] = hypothesis_cost;
      costs[1] = first_cost;
      costs[2] = second_cost;
      costs[3] = third_cost;
      costs[4] = fourth_cost;
    }
  }
}
#endif

#ifdef COLMAP_PATCH_MATCH_TESTING
// Views are passed only in test builds; kTrace=false eliminates all writes.
struct SweepTrace {
  GpuMatView<int> draws;
  GpuMatView<int> winners;
  GpuMatView<int> support;
  GpuMatView<float> hypotheses;
  GpuMatView<float> costs;
  GpuMatView<float> forward;
};
#define SWEEP_TRACE_ARGUMENT \
  , SweepTrace {}
#else
#define SWEEP_TRACE_ARGUMENT
#endif

template <int kWindowSize,
          int kWindowStep,
          bool kGeomConsistencyTerm = false,
          bool kFilterPhotoConsistency = false,
          bool kFilterGeomConsistency = false,
          bool kCooperative = false,
          int kBlockWidth = THREADS_PER_BLOCK,
          int kLookAheadCapacity = 0
#ifdef COLMAP_PATCH_MATCH_TESTING
          ,
          bool kTrace = false
#endif
          >
__global__ void SweepFromTopToBottom(
    const RefCamera camera,
    GpuMatView<float> global_workspace,
    GpuMatView<curandState> rand_state_map,
    const PatchMatchRngPerimeter rand_state_layout,
    const unsigned int rand_state_rotation,
    GpuMatView<float> cost_map,
    GpuMatView<float> depth_map,
    GpuMatView<float> normal_map,
    PackedConsistencyMaskView consistency_mask,
    GpuMatView<float> sel_prob_map,
    const GpuMatView<float> prev_sel_prob_map,
    const cudaTextureObject_t ref_image_texture,
    const internal::ReferenceStatisticsView ref_statistics,
    const cudaTextureObject_t src_images_texture,
    const cudaTextureObject_t src_depth_maps_texture,
    const cudaTextureObject_t poses_texture,
    const SweepOptions options,
    LookAheadViews look_ahead
#ifdef COLMAP_PATCH_MATCH_TESTING
    ,
    SweepTrace trace
#endif
) {
  constexpr bool kLookAhead = kLookAheadCapacity != 0;
  static_assert(kLookAheadCapacity == 0 || kLookAheadCapacity == 16 ||
                kLookAheadCapacity == 32);
  static_assert(!kLookAhead || (kCooperative && kBlockWidth == 8));
  // The two retained incumbent source pairs are owned by y and y + 16.
  static_assert(!kLookAhead || 4 * kCooperativeSampleBatchSize == 16,
                "Look-ahead history retention requires sixteen source workers");
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const bool owner = !kCooperative || threadIdx.y == 0;
  constexpr int kSampleBatchSize =
      kCooperative && kBlockWidth == 8 ? kCooperativeSampleBatchSize : 1;
  constexpr int kSampleWindowSize =
      kCooperative && kBlockWidth == 8 ? kCooperativeSampleWindowSize : 1;
  constexpr int kCachedSources =
      kLookAhead ? kLookAheadCapacity : kSampleWindowSize;
  __shared__ CooperativeNccJob<kCooperative,
                              kBlockWidth,
                              kSampleWindowSize,
                              kLookAheadCapacity,
                              kGeomConsistencyTerm> cooperative_job;
  // Legacy sample aliases become the first sixteen selected IDs in band mode.
  // Only longer rows need the existing global uniform slots as tail-ID storage.
  __shared__ int cooperative_sample_alias[kSampleWindowSize][kBlockWidth];
  __shared__ uint32_t row_source_mask[kLookAhead ? kBlockWidth : 1];

  LikelihoodComputer likelihood_computer(options.ncc_sigma,
                                         options.min_triangulation_angle,
                                         options.incident_angle_sigma);

  // The two logical M*S regions are packed; allocation pitch is not stride.
  const int source_stride = global_workspace.GetWidth();
  float* forward_message = nullptr;
  float* sampling_probs = nullptr;
  if (col < cost_map.GetWidth()) {
    forward_message = &global_workspace.GetPtr()[col];
    sampling_probs =
        &global_workspace
             .GetPtr()[source_stride * global_workspace.GetHeight() + col];
  }

  //////////////////////////////////////////////////////////////////////////////
  // Compute backward message for all rows. Note that the backward messages are
  // temporarily stored in the sel_prob_map and replaced row by row as the
  // updated forward messages are computed further below.
  //////////////////////////////////////////////////////////////////////////////

  if (!kLookAhead && col < cost_map.GetWidth()) {
    const int first_source = kCooperative ? threadIdx.y : 0;
    const int source_step = kCooperative ? blockDim.y : 1;
    for (int image_idx = first_source; image_idx < cost_map.GetDepth();
         image_idx += source_step) {
      InitializeSourceMessages(likelihood_computer,
                               cost_map,
                               sel_prob_map,
                               forward_message,
                               source_stride,
                               col,
                               image_idx);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Estimate parameters for remaining rows and compute selection probabilities.
  //////////////////////////////////////////////////////////////////////////////

  typedef PhotoConsistencyCostComputer<kWindowSize, kWindowStep, kBlockWidth>
      PhotoConsistencyCostComputerType;
  PhotoConsistencyCostComputerType pcc_computer(camera,
                                                ref_image_texture,
                                                src_images_texture,
                                                poses_texture,
                                                options.sigma_spatial,
                                                options.sigma_color);
  pcc_computer.col = col;

  __shared__ float local_ref_image_data
      [PhotoConsistencyCostComputerType::LocalRefImageType::kDataSize];
  pcc_computer.local_ref_image.data = &local_ref_image_data[0];
  __shared__ float
      local_weight_data[PhotoConsistencyCostComputerType::kWeightCacheSize];
  pcc_computer.local_weights = &local_weight_data[0];

  struct ParamState {
    float depth = 0.0f;
    float normal[3] = {0};
  };

  // Parameters of previous pixel in column.
  ParamState prev_param_state;
  // Parameters of current pixel in column.
  ParamState curr_param_state;
  // Randomly sampled parameters.
  ParamState rand_param_state;
  // Cuda PRNG state for random sampling.
#ifndef COLMAP_HIP_ENABLED
  __shared__ float
      source_current_cost[kCooperative && kBlockWidth == 8 ? kCachedSources
                                                           : 1]
                         [kCooperative && kBlockWidth == 8 ? kBlockWidth : 1];
#endif
  curandState rand_state;

  if (owner && col < cost_map.GetWidth()) {
    if constexpr (!kLookAhead) {
      rand_state = rand_state_map.Get(
          0, rand_state_layout.SweepIndex(rand_state_rotation, col));
    }
    if (kLookAhead && look_ahead.first_row > 0) {
      prev_param_state.depth = look_ahead.previous_parameters.Get(0, col, 0);
      for (int axis = 0; axis < 3; ++axis) {
        prev_param_state.normal[axis] =
            look_ahead.previous_parameters.Get(0, col, axis + 1);
      }
    } else {
      prev_param_state.depth = depth_map.Get(0, col);
      normal_map.GetSlice(0, col, prev_param_state.normal);
    }
  }

  const int first_row = kLookAhead ? look_ahead.first_row : 0;
  const int last_row = kLookAhead ? first_row + look_ahead.num_rows
                                 : static_cast<int>(cost_map.GetHeight());
  for (int row = first_row; row < last_row; ++row) {
    if constexpr (!kCooperative) {
      // The scalar refresh has no end-of-row barrier before tile reuse.
      __syncthreads();
    }
    // Note that this must be executed even for pixels outside the borders,
    // since pixels are used in the local neighborhood of the current pixel.
    if constexpr (kCooperative) {
      // The owner loads the reference tile; groups fill independent weight
      // rows. The first barrier publishes the tile, the second its weights.
      if (owner) pcc_computer.ReadRef(row, kLookAhead && row == first_row);
      __syncthreads();
      if constexpr (kBlockWidth == 8 &&
                    PhotoConsistencyCostComputerType::kCacheWeights) {
        pcc_computer.ReadWeights(threadIdx.y, 4 * kSampleBatchSize);
      } else {
        if (owner) pcc_computer.ReadWeights();
      }
      __syncthreads();
    } else {
      pcc_computer.Read(row);
      if (col >= cost_map.GetWidth()) continue;
    }

    constexpr int kNumCosts = 5;
    float costs[kNumCosts] = {0};
    float depths[kNumCosts] = {0};
    const float* normals[kNumCosts] = {nullptr};
    // Source-chain ownership is y, y+16, independently of compact NCC ranks.
    // Keep both pairs for capacity32: an incumbent winner needs these exact
    // unweighted values again, but any changed winner must recompute them.
    float incumbent_alpha0;
    float incumbent_prob0;
    float incumbent_alpha1;
    float incumbent_prob1;
    bool empty_sampling_support = false;
    if (owner && col < cost_map.GetWidth()) {
      pcc_computer.row = row;
      const float2 statistics = ref_statistics.Get(row, col);
      pcc_computer.local_ref_sum = statistics.x;
      pcc_computer.local_ref_squared_sum = statistics.y;

      // Propagate the depth at which the current ray intersects with the plane
      // of the normal of the previous ray. This helps to better estimate
      // the depth of very oblique structures, i.e. pixels whose normal
      // direction is significantly different from their viewing direction.
      if (row > 0) {
        prev_param_state.depth =
            internal::PropagateDepth(camera.inv_K,
                                     prev_param_state.depth,
                                     prev_param_state.normal,
                                     row - 1,
                                     row,
                                     col);
      }

      // Read parameters for current pixel from previous sweep.
      curr_param_state.depth = depth_map.Get(row, col);
      normal_map.GetSlice(row, col, curr_param_state.normal);

      if constexpr (kLookAhead) {
        rand_param_state.depth =
            look_ahead.random_parameters.Get(row - first_row, col, 0);
        for (int axis = 0; axis < 3; ++axis) {
          rand_param_state.normal[axis] = look_ahead.random_parameters.Get(
              row - first_row, col, axis + 1);
        }
      } else {
        rand_param_state.depth = PerturbDepth(
            options.perturbation, curr_param_state.depth, &rand_state);
        PerturbNormal(camera,
                      row,
                      col,
                      options.perturbation * M_PI,
                      curr_param_state.normal,
                      &rand_state,
                      rand_param_state.normal);
      }
    }

    // Compute independent source probabilities before the ordered CDF update.
    if (col < cost_map.GetWidth()) {
      if constexpr (kCooperative) {
        if (!owner) {
          curr_param_state.depth = depth_map.Get(row, col);
          normal_map.GetSlice(row, col, curr_param_state.normal);
        }
      }
      float point[3];
      ComputePointAtDepth(camera, row, col, curr_param_state.depth, point);

      const int first_source = kCooperative ? threadIdx.y : 0;
      const int source_step = kCooperative ? blockDim.y : 1;
      for (int image_idx = first_source; image_idx < cost_map.GetDepth();
           image_idx += source_step) {
        const float cost = cost_map.Get(row, col, image_idx);
#ifndef COLMAP_HIP_ENABLED
        if constexpr (kCooperative && kBlockWidth == 8) {
          if (image_idx < kCachedSources)
            source_current_cost[image_idx][threadIdx.x] = cost;
        }
#endif
        const float alpha = likelihood_computer.ComputeForwardMessage(
            cost, forward_message[image_idx * source_stride]);
        const float beta = sel_prob_map.Get(row, col, image_idx);
        const float prev_prob = prev_sel_prob_map.Get(row, col, image_idx);
        const float sel_prob = likelihood_computer.ComputeSelProb(
            alpha, beta, prev_prob, options.prev_sel_prob_weight);
        if constexpr (kLookAhead) {
          if constexpr (kLookAheadCapacity == 32) {
            if (image_idx >= 16) {
              incumbent_alpha1 = alpha;
              incumbent_prob1 = sel_prob;
            } else {
              incumbent_alpha0 = alpha;
              incumbent_prob0 = sel_prob;
            }
          } else {
            incumbent_alpha0 = alpha;
            incumbent_prob0 = sel_prob;
          }
        }

        float cos_triangulation_angle;
        float cos_incident_angle;
        ComputeViewingAngles(poses_texture,
                             point,
                             curr_param_state.normal,
                             image_idx,
                             &cos_triangulation_angle,
                             &cos_incident_angle);
        const float tri_prob =
            likelihood_computer.ComputeTriProb(cos_triangulation_angle);
        const float inc_prob =
            likelihood_computer.ComputeIncProb(cos_incident_angle);

        float H[9];
        ComposeHomography(camera,
                          poses_texture,
                          image_idx,
                          row,
                          col,
                          curr_param_state.depth,
                          curr_param_state.normal,
                          H);
        const float res_prob =
            likelihood_computer.ComputeResolutionProb<kWindowSize>(H, row, col);

        sampling_probs[image_idx * source_stride] =
            sel_prob * tri_prob * inc_prob * res_prob;
      }
    }
    if constexpr (kCooperative) {
      // Finish every disjoint source probability before the ordered CDF update.
      __syncthreads();
    }
    if (owner && col < cost_map.GetWidth()) {
      empty_sampling_support = !TransformPDFToCDF(
          sampling_probs, cost_map.GetDepth(), source_stride);

      // Compute matching cost using Monte Carlo sampling of source images.
      // Images with higher selection probability are more likely to be sampled.
      // Hence, if only very few source images see the reference image pixel,
      // the same source image is likely to be sampled many times. Instead of
      // taking the best K probabilities, this sampling scheme has the advantage
      // of being adaptive to any distribution of selection probabilities.

      depths[0] = curr_param_state.depth;
      depths[1] = prev_param_state.depth;
      depths[2] = rand_param_state.depth;
      depths[3] = curr_param_state.depth;
      depths[4] = rand_param_state.depth;
      normals[0] = curr_param_state.normal;
      normals[1] = prev_param_state.normal;
      normals[2] = rand_param_state.normal;
      normals[3] = rand_param_state.normal;
      normals[4] = curr_param_state.normal;
      if constexpr (kCooperative) {
        const int lane = threadIdx.x;
        cooperative_job.ref_sum[lane] = pcc_computer.local_ref_sum;
        cooperative_job.ref_squared_sum[lane] =
            pcc_computer.local_ref_squared_sum;
        for (int hypothesis = 0; hypothesis < (kLookAhead ? 1 : 4);
             ++hypothesis) {
          cooperative_job.depths[hypothesis][lane] = depths[hypothesis + 1];
          for (int axis = 0; axis < 3; ++axis) {
            cooperative_job.normals[hypothesis][axis][lane] =
                normals[hypothesis + 1][axis];
          }
        }
      }
    }
    if constexpr (kCooperative) {
      // Parameters and reference sums stay immutable through this row.
      __syncthreads();
      if (col < cost_map.GetWidth()) {
        pcc_computer.row = row;
        pcc_computer.local_ref_sum = cooperative_job.ref_sum[threadIdx.x];
        pcc_computer.local_ref_squared_sum =
            cooperative_job.ref_squared_sum[threadIdx.x];
      }
    }

    if constexpr (kLookAhead) {
      // The row's CDF is immutable during evaluation. Consume each raw draw
      // once, retain its first sixteen source IDs in shared storage and any
      // tail IDs in consumed float slots, and compact the row only once. A NaN
      // CDF can select no source while its original support status stays true;
      // do not derive support from this mask.
      if (owner) {
        uint32_t sampled_sources = 0;
        if (col < cost_map.GetWidth()) {
          for (int sample = 0; sample < options.num_samples; ++sample) {
            const float uniform =
                look_ahead.uniforms.Get(row - first_row, col, sample);
            const int source = SampleSource(
                sampling_probs, cost_map.GetDepth(), source_stride, uniform);
            if (sample < kSampleWindowSize) {
              cooperative_sample_alias[sample][threadIdx.x] = source;
            } else {
              look_ahead.uniforms.Set(row - first_row, col, sample,
                                       static_cast<float>(source));
            }
#ifdef COLMAP_PATCH_MATCH_TESTING
            if constexpr (kTrace) trace.draws.Set(row, col, sample, source);
#endif
            if (source >= 0) sampled_sources |= uint32_t{1} << source;
          }
        }
        row_source_mask[threadIdx.x] = sampled_sources;
        uint32_t remaining_sources = sampled_sources;
        int slot = 0;
        while (remaining_sources != 0) {
          cooperative_job.source[slot++][threadIdx.x] =
              __ffs(remaining_sources) - 1;
          remaining_sources &= remaining_sources - 1;
        }
      }
      __syncthreads();
      if (col < cost_map.GetWidth()) {
        const int num_sources =
            __popc(row_source_mask[threadIdx.x]);
        float propagated_normal[3];
        for (int axis = 0; axis < 3; ++axis) {
          propagated_normal[axis] =
              cooperative_job.normals[0][axis][threadIdx.x];
        }
        pcc_computer.depth = cooperative_job.depths[0][threadIdx.x];
        pcc_computer.normal = propagated_normal;
        // Keep expensive work dense by compact rank, even when selected source
        // IDs are sparse. Cache values remain live through winner/filter reuse.
        for (int slot = threadIdx.y; slot < num_sources; slot += blockDim.y) {
          const int source = cooperative_job.source[slot][threadIdx.x];
          pcc_computer.src_image_idx = source;
          cooperative_job.Result(slot, 0, threadIdx.x) = pcc_computer.Compute();
          for (int hypothesis = 1; hypothesis < 4; ++hypothesis) {
            cooperative_job.Result(slot, hypothesis, threadIdx.x) =
                look_ahead.costs.Get(row - first_row,
                                     col,
                                     (hypothesis - 1) * cost_map.GetDepth() +
                                         source);
          }
          if constexpr (kGeomConsistencyTerm) {
            cooperative_job.GeometricCost(slot, 0, threadIdx.x) =
                look_ahead.geometric_costs.Get(row - first_row, col, source);
            cooperative_job.GeometricCost(slot, 1, threadIdx.x) =
                ComputeGeomConsistencyCost(camera,
                                           poses_texture,
                                           src_depth_maps_texture,
                                           row,
                                           col,
                                           pcc_computer.depth,
                                           source,
                                           options.geom_consistency_max_cost);
            cooperative_job.GeometricCost(slot, 2, threadIdx.x) =
                look_ahead.geometric_costs.Get(
                    row - first_row, col, cost_map.GetDepth() + source);
          }
        }
      }
      // Every actual source is now cached. Invalid columns participate too.
      __syncthreads();
#ifndef COLMAP_HIP_ENABLED
      FoldLookAheadSamples<kLookAheadCapacity, kGeomConsistencyTerm>(
          look_ahead.uniforms,
          row - first_row,
          col,
          cost_map.GetWidth(),
          options.num_samples,
          options.geom_consistency_regularizer,
          row_source_mask[threadIdx.x],
          cooperative_sample_alias,
          source_current_cost,
          cooperative_job,
          costs);
#endif
    } else {
      for (int sample = 0; sample < options.num_samples;
           sample += options.num_samples - sample < kSampleWindowSize
                         ? options.num_samples - sample
                         : kSampleWindowSize) {
        const int sample_count = options.num_samples - sample < kSampleWindowSize
                                     ? options.num_samples - sample
                                     : kSampleWindowSize;
        float geometric_costs[3] = {0};
        if (owner && col < cost_map.GetWidth()) {
          int num_unique_sources = 0;
          uint32_t sampled_sources = 0;
          const bool use_source_mask = cost_map.GetDepth() <= 32;
          for (int slot = 0; slot < kSampleWindowSize; ++slot) {
            pcc_computer.src_image_idx = -1;
            if (slot < sample_count) {
              // Draw only actual samples, in the original RNG order.
              const float uniform = curand_uniform(&rand_state);
              pcc_computer.src_image_idx = SampleSource(
                  sampling_probs, cost_map.GetDepth(), source_stride, uniform);
#ifdef COLMAP_PATCH_MATCH_TESTING
              if constexpr (kTrace) {
                trace.draws.Set(
                    row, col, sample + slot, pcc_computer.src_image_idx);
              }
#endif
            }
            if constexpr (kCooperative) {
              if constexpr (kSampleWindowSize > 1) {
                cooperative_sample_alias[slot][threadIdx.x] =
                    internal::AppendSampledSource(pcc_computer.src_image_idx,
                                                  use_source_mask,
                                                  sampled_sources,
                                                  num_unique_sources,
                                                  cooperative_job.source,
                                                  threadIdx.x);
              } else {
                cooperative_job.source[slot][threadIdx.x] =
                    pcc_computer.src_image_idx;
              }
            } else if (pcc_computer.src_image_idx >= 0) {
              if constexpr (kGeomConsistencyTerm) {
                for (int i = 0; i < 3; ++i) {
                  geometric_costs[i] = ComputeGeomConsistencyCost(
                      camera,
                      poses_texture,
                      src_depth_maps_texture,
                      row,
                      col,
                      depths[i],
                      pcc_computer.src_image_idx,
                      options.geom_consistency_max_cost);
                }
              }
              costs[0] += cost_map.Get(row, col, pcc_computer.src_image_idx);
              if constexpr (kGeomConsistencyTerm) {
                costs[0] +=
                    options.geom_consistency_regularizer * geometric_costs[0];
              }
            }
          }
          if constexpr (kCooperative && kSampleWindowSize > 1) {
            internal::PackSampledSources<kSampleWindowSize>(
                use_source_mask,
                sampled_sources,
                num_unique_sources,
                sample_count,
                cooperative_job.source,
                cooperative_sample_alias,
                threadIdx.x);
          }
        }

        if constexpr (kCooperative) {
          if (owner && col >= cost_map.GetWidth()) {
            for (int slot = 0; slot < kSampleWindowSize; ++slot) {
              cooperative_job.source[slot][threadIdx.x] = -1;
              if constexpr (kSampleWindowSize > 1) {
                cooperative_sample_alias[slot][threadIdx.x] = -1;
              }
            }
          }
          // Invalid columns and padded final-window samples reach both barriers.
          __syncthreads();
          const int hypothesis = threadIdx.y % 4;
          const int worker_limit = kSampleWindowSize == 1 ? 1 : sample_count;
          for (int slot = threadIdx.y / 4; slot < worker_limit;
               slot += kSampleBatchSize) {
            const int source = cooperative_job.source[slot][threadIdx.x];
            if (col < cost_map.GetWidth() && source >= 0) {
              if constexpr (kGeomConsistencyTerm) {
                if (hypothesis < 3) {
                  const int depth_hypothesis =
                      hypothesis == 0 ? 2 : hypothesis - 1;
                  cooperative_job.GeometricCost(slot, hypothesis, threadIdx.x) =
                      ComputeGeomConsistencyCost(
                          camera,
                          poses_texture,
                          src_depth_maps_texture,
                          row,
                          col,
                          cooperative_job.depths[depth_hypothesis][threadIdx.x],
                          source,
                          options.geom_consistency_max_cost);
                }
              }
              float hypothesis_normal[3];
              for (int axis = 0; axis < 3; ++axis) {
                hypothesis_normal[axis] =
                    cooperative_job.normals[hypothesis][axis][threadIdx.x];
              }
              pcc_computer.src_image_idx = source;
              pcc_computer.depth =
                  cooperative_job.depths[hypothesis][threadIdx.x];
              pcc_computer.normal = hypothesis_normal;
              cooperative_job.Result(slot, hypothesis, threadIdx.x) =
                  pcc_computer.Compute();
            }
          }
          __syncthreads();
          if (owner && col < cost_map.GetWidth()) {
            for (int slot = 0; slot < sample_count; ++slot) {
              int result_slot = slot;
              if constexpr (kSampleWindowSize > 1) {
                result_slot = cooperative_sample_alias[slot][threadIdx.x];
                if (result_slot < 0) continue;
              }
              const int source = cooperative_job.source[result_slot][threadIdx.x];
              if (source < 0) continue;
              // Reuse identical evaluations, but retain every original sample
              // add.
#ifndef COLMAP_HIP_ENABLED
              if constexpr (kBlockWidth == 8) {
                costs[0] += source < kSampleWindowSize
                                ? source_current_cost[source][threadIdx.x]
                                : cost_map.Get(row, col, source);
              } else
#endif
              {
                costs[0] += cost_map.Get(row, col, source);
              }
              if constexpr (kGeomConsistencyTerm) {
                for (int i = 0; i < 3; ++i) {
                  geometric_costs[i] =
                      cooperative_job.GeometricCost(result_slot, i, threadIdx.x);
                }
                costs[0] +=
                    options.geom_consistency_regularizer * geometric_costs[0];
              }
              for (int i = 1; i < kNumCosts; ++i) {
                costs[i] +=
                    cooperative_job.Result(result_slot, i - 1, threadIdx.x);
                if (kGeomConsistencyTerm) {
                  costs[i] += options.geom_consistency_regularizer *
                              geometric_costs[i == 3 ? 0 : (i == 4 ? 2 : i)];
                }
              }
            }
          }
        } else {
          if (pcc_computer.src_image_idx == -1) continue;
          for (int i = 1; i < kNumCosts; ++i) {
            pcc_computer.depth = depths[i];
            pcc_computer.normal = normals[i];
            costs[i] += pcc_computer.Compute();
            if (kGeomConsistencyTerm) {
              costs[i] += options.geom_consistency_regularizer *
                          geometric_costs[i == 3 ? 0 : (i == 4 ? 2 : i)];
            }
          }
        }
      }
    }

    int min_cost_idx = 0;
    float best_depth = 0.0f;
    const float* best_normal = nullptr;
    if (owner && col < cost_map.GetWidth()) {
      // Find the parameters of the minimum cost.
      min_cost_idx = empty_sampling_support ? 0 : FindMinCost<kNumCosts>(costs);
#ifdef COLMAP_PATCH_MATCH_TESTING
      if constexpr (kTrace) {
        trace.winners.Set(row, col, min_cost_idx);
        trace.support.Set(row, col, !empty_sampling_support);
        for (int i = 0; i < kNumCosts; ++i) {
          trace.costs.Set(row, col, i, costs[i]);
          trace.hypotheses.Set(row, col, 4 * i, depths[i]);
          for (int axis = 0; axis < 3; ++axis) {
            trace.hypotheses.Set(row, col, 4 * i + axis + 1, normals[i][axis]);
          }
        }
      }
#endif
      best_depth = depths[min_cost_idx];
      best_normal = normals[min_cost_idx];

      // Save best new parameters.
      depth_map.Set(row, col, best_depth);
      normal_map.SetSlice(row, col, best_normal);

      if constexpr (kCooperative) {
        const int lane = threadIdx.x;
        cooperative_job.best_depth[lane] = best_depth;
        cooperative_job.best_idx[lane] = min_cost_idx;
        for (int axis = 0; axis < 3; ++axis) {
          cooperative_job.best_normal[axis][lane] = best_normal[axis];
        }
      }
    }

    if constexpr (kCooperative) {
      // Publish the winner before independent source-chain updates.
      __syncthreads();
    }
    if (col < cost_map.GetWidth()) {
      int winner = min_cost_idx;
      float winning_normal[3];
      if constexpr (kCooperative) {
        const int lane = threadIdx.x;
        winner = cooperative_job.best_idx[lane];
        for (int axis = 0; axis < 3; ++axis) {
          winning_normal[axis] = cooperative_job.best_normal[axis][lane];
        }
        pcc_computer.depth = cooperative_job.best_depth[lane];
        pcc_computer.normal = winning_normal;
      } else {
        pcc_computer.depth = best_depth;
        pcc_computer.normal = best_normal;
      }
      float* source_forward_message = &global_workspace.GetPtr()[col];
      const int first_source = kCooperative ? threadIdx.y : 0;
      const int source_step = kCooperative ? blockDim.y : 1;
      for (int image_idx = first_source; image_idx < cost_map.GetDepth();
           image_idx += source_step) {
        float cost;
        if (winner == 0) {
          cost = cost_map.Get(row, col, image_idx);
        } else {
          if constexpr (kLookAhead) {
            if (winner >= 2) {
              cost = look_ahead.costs.Get(
                  row - first_row,
                  col,
                  (winner - 2) * cost_map.GetDepth() + image_idx);
            } else if (winner == 1 && (row_source_mask[threadIdx.x] &
                                       (uint32_t{1} << image_idx)) != 0) {
              // Every selected source remains live through refresh, not just
              // those in the last sixteen draws. Rank needs no list search.
              const uint32_t mask = row_source_mask[threadIdx.x];
              const uint32_t bit = uint32_t{1} << image_idx;
              const int slot = __popc(mask & (bit - 1));
              cost = cooperative_job.Result(slot, 0, threadIdx.x);
            } else {
              pcc_computer.src_image_idx = image_idx;
              cost = pcc_computer.Compute();
            }
          } else {
            pcc_computer.src_image_idx = image_idx;
            cost = pcc_computer.Compute();
          }
          cost_map.Set(row, col, image_idx, cost);
        }

        float alpha;
        float prob;
        if (kLookAhead && winner == 0) {
          if constexpr (kLookAheadCapacity == 32) {
            alpha = image_idx < 16 ? incumbent_alpha0 : incumbent_alpha1;
            prob = image_idx < 16 ? incumbent_prob0 : incumbent_prob1;
          } else {
            alpha = incumbent_alpha0;
            prob = incumbent_prob0;
          }
        } else {
          alpha = likelihood_computer.ComputeForwardMessage(
              cost, source_forward_message[image_idx * source_stride]);
          const float beta = sel_prob_map.Get(row, col, image_idx);
          const float prev_prob = prev_sel_prob_map.Get(row, col, image_idx);
          prob = likelihood_computer.ComputeSelProb(
              alpha, beta, prev_prob, options.prev_sel_prob_weight);
        }
        source_forward_message[image_idx * source_stride] = alpha;
        sel_prob_map.Set(row, col, image_idx, prob);
#ifdef COLMAP_PATCH_MATCH_TESTING
        if constexpr (kTrace) {
          trace.forward.Set(row, col, image_idx, alpha);
        }
#endif
      }
    }
    if constexpr (kCooperative) {
      // Filtering reads the updated source probabilities. Band geometry stays
      // immutable through filtering; the next row's tile barrier joins every
      // owner before any row-cache field is overwritten.
      __syncthreads();
    }

    if (owner && col < cost_map.GetWidth()) {
      if (kFilterPhotoConsistency || kFilterGeomConsistency) {
        int num_consistent = 0;

        float best_point[3];
        ComputePointAtDepth(camera, row, col, best_depth, best_point);

        const float min_ncc_prob =
            likelihood_computer.ComputeNCCProb(1.0f - options.filter_min_ncc);
        const float cos_min_triangulation_angle =
            cos(options.filter_min_triangulation_angle);

        // Reuse only a fresh evaluation of this exact winning depth. The term
        // and filter both call ComputeGeomConsistencyCost with max_cost above;
        // filter-only specializations have no table and keep the live call.
        const auto winning_geometric_cost = [&](const int image_idx) {
          if constexpr (kLookAhead && kGeomConsistencyTerm) {
            if (min_cost_idx == 0 || min_cost_idx == 3) {
              return look_ahead.geometric_costs.Get(
                  row - first_row, col, image_idx);
            }
            if (min_cost_idx == 2 || min_cost_idx == 4) {
              return look_ahead.geometric_costs.Get(
                  row - first_row, col, cost_map.GetDepth() + image_idx);
            }
            const uint32_t mask = row_source_mask[threadIdx.x];
            const uint32_t bit = uint32_t{1} << image_idx;
            if ((mask & bit) != 0) {
              return cooperative_job.GeometricCost(
                  __popc(mask & (bit - 1)), 1, threadIdx.x);
            }
          }
          return ComputeGeomConsistencyCost(camera,
                                            poses_texture,
                                            src_depth_maps_texture,
                                            row,
                                            col,
                                            best_depth,
                                            image_idx,
                                            options.geom_consistency_max_cost);
        };

        for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
          float cos_triangulation_angle;
          float cos_incident_angle;
          ComputeViewingAngles(poses_texture,
                               best_point,
                               best_normal,
                               image_idx,
                               &cos_triangulation_angle,
                               &cos_incident_angle);
          if (cos_triangulation_angle > cos_min_triangulation_angle ||
              cos_incident_angle <= 0.0f) {
            continue;
          }

          if (!kFilterGeomConsistency) {
            if (sel_prob_map.Get(row, col, image_idx) >= min_ncc_prob) {
              consistency_mask.Set(row, col, image_idx, 1);
              num_consistent += 1;
            }
          } else if (!kFilterPhotoConsistency) {
            if (winning_geometric_cost(image_idx) <=
                options.filter_geom_consistency_max_cost) {
              consistency_mask.Set(row, col, image_idx, 1);
              num_consistent += 1;
            }
          } else {
            if (sel_prob_map.Get(row, col, image_idx) >= min_ncc_prob &&
                winning_geometric_cost(image_idx) <=
                    options.filter_geom_consistency_max_cost) {
              consistency_mask.Set(row, col, image_idx, 1);
              num_consistent += 1;
            }
          }
        }

        if (num_consistent < options.filter_min_num_consistent) {
          depth_map.Set(row, col, 0.0f);
          normal_map.Set(row, col, 0, 0.0f);
          normal_map.Set(row, col, 1, 0.0f);
          normal_map.Set(row, col, 2, 0.0f);
          consistency_mask.Clear(row, col);
        }
      }

      // Update previous depth for next row.
      prev_param_state.depth = best_depth;
      for (int i = 0; i < 3; ++i) {
        prev_param_state.normal[i] = best_normal[i];
      }
    }
  }

  if (owner && col < cost_map.GetWidth()) {
    if constexpr (kLookAhead) {
      // Filtering may zero the public map. Propagation carries the selected,
      // unfiltered tuple, just as it does inside an uninterrupted Sweep.
      look_ahead.previous_parameters.Set(0, col, 0, prev_param_state.depth);
      for (int axis = 0; axis < 3; ++axis) {
        look_ahead.previous_parameters.Set(
            0, col, axis + 1, prev_param_state.normal[axis]);
      }
    } else {
      rand_state_map.Set(0,
                         rand_state_layout.SweepIndex(rand_state_rotation, col),
                         rand_state);
    }
  }
}

// Both production and the traced private probe use this dispatcher. Band
// kernels stay in the default stream, including the full backward prepass.
template <int kWindowSize,
          int kWindowStep,
          bool kGeomConsistencyTerm,
          bool kFilterPhotoConsistency,
          bool kFilterGeomConsistency,
          bool kCooperative,
          int kBlockWidth
#ifdef COLMAP_PATCH_MATCH_TESTING
          ,
          bool kTrace = false
#endif
          >
void LaunchSweep(LookAheadWorkspace* workspace,
                 const RefCamera camera,
                 GpuMatView<float> global_workspace,
                 GpuMatView<curandState> rand_state_map,
                 const PatchMatchRngPerimeter rand_state_layout,
                 const unsigned int rand_state_rotation,
                 GpuMatView<float> cost_map,
                 GpuMatView<float> depth_map,
                 GpuMatView<float> normal_map,
                 PackedConsistencyMaskView consistency_mask,
                 GpuMatView<float> sel_prob_map,
                 const GpuMatView<float> prev_sel_prob_map,
                 const cudaTextureObject_t ref_image_texture,
                 const internal::ReferenceStatisticsView ref_statistics,
                 const cudaTextureObject_t src_images_texture,
                 const cudaTextureObject_t src_depth_maps_texture,
                 const cudaTextureObject_t poses_texture,
                 const SweepOptions options
#ifdef COLMAP_PATCH_MATCH_TESTING
                 ,
                 SweepTrace trace
#endif
) {
  const dim3 grid((cost_map.GetWidth() + kBlockWidth - 1) / kBlockWidth);
  const dim3 block(kBlockWidth,
                    kCooperative ? 4 * kCooperativeSampleBatchSize : 1);
#define LAUNCH_SWEEP_KERNEL(LOOK_AHEAD, BAND)           \
  SweepFromTopToBottom<kWindowSize,                     \
                       kWindowStep,                     \
                       kGeomConsistencyTerm,            \
                       kFilterPhotoConsistency,         \
                       kFilterGeomConsistency,          \
                       kCooperative,                    \
                       kBlockWidth,                     \
                       LOOK_AHEAD SWEEP_TRACE_TEMPLATE> \
      <<<grid, block>>>(camera,                         \
                        global_workspace,               \
                        rand_state_map,                 \
                        rand_state_layout,              \
                        rand_state_rotation,            \
                        cost_map,                       \
                        depth_map,                      \
                        normal_map,                     \
                        consistency_mask,               \
                        sel_prob_map,                   \
                        prev_sel_prob_map,              \
                        ref_image_texture,              \
                        ref_statistics,          \
                        src_images_texture,             \
                        src_depth_maps_texture,         \
                        poses_texture,                  \
                        options,                        \
                        BAND SWEEP_TRACE_VALUE)
#ifdef COLMAP_PATCH_MATCH_TESTING
#define SWEEP_TRACE_TEMPLATE , kTrace
#define SWEEP_TRACE_VALUE , trace
#else
#define SWEEP_TRACE_TEMPLATE
#define SWEEP_TRACE_VALUE
#endif
#ifndef COLMAP_HIP_ENABLED
  if constexpr (kCooperative && kWindowSize == 11 && kWindowStep == 1) {
    if (workspace != nullptr) {
      InitializeSweepMessages<<<
          dim3((cost_map.GetWidth() + 31) / 32,
               (cost_map.GetDepth() + 3) / 4),
          dim3(32, 4)>>>(global_workspace, cost_map, sel_prob_map, options);
      for (int first_row = 0; first_row < cost_map.GetHeight();
           first_row += workspace->Rows()) {
        const int num_rows =
            std::min(workspace->Rows(),
                     static_cast<int>(cost_map.GetHeight()) - first_row);
        const LookAheadViews band =
            workspace->Views(rand_state_rotation, first_row, num_rows);
        GenerateBandHypotheses<<<(cost_map.GetWidth() + 31) / 32, 32>>>(
            camera,
            depth_map,
            normal_map,
            rand_state_map,
            rand_state_layout,
            rand_state_rotation,
            options,
            band);
        PrecomputeBandCosts<kGeomConsistencyTerm><<<
            dim3((cost_map.GetWidth() + 7) / 8,
                 (num_rows + kLookAheadStripeRows - 1) / kLookAheadStripeRows),
            dim3(8, 12)>>>(camera,
                           depth_map,
                           normal_map,
                           ref_image_texture,
                           ref_statistics,
                           src_images_texture,
                           src_depth_maps_texture,
                           poses_texture,
                           options,
                           band);
        if (cost_map.GetDepth() <= 16) {
          LAUNCH_SWEEP_KERNEL(16, band);
        } else {
          LAUNCH_SWEEP_KERNEL(32, band);
        }
      }
      return;
    }
  }
#endif
  LAUNCH_SWEEP_KERNEL(0, LookAheadViews{});
#undef SWEEP_TRACE_VALUE
#undef SWEEP_TRACE_TEMPLATE
#undef LAUNCH_SWEEP_KERNEL
}

PatchMatchCuda::PatchMatchCuda(const PatchMatchOptions& options,
                               const PatchMatch::Problem& problem)
    : options_(options),
      problem_(problem),
      ref_width_(0),
      ref_height_(0),
      rotation_in_half_pi_(0) {
  SetBestCudaDevice(std::stoi(options_.gpu_index));
  InitRefImage();
  InitSourceImages();
  InitTransforms();
  InitWorkspaceMemory();
}

void PatchMatchCuda::Run() {
  consistency_mask_valid_ = false;
  THROW_CHECK_NOTNULL(prev_sel_prob_map_.get());
  THROW_CHECK_NOTNULL(sel_prob_map_.get());

#define CASE_WINDOW_RADIUS(window_radius, window_step)              \
  case window_radius:                                               \
    RunWithWindowSizeAndStep<2 * window_radius + 1, window_step>(); \
    break;

#define CASE_WINDOW_STEP(window_step)                          \
  case window_step:                                            \
    switch (options_.window_radius) {                          \
      CASE_WINDOW_RADIUS(1, window_step)                       \
      CASE_WINDOW_RADIUS(2, window_step)                       \
      CASE_WINDOW_RADIUS(3, window_step)                       \
      CASE_WINDOW_RADIUS(4, window_step)                       \
      CASE_WINDOW_RADIUS(5, window_step)                       \
      CASE_WINDOW_RADIUS(6, window_step)                       \
      CASE_WINDOW_RADIUS(7, window_step)                       \
      CASE_WINDOW_RADIUS(8, window_step)                       \
      CASE_WINDOW_RADIUS(9, window_step)                       \
      CASE_WINDOW_RADIUS(10, window_step)                      \
      CASE_WINDOW_RADIUS(11, window_step)                      \
      CASE_WINDOW_RADIUS(12, window_step)                      \
      CASE_WINDOW_RADIUS(13, window_step)                      \
      CASE_WINDOW_RADIUS(14, window_step)                      \
      CASE_WINDOW_RADIUS(15, window_step)                      \
      CASE_WINDOW_RADIUS(16, window_step)                      \
      CASE_WINDOW_RADIUS(17, window_step)                      \
      CASE_WINDOW_RADIUS(18, window_step)                      \
      CASE_WINDOW_RADIUS(19, window_step)                      \
      CASE_WINDOW_RADIUS(20, window_step)                      \
      default: {                                               \
        LOG(ERROR) << "Window size " << options_.window_radius \
                   << " not supported";                        \
        break;                                                 \
      }                                                        \
    }                                                          \
    break;

  switch (options_.window_step) {
    CASE_WINDOW_STEP(1)
    CASE_WINDOW_STEP(2)
    default: {
      LOG(ERROR) << "Window step " << options_.window_step << " not supported";
      break;
    }
  }

#undef CASE_WINDOW_RADIUS
#undef CASE_WINDOW_STEP
}

DepthMap PatchMatchCuda::GetDepthMap() const {
  return DepthMap(
      depth_map_->CopyToMat(), options_.depth_min, options_.depth_max);
}

NormalMap PatchMatchCuda::GetNormalMap() const {
  return NormalMap(normal_map_->CopyToMat());
}

Mat<float> PatchMatchCuda::GetSelProbMap() const {
  THROW_CHECK_NOTNULL(prev_sel_prob_map_.get());
  return prev_sel_prob_map_->CopyToMat();
}

std::vector<int> PatchMatchCuda::GetConsistentImageIdxs() const {
  if (!consistency_mask_valid_) {
    return {};
  }
  const Mat<uint8_t> mask = consistency_mask_->CopyToMat();
  const size_t num_sources = problem_.src_image_idxs.size();
  const size_t num_planes = PackedConsistencyMaskDepth(num_sources);
  THROW_CHECK_EQ(mask.GetDepth(), num_planes);
  std::vector<int> consistent_image_idxs;
  std::vector<int> pixel_consistent_image_idxs;
  pixel_consistent_image_idxs.reserve(num_sources);
  for (size_t r = 0; r < mask.GetHeight(); ++r) {
    for (size_t c = 0; c < mask.GetWidth(); ++c) {
      pixel_consistent_image_idxs.clear();
      for (size_t plane = 0; plane < num_planes; ++plane) {
        const uint8_t packed = mask.Get(r, c, plane);
        if (packed == 0) {
          continue;
        }
        const size_t first = plane * kConsistencyFlagsPerByte;
        const size_t count =
            std::min(kConsistencyFlagsPerByte, num_sources - first);
        for (size_t bit = 0; bit < count; ++bit) {
          if ((packed & PackedConsistencyMaskBit(bit)) != 0) {
            pixel_consistent_image_idxs.push_back(
                problem_.src_image_idxs[first + bit]);
          }
        }
      }
      if (pixel_consistent_image_idxs.size() > 0) {
        consistent_image_idxs.push_back(c);
        consistent_image_idxs.push_back(r);
        consistent_image_idxs.push_back(pixel_consistent_image_idxs.size());
        consistent_image_idxs.insert(consistent_image_idxs.end(),
                                     pixel_consistent_image_idxs.begin(),
                                     pixel_consistent_image_idxs.end());
      }
    }
  }
  return consistent_image_idxs;
}

template <int kWindowSize, int kWindowStep>
void PatchMatchCuda::RunWithWindowSizeAndStep() {
  // Wait for all initializations to finish.
  CUDA_SYNC_AND_CHECK();

  CudaTimer total_timer;
  CudaTimer init_timer;
  std::unique_ptr<LookAheadWorkspace> look_ahead_workspace;
#ifndef COLMAP_HIP_ENABLED
  if constexpr (kWindowSize == 11 && kWindowStep == 1) {
    look_ahead_workspace = MakeLookAheadWorkspace(
        ref_width_,
        ref_height_,
        problem_.src_image_idxs.size(),
        options_.num_samples,
        options_.geom_consistency);
  }
#endif

  ComputeCudaConfig();
  dim3 initial_cost_grid = sweep_grid_size_;
  initial_cost_grid.y = (cost_map_->GetHeight() + kInitialCostStripeRows - 1) /
                        kInitialCostStripeRows;
  ComputeInitialCost<kWindowSize, kWindowStep>
      <<<initial_cost_grid, sweep_block_size_>>>(
          MakeRefCamera(ref_K_host_[rotation_in_half_pi_],
                        ref_inv_K_host_[rotation_in_half_pi_]),
          cost_map_->View(),
          depth_map_->View(),
          normal_map_->View(),
          ref_image_texture_->GetObj(),
          ReferenceStatistics(),
          src_images_texture_->GetObj(),
          poses_texture_[0]->GetObj(),
          options_.sigma_spatial,
          options_.sigma_color);
  CUDA_SYNC_AND_CHECK();

  init_timer.Print("Initialization");

  const float total_num_steps = options_.num_iterations * 4;

  SweepOptions sweep_options;
  sweep_options.depth_min = options_.depth_min;
  sweep_options.depth_max = options_.depth_max;
  sweep_options.sigma_spatial = options_.sigma_spatial;
  sweep_options.sigma_color = options_.sigma_color;
  sweep_options.num_samples = options_.num_samples;
  sweep_options.ncc_sigma = options_.ncc_sigma;
  sweep_options.min_triangulation_angle =
      DEG2RAD(options_.min_triangulation_angle);
  sweep_options.incident_angle_sigma = options_.incident_angle_sigma;
  sweep_options.geom_consistency_regularizer =
      options_.geom_consistency_regularizer;
  sweep_options.geom_consistency_max_cost = options_.geom_consistency_max_cost;
  sweep_options.filter_min_ncc = options_.filter_min_ncc;
  sweep_options.filter_min_triangulation_angle =
      DEG2RAD(options_.filter_min_triangulation_angle);
  sweep_options.filter_min_num_consistent = options_.filter_min_num_consistent;
  sweep_options.filter_geom_consistency_max_cost =
      options_.filter_geom_consistency_max_cost;

#ifdef COLMAP_HIP_ENABLED
  constexpr bool kCooperative = false;
#else
  constexpr bool kCooperative = kWindowSize == 11 && kWindowStep == 1;
#endif
  // Only radius 5 / step 1 uses cooperative sweeping (and never on HIP).
  // Initial-cost and element-wise kernels retain their existing launch shape.
  constexpr int kSweepBlockWidth = kCooperative ? 8 : THREADS_PER_BLOCK;

  for (int iter = 0; iter < options_.num_iterations; ++iter) {
    CudaTimer iter_timer;

    for (int sweep = 0; sweep < 4; ++sweep) {
      CudaTimer sweep_timer;
      // Expenentially reduce amount of perturbation during the optimization.
      sweep_options.perturbation = 1.0f / std::pow(2.0f, iter + sweep / 4.0f);

      // Linearly increase the influence of previous selection probabilities.
      sweep_options.prev_sel_prob_weight =
          static_cast<float>(iter * 4 + sweep) / total_num_steps;

      const bool last_sweep = iter == options_.num_iterations - 1 && sweep == 3;

#define CALL_SWEEP_FUNC                                                       \
  LaunchSweep<kWindowSize,                                                    \
              kWindowStep,                                                    \
              kGeomConsistencyTerm,                                           \
              kFilterPhotoConsistency,                                        \
              kFilterGeomConsistency,                                         \
              kCooperative,                                                   \
              kSweepBlockWidth>(                                              \
      look_ahead_workspace.get(),                                             \
      MakeRefCamera(ref_K_host_[rotation_in_half_pi_],                        \
                    ref_inv_K_host_[rotation_in_half_pi_]),                   \
      global_workspace_->View(),                                              \
      rand_state_map_->View(),                                                \
      (PatchMatchRngPerimeter{ref_width_, ref_height_}),                      \
      static_cast<unsigned int>(rotation_in_half_pi_),                        \
      cost_map_->View(),                                                      \
      depth_map_->View(),                                                     \
      normal_map_->View(),                                                    \
      PackedConsistencyMaskView{consistency_mask_->View()},                   \
      sel_prob_map_->View(),                                                  \
      prev_sel_prob_map_->View(),                                             \
      ref_image_texture_->GetObj(),                                           \
      ReferenceStatistics(),                                  \
      src_images_texture_->GetObj(),                                          \
      src_depth_maps_texture_ == nullptr ? 0                                  \
                                         : src_depth_maps_texture_->GetObj(), \
      poses_texture_[rotation_in_half_pi_]->GetObj(),                         \
      sweep_options SWEEP_TRACE_ARGUMENT);

      if (last_sweep) {
        if (options_.filter) {
          consistency_mask_ = std::make_unique<GpuMat<uint8_t>>(
              cost_map_->GetWidth(),
              cost_map_->GetHeight(),
              PackedConsistencyMaskDepth(cost_map_->GetDepth()));
          consistency_mask_->FillWithScalar(0);
        }
        if (options_.geom_consistency) {
          const bool kGeomConsistencyTerm = true;
          if (options_.filter) {
            const bool kFilterPhotoConsistency = true;
            const bool kFilterGeomConsistency = true;
            CALL_SWEEP_FUNC
          } else {
            const bool kFilterPhotoConsistency = false;
            const bool kFilterGeomConsistency = false;
            CALL_SWEEP_FUNC
          }
        } else {
          const bool kGeomConsistencyTerm = false;
          if (options_.filter) {
            const bool kFilterPhotoConsistency = true;
            const bool kFilterGeomConsistency = false;
            CALL_SWEEP_FUNC
          } else {
            const bool kFilterPhotoConsistency = false;
            const bool kFilterGeomConsistency = false;
            CALL_SWEEP_FUNC
          }
        }
      } else {
        const bool kFilterPhotoConsistency = false;
        const bool kFilterGeomConsistency = false;
        if (options_.geom_consistency) {
          const bool kGeomConsistencyTerm = true;
          CALL_SWEEP_FUNC
        } else {
          const bool kGeomConsistencyTerm = false;
          CALL_SWEEP_FUNC
        }
      }

#undef CALL_SWEEP_FUNC

      CUDA_SYNC_AND_CHECK();
      if (last_sweep) {
        // No later kernel uses the band storage; release it before rotation.
        look_ahead_workspace.reset();
      }

      Rotate();

      // Rotate selected image map.
      if (last_sweep && options_.filter) {
        {
          std::unique_ptr<GpuMat<uint8_t>> rot_consistency_mask_(
              new GpuMat<uint8_t>(cost_map_->GetWidth(),
                                  cost_map_->GetHeight(),
                                  consistency_mask_->GetDepth()));
          consistency_mask_->Rotate(rot_consistency_mask_.get());
          consistency_mask_.swap(rot_consistency_mask_);
        }
      }

      sweep_timer.Print(" Sweep " + std::to_string(sweep + 1));
    }

    iter_timer.Print("Iteration " + std::to_string(iter + 1));
  }

  look_ahead_workspace.reset();
  total_timer.Print("Total");
  consistency_mask_valid_ = options_.filter;
}

#undef SWEEP_TRACE_ARGUMENT

void PatchMatchCuda::ComputeCudaConfig() {
  sweep_block_size_.x = THREADS_PER_BLOCK;
  sweep_block_size_.y = 1;
  sweep_block_size_.z = 1;
  sweep_grid_size_.x = (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
  sweep_grid_size_.y = 1;
  sweep_grid_size_.z = 1;

  elem_wise_block_size_.x = THREADS_PER_BLOCK;
  elem_wise_block_size_.y = THREADS_PER_BLOCK;
  elem_wise_block_size_.z = 1;
  elem_wise_grid_size_.x = (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
  elem_wise_grid_size_.y =
      (depth_map_->GetHeight() - 1) / THREADS_PER_BLOCK + 1;
  elem_wise_grid_size_.z = 1;
}

void PatchMatchCuda::BindRefImageTexture() {
  cudaTextureDesc texture_desc;
  memset(&texture_desc, 0, sizeof(texture_desc));
  texture_desc.addressMode[0] = cudaAddressModeBorder;
  texture_desc.addressMode[1] = cudaAddressModeBorder;
  texture_desc.addressMode[2] = cudaAddressModeBorder;
  texture_desc.filterMode = cudaFilterModePoint;
  texture_desc.readMode = cudaReadModeNormalizedFloat;
  texture_desc.normalizedCoords = false;
  ref_image_texture_ = CudaArrayLayeredTexture<uint8_t>::FromGpuMat(
      texture_desc, *ref_image_);
}

internal::ReferenceStatisticsView PatchMatchCuda::ReferenceStatistics() const {
  const GpuMatView<float> empty{nullptr, 0, 0, 0, 0};
  return {ref_sum_image_ ? ref_sum_image_->View() : empty,
          ref_squared_sum_image_ ? ref_squared_sum_image_->View() : empty,
          ref_statistics_texture_ ? ref_statistics_texture_->GetObj() : 0,
          PatchMatchRngPerimeter{ref_width_, ref_height_},
          static_cast<unsigned int>(rotation_in_half_pi_)};
}

void PatchMatchCuda::InitRefImage() {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

  ref_width_ = ref_image.GetWidth();
  ref_height_ = ref_image.GetHeight();

  // Preserve the original filtering arithmetic and its ordered tap folds.
  // Only ownership/read coordinates change after this one original-image pass.
  {
    GpuMatRefImage filtered(ref_width_, ref_height_);
    filtered.Filter(ref_image.GetBitmap().RowMajorData().data(),
                    options_.window_radius,
                    options_.window_step,
                    options_.sigma_spatial,
                    options_.sigma_color);
    ref_image_ = std::move(filtered.image);
#ifndef COLMAP_HIP_ENABLED
    if (options_.window_radius == 5 && options_.window_step == 1) {
      static_assert(sizeof(uint2) == 2 * sizeof(float));
      GpuMat<uint2> packed(ref_width_, ref_height_);
      const dim3 block(32, 16);
      const dim3 grid((ref_width_ - 1) / block.x + 1,
                      (ref_height_ - 1) / block.y + 1);
      PackReferenceStatistics<<<grid, block>>>(filtered.sum_image->View(),
                                               filtered.squared_sum_image->View(),
                                               packed.View());
      CUDA_SYNC_AND_CHECK();
      // The pack has finished reading both float planes. Release them before
      // allocating the immutable array instead of overlapping all three copies.
      filtered.sum_image.reset();
      filtered.squared_sum_image.reset();
      cudaTextureDesc desc = {};
      desc.addressMode[0] = cudaAddressModeClamp;
      desc.addressMode[1] = cudaAddressModeClamp;
      desc.filterMode = cudaFilterModePoint;
      desc.readMode = cudaReadModeElementType;
      desc.normalizedCoords = 0;
      ref_statistics_texture_ =
          CudaArrayLayeredTexture<uint2>::FromGpuMat(desc, packed);
    } else
#endif
    {
      ref_sum_image_ = std::move(filtered.sum_image);
      ref_squared_sum_image_ = std::move(filtered.squared_sum_image);
    }
  }

  BindRefImageTexture();
}

void PatchMatchCuda::InitSourceImages() {
  // Determine maximum image size.
  size_t max_width = 0;
  size_t max_height = 0;
  for (const auto image_idx : problem_.src_image_idxs) {
    const Image& image = problem_.images->at(image_idx);
    if (image.GetWidth() > max_width) {
      max_width = image.GetWidth();
    }
    if (image.GetHeight() > max_height) {
      max_height = image.GetHeight();
    }
  }

  // Upload source images to device.
  {
    // Copy source images to contiguous memory block.
    const uint8_t kDefaultValue = 0;
    std::vector<uint8_t> src_images_host_data(
        static_cast<size_t>(max_width * max_height *
                            problem_.src_image_idxs.size()),
        kDefaultValue);
    for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
      const Image& image = problem_.images->at(problem_.src_image_idxs[i]);
      const Bitmap& bitmap = image.GetBitmap();
      // The texture reads each layer with a row stride of max_width, while
      // the bitmap rows are packed with the image's own width. Images
      // narrower than max_width must therefore be copied row by row - a
      // single contiguous copy would displace row r by r * (max_width -
      // width) pixels. Example for a 2x2 image in a 3-wide layer:
      //   bitmap [a b; c d] -> contiguous copy [a b c; d 0 0] (wrong)
      //                        row-wise copy   [a b 0; c d 0] (correct)
      const uint8_t* src = bitmap.RowMajorData().data();
      uint8_t* dest = src_images_host_data.data() + max_width * max_height * i;
      if (image.GetWidth() == max_width) {
        memcpy(dest, src, max_width * image.GetHeight());
      } else {
        for (size_t r = 0; r < image.GetHeight(); ++r) {
          memcpy(dest, src + r * image.GetWidth(), image.GetWidth());
          dest += max_width;
        }
      }
    }

    // Create source images texture.
    cudaTextureDesc texture_desc;
    memset(&texture_desc, 0, sizeof(texture_desc));
    texture_desc.addressMode[0] = cudaAddressModeBorder;
    texture_desc.addressMode[1] = cudaAddressModeBorder;
    texture_desc.addressMode[2] = cudaAddressModeBorder;
    texture_desc.filterMode = cudaFilterModeLinear;
#ifdef COLMAP_HIP_ENABLED
    // AMD CDNA/gfx9 GPUs do not support linear filtering of layered textures.
    // Detect the actual device at runtime (so a binary built for several
    // architectures still picks the right path per GPU) and fall back to point
    // filtering there; the gfx9 device code emulates bilinear filtering in the
    // kernel (see SampleLayeredBilinear). Non-gfx9 GPUs keep hardware
    // filtering.
    int device = 0;
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    cudaDeviceProp device_prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_prop, device));
    if (std::string(device_prop.gcnArchName).rfind("gfx9", 0) == 0) {
      texture_desc.filterMode = cudaFilterModePoint;
    }
#endif
    texture_desc.readMode = cudaReadModeNormalizedFloat;
    texture_desc.normalizedCoords = false;
    src_images_texture_ = CudaArrayLayeredTexture<uint8_t>::FromHostArray(
        texture_desc,
        max_width,
        max_height,
        problem_.src_image_idxs.size(),
        src_images_host_data.data());
  }

  // Upload source depth maps to device.
  if (options_.geom_consistency) {
    const float kDefaultValue = 0.0f;
    std::vector<float> src_depth_maps_host_data(
        static_cast<size_t>(max_width * max_height *
                            problem_.src_image_idxs.size()),
        kDefaultValue);
    for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
      const DepthMap& depth_map =
          problem_.depth_maps->at(problem_.src_image_idxs[i]);
      float* dest =
          src_depth_maps_host_data.data() + max_width * max_height * i;
      if (depth_map.GetWidth() == max_width) {
        memcpy(dest,
               depth_map.GetPtr(),
               max_width * depth_map.GetHeight() * sizeof(float));
      } else {
        for (size_t r = 0; r < depth_map.GetHeight(); ++r) {
          memcpy(dest,
                 depth_map.GetPtr() + r * depth_map.GetWidth(),
                 depth_map.GetWidth() * sizeof(float));
          dest += max_width;
        }
      }
    }

    // Create source depth maps texture.
    cudaTextureDesc texture_desc;
    memset(&texture_desc, 0, sizeof(texture_desc));
    texture_desc.addressMode[0] = cudaAddressModeBorder;
    texture_desc.addressMode[1] = cudaAddressModeBorder;
    texture_desc.addressMode[2] = cudaAddressModeBorder;
    texture_desc.filterMode = cudaFilterModePoint;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = false;
    src_depth_maps_texture_ = CudaArrayLayeredTexture<float>::FromHostArray(
        texture_desc,
        max_width,
        max_height,
        problem_.src_image_idxs.size(),
        src_depth_maps_host_data.data());
  }
}

void PatchMatchCuda::InitTransforms() {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

  //////////////////////////////////////////////////////////////////////////////
  // Generate rotated versions (counter-clockwise) of calibration matrix.
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < 4; ++i) {
    ref_K_host_[i][0] = ref_image.GetK()[0];
    ref_K_host_[i][1] = ref_image.GetK()[2] - 0.5f;
    ref_K_host_[i][2] = ref_image.GetK()[4];
    ref_K_host_[i][3] = ref_image.GetK()[5] - 0.5f;
  }

  // Rotated by 90 degrees.
  std::swap(ref_K_host_[1][0], ref_K_host_[1][2]);
  std::swap(ref_K_host_[1][1], ref_K_host_[1][3]);
  ref_K_host_[1][3] = ref_width_ - 1 - ref_K_host_[1][3];

  // Rotated by 180 degrees.
  ref_K_host_[2][1] = ref_width_ - 1 - ref_K_host_[2][1];
  ref_K_host_[2][3] = ref_height_ - 1 - ref_K_host_[2][3];

  // Rotated by 270 degrees.
  std::swap(ref_K_host_[3][0], ref_K_host_[3][2]);
  std::swap(ref_K_host_[3][1], ref_K_host_[3][3]);
  ref_K_host_[3][1] = ref_height_ - 1 - ref_K_host_[3][1];

  // Extract 1/fx, -cx/fx, fy, -cy/fy.
  for (size_t i = 0; i < 4; ++i) {
    ref_inv_K_host_[i][0] = 1.0f / ref_K_host_[i][0];
    ref_inv_K_host_[i][1] = -ref_K_host_[i][1] / ref_K_host_[i][0];
    ref_inv_K_host_[i][2] = 1.0f / ref_K_host_[i][2];
    ref_inv_K_host_[i][3] = -ref_K_host_[i][3] / ref_K_host_[i][2];
  }

  //////////////////////////////////////////////////////////////////////////////
  // Generate rotated versions of camera poses.
  //////////////////////////////////////////////////////////////////////////////

  float rotated_R[9];
  memcpy(rotated_R, ref_image.GetR(), 9 * sizeof(float));

  float rotated_T[3];
  memcpy(rotated_T, ref_image.GetT(), 3 * sizeof(float));

  // Matrix for 90deg rotation around Z-axis in counter-clockwise direction.
  const float R_z90[9] = {0, 1, 0, -1, 0, 0, 0, 0, 1};

  cudaTextureDesc texture_desc;
  memset(&texture_desc, 0, sizeof(texture_desc));
  texture_desc.addressMode[0] = cudaAddressModeBorder;
  texture_desc.addressMode[1] = cudaAddressModeBorder;
  texture_desc.addressMode[2] = cudaAddressModeBorder;
  texture_desc.filterMode = cudaFilterModePoint;
  texture_desc.readMode = cudaReadModeElementType;
  texture_desc.normalizedCoords = false;

  for (size_t i = 0; i < 4; ++i) {
    const size_t kNumTformParams = 4 + 9 + 3 + 3 + 12 + 12;
    std::vector<float> poses_host_data(kNumTformParams *
                                       problem_.src_image_idxs.size());
    int offset = 0;
    for (const auto image_idx : problem_.src_image_idxs) {
      const Image& image = problem_.images->at(image_idx);

      float src_K[9];
      memcpy(src_K, image.GetK(), 9 * sizeof(float));
      src_K[2] -= 0.5f;
      src_K[5] -= 0.5f;

      const float K[4] = {src_K[0], src_K[2], src_K[4], src_K[5]};
      memcpy(poses_host_data.data() + offset, K, 4 * sizeof(float));
      offset += 4;

      float rel_R[9];
      float rel_T[3];
      ComputeRelativePose(
          rotated_R, rotated_T, image.GetR(), image.GetT(), rel_R, rel_T);
      memcpy(poses_host_data.data() + offset, rel_R, 9 * sizeof(float));
      offset += 9;
      memcpy(poses_host_data.data() + offset, rel_T, 3 * sizeof(float));
      offset += 3;

      float C[3];
      ComputeProjectionCenter(rel_R, rel_T, C);
      memcpy(poses_host_data.data() + offset, C, 3 * sizeof(float));
      offset += 3;

      float P[12];
      ComposeProjectionMatrix(src_K, rel_R, rel_T, P);
      memcpy(poses_host_data.data() + offset, P, 12 * sizeof(float));
      offset += 12;

      float inv_P[12];
      ComposeInverseProjectionMatrix(src_K, rel_R, rel_T, inv_P);
      memcpy(poses_host_data.data() + offset, inv_P, 12 * sizeof(float));
      offset += 12;
    }

    poses_texture_[i] = CudaArrayLayeredTexture<float>::FromHostArray(
        texture_desc,
        kNumTformParams,
        problem_.src_image_idxs.size(),
        1,
        poses_host_data.data());

    RotatePose(R_z90, rotated_R, rotated_T);
  }
}

void PatchMatchCuda::InitWorkspaceMemory() {
  THROW_CHECK_GT(ref_width_, 0);
  THROW_CHECK_GT(ref_height_, 0);
  const size_t max_size = std::numeric_limits<size_t>::max();
  if (ref_width_ > 1 && ref_height_ > 1) {
    THROW_CHECK_LE(ref_width_ - 1, max_size / 2);
    THROW_CHECK_LE(ref_height_ - 1, (max_size - 2 * (ref_width_ - 1)) / 2);
  }
  const PatchMatchRngPerimeter layout{ref_width_, ref_height_};
  const size_t count = layout.Count();
  THROW_CHECK_LE(count, max_size / sizeof(curandState));
  rand_state_map_ = std::make_unique<GpuMat<curandState>>(count, 1);
  if (options_.geom_consistency) {
    constexpr size_t kThreads = 256;
    const size_t blocks = (count - 1) / kThreads + 1;
    THROW_CHECK_LE(blocks, std::numeric_limits<int>::max());
    InitPerimeter<<<static_cast<unsigned int>(blocks), kThreads>>>(
        rand_state_map_->View(), layout);
  }

  depth_map_ = std::make_unique<GpuMat<float>>(ref_width_, ref_height_);
  if (options_.geom_consistency) {
    const DepthMap& init_depth_map =
        problem_.depth_maps->at(problem_.ref_image_idx);
    depth_map_->CopyToDevice(init_depth_map.GetPtr(),
                             init_depth_map.GetWidth() * sizeof(float));
  }

  normal_map_ = std::make_unique<GpuMat<float>>(ref_width_, ref_height_, 3);

  // Note that it is not necessary to keep the selection probability map in
  // memory for all pixels. Theoretically, it is possible to incorporate
  // the temporary selection probabilities in the global_workspace_.
  // However, it is useful to keep the probabilities for the entire image
  // in memory, so that it can be exported.
  sel_prob_map_ = std::make_unique<GpuMat<float>>(
      ref_width_, ref_height_, problem_.src_image_idxs.size());
  prev_sel_prob_map_ = std::make_unique<GpuMat<float>>(
      ref_width_, ref_height_, problem_.src_image_idxs.size());
  prev_sel_prob_map_->FillWithScalar(0.5f);

  cost_map_ = std::make_unique<GpuMat<float>>(
      ref_width_, ref_height_, problem_.src_image_idxs.size());

  // Two packed M*S regions fit in the pitched allocation; M is the logical
  // source stride, independent of the allocated pitch.
  const int ref_max_dim = std::max(ref_width_, ref_height_);
  global_workspace_ = std::make_unique<GpuMat<float>>(
      ref_max_dim, problem_.src_image_idxs.size(), 2);

  consistency_mask_ = std::make_unique<GpuMat<uint8_t>>(0, 0, 0);

  ComputeCudaConfig();

  if (options_.geom_consistency) {
    const NormalMap& init_normal_map =
        problem_.normal_maps->at(problem_.ref_image_idx);
    normal_map_->CopyToDevice(init_normal_map.GetPtr(),
                              init_normal_map.GetWidth() * sizeof(float));
  } else {
    InitDepthNormalMaps<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
        MakeRefCamera(ref_K_host_[rotation_in_half_pi_],
                      ref_inv_K_host_[rotation_in_half_pi_]),
        depth_map_->View(),
        normal_map_->View(),
        rand_state_map_->View(),
        layout,
        options_.depth_min,
        options_.depth_max);
  }
  CUDA_SYNC_AND_CHECK();
}

void PatchMatchCuda::Rotate() {
  // The completed sweep no longer needs the previous selection probabilities.
  prev_sel_prob_map_.reset();

  rotation_in_half_pi_ = (rotation_in_half_pi_ + 1) % 4;

  size_t width;
  size_t height;
  if (rotation_in_half_pi_ % 2 == 0) {
    width = ref_width_;
    height = ref_height_;
  } else {
    width = ref_height_;
    height = ref_width_;
  }

  // RNG remains in canonical perimeter coordinates across all rotations.

  // Rotate depth map.
  {
    std::unique_ptr<GpuMat<float>> rotated_depth_map(
        new GpuMat<float>(width, height));
    depth_map_->Rotate(rotated_depth_map.get());
    depth_map_.swap(rotated_depth_map);
  }

  // Rotate normal map.
  {
    RotateNormalMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
        normal_map_->View());
    std::unique_ptr<GpuMat<float>> rotated_normal_map(
        new GpuMat<float>(width, height, 3));
    normal_map_->Rotate(rotated_normal_map.get());
    normal_map_.swap(rotated_normal_map);
  }

  // Rotate byte pixels. Only fallback moment planes rotate; the default CUDA
  // texture retains original-orientation float representations for every Run.
  {
    auto rotated_ref_image = std::make_unique<GpuMat<uint8_t>>(width, height);
    std::unique_ptr<GpuMat<float>> rotated_sum, rotated_squared_sum;
    if (ref_sum_image_) {
      rotated_sum = std::make_unique<GpuMat<float>>(width, height);
      rotated_squared_sum = std::make_unique<GpuMat<float>>(width, height);
    }
    ref_image_->Rotate(rotated_ref_image.get());
    if (ref_sum_image_) {
      ref_sum_image_->Rotate(rotated_sum.get());
      ref_squared_sum_image_->Rotate(rotated_squared_sum.get());
      ref_sum_image_.swap(rotated_sum);
      ref_squared_sum_image_.swap(rotated_squared_sum);
    }
    ref_image_.swap(rotated_ref_image);
    BindRefImageTexture();
  }

  // Rotate selection probability map.
  prev_sel_prob_map_ = std::make_unique<GpuMat<float>>(
      width, height, problem_.src_image_idxs.size());
  sel_prob_map_->Rotate(prev_sel_prob_map_.get());
  sel_prob_map_.reset();

  // Rotate cost map.
  {
    std::unique_ptr<GpuMat<float>> rotated_cost_map(
        new GpuMat<float>(width, height, problem_.src_image_idxs.size()));
    cost_map_->Rotate(rotated_cost_map.get());
    cost_map_.swap(rotated_cost_map);
  }

  // The next sweep overwrites this scratch map before reading it. Allocate it
  // after releasing the old cost map to avoid an extra live scalar map.
  sel_prob_map_ = std::make_unique<GpuMat<float>>(
      width, height, problem_.src_image_idxs.size());

  // Recompute Cuda configuration for rotated reference image.
  ComputeCudaConfig();
}

#ifdef COLMAP_PATCH_MATCH_TESTING
// Private bridge for the direct ownership regression. It seeds the existing
// PRNG state and computes the same initial cost used by Run, before any sweep.
namespace internal {
namespace {
__global__ void CaptureReferenceStatistics(
    const ReferenceStatisticsView statistics, GpuMatView<uint32_t> bits) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < bits.GetHeight() && col < bits.GetWidth()) {
    const float2 value = statistics.Get(row, col);
    bits.Set(row, col, 0, __float_as_uint(value.x));
    bits.Set(row, col, 1, __float_as_uint(value.y));
  }
}

// Independent producer oracle: use one column owner, visit every source and
// proposal without aliasing, and execute the retained literal PR59 NCC body.
// The shared geometry helper is unchanged; this fixture tests its depth routing
// and destination publication, not the reprojection arithmetic itself.
__global__ void ComputeFreshProposalReference(
    const RefCamera camera,
    const GpuMatView<float> depth,
    const GpuMatView<float> normal,
    const cudaTextureObject_t ref_texture,
    const ReferenceStatisticsView statistics,
    const cudaTextureObject_t sources_texture,
    const cudaTextureObject_t depths_texture,
    const cudaTextureObject_t poses_texture,
    const SweepOptions options,
    const LookAheadViews band,
    GpuMatView<float> costs,
    GpuMatView<float> geometry) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  using Computer = PhotoConsistencyCostComputer<11, 1>;
  Computer computer(camera,
                    ref_texture,
                    sources_texture,
                    poses_texture,
                    options.sigma_spatial,
                    options.sigma_color);
  __shared__ float reference[Computer::LocalRefImageType::kDataSize];
  __shared__ float weights[Computer::kWeightCacheSize];
  computer.local_ref_image.data = reference;
  computer.local_weights = weights;
  computer.col = col;
  const int num_sources = costs.GetDepth() / 3;
  for (int local_row = 0; local_row < band.num_rows; ++local_row) {
    const int row = band.first_row + local_row;
    __syncthreads();
    computer.Read(row, local_row == 0);
    if (col >= depth.GetWidth()) continue;
    computer.row = row;
    const float2 moments = statistics.Get(row, col);
    computer.local_ref_sum = moments.x;
    computer.local_ref_squared_sum = moments.y;
    float current_normal[3], random_normal[3];
    normal.GetSlice(row, col, current_normal);
    for (int axis = 0; axis < 3; ++axis) {
      random_normal[axis] =
          band.random_parameters.Get(local_row, col, axis + 1);
    }
    const float current_depth = depth.Get(row, col);
    const float random_depth = band.random_parameters.Get(local_row, col, 0);
    for (int source = 0; source < num_sources; ++source) {
      computer.src_image_idx = source;
      for (int hypothesis = 0; hypothesis < 3; ++hypothesis) {
        computer.depth = hypothesis == 1 ? current_depth : random_depth;
        computer.normal = hypothesis == 2 ? current_normal : random_normal;
        costs.Set(local_row,
                  col,
                  hypothesis * num_sources + source,
                  computer.ComputeReferenceForTesting());
      }
      geometry.Set(
          local_row,
          col,
          source,
          ComputeGeomConsistencyCost(camera,
                                     poses_texture,
                                     depths_texture,
                                     row,
                                     col,
                                     current_depth,
                                     source,
                                     options.geom_consistency_max_cost));
      geometry.Set(
          local_row,
          col,
          num_sources + source,
          ComputeGeomConsistencyCost(camera,
                                     poses_texture,
                                     depths_texture,
                                     row,
                                     col,
                                     random_depth,
                                     source,
                                     options.geom_consistency_max_cost));
    }
  }
}

Mat<float> CopyActiveBand(const GpuMatView<float> view,
                          const size_t width,
                          const size_t rows) {
  Mat<float> result(width, rows, view.GetDepth());
  for (size_t plane = 0; plane < view.GetDepth(); ++plane) {
    const auto* source = reinterpret_cast<const char*>(view.GetPtr()) +
                         plane * view.GetHeight() * view.pitch;
    CUDA_SAFE_CALL(cudaMemcpy2D(result.GetPtr() + plane * width * rows,
                                width * sizeof(float),
                                source,
                                view.pitch,
                                width * sizeof(float),
                                rows,
                                cudaMemcpyDeviceToHost));
  }
  return result;
}

__global__ void ConstructorContinuation(GpuMatView<curandState> states,
                                        GpuMatView<uint32_t> draws) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= states.GetWidth()) return;
  curandState state = states.Get(0, index);
  for (int i = 0; i < 3; ++i) draws.Set(0, index, i, curand(&state));
}

__global__ void SeedOwnershipTest(GpuMatView<curandState> states,
                                  const PatchMatchRngPerimeter layout,
                                  const unsigned long long seed) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= layout.Count()) return;
  size_t row, col;
  layout.Coordinates(index, &row, &col);
  // Preserve the private probe's original dense row-major sequence identity.
  curand_init(seed, row * layout.width + col, 0, &states.GetRef(0, index));
}
__global__ void SweepContinuation(GpuMatView<curandState> states,
                                  const PatchMatchRngPerimeter layout,
                                  const unsigned int rotation,
                                  GpuMatView<uint32_t> draws) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < draws.GetWidth()) {
    curandState state = states.Get(0, layout.SweepIndex(rotation, col));
    for (int i = 0; i < 3; ++i) draws.Set(0, col, i, curand(&state));
  }
}
}  // namespace

struct PatchMatchCudaTestAccess {
  static FreshProposalAliasComparison FreshProposalAliases(PatchMatchCuda& pm) {
    THROW_CHECK(pm.options_.geom_consistency);
    THROW_CHECK_EQ(pm.rotation_in_half_pi_, 0);
    THROW_CHECK_EQ(pm.options_.window_radius, 5);
    THROW_CHECK_EQ(pm.options_.window_step, 1);
    constexpr int first_row = 7;
    constexpr int rows = 9;  // Two producer stripes, with a one-row tail.
    const size_t width = pm.depth_map_->GetWidth();
    const size_t sources = pm.cost_map_->GetDepth();
    THROW_CHECK_GE(pm.depth_map_->GetHeight(), first_row + rows);
    auto workspace = MakeLookAheadWorkspace(
        pm.ref_width_,
        pm.ref_height_,
        sources,
        std::max(pm.options_.num_samples, static_cast<int>(sources)),
        true);
    THROW_CHECK_NOTNULL(workspace.get());
    THROW_CHECK_GE(workspace->Rows(), rows);
    LookAheadViews band = workspace->Views(0, first_row, rows);
    const auto current_depth = pm.depth_map_->CopyToMat();
    const auto current_normal = pm.normal_map_->CopyToMat();
    Mat<float> random(width, rows, 4);
    Mat<int> aliases(width, rows, 1);
    for (int local_row = 0; local_row < rows; ++local_row) {
      for (size_t col = 0; col < width; ++col) {
        const int row = first_row + local_row;
        const int pattern = (local_row + col) % 4;
        const bool same_depth = pattern == 0 || pattern == 1;
        const bool same_normal = pattern == 0 || pattern == 2;
        const float depth = current_depth.Get(row, col);
        random.Set(local_row, col, 0, same_depth ? depth : 1.15f * depth + .1f);
        float normal[3];
        for (int axis = 0; axis < 3; ++axis) {
          normal[axis] = current_normal.Get(row, col, axis);
        }
        if (!same_normal) {
          normal[0] += .07f;
          normal[1] -= .04f;
          const float inverse =
              1.0f / std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                               normal[2] * normal[2]);
          for (float& value : normal) value *= inverse;
        }
        for (int axis = 0; axis < 3; ++axis)
          random.Set(local_row, col, axis + 1, normal[axis]);
        const auto same_bits = [](const float lhs, const float rhs) {
          return std::memcmp(&lhs, &rhs, sizeof(float)) == 0;
        };
        const bool equal_depth = same_bits(random.Get(local_row, col, 0),
                                           current_depth.Get(row, col));
        bool equal_normal = true;
        for (int axis = 0; axis < 3; ++axis) {
          equal_normal &= same_bits(random.Get(local_row, col, axis + 1),
                                    current_normal.Get(row, col, axis));
        }
        THROW_CHECK_EQ(equal_depth, same_depth);
        THROW_CHECK_EQ(equal_normal, same_normal);
        aliases.Set(local_row, col, int(equal_depth) + 2 * int(equal_normal));
      }
    }
    for (size_t plane = 0; plane < 4; ++plane) {
      auto* destination =
          reinterpret_cast<char*>(band.random_parameters.GetPtr()) +
          plane * band.random_parameters.GetHeight() *
              band.random_parameters.pitch;
      CUDA_SAFE_CALL(cudaMemcpy2D(destination,
                                  band.random_parameters.pitch,
                                  random.GetPtr() + plane * width * rows,
                                  width * sizeof(float),
                                  width * sizeof(float),
                                  rows,
                                  cudaMemcpyHostToDevice));
    }
    // Missing h0 alias publication must leave a visibly invalid destination.
    for (auto view : {band.costs, band.geometric_costs}) {
      CUDA_SAFE_CALL(cudaMemset2D(view.GetPtr(),
                                  view.pitch,
                                  0xff,
                                  view.GetWidth() * sizeof(float),
                                  view.GetHeight() * view.GetDepth()));
    }
    // A freshly evaluated current tuple must not be substituted with an
    // incoming incumbent cost just because its parameters happen to match.
    constexpr float incoming_cost = 1.75f;
    pm.cost_map_->FillWithScalar(incoming_cost);
    GpuMat<float> reference(width, rows, 3 * sources);
    GpuMat<float> reference_geometry(width, rows, 2 * sources);
    SweepOptions options;
    options.sigma_spatial = pm.options_.sigma_spatial;
    options.sigma_color = pm.options_.sigma_color;
    options.geom_consistency_max_cost = pm.options_.geom_consistency_max_cost;
    const RefCamera camera =
        MakeRefCamera(pm.ref_K_host_[0], pm.ref_inv_K_host_[0]);
    PrecomputeBandCosts<true>
        <<<dim3((width + 7) / 8,
                (rows + kLookAheadStripeRows - 1) / kLookAheadStripeRows),
           dim3(8, 12)>>>(camera,
                          pm.depth_map_->View(),
                          pm.normal_map_->View(),
                          pm.ref_image_texture_->GetObj(),
                          pm.ReferenceStatistics(),
                          pm.src_images_texture_->GetObj(),
                          pm.src_depth_maps_texture_->GetObj(),
                          pm.poses_texture_[0]->GetObj(),
                          options,
                          band);
    ComputeFreshProposalReference<<<(width + 31) / 32, 32>>>(
        camera,
        pm.depth_map_->View(),
        pm.normal_map_->View(),
        pm.ref_image_texture_->GetObj(),
        pm.ReferenceStatistics(),
        pm.src_images_texture_->GetObj(),
        pm.src_depth_maps_texture_->GetObj(),
        pm.poses_texture_[0]->GetObj(),
        options,
        band,
        reference.View(),
        reference_geometry.View());
    CUDA_SYNC_AND_CHECK();
    return {CopyActiveBand(band.costs, width, rows),
            reference.CopyToMat(),
            CopyActiveBand(band.geometric_costs, width, rows),
            reference_geometry.CopyToMat(),
            std::move(aliases),
            incoming_cost};
  }

  template <int kWindowSize>
  static NccCostComparison NccCostsForWindow(PatchMatchCuda& pm) {
    pm.ComputeCudaConfig();
    const size_t width = pm.cost_map_->GetWidth();
    const size_t height = pm.cost_map_->GetHeight();
    const size_t sources = pm.cost_map_->GetDepth();
    GpuMat<float> candidate(width, height, sources);
    GpuMat<float> reference(width, height, sources);
    dim3 grid = pm.sweep_grid_size_;
    grid.y = (height + kInitialCostStripeRows - 1) / kInitialCostStripeRows;
#define COMPUTE_NCC_COST(REFERENCE, OUTPUT)                           \
  ComputeInitialCost<kWindowSize, 1, REFERENCE>                       \
      <<<grid, pm.sweep_block_size_>>>(                               \
          MakeRefCamera(pm.ref_K_host_[pm.rotation_in_half_pi_],      \
                        pm.ref_inv_K_host_[pm.rotation_in_half_pi_]), \
          OUTPUT.View(),                                              \
          pm.depth_map_->View(),                                      \
          pm.normal_map_->View(),                                     \
          pm.ref_image_texture_->GetObj(),                            \
          pm.ReferenceStatistics(),                                  \
          pm.src_images_texture_->GetObj(),                           \
          pm.poses_texture_[pm.rotation_in_half_pi_]->GetObj(),       \
          pm.options_.sigma_spatial,                                  \
          pm.options_.sigma_color)
    COMPUTE_NCC_COST(false, candidate);
    COMPUTE_NCC_COST(true, reference);
#undef COMPUTE_NCC_COST
    CUDA_SYNC_AND_CHECK();
    return {candidate.CopyToMat(), reference.CopyToMat()};
  }

  static NccCostComparison NccCosts(PatchMatchCuda& pm) {
    THROW_CHECK_EQ(pm.options_.window_step, 1);
    if (pm.options_.window_radius == 5) return NccCostsForWindow<11>(pm);
    THROW_CHECK_EQ(pm.options_.window_radius, 2);
    return NccCostsForWindow<5>(pm);
  }

  static void LegacyReferenceStatistics(PatchMatchCuda& pm) {
    THROW_CHECK_EQ(pm.rotation_in_half_pi_, 0);
    const auto& bitmap =
        pm.problem_.images->at(pm.problem_.ref_image_idx).GetBitmap();
    GpuMatRefImage filtered(pm.ref_width_, pm.ref_height_);
    filtered.Filter(bitmap.RowMajorData().data(),
                    pm.options_.window_radius,
                    pm.options_.window_step,
                    pm.options_.sigma_spatial,
                    pm.options_.sigma_color);
    pm.ref_sum_image_ = std::move(filtered.sum_image);
    pm.ref_squared_sum_image_ = std::move(filtered.squared_sum_image);
    pm.ref_statistics_texture_.reset();
  }

  static Mat<uint32_t> ConstructorState(PatchMatchCuda& pm) {
    const size_t count = pm.rand_state_map_->GetWidth();
    GpuMat<uint32_t> draws(count, 1, 3);
    ConstructorContinuation<<<(count + 255) / 256, 256>>>(
        pm.rand_state_map_->View(), draws.View());
    CUDA_SYNC_AND_CHECK();
    return draws.CopyToMat();
  }

  static Mat<float> InitialCost(PatchMatchCuda& pm,
                                const unsigned long long seed) {
    const PatchMatchRngPerimeter layout{pm.ref_width_, pm.ref_height_};
    SeedOwnershipTest<<<(layout.Count() + 255) / 256, 256>>>(
        pm.rand_state_map_->View(), layout, seed);
    pm.ComputeCudaConfig();
    dim3 cost_grid = pm.sweep_grid_size_;
    cost_grid.y = (pm.cost_map_->GetHeight() + kInitialCostStripeRows - 1) /
                  kInitialCostStripeRows;
    ComputeInitialCost<11, 1><<<cost_grid, pm.sweep_block_size_>>>(
        MakeRefCamera(pm.ref_K_host_[pm.rotation_in_half_pi_],
                      pm.ref_inv_K_host_[pm.rotation_in_half_pi_]),
        pm.cost_map_->View(),
        pm.depth_map_->View(),
        pm.normal_map_->View(),
        pm.ref_image_texture_->GetObj(),
        pm.ReferenceStatistics(),
        pm.src_images_texture_->GetObj(),
        pm.poses_texture_[pm.rotation_in_half_pi_]->GetObj(),
        pm.options_.sigma_spatial,
        pm.options_.sigma_color);
    CUDA_SYNC_AND_CHECK();
    return pm.cost_map_->CopyToMat();
  }

  static SweepTraceResult Sweep(PatchMatchCuda& pm,
                                const bool cooperative,
                                const unsigned long long seed,
                                const bool initialize = true,
                                const int sequence_step = 0) {
    pm.consistency_mask_valid_ = false;
    std::unique_ptr<LookAheadWorkspace> look_ahead_workspace;
    if (cooperative) {
      look_ahead_workspace = MakeLookAheadWorkspace(
          pm.ref_width_,
          pm.ref_height_,
          pm.cost_map_->GetDepth(),
          pm.options_.num_samples,
          true);  // The existing private probe instantiates the GEO term.
    }
    const Mat<float> initial_cost =
        initialize ? InitialCost(pm, seed) : pm.cost_map_->CopyToMat();
    const size_t width = pm.cost_map_->GetWidth();
    const size_t height = pm.cost_map_->GetHeight();
    const size_t sources = pm.cost_map_->GetDepth();
    GpuMat<int> draws(width, height, pm.options_.num_samples);
    GpuMat<int> winners(width, height);
    GpuMat<int> support(width, height);
    GpuMat<float> hypotheses(width, height, 20);
    GpuMat<float> costs(width, height, 5);
    GpuMat<float> forward(width, height, sources);
    GpuMat<uint32_t> continuation(width, 1, 3);
    GpuMat<uint32_t> continuation_before(width, 1, 3);
    GpuMat<uint32_t> reference_statistics(width, height, 2);
    CaptureReferenceStatistics<<<dim3((width + 31) / 32, (height + 15) / 16),
                                 dim3(32, 16)>>>(
        pm.ReferenceStatistics(), reference_statistics.View());
    SweepContinuation<<<(width + 31) / 32, 32>>>(
        pm.rand_state_map_->View(),
        PatchMatchRngPerimeter{pm.ref_width_, pm.ref_height_},
        static_cast<unsigned int>(pm.rotation_in_half_pi_),
        continuation_before.View());
    const SweepTrace trace{draws.View(),
                           winners.View(),
                           support.View(),
                           hypotheses.View(),
                           costs.View(),
                           forward.View()};
    SweepOptions options;
    // One iteration's production perturbation/previous-probability schedule.
    options.perturbation = 1.0f / std::pow(2.0f, sequence_step / 4.0f);
    options.prev_sel_prob_weight = static_cast<float>(sequence_step) / 4.0f;
    options.depth_min = pm.options_.depth_min;
    options.depth_max = pm.options_.depth_max;
    options.sigma_spatial = pm.options_.sigma_spatial;
    options.sigma_color = pm.options_.sigma_color;
    options.num_samples = pm.options_.num_samples;
    options.ncc_sigma = pm.options_.ncc_sigma;
    options.min_triangulation_angle =
        DEG2RAD(pm.options_.min_triangulation_angle);
    options.incident_angle_sigma = pm.options_.incident_angle_sigma;
    options.geom_consistency_regularizer =
        pm.options_.geom_consistency_regularizer;
    options.geom_consistency_max_cost = pm.options_.geom_consistency_max_cost;
    options.filter_min_ncc = pm.options_.filter_min_ncc;
    options.filter_min_triangulation_angle =
        DEG2RAD(pm.options_.filter_min_triangulation_angle);
    options.filter_min_num_consistent = pm.options_.filter_min_num_consistent;
    options.filter_geom_consistency_max_cost =
        pm.options_.filter_geom_consistency_max_cost;
    if (pm.options_.filter) {
      pm.consistency_mask_ = std::make_unique<GpuMat<uint8_t>>(
          width, height, PackedConsistencyMaskDepth(sources));
      pm.consistency_mask_->FillWithScalar(0);
    }
    const RefCamera camera =
        MakeRefCamera(pm.ref_K_host_[pm.rotation_in_half_pi_],
                      pm.ref_inv_K_host_[pm.rotation_in_half_pi_]);

    // Use the production dispatcher so eligible traces exercise every band.
    // Only the first sequence sweep seeds states and recomputes initial NCC.
    // Later sweeps consume the actual rotated maps and shared corner histories.
#define LAUNCH_TEST_SWEEP(COOPERATIVE, WIDTH, FILTER)                 \
  LaunchSweep<11, 1, true, FILTER, FILTER, COOPERATIVE, WIDTH, true>( \
      look_ahead_workspace.get(),                                     \
      camera,                                                         \
      pm.global_workspace_->View(),                                   \
      pm.rand_state_map_->View(),                                     \
      (PatchMatchRngPerimeter{pm.ref_width_, pm.ref_height_}),        \
      static_cast<unsigned int>(pm.rotation_in_half_pi_),             \
      pm.cost_map_->View(),                                           \
      pm.depth_map_->View(),                                          \
      pm.normal_map_->View(),                                         \
      PackedConsistencyMaskView{pm.consistency_mask_->View()},        \
      pm.sel_prob_map_->View(),                                       \
      pm.prev_sel_prob_map_->View(),                                  \
      pm.ref_image_texture_->GetObj(),                                \
      pm.ReferenceStatistics(),                       \
      pm.src_images_texture_->GetObj(),                               \
      pm.src_depth_maps_texture_->GetObj(),                           \
      pm.poses_texture_[pm.rotation_in_half_pi_]->GetObj(),           \
      options,                                                        \
      trace)
    if (cooperative) {
      if (pm.options_.filter) {
        LAUNCH_TEST_SWEEP(true, 8, true);
      } else {
        LAUNCH_TEST_SWEEP(true, 8, false);
      }
    } else {
      if (pm.options_.filter) {
        LAUNCH_TEST_SWEEP(false, THREADS_PER_BLOCK, true);
      } else {
        LAUNCH_TEST_SWEEP(false, THREADS_PER_BLOCK, false);
      }
    }
#undef LAUNCH_TEST_SWEEP
    CUDA_SYNC_AND_CHECK();
    SweepContinuation<<<(width + 31) / 32, 32>>>(
        pm.rand_state_map_->View(),
        PatchMatchRngPerimeter{pm.ref_width_, pm.ref_height_},
        static_cast<unsigned int>(pm.rotation_in_half_pi_),
        continuation.View());
    CUDA_SYNC_AND_CHECK();
    pm.consistency_mask_valid_ = pm.options_.filter;
    return {draws.CopyToMat(),
            winners.CopyToMat(),
            support.CopyToMat(),
            hypotheses.CopyToMat(),
            costs.CopyToMat(),
            forward.CopyToMat(),
            initial_cost,
            pm.cost_map_->CopyToMat(),
            pm.depth_map_->CopyToMat(),
            pm.normal_map_->CopyToMat(),
            pm.sel_prob_map_->CopyToMat(),
            continuation.CopyToMat(),
            continuation_before.CopyToMat(),
            reference_statistics.CopyToMat(),
            pm.ref_statistics_texture_ != nullptr};
  }

  static std::vector<SweepTraceResult> SweepSequence(
      PatchMatchCuda& pm,
      const bool cooperative,
      const unsigned long long seed) {
    THROW_CHECK(!pm.options_.filter);
    THROW_CHECK_EQ(pm.rotation_in_half_pi_, 0);
    std::vector<SweepTraceResult> result;
    result.reserve(4);
    for (int step = 0; step < 4; ++step) {
      result.push_back(Sweep(pm, cooperative, seed, step == 0, step));
      // Sweep synchronizes before any map/view is released or rotated.
      pm.Rotate();
    }
    return result;
  }
};

NccCostComparison ProbeNccCosts(PatchMatchCuda& pm) {
  return PatchMatchCudaTestAccess::NccCosts(pm);
}

FreshProposalAliasComparison ProbeFreshProposalAliases(PatchMatchCuda& pm) {
  return PatchMatchCudaTestAccess::FreshProposalAliases(pm);
}

Mat<uint32_t> ProbeConstructorContinuation(PatchMatchCuda& pm) {
  return PatchMatchCudaTestAccess::ConstructorState(pm);
}

SweepTraceResult ProbeSweep(PatchMatchCuda& pm,
                            const bool cooperative,
                            const unsigned long long seed) {
  return PatchMatchCudaTestAccess::Sweep(pm, cooperative, seed);
}

std::vector<SweepTraceResult> ProbeSweepSequence(
    PatchMatchCuda& pm, const bool cooperative, const unsigned long long seed) {
  return PatchMatchCudaTestAccess::SweepSequence(pm, cooperative, seed);
}

void UseLegacyReferenceStatistics(PatchMatchCuda& pm) {
  PatchMatchCudaTestAccess::LegacyReferenceStatistics(pm);
}

Mat<float> ProbeOwnershipInitialCost(PatchMatchCuda& pm,
                                     const unsigned long long seed) {
  return PatchMatchCudaTestAccess::InitialCost(pm, seed);
}
}  // namespace internal
#endif  // COLMAP_PATCH_MATCH_TESTING

}  // namespace mvs
}  // namespace colmap
