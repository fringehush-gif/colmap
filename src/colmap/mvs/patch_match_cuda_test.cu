// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/patch_match_cuda_test.h"
#include "colmap/mvs/patch_match_geometry.cuh"
#include "colmap/mvs/patch_match_sampling.cuh"
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <array>
#include <limits>
#include <utility>
#include <vector>
namespace colmap {
namespace mvs {
namespace {
// Retained from exact public 020cbb9e: gpu_mat_prng.cu's initializer,
// gpu_mat.h's float fill, and patch_match_cuda.cu's normal initialization.
// Keep this legacy arm independent of GpuMatPRNGSeed, DrawUniformInterval,
// InitDepthNormalMaps and GenerateRandomNormal in the candidate owner.
__global__ void LegacyConstructorSeed(GpuMatView<curandState> output) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t uniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
  const size_t id = uniqueBlockIndex * blockDim.y * blockDim.x +
                    threadIdx.y * blockDim.x + threadIdx.x;
  if (col < output.GetWidth() && row < output.GetHeight()) {
    curand_init(id, 0, 0, &output.GetRef(row, col));
  }
}

__global__ void LegacyConstructorDepth(GpuMatView<float> depth,
                                       GpuMatView<curandState> states,
                                       const float depth_min,
                                       const float depth_max) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < depth.GetHeight() && col < depth.GetWidth()) {
    curandState state = states.Get(row, col);
    const float value =
        curand_uniform(&state) * (depth_max - depth_min) + depth_min;
    depth.Set(row, col, value);
    states.Set(row, col, state);
  }
}

__global__ void LegacyConstructorNormal(const float4 inv_K,
                                        GpuMatView<float> normals,
                                        GpuMatView<curandState> states,
                                        GpuMatView<int> normal_pairs) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col >= normals.GetWidth() || row >= normals.GetHeight()) return;
  curandState state = states.Get(row, col);
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  int pairs = 0;
  while (s >= 1.0f) {
    v1 = 2.0f * curand_uniform(&state) - 1.0f;
    v2 = 2.0f * curand_uniform(&state) - 1.0f;
    s = v1 * v1 + v2 * v2;
    ++pairs;  // Observation only; it does not consume or reorder a draw.
  }
  const float s_norm = sqrt(1.0f - s);
  float normal[3];
  normal[0] = 2.0f * v1 * s_norm;
  normal[1] = 2.0f * v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;
  const float view_ray[3] = {
      inv_K.x * col + inv_K.y, inv_K.z * row + inv_K.w, 1.0f};
  if (normal[0] * view_ray[0] + normal[1] * view_ray[1] +
          normal[2] * view_ray[2] >
      0) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
  normals.SetSlice(row, col, normal);
  normal_pairs.Set(row, col, pairs);
  states.Set(row, col, state);
}

__global__ void LegacyConstructorContinuation(GpuMatView<curandState> states,
                                              GpuMatView<uint32_t> draws) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= states.GetHeight() || col >= states.GetWidth()) return;
  curandState state = states.Get(row, col);
  for (int i = 0; i < 3; ++i) draws.Set(row, col, i, curand(&state));
}

__global__ void ProbeSamplingSupportKernel(int* errors) {
  int failures = 0;
  const int strides[] = {1, 8, 23, 32, 47};
  for (int test = 0; test < 5; ++test) {
    const int stride = strides[test];
    float actual[3 * 47];
    float expected[3 * 47];
    for (int i = 0; i < 3 * 47; ++i) actual[i] = expected[i] = -17.0f;
    actual[0] = actual[stride] = actual[2 * stride] = 0.0f;
    failures += TransformPDFToCDF(actual, 3, stride);
    failures += actual[0] != 0.0f || actual[stride] != 0.0f ||
                actual[2 * stride] != 0.0f;
    failures += SampleSource(actual, 3, stride, FLT_EPSILON / 2) != -1;
    actual[stride] = 1.0f;
    actual[2 * stride] = 3.0f;
    for (int i = 0; i < 3 * 47; ++i) expected[i] = actual[i];
    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) sum += expected[i * stride];
    const float inverse = 1.0f / sum;
    float cumulative = 0.0f;
    for (int i = 0; i < 3; ++i) {
      const float probability = expected[i * stride] * inverse;
      cumulative += probability;
      expected[i * stride] = cumulative;
    }
    failures += !TransformPDFToCDF(actual, 3, stride);
    for (int i = 0; i < 3 * 47; ++i) {
      failures += __float_as_uint(actual[i]) != __float_as_uint(expected[i]);
    }
    failures += SampleSource(actual, 3, stride, FLT_EPSILON / 2) != 1;
    failures += SampleSource(actual, 3, stride, .5f) != 2;
    failures += SampleSource(actual, 3, stride, 1.0f) != 2;
    actual[0] = actual[stride] = actual[2 * stride] = 0.0f;
    failures += TransformPDFToCDF(actual, 3, stride);
    const int windows[] = {1, 16};
    const int sample_counts[] = {1, 17, 19};
    for (int window_idx = 0; window_idx < 2; ++window_idx) {
      const int window = windows[window_idx];
      for (int sample_idx = 0; sample_idx < 3; ++sample_idx) {
        const int samples = sample_counts[sample_idx];
        curandState state;
        curandState reference;
        curand_init(9182, 0, 0, &state);
        curand_init(9182, 0, 0, &reference);
        for (int begin = 0; begin < samples; begin += window) {
          const int count = min(window, samples - begin);
          for (int slot = 0; slot < window; ++slot) {
            if (slot < count) {
              const float uniform = curand_uniform(&state);
              failures += SampleSource(actual, 3, stride, uniform) != -1;
            }
          }
        }
        for (int i = 0; i < samples; ++i) (void)curand_uniform(&reference);
        for (int i = 0; i < 3; ++i)
          failures += curand(&state) != curand(&reference);
      }
    }
  }
  *errors = failures;
}

__global__ void ProbePropagatedDepthKernel(const float4 inv_K,
                                           const float depth,
                                           const float3 normal,
                                           const float row1,
                                           const float row2,
                                           const float col,
                                           float* output) {
  const float n[3] = {normal.x, normal.y, normal.z};
  const float calibration[4] = {inv_K.x, inv_K.y, inv_K.z, inv_K.w};
  *output = internal::PropagateDepth(calibration, depth, n, row1, row2, col);
}

template <bool kCooperative, int kSampleWindowSize>
__global__ void ProbeSampledSourcesKernel(const GpuMatView<int> draws,
                                          const GpuMatView<float> values,
                                          GpuMatView<int> restored,
                                          GpuMatView<float> folded) {
  constexpr int kLanes = 8;
  const int lane = threadIdx.x;
  float sum = 0.0f;
  if constexpr (!kCooperative) {
    for (int sample = 0; sample < draws.width; ++sample) {
      const int source = draws.Get(lane, sample);
      restored.Set(lane, sample, source);
      if (source >= 0) sum += values.Get(0, source);
    }
  } else {
    __shared__ int sources[kSampleWindowSize][kLanes];
    __shared__ int aliases[kSampleWindowSize][kLanes];
    __shared__ float results[kSampleWindowSize][kLanes];
    for (int begin = 0; begin < draws.width; begin += kSampleWindowSize) {
      const int sample_count =
          min(kSampleWindowSize, static_cast<int>(draws.width) - begin);
      if constexpr (kSampleWindowSize > 1) {
        int num_unique_sources = 0;
        uint32_t sampled_sources = 0;
        const bool use_source_mask = values.width <= 32;
        for (int slot = 0; slot < kSampleWindowSize; ++slot) {
          const int source =
              slot < sample_count ? draws.Get(lane, begin + slot) : -1;
          aliases[slot][lane] =
              internal::AppendSampledSource(source,
                                            use_source_mask,
                                            sampled_sources,
                                            num_unique_sources,
                                            sources,
                                            lane);
        }
        internal::PackSampledSources<kSampleWindowSize>(use_source_mask,
                                                        sampled_sources,
                                                        num_unique_sources,
                                                        sample_count,
                                                        sources,
                                                        aliases,
                                                        lane);
      } else {
        sources[0][lane] = draws.Get(lane, begin);
      }
      for (int slot = 0; slot < kSampleWindowSize; ++slot) {
        const int source = sources[slot][lane];
        if (source >= 0) results[slot][lane] = values.Get(0, source);
      }
      for (int slot = 0; slot < sample_count; ++slot) {
        const int job = kSampleWindowSize > 1 ? aliases[slot][lane] : slot;
        const int source = job >= 0 ? sources[job][lane] : -1;
        restored.Set(lane, begin + slot, source);
        if (source >= 0) sum += results[job][lane];
      }
    }
  }
  folded.Set(0, lane, sum);
}

}  // namespace
namespace internal {
PhotoInitializationResult ProbeLegacyPhotoInitialization(
    const int width,
    const int height,
    const std::array<float, 4>& inv_K,
    const float depth_min,
    const float depth_max) {
  GpuMat<curandState> states(width, height);
  GpuMat<float> depth(width, height);
  GpuMat<float> normal(width, height, 3);
  GpuMat<uint32_t> continuation(width, height, 3);
  GpuMat<int> normal_pairs(width, height);
  // Original GpuMat PRNG/fill shape and PatchMatch element-wise shape differ.
  const dim3 map_block(32, 16);
  const dim3 map_grid((width + 31) / 32, (height + 15) / 16);
  LegacyConstructorSeed<<<map_grid, map_block>>>(states.View());
  LegacyConstructorDepth<<<map_grid, map_block>>>(
      depth.View(), states.View(), depth_min, depth_max);
  const dim3 normal_block(32, 32);
  const dim3 normal_grid((width + 31) / 32, (height + 31) / 32);
  LegacyConstructorNormal<<<normal_grid, normal_block>>>(
      make_float4(inv_K[0], inv_K[1], inv_K[2], inv_K[3]),
      normal.View(),
      states.View(),
      normal_pairs.View());
  LegacyConstructorContinuation<<<map_grid, map_block>>>(states.View(),
                                                         continuation.View());
  CUDA_SYNC_AND_CHECK();
  return {depth.CopyToMat(),
          normal.CopyToMat(),
          continuation.CopyToMat(),
          normal_pairs.CopyToMat()};
}

int ProbeSamplingSupport() {
  GpuMat<int> errors(1, 1);
  ProbeSamplingSupportKernel<<<1, 1>>>(errors.GetPtr());
  CUDA_SYNC_AND_CHECK();
  return errors.CopyToMat().GetData()[0];
}

float ProbePropagatedDepth(const std::array<float, 4>& K,
                           const float depth,
                           const std::array<float, 3>& normal,
                           const float row1,
                           const float row2,
                           const float col) {
  const float inv_K[4] = {1 / K[0], -K[1] / K[0], 1 / K[2], -K[3] / K[2]};
  GpuMat<float> output(1, 1);
  ProbePropagatedDepthKernel<<<1, 1>>>(
      make_float4(inv_K[0], inv_K[1], inv_K[2], inv_K[3]),
      depth,
      make_float3(normal[0], normal[1], normal[2]),
      row1,
      row2,
      col,
      output.GetPtr());
  CUDA_SYNC_AND_CHECK();
  return output.CopyToMat().GetData()[0];
}

// Private test bridge; no public option or header API. The scalar path folds
// raw draws directly, while cooperative paths use the production pack helpers.
std::pair<std::vector<int>, std::vector<float>> ProbeSampledSources(
    const std::vector<int>& draws,
    const std::vector<float>& values,
    const int sample_window_size,
    const bool cooperative) {
  constexpr int kLanes = 8;
  THROW_CHECK(!draws.empty() && draws.size() % kLanes == 0);
  THROW_CHECK(!values.empty());
  THROW_CHECK_LE(draws.size(),
                 static_cast<size_t>(std::numeric_limits<int>::max()));
  THROW_CHECK_LE(values.size(),
                 static_cast<size_t>(std::numeric_limits<int>::max()));
  THROW_CHECK(sample_window_size == 1 || sample_window_size == 16);
  for (const int source : draws) {
    THROW_CHECK_GE(source, -1);
    THROW_CHECK(source < 0 || static_cast<size_t>(source) < values.size());
  }
  const size_t count = draws.size() / kLanes;
  GpuMat<int> device_draws(count, kLanes);
  GpuMat<float> device_values(values.size(), 1);
  GpuMat<int> restored(count, kLanes);
  GpuMat<float> folded(kLanes, 1);
  CUDA_SAFE_CALL(cudaMemcpy2D(device_draws.GetPtr(),
                              device_draws.GetPitch(),
                              draws.data(),
                              count * sizeof(int),
                              count * sizeof(int),
                              kLanes,
                              cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(device_values.GetPtr(),
                            values.data(),
                            values.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  if (!cooperative) {
    ProbeSampledSourcesKernel<false, 1><<<1, kLanes>>>(device_draws.View(),
                                                       device_values.View(),
                                                       restored.View(),
                                                       folded.View());
  } else if (sample_window_size == 1) {
    ProbeSampledSourcesKernel<true, 1><<<1, kLanes>>>(device_draws.View(),
                                                      device_values.View(),
                                                      restored.View(),
                                                      folded.View());
  } else {
    ProbeSampledSourcesKernel<true, 16><<<1, kLanes>>>(device_draws.View(),
                                                       device_values.View(),
                                                       restored.View(),
                                                       folded.View());
  }
  CUDA_SYNC_AND_CHECK();
  return {restored.CopyToMat().GetData(), folded.CopyToMat().GetData()};
}

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
