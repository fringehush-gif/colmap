// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Included in patch_match_cuda.cu after SweepOptions.
// Experimental weighted checkerboard search; native remains the default.

#include "experimental_acmh_cost.cuh"

namespace experimental_acmh {

// Frozen HMM posterior: source chains run independently. For the checkpoint
// path, only one existing sqrt(height)-row stripe and its boundaries are live.
// Previous probabilities can alias output because each is read before write.
template <bool kCheckpoint>
__global__ void FreezeHmm(GpuMatView<float> costs,
                          GpuMatView<float> posterior,
                          GpuMatView<float> previous,
                          GpuMatView<float> boundaries,
                          GpuMatView<float> stripe,
                          SweepOptions options) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int s = blockIdx.y;
  if (c >= costs.GetWidth()) return;
  LikelihoodComputer likelihood(options.ncc_sigma,
                                options.min_triangulation_angle,
                                options.incident_angle_sigma);
  const int h = costs.GetHeight();
  const int rows = kCheckpoint ? stripe.GetHeight() : h;
  float beta = 0.5f;
  for (int r = h - 1; r >= 0; --r) {
    beta = likelihood.ComputeBackwardMessage(costs.Get(r, c, s), beta);
    if constexpr (kCheckpoint) {
      if (r % rows == 0) boundaries.Set(r / rows, c, s, beta);
    } else {
      posterior.Set(r, c, s, beta);
    }
  }
  float alpha = 0.5f;
  for (int start = 0; start < h; start += rows) {
    const int end = min(h, start + rows);
    if constexpr (kCheckpoint) {
      beta = end < h ? boundaries.Get(end / rows, c, s) : 0.5f;
      for (int r = end - 1; r >= start; --r) {
        beta = likelihood.ComputeBackwardMessage(costs.Get(r, c, s), beta);
        stripe.Set(r - start, c, s, beta);
      }
    }
    for (int r = start; r < end; ++r) {
      alpha = likelihood.ComputeForwardMessage(costs.Get(r, c, s), alpha);
      const float b =
          kCheckpoint ? stripe.Get(r - start, c, s) : posterior.Get(r, c, s);
      posterior.Set(
          r,
          c,
          s,
          likelihood.ComputeSelProb(
              alpha, b, previous.Get(r, c, s), options.prev_sel_prob_weight));
    }
  }
}

struct Plane {
  float depth;
  float normal[3];
};

__device__ inline Plane ReadPlane(GpuMatView<float> depths,
                                  GpuMatView<float> normals,
                                  int r,
                                  int c) {
  Plane p;
  p.depth = depths.Get(r, c);
  normals.GetSlice(r, c, p.normal);
  return p;
}

// Plane n.X = d*n.ray(donor). Depth is camera z, never Euclidean range.
__device__ inline Plane Transfer(const RefCamera& camera,
                                 Plane donor,
                                 int dr,
                                 int dc,
                                 int r,
                                 int c,
                                 const SweepOptions& options,
                                 const Plane& fallback) {
  const float a[3] = {camera.inv_K[0] * dc + camera.inv_K[1],
                      camera.inv_K[2] * dr + camera.inv_K[3],
                      1.0f};
  const float b[3] = {camera.inv_K[0] * c + camera.inv_K[1],
                      camera.inv_K[2] * r + camera.inv_K[3],
                      1.0f};
  const float denominator = DotProduct3(donor.normal, b);
  if (!(denominator < -1e-5f)) return fallback;
  donor.depth *= DotProduct3(donor.normal, a) / denominator;
  if (!isfinite(donor.depth) || donor.depth < options.depth_min ||
      donor.depth > options.depth_max)
    return fallback;
  return donor;
}

// Counter mixing seeds the per-pixel curand state for plane refinement
// and random escape; no persistent per-pixel RNG allocation is needed.
__device__ inline unsigned int Mix(unsigned int x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

// The first checkerboard color has no preceding joint-view winner. Bootstrap
// its donor rank from the initial ordered NCC tensor, normalized to the
// later per-view winning-score scale.
__global__ void InitialDonorMean(GpuMatView<float> costs,
                                 GpuMatView<float> scores) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (c >= costs.GetWidth()) return;
  const int sources = costs.GetDepth();
  if (sources <= 0) {
    scores.Set(r, c, FLT_MAX);
    return;
  }
  float sum = 0.0f;
  for (int s = 0; s < sources; ++s) sum += costs.Get(r, c, s);
  scores.Set(r, c, isfinite(sum) ? sum / sources : FLT_MAX);
}

// One warp owns two rows of sixteen checkerboard pixels each. All lanes
// cooperate in publishing the compact halo before any bounds return. Donors
// remain opposite color, so scheduling order within a color cannot change data.
template <int kWindowSize, int kWindowStep, bool kGeometry>
__global__ void Search(const RefCamera camera,
                       GpuMatView<float> costs,
                       GpuMatView<float> depths,
                       GpuMatView<float> normals,
                       GpuMatView<float> posterior,
                       GpuMatView<float> donor_scores,
                       cudaTextureObject_t ref_texture,
                       GpuMatView<float> ref_sum,
                       GpuMatView<float> ref_squared_sum,
                       cudaTextureObject_t sources,
                       cudaTextureObject_t source_depths,
                       cudaTextureObject_t poses,
                       SweepOptions options,
                       int color,
                       unsigned int step) {
  using Cost = CompactPhotoConsistencyCostComputer<kWindowSize, kWindowStep>;
  constexpr int radius = kWindowSize / 2;
  constexpr int tile_width = Cost::kTileWidth;
  constexpr int tile_rows = Cost::kTileRows;
  const int lane = threadIdx.x;
  const int row_in_tile = lane / 16;
  const int r = 2 * blockIdx.y + row_in_tile;
  const int local_column = 2 * (lane % 16) + ((r + color) & 1);
  const int c = blockIdx.x * 32 + local_column;
  __shared__ float tile[tile_width * tile_rows];
  __shared__ float weights[Cost::kWeightCacheSize];
  for (int i = lane; i < tile_width * tile_rows; i += 32) {
    tile[i] = tex2D<float>(ref_texture,
                           blockIdx.x * 32 - radius + i % tile_width,
                           blockIdx.y * 2 - radius + i / tile_width);
  }
  __syncthreads();
  Cost cost(camera,
            ref_texture,
            sources,
            poses,
            options.sigma_spatial,
            options.sigma_color);
  cost.local_ref_image.data = tile;
  cost.local_weights = weights;
  cost.col = c;
  cost.row = r;
  cost.ref_patch_start = row_in_tile * tile_width + local_column;
  cost.ref_center_index = cost.ref_patch_start + radius * tile_width + radius;
  // ReadWeights has no barrier and only accesses the published tile.
  cost.ReadWeights();
  __syncthreads();
  if (r >= depths.GetHeight() || c >= depths.GetWidth()) return;
  cost.local_ref_sum = ref_sum.Get(r, c);
  cost.local_ref_squared_sum = ref_squared_sum.Get(r, c);
  Plane candidates[15];
  candidates[0] = ReadPlane(depths, normals, r, c);
  // Four seven-site V regions and four eleven-site axial strips. Every
  // offset has odd checkerboard parity, so donors are stable within a color.
  for (int region = 0; region < 8; ++region) {
    float best = FLT_MAX;
    candidates[1 + region] = candidates[0];
    const int n = region < 4 ? 7 : 11;
    for (int site = 0; site < n; ++site) {
      int axial, lateral;
      if (region < 4) {
        axial = site == 0 ? 1 : 2 + (site - 1) / 2;
        lateral = site == 0 ? 0 : ((site & 1) ? -1 : 1) * (axial - 1);
      } else {
        axial = 2 * site + 1;
        lateral = 0;
      }
      const int direction = region & 3;
      const int dr = direction == 0 ? -axial : direction == 1 ? axial : lateral;
      const int dc = direction == 2 ? -axial : direction == 3 ? axial : lateral;
      const int rr = r + dr;
      const int cc = c + dc;
      if (rr < 0 || cc < 0 || rr >= depths.GetHeight() ||
          cc >= depths.GetWidth())
        continue;
      const float score = donor_scores.Get(rr, cc);
      if (score < best) {
        best = score;
        candidates[1 + region] = Transfer(camera,
                                          ReadPlane(depths, normals, rr, cc),
                                          rr,
                                          cc,
                                          r,
                                          c,
                                          options,
                                          candidates[0]);
      }
    }
  }
  const unsigned int pixel =
      static_cast<unsigned int>(r * depths.GetWidth() + c);
  curandState random;
  curand_init(Mix(pixel ^ Mix(step + 1u)), 0, 0, &random);
  candidates[9] = candidates[0];
  candidates[9].depth = min(
      options.depth_max,
      max(options.depth_min,
          PerturbDepth(options.perturbation, candidates[0].depth, &random)));
  PerturbNormal(camera,
                r,
                c,
                options.perturbation * M_PI,
                candidates[0].normal,
                &random,
                candidates[9].normal);
  candidates[10] = candidates[0];
  candidates[10].depth = candidates[9].depth;
  candidates[11] = candidates[9];
  candidates[11].depth = candidates[0].depth;
  // A global escape supplies a fresh depth/normal pair when all propagated
  // planes and local perturbations have inherited a poor initialization.
  candidates[12].depth =
      GenerateRandomDepth(options.depth_min, options.depth_max, &random);
  GenerateRandomNormal(camera, r, c, &random, candidates[12].normal);
  candidates[13] = candidates[0];
  candidates[13].depth = candidates[12].depth;
  candidates[14] = candidates[12];
  candidates[14].depth = candidates[0].depth;

  float best_prior = -1.0f;
  int prior_source = -1;
  for (int s = 0; s < costs.GetDepth(); ++s) {
    const float prior = posterior.Get(r, c, s);
    if (prior > best_prior) {
      best_prior = prior;
      prior_source = s;
    }
  }
  float scores[15] = {0};
  float native_scores[15] = {0};
  float joint_total = 0.0f;
  float native_total = 0.0f;
  float current_point[3];
  ComputePointAtDepth(camera, r, c, candidates[0].depth, current_point);
  LikelihoodComputer likelihood(options.ncc_sigma,
                                options.min_triangulation_angle,
                                options.incident_angle_sigma);
  // Native invokes four directional sweeps per PatchMatch iteration.
  const float iteration = float(step / 4);
  const float tau = 0.8f * expf(-(iteration * iteration) / 90.0f);
  // Jointly choose a source from the eight propagated hypotheses. The
  // selected source weight is shared across all twelve candidates, including
  // incumbent and refinement. A small floor preserves a defined fallback.
  for (int source = 0; source < costs.GetDepth(); ++source) {
    cost.src_image_idx = source;
    float ncc[15];
    ncc[0] = costs.Get(r, c, source);
    int good = 0, bad = 0;
    float good_weight = 0.0f;
    for (int h = 1; h <= 8; ++h) {
      cost.depth = candidates[h].depth;
      cost.normal = candidates[h].normal;
      ncc[h] = cost.Compute();
      if (ncc[h] < tau) {
        ++good;
        good_weight += expf(-(ncc[h] * ncc[h]) / 0.18f);
      }
      if (ncc[h] > 1.2f) ++bad;
    }
    const bool selected = good > 2 && bad < 3;
    float weight = selected ? good_weight / float(good) : 0.02f;
    if (source == prior_source) weight *= selected ? 2.0f : 0.2f;
    joint_total += weight;
    float tri, inc, H[9];
    ComputeViewingAngles(
        poses, current_point, candidates[0].normal, source, &tri, &inc);
    ComposeHomography(camera,
                      poses,
                      source,
                      r,
                      c,
                      candidates[0].depth,
                      candidates[0].normal,
                      H);
    float native_weight =
        posterior.Get(r, c, source) * likelihood.ComputeTriProb(tri) *
        likelihood.ComputeIncProb(inc) *
        likelihood.ComputeResolutionProb<kWindowSize>(H, r, c);
    if (!isfinite(native_weight) || native_weight < 0.0f) native_weight = 0.0f;
    native_total += native_weight;
    for (int h = 9; h < 15; ++h) {
      cost.depth = candidates[h].depth;
      cost.normal = candidates[h].normal;
      ncc[h] = cost.Compute();
    }
    for (int h = 0; h < 15; ++h) {
      float value = ncc[h];
      if constexpr (kGeometry) {
        value += options.geom_consistency_regularizer *
                 ComputeGeomConsistencyCost(camera,
                                            poses,
                                            source_depths,
                                            r,
                                            c,
                                            candidates[h].depth,
                                            source,
                                            options.geom_consistency_max_cost);
      }
      scores[h] += weight * value;
      native_scores[h] += native_weight * value;
    }
  }
  const float inv_joint = 1.0f / joint_total;
  const float inv_native = native_total > 0.0f ? 1.0f / native_total : 0.0f;
  for (int h = 0; h < 15; ++h) {
    scores[h] =
        native_total > 0.0f
            ? 0.5f * (scores[h] * inv_joint + native_scores[h] * inv_native)
            : scores[h] * inv_joint;
  }
  // Strict comparison keeps the incumbent on exact ties (e.g. flat texture).
  int winner = 0;
  for (int h = 1; h < 15; ++h)
    if (isfinite(scores[h]) && scores[h] < scores[winner]) winner = h;
  if (winner != 0) {
    depths.Set(r, c, candidates[winner].depth);
    normals.SetSlice(r, c, candidates[winner].normal);
    cost.depth = candidates[winner].depth;
    cost.normal = candidates[winner].normal;
    for (int s = 0; s < costs.GetDepth(); ++s) {
      cost.src_image_idx = s;
      costs.Set(r, c, s, cost.Compute());
    }
  }
  // Same normalized objective that chose this pixel's winning plane. The
  // opposite checkerboard color reads it on the next launch; no lane in this
  // launch reads a score written by another lane of the same color.
  const float score = scores[winner];
  donor_scores.Set(r, c, isfinite(score) ? score : FLT_MAX);
}

template <bool kPhoto, bool kGeometry>
__global__ void Filter(const RefCamera camera,
                       GpuMatView<float> depth,
                       GpuMatView<float> normal,
                       GpuMatView<float> posterior,
                       PackedConsistencyMaskView mask,
                       cudaTextureObject_t poses,
                       cudaTextureObject_t source_depth,
                       SweepOptions options) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (c >= depth.GetWidth()) return;
  const Plane p = ReadPlane(depth, normal, r, c);
  float point[3];
  ComputePointAtDepth(camera, r, c, p.depth, point);
  LikelihoodComputer likelihood(options.ncc_sigma,
                                options.min_triangulation_angle,
                                options.incident_angle_sigma);
  const float threshold =
      likelihood.ComputeNCCProb(1.0f - options.filter_min_ncc);
  int consistent = 0;
  for (int s = 0; s < posterior.GetDepth(); ++s) {
    float tri, inc;
    ComputeViewingAngles(poses, point, p.normal, s, &tri, &inc);
    bool accepted =
        tri <= cos(options.filter_min_triangulation_angle) && inc > 0;
    if constexpr (kPhoto)
      accepted = accepted && posterior.Get(r, c, s) >= threshold;
    if constexpr (kGeometry)
      accepted = accepted && ComputeGeomConsistencyCost(
                                 camera,
                                 poses,
                                 source_depth,
                                 r,
                                 c,
                                 p.depth,
                                 s,
                                 options.geom_consistency_max_cost) <=
                                 options.filter_geom_consistency_max_cost;
    mask.Set(r, c, s, accepted ? 1 : 0);
    consistent += accepted;
  }
  if (consistent < options.filter_min_num_consistent) {
    depth.Set(r, c, 0.0f);
    const float zero[3] = {0, 0, 0};
    normal.SetSlice(r, c, zero);
    for (int s = 0; s < posterior.GetDepth(); ++s) mask.Set(r, c, s, 0);
  }
}

template <int kWindowSize,
          int kWindowStep,
          bool kGeometry,
          bool kFilterPhoto,
          bool kFilterGeometry,
          bool kCheckpoint>
void Launch(const RefCamera camera,
            GpuMatView<float> costs,
            GpuMatView<float> depth,
            GpuMatView<float> normals,
            PackedConsistencyMaskView mask,
            GpuMatView<float> posterior,
            GpuMatView<float> previous,
            GpuMatView<float> boundaries,
            GpuMatView<float> stripe,
            cudaTextureObject_t ref_texture,
            GpuMatView<float> ref_sum,
            GpuMatView<float> ref_squared_sum,
            cudaTextureObject_t sources,
            cudaTextureObject_t source_depths,
            cudaTextureObject_t poses,
            SweepOptions options,
            unsigned int step,
            GpuMatView<float> donor_scores) {
  const dim3 hmm_grid((costs.GetWidth() + 127) / 128, costs.GetDepth());
  const dim3 pixel_grid((costs.GetWidth() + 31) / 32, costs.GetHeight());
  const dim3 search_grid((costs.GetWidth() + 31) / 32,
                         (costs.GetHeight() + 1) / 2);
  VLOG(1) << "ACMH joint work: pixels=" << costs.GetWidth() * costs.GetHeight()
          << ", sources=" << costs.GetDepth()
          << ", propagated NCC/source=8, refinement NCC/source=6"
          << ", HMM full-chain passes=" << (kCheckpoint ? 6 : 4);
  CudaTimer prepass;
  FreezeHmm<kCheckpoint><<<hmm_grid, 128>>>(
      costs, posterior, previous, boundaries, stripe, options);
  CUDA_SYNC_AND_CHECK();
  prepass.Print(" Checkerboard frozen HMM");
  CudaTimer search;
  for (int color = 0; color < 2; ++color) {
    Search<kWindowSize, kWindowStep, kGeometry>
        <<<search_grid, 32>>>(camera,
                              costs,
                              depth,
                              normals,
                              posterior,
                              donor_scores,
                              ref_texture,
                              ref_sum,
                              ref_squared_sum,
                              sources,
                              source_depths,
                              poses,
                              options,
                              color,
                              step);
  }
  CUDA_SYNC_AND_CHECK();
  search.Print(" ACMH joint NCC search (two colors, fifteen candidates)");
  CudaTimer refresh;
  // Output posterior must describe the winning costs. No second application
  // of the previous-sweep prior: this refresh uses the new likelihood only.
  options.prev_sel_prob_weight = 0.0f;
  FreezeHmm<kCheckpoint><<<hmm_grid, 128>>>(
      costs, posterior, posterior, boundaries, stripe, options);
  CUDA_SYNC_AND_CHECK();
  refresh.Print(" Checkerboard winner HMM refresh");
  if constexpr (kFilterPhoto || kFilterGeometry) {
    Filter<kFilterPhoto, kFilterGeometry><<<pixel_grid, 32>>>(
        camera, depth, normals, posterior, mask, poses, source_depths, options);
  }
}

}  // namespace experimental_acmh
