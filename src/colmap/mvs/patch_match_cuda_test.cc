// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/patch_match_cuda.h"

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_cuda_test.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace internal {
float ProbePropagatedDepth(const std::array<float, 4>& K,
                           float depth,
                           const std::array<float, 3>& normal,
                           float row1,
                           float row2,
                           float col);
std::pair<std::vector<int>, std::vector<float>> ProbeSampledSources(
    const std::vector<int>&, const std::vector<float>&, int, bool);
int ProbeSamplingSupport();
std::vector<uint32_t> ProbeSelectionProbability(const std::vector<uint32_t>&);
Mat<float> ProbeOwnershipInitialCost(PatchMatchCuda&, unsigned long long);
}  // namespace internal
namespace {
constexpr float kDepthMin = 1.0f;
constexpr float kDepthMax = 8.0f;
constexpr float kPlaneZ = 4.0f;
struct Inputs {
  std::vector<Image> images;
  std::vector<DepthMap> depths;
  std::vector<NormalMap> normals;
};

// Two independent reference problems, each with differently calibrated source
// views. All images are rendered from the same textured world plane. Sizes are
// deliberately non-square, odd, and not CUDA block multiples.
Inputs MakeInputs(const int variant,
                  const int num_source_images = 2,
                  const int ref_height = 0,
                  const int ref_width = 0) {
  Inputs inputs;
  for (int i = 0; i <= num_source_images; ++i) {
    const int width =
        (ref_width > 0 ? ref_width : (variant == 0 ? 47 : 39)) + 2 * i;
    const int height =
        (ref_height > 0 ? ref_height : (variant == 0 ? 35 : 53)) + 2 * i;
    const float fx = (variant == 0 ? 48.0f : 69.0f) + 3.0f * i;
    const float fy = (variant == 0 ? 57.0f : 44.0f) + 2.0f * i;
    const float cx = width * 0.5f + 0.7f + i * 0.2f;
    const float cy = height * 0.5f - 1.1f - i * 0.15f;
    const float yaw = (i - 1) * 0.035f;
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    const float camera_x = (i - 1) * 0.25f;
    const float camera_y = (i - 1) * 0.04f;
    const float K[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    const float R[9] = {c, 0, s, 0, 1, 0, -s, 0, c};
    const float T[3] = {-c * camera_x, -camera_y, s * camera_x};

    Image image(
        "synthetic-" + std::to_string(variant) + "-" + std::to_string(i),
        width,
        height,
        K,
        R,
        T);
    Bitmap bitmap(width, height, false);
    DepthMap depth(width, height, kDepthMin, kDepthMax);
    NormalMap normal(width, height);
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        // COLMAP's undistorted calibration uses pixel centers at (col+.5,
        // row+.5). Invert R to intersect the camera ray with world Z=4.
        const float ray_x = (col + 0.5f - cx) / fx;
        const float ray_y = (row + 0.5f - cy) / fy;
        const float world_ray_x = c * ray_x - s;
        const float world_ray_z = s * ray_x + c;
        const float z = kPlaneZ / world_ray_z;
        const float x = camera_x + z * world_ray_x;
        const float y = camera_y + z * ray_y;
        const float value = 128.0f + 42.0f * std::sin(8.0f * x + 3.0f * y) +
                            35.0f * std::cos(5.0f * x - 11.0f * y) +
                            24.0f * std::sin(17.0f * x + 13.0f * y);
        bitmap.SetPixel(col,
                        row,
                        BitmapColor<uint8_t>(static_cast<uint8_t>(
                            std::clamp(std::round(value), 0.0f, 255.0f))));
        depth.Set(row, col, z);
        normal.Set(row, col, 0, -s);
        normal.Set(row, col, 1, 0.0f);
        normal.Set(row, col, 2, -c);
      }
    }
    image.SetBitmap(std::move(bitmap));
    inputs.images.push_back(std::move(image));
    inputs.depths.push_back(std::move(depth));
    inputs.normals.push_back(std::move(normal));
  }
  return inputs;
}

void MakeSourceTexturesDistinct(Inputs* inputs) {
  // Spatial errors distinguish sources; an affine intensity change would not
  // distinguish their NCC values reliably.
  for (int source = 1; source < static_cast<int>(inputs->images.size()); ++source) {
    Bitmap bitmap = inputs->images[source].GetBitmap().Clone();
    for (int row = 0; row < bitmap.Height(); ++row) {
      for (int col = 0; col < bitmap.Width(); ++col) {
        const int perturbation =
            ((row * (source + 1) + col * (2 * source + 1)) % 17) - 8;
        const int value = bitmap.GetPixel(col, row)->r + perturbation;
        bitmap.SetPixel(col, row, BitmapColor<uint8_t>(
            static_cast<uint8_t>(std::clamp(value, 0, 255))));
      }
    }
    inputs->images[source].SetBitmap(std::move(bitmap));
  }
}

void AddDistinctSourceDepthErrors(Inputs* inputs) {
  for (int source = 1; source < static_cast<int>(inputs->depths.size()); ++source) {
    auto& depth = inputs->depths[source];
    for (int row = 0; row < static_cast<int>(depth.GetHeight()); ++row) {
      for (int col = 0; col < static_cast<int>(depth.GetWidth()); ++col) {
        const float scale = 1.0f + 0.003f * source +
                            0.0004f * ((row + 3 * col) % 7 - 3);
        depth.Set(row, col, depth.Get(row, col) * scale);
      }
    }
  }
}

PatchMatchOptions MakeOptions(const bool geometry,
                              const bool filter,
                              const bool /*unused*/) {
  PatchMatchOptions options;
  options.gpu_index = "0";
  options.depth_min = kDepthMin;
  options.depth_max = kDepthMax;
  options.window_radius = 2;
  options.window_step = 1;
  options.num_samples = 3;
  // Two iterations cross both alternating orientations repeatedly.
  options.num_iterations = 2;
  options.geom_consistency = geometry;
  options.filter = filter;
  // Keep full-storage controls explicit when the public CUDA default changes.
  options.filter_min_ncc = -1.0;
  options.filter_min_triangulation_angle = 0.0;
  options.filter_min_num_consistent = 1;
  options.filter_geom_consistency_max_cost = 10.0;
  return options;
}

struct Result {
  DepthMap depths;
  NormalMap normals;
  Mat<float> selection;
  // Query graph through its public API, preserving each pixel and ordered IDs.
  std::vector<std::vector<int>> consistent_images;
};

Result RunPatchMatch(Inputs* inputs,
                     const PatchMatchOptions& options,
                     const std::vector<int>& ordered_sources = {}) {
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = ordered_sources;
  if (problem.src_image_idxs.empty()) {
    for (size_t i = 1; i < inputs->images.size(); ++i) {
      problem.src_image_idxs.push_back(static_cast<int>(i));
    }
  }
  problem.images = &inputs->images;
  problem.depth_maps = &inputs->depths;
  problem.normal_maps = &inputs->normals;
  PatchMatch patch_match(options, problem);
  patch_match.Run();
  Result result{patch_match.GetDepthMap(),
                patch_match.GetNormalMap(),
                patch_match.GetSelProbMap(),
                {}};
  const ConsistencyGraph graph = patch_match.GetConsistencyGraph();
  for (size_t row = 0; row < result.depths.GetHeight(); ++row) {
    for (size_t col = 0; col < result.depths.GetWidth(); ++col) {
      int num_images = 0;
      const int* image_idxs = nullptr;
      graph.GetImageIdxs(row, col, &num_images, &image_idxs);
      result.consistent_images.emplace_back();
      if (num_images > 0) {
        result.consistent_images.back().assign(image_idxs,
                                               image_idxs + num_images);
      }
    }
  }
  return result;
}

void ExpectExactlyEqual(const Mat<float>& expected, const Mat<float>& actual) {
  ASSERT_EQ(expected.GetWidth(), actual.GetWidth());
  ASSERT_EQ(expected.GetHeight(), actual.GetHeight());
  ASSERT_EQ(expected.GetDepth(), actual.GetDepth());
  ASSERT_EQ(expected.GetData().size(), actual.GetData().size());
  for (size_t i = 0; i < expected.GetData().size(); ++i) {
    const float a = expected.GetData()[i];
    const float b = actual.GetData()[i];
    ASSERT_TRUE(std::isfinite(a)) << i;
    ASSERT_TRUE(std::isfinite(b)) << i;
    // These delivery/allocation/scheduling paths preserve the arithmetic.
    // Require byte equality within the same executable and CUDA device only.
    ASSERT_EQ(std::memcmp(&a, &b, sizeof(float)), 0)
        << "element " << i << ": " << a << " vs " << b;
  }
}

void ExpectNonVacuousGeometry(const internal::SweepTraceResult& control,
                              const internal::SweepTraceResult& changed) {
  ExpectExactlyEqual(control.initial_cost, changed.initial_cost);
  int source_sensitive_pixels = 0;
  int distinct_depth_geometry_pixels = 0;
  for (size_t row = 0; row < changed.draws.GetHeight(); ++row) {
    for (size_t col = 0; col < changed.draws.GetWidth(); ++col) {
      bool same_inputs = true;
      for (int component = 0; component < 20; ++component) {
        const float a = control.hypotheses.Get(row, col, component);
        const float b = changed.hypotheses.Get(row, col, component);
        same_inputs = same_inputs && std::memcmp(&a, &b, sizeof(float)) == 0;
      }
      int first_source = -1;
      bool distinct_sources = false;
      double incumbent_photo_sum = 0.0;
      double winner_photo_sum = 0.0;
      for (size_t sample = 0; sample < changed.draws.GetDepth(); ++sample) {
        const int source = changed.draws.Get(row, col, sample);
        same_inputs = same_inputs && source == control.draws.Get(row, col, sample);
        if (source < 0) continue;
        if (first_source < 0) first_source = source;
        distinct_sources = distinct_sources || source != first_source;
        incumbent_photo_sum += changed.initial_cost.Get(row, col, source);
        winner_photo_sum += changed.cost.Get(row, col, source);
      }
      // Only source depth maps differ. Matching tuples/draws and identical
      // incoming NCC, but a changed incumbent total, isolate consumed GEO.
      // All-equal clamped reprojections cannot satisfy this check.
      if (same_inputs && distinct_sources &&
          std::abs(control.hypothesis_costs.Get(row, col, 0) -
                   changed.hypothesis_costs.Get(row, col, 0)) > 1e-3f) {
        ++source_sensitive_pixels;
      }
      const int winner = changed.winners.Get(row, col);
      if (winner == 2 || winner == 4) {
        // For a random-depth winner, the exported cost map is its photo-only
        // NCC. Subtraction estimates the two consumed GEO contributions. The
        // coarse threshold is a nonvacuity check, not a floating-point oracle.
        const double current_geometry =
            changed.hypothesis_costs.Get(row, col, 0) - incumbent_photo_sum;
        const double random_geometry =
            changed.hypothesis_costs.Get(row, col, winner) - winner_photo_sum;
        if (distinct_sources &&
            std::abs(current_geometry - random_geometry) > 1e-3) {
          ++distinct_depth_geometry_pixels;
        }
      }
    }
  }
  EXPECT_GT(source_sensitive_pixels, 0);
  EXPECT_GT(distinct_depth_geometry_pixels, 0);
}

void ExpectEquivalent(const Result& expected, const Result& actual) {
  ExpectExactlyEqual(expected.depths, actual.depths);
  ExpectExactlyEqual(expected.normals, actual.normals);
  ExpectExactlyEqual(expected.selection, actual.selection);
  // Discrete source memberships must agree exactly, including empty pixels.
  ASSERT_EQ(expected.consistent_images.size(), actual.consistent_images.size());
  for (size_t i = 0; i < expected.consistent_images.size(); ++i) {
    ASSERT_EQ(expected.consistent_images[i], actual.consistent_images[i]) << i;
  }
}

void ExpectNonempty(const Result& result,
                    const Inputs& inputs,
                    const bool filter) {
  EXPECT_EQ(result.depths.GetWidth(), inputs.images[0].GetWidth());
  EXPECT_EQ(result.depths.GetHeight(), inputs.images[0].GetHeight());
  EXPECT_EQ(result.normals.GetDepth(), 3);
  EXPECT_EQ(result.selection.GetDepth(), inputs.images.size() - 1);
  EXPECT_EQ(result.depths.GetData().size(),
            result.depths.GetWidth() * result.depths.GetHeight());
  size_t valid_depths = 0;
  for (const float depth : result.depths.GetData()) {
    ASSERT_TRUE(std::isfinite(depth));
    if (depth > 0.0f) {
      ++valid_depths;
    }
  }
  // Propagation and perturbation can leave the initial sampling range.
  EXPECT_GT(valid_depths, 0);
  size_t consistent_pixels = 0;
  for (const auto& image_idxs : result.consistent_images) {
    consistent_pixels += !image_idxs.empty();
    for (const int image_idx : image_idxs) {
      EXPECT_GT(image_idx, 0);
      EXPECT_LT(image_idx, inputs.images.size());
    }
  }
  if (filter) {
    EXPECT_GT(consistent_pixels, 0);
  } else {
    EXPECT_EQ(consistent_pixels, 0);
  }
}

class CudaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int num_devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&num_devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && num_devices == 0)) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

TEST_F(CudaTest, DirectInstancesOwnReferenceCalibration) {
  auto a = MakeInputs(0);
  auto b = MakeInputs(1);
  ASSERT_NE(a.images[0].GetK()[0], b.images[0].GetK()[0]);
  ASSERT_NE(a.images[0].GetK()[4], b.images[0].GetK()[4]);
  auto options = MakeOptions(true, false, false);
  options.window_radius = 5;
  options.window_step = 1;
  options.sigma_spatial = 5.0;
  options.num_samples = 17;
  auto make_problem = [](Inputs& inputs) {
    PatchMatch::Problem problem;
    problem.ref_image_idx = 0;
    problem.src_image_idxs = {1, 2};
    problem.images = &inputs.images;
    problem.depth_maps = &inputs.depths;
    problem.normal_maps = &inputs.normals;
    return problem;
  };
  const auto problem_a = make_problem(a);
  const auto problem_b = make_problem(b);
  constexpr unsigned long long kSeed = 12345;

  PatchMatchCuda isolated(options, problem_a);
  const Mat<float> isolated_cost =
      internal::ProbeOwnershipInitialCost(isolated, kSeed);
  isolated.Run();
  const auto isolated_depth = isolated.GetDepthMap();
  const auto isolated_normal = isolated.GetNormalMap();
  const auto isolated_selection = isolated.GetSelProbMap();

  PatchMatchCuda interleaved(options, problem_a);
  PatchMatchCuda other(options, problem_b);
  const Mat<float> interleaved_cost =
      internal::ProbeOwnershipInitialCost(interleaved, kSeed);
  interleaved.Run();
  ASSERT_GT(isolated_cost.GetData().size(), 0);
  EXPECT_TRUE(std::any_of(
      isolated_cost.GetData().begin(),
      isolated_cost.GetData().end(),
      [](float cost) { return std::isfinite(cost) && cost < 1.9f; }));
  ExpectExactlyEqual(isolated_cost, interleaved_cost);
  ExpectExactlyEqual(isolated_depth, interleaved.GetDepthMap());
  ExpectExactlyEqual(isolated_normal, interleaved.GetNormalMap());
  ExpectExactlyEqual(isolated_selection, interleaved.GetSelProbMap());
  EXPECT_TRUE(std::any_of(
      isolated_depth.GetData().begin(),
      isolated_depth.GetData().end(),
      [](float depth) { return std::isfinite(depth) && depth > 0; }));
}

TEST_F(CudaTest, NccNormalizationMatchesRetainedArithmeticAcrossRows) {
  constexpr int kWidth = 35, kHeight = 67;
  auto inputs = MakeInputs(0, 3, kHeight, kWidth);
  // One complete flat reference patch and many textured patches exercise both
  // the early rejection and the remaining NCC arithmetic. The height crosses
  // an initial-cost stripe and leaves a partial three-row tail.
  Bitmap reference = inputs.images[0].GetBitmap().Clone();
  for (int row = 16; row <= 34; ++row) {
    for (int col = 10; col <= 24; ++col) {
      reference.SetPixel(col, row, BitmapColor<uint8_t>(128));
    }
  }
  inputs.images[0].SetBitmap(std::move(reference));
  // On positive-reference patches the first source rejects by source variance
  // only after its complete weight fold; later useful sources reuse that fold.
  Bitmap constant = inputs.images[3].GetBitmap().Clone();
  constant.Fill(BitmapColor<uint8_t>(128));
  inputs.images[3].SetBitmap(std::move(constant));
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {3, 2, 1};
  problem.images = &inputs.images;
  problem.depth_maps = &inputs.depths;
  problem.normal_maps = &inputs.normals;
  for (const int radius : {5, 2}) {
    SCOPED_TRACE(::testing::Message() << "radius=" << radius);
    auto options = MakeOptions(true, false, false);
    options.window_radius = radius;
    options.sigma_spatial = 5.0;
    PatchMatchCuda pm(options, problem);
    const auto costs = internal::ProbeNccCosts(pm);
    ExpectExactlyEqual(costs.reference, costs.candidate);
    ASSERT_EQ(costs.candidate.GetWidth(), kWidth);
    ASSERT_EQ(costs.candidate.GetHeight(), kHeight);
    ASSERT_EQ(costs.candidate.GetDepth(), 3);
    int useful_after_source_rejection = 0;
    int distinct_useful_sources = 0;
    int useful_final_stripe = 0;
    for (int row = 0; row < kHeight; ++row) {
      for (int col = 0; col < kWidth; ++col) {
        const float first = costs.reference.Get(row, col, 0);
        const float second = costs.reference.Get(row, col, 1);
        const float third = costs.reference.Get(row, col, 2);
        useful_after_source_rejection += first == 2.0f && second < 1.9f;
        distinct_useful_sources +=
            second < 1.9f && third < 1.9f && second != third;
        useful_final_stripe += row >= 64 && (second < 1.9f || third < 1.9f);
      }
    }
    EXPECT_GT(useful_after_source_rejection, 0);
    EXPECT_GT(distinct_useful_sources, 0);
    EXPECT_GT(useful_final_stripe, 0);
    for (int row = 21; row <= 29; ++row) {
      for (int col = 15; col <= 19; ++col) {
        for (int source = 0; source < 3; ++source) {
          EXPECT_EQ(costs.candidate.Get(row, col, source), 2.0f);
        }
      }
    }
  }
}

TEST_F(CudaTest, FreshProposalAliasesMatchIndependentProducer) {
  constexpr int kWidth = 35, kHeight = 19, kSources = 5;
  auto inputs = MakeInputs(1, kSources, kHeight, kWidth);
  MakeSourceTexturesDistinct(&inputs);
  AddDistinctSourceDepthErrors(&inputs);
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {1, 2, 3, 4, 5};
  problem.images = &inputs.images;
  problem.depth_maps = &inputs.depths;
  problem.normal_maps = &inputs.normals;
  auto options = MakeOptions(true, false, false);
  options.window_radius = 5;
  options.sigma_spatial = 5.0;
  options.num_samples = 17;
  PatchMatchCuda pm(options, problem);
  const auto result = internal::ProbeFreshProposalAliases(pm);

  // The probe stages a nonzero first row and nine rows, crossing an eight-row
  // producer stripe. Width35 leaves both eight- and32-column partial blocks.
  // All output planes are poisoned before the actual producer is launched.
  ExpectExactlyEqual(result.reference, result.candidate);
  ExpectExactlyEqual(result.reference_geometry, result.candidate_geometry);
  ASSERT_EQ(result.candidate.GetWidth(), kWidth);
  ASSERT_EQ(result.candidate.GetHeight(), 9);
  ASSERT_EQ(result.candidate.GetDepth(), 3 * kSources);
  ASSERT_EQ(result.candidate_geometry.GetDepth(), 2 * kSources);
  ASSERT_EQ(result.aliases.GetWidth(), kWidth);
  ASSERT_EQ(result.aliases.GetHeight(), 9);

  std::array<int, 4> alias_pixels{}, useful_geometry{};
  std::array<std::set<float>, 4> useful_costs;
  int fresh_current_tuple_differs_from_incumbent = 0;
  int useful_tail = 0;
  for (int row = 0; row < 9; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      const int aliases = result.aliases.Get(row, col);
      ASSERT_GE(aliases, 0);
      ASSERT_LE(aliases, 3);
      ++alias_pixels[aliases];
      for (int source = 0; source < kSources; ++source) {
        for (int hypothesis = 0; hypothesis < 3; ++hypothesis) {
          const float cost =
              result.reference.Get(row, col, hypothesis * kSources + source);
          if (cost > 0.0f && cost < 2.0f) {
            useful_costs[aliases].insert(cost);
            useful_tail += row == 8 && col >= 32;
          }
        }
        for (int depth_kind = 0; depth_kind < 2; ++depth_kind) {
          const float cost = result.reference_geometry.Get(
              row, col, depth_kind * kSources + source);
          useful_geometry[aliases] +=
              cost > 0.0f && cost < options.geom_consistency_max_cost;
        }
        if ((aliases & 1) != 0) {
          // h2 uses the current normal. When the random depth is identical,
          // h2 is a fresh evaluation of the incoming tuple, whose old cost
          // is deliberately different. Parameter equality is insufficient
          // permission to substitute the incoming incumbent cost map.
          const float fresh_current =
              result.reference.Get(row, col, 2 * kSources + source);
          fresh_current_tuple_differs_from_incumbent +=
              fresh_current > 0.0f && fresh_current < 2.0f &&
              fresh_current != result.incoming_cost;
        }
      }
    }
  }
  for (int aliases = 0; aliases < 4; ++aliases) {
    SCOPED_TRACE(::testing::Message() << "alias flags=" << aliases);
    EXPECT_GT(alias_pixels[aliases], 0);
    EXPECT_GT(useful_costs[aliases].size(), 1);
    EXPECT_GT(useful_geometry[aliases], 0);
  }
  EXPECT_GT(fresh_current_tuple_differs_from_incumbent, 0);
  EXPECT_GT(useful_tail, 0);
}

TEST_F(CudaTest, PhotoConstructorMatchesRetainedThreeStageInitialization) {
  constexpr int kWidth = 35, kHeight = 19;
  auto inputs = MakeInputs(1, 2, kHeight, kWidth);
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {2, 1};
  problem.images = &inputs.images;
  auto options = MakeOptions(false, false, false);
  options.window_radius = 5;
  const float* K = inputs.images[0].GetK();
  ASSERT_NE(K[0], K[4]);
  const std::array<float, 4> inv_K = {
      1.0f / K[0], -(K[2] - 0.5f) / K[0], 1.0f / K[4], -(K[5] - 0.5f) / K[4]};
  const auto expected = internal::ProbeLegacyPhotoInitialization(
      kWidth, kHeight, inv_K, options.depth_min, options.depth_max);

  PatchMatchCuda actual(options, problem);
  // No Run, ProbeSweep or probe reseed intervenes before these observations.
  ExpectExactlyEqual(expected.depth, actual.GetDepthMap());
  ExpectExactlyEqual(expected.normal, actual.GetNormalMap());
  const auto retained = internal::ProbeConstructorContinuation(actual);
  std::vector<std::pair<int, int>> perimeter;
  for (int col = 0; col < kWidth; ++col) perimeter.emplace_back(0, col);
  for (int row = 1; row < kHeight; ++row)
    perimeter.emplace_back(row, kWidth - 1);
  for (int col = kWidth - 2; col >= 0; --col)
    perimeter.emplace_back(kHeight - 1, col);
  for (int row = kHeight - 2; row > 0; --row) perimeter.emplace_back(row, 0);
  ASSERT_EQ(retained.GetWidth(), perimeter.size());
  ASSERT_EQ(retained.GetHeight(), 1);
  ASSERT_EQ(retained.GetDepth(), 3);
  int perimeter_retries = 0;
  for (size_t index = 0; index < perimeter.size(); ++index) {
    const auto [row, col] = perimeter[index];
    perimeter_retries += expected.normal_pairs.Get(row, col) > 1;
    for (int draw = 0; draw < 3; ++draw) {
      EXPECT_EQ(retained.Get(0, index, draw),
                expected.continuation.Get(row, col, draw));
    }
  }
  EXPECT_GT(perimeter_retries, 0);
  EXPECT_TRUE(std::any_of(expected.normal_pairs.GetData().begin(),
                          expected.normal_pairs.GetData().end(),
                          [](int pairs) { return pairs > 1; }));
  int distinct_components = 0;
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      const float nx = expected.normal.Get(row, col, 0);
      const float ny = expected.normal.Get(row, col, 1);
      const float nz = expected.normal.Get(row, col, 2);
      distinct_components += nx != ny && ny != nz && nx != nz;
    }
  }
  EXPECT_GT(distinct_components, kWidth * kHeight / 2);
}

// Compose real band execution, perimeter histories and Rotate. The independent
// perimeter test compares integer ownership/seed evolution against a dense map;
// this test additionally requires the production scalar/band traces to agree.
TEST_F(CudaTest, BandAndScalarKeepPerimeterHistoriesAcrossFourRotations) {
  constexpr int kWidth = 35, kHeight = 257, kSources = 17, kSamples = 17;
  constexpr unsigned long long kSeed = 9182;
  auto inputs = MakeInputs(0, 2, kHeight, kWidth);
  const Image source_image = inputs.images[2];
  const DepthMap source_depth = inputs.depths[2];
  const NormalMap source_normal = inputs.normals[2];
  inputs.images[1] = source_image;
  inputs.depths[1] = source_depth;
  inputs.normals[1] = source_normal;
  while (inputs.images.size() < kSources + 1) {
    inputs.images.push_back(source_image);
    inputs.depths.push_back(source_depth);
    inputs.normals.push_back(source_normal);
  }
  MakeSourceTexturesDistinct(&inputs);
  AddDistinctSourceDepthErrors(&inputs);
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      inputs.depths[0].Set(row, col, row % 2 == 0 ? 3.2f : 5.2f);
    }
  }
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  for (int i = 1; i <= kSources; ++i) problem.src_image_idxs.push_back(i);
  problem.images = &inputs.images;
  problem.depth_maps = &inputs.depths;
  problem.normal_maps = &inputs.normals;
  auto options = MakeOptions(true, false, false);
  options.window_radius = 5;
  options.num_samples = kSamples;
  options.sigma_spatial = 5.0;
  options.num_iterations = 1;

  PatchMatchCuda scalar(options, problem);
  PatchMatchCuda band(options, problem);
  const auto expected = internal::ProbeSweepSequence(scalar, false, kSeed);
  const auto actual = internal::ProbeSweepSequence(band, true, kSeed);
  ASSERT_EQ(expected.size(), 4);
  ASSERT_EQ(actual.size(), expected.size());
  for (int step = 0; step < 4; ++step) {
    SCOPED_TRACE(::testing::Message() << "orientation=" << step);
    const auto& a = expected[step];
    const auto& b = actual[step];
    const int width = step % 2 == 0 ? kWidth : kHeight;
    const int height = step % 2 == 0 ? kHeight : kWidth;
    ASSERT_EQ(a.draws.GetWidth(), width);
    ASSERT_EQ(a.draws.GetHeight(), height);
    ASSERT_EQ(a.draws.GetData(), b.draws.GetData());
    ASSERT_EQ(a.winners.GetData(), b.winners.GetData());
    ASSERT_EQ(a.support.GetData(), b.support.GetData());
    ExpectExactlyEqual(a.initial_cost, b.initial_cost);
    ExpectExactlyEqual(a.hypotheses, b.hypotheses);
    ExpectExactlyEqual(a.hypothesis_costs, b.hypothesis_costs);
    ExpectExactlyEqual(a.forward, b.forward);
    ExpectExactlyEqual(a.cost, b.cost);
    ExpectExactlyEqual(a.depth, b.depth);
    ExpectExactlyEqual(a.normal, b.normal);
    ExpectExactlyEqual(a.selection, b.selection);
    ASSERT_EQ(a.continuation.GetData(), b.continuation.GetData());
    ASSERT_EQ(a.continuation_before.GetData(), b.continuation_before.GetData());
    EXPECT_NE(a.continuation.GetData(), a.continuation_before.GetData());
    EXPECT_TRUE(std::any_of(a.support.GetData().begin(),
                            a.support.GetData().end(),
                            [](int supported) { return supported != 0; }));
    EXPECT_TRUE(std::any_of(a.draws.GetData().begin(),
                            a.draws.GetData().end(),
                            [](int source) { return source == kSources - 1; }));
    if (step > 0) {
      const auto& previous = actual[step - 1].continuation;
      // CCW rotation maps the prior top-right corner to the new top-left.
      // No new draws, reseed or state copy may intervene between the sweeps.
      for (int draw = 0; draw < 3; ++draw) {
        EXPECT_EQ(previous.Get(0, previous.GetWidth() - 1, draw),
                  b.continuation_before.Get(0, 0, draw));
      }
    }
    if (height > 128) {
      for (int boundary : {128, 256}) {
        int supported = 0;
        for (int col = 0; col < width; ++col) {
          supported += a.support.Get(boundary, col);
        }
        EXPECT_GT(supported, 0) << "band boundary " << boundary;
      }
    }
  }
  ExpectExactlyEqual(scalar.GetDepthMap(), band.GetDepthMap());
  ExpectExactlyEqual(scalar.GetNormalMap(), band.GetNormalMap());
  ExpectExactlyEqual(scalar.GetSelProbMap(), band.GetSelProbMap());
}

TEST_F(CudaTest, ImmutableReferenceStatisticsMatchRotatedPlanesAcrossBands) {
  constexpr int kWidth = 35, kHeight = 259, kSources = 8;
  auto inputs = MakeInputs(1, kSources, kHeight, kWidth);
  MakeSourceTexturesDistinct(&inputs);
  AddDistinctSourceDepthErrors(&inputs);
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  for (int source = 1; source <= kSources; ++source) {
    problem.src_image_idxs.push_back(source);
  }
  problem.images = &inputs.images;
  problem.depth_maps = &inputs.depths;
  problem.normal_maps = &inputs.normals;
  auto options = MakeOptions(true, false, false);
  options.window_radius = 5;
  options.num_samples = 17;
  options.num_iterations = 1;
  options.sigma_spatial = 5.0;

  PatchMatchCuda rotated_planes(options, problem);
  PatchMatchCuda immutable(options, problem);
  // This control retains a single original-orientation Filter result and uses
  // actual GpuMat rotations. It never calls OriginalCoordinates for moments.
  internal::UseLegacyReferenceStatistics(rotated_planes);
  const auto expected =
      internal::ProbeSweepSequence(rotated_planes, false, 7147);
  const auto actual = internal::ProbeSweepSequence(immutable, true, 7147);
  ASSERT_EQ(expected.size(), 4);
  ASSERT_EQ(actual.size(), 4);
  for (int orientation = 0; orientation < 4; ++orientation) {
    SCOPED_TRACE(::testing::Message() << "orientation=" << orientation);
    const auto& a = expected[orientation];
    const auto& b = actual[orientation];
    ASSERT_FALSE(a.immutable_reference_statistics);
    ASSERT_TRUE(b.immutable_reference_statistics);
    ASSERT_EQ(a.reference_statistics.GetWidth(),
              orientation % 2 == 0 ? kWidth : kHeight);
    ASSERT_EQ(a.reference_statistics.GetHeight(),
              orientation % 2 == 0 ? kHeight : kWidth);
    ASSERT_EQ(a.reference_statistics.GetData(),
              b.reference_statistics.GetData());
    int distinct_moments = 0;
    for (uint32_t word : a.reference_statistics.GetData()) {
      distinct_moments += word != a.reference_statistics.Get(0, 0);
    }
    EXPECT_GT(distinct_moments, kWidth * kHeight);
    ASSERT_EQ(a.draws.GetData(), b.draws.GetData());
    ASSERT_EQ(a.winners.GetData(), b.winners.GetData());
    ASSERT_EQ(a.support.GetData(), b.support.GetData());
    ExpectExactlyEqual(a.initial_cost, b.initial_cost);
    ExpectExactlyEqual(a.hypotheses, b.hypotheses);
    ExpectExactlyEqual(a.hypothesis_costs, b.hypothesis_costs);
    ExpectExactlyEqual(a.forward, b.forward);
    ExpectExactlyEqual(a.cost, b.cost);
    ExpectExactlyEqual(a.depth, b.depth);
    ExpectExactlyEqual(a.normal, b.normal);
    ExpectExactlyEqual(a.selection, b.selection);
    ASSERT_EQ(a.continuation.GetData(), b.continuation.GetData());
    if (a.draws.GetHeight() > 128) {
      for (int row : {128, 256}) {
        int supported = 0, row_sensitive = 0;
        for (size_t col = 0; col < a.draws.GetWidth(); ++col) {
          supported += a.support.Get(row, col) != 0;
          row_sensitive += a.reference_statistics.Get(row, col) !=
                           a.reference_statistics.Get(0, col);
        }
        EXPECT_GT(supported, 0) << "band boundary " << row;
        EXPECT_GT(row_sensitive, 0) << "global row must differ from band row";
      }
    }
  }
  // Both owners finish in original orientation and keep the ordinary getters.
  ExpectExactlyEqual(rotated_planes.GetDepthMap(), immutable.GetDepthMap());
  ExpectExactlyEqual(rotated_planes.GetNormalMap(), immutable.GetNormalMap());
  ExpectExactlyEqual(rotated_planes.GetSelProbMap(), immutable.GetSelProbMap());
}

// Compute the ray by undoing the image rotation, then rotating the camera ray.
// This does not reuse the device's rotated calibration or propagation helper.
std::array<double, 3> ReferenceRay(const Image& image,
                                   const int orientation,
                                   const int row,
                                   const int col) {
  double x = col;
  double y = row;
  switch (orientation) {
    case 1:
      x = image.GetWidth() - 1 - row;
      y = col;
      break;
    case 2:
      x = image.GetWidth() - 1 - col;
      y = image.GetHeight() - 1 - row;
      break;
    case 3:
      x = row;
      y = image.GetHeight() - 1 - col;
      break;
  }
  const float* K = image.GetK();
  x = (x + 0.5 - K[2]) / K[0];
  y = (y + 0.5 - K[5]) / K[4];
  switch (orientation) {
    case 1:
      return {y, -x, 1};
    case 2:
      return {-x, -y, 1};
    case 3:
      return {-y, x, 1};
    default:
      return {x, y, 1};
  }
}

void ExpectPropagationAnchors(const internal::SweepTraceResult& trace,
                              const Image& image,
                              const int orientation) {
  const int width = static_cast<int>(trace.hypotheses.GetWidth());
  const int height = static_cast<int>(trace.hypotheses.GetHeight());
  ASSERT_GT(width, 0);
  ASSERT_GT(height, 1);
  for (int row = 0; row < height; ++row) {
    int changed_depths = 0;
    for (int col = 0; col < width; ++col) {
      SCOPED_TRACE(::testing::Message() << "row=" << row << " col=" << col);
      // Zero NCC emissions keep the deliberately oblique incumbent selected.
      ASSERT_EQ(trace.support.Get(row, col), 0);
      ASSERT_EQ(trace.winners.Get(row, col), 0);
      const int source_row = row == 0 ? 0 : row - 1;
      const float source_depth = trace.hypotheses.Get(source_row, col, 0);
      const float propagated_depth = trace.hypotheses.Get(row, col, 4);
      for (int axis = 0; axis < 3; ++axis) {
        EXPECT_EQ(trace.hypotheses.Get(row, col, 5 + axis),
                  trace.hypotheses.Get(source_row, col, 1 + axis));
      }
      if (row == 0) {
        EXPECT_EQ(propagated_depth, source_depth);
        continue;
      }
      const auto source_ray = ReferenceRay(image, orientation, row - 1, col);
      const auto target_ray = ReferenceRay(image, orientation, row, col);
      double source_dot = 0;
      double target_dot = 0;
      for (int axis = 0; axis < 3; ++axis) {
        const double normal = trace.hypotheses.Get(source_row, col, 1 + axis);
        source_dot += normal * source_ray[axis];
        target_dot += normal * target_ray[axis];
      }
      // This fixture stays far from the production fallback and plane horizon.
      ASSERT_GT(std::abs(target_dot), 0.1);
      const double expected = source_depth * source_dot / target_dot;
      ASSERT_GT(expected, 0);
      const double tolerance = 2e-5 * expected;
      EXPECT_NEAR(propagated_depth, expected, tolerance);
      changed_depths += std::abs(expected - source_depth) > 10 * tolerance;
    }
    // Every nonzero row, including each band's first row, needs a real
    // transfer.
    if (row > 0) {
      EXPECT_GT(changed_depths, 0) << "row=" << row;
    }
  }
}

TEST_F(CudaTest, PropagationUsesTheActualSourcePixel) {
  constexpr int kWidth = 35;
  constexpr int kHeight = 259;
  auto inputs = MakeInputs(0, 1, kHeight, kWidth);
  for (Image& image : inputs.images) {
    Bitmap bitmap(image.GetWidth(), image.GetHeight(), false);
    bitmap.Fill(BitmapColor<uint8_t>(0));
    image.SetBitmap(std::move(bitmap));
  }
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      inputs.depths[0].Set(row, col, 3.2f + 0.001f * row);
      inputs.normals[0].Set(row, col, 0, 0.2f);
      inputs.normals[0].Set(row, col, 1, 0.3f);
      inputs.normals[0].Set(row, col, 2, -std::sqrt(0.87f));
    }
  }
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {1};
  problem.images = &inputs.images;
  problem.depth_maps = &inputs.depths;
  problem.normal_maps = &inputs.normals;
  auto options = MakeOptions(true, false, false);
  options.window_radius = 5;
  options.sigma_spatial = 5.0;
  options.ncc_sigma = 0.01;
  options.num_samples = 1;
  options.num_iterations = 1;
  ASSERT_TRUE(options.Check());

  // Constant images give cost=2. At this sigma the emission is exactly zero,
  // so the real sweep retains the seed while still tracing every hypothesis.
  // This makes every propagation boundary informative without relying on RNG
  // winners or claiming that the incorrect candidate must win its comparison.
  for (const bool cooperative : {false, true}) {
    SCOPED_TRACE(cooperative);
    PatchMatchCuda patch_match(options, problem);
    const auto traces =
        internal::ProbeSweepSequence(patch_match, cooperative, 1873);
    ASSERT_EQ(traces.size(), 4);
    for (int orientation = 0; orientation < 4; ++orientation) {
      SCOPED_TRACE(orientation);
      ExpectPropagationAnchors(
          traces[orientation], inputs.images[0], orientation);
    }
  }
}

TEST_F(CudaTest, SameWindowScalarAndCooperativeSweep) {
  constexpr int kWidth = 35;
  constexpr unsigned long long kSeed = 9182;
  static_assert(kWidth % 8 != 0 && kWidth % 32 != 0);

  struct Case {
    int sources, samples, height;
    bool look_ahead;
  };
  const Case cases[] = {
      {8, 17, 257, true},
      {31, 33, 65, true},   // Last second-history slot is inactive, not source 31.
      {32, 15, 65, false},  // Default sample count and a partial final window.
      {32, 16, 65, false},  // A complete final window.
      {32, 17, 65, false},  // Retain the S>P fallback control.
      {32, 33, 65, true},   // Two source slots per worker; P exceeds 32.
      {33, 33, 65, false},  // Source-capacity fallback while S<=P.
  };
  for (const auto& c : cases) {
    const int sources = c.sources;
    const int kSamples = c.samples;
    const int kHeight = c.height;
    SCOPED_TRACE(::testing::Message()
                 << "sources=" << sources << " samples=" << kSamples
                 << " height=" << kHeight);
    auto inputs = MakeInputs(0, 2, kHeight, kWidth);
    // Repeating a real overlapping source makes the high IDs eligible for
    // sampling rather than placing later cameras outside the reference view.
    const Image source_image = inputs.images[2];
    const DepthMap source_depth = inputs.depths[2];
    const NormalMap source_normal = inputs.normals[2];
    inputs.images[1] = source_image;
    inputs.depths[1] = source_depth;
    inputs.normals[1] = source_normal;
    while (inputs.images.size() < static_cast<size_t>(sources) + 1) {
      inputs.images.push_back(source_image);
      inputs.depths.push_back(source_depth);
      inputs.normals.push_back(source_normal);
    }
    MakeSourceTexturesDistinct(&inputs);
    // Alternating non-ground-truth depths provide a useful propagation and
    // perturbation challenge; the source geometry remains valid.
    for (int row = 0; row < kHeight; ++row) {
      for (int col = 0; col < kWidth; ++col) {
        inputs.depths[0].Set(row, col, row % 2 == 0 ? 3.2f : 5.2f);
      }
    }
    PatchMatch::Problem problem;
    problem.ref_image_idx = 0;
    for (int i = 1; i <= sources; ++i) problem.src_image_idxs.push_back(i);
    problem.images = &inputs.images;
    problem.depth_maps = &inputs.depths;
    problem.normal_maps = &inputs.normals;
    auto options = MakeOptions(true, false, false);
    options.window_radius = 5;
    options.window_step = 1;
    options.sigma_spatial = 5.0;
    options.num_samples = kSamples;

    std::optional<internal::SweepTraceResult> geometry_control;
    if (c.look_ahead) {
      {
        PatchMatchCuda control(options, problem);
        geometry_control.emplace(internal::ProbeSweep(control, false, kSeed));
      }
      // Keep RGB/calibration identical to the control. Source-depth errors
      // must exercise the GEO table's source and current/random indices.
      AddDistinctSourceDepthErrors(&inputs);
    }

    PatchMatchCuda scalar(options, problem);
    PatchMatchCuda cooperative(options, problem);
    const auto a = internal::ProbeSweep(scalar, false, kSeed);
    const auto b = internal::ProbeSweep(cooperative, true, kSeed);
    ASSERT_EQ(a.draws.GetWidth(), kWidth);
    ASSERT_EQ(a.draws.GetHeight(), kHeight);
    ASSERT_EQ(a.draws.GetDepth(), kSamples);
    ASSERT_EQ(a.draws.GetData(), b.draws.GetData());
    ASSERT_EQ(a.winners.GetData(), b.winners.GetData());
    ASSERT_EQ(a.support.GetData(), b.support.GetData());
    ExpectExactlyEqual(a.initial_cost, b.initial_cost);
    ExpectExactlyEqual(a.hypotheses, b.hypotheses);
    ExpectExactlyEqual(a.hypothesis_costs, b.hypothesis_costs);
    ExpectExactlyEqual(a.forward, b.forward);
    ExpectExactlyEqual(a.cost, b.cost);
    ExpectExactlyEqual(a.depth, b.depth);
    ExpectExactlyEqual(a.normal, b.normal);
    ExpectExactlyEqual(a.selection, b.selection);
    if (geometry_control) ExpectNonVacuousGeometry(*geometry_control, a);
    for (const auto* map : {&a.selection, &b.selection}) {
      for (float value : map->GetData()) {
        ASSERT_TRUE(std::isfinite(value));
        ASSERT_GE(value, 0.0f);
        ASSERT_LE(value, 1.0f);
      }
    }
    ASSERT_EQ(a.continuation.GetWidth(), kWidth);
    ASSERT_EQ(a.continuation.GetDepth(), 3);
    ASSERT_EQ(a.continuation.GetData(), b.continuation.GetData());
    EXPECT_TRUE(std::any_of(a.continuation.GetData().begin(),
                            a.continuation.GetData().end(),
                            [](uint32_t draw) { return draw != 0; }));

    int supported_pixels = 0;
    int supported_rows = 0;
    int duplicate_pixels = 0;
    int cross_window_duplicate_pixels = 0;
    int high30 = 0;
    int high31 = 0;
    int high32 = 0;
    int source_30_14_distinct_sampled_cost_pixels = 0;
    int source_31_15_distinct_sampled_cost_pixels = 0;
    int strict_new_winners = 0;
    int propagated_earlier_window_sources = 0;
    int nontrivial_incumbent_pixels = 0;
    int nontrivial_second_pair_incumbent_pixels = 0;
    int distinct_sampled_costs = 0;
    int supported_band_boundary_pixels[2] = {0, 0};
    const int last_window_start = ((kSamples - 1) / 16) * 16;
    for (int row = 0; row < kHeight; ++row) {
      bool row_supported = false;
      for (int col = 0; col < kWidth; ++col) {
        const int supported = a.support.Get(row, col);
        ASSERT_TRUE(supported == 0 || supported == 1);
        supported_pixels += supported;
        if (row == 128) supported_band_boundary_pixels[0] += supported;
        if (row == 256) supported_band_boundary_pixels[1] += supported;
        row_supported = row_supported || supported;
        std::vector<int> first_window(sources, -1);
        std::vector<bool> in_final_window(sources, false);
        bool duplicate = false;
        bool cross_window_duplicate = false;
        int first_sampled_source = -1;
        bool distinct_sampled_cost = false;
        for (int sample = 0; sample < kSamples; ++sample) {
          const int source = a.draws.Get(row, col, sample);
          ASSERT_GE(source, -1);
          ASSERT_LT(source, sources);
          if (source >= 0) {
            ASSERT_EQ(supported, 1);
            if (sample < 16) {
              duplicate = duplicate || first_window[source] >= 0;
            }
            if (first_window[source] >= 0) {
              cross_window_duplicate =
                  cross_window_duplicate || first_window[source] < sample / 16;
            } else {
              first_window[source] = sample / 16;
            }
            if (sample >= last_window_start) in_final_window[source] = true;
            if (first_sampled_source < 0) {
              first_sampled_source = source;
            } else {
              distinct_sampled_cost = distinct_sampled_cost ||
                  a.initial_cost.Get(row, col, source) !=
                      a.initial_cost.Get(row, col, first_sampled_source);
            }
            high30 += source == 30;
            high31 += source == 31;
            high32 += source == 32;
          }
        }
        duplicate_pixels += duplicate;
        cross_window_duplicate_pixels += cross_window_duplicate;
        distinct_sampled_costs += distinct_sampled_cost;
        if (sources == 31 && first_window[30] >= 0 && first_window[14] >= 0 &&
            std::count_if(first_window.begin(), first_window.end(),
                          [](int window) { return window >= 0; }) > 16) {
          const float high_cost = a.initial_cost.Get(row, col, 30);
          const float other_cost = a.initial_cost.Get(row, col, 14);
          // Worker 14 has a live second history; worker 15 has none. Require
          // distinct useful values so a partial second-history row is tested.
          source_30_14_distinct_sampled_cost_pixels +=
              std::isfinite(high_cost) && std::isfinite(other_cost) &&
              high_cost < 1.9f && other_cost < 1.9f && high_cost != other_cost;
        }
        if (sources == 32 && c.look_ahead && first_window[31] >= 0 &&
            first_window[15] >= 0 &&
            std::count_if(first_window.begin(), first_window.end(),
                          [](int window) { return window >= 0; }) > 16) {
          const float high_cost = a.initial_cost.Get(row, col, 31);
          const float other_cost = a.initial_cost.Get(row, col, 15);
          // Join sampled source 31 with a distinct useful source cost and a
          // compact rank >=16. The RGB perturbation repeats every 17 IDs, so
          // source 14 would be an unsuitable comparison; source 15 differs.
          source_31_15_distinct_sampled_cost_pixels +=
              std::isfinite(high_cost) && std::isfinite(other_cost) &&
              high_cost < 1.9f && other_cost < 1.9f && high_cost != other_cost;
        }
        const int winner = a.winners.Get(row, col);
        ASSERT_GE(winner, 0);
        ASSERT_LT(winner, 5);
        if (supported && winner == 1) {
          for (int source = 0; source < sources; ++source) {
            // A whole-row cache can reuse this propagated result even though
            // it was absent from the final 16-draw window. This is a coverage
            // counter, not evidence that an evaluation was actually avoided.
            propagated_earlier_window_sources +=
                first_window[source] >= 0 && !in_final_window[source];
          }
        }
        if (supported && winner == 0 && row > 0) {
          bool nontrivial_forward = false;
          bool nontrivial_second_pair = false;
          for (int source = 0; source < sources; ++source) {
            const bool nontrivial = a.forward.Get(row - 1, col, source) != 0.5f;
            nontrivial_forward = nontrivial_forward || nontrivial;
            if (source >= 16 && nontrivial) {
              // The second source's pair must differ from the first pair
              // owned by this worker, not merely share a non-boundary value.
              nontrivial_second_pair = nontrivial_second_pair ||
                  a.forward.Get(row, col, source) !=
                      a.forward.Get(row, col, source - 16) ||
                  a.selection.Get(row, col, source) !=
                      a.selection.Get(row, col, source - 16);
            }
          }
          nontrivial_incumbent_pixels += nontrivial_forward;
          nontrivial_second_pair_incumbent_pixels += nontrivial_second_pair;
        }
        bool different = false;
        for (int component = 0; component < 4; ++component) {
          const float chosen =
              a.hypotheses.Get(row, col, winner * 4 + component);
          const float incumbent = a.hypotheses.Get(row, col, component);
          different = different || chosen != incumbent;
          const float published = component == 0
                                      ? a.depth.Get(row, col)
                                      : a.normal.Get(row, col, component - 1);
          // Preserve the selected tuple's exact bits, including signed zero.
          ASSERT_EQ(std::memcmp(&chosen, &published, sizeof(float)), 0);
        }
        if (supported && winner > 0 && different &&
            a.hypothesis_costs.Get(row, col, winner) <
                a.hypothesis_costs.Get(row, col, 0)) {
          ++strict_new_winners;
        }
      }
      supported_rows += row_supported;
    }
    EXPECT_GT(supported_pixels, 0);
    EXPECT_GT(supported_rows, 1);
    EXPECT_GT(duplicate_pixels, 0);
    if (sources == 31) {
      EXPECT_GT(high30, 0);
    }
    if (sources >= 32) {
      EXPECT_GT(high31, 0);
    }
    EXPECT_GT(distinct_sampled_costs, 0);
    if (c.look_ahead) {
      EXPECT_GT(cross_window_duplicate_pixels, 0);
      EXPECT_GT(propagated_earlier_window_sources, 0);
      // Incumbent reuse must consume nontrivial source history, not only
      // empty-support pixels or the initial 0.5 forward messages.
      EXPECT_GT(nontrivial_incumbent_pixels, 0);
    }
    if (sources == 8) {
      // S=8/P=17 selects look-ahead. Height 257 crosses both 128-row
      // boundaries; the same probe must compare all three bands.
      EXPECT_GT(supported_band_boundary_pixels[0], 0);
      EXPECT_GT(supported_band_boundary_pixels[1], 0);
    }
    if (sources == 31) {
      EXPECT_GT(source_30_14_distinct_sampled_cost_pixels, 0);
      EXPECT_GT(nontrivial_second_pair_incumbent_pixels, 0);
    }
    if (sources == 32 && c.look_ahead) {
      EXPECT_GT(source_31_15_distinct_sampled_cost_pixels, 0);
      EXPECT_GT(nontrivial_second_pair_incumbent_pixels, 0);
    }
    if (sources == 33) {
      EXPECT_GT(high32, 0);
    }
    EXPECT_GT(strict_new_winners, 0);
    const std::string property_prefix = "sources_" + std::to_string(sources) +
                                        "_samples_" + std::to_string(kSamples);
    RecordProperty(property_prefix + "_supported_pixels", supported_pixels);
    RecordProperty(property_prefix + "_duplicate_pixels", duplicate_pixels);
    RecordProperty(property_prefix + "_cross_window_duplicate_pixels",
                   cross_window_duplicate_pixels);
    RecordProperty(property_prefix + "_propagated_earlier_window_sources",
                   propagated_earlier_window_sources);
    RecordProperty(property_prefix + "_nontrivial_incumbent_pixels",
                   nontrivial_incumbent_pixels);
    RecordProperty(property_prefix + "_nontrivial_second_pair_incumbent_pixels",
                   nontrivial_second_pair_incumbent_pixels);
    RecordProperty(property_prefix + "_source_31_15_distinct_sampled_cost_pixels",
                   source_31_15_distinct_sampled_cost_pixels);
    RecordProperty(property_prefix + "_source_30_14_distinct_sampled_cost_pixels",
                   source_30_14_distinct_sampled_cost_pixels);
    RecordProperty(property_prefix + "_draws_30", high30);
    RecordProperty(property_prefix + "_draws_31", high31);
    RecordProperty(property_prefix + "_draws_32", high32);
    RecordProperty(property_prefix + "_strict_winners", strict_new_winners);
  }
}

TEST_F(CudaTest, SamplingSupportHelpersPreserveNativeDrawCount) {
  EXPECT_EQ(internal::ProbeSamplingSupport(), 0);
}

TEST_F(CudaTest, EmptySamplingSupportRetainsScalarAndCooperativeIncumbents) {
  int configurations = 0;
  for (const int radius : {2, 5}) {
    for (const int sources : {2, 33}) {
      for (const int samples : {1, 17}) {
        SCOPED_TRACE(::testing::Message() << "radius=" << radius << " sources="
                                          << sources << " samples=" << samples);
        auto inputs = MakeInputs(0, sources, 19, 23);
        for (auto& image : inputs.images) {
          Bitmap black(image.GetWidth(), image.GetHeight(), false);
          black.Fill(BitmapColor<uint8_t>(0));
          image.SetBitmap(std::move(black));
        }
        auto options = MakeOptions(true, false, false);
        options.window_radius = radius;
        options.num_samples = samples;
        // Constant patches have cost2; its FP32 likelihood exp(-512)
        // underflows to zero, giving finite empty sampling support.
        options.ncc_sigma = 1.0f / 16.0f;
        const auto result = RunPatchMatch(&inputs, options);
        ++configurations;
        ASSERT_EQ(result.depths.GetWidth(), 23);
        ASSERT_EQ(result.depths.GetHeight(), 19);
        ASSERT_EQ(result.depths.GetDepth(), 1);
        ASSERT_EQ(result.depths.GetData().size(), 23 * 19);
        ASSERT_EQ(result.normals.GetWidth(), 23);
        ASSERT_EQ(result.normals.GetHeight(), 19);
        ASSERT_EQ(result.normals.GetDepth(), 3);
        ASSERT_EQ(result.normals.GetData().size(), 23 * 19 * 3);
        ASSERT_EQ(result.selection.GetWidth(), 23);
        ASSERT_EQ(result.selection.GetHeight(), 19);
        ASSERT_EQ(result.selection.GetDepth(), sources);
        ASSERT_EQ(result.selection.GetData().size(), 23 * 19 * sources);
        ExpectExactlyEqual(inputs.depths[0], result.depths);
        ExpectExactlyEqual(inputs.normals[0], result.normals);
        EXPECT_TRUE(
            std::all_of(result.selection.GetData().begin(),
                        result.selection.GetData().end(),
                        [](float probability) { return probability == 0.0f; }));
      }
    }
  }
  EXPECT_EQ(configurations, 8);
  RecordProperty("empty_support_configurations", configurations);
}

TEST_F(CudaTest, PropagationPreservesCompletePlaneAcrossSweepRotations) {
  std::array<float, 4> K{800, 640, 750, 480};
  std::array<float, 3> normal{0.6f, 0.4f, -std::sqrt(0.48f)};
  int width = 1280;
  int height = 960;
  for (int rotation = 0; rotation < 4; ++rotation) {
    SCOPED_TRACE(rotation);
    const Eigen::Vector3d n(normal[0], normal[1], normal[2]);
    for (const float col : {0.1f * width, K[1], 0.86f * width}) {
      const float row1 = 0.68f * height;
      const float row2 = row1 + 1;
      const Eigen::Vector3d ray1(
          (double(col) - K[1]) / K[0], (double(row1) - K[3]) / K[2], 1);
      const Eigen::Vector3d ray2(
          (double(col) - K[1]) / K[0], (double(row2) - K[3]) / K[2], 1);
      ASSERT_LT(n.dot(ray1), 0);
      ASSERT_LT(n.dot(ray2), 0);
      for (const float depth : {0.0001f, 10.0f, 1000000.0f}) {
        SCOPED_TRACE(depth);
        const Eigen::Vector3d point1 = double(depth) * ray1;
        const double expected = n.dot(point1) / n.dot(ray2);
        const float actual =
            internal::ProbePropagatedDepth(K, depth, normal, row1, row2, col);
        ASSERT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, expected, 2e-6 * std::abs(expected));
        // A propagated candidate must still lie on the original 3D plane.
        EXPECT_NEAR(n.dot(double(actual) * ray2),
                    n.dot(point1),
                    2e-6 * std::abs(n.dot(point1)));
      }
    }
    const float previous_cx = K[1];
    K = {K[2], K[3], K[0], width - 1.0f - previous_cx};
    normal = {normal[1], -normal[0], normal[2]};
    std::swap(width, height);
  }
}

TEST_F(CudaTest, PropagationRetainsDepthForNearParallelRay) {
  const std::array<float, 4> K{1, 0, 1, 0};
  const std::array<float, 3> normal{0, -1, 0};
  for (const float row2 : {0.0f, 0.000005f}) {
    EXPECT_FLOAT_EQ(internal::ProbePropagatedDepth(K, 3, normal, 1, row2, 0),
                    3);
  }
  EXPECT_NEAR(internal::ProbePropagatedDepth(K, 3, normal, 1, 0.00002f, 0),
              150000.0f,
              1.0f);
}

TEST_F(CudaTest, PropagationRetainsDepthAcrossPlaneHorizon) {
  const std::array<float, 4> K{1000, 0, 1000, 0};
  Eigen::Vector3f n(0.7f, 0.7f, -0.70035f);
  n.normalize();
  const std::array<float, 3> normal{n.x(), n.y(), n.z()};
  ASSERT_LT(n.cast<double>().dot(Eigen::Vector3d(1, 0, 1)), 0);
  ASSERT_GT(n.cast<double>().dot(Eigen::Vector3d(1, 0.001, 1)), 0);
  EXPECT_FLOAT_EQ(internal::ProbePropagatedDepth(K, 10, normal, 0, 1, 1000),
                  10);
}

TEST_F(CudaTest, PropagationAvoidsUnrepresentableDepth) {
  const std::array<float, 4> K{1, 0, 1, 0};
  const float component = std::sqrt(0.5f);
  const std::array<float, 3> normal{0, -component, -component};
  const float depth = std::numeric_limits<float>::max();
  EXPECT_FLOAT_EQ(internal::ProbePropagatedDepth(K, depth, normal, 1, 0, 0),
                  depth);
  // Dividing the plane terms before multiplying avoids an intermediate
  // overflow when the resulting depth is still representable.
  const double expected = double(depth) * 2 / 3;
  EXPECT_NEAR(internal::ProbePropagatedDepth(K, depth, normal, 1, 2, 0),
              expected,
              2e-6 * expected);
}

TEST_F(CudaTest, SourcePackingPreservesDrawIdentityAndOrderedFold) {
  constexpr int kLanes = 8;
  for (const int sources : {32, 33}) {
    std::vector<float> values(sources);
    for (int source = 0; source < sources; ++source) {
      values[source] = (source % 7 + 1) * 0.125f;
    }
    values[0] = 1.0f;
    values[31] = std::ldexp(1.0f, -24);
    values[17] = 0.125f;
    values[1] = 0.75f;
    const auto check = [&](const std::vector<int>& pattern) {
      const size_t count = pattern.size();
      std::vector<int> draws(count * kLanes);
      std::vector<float> expected(kLanes, 0.0f);
      for (int lane = 0; lane < kLanes; ++lane) {
        for (size_t sample = 0; sample < count; ++sample) {
          const int original = pattern[sample];
          const int source = original < 0 ? -1 : (original + lane) % sources;
          draws[lane * count + sample] = source;
          // Independent scalar oracle: no packing, deduplication, ranking,
          // masks or alias mapping, and no production fold helper.
          if (source >= 0) expected[lane] += values[source];
        }
      }
      for (const auto& config :
           {std::pair{false, 1}, std::pair{true, 1}, std::pair{true, 16}}) {
        SCOPED_TRACE(::testing::Message()
                     << "sources=" << sources << " samples=" << count
                     << " cooperative=" << config.first
                     << " window=" << config.second);
        const auto actual = internal::ProbeSampledSources(
            draws, values, config.second, config.first);
        ASSERT_EQ(actual.first, draws);
        ASSERT_EQ(actual.second.size(), expected.size());
        for (int lane = 0; lane < kLanes; ++lane) {
          EXPECT_EQ(actual.second[lane], expected[lane]) << lane;
        }
      }
    };
    const std::array<int, 6> seed = {31, 0, 31, 17, -1, 1};
    for (const size_t samples : {15, 16, 17}) {
      std::vector<int> pattern(samples);
      for (size_t i = 0; i < samples; ++i) pattern[i] = seed[i % seed.size()];
      if (sources == 33) pattern[7] = 32;
      pattern.back() = -1;
      check(pattern);
    }
    check(std::vector<int>(17, -1));
    // Confirm that this fixture detects addition reordering in the valid,
    // finite nonnegative cost range instead of using only exact sums.
    float ordered = 0;
    for (const int source : {0, 31, 31}) ordered += values[source];
    float reordered = 0;
    for (const int source : {31, 31, 0}) reordered += values[source];
    ASSERT_NE(ordered, reordered);
    check({0, 31, 31});
  }
}

// Reuses the maintained analytic-image fixture and native public getters.
// Repeatability is only a regression check, not a scalar/cooperative oracle.
TEST_F(CudaTest, NativeDenseSingleReferenceVariedShapesAndModes) {
  struct Case {
    int sources, samples, width, height, radius, step;
    bool geo, filter;
  };
  const Case cases[] = {
      {1, 15, 47, 35, 5, 1, false, false},
      {17, 16, 39, 67, 5, 1, false, true},
      {32, 17, 35, 65, 5, 1, true, true},
      {33, 15, 17, 129, 5, 1, true, false},
      {2, 3, 39, 53, 2, 1, true, true},
      {2, 4, 35, 67, 5, 2, false, true},
      {1, 17, 17, 65, 17, 2, false, false},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message()
                 << c.sources << "/" << c.samples << " " << c.width << "x"
                 << c.height << " radius=" << c.radius << " step=" << c.step
                 << " geo=" << c.geo << " filter=" << c.filter);
    Inputs inputs = MakeInputs(0, c.sources, c.height, c.width);
    auto options = MakeOptions(c.geo, c.filter, false);
    options.window_radius = c.radius;
    options.window_step = c.step;
    options.num_samples = c.samples;
    const Result first = RunPatchMatch(&inputs, options);
    const Result repeated = RunPatchMatch(&inputs, options);
    ExpectNonempty(first, inputs, c.filter);
    ExpectNonempty(repeated, inputs, c.filter);
    ExpectEquivalent(first, repeated);
  }
}

TEST_F(CudaTest, InvalidSourceDepthIsExcludedWithoutDiscardingUsefulViews) {
  auto inputs = MakeInputs(0, 6);
  auto options = MakeOptions(true, true, false);
  const PatchMatchOptions defaults;
  options.window_radius = 5;
  options.num_samples = 17;  // S=6 must enter the look-ahead/filter-reuse path.
  options.sigma_spatial = defaults.sigma_spatial;
  options.geom_consistency_max_cost = defaults.geom_consistency_max_cost;
  options.filter_geom_consistency_max_cost =
      defaults.filter_geom_consistency_max_cost;
  const std::vector<int> order{3, 6, 1, 5, 2, 4};
  const Result valid = RunPatchMatch(&inputs, options, order);
  ExpectNonempty(valid, inputs, true);
  size_t source_six_memberships = 0;
  for (const auto& members : valid.consistent_images)
    source_six_memberships += std::count(members.begin(), members.end(), 6);
  ASSERT_GT(source_six_memberships, 0)
      << "The all-valid control must observe source 6";
  for (size_t row = 0; row < inputs.depths[6].GetHeight(); ++row) {
    for (size_t col = 0; col < inputs.depths[6].GetWidth(); ++col)
      inputs.depths[6].Set(row, col, 0.0f);
  }
  const Result result = RunPatchMatch(&inputs, options, order);
  ExpectNonempty(result, inputs, true);
  size_t useful_memberships = 0;
  for (const auto& members : result.consistent_images) {
    for (const int source : members) {
      EXPECT_NE(source, 6);
      ++useful_memberships;
    }
  }
  EXPECT_GT(useful_memberships, 0);
}

TEST_F(CudaTest, AllInvalidSourceDepthsPublishSizedRejectedOutputs) {
  auto inputs = MakeInputs(0);
  for (size_t source = 1; source < inputs.depths.size(); ++source) {
    for (size_t row = 0; row < inputs.depths[source].GetHeight(); ++row) {
      for (size_t col = 0; col < inputs.depths[source].GetWidth(); ++col)
        inputs.depths[source].Set(row, col, 0.0f);
    }
  }
  auto options = MakeOptions(true, true, false);
  const PatchMatchOptions defaults;
  options.window_radius = 5;
  options.sigma_spatial = defaults.sigma_spatial;
  options.geom_consistency_max_cost = defaults.geom_consistency_max_cost;
  options.filter_geom_consistency_max_cost =
      defaults.filter_geom_consistency_max_cost;
  const Result result = RunPatchMatch(&inputs, options);
  ASSERT_EQ(result.depths.GetWidth(), inputs.images[0].GetWidth());
  ASSERT_EQ(result.depths.GetHeight(), inputs.images[0].GetHeight());
  ASSERT_GT(result.depths.GetData().size(), 0);
  ASSERT_EQ(result.normals.GetWidth(), result.depths.GetWidth());
  ASSERT_EQ(result.normals.GetHeight(), result.depths.GetHeight());
  ASSERT_EQ(result.normals.GetDepth(), 3);
  ASSERT_EQ(result.selection.GetWidth(), result.depths.GetWidth());
  ASSERT_EQ(result.selection.GetHeight(), result.depths.GetHeight());
  ASSERT_EQ(result.selection.GetDepth(), inputs.images.size() - 1);
  ASSERT_EQ(result.normals.GetData().size(),
            result.depths.GetData().size() * 3);
  ASSERT_EQ(result.selection.GetData().size(),
            result.depths.GetData().size() * (inputs.images.size() - 1));
  ASSERT_EQ(result.consistent_images.size(), result.depths.GetData().size());
  for (const float depth : result.depths.GetData()) EXPECT_EQ(depth, 0.0f);
  for (const float normal : result.normals.GetData()) EXPECT_EQ(normal, 0.0f);
  for (const float probability : result.selection.GetData()) {
    EXPECT_TRUE(std::isfinite(probability));
    EXPECT_GE(probability, 0.0f);
    EXPECT_LE(probability, 1.0f);
  }
  for (const auto& members : result.consistent_images)
    EXPECT_TRUE(members.empty());
}
TEST_F(CudaTest, LookAheadPreservesUnfilteredCarryAcrossBands) {
  constexpr int kWidth = 35;
  constexpr int kHeight = 257;
  constexpr int kSources = 8;
  constexpr unsigned long long kSeed = 9182;
  auto inputs = MakeInputs(0, 2, kHeight, kWidth);
  const Image source_image = inputs.images[2];
  const DepthMap source_depth = inputs.depths[2];
  const NormalMap source_normal = inputs.normals[2];
  inputs.images[1] = source_image;
  inputs.depths[1] = source_depth;
  inputs.normals[1] = source_normal;
  while (inputs.images.size() < kSources + 1) {
    inputs.images.push_back(source_image);
    inputs.depths.push_back(source_depth);
    inputs.normals.push_back(source_normal);
  }
  MakeSourceTexturesDistinct(&inputs);
  AddDistinctSourceDepthErrors(&inputs);
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      inputs.depths[0].Set(row, col, row % 2 == 0 ? 3.2f : 5.2f);
    }
  }
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  for (int i = 1; i <= kSources; ++i) problem.src_image_idxs.push_back(i);
  problem.images = &inputs.images;
  problem.depth_maps = &inputs.depths;
  problem.normal_maps = &inputs.normals;
  auto options = MakeOptions(true, true, false);
  options.window_radius = 5;
  options.window_step = 1;
  options.sigma_spatial = 5.0;
  options.num_samples = 17;
  // Legal under Check(); more required sources than exist forces rejection.
  options.filter_min_num_consistent = kSources + 1;
  ASSERT_TRUE(options.Check());

  PatchMatchCuda scalar(options, problem);
  PatchMatchCuda cooperative(options, problem);
  const auto a = internal::ProbeSweep(scalar, false, kSeed);
  const auto b = internal::ProbeSweep(cooperative, true, kSeed);
  EXPECT_EQ(a.draws.GetData(), b.draws.GetData());
  EXPECT_EQ(a.winners.GetData(), b.winners.GetData());
  EXPECT_EQ(a.support.GetData(), b.support.GetData());
  EXPECT_EQ(a.continuation.GetData(), b.continuation.GetData());
  ExpectExactlyEqual(a.initial_cost, b.initial_cost);
  ExpectExactlyEqual(a.hypotheses, b.hypotheses);
  ExpectExactlyEqual(a.hypothesis_costs, b.hypothesis_costs);
  ExpectExactlyEqual(a.forward, b.forward);
  ExpectExactlyEqual(a.cost, b.cost);
  ExpectExactlyEqual(a.selection, b.selection);
  ExpectExactlyEqual(a.depth, b.depth);
  ExpectExactlyEqual(a.normal, b.normal);
  EXPECT_TRUE(std::all_of(b.depth.GetData().begin(), b.depth.GetData().end(),
                         [](float value) { return value == 0.0f; }));
  EXPECT_TRUE(std::all_of(b.normal.GetData().begin(), b.normal.GetData().end(),
                         [](float value) { return value == 0.0f; }));
  for (const int row : {128, 256}) {
    int positive_propagations = 0;
    for (int col = 0; col < kWidth; ++col) {
      EXPECT_EQ(b.depth.Get(row - 1, col), 0.0f);
      positive_propagations += b.hypotheses.Get(row, col, 4) > 0.0f;
    }
    // Reloading the filtered previous row would instead propagate zero.
    EXPECT_GT(positive_propagations, 0) << "band starting at row " << row;
  }
}


TEST_F(CudaTest, SelectionProbabilityCapturedTupleAndDomain) {
  // Captured alpha, beta, previous and weight, retained as exact binary32 bits.
  const std::vector<uint32_t> input = {
      0x3f7ff9b0, 0x3f7ff844, 0x3f000000, 0x00000000, 0x00000000, 0x00000000,
      0x3f000000, 0x00000000, 0x3f800000, 0x3f800000, 0x3f000000, 0x00000000,
      0x3f000000, 0x3f000000, 0x3f000000, 0x3e800000, 0x3f800000, 0x3f800000,
      0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x3f000000, 0x00000000,
      0x7fc00000, 0x3f000000, 0x3f000000, 0x00000000, 0xbe800000, 0x3f000000,
      0x3f000000, 0x00000000};
  const auto out = internal::ProbeSelectionProbability(input);
  ASSERT_EQ(out.size(), 8);
  EXPECT_EQ(out[0], 0x3f800000u);  // Captured valid tuple: strict <= 1.
  EXPECT_EQ(out[1], 0x00000000u);
  EXPECT_EQ(out[2], 0x3f800000u);
  EXPECT_EQ(out[3], 0x3f000000u);
  EXPECT_EQ(out[4], 0x00000000u);
  EXPECT_GT(out[5] & 0x7fffffffu, 0x7f800000u);  // Singular 0/0 remains NaN.
  EXPECT_GT(out[6] & 0x7fffffffu, 0x7f800000u);  // NaN input remains NaN.
  EXPECT_EQ(out[7], 0xbe800000u);  // Invalid input is not silently clamped.
  std::vector<uint32_t> many;
  for (int i = 0; i < 40; ++i) {
    many.insert(many.end(), input.begin(), input.begin() + 4);
  }
  const auto many_out = internal::ProbeSelectionProbability(many);
  ASSERT_EQ(many_out.size(), 40);
  for (const auto value : many_out) EXPECT_EQ(value, 0x3f800000u);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
