// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/packed_consistency_mask.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_rng_perimeter.h"
#include "colmap/util/cudacc.h"

#include <gtest/gtest.h>

#ifndef COLMAP_HIP_ENABLED
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#endif

namespace colmap {
namespace mvs {
namespace {

#ifdef COLMAP_HIP_ENABLED
TEST(PatchMatchCuda, RotationBufferReuseRequiresCuda) {
  GTEST_SKIP() << "The rotation-buffer option is CUDA-only";
}
#else

TEST(PatchMatchCuda, PackedConsistencyMaskDepthCoversSourceTails) {
  for (const auto& [sources, expected] :
       std::array<std::pair<std::size_t, std::size_t>, 8>{{{0, 0},
                                                           {1, 1},
                                                           {7, 1},
                                                           {8, 1},
                                                           {9, 2},
                                                           {17, 3},
                                                           {32, 4},
                                                           {33, 5}}}) {
    EXPECT_EQ(PackedConsistencyMaskDepth(sources), expected) << sources;
  }
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(PackedConsistencyMaskDepth(kMax), kMax / 8 + 1);
}

TEST(PatchMatchCuda, RngPerimeterLayoutPreservesRotatedRowZero) {
  const std::array<std::pair<std::size_t, std::size_t>, 7> shapes = {
      {{1, 1}, {1, 9}, {9, 1}, {2, 7}, {7, 2}, {33, 17}, {47, 35}}};
  for (const auto& [width, height] : shapes) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    const PatchMatchRngPerimeter layout{width, height};
    const std::size_t expected_count =
        width == 1 ? height
                   : (height == 1 ? width : 2 * width + 2 * height - 4);
    ASSERT_EQ(layout.Count(), expected_count);
    std::vector<bool> seen(expected_count, false);
    using Row = std::vector<std::size_t>;
    std::vector<Row> rotated(height, Row(width));
    for (std::size_t row = 0; row < height; ++row) {
      for (std::size_t col = 0; col < width; ++col) {
        rotated[row][col] = row * width + col;
        if (row != 0 && col != 0 && row + 1 != height && col + 1 != width) {
          continue;
        }
        const std::size_t index = layout.Index(row, col);
        ASSERT_LT(index, expected_count);
        EXPECT_FALSE(seen[index]);
        seen[index] = true;
        std::size_t recovered_row = 0, recovered_col = 0;
        layout.Coordinates(index, &recovered_row, &recovered_col);
        EXPECT_EQ(recovered_row, row);
        EXPECT_EQ(recovered_col, col);
      }
    }
    for (const bool visited : seen) EXPECT_TRUE(visited);

    for (unsigned rotation = 0; rotation < 4; ++rotation) {
      const std::size_t current_height = rotated.size();
      const std::size_t current_width = rotated.front().size();
      std::vector<bool> active(expected_count, false);
      for (std::size_t col = 0; col < current_width; ++col) {
        const std::size_t index = layout.SweepIndex(rotation, col);
        ASSERT_LT(index, expected_count);
        EXPECT_FALSE(active[index]);
        active[index] = true;
        std::size_t original_row = 0, original_col = 0;
        layout.Coordinates(index, &original_row, &original_col);
        EXPECT_EQ(original_row * width + original_col, rotated[0][col]);
      }
      std::vector<Row> next(current_width, Row(current_height));
      for (std::size_t row = 0; row < current_height; ++row) {
        for (std::size_t col = 0; col < current_width; ++col) {
          next[current_width - 1 - col][row] = rotated[row][col];
        }
      }
      rotated.swap(next);
    }
    EXPECT_EQ(layout.SweepIndex(0, width - 1), layout.SweepIndex(1, 0));
    EXPECT_EQ(layout.SweepIndex(1, height - 1), layout.SweepIndex(2, 0));
    EXPECT_EQ(layout.SweepIndex(2, width - 1), layout.SweepIndex(3, 0));
    EXPECT_EQ(layout.SweepIndex(3, height - 1), layout.SweepIndex(0, 0));
  }
}

constexpr float kPlaneZ = 4.0f;
constexpr float kDepthMin = 3.0f;
constexpr float kDepthMax = 5.0f;

struct Inputs {
  std::vector<Image> images;
  std::vector<DepthMap> depths;
  std::vector<NormalMap> normals;
};

// Two independent reference problems, each with differently calibrated source
// views. All images are rendered from the same textured world plane. Sizes are
// deliberately non-square, odd, and not CUDA block multiples.
Inputs MakeInputs(const int variant, const int num_source_images = 2) {
  Inputs inputs;
  for (int i = 0; i <= num_source_images; ++i) {
    const int width = (variant == 0 ? 47 : 39) + 2 * i;
    const int height = (variant == 0 ? 35 : 53) + 2 * i;
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

PatchMatchOptions MakeOptions(const bool geometry,
                              const bool filter,
                              const bool reuse) {
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
  options.rotation_buffer_reuse = reuse;
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
  std::vector<std::vector<int> > consistent_images;
};

Result RunPatchMatch(Inputs* inputs, const PatchMatchOptions& options) {
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  for (size_t i = 1; i < inputs->images.size(); ++i) {
    problem.src_image_idxs.push_back(static_cast<int>(i));
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

// Parameter order: geometric consistency, filtering.
class PatchMatchCudaTest
    : public CudaTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool> > {};

class SourceUploadTest : public CudaTest,
                         public ::testing::WithParamInterface<bool> {};

TEST_P(SourceUploadTest, DirectUploadPreservesAlgorithmOutputs) {
  Inputs inputs = MakeInputs(0);
  PatchMatchOptions options = MakeOptions(GetParam(), true, false);
  options.direct_source_upload = false;
  const Result padded = RunPatchMatch(&inputs, options);
  options.direct_source_upload = true;
  const Result direct = RunPatchMatch(&inputs, options);
  ExpectNonempty(padded, inputs, true);
  ExpectNonempty(direct, inputs, true);
  ExpectEquivalent(padded, direct);
}

TEST_F(CudaTest, ExplicitAndDefaultStreamHelpers) {
  // Preserve cleanup even when a CUDA allocation or operation assertion fails.
  struct Resources {
    cudaStream_t stream = nullptr;
    uint8_t* device = nullptr;
    uint8_t* host = nullptr;
    ~Resources() {
      if (stream != nullptr) {
        cudaStreamSynchronize(stream);
      }
      cudaStreamSynchronize(CudaDefaultStream());
      if (host != nullptr) {
        cudaFreeHost(host);
      }
      if (device != nullptr) {
        cudaFree(device);
      }
      if (stream != nullptr) {
        cudaStreamDestroy(stream);
      }
    }
  } resources;
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&resources.device), 1),
            cudaSuccess);
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void**>(&resources.host), 1),
            cudaSuccess);
  {
    CudaTimer timer(resources.stream);
    ASSERT_EQ(cudaMemsetAsync(resources.device, 0x5a, 1, resources.stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(resources.host,
                              resources.device,
                              1,
                              cudaMemcpyDeviceToHost,
                              resources.stream),
              cudaSuccess);
    CudaSyncAndCheck(__FILE__, __LINE__, resources.stream);
    EXPECT_EQ(resources.host[0], 0x5a);
    timer.Print("Explicit nonblocking stream helper test");
  }
  EXPECT_EQ(CudaDefaultStream(), cudaStreamPerThread);
  {
    CudaTimer timer;
    ASSERT_EQ(cudaMemsetAsync(resources.device, 0xa5, 1, CudaDefaultStream()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(resources.host,
                              resources.device,
                              1,
                              cudaMemcpyDeviceToHost,
                              CudaDefaultStream()),
              cudaSuccess);
    CudaSyncAndCheck(__FILE__, __LINE__);
    EXPECT_EQ(resources.host[0], 0xa5);
    timer.Print("Per-thread default stream helper test");
  }
}

TEST_F(CudaTest, WideWindowsPreserveOutputs) {
  Inputs inputs = MakeInputs(0);
  // Cover both boundaries between the 8-, 16-, and 32-column tiles.
  for (const int radius : {8, 9, 16, 17}) {
    SCOPED_TRACE(radius);
    PatchMatchOptions options = MakeOptions(false, false, false);
    options.window_radius = radius;
    options.num_iterations = 1;
    options.num_samples = 1;
    const Result original = RunPatchMatch(&inputs, options);
    options.rotation_buffer_reuse = true;
    const Result reused = RunPatchMatch(&inputs, options);
    ExpectNonempty(original, inputs, false);
    ExpectNonempty(reused, inputs, false);
    ExpectEquivalent(original, reused);
  }
}

TEST_F(CudaTest, GeometricCostGroupsPreserveOutputs) {
  Inputs inputs = MakeInputs(0, 5);
  // Cover shared geometric costs in both narrow and wide cooperative tiles.
  for (const int radius : {5, 17}) {
    for (const bool filter : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "radius=" << radius << ", filter=" << filter);
      PatchMatchOptions options = MakeOptions(true, filter, false);
      options.window_radius = radius;
      options.num_iterations = 1;
      options.num_samples = 1;
      // The wide case targets shared-result safety rather than patch workload.
      if (radius == 17) options.window_step = 2;
      const Result original = RunPatchMatch(&inputs, options);
      options.rotation_buffer_reuse = true;
      const Result reused = RunPatchMatch(&inputs, options);
      ExpectNonempty(original, inputs, filter);
      ExpectNonempty(reused, inputs, filter);
      ExpectEquivalent(original, reused);
    }
  }
}

TEST_F(CudaTest, EvenSampleGroupsPreserveOutputs) {
  // One source forces duplicate picks; seventeen cover all groups and a wrap.
  for (const int num_sources : {1, 17}) {
    SCOPED_TRACE(num_sources);
    Inputs inputs = MakeInputs(0, num_sources);
    for (const int num_samples : {2, 4}) {
      SCOPED_TRACE(num_samples);
      PatchMatchOptions options = MakeOptions(true, true, false);
      options.num_iterations = 1;
      options.num_samples = num_samples;
      const Result original = RunPatchMatch(&inputs, options);
      options.rotation_buffer_reuse = true;
      const Result reused = RunPatchMatch(&inputs, options);
      ExpectNonempty(original, inputs, true);
      ExpectNonempty(reused, inputs, true);
      ExpectEquivalent(original, reused);
    }
  }
}

TEST_F(CudaTest, SampleWindowBoundariesPreserveOutputs) {
  // Cover source-mask capacity and its larger-source fallback.
  for (const int num_sources : {1, 17, 32, 33}) {
    SCOPED_TRACE(num_sources);
    Inputs inputs = MakeInputs(0, num_sources);
    for (const int num_samples : {16, 17}) {
      SCOPED_TRACE(num_samples);
      PatchMatchOptions options = MakeOptions(true, true, false);
      options.num_iterations = 1;
      options.num_samples = num_samples;
      const Result original = RunPatchMatch(&inputs, options);
      options.rotation_buffer_reuse = true;
      const Result reused = RunPatchMatch(&inputs, options);
      ExpectNonempty(original, inputs, true);
      ExpectNonempty(reused, inputs, true);
      ExpectEquivalent(original, reused);
    }
  }
}

TEST_P(PatchMatchCudaTest, UnevenSourceGroupsPreserveOutputs) {
  const auto [geometry, filter] = GetParam();
  // Five sources exercise every cooperative group and a second source in one
  // group. Odd image sizes also leave invalid lanes at the tile boundary.
  Inputs inputs = MakeInputs(0, 5);
  PatchMatchOptions options = MakeOptions(geometry, filter, false);
  const Result original = RunPatchMatch(&inputs, options);
  options.rotation_buffer_reuse = true;
  const Result reused = RunPatchMatch(&inputs, options);
  ExpectNonempty(original, inputs, filter);
  ExpectNonempty(reused, inputs, filter);
  ExpectEquivalent(original, reused);
  // Allow FP32 rounding at the probability bounds; parity above stays exact.
  constexpr float kProbabilityTolerance = 1e-6f;
  for (const float probability : original.selection.GetData()) {
    EXPECT_GE(probability, -kProbabilityTolerance);
    EXPECT_LE(probability, 1.0f + kProbabilityTolerance);
  }
}

TEST_P(PatchMatchCudaTest, RotationBufferReusePreservesAlgorithmOutputs) {
  const auto [geometry, filter] = GetParam();
  Inputs inputs = MakeInputs(0);
  const Result legacy =
      RunPatchMatch(&inputs, MakeOptions(geometry, filter, false));
  const Result reused =
      RunPatchMatch(&inputs, MakeOptions(geometry, filter, true));
  ExpectNonempty(legacy, inputs, filter);
  ExpectNonempty(reused, inputs, filter);
  ExpectEquivalent(legacy, reused);
}

TEST_P(PatchMatchCudaTest, SeparateCalibrationsMatchSerialOnSameDevice) {
  const auto [geometry, filter] = GetParam();
  for (const bool reuse : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "rotation_buffer_reuse=" << reuse);
    const PatchMatchOptions options = MakeOptions(geometry, filter, reuse);
    Inputs serial_a = MakeInputs(0);
    Inputs serial_b = MakeInputs(1);
    ASSERT_NE(serial_a.images[0].GetK()[0], serial_b.images[0].GetK()[0]);
    ASSERT_NE(serial_a.images[0].GetK()[4], serial_b.images[0].GetK()[4]);
    const Result expected_a = RunPatchMatch(&serial_a, options);
    const Result expected_b = RunPatchMatch(&serial_b, options);
    ExpectNonempty(expected_a, serial_a, filter);
    ExpectNonempty(expected_b, serial_b, filter);
    // A bounded repetition increases the opportunity for interleaving without
    // sleeps, external datasets, hooks into kernels, or a stress-test workload.
    for (int repetition = 0; repetition < 2; ++repetition) {
      SCOPED_TRACE(repetition);
      Inputs concurrent_a = MakeInputs(0);
      Inputs concurrent_b = MakeInputs(1);
      std::promise<void> start;
      std::promise<void> entered_a;
      std::promise<void> entered_b;
      const std::shared_future<void> ready = start.get_future().share();
      auto a = std::async(std::launch::async, [&]() {
        entered_a.set_value();
        ready.wait();
        return RunPatchMatch(&concurrent_a, options);
      });
      auto b = std::async(std::launch::async, [&]() {
        entered_b.set_value();
        ready.wait();
        return RunPatchMatch(&concurrent_b, options);
      });
      entered_a.get_future().wait();
      entered_b.get_future().wait();
      start.set_value();
      const Result actual_a = a.get();
      const Result actual_b = b.get();
      ExpectNonempty(actual_a, concurrent_a, filter);
      ExpectNonempty(actual_b, concurrent_b, filter);
      ExpectEquivalent(expected_a, actual_a);
      ExpectEquivalent(expected_b, actual_b);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(PhotoAndGeometry,
                         PatchMatchCudaTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));
INSTANTIATE_TEST_SUITE_P(PhotoAndGeometry, SourceUploadTest, ::testing::Bool());
#endif

}  // namespace
}  // namespace mvs
}  // namespace colmap
