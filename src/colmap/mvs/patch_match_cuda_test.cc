// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/packed_consistency_mask.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_hmm_checkpoint.h"
#include "colmap/mvs/patch_match_rng_perimeter.h"
#include "colmap/mvs/patch_match_source_array.h"
#include "colmap/util/cudacc.h"

#include <gtest/gtest.h>

#ifndef COLMAP_HIP_ENABLED
#include "colmap/mvs/patch_match_cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
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

TEST(PatchMatchCuda, CheckpointHmmStripeRowsRoundsUpWithoutOverflow) {
  const std::array<std::size_t, 17> heights = {
      1,
      2,
      3,
      4,
      5,
      8,
      9,
      10,
      17,
      35,
      47,
      129,
      1024,
      1536,
      2048,
      3072,
      std::numeric_limits<std::size_t>::max()};
  for (const std::size_t height : heights) {
    SCOPED_TRACE(height);
    const std::size_t rows = PatchMatchHmmCheckpoint::StripeRows(height);
    ASSERT_GT(rows, 0);
    // Characterize ceil(sqrt(height)) without squaring and overflowing.
    EXPECT_GE(rows, height / rows + (height % rows != 0));
    if (rows > 1) {
      const std::size_t previous = rows - 1;
      EXPECT_LT(previous, height / previous + (height % previous != 0));
    }
  }
}

TEST(PatchMatchCuda, CheckpointHmmRejectsInvalidAllocationShapes) {
  // Every rejection occurs before attempting a device allocation.
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint::StripeRows(0));
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint(0, 1, 1));
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint(1, 0, 1));
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint(
      std::size_t(std::numeric_limits<int>::max()) + 1, 1, 1));
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint(
      128, 128, std::numeric_limits<std::size_t>::max()));
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
      for (std::size_t row = 0; row < current_height; ++row) {
        for (std::size_t col = 0; col < current_width; ++col) {
          std::size_t original_row = 0, original_col = 0;
          layout.OriginalCoordinates(
              rotation, row, col, &original_row, &original_col);
          ASSERT_LT(original_row, height);
          ASSERT_LT(original_col, width);
          EXPECT_EQ(original_row * width + original_col, rotated[row][col]);
        }
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
  // Keep full-storage controls explicit when the public CUDA default changes.
  options.checkpoint_hmm = false;
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

TEST_F(CudaTest, NondefaultFilteredPhotoPreservesRotatedStatistics) {
  Inputs inputs = MakeInputs(0, 4, 77, 95);
  auto options = MakeOptions(false, true, false);
  options.window_radius = 13;  // Full27/step2 uses the original float planes.
  options.window_step = 2;
  options.num_samples = PatchMatchOptions{}.num_samples;
  const Result native_rotation = RunPatchMatch(&inputs, options);
  options.rotation_buffer_reuse = true;
  const Result retained_rotation = RunPatchMatch(&inputs, options);
  ExpectNonempty(native_rotation, inputs, true);
  ExpectNonempty(retained_rotation, inputs, true);
  ExpectEquivalent(native_rotation, retained_rotation);
}

TEST_F(CudaTest, IntegerStatisticsArrayPreservesChannelBits) {
  // Exercise the existing typed-array wrapper with special FP32 representations
  // carried as integers. This is array/descriptor coverage; the compiled map
  // oracle must additionally cover production packing and device fetches.
  const std::array<uint2, 8> input = {{
      make_uint2(0x00000000u, 0x80000000u),
      make_uint2(0x00000001u, 0x80000001u),
      make_uint2(0x007fffffu, 0x807fffffu),
      make_uint2(0x00800000u, 0x80800000u),
      make_uint2(0x3f800000u, 0xbf800000u),
      make_uint2(0x7f7fffffu, 0xff7fffffu),
      make_uint2(0x7f800000u, 0xff800000u),
      make_uint2(0x7fc12345u, 0x7fa12345u),
  }};
  cudaTextureDesc descriptor = {};
  descriptor.addressMode[0] = cudaAddressModeClamp;
  descriptor.addressMode[1] = cudaAddressModeClamp;
  descriptor.filterMode = cudaFilterModePoint;
  descriptor.readMode = cudaReadModeElementType;
  descriptor.normalizedCoords = 0;
  auto texture = CudaArrayLayeredTexture<uint2>::FromHostArray(
      descriptor, 4, 2, 1, input.data());
  cudaTextureDesc actual = {};
  CUDA_SAFE_CALL(cudaGetTextureObjectTextureDesc(&actual, texture->GetObj()));
  EXPECT_EQ(actual.filterMode, cudaFilterModePoint);
  EXPECT_EQ(actual.readMode, cudaReadModeElementType);
  EXPECT_EQ(actual.normalizedCoords, 0);
  cudaResourceDesc resource = {};
  CUDA_SAFE_CALL(
      cudaGetTextureObjectResourceDesc(&resource, texture->GetObj()));
  ASSERT_EQ(resource.resType, cudaResourceTypeArray);
  cudaChannelFormatDesc channels = {};
  cudaExtent extent = {};
  unsigned flags = 0;
  CUDA_SAFE_CALL(
      cudaArrayGetInfo(&channels, &extent, &flags, resource.res.array.array));
  EXPECT_EQ(channels.x, 32);
  EXPECT_EQ(channels.y, 32);
  EXPECT_EQ(channels.z, 0);
  EXPECT_EQ(channels.w, 0);
  EXPECT_EQ(channels.f, cudaChannelFormatKindUnsigned);
  EXPECT_NE(flags & cudaArrayLayered, 0);
  EXPECT_EQ(extent.width, 4);
  EXPECT_EQ(extent.height, 2);
  EXPECT_EQ(extent.depth, 1);
  std::array<uint2, 8> output = {};
  cudaMemcpy3DParms copy = {};
  copy.srcArray = resource.res.array.array;
  copy.dstPtr = make_cudaPitchedPtr(
      output.data(), 4 * sizeof(uint2), 4 * sizeof(uint2), 2);
  copy.extent = make_cudaExtent(4, 2, 1);
  copy.kind = cudaMemcpyDeviceToHost;
  CUDA_SAFE_CALL(cudaMemcpy3D(&copy));
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(input[i].x, output[i].x);
    EXPECT_EQ(input[i].y, output[i].y);
  }
}

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

void ExpectPairedDefaultSweepOutputs(const bool reuse,
                                     const bool tall = false) {
  const auto make_problem = [](Inputs& inputs) {
    PatchMatch::Problem problem;
    problem.ref_image_idx = 0;
    for (size_t i = 1; i < inputs.images.size(); ++i) {
      problem.src_image_idxs.push_back(static_cast<int>(i));
    }
    problem.images = &inputs.images;
    problem.depth_maps = &inputs.depths;
    problem.normal_maps = &inputs.normals;
    return problem;
  };
  const auto read = [](const PatchMatchCuda& cuda) {
    return std::make_tuple(cuda.GetDepthMap(),
                           cuda.GetNormalMap(),
                           cuda.GetSelProbMap(),
                           cuda.GetConsistentImageIdxs());
  };
  struct Streams {
    cudaStream_t first = nullptr, second = nullptr;
    ~Streams() {
      if (second) (void)cudaStreamDestroy(second);
      if (first) (void)cudaStreamDestroy(first);
    }
  };
  for (const bool geometry : {false, true}) {
    SCOPED_TRACE(::testing::Message()
                 << "geometry=" << geometry << " reuse=" << reuse);
    // Odd, differently calibrated references and nine sources exercise both
    // orientations and a final consistency-mask byte tail.
    Inputs first_inputs = MakeInputs(0, 9, tall ? 149 : 0);
    Inputs second_inputs = MakeInputs(1, 9, tall ? 153 : 0);
    auto options = MakeOptions(geometry, geometry, reuse);
    const PatchMatchOptions defaults;
    options.window_radius = defaults.window_radius;
    options.window_step = defaults.window_step;
    options.num_samples = defaults.num_samples;
    options.num_iterations = defaults.num_iterations;
    options.sigma_spatial = options.window_radius;
    auto first_problem = make_problem(first_inputs);
    auto second_problem = make_problem(second_inputs);
    const auto serial = [&](const PatchMatch::Problem& problem) {
      PatchMatchCuda cuda(options, problem);
      cuda.Run();
      return read(cuda);
    };
    const auto expected_first = serial(first_problem);
    const auto expected_second = serial(second_problem);
    if (reuse && geometry) {
      const auto& first_mask = std::get<3>(expected_first);
      const auto& second_mask = std::get<3>(expected_second);
      EXPECT_FALSE(first_mask.empty() && second_mask.empty());
    }
    PatchMatchCuda first(options, first_problem);
    PatchMatchCuda second(options, second_problem);
    Streams streams;
    ASSERT_EQ(cudaStreamCreateWithFlags(&streams.first, cudaStreamNonBlocking),
              cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&streams.second, cudaStreamNonBlocking),
              cudaSuccess);
    RunPatchMatchPair(first, second, streams.first, streams.second);
    const auto compare = [&](const auto& expected, const PatchMatchCuda& cuda) {
      const auto actual = read(cuda);
      ExpectExactlyEqual(std::get<0>(expected), std::get<0>(actual));
      ExpectExactlyEqual(std::get<1>(expected), std::get<1>(actual));
      ExpectExactlyEqual(std::get<2>(expected), std::get<2>(actual));
      EXPECT_EQ(std::get<3>(expected), std::get<3>(actual));
      const auto& depths = std::get<0>(actual).GetData();
      EXPECT_GT(std::count_if(depths.begin(),
                              depths.end(),
                              [](float depth) { return depth > 0.0f; }),
                0);
    };
    compare(expected_first, first);
    compare(expected_second, second);
  }
}

TEST_F(CudaTest, PairedDefaultSweepsPreserveIndependentOutputs) {
  ExpectPairedDefaultSweepOutputs(false);
}

TEST_F(CudaTest, PairedBankedRotationPreservesIndependentOutputs) {
  ExpectPairedDefaultSweepOutputs(true);
}

TEST_F(CudaTest, PairedTallInitialCostStripeTailsPreserveOutputs) {
  ExpectPairedDefaultSweepOutputs(false, true);
}

void ExpectReferenceBatchOutputs(const size_t count,
                                 const bool geometry,
                                 const bool reuse,
                                 const bool checkpoint = false) {
  const auto make_problem = [](Inputs& images) {
    PatchMatch::Problem problem;
    problem.ref_image_idx = 0;
    for (size_t i = 1; i < images.images.size(); ++i)
      problem.src_image_idxs.push_back(static_cast<int>(i));
    problem.images = &images.images;
    problem.depth_maps = &images.depths;
    problem.normal_maps = &images.normals;
    return problem;
  };
  auto options = MakeOptions(geometry, geometry, reuse);
  const PatchMatchOptions defaults;
  options.window_radius = defaults.window_radius;
  options.window_step = defaults.window_step;
  options.num_samples = defaults.num_samples;
  options.num_iterations = defaults.num_iterations;
  options.sigma_spatial = options.window_radius;
  std::vector<std::unique_ptr<Inputs> > inputs;
  std::vector<PatchMatch::Problem> problems;
  for (size_t i = 0; i < count; ++i) {
    inputs.push_back(std::make_unique<Inputs>(
        MakeInputs(static_cast<int>(i % 2),
                   9,
                   checkpoint ? 129 : (i % 2 ? 153 : 149),
                   checkpoint ? 257 : 0)));
    problems.push_back(make_problem(*inputs.back()));
  }
  const auto construct = [&]() {
    std::vector<std::unique_ptr<PatchMatchCuda> > owners;
    for (const auto& problem : problems)
      owners.push_back(std::make_unique<PatchMatchCuda>(options, problem));
    return owners;
  };
  const auto read = [](const PatchMatchCuda& cuda) {
    return std::make_tuple(cuda.GetDepthMap(),
                           cuda.GetNormalMap(),
                           cuda.GetSelProbMap(),
                           cuda.GetConsistentImageIdxs());
  };
  struct Streams {
    std::vector<cudaStream_t> values;
    explicit Streams(size_t count) : values(count, nullptr) {
      for (auto& stream : values)
        CUDA_SAFE_CALL(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Streams() {
      for (auto stream : values)
        if (stream) (void)cudaStreamDestroy(stream);
    }
  } streams(count);
  auto incumbent = construct();
  for (size_t i = 0; i < count; i += 2) {
    if (i + 1 < count) {
      RunPatchMatchPair(*incumbent[i],
                        *incumbent[i + 1],
                        streams.values[i],
                        streams.values[i + 1]);
    } else {
      incumbent[i]->Run();
    }
  }
  using Snapshot =
      std::tuple<DepthMap, NormalMap, Mat<float>, std::vector<int> >;
  std::vector<Snapshot> expected;
  for (const auto& owner : incumbent) expected.push_back(read(*owner));
  if (!checkpoint) incumbent.clear();
  options.checkpoint_hmm = checkpoint;
  auto batch = construct();
  if (checkpoint) {
    for (size_t i = 0; i < count; ++i)
      EXPECT_LT(batch[i]->PairPeakBytes(), incumbent[i]->PairPeakBytes());
  }
  std::vector<PatchMatchCuda*> views;
  for (auto& owner : batch) views.push_back(owner.get());
  RunPatchMatchBatch(views, streams.values);
  for (size_t i = 0; i < count; ++i) {
    SCOPED_TRACE(::testing::Message() << "count=" << count << " view=" << i
                                      << " geometry=" << geometry << " reuse="
                                      << reuse << " checkpoint=" << checkpoint);
    const auto actual = read(*batch[i]);
    ExpectExactlyEqual(std::get<0>(expected[i]), std::get<0>(actual));
    ExpectExactlyEqual(std::get<1>(expected[i]), std::get<1>(actual));
    ExpectExactlyEqual(std::get<2>(expected[i]), std::get<2>(actual));
    EXPECT_EQ(std::get<3>(expected[i]), std::get<3>(actual));
  }
  if (checkpoint) {
    for (size_t i = 0; i < count; ++i) {
      incumbent[i]->Run();
      batch[i]->Run();
      const auto expected_next = read(*incumbent[i]);
      const auto actual_next = read(*batch[i]);
      ExpectExactlyEqual(std::get<0>(expected_next), std::get<0>(actual_next));
      ExpectExactlyEqual(std::get<1>(expected_next), std::get<1>(actual_next));
      ExpectExactlyEqual(std::get<2>(expected_next), std::get<2>(actual_next));
      EXPECT_EQ(std::get<3>(expected_next), std::get<3>(actual_next));
    }
  }
}

TEST_F(CudaTest, ThreeViewPhotoBatchPreservesOddTallOutputs) {
  ExpectReferenceBatchOutputs(3, false, false);
}

TEST_F(CudaTest, FiveViewPhotoBatchPreservesOddTallOutputs) {
  ExpectReferenceBatchOutputs(5, false, false);
}

TEST_F(CudaTest, FiveViewGeometricBankedBatchPreservesFinalMask) {
  ExpectReferenceBatchOutputs(5, true, true);
}

TEST_F(CudaTest, ThreeViewCheckpointPhotoBatchPreservesFullStorageOutputs) {
  ExpectReferenceBatchOutputs(3, false, true, true);
}

TEST_F(CudaTest, FiveViewCheckpointPhotoBatchPreservesFullStorageOutputs) {
  ExpectReferenceBatchOutputs(5, false, true, true);
}

TEST_F(CudaTest, ThreeViewCheckpointGeometricBatchPreservesFullStorageOutputs) {
  ExpectReferenceBatchOutputs(3, true, true, true);
}

TEST_F(CudaTest, FiveViewCheckpointGeometricBatchPreservesFullStorageOutputs) {
  ExpectReferenceBatchOutputs(5, true, true, true);
}

TEST_F(CudaTest, SharedGeometryBankPreservesRepeatedBatchAndStandaloneRuns) {
  for (const auto& [width, height] :
       std::array<std::pair<int, int>, 2>{{{128, 256}, {257, 129}}}) {
    for (const bool geometry : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << width << "x" << height << " geometry=" << geometry);
      std::vector<Inputs> inputs;
      for (int i = 0; i < 3; ++i)
        inputs.push_back(MakeInputs(i, 2, height, width));
      auto options = MakeOptions(geometry, geometry, true);
      const PatchMatchOptions defaults;
      options.window_radius = defaults.window_radius;
      options.window_step = defaults.window_step;
      options.num_samples = defaults.num_samples;
      options.num_iterations = defaults.num_iterations;
      options.sigma_spatial = options.window_radius;
      std::vector<PatchMatch::Problem> problems(3);
      std::vector<std::unique_ptr<PatchMatchCuda> > control, candidate;
      for (size_t i = 0; i < problems.size(); ++i) {
        auto& problem = problems[i];
        problem.ref_image_idx = 0;
        problem.src_image_idxs = {1, 2};
        problem.images = &inputs[i].images;
        problem.depth_maps = &inputs[i].depths;
        problem.normal_maps = &inputs[i].normals;
        control.push_back(std::make_unique<PatchMatchCuda>(options, problem));
        ASSERT_TRUE(control.back()->CanShareGeometryRotation());
        candidate.push_back(std::make_unique<PatchMatchCuda>(
            options,
            problem,
            std::shared_ptr<const PatchMatchSparseSources>{},
            i > 0 ? control[0].get() : nullptr));
        ASSERT_TRUE(candidate.back()->CanShareGeometryRotation());
        EXPECT_EQ(candidate.back()->PairPeakBytes(),
                  control.back()->PairPeakBytes());
      }
      std::vector<PatchMatchCuda*> views;
      std::vector<cudaStream_t> streams(3, nullptr);
      struct Cleanup {
        std::vector<cudaStream_t>& streams;
        ~Cleanup() {
          for (auto stream : streams)
            if (stream) (void)cudaStreamDestroy(stream);
        }
      } cleanup{streams};
      for (size_t i = 0; i < 3; ++i) {
        views.push_back(candidate[i].get());
        CUDA_SAFE_CALL(
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      }
      const auto compare = [](const PatchMatchCuda& expected,
                              const PatchMatchCuda& actual) {
        ExpectExactlyEqual(expected.GetDepthMap(), actual.GetDepthMap());
        ExpectExactlyEqual(expected.GetNormalMap(), actual.GetNormalMap());
        ExpectExactlyEqual(expected.GetSelProbMap(), actual.GetSelProbMap());
        EXPECT_EQ(expected.GetConsistentImageIdxs(),
                  actual.GetConsistentImageIdxs());
        EXPECT_GE(actual.PairPeakBytes(), expected.PairPeakBytes());
      };
      for (int repeat = 0; repeat < 2; ++repeat) {
        for (auto& owner : control) owner->Run();
        RunPatchMatchBatch(views, streams);
        for (size_t i = 0; i < 3; ++i) compare(*control[i], *candidate[i]);
      }
      // Non-first owners have no private alternate pair when leaving the batch.
      // Standalone Run must provision it and preserve logical views/readback.
      for (size_t i = 0; i < 3; ++i) {
        control[i]->Run();
        candidate[i]->Run();
        compare(*control[i], *candidate[i]);
      }
      control[0]->Run();
      control[1]->Run();
      RunPatchMatchPair(*candidate[0], *candidate[1], streams[0], streams[1]);
      compare(*control[0], *candidate[0]);
      compare(*control[1], *candidate[1]);
    }
  }
}

TEST_F(CudaTest, SparseSourceAliasesPreserveOrderedPaddedOutputs) {
  // One workspace identity; five references reuse differently ordered subsets.
  // Varying source dimensions exercise the original zero-padded layer extent.
  Inputs input = MakeInputs(0, 9, 257, 513);
  for (const bool geometry : {false, true}) {
    for (const bool reuse : {false, true}) {
      auto options = MakeOptions(geometry, geometry, reuse);
      const PatchMatchOptions defaults;
      options.window_radius = defaults.window_radius;
      options.window_step = defaults.window_step;
      options.num_samples = defaults.num_samples;
      options.num_iterations = defaults.num_iterations;
      options.sigma_spatial = options.window_radius;
      // Native PrepareProblem owns different vectors for each reference.
      std::vector<std::vector<Image> > images(5, input.images);
      std::vector<std::vector<DepthMap> > depths(5, input.depths);
      std::vector<PatchMatch::Problem> problems(5);
      std::vector<const PatchMatch::Problem*> pointers;
      for (int ref = 0; ref < 5; ++ref) {
        auto& problem = problems[ref];
        problem.ref_image_idx = ref;
        problem.images = &images[ref];
        problem.depth_maps = &depths[ref];
        problem.normal_maps = &input.normals;
        for (int source = 0; source < 10; ++source)
          if (source != ref) problem.src_image_idxs.push_back(source);
        std::rotate(problem.src_image_idxs.begin(),
                    problem.src_image_idxs.begin() + ref,
                    problem.src_image_idxs.end());
        pointers.push_back(&problem);
      }
      EXPECT_EQ(PatchMatchSparseSources::Budget({nullptr, nullptr, nullptr},
                                                geometry),
                0);
      auto invalid = problems[0];
      invalid.images = nullptr;
      EXPECT_EQ(PatchMatchSparseSources::Budget(
                    {&invalid, &problems[1], &problems[2]}, geometry),
                0);
      invalid = problems[0];
      invalid.src_image_idxs.clear();
      EXPECT_EQ(PatchMatchSparseSources::Budget(
                    {&invalid, &problems[1], &problems[2]}, geometry),
                0);
      using Snapshot =
          std::tuple<DepthMap, NormalMap, Mat<float>, std::vector<int> >;
      const auto read = [](const PatchMatchCuda& owner) {
        return Snapshot(owner.GetDepthMap(),
                        owner.GetNormalMap(),
                        owner.GetSelProbMap(),
                        owner.GetConsistentImageIdxs());
      };
      std::vector<Snapshot> expected;
      for (const auto& problem : problems) {
        PatchMatchCuda independent(options, problem);
        independent.Run();
        expected.push_back(read(independent));
        const auto& depth = std::get<0>(expected.back()).GetData();
        EXPECT_TRUE(std::any_of(depth.begin(), depth.end(), [](float value) {
          return value > 0.0f;
        }));
      }
      ASSERT_GT(PatchMatchSparseSources::Budget(pointers, geometry), 0);
      auto shared = PatchMatchSparseSources::Create(pointers, geometry);
      ASSERT_TRUE(shared);
      std::vector<std::unique_ptr<PatchMatchCuda> > owners;
      owners.push_back(std::make_unique<PatchMatchCuda>(options, problems[0]));
      owners[0]->UseSparseSources(shared);
      for (size_t i = 1; i < problems.size(); ++i)
        owners.push_back(
            std::make_unique<PatchMatchCuda>(options, problems[i], shared));
      shared.reset();  // Owners, not the caller, must retain the bank.
      std::vector<PatchMatchCuda*> views;
      std::vector<cudaStream_t> streams(owners.size(), nullptr);
      struct StreamCleanup {
        std::vector<cudaStream_t>& values;
        ~StreamCleanup() {
          for (auto stream : values) {
            if (stream) {
              (void)cudaStreamSynchronize(stream);
              (void)cudaStreamDestroy(stream);
            }
          }
        }
      } cleanup{streams};
      for (auto& stream : streams)
        CUDA_SAFE_CALL(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      for (auto& owner : owners) views.push_back(owner.get());
      RunPatchMatchBatch(views, streams);
      for (size_t i = 0; i < owners.size(); ++i) {
        const auto actual = read(*owners[i]);
        ExpectExactlyEqual(std::get<0>(expected[i]), std::get<0>(actual));
        ExpectExactlyEqual(std::get<1>(expected[i]), std::get<1>(actual));
        ExpectExactlyEqual(std::get<2>(expected[i]), std::get<2>(actual));
        EXPECT_EQ(std::get<3>(expected[i]), std::get<3>(actual));
      }
    }
  }
}

TEST_F(CudaTest, BatchAdmissionCanBeatIneligiblePairs) {
  // Search grid widths against the queried device rather than baking in an
  // sm86/SM-count cutoff. This exercises the large-grid path where the
  // incumbent schedules singletons but the five-view aggregate saves waves.
  bool found = false;
  for (size_t width = 64; width <= 65536; width += 64) {
    const std::vector<std::pair<size_t, size_t> > shapes(5, {width, 3});
    if (!CanPairPatchMatchViews(width, 3, width, 3, false) &&
        CanBatchPatchMatchViews(shapes, false)) {
      found = true;
      break;
    }
  }
  if (!found) GTEST_SKIP() << "No beneficial five-grid case on this device";
  EXPECT_TRUE(found);
}

TEST_F(CudaTest, BatchAdmissionRejectsUnsupportedGroupSizes) {
  EXPECT_FALSE(CanBatchPatchMatchViews({{47, 149}, {39, 153}}, false));
  EXPECT_FALSE(CanBatchPatchMatchViews(
      std::vector<std::pair<size_t, size_t> >(6, {47, 149}), false));
  EXPECT_FALSE(
      CanBatchPatchMatchViews({{47, 149}, {0, 153}, {47, 149}}, false));
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

TEST_F(CudaTest, TallInitialCostStripeTailsPreserveOutputs) {
  // The two odd reference heights leave 21- and 25-row final stripes. Odd
  // widths also retain the existing pitched-storage and border-lane coverage.
  for (const int variant : {0, 1}) {
    Inputs inputs = MakeInputs(variant, 2, variant == 0 ? 149 : 153);
    for (const int radius : {2, 5, 9, 17}) {
      SCOPED_TRACE(::testing::Message()
                   << "variant=" << variant << " radius=" << radius);
      PatchMatchOptions options = MakeOptions(false, false, false);
      options.window_radius = radius;
      options.sigma_spatial = radius;
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

TEST_F(CudaTest, DuplicatedSourcesPreservePhotoGeometryAcrossCacheBoundary) {
  // The same source bitmap, calibration, and pose occupy every source layer.
  // Sampling can choose different IDs as the CDF length changes, but each
  // selected source contributes the same photo cost. Compare geometry only:
  // selection maps necessarily have different source dimensions.
  const Inputs seed = MakeInputs(0, 1);
  const auto duplicate_source = [&seed](const int num_sources) {
    Inputs inputs;
    inputs.images.push_back(seed.images[0]);
    inputs.depths.push_back(seed.depths[0]);
    inputs.normals.push_back(seed.normals[0]);
    for (int i = 0; i < num_sources; ++i) {
      inputs.images.push_back(seed.images[1]);
      inputs.depths.push_back(seed.depths[1]);
      inputs.normals.push_back(seed.normals[1]);
    }
    return inputs;
  };

  PatchMatchOptions options = MakeOptions(false, false, false);
  options.window_radius = 5;
  options.window_step = 1;
  options.num_samples = 15;
  options.num_iterations = 5;
  options.sigma_spatial = options.window_radius;

  Inputs nine_sources = duplicate_source(9);
  Inputs thirty_three_sources = duplicate_source(33);
  const Result nine = RunPatchMatch(&nine_sources, options);
  const Result thirty_three = RunPatchMatch(&thirty_three_sources, options);
  ExpectNonempty(nine, nine_sources, false);
  ExpectNonempty(thirty_three, thirty_three_sources, false);
  EXPECT_EQ(nine.selection.GetDepth(), 9);
  EXPECT_EQ(thirty_three.selection.GetDepth(), 33);
  ExpectExactlyEqual(nine.depths, thirty_three.depths);
  ExpectExactlyEqual(nine.normals, thirty_three.normals);
}

TEST_P(PatchMatchCudaTest, CheckpointHmmPreservesAlgorithmOutputs) {
  const auto [geometry, filter] = GetParam();
  for (const int num_sources : {1, 9, 17}) {
    for (const bool reuse : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "sources=" << num_sources << ", reuse=" << reuse);
      // The 47x35 reference has stripe tails in both orientations, unequal
      // plane strides, and odd sizes; source counts also cross packed tails.
      Inputs inputs = MakeInputs(0, num_sources);
      PatchMatchOptions options = MakeOptions(geometry, filter, reuse);
      options.window_radius = 5;
      options.checkpoint_hmm = false;
      const Result original = RunPatchMatch(&inputs, options);
      options.checkpoint_hmm = true;
      const Result checkpointed = RunPatchMatch(&inputs, options);
      ExpectNonempty(original, inputs, filter);
      ExpectNonempty(checkpointed, inputs, filter);
      ExpectEquivalent(original, checkpointed);
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
