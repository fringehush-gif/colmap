// SPDX-License-Identifier: BSD-3-Clause
// Exact-depth consumer regression tests. No performance claim.
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/mat.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/testing.h"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace colmap::mvs {
// Inspect exact-depth ownership through the Fusion test peer.
struct StereoFusionExactDepthTestPeer {
  static std::vector<std::vector<char>> Masks(const StereoFusion& fusion) {
    std::vector<std::vector<char>> masks;
    for (size_t index = 0; index < fusion.fused_pixel_masks_.size(); ++index) {
      const auto& [width, height] = fusion.depth_map_sizes_.at(index);
      masks.push_back(fusion.fused_pixel_masks_[index].Snapshot(
          0, static_cast<size_t>(width) * height));
    }
    return masks;
  }
};
}  // namespace colmap::mvs

namespace {
using colmap::mvs::BoundedWorkspace;
using colmap::mvs::Mat;
using colmap::mvs::StereoFusion;
using colmap::mvs::StereoFusionOptions;
using colmap::mvs::Workspace;
namespace fs = std::filesystem;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
uint32_t Bits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
template <class F>
void Refuses(F action, const char* message) {
  bool refused = false;
  try {
    action();
  } catch (const std::exception&) {
    refused = true;
  }
  Require(refused, message);
}

// Independent byte oracle: no production Mat writer/codec in source fixtures.
// Current native mapping is explicitly Linux/little-endian/IEEE binary32.
void WriteRaw(const fs::path& path,
              size_t width,
              size_t height,
              size_t channels,
              const std::vector<float>& values) {
  Require(values.size() == width * height * channels, "bad fixture size");
  std::ofstream out(path, std::ios::binary);
  out << width << '&' << height << '&' << channels << '&';
  for (float value : values) {
    const auto bits = Bits(value);
    for (int shift = 0; shift < 32; shift += 8) out.put((bits >> shift) & 255);
  }
  Require(out.good(), "raw fixture write failed");
}
struct Fixture {
  fs::path root;
  std::vector<float> depths;
};
Fixture MakeFixture(const fs::path& root,
                    float ordinary_depth,
                    bool wide_span,
                    bool zero_normals = false,
                    int views = 2) {
  Require(!fs::exists(root), "fixture root must be fresh");
  for (const auto& dir : {root / "sparse",
                          root / "images",
                          root / "stereo" / "depth_maps",
                          root / "stereo" / "normal_maps"})
    fs::create_directories(dir);
  std::ofstream(root / "sparse" / "cameras.txt") << "1 PINHOLE 8 2 8 8 4 1\n";
  std::ofstream points(root / "sparse" / "points3D.txt");
  points << "1 0 0 2 20 40 60 0 1 0 2 0" << (views == 3 ? " 3 0\n" : "\n");
  std::ofstream images(root / "sparse" / "images.txt");
  std::ofstream roster(root / "stereo" / "fusion.cfg");
  std::vector<float> depths(16, ordinary_depth);
  depths[1] = 0.0f;
  if (wide_span) {
    depths[0] = 1.0f;
    depths[15] = 1040.0f;
  }
  for (int view = 0; view < views; ++view) {
    const std::string name = "seed-" + std::to_string(view) + ".png";
    // Positive tx gives one pixel/depth parallax in the second camera.
    images << view + 1 << " 1 0 0 0 " << (view ? "0.125" : "0") << " 0 0 1 "
           << name << "\n4 1 1\n";
    roster << name << '\n';
    colmap::Bitmap rgb(8, 2, true);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 8; ++x)
        rgb.SetPixel(x, y, colmap::BitmapColor<uint8_t>(20, 40, 60));
    Require(rgb.Write(root / "images" / name), "RGB write failed");
    WriteRaw(root / "stereo" / "depth_maps" / (name + ".geometric.bin"),
             8,
             2,
             1,
             depths);
    std::vector<float> normals(48, 0.0f);
    if (!zero_normals)
      for (size_t i = 32; i < 48; ++i) normals[i] = -1.0f;
    if (view == 2)
      for (size_t i = 0; i < 16; ++i) {
        normals[i] = 0.25f;
        normals[32 + i] = -0.75f;
      }
    WriteRaw(root / "stereo" / "normal_maps" / (name + ".geometric.bin"),
             8,
             2,
             3,
             normals);
  }
  Require(images.good() && roster.good(), "sparse fixture write failed");
  return {root, depths};
}

StereoFusionOptions Options(bool exact) {
  StereoFusionOptions o;
  o.num_threads = 1;
  o.min_num_pixels = 2;
  o.max_num_pixels = 8;
  o.max_traversal_depth = 3;
  o.check_num_images = 2;
  o.bounded_cache = true;
  o.bounded_cache_components = true;
  o.bounded_cache_exact_depth = exact;
  o.bounded_cache_normal_mapped_bytes = exact ? 1024 * 1024 : 0;
  o.bounded_cache_normal_mapped_files = exact ? 1 : 0;
  o.cache_size = 0.001;
  return o;
}
std::unique_ptr<StereoFusion> RunFusionFixture(const Fixture& fixture,
                                               StereoFusionOptions options) {
  Require(options.Check(), "valid exact-depth options refused");
  auto fusion = std::make_unique<StereoFusion>(
      options, fixture.root, "COLMAP", "", "geometric");
  fusion->Run();
  return fusion;
}
bool SamePoints(const StereoFusion& a, const StereoFusion& b) {
  if (a.GetFusedPointsVisibility() != b.GetFusedPointsVisibility() ||
      a.GetFusedPoints().size() != b.GetFusedPoints().size())
    return false;
  for (size_t i = 0; i < a.GetFusedPoints().size(); ++i) {
    const auto& x = a.GetFusedPoints()[i];
    const auto& y = b.GetFusedPoints()[i];
    if (Bits(x.x) != Bits(y.x) || Bits(x.y) != Bits(y.y) ||
        Bits(x.z) != Bits(y.z) || Bits(x.nx) != Bits(y.nx) ||
        Bits(x.ny) != Bits(y.ny) || Bits(x.nz) != Bits(y.nz) || x.r != y.r ||
        x.g != y.g || x.b != y.b)
      return false;
  }
  return true;
}
StereoFusionOptions Variant(StereoFusionOptions o, int variant) {
  if (variant == 0) o.bounding_box.second(0) = 0.25f;
  if (variant == 1) o.max_num_pixels = 2;
  if (variant == 2) o.max_traversal_depth = 1;
  if (variant == 3) o.min_num_pixels = 3;
  return o;
}
void AssertParity(const StereoFusion& raw, const StereoFusion& exact) {
  Require(SamePoints(raw, exact), "all point bits/order or VIS differ");
  Require(colmap::mvs::StereoFusionExactDepthTestPeer::Masks(raw) ==
              colmap::mvs::StereoFusionExactDepthTestPeer::Masks(exact),
          "final visited masks differ");
  const auto decoded = exact.GetBoundedCacheStats();
  Require(decoded && decoded->peak_charged_bytes <= decoded->capacity_bytes &&
              decoded->reserved_bytes == 0,
          "decoded charge/release differs");
  const auto mapped = exact.GetMappedNormalStats();
  Require(mapped.peak_charged_bytes <= mapped.capacity_bytes &&
              mapped.peak_charged_files <= mapped.capacity_files &&
              mapped.source_peak_live_files <= mapped.capacity_files &&
              mapped.active_copies == 0 && mapped.reserved_bytes == 0 &&
              mapped.reserved_files == 0,
          "mapped cap or release differs");
}
void VerifyFusionOutputParity(const Fixture& fixture) {
  auto raw = RunFusionFixture(fixture, Options(false));
  auto exact = RunFusionFixture(fixture, Options(true));
  Require(!raw->GetFusedPoints().empty(), "fixture did not fuse any point");
  AssertParity(*raw, *exact);
  const auto reads = exact->GetDepthExactStateReadCounts();
  Require(reads.first > 0 && reads.second > 0,
          "root and candidate original copies not both exercised");
  Require(raw->GetDepthExactStateReadCounts() ==
              std::make_pair(size_t{0}, size_t{0}),
          "default path read originals");
  exact->Run();
  AssertParity(*raw, *exact);
  Require(exact->GetDepthExactStateReadCounts() == reads,
          "reused Run accumulated exact-read counters");
  for (int variant = 0; variant < 4; ++variant) {
    auto a = RunFusionFixture(fixture, Variant(Options(false), variant));
    auto b = RunFusionFixture(fixture, Variant(Options(true), variant));
    AssertParity(*a, *b);
  }
  auto neighborhood = Options(true);
  neighborhood.bounded_cache_neighborhood = true;
  auto pinned_mask = RunFusionFixture(fixture, neighborhood);
  AssertParity(*raw, *pinned_mask);
  Require(pinned_mask->GetDepthExactStateReadCounts() == reads,
          "decoded mask pins changed original-copy ordering/counts");
}
Workspace::Options WorkspaceOptions(const fs::path& root) {
  Workspace::Options o;
  o.workspace_path = root;
  o.workspace_format = "COLMAP";
  o.input_type = "geometric";
  o.cache_size = 0.001;
  o.num_threads = 1;
  o.exact_depth_state = true;
  o.mapped_normal_bytes_limit = 1024 * 1024;
  o.mapped_normal_files_limit = 1;
  return o;
}
void VerifyScalarReaderAdmissionAndLifetime(const Fixture& f) {
  BoundedWorkspace ws(WorkspaceOptions(f.root));
  Mat<float> scratch(1, 1, 1);
  ws.PrepareOriginalDepth(0, 8, 2);
  for (int repeat = 0; repeat < 3; ++repeat) {
    Require(Bits(ws.ReadOriginalDepth(0, 0, 2, scratch)) == Bits(f.depths[2]),
            "exact depth changed across repeated scalar reads");
    Require(scratch.GetWidth() == 1 && scratch.GetHeight() == 1 &&
                scratch.GetDepth() == 1,
            "scalar scratch shape differs");
  }
  Refuses([&] { ws.ReadOriginalDepth(0, 2, 0, scratch); },
          "out-of-grid read accepted");
  Refuses([&] { ws.ReadOriginalDepth(1, 0, 0, scratch); },
          "unadmitted image read accepted");
  const auto stats = ws.GetMappedNormalStats();
  Require(stats.peak_charged_files == 1 && stats.active_copies == 0 &&
              stats.reserved_files == 0,
          "one-file scalar owner lifetime proof failed");
  Refuses([&] { ws.PrepareOriginalDepth(0, 4, 1); }, "grid mismatch admitted");
  for (int variant = 0; variant < 2; ++variant) {
    auto o = WorkspaceOptions(f.root);
    if (variant == 0) o.exact_depth_state = false;
    if (variant == 1) o.mapped_normal_bytes_limit = 1;
    Refuses(
        [&] {
          BoundedWorkspace invalid(o);
          invalid.PrepareOriginalDepth(0, 8, 2);
        },
        "invalid backing/domain/budget admitted");
  }
  std::cout << "scalar_reads=3 scalar_bits=exact receiver_shape=1x1x1"
            << " owner_evictions=" << stats.evictions << '\n';
}
void VerifyInvalidDepthSourceRefusal(const fs::path& base) {
  for (int variant = 0; variant < 4; ++variant) {
    const auto f = MakeFixture(
        base / ("bad-source-" + std::to_string(variant)), 2.0f, false);
    const auto path = f.root / "stereo/depth_maps/seed-0.png.geometric.bin";
    if (variant == 0) WriteRaw(path, 8, 2, 3, std::vector<float>(48, 2.0f));
    if (variant == 1) {
      std::ofstream out(path, std::ios::binary);
      out << "8&2&1&";
    }
    if (variant == 2) WriteRaw(path, 16, 4, 1, std::vector<float>(64, 2.0f));
    if (variant == 3) {
      auto bad = f.depths;
      bad[2] = std::numeric_limits<float>::infinity();
      WriteRaw(path, 8, 2, 1, bad);
    }
    BoundedWorkspace ws(WorkspaceOptions(f.root));
    Mat<float> scratch(1, 1, 1);
    if (variant < 3)
      Refuses([&] { ws.PrepareOriginalDepth(0, 8, 2); },
              "malformed/scalar-grid source admitted");
    else {
      ws.PrepareOriginalDepth(0, 8, 2);
      Refuses([&] { ws.ReadOriginalDepth(0, 0, 2, scratch); },
              "nonfinite copy accepted");
    }
    const auto s = ws.GetMappedNormalStats();
    Require(
        s.active_copies == 0 && s.reserved_files == 0 && s.reserved_bytes == 0,
        "exception leaked pin/reservation");
  }
}
void VerifyExactDepthOptions() {
  Require(!StereoFusionOptions().bounded_cache_exact_depth,
          "exact-depth unexpectedly default-on");
  const auto valid = Options(true);
  Require(valid.Check(), "valid exact-depth options refused");
  for (int variant = 0; variant < 6; ++variant) {
    auto o = valid;
    switch (variant) {
      case 0:
        o.bounded_cache = false;
        break;
      case 1:
        o.bounded_cache_components = false;
        break;
      case 2:
        o.bounded_cache_normal_tile_size = 4;
        break;
      case 3:
        o.bounded_cache_normal_mapped_bytes = 0;
        break;
      case 4:
        o.bounded_cache_normal_mapped_files = 0;
        break;
      case 5:
        o.num_threads = 2;
        break;
    }
    Require(!o.Check(), "incompatible exact-depth options accepted");
  }
  auto automatic = valid;
  automatic.num_threads = -1;
  Require(!automatic.Check(), "automatic row thread count accepted");
  auto legacy = valid;
  legacy.bounded_cache_exact_depth = false;
  Require(!legacy.Check(), "one-file non-tile normal mapping rule weakened");
  legacy.bounded_cache_normal_mapped_bytes = 0;
  legacy.bounded_cache_normal_mapped_files = 0;
  Require(legacy.Check(), "ordinary bounded options changed");
}
void VerifyPositiveMaskAccountingAndRefusal(const fs::path& root) {
  using Data = BoundedWorkspace::ImageData;
  Require(Data::PositiveDepthMaskBytes(std::numeric_limits<size_t>::max()) ==
              std::numeric_limits<size_t>::max() / 8 + 1,
          "mask ceil overflowed size_t");
  Require(Data::PositiveDepthMaskBytes(0) == 0, "empty mask ceil differs");
  struct Environment {
    fenv_t saved;
#if defined(__SSE__)
    unsigned csr = _mm_getcsr();
#endif
    Environment() {
      Require(std::feholdexcept(&saved) == 0,
              "cannot save/mask FP environment");
    }
    ~Environment() {
      std::fesetenv(&saved);
#if defined(__SSE__)
      _mm_setcsr(csr);
#endif
    }
  } environment;
  const std::array<float, 8> classes{0.0f,
                                     -0.0f,
                                     -2.0f,
                                     std::numeric_limits<float>::denorm_min(),
                                     1.0f,
                                     std::numeric_limits<float>::max(),
                                     -std::numeric_limits<float>::denorm_min(),
                                     2.0f};
  size_t cases = 0;
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    Require(std::fesetround(rounding) == 0, "cannot set test rounding mode");
#if defined(__SSE__)
    constexpr unsigned kFlushStates = 4;
#else
    constexpr unsigned kFlushStates = 1;
#endif
    for (unsigned flush = 0; flush < kFlushStates; ++flush) {
#if defined(__SSE__)
      _mm_setcsr((_mm_getcsr() & ~((1u << 15) | (1u << 6))) |
                 ((flush & 1) ? 1u << 15 : 0) | ((flush & 2) ? 1u << 6 : 0));
#else
      Require(flush == 0, "mask FTZ/DAZ fixture requires SSE host");
#endif
      for (size_t count : {size_t{1}, size_t{7}, size_t{8}, size_t{9}}) {
        const auto f =
            MakeFixture(root / ("mask-" + std::to_string(cases)), 2.0f, false);
        std::ofstream(f.root / "sparse/cameras.txt")
            << "1 PINHOLE " << count << " 1 8 8 0 0\n";
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
          values[i] = classes[i % classes.size()];
        colmap::Bitmap rgb(count, 1, true);
        for (size_t i = 0; i < count; ++i)
          rgb.SetPixel(i, 0, colmap::BitmapColor<uint8_t>(20, 40, 60));
        for (const auto* name : {"seed-0.png", "seed-1.png"}) {
          Require(rgb.Write(f.root / "images" / name),
                  "mask RGB fixture failed");
          WriteRaw(f.root / "stereo/depth_maps" /
                       (std::string(name) + ".geometric.bin"),
                   count,
                   1,
                   1,
                   values);
          WriteRaw(f.root / "stereo/normal_maps" /
                       (std::string(name) + ".geometric.bin"),
                   count,
                   1,
                   3,
                   std::vector<float>(3 * count, 0));
        }
        auto raw_options = WorkspaceOptions(f.root);
        raw_options.exact_depth_state = false;
        auto mask_options = raw_options;
        mask_options.exact_depth_state = true;
        BoundedWorkspace raw(raw_options), masked(mask_options);
        const auto component = BoundedWorkspace::Component::kDepth;
        const size_t mask_bytes = count / 8 + (count % 8 != 0);
        Require(masked.GetLoadReservationBytes(0, component) ==
                    4 * count + mask_bytes,
                "mask conversion charge is not FP32 source plus mask");
        Require(masked.GetResidentBytesEstimate(0, component) == mask_bytes,
                "mask resident estimate differs");
        Require(raw.GetLoadReservationBytes(0, component) == 4 * count &&
                    raw.GetResidentBytesEstimate(0, component) == 4 * count,
                "ordinary FP32 accounting changed");
        auto mask = masked.AcquireComponent(0, component);
        auto codes = raw.AcquireComponent(0, component);
        Require(
            mask->positive_depth_mask.size() == mask_bytes &&
                mask->NumBytes() == mask_bytes &&
                mask->depth_map.GetData().empty(),
            "exact mask retained code/FP32 raster or wrong capacity charge");
        Require(codes->positive_depth_mask.empty() &&
                    codes->depth_map.GetData().size() == count,
                "ordinary FP32 representation changed");
        for (size_t i = 0; i < count; ++i) {
          volatile float original =
              values[i];  // Runtime FP comparison, not sign/nonzero bits.
          Require(mask->HasPositiveDepth(0, i) == (original > 0.0f) &&
                      mask->HasPositiveDepth(0, i) ==
                          (codes->GetDepth(0, i) > 0.0f),
                  "mask disagrees with native FP32 DAZ-positive class");
        }
        if (count % 8)
          Require((mask->positive_depth_mask.back() >> (count % 8)) == 0,
                  "mask tail bits are not zero");
        Refuses([&] { mask->GetDepth(0, 0); },
                "mask fabricated a decoded depth");
        Refuses([&] { mask->HasPositiveDepth(1, 0); }, "mask accepted bad row");
        Refuses([&] { mask->HasPositiveDepth(0, count); },
                "mask accepted bad column");
        mask.reset();
        codes.reset();
        Require(masked.GetCacheStats().reserved_bytes == 0,
                "mask load leaked reservation");
        ++cases;
      }
    }
  }
  const size_t mask_cases = cases;
  // A nonfinite last sample must fail even though earlier positive bits exist.
  for (float invalid : {std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    const auto f = MakeFixture(
        root / ("mask-nonfinite-" + std::to_string(cases++)), 2.0f, false);
    auto values = f.depths;
    values.back() = invalid;
    WriteRaw(
        f.root / "stereo/depth_maps/seed-0.png.geometric.bin", 8, 2, 1, values);
    auto options = WorkspaceOptions(f.root);
    options.exact_depth_state = true;
    BoundedWorkspace ws(options);
    Refuses(
        [&] { ws.AcquireComponent(0, BoundedWorkspace::Component::kDepth); },
        "mask did not validate every sample finite");
    Require(ws.GetCacheStats().reserved_bytes == 0,
            "failed mask load leaked reservation");
  }
  std::cout << "positive_mask_rounding_ftzdaz_boundary_cases=" << mask_cases
            << " full_scan_nonfinite_cases=3"
            << " tail_bits=zero ordinary_fp32=unchanged decoded_mask=refused\n";
}
void VerifyDepthLeaseLifetime(const fs::path& root) {
  const auto f = MakeFixture(root / "held-normal", 2.00002f, true);
  auto options = WorkspaceOptions(f.root);
  options.mapped_normal_files_limit = 2;
  BoundedWorkspace::MappedNormalLease retained;
  {
    BoundedWorkspace ws(options);
    ws.PrepareOriginalDepth(0, 8, 2);
    retained = ws.AcquireMappedNormal(0);
    Mat<float> scratch(1, 1, 1);
    Require(Bits(ws.ReadOriginalDepth(0, 0, 2, scratch)) == Bits(f.depths[2]),
            "held normal aliased same-image scalar depth");
    Require(Bits(retained->ReadFloat3(0, 2)[2]) == Bits(-1.0f),
            "scalar depth replaced held normal owner");
    const auto stats = ws.GetMappedNormalStats();
    Require(stats.active_copies == 1 && stats.peak_charged_files == 2 &&
                stats.reserved_files == 0,
            "held lease charge/lifetime differs");
  }
  Require(Bits(retained->ReadFloat3(0, 2)[2]) == Bits(-1.0f),
          "normal lease did not survive workspace destruction");
  retained.reset();
  const auto path = f.root / "stereo/depth_maps/seed-0.png.geometric.bin";
  colmap::mvs::ImmutableMapRegion scalar(path);
  Refuses([&] { scalar.ReadFloat3(0, 0); },
          "normal gather accepted scalar map");
  Refuses([&] { colmap::mvs::ImmutableMapRegion window(path, 0, 1); },
          "three-plane normal window accepted scalar map");
}
}  // namespace

TEST(ExactDepthState, OptionBoundaries) {
  EXPECT_NO_THROW(VerifyExactDepthOptions());
}
TEST(ExactDepthState, FullPointBitsVisibilityMasksAndPredicates) {
  if (!colmap::mvs::ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = colmap::CreateTestDir();
  const auto constant = MakeFixture(root / "constant", 2.0f, false);
  const auto crossing = MakeFixture(root / "half-pixel", 2.00002f, true);
  Require(std::round(1.0f / crossing.depths[2]) == 0,
          "exact half-pixel boundary fixture differs");
  VerifyFusionOutputParity(constant);
  VerifyFusionOutputParity(crossing);
  const auto degenerate = MakeFixture(root / "zero-normals", 2.0f, false, true);
  auto raw = RunFusionFixture(degenerate, Options(false));
  auto exact = RunFusionFixture(degenerate, Options(true));
  AssertParity(*raw, *exact);
  Require(raw->GetFusedPoints().empty(), "zero median normal emitted");
  const auto nonpositive = MakeFixture(root / "nonpositive", -2.0f, false);
  raw = RunFusionFixture(nonpositive, Options(false));
  exact = RunFusionFixture(nonpositive, Options(true));
  AssertParity(*raw, *exact);
  Require(exact->GetFusedPoints().empty() &&
              exact->GetDepthExactStateReadCounts() ==
                  std::make_pair(size_t{0}, size_t{0}),
          "nonpositive mask copied depth or emitted points");
}
TEST(ExactDepthState, ScalarAdmissionKeysEvictionAndLeaseLifetime) {
  if (!colmap::mvs::ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = colmap::CreateTestDir();
  VerifyScalarReaderAdmissionAndLifetime(
      MakeFixture(root / "reader", 2.00002f, true));
  VerifyInvalidDepthSourceRefusal(root);
  VerifyDepthLeaseLifetime(root);
}
TEST(ExactDepthState, PositiveMaskAccountingRoundingAndNonfiniteRefusal) {
  if (!colmap::mvs::ImmutableMapRegion::Supported()) GTEST_SKIP();
  VerifyPositiveMaskAccountingAndRefusal(colmap::CreateTestDir());
}

// Owner-pin regression tests for the mapping source pool.
#include "colmap/sensor/bitmap_reader.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
namespace colmap::mvs {
struct BoundedWorkspaceExactDepthPinTestPeer {
  static std::array<size_t, 6> Metadata(const BoundedWorkspace& ws) {
    struct Before {
      size_t width, height;
      std::filesystem::path path;
    };
    return {ws.original_depth_sources_.size(),
            ws.original_depth_sources_.capacity(),
            sizeof(BoundedWorkspace::OriginalDepthSource) - sizeof(Before),
            ws.original_depth_pins_.capacity(),
            sizeof(BoundedWorkspace::MappedNormalPool::Lease),
            sizeof(ws.original_depth_pins_) +
                sizeof(ws.depth_mapped_neighborhood_stats_)};
  }
};
}  // namespace colmap::mvs

namespace {
size_t MetadataBase(const BoundedWorkspace& ws, bool report) {
  const auto m =
      colmap::mvs::BoundedWorkspaceExactDepthPinTestPeer::Metadata(ws);
  Require(m[1] >= m[0] && m[2] > 0 && m[4] > 0 && m[5] > 0,
          "invalid measured native metadata layout");
  if (report)
    std::cout << "pin_metadata source_size=" << m[0]
              << " source_capacity=" << m[1] << " source_extra_sizeof=" << m[2]
              << " lease_capacity=" << m[3] << " lease_sizeof=" << m[4]
              << " fixed_sizeof=" << m[5] << '\n';
  return m[1] * m[2] + m[3] * m[4] + m[5];
}
void PinCap(const BoundedWorkspace& ws,
            size_t bytes,
            size_t files,
            size_t active) {
  const auto s = ws.GetMappedNormalStats();
  Require(s.peak_charged_bytes <= bytes && s.peak_charged_files <= files &&
              s.reserved_bytes == 0 && s.reserved_files == 0 &&
              s.active_copies == active && s.waits == 0,
          "pin cap/reservation/lifetime or self-wait failed");
}
void VerifyDirectOwnerPins(const fs::path& root) {
  const auto f = MakeFixture(root / "pin-owner", 2.0f, false, false, 3);
  for (int i = 0; i < 3; ++i)
    WriteRaw(f.root / "stereo/depth_maps" /
                 ("seed-" + std::to_string(i) + ".png.geometric.bin"),
             8,
             2,
             1,
             std::vector<float>(16, float(2 + i)));
  const auto path = f.root / "stereo/depth_maps/seed-0.png.geometric.bin";
  const size_t page =
      colmap::mvs::ImmutableMapRegion::GetMappingReservationBytes(path);
  auto o = WorkspaceOptions(f.root);
  o.exact_depth_state = true;
  o.mapped_normal_bytes_limit = 2 * page;
  o.mapped_normal_files_limit = 2;
  BoundedWorkspace ws(o);
  Mat<float> scratch(1, 1, 1);
  for (int i = 0; i < 3; ++i) ws.PrepareOriginalDepth(i, 8, 2);
  Require(ws.GetDepthMappedNeighborhoodStats().metadata_peak_bytes ==
              MetadataBase(ws, true),
          "scalar-off source metadata capacity was not accounted");
  // Full roster requires 3 pages, but root neighborhood {0} + worst excluded
  // page fits two. Support must not depend on the complete roster fitting.
  Require(ws.PrepareOriginalDepthNeighborhood({0}, {}),
          "partial-roster neighborhood did not admit");
  const std::vector<uint8_t> selected_control(3, uint8_t{0});
  Require(ws.GetDepthMappedNeighborhoodStats().metadata_peak_bytes ==
              MetadataBase(ws, true) + sizeof(selected_control) +
                  selected_control.capacity() * sizeof(uint8_t),
          "pin metadata peak ignored allocated vector capacities");
  PinCap(ws, 2 * page, 2, 1);
  for (int i = 0; i < 10; ++i)
    Require(Bits(ws.ReadOriginalDepth(0, 0, 2, scratch)) == Bits(2.0f),
            "borrowed scalar bits differ");
  for (int i : {1, 2})
    Require(Bits(ws.ReadOriginalDepth(i, 0, 2, scratch)) == Bits(float(2 + i)),
            "excluded scalar bits differ");
  Require(Bits(ws.ReadOriginalDepth(0, 0, 2, scratch)) == Bits(2.0f),
          "excluded eviction invalidated borrowed owner");
  const auto reads = ws.GetDepthMappedNeighborhoodStats();
  Require(reads.source_admissions == 3 && reads.pin_acquires == 1 &&
              reads.borrowed_reads == 11 && reads.scalar_acquires == 2,
          "borrow/pin/admission/scalar cadence differs");
  const auto mapped = ws.GetMappedNormalStats();
  Require(
      mapped.loads + mapped.hits ==
          reads.source_admissions + reads.pin_acquires + reads.scalar_acquires,
      "new acquisition partition differs");
  ws.ReleaseOriginalDepthNeighborhood();
  PinCap(ws, 2 * page, 2, 0);
  Require(!ws.PrepareOriginalDepthNeighborhood({0, 1}, {}),
          "excluded-map headroom was not reserved");
  PinCap(ws, 2 * page, 2, 0);
  Refuses([&] { ws.PrepareOriginalDepthNeighborhood({0, 0}, {}); },
          "duplicate source admitted");
  Refuses([&] { ws.PrepareOriginalDepthNeighborhood({3}, {}); },
          "invalid source index admitted");
  PinCap(ws, 2 * page, 2, 0);
  auto one = o;
  one.mapped_normal_bytes_limit = page;
  one.mapped_normal_files_limit = 1;
  BoundedWorkspace fallback(one);
  for (int i = 0; i < 3; ++i) fallback.PrepareOriginalDepth(i, 8, 2);
  Require(!fallback.PrepareOriginalDepthNeighborhood({0}, {}),
          "capacity1 pins excluded headroom");
  for (int i : {0, 1, 2})
    Require(Bits(fallback.ReadOriginalDepth(i, 0, 2, scratch)) ==
                Bits(float(2 + i)),
            "capacity1 fallback bits differ");
  PinCap(fallback, page, 1, 0);
  // Two admitted sources, no excluded set: whole neighborhood can fit. Fail
  // after the first successful Acquire, without mutating mapped source files.
  BoundedWorkspace interrupted(o);
  interrupted.PrepareOriginalDepth(0, 8, 2);
  interrupted.PrepareOriginalDepth(1, 8, 2);
  int calls = 0;
  Refuses(
      [&] {
        interrupted.PrepareOriginalDepthNeighborhood({0, 1}, [&] {
          if (++calls == 2)
            throw std::runtime_error("injected native stop callback failure");
          return false;
        });
      },
      "pin preparation did not propagate callback throw");
  Require(calls == 2, "throw did not exercise a partially acquired set");
  PinCap(interrupted, 2 * page, 2, 0);
  calls = 0;
  Require(!interrupted.PrepareOriginalDepthNeighborhood(
              {0, 1}, [&] { return ++calls == 2; }),
          "pin preparation ignored cancellation");
  Require(
      calls == 2 &&
          interrupted.GetDepthMappedNeighborhoodStats().cancelled_roots == 1,
      "partial pin cancellation not exercised");
  PinCap(interrupted, 2 * page, 2, 0);
  Require(interrupted.PrepareOriginalDepthNeighborhood({0, 1}, {}),
          "complete set did not recover after throw/cancel");
  PinCap(interrupted, 2 * page, 2, 2);
  interrupted.ReleaseOriginalDepthNeighborhood();
  PinCap(interrupted, 2 * page, 2, 0);
  // Byte admission alone is insufficient: require an excluded file slot too.
  auto handles = o;
  handles.mapped_normal_bytes_limit = 3 * page;
  BoundedWorkspace limited(handles);
  for (int i = 0; i < 3; ++i) limited.PrepareOriginalDepth(i, 8, 2);
  Require(!limited.PrepareOriginalDepthNeighborhood({0, 1}, {}),
          "excluded file slot was not reserved");
  PinCap(limited, 3 * page, 2, 0);
  std::cout << "pin_owner_partial_roster=pass excluded_demand=pass "
               "capacity1=scalar_fallback"
            << " partial_throw_cancel=release scratch=unchanged\n";
}
void VerifyPinnedFusionOutputParity(const Fixture& f) {
  const size_t page =
      colmap::mvs::ImmutableMapRegion::GetMappingReservationBytes(
          f.root / "stereo/depth_maps/seed-0.png.geometric.bin");
  for (int variant = 0; variant < 4; ++variant) {
    auto raw_options = Variant(Options(false), variant);
    raw_options.bounded_cache_neighborhood = true;
    auto raw = RunFusionFixture(f, raw_options);
    for (bool complete : {false, true}) {
      auto exact_options = Variant(Options(true), variant);
      exact_options.bounded_cache_neighborhood = true;
      exact_options.bounded_cache_normal_mapped_bytes =
          (complete ? 3 : 1) * page;
      exact_options.bounded_cache_normal_mapped_files = complete ? 3 : 1;
      auto exact = RunFusionFixture(f, exact_options);
      Require(
          SamePoints(*raw, *exact) &&
              colmap::mvs::StereoFusionExactDepthTestPeer::Masks(*raw) ==
                  colmap::mvs::StereoFusionExactDepthTestPeer::Masks(*exact),
          "pin complete/fallback changed full FP32 points/VIS/contributor "
          "order/visited state");
      const auto stats = exact->GetDepthMappedNeighborhoodStats();
      const auto copies = exact->GetDepthExactStateReadCounts();
      Require(stats.borrowed_reads + stats.scalar_acquires ==
                  copies.first + copies.second,
              "scalar copy partition differs");
      Require(stats.root_preparations ==
                      stats.complete_roots + stats.fallback_roots &&
                  stats.cancelled_roots == 0,
              "root complete/fallback partition differs");
      Require(complete ? stats.complete_roots > 0
                       : stats.complete_roots == 0 && stats.fallback_roots > 0,
              "required complete/fallback path not exercised");
      if (complete && variant != 3)
        Require(stats.borrowed_reads > 0, "pin borrowed reads not exercised");
      const auto mapped = exact->GetMappedNormalStats();
      Require(mapped.loads + mapped.hits == stats.source_admissions +
                                                stats.pin_acquires +
                                                stats.scalar_acquires,
              "fusion acquisition partition differs");
      Require(mapped.peak_charged_bytes <=
                      size_t(exact_options.bounded_cache_normal_mapped_bytes) &&
                  mapped.peak_charged_files <=
                      size_t(exact_options.bounded_cache_normal_mapped_files) &&
                  mapped.active_copies == 0 && mapped.reserved_files == 0 &&
                  mapped.reserved_bytes == 0 && mapped.waits == 0,
              "fusion pin lifetime/cap/self-wait failed");
      AssertParity(*raw, *exact);
    }
  }
}

void VerifyAsyncPinCancellation(const Fixture& fixture) {
  const size_t page =
      colmap::mvs::ImmutableMapRegion::GetMappingReservationBytes(
          fixture.root / "stereo/depth_maps/seed-0.png.geometric.bin");
  auto options = Options(true);
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_mapped_bytes = 3 * page;
  options.bounded_cache_normal_mapped_files = 3;
  StereoFusion fusion(options, fixture.root, "COLMAP", "", "geometric");
  std::atomic<bool> stop{false}, done{false};
  std::mutex mutex;
  std::condition_variable barrier;
  bool parked = false, release = false;
  size_t recorded_leases = 0;
  std::exception_ptr error;
  std::vector<fs::path> paths;
  for (int i = 0; i < 3; ++i)
    paths.push_back(fixture.root / "images" /
                    ("seed-" + std::to_string(i) + ".png"));
  // Current row callbacks have no stop hook. Park the existing prefill worker
  // through its scoped reader, after original-depth pins are complete, without
  // adding production cancellation callbacks or mutating immutable sources.
  auto scope = colmap::ScopedBitmapReader::Create(
      paths,
      [&](const fs::path&, bool, bool, colmap::Bitmap* target) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!parked) {
          recorded_leases = fusion.GetMappedNormalStats().active_copies;
          parked = true;
          barrier.notify_all();
          barrier.wait(lock, [&] { return release; });
        }
        *target = colmap::Bitmap(8, 2, true);
        target->Fill(colmap::BitmapColor<uint8_t>(20, 40, 60));
        return true;
      },
      {{8, 2}, {8, 2}, {8, 2}});
  fusion.SetCheckIfStoppedFunc([&] { return stop.load(); });
  std::thread driver([&] {
    try {
      fusion.Run();
    } catch (...) {
      error = std::current_exception();
    }
    done.store(true);
    barrier.notify_all();
  });
  struct JoinOnExit {
    std::thread& driver;
    std::atomic<bool>& stop;
    std::mutex& mutex;
    std::condition_variable& barrier;
    bool& release;
    ~JoinOnExit() {
      stop.store(true);
      {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
      }
      barrier.notify_all();
      if (driver.joinable()) driver.join();
    }
  } join{driver, stop, mutex, barrier, release};
  {
    std::unique_lock<std::mutex> lock(mutex);
    const bool reached = barrier.wait_for(
        lock, std::chrono::seconds(5), [&] { return parked || done.load(); });
    Require(reached && parked && !done.load() && recorded_leases > 0,
            "cancellation did not park a live prefill worker with depth pins");
    stop.store(true);
    release = true;
  }
  barrier.notify_all();
  driver.join();
  if (error) std::rethrow_exception(error);
  const auto mapped = fusion.GetMappedNormalStats();
  const auto roots = fusion.GetDepthMappedNeighborhoodStats();
  Require(mapped.active_copies == 0 && mapped.reserved_bytes == 0 &&
              mapped.reserved_files == 0 && mapped.waits == 0,
          "cooperative cancellation retained pins or self-waited");
  Require(roots.complete_roots > 0 && roots.pin_acquires > 0 &&
              roots.root_preparations == roots.complete_roots +
                                             roots.fallback_roots +
                                             roots.cancelled_roots,
          "cancellation root partition differs");
  scope->Close();
  fusion.SetCheckIfStoppedFunc([] { return false; });
  fusion.Run();
  auto raw = RunFusionFixture(fixture, Options(false));
  AssertParity(*raw, fusion);
}
void VerifyPinPrefillFailureRecovery(const Fixture& fixture) {
  auto options = Options(true);
  options.bounded_cache_neighborhood = true;
  options.bounded_cache_normal_mapped_files = 3;
  StereoFusion fusion(options, fixture.root, "COLMAP", "", "geometric");
  std::atomic<size_t> failures{0};
  auto scope = colmap::ScopedBitmapReader::Create(
      {fixture.root / "images/seed-0.png"},
      [&](const fs::path&, bool, bool, colmap::Bitmap*) {
        ++failures;
        return false;
      },
      {{8, 2}});
  Refuses([&] { fusion.Run(); }, "prefill failure did not propagate");
  const auto mapped = fusion.GetMappedNormalStats();
  Require(failures > 0 && mapped.active_copies == 0 &&
              mapped.reserved_bytes == 0 && mapped.reserved_files == 0,
          "failed prefill left depth-owner pins alive after join");
  scope->Close();
  fusion.Run();
  auto raw = RunFusionFixture(fixture, Options(false));
  AssertParity(*raw, fusion);
}
}  // namespace

TEST(ExactDepthOwnerPins, ExcludedHeadroomCadenceMetadataAndPartialRelease) {
  if (!colmap::mvs::ImmutableMapRegion::Supported()) GTEST_SKIP();
  VerifyDirectOwnerPins(colmap::CreateTestDir());
}
TEST(ExactDepthOwnerPins, CompleteAndFallbackPreserveAllNativeOutput) {
  if (!colmap::mvs::ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = colmap::CreateTestDir();
  const auto fixture =
      MakeFixture(root / "pin-fusion", 2.00002f, true, false, 3);
  VerifyPinnedFusionOutputParity(fixture);
}
TEST(ExactDepthOwnerPins, LivePrefillCancellationFailureAndRetryRelease) {
  if (!colmap::mvs::ImmutableMapRegion::Supported()) GTEST_SKIP();
  const auto root = colmap::CreateTestDir();
  const auto fixture =
      MakeFixture(root / "pin-cancel", 2.00002f, true, false, 3);
  VerifyAsyncPinCancellation(fixture);
  VerifyPinPrefillFailureRecovery(fixture);
}

// Scalar depth-sampling regression tests; no performance claim.
namespace {
using colmap::mvs::ImmutableMapRegion;
void WriteFloatMap(const fs::path& path,
                   size_t width,
                   size_t height,
                   size_t channels,
                   const float* values,
                   size_t count) {
  WriteRaw(path,
           width,
           height,
           channels,
           std::vector<float>(values, values + count));
}
template <class F>
void ExpectThrows(const char* message, F action) {
  Refuses(action, message);
}
void TestDepthScalarSamples(const fs::path& root) {
  static_assert(sizeof(float) == sizeof(uint32_t));

  const auto file = root / "scalar.bin";
  constexpr size_t width = 5;
  constexpr size_t height = 2;
  // The six-byte header leaves the first FP32 sample unaligned. The grid is
  // intentionally asymmetric so row-stride mistakes cannot hide in a square.
  const std::array<uint32_t, width * height> expected_bits{
      0x00000000u,  // +0
      0x80000000u,  // -0
      0x00000001u,  // positive subnormal
      0x80000001u,  // negative subnormal
      0x7f7fffffu,  // largest finite positive value
      0x00800000u,  // smallest positive normal value
      0xff7fffffu,  // largest finite negative magnitude
      0x7fc12345u,  // quiet NaN with payload
      0x7f800000u,  // +infinity
      0xff800000u,  // -infinity
  };
  std::array<float, width * height> payload{};
  for (size_t i = 0; i < expected_bits.size(); ++i) {
    std::memcpy(&payload[i], &expected_bits[i], sizeof(float));
  }
  WriteFloatMap(file, width, height, 1, payload.data(), payload.size());

  const ImmutableMapRegion mapped(file);
  Mat<float> oracle(1, 1, 1);
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      const size_t index = row * width + col;
      const float sample = mapped.ReadFloat1Sample(row, col, width, height);
      uint32_t actual_bits = 0;
      std::memcpy(&actual_bits, &sample, sizeof(actual_bits));
      Require(actual_bits == expected_bits[index],
              "scalar sample read changed source FP32 bits");

      oracle.ReadRegion(mapped, col, row, 1, 1);
      uint32_t oracle_bits = 0;
      std::memcpy(&oracle_bits, oracle.GetPtr(), sizeof(oracle_bits));
      Require(actual_bits == oracle_bits,
              "scalar sample differs from one-pixel ReadRegion oracle");
    }
  }

  // Exercise many additional finite bit patterns without numeric conversion.
  const auto finite_file = root / "finite-scalar.bin";
  constexpr size_t finite_width = 7;
  constexpr size_t finite_height = 3;
  std::array<uint32_t, finite_width * finite_height> finite_bits{};
  std::array<float, finite_width * finite_height> finite_payload{};
  for (size_t i = 0; i < finite_bits.size(); ++i) {
    const uint32_t magnitude =
        static_cast<uint32_t>((i * 0x00123457u + 0x00800001u) & 0x7f7fffffu);
    finite_bits[i] = magnitude | ((i & 1u) != 0 ? 0x80000000u : 0u);
    std::memcpy(&finite_payload[i], &finite_bits[i], sizeof(float));
  }
  WriteFloatMap(finite_file,
                finite_width,
                finite_height,
                1,
                finite_payload.data(),
                finite_payload.size());
  const ImmutableMapRegion finite_mapped(finite_file);
  for (size_t row = 0; row < finite_height; ++row) {
    for (size_t col = 0; col < finite_width; ++col) {
      const size_t index = row * finite_width + col;
      const float first =
          finite_mapped.ReadFloat1Sample(row, col, finite_width, finite_height);
      const float second =
          finite_mapped.ReadFloat1Sample(row, col, finite_width, finite_height);
      uint32_t first_bits = 0;
      uint32_t second_bits = 0;
      std::memcpy(&first_bits, &first, sizeof(first_bits));
      std::memcpy(&second_bits, &second, sizeof(second_bits));
      Require(first_bits == finite_bits[index] && second_bits == first_bits,
              "finite scalar sample bits or repeated read mismatch");
    }
  }

  ExpectThrows("scalar column bound",
               [&] { (void)mapped.ReadFloat1Sample(0, width, width, height); });
  ExpectThrows("scalar row bound", [&] {
    (void)mapped.ReadFloat1Sample(height, 0, width, height);
  });
  ExpectThrows("scalar effective width mismatch",
               [&] { (void)mapped.ReadFloat1Sample(0, 0, width + 1, height); });
  ExpectThrows("scalar effective height mismatch",
               [&] { (void)mapped.ReadFloat1Sample(0, 0, width, height + 1); });

  // Current parser rejects unsupported two-channel backing at
  // construction. Three-channel normals map successfully but reject this API.
  const auto wrong_depth = root / "two-channel.bin";
  WriteRaw(wrong_depth, 2, 1, 2, {1, 2, 3, 4});
  Refuses([&] { const ImmutableMapRegion refused(wrong_depth); },
          "two-channel source admission accepted");
  const auto normal = root / "normal.bin";
  WriteRaw(normal, 2, 1, 3, {1, 2, 3, 4, 5, 6});
  const ImmutableMapRegion normal_map(normal);
  Refuses([&] { normal_map.ReadFloat1Sample(0, 0, 2, 1); },
          "scalar accepted three-channel normal owner");
  const ImmutableMapRegion normal_window(normal, 0, 1);
  Refuses([&] { normal_window.ReadFloat1Sample(0, 0, 2, 1); },
          "scalar accepted normal window instead of complete depth owner");
}
void VerifyScalarScratchLifetime(const fs::path& root) {
  const auto fixture = MakeFixture(root / "scalar-scratch", 2.00002f, true);
  auto options = WorkspaceOptions(fixture.root);
  options.exact_depth_state = true;
  BoundedWorkspace workspace(options);
  workspace.PrepareOriginalDepth(0, 8, 2);
  Mat<float> scratch(3, 2, 2);
  for (size_t i = 0; i < scratch.GetData().size(); ++i)
    scratch.GetPtr()[i] = float(i + 1);
  const auto* allocation = scratch.GetPtr();
  const auto before = scratch.GetData();
  for (size_t col : {size_t{2}, size_t{3}})
    Require(Bits(workspace.ReadOriginalDepth(0, 0, col, scratch)) ==
                Bits(fixture.depths[col]),
            "live scalar consumer changed original depth");
  Refuses([&] { workspace.ReadOriginalDepth(0, 2, 0, scratch); },
          "live scalar consumer accepted out-of-grid sample");
  Require(scratch.GetWidth() == 3 && scratch.GetHeight() == 2 &&
              scratch.GetDepth() == 2 && scratch.GetPtr() == allocation &&
              scratch.GetData() == before,
          "scalar consumer mutated public scratch argument");
  const auto stats = workspace.GetMappedNormalStats();
  Require(stats.active_copies == 0 && stats.reserved_bytes == 0 &&
              stats.reserved_files == 0,
          "scalar consumer retained demand owner");
}

}  // namespace
TEST(ExactDepthScalar, UnalignedBitsBoundsAndSourceRefusal) {
  if (!colmap::mvs::ImmutableMapRegion::Supported()) GTEST_SKIP();
  TestDepthScalarSamples(colmap::CreateTestDir());
}
TEST(ExactDepthScalar, LiveConsumerPreservesScratchAndLeaseLifetime) {
  if (!colmap::mvs::ImmutableMapRegion::Supported()) GTEST_SKIP();
  VerifyScalarScratchLifetime(colmap::CreateTestDir());
}
