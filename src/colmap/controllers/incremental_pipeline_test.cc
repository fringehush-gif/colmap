// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/incremental_pipeline.h"

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <thread>

#include <OpenImageIO/imageio.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

enum class ColorReadMode { kSuccess, kTransient, kFailure };
ColorReadMode color_read_mode;
std::thread::id mapping_thread;
std::atomic<int> injected_prefetch_failures{0};
std::atomic<int> serial_successes{0};
std::atomic<int> async_scanlines{0};
std::map<std::string, int> color_image_tags;

using Color = std::array<uint8_t, 3>;
using Observation = std::pair<std::string, point2D_t>;
using Association = std::vector<Observation>;
using ColorState = std::map<Association, Color>;

struct ColorRunResult {
  std::vector<std::pair<std::vector<std::string>, ColorState>> stages;
  ColorState final;
};

std::string ColorPathKey(const std::filesystem::path& path) {
  return path.lexically_normal().generic_string();
}

Color PointColor(const Point3D& point) {
  return {point.color(0), point.color(1), point.color(2)};
}

// Each channel is affine over the raster, so bilinear sampling has a simple
// independent oracle at the observation coordinate. Values stay in uint8.
std::optional<Eigen::Vector3d> ExpectedSample(const std::filesystem::path& root,
                                              const Image& image,
                                              const Point2D& point2D) {
  const double x = point2D.xy(0) - 0.5;
  const double y = point2D.xy(1) - 0.5;
  if (!(x >= 0 && x < 127 && y >= 0 && y < 127)) return std::nullopt;
  const int tag = color_image_tags.at(ColorPathKey(root / image.Name()));
  return Eigen::Vector3d(static_cast<float>(11 + 3 * tag + y),
                         static_cast<float>(29 + 3 * tag + x),
                         static_cast<float>(43 + 2 * tag + y));
}

ColorState CaptureColors(const Reconstruction& reconstruction) {
  ColorState state;
  for (const auto& [id, point] : reconstruction.Points3D()) {
    Association observations;
    for (const TrackElement& element : point.track.Elements()) {
      const Image& image = reconstruction.Image(element.image_id);
      EXPECT_LT(element.point2D_idx, image.NumPoints2D());
      if (element.point2D_idx >= image.NumPoints2D()) continue;
      EXPECT_EQ(image.Point2D(element.point2D_idx).point3D_id, id);
      observations.emplace_back(image.Name(), element.point2D_idx);
    }
    std::sort(observations.begin(), observations.end());
    EXPECT_FALSE(observations.empty());
    EXPECT_TRUE(
        state.emplace(std::move(observations), PointColor(point)).second);
  }
  return state;
}

void CheckNewImageColors(const Reconstruction& reconstruction,
                         const std::filesystem::path& root,
                         const std::set<std::string>& new_images) {
  for (const auto& [_, point] : reconstruction.Points3D()) {
    bool eligible = false;
    for (const TrackElement& element : point.track.Elements()) {
      const Image& image = reconstruction.Image(element.image_id);
      if (!new_images.count(image.Name()) ||
          element.point2D_idx >= image.NumPoints2D()) {
        continue;
      }
      eligible |=
          ExpectedSample(root, image, image.Point2D(element.point2D_idx))
              .has_value();
    }
    if (eligible) {
      EXPECT_NE(PointColor(point), (Color{0, 0, 0}));
    }
  }
}

void CheckAssociatedColors(const Reconstruction& reconstruction,
                           const std::filesystem::path& root,
                           const bool require_all_colored) {
  const auto registered_ids = reconstruction.RegImageIds();
  const std::set<image_t> registered(registered_ids.begin(),
                                     registered_ids.end());
  std::set<Color> distinct_colors;
  for (const auto& [id, point] : reconstruction.Points3D()) {
    std::set<Color> observation_colors;
    for (const TrackElement& element : point.track.Elements()) {
      if (!registered.count(element.image_id)) continue;
      const Image& image = reconstruction.Image(element.image_id);
      EXPECT_LT(element.point2D_idx, image.NumPoints2D());
      if (element.point2D_idx >= image.NumPoints2D()) continue;
      const Point2D& point2D = image.Point2D(element.point2D_idx);
      EXPECT_EQ(point2D.point3D_id, id);
      if (const auto sample = ExpectedSample(root, image, point2D)) {
        Color expected;
        for (size_t channel = 0; channel < expected.size(); ++channel) {
          expected[channel] =
              static_cast<uint8_t>(std::round((*sample)(channel)));
        }
        observation_colors.insert(expected);
      }
    }
    const Color color = PointColor(point);
    if (require_all_colored) {
      EXPECT_FALSE(observation_colors.empty());
      EXPECT_NE(color, (Color{0, 0, 0}));
    }
    if (color != Color{0, 0, 0}) {
      EXPECT_TRUE(observation_colors.count(color))
          << "Color has no matching image/pixel observation";
      distinct_colors.insert(color);
    }
  }
  if (require_all_colored) {
    EXPECT_GT(distinct_colors.size(), 1);
  }
}

class MapperColorTestInput : public OIIO::ImageInput {
 public:
  const char* format_name() const override { return "png"; }
  bool open(const std::string& name, OIIO::ImageSpec& spec) override {
    if (std::filesystem::path(name).extension() != ".colmap_mapper_color") {
      return false;
    }
    const auto tag =
        color_image_tags.find(ColorPathKey(std::filesystem::path(name)));
    if (tag == color_image_tags.end()) return false;
    image_tag_ = tag->second;
    is_mapping_thread_ = std::this_thread::get_id() == mapping_thread;
    m_spec = OIIO::ImageSpec(128, 128, 3, OIIO::TypeDesc::UINT8);
    spec = m_spec;
    return true;
  }
  bool close() override { return true; }
  bool read_native_scanline(int, int, int y, int, void* data) override {
    if (!is_mapping_thread_) ++async_scanlines;
    if (color_read_mode == ColorReadMode::kFailure ||
        (color_read_mode == ColorReadMode::kTransient && !is_mapping_thread_)) {
      if (color_read_mode == ColorReadMode::kTransient) {
        ++injected_prefetch_failures;
      }
      errorfmt("Injected mapper color read failure");
      return false;
    }
    auto* row = static_cast<uint8_t*>(data);
    for (int x = 0; x < 128; ++x) {
      row[3 * x + 0] = static_cast<uint8_t>(11 + 3 * image_tag_ + y);
      row[3 * x + 1] = static_cast<uint8_t>(29 + 3 * image_tag_ + x);
      row[3 * x + 2] = static_cast<uint8_t>(43 + 2 * image_tag_ + y);
    }
    if (is_mapping_thread_) ++serial_successes;
    return true;
  }

 private:
  bool is_mapping_thread_ = false;
  int image_tag_ = 0;
};

void CheckMapperColorReads(const ColorReadMode mode,
                           const bool overlap,
                           const int cameras_per_rig,
                           ColorRunResult* result) {
  static const char* extensions[] = {"colmap_mapper_color", nullptr};
  static const bool registered = [] {
    OIIO::declare_imageio_format(
        "colmap_mapper_color",
        []() -> OIIO::ImageInput* { return new MapperColorTestInput; },
        extensions,
        nullptr,
        nullptr,
        "test");
    return true;
  }();
  (void)registered;
  // CreateTestDir cleans once per TEST, while the success test runs both rig
  // sizes. Keep their databases separate so the second fixture starts empty.
  const auto root = CreateTestDir() /
                    ("cameras_" + std::to_string(cameras_per_rig)) /
                    (overlap ? "overlap" : "serial");
  std::filesystem::create_directories(root);
  const auto database = Database::Open(root / "database.db");
  Reconstruction truth;
  SyntheticDatasetOptions synthetic;
  synthetic.num_rigs = 2;
  synthetic.num_cameras_per_rig = cameras_per_rig;
  synthetic.num_frames_per_rig = 7;
  synthetic.num_points3D = 50;
  synthetic.camera_has_prior_focal_length = false;
  synthetic.camera_width = 128;
  synthetic.camera_height = 128;
  synthetic.camera_params = {160, 64, 64, 0.05};
  synthetic.image_extension = ".colmap_mapper_color";
  SetPRNGSeed(0);
  SynthesizeDataset(synthetic, &truth, database.get());
  std::vector<std::string> image_names;
  for (const auto& [_, image] : truth.Images()) {
    image_names.push_back(image.Name());
    std::ofstream(root / image.Name()).close();
  }
  std::sort(image_names.begin(), image_names.end());
  ASSERT_TRUE(std::adjacent_find(image_names.begin(), image_names.end()) ==
              image_names.end());
  color_image_tags.clear();
  for (size_t i = 0; i < image_names.size(); ++i) {
    color_image_tags.emplace(ColorPathKey(root / image_names[i]),
                             static_cast<int>(i + 1));
  }
  ASSERT_LT(image_names.size(), 30);

  auto manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->image_path = root;
  options->extract_colors = true;
  options->overlap_color_decode = overlap;
  options->num_threads = 1;
  options->random_seed = 0;
  color_read_mode = mode;
  mapping_thread = std::this_thread::get_id();
  injected_prefetch_failures = 0;
  serial_successes = 0;
  async_scanlines = 0;
  int later_images = 0;
  int prefinal_async_scanlines = 0;
  std::set<std::string> registered_names;
  IncrementalPipeline mapper(options, database, manager);
  mapper.AddCallback(IncrementalPipeline::INITIAL_IMAGE_PAIR_REG_CALLBACK, [&] {
    ASSERT_EQ(manager->Size(), 1);
    auto& reconstruction = *manager->Get(0);
    for (const auto& [id, point] : reconstruction.Points3D()) {
      reconstruction.Point3D(id).color.setZero();
    }
    for (const image_t id : reconstruction.RegImageIds()) {
      registered_names.insert(reconstruction.Image(id).Name());
    }
    serial_successes = 0;
    async_scanlines = 0;
  });
  mapper.AddCallback(IncrementalPipeline::NEXT_IMAGE_REG_CALLBACK, [&] {
    ++later_images;
    const auto& reconstruction = *manager->Get(0);
    std::set<std::string> new_images;
    for (const image_t id : reconstruction.RegImageIds()) {
      const std::string& name = reconstruction.Image(id).Name();
      if (registered_names.insert(name).second) new_images.insert(name);
    }
    EXPECT_FALSE(new_images.empty());
    if (mode != ColorReadMode::kFailure) {
      CheckNewImageColors(reconstruction, root, new_images);
      CheckAssociatedColors(
          reconstruction, root, /*require_all_colored=*/false);
    }
    result->stages.emplace_back(
        std::vector<std::string>(new_images.begin(), new_images.end()),
        CaptureColors(reconstruction));
    prefinal_async_scanlines = async_scanlines.load();
  });
  mapper.Run();
  ASSERT_EQ(manager->Size(), 1);
  ASSERT_GT(later_images, 0);
  const auto& reconstruction = *manager->Get(0);
  EXPECT_EQ(reconstruction.NumRegImages(), truth.NumImages());
  result->final = CaptureColors(reconstruction);
  ASSERT_FALSE(result->final.empty());
  if (mode == ColorReadMode::kFailure) {
    for (const auto& [_, color] : result->final) {
      EXPECT_EQ(color, (Color{0, 0, 0}));
    }
  } else {
    CheckAssociatedColors(reconstruction, root, /*require_all_colored=*/true);
    if (!overlap || mode == ColorReadMode::kTransient || cameras_per_rig > 1) {
      EXPECT_GT(serial_successes.load(), 0);
    } else {
      EXPECT_EQ(serial_successes.load(), 0);
    }
  }
  if (overlap && mode != ColorReadMode::kFailure) {
    EXPECT_GT(prefinal_async_scanlines, 0);
  } else if (!overlap) {
    EXPECT_EQ(prefinal_async_scanlines, 0);
  }
  if (overlap && mode == ColorReadMode::kTransient) {
    EXPECT_GT(injected_prefetch_failures.load(), 0);
  }
}

void CheckColorPair(const ColorReadMode mode, const int cameras_per_rig) {
  ColorRunResult serial;
  ColorRunResult overlap;
  CheckMapperColorReads(mode, /*overlap=*/false, cameras_per_rig, &serial);
  CheckMapperColorReads(mode, /*overlap=*/true, cameras_per_rig, &overlap);
  ASSERT_FALSE(serial.stages.empty());
  EXPECT_EQ(serial.stages, overlap.stages);
  EXPECT_EQ(serial.final, overlap.final);
}

TEST(IncrementalPipeline, ColorReadStableSuccess) {
  for (const int cameras_per_rig : {1, 2}) {
    SCOPED_TRACE(cameras_per_rig);
    CheckColorPair(ColorReadMode::kSuccess, cameras_per_rig);
  }
}

TEST(IncrementalPipeline, ColorReadTransientPrefetchFailure) {
  CheckColorPair(ColorReadMode::kTransient, /*cameras_per_rig=*/1);
}

TEST(IncrementalPipeline, ColorReadStableFailure) {
  CheckColorPair(ColorReadMode::kFailure, /*cameras_per_rig=*/1);
}

TEST(IncrementalPipeline, WithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(std::make_shared<IncrementalPipelineOptions>(),
                             database,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, WithoutNoiseSphericalCameras) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 5;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_model_id =
      EquirectangularCameraModel::model_id;
  synthetic_dataset_options.camera_width = 1000;
  synthetic_dataset_options.camera_height = 500;
  synthetic_dataset_options.camera_params = {1000, 500};
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(std::make_shared<IncrementalPipelineOptions>(),
                             database,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  const Reconstruction& reconstruction = *reconstruction_manager->Get(0);
  EXPECT_EQ(reconstruction.NumRegImages(), gt_reconstruction.NumImages());
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, WithoutNoiseAndWithNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.05;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  for (const bool refine_sensor_from_rig : {true, false}) {
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto options = std::make_shared<IncrementalPipelineOptions>();
    options->ba_refine_sensor_from_rig = refine_sensor_from_rig;
    IncrementalPipeline mapper(options, database, reconstruction_manager);
    mapper.Run();

    ASSERT_EQ(reconstruction_manager->Size(), 1);
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(
                    *reconstruction_manager->Get(0),
                    /*max_rotation_error_deg=*/1e-2,
                    /*max_proj_center_error=*/1e-3,
                    /*max_scale_error=*/refine_sensor_from_rig ? 1e-2 : 1e-4));
  }
}

TEST(IncrementalPipeline, UnknownSensorFromRigExitsGracefully) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.05;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Set one of the sensor from rig to unknown.
  auto rig = database->ReadAllRigs()[0];
  rig.NonRefSensors().begin()->second.reset();
  database->UpdateRig(rig);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  IncrementalPipeline mapper(options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 0);
}

TEST(IncrementalPipeline, WithNonTrivialFramesAndConstantRigsAndCameras) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.05;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  constexpr int kConstantRigId = 1;
  constexpr int kConstantCameraId = 1;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->constant_rigs.insert(kConstantRigId);
  options->constant_cameras.insert(kConstantCameraId);
  IncrementalPipeline mapper(options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  auto& reconstruction = *reconstruction_manager->Get(0);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-3));

  for (const auto& [sensor_id, sensor_from_rig] :
       reconstruction.Rig(kConstantRigId).NonRefSensors()) {
    EXPECT_THAT(
        sensor_from_rig.value(),
        Rigid3dNear(
            gt_reconstruction.Rig(kConstantRigId).SensorFromRig(sensor_id),
            /*rtol=*/1e-4,
            /*ttol=*/1e-4));
  }
  EXPECT_EQ(reconstruction.Camera(kConstantCameraId).params,
            gt_reconstruction.Camera(kConstantCameraId).params);
}

TEST(IncrementalPipeline, WithoutNoiseAndWithPanoramicNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  for (const bool refine_sensor_from_rig : {true, false}) {
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto options = std::make_shared<IncrementalPipelineOptions>();
    options->ba_refine_sensor_from_rig = refine_sensor_from_rig;
    IncrementalPipeline mapper(options, database, reconstruction_manager);
    mapper.Run();

    ASSERT_EQ(reconstruction_manager->Size(), 1);
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(*reconstruction_manager->Get(0),
                                   /*max_rotation_error_deg=*/1e-2,
                                   /*max_proj_center_error=*/1e-3));
  }
}

TEST(IncrementalPipeline, WithPriorFocalLength) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(std::make_shared<IncrementalPipelineOptions>(),
                             database,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, WithNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(std::make_shared<IncrementalPipelineOptions>(),
                             database,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  auto reconstruction = reconstruction_manager->Get(0);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));

  // After the pipeline runs, point3D.error must be in pixel units, i.e.
  // equal to what UpdatePoint3DErrors would recompute.
  ASSERT_GT(reconstruction->NumPoints3D(), 0u);
  const double mean_after_run = reconstruction->ComputeMeanReprojectionError();
  reconstruction->UpdatePoint3DErrors();
  EXPECT_DOUBLE_EQ(mean_after_run,
                   reconstruction->ComputeMeanReprojectionError());
}

TEST(IncrementalPipeline, IgnoreRedundantPoints3D) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->mapper.ba_global_ignore_redundant_points3D = true;
  options->mapper.ba_global_ignore_redundant_points3D_min_coverage_gain = 0.5;
  IncrementalPipeline mapper(options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, StructureLessRegistrationOnly) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->structure_less_registration_only = true;
  IncrementalPipeline mapper(options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-3,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, MultiReconstruction) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction1;
  Reconstruction gt_reconstruction2;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction1, database.get());
  synthetic_dataset_options.num_frames_per_rig = 4;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction2, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto mapper_options = std::make_shared<IncrementalPipelineOptions>();
  mapper_options->min_model_size = 4;
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 2);
  Reconstruction* computed_reconstruction1 = nullptr;
  Reconstruction* computed_reconstruction2 = nullptr;
  if (reconstruction_manager->Get(0)->NumRegImages() == 5) {
    computed_reconstruction1 = reconstruction_manager->Get(0).get();
    computed_reconstruction2 = reconstruction_manager->Get(1).get();
  } else {
    computed_reconstruction1 = reconstruction_manager->Get(1).get();
    computed_reconstruction2 = reconstruction_manager->Get(0).get();
  }
  EXPECT_THAT(gt_reconstruction1,
              ReconstructionNear(*computed_reconstruction1,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
  EXPECT_THAT(gt_reconstruction2,
              ReconstructionNear(*computed_reconstruction2,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, FixExistingFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  for (const bool fix_existing_frames : {false, true}) {
    if (fix_existing_frames) {
      ASSERT_EQ(reconstruction_manager->Size(), 1);
      Reconstruction& reconstruction = *reconstruction_manager->Get(0);
      // De-register a frame that expect to be re-registered in the second run.
      reconstruction.DeRegisterFrame(1);
      // Clear all the observations of one image but keep it registered. We do
      // not expect fixed images to be filtered (due to insufficient
      // observations).
      Image& image2 = reconstruction.Image(2);
      for (point2D_t point2D_idx = 0; point2D_idx < image2.NumPoints2D();
           ++point2D_idx) {
        if (image2.Point2D(point2D_idx).HasPoint3D()) {
          reconstruction.DeleteObservation(image2.ImageId(), point2D_idx);
        }
      }
    }
    options->fix_existing_frames = fix_existing_frames;
    IncrementalPipeline mapper(options, database, reconstruction_manager);
    mapper.Run();

    ASSERT_EQ(reconstruction_manager->Size(), 1);
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(*reconstruction_manager->Get(0),
                                   /*max_rotation_error_deg=*/1e-2,
                                   /*max_proj_center_error=*/1e-4));
  }
}

TEST(IncrementalPipeline, ChainedMatches) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::CHAINED;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->num_threads = 1;
  IncrementalPipeline mapper(options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, PriorBasedSfMWithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.prior_position_stddev = 0.0;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();
  mapper_options->use_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);

  // No noise on prior so do not align gt & computed (expected to be aligned
  // from PositionPriorBundleAdjustment)
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02,
                                 /*align=*/false));
}

TEST(IncrementalPipeline, PriorBasedSfMWithoutNoiseAndWithNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;

  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Match the common rig setup where only the reference sensor has absolute
  // positions. Registering two frames then yields many images but only two
  // usable pose priors.
  FlatHashSet<sensor_t> ref_sensor_ids;
  for (const auto& [_, rig] : gt_reconstruction.Rigs()) {
    ref_sensor_ids.insert(rig.RefSensorId());
  }
  const std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  database->ClearPosePriors();
  for (const PosePrior& pose_prior : pose_priors) {
    if (ref_sensor_ids.count(pose_prior.corr_data_id.sensor_id)) {
      database->WritePosePrior(pose_prior);
    }
  }

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02,
                                 /*align=*/false));
}

TEST(IncrementalPipeline, PriorBasedSfMWithNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.prior_position_stddev = 1.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(IncrementalPipeline, GPSPriorBasedSfMWithNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;

  synthetic_dataset_options.prior_position = true;
  synthetic_dataset_options.prior_position_coordinate_system =
      PosePrior::CoordinateSystem::WGS84;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.prior_position_stddev = 1.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(IncrementalPipeline, SfMWithRandomSeedStability) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.prior_position = false;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto run_mapper = [&](int num_threads, int random_seed) {
    auto mapper_options = std::make_shared<IncrementalPipelineOptions>();
    mapper_options->use_prior_position = false;
    mapper_options->num_threads = num_threads;
    mapper_options->random_seed = random_seed;
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    IncrementalPipeline mapper(
        mapper_options, database, reconstruction_manager);
    mapper.Run();
    EXPECT_EQ(reconstruction_manager->Size(), 1);
    return reconstruction_manager;
  };

  constexpr int kRandomSeed = 42;

  auto reconstruction_manager0 =
      run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
  auto reconstruction_manager1 =
      run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
  EXPECT_THAT(*reconstruction_manager0->Get(0),
              ReconstructionEq(*reconstruction_manager1->Get(0)));
}

TEST(IncrementalPipeline, PriorBasedSfMWithRandomSeedStability) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.1;
  synthetic_noise_options.prior_position_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto run_mapper = [&](int num_threads, int random_seed) {
    auto mapper_options = std::make_shared<IncrementalPipelineOptions>();
    mapper_options->use_prior_position = true;
    mapper_options->num_threads = num_threads;
    mapper_options->random_seed = random_seed;
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    IncrementalPipeline mapper(
        mapper_options, database, reconstruction_manager);
    mapper.Run();
    EXPECT_EQ(reconstruction_manager->Size(), 1);
    return reconstruction_manager;
  };

  constexpr int kRandomSeed = 42;

  auto reconstruction_manager0 =
      run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
  auto reconstruction_manager1 =
      run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
  EXPECT_THAT(*reconstruction_manager0->Get(0),
              ReconstructionEq(*reconstruction_manager1->Get(0)));
}

TEST(IncrementalPipelineOptions, DefaultPriorPositionLossScale) {
  const IncrementalPipelineOptions options;
  EXPECT_DOUBLE_EQ(options.prior_position_loss_scale,
                   std::sqrt(kChiSquare95ThreeDof));
  EXPECT_DOUBLE_EQ(IncrementalMapper::Options().prior_position_loss_scale,
                   std::sqrt(kChiSquare95ThreeDof));
  EXPECT_DOUBLE_EQ(
      CeresPosePriorBundleAdjustmentOptions().prior_position_loss_scale,
      std::sqrt(kChiSquare95ThreeDof));
}

TEST(IncrementalPipelineOptions, DefaultSolverThresholds) {
  const IncrementalPipelineOptions options;
  const CeresBundleAdjustmentOptions defaults;
  for (const auto& ba_options :
       {options.LocalBundleAdjustment(), options.GlobalBundleAdjustment()}) {
    ASSERT_TRUE(ba_options.ceres);
    EXPECT_EQ(ba_options.ceres->min_num_images_gpu_solver,
              defaults.min_num_images_gpu_solver);
    EXPECT_EQ(ba_options.ceres->max_num_images_direct_dense_cpu_solver,
              defaults.max_num_images_direct_dense_cpu_solver);
    EXPECT_EQ(ba_options.ceres->max_num_images_direct_sparse_cpu_solver,
              defaults.max_num_images_direct_sparse_cpu_solver);
    EXPECT_EQ(ba_options.ceres->max_num_images_direct_dense_gpu_solver,
              defaults.max_num_images_direct_dense_gpu_solver);
    EXPECT_EQ(ba_options.ceres->max_num_images_direct_sparse_gpu_solver,
              defaults.max_num_images_direct_sparse_gpu_solver);
  }
}

TEST(IncrementalPipelineOptions, PropagatesSolverThresholds) {
  IncrementalPipelineOptions options;
  options.ba_min_num_images_gpu_solver = 7;
  options.ba_max_num_images_direct_dense_cpu_solver = 11;
  options.ba_max_num_images_direct_sparse_cpu_solver = 101;
  options.ba_max_num_images_direct_dense_gpu_solver = 23;
  options.ba_max_num_images_direct_sparse_gpu_solver = 1001;
  EXPECT_TRUE(options.Check());
  for (const auto& ba_options :
       {options.LocalBundleAdjustment(), options.GlobalBundleAdjustment()}) {
    ASSERT_TRUE(ba_options.ceres);
    EXPECT_EQ(ba_options.ceres->min_num_images_gpu_solver, 7);
    EXPECT_EQ(ba_options.ceres->max_num_images_direct_dense_cpu_solver, 11);
    EXPECT_EQ(ba_options.ceres->max_num_images_direct_sparse_cpu_solver, 101);
    EXPECT_EQ(ba_options.ceres->max_num_images_direct_dense_gpu_solver, 23);
    EXPECT_EQ(ba_options.ceres->max_num_images_direct_sparse_gpu_solver, 1001);
    ASSERT_TRUE(ba_options.caspar);
    EXPECT_EQ(ba_options.caspar->solver_iter_max,
              CasparBundleAdjustmentOptions().solver_iter_max);
  }
}

TEST(IncrementalPipelineOptions, CheckSolverThresholds) {
  IncrementalPipelineOptions options;
  options.ba_max_num_images_direct_dense_cpu_solver =
      options.ba_max_num_images_direct_sparse_cpu_solver;
  EXPECT_FALSE(options.Check());
  options.ba_max_num_images_direct_dense_cpu_solver = 50;
  EXPECT_TRUE(options.Check());
  options.ba_max_num_images_direct_dense_gpu_solver =
      options.ba_max_num_images_direct_sparse_gpu_solver;
  EXPECT_FALSE(options.Check());
}

TEST(IncrementalPipelineOptions, PropagatesExplicitMaxNumIterations) {
  // Explicitly set iteration bounds must be forwarded to both backends,
  // including values that coincide with the previous compiled-in defaults.
  IncrementalPipelineOptions options;
  options.ba_local_max_num_iterations = 25;
  options.ba_global_max_num_iterations = 50;

  const BundleAdjustmentOptions local_options = options.LocalBundleAdjustment();
  ASSERT_TRUE(local_options.ceres);
  EXPECT_EQ(local_options.ceres->solver_options.max_num_iterations, 25);
  ASSERT_TRUE(local_options.caspar);
  EXPECT_EQ(local_options.caspar->solver_iter_max, 25);

  const BundleAdjustmentOptions global_options =
      options.GlobalBundleAdjustment();
  ASSERT_TRUE(global_options.ceres);
  EXPECT_EQ(global_options.ceres->solver_options.max_num_iterations, 50);
  ASSERT_TRUE(global_options.caspar);
  EXPECT_EQ(global_options.caspar->solver_iter_max, 50);
}

TEST(IncrementalPipelineOptions, DefaultMaxNumIterationsUsesBackendDefaults) {
  // With the -1 sentinel default, each backend must keep its own default:
  // Ceres the previous 25/50 mapper defaults, Caspar its own tuned
  // solver_iter_max (see review discussion on #4382/PR #4527).
  const IncrementalPipelineOptions options;
  ASSERT_EQ(options.ba_local_max_num_iterations, -1);
  ASSERT_EQ(options.ba_global_max_num_iterations, -1);
  const int caspar_default = CasparBundleAdjustmentOptions().solver_iter_max;

  const BundleAdjustmentOptions local_options = options.LocalBundleAdjustment();
  ASSERT_TRUE(local_options.ceres);
  EXPECT_EQ(local_options.ceres->solver_options.max_num_iterations, 25);
  ASSERT_TRUE(local_options.caspar);
  EXPECT_EQ(local_options.caspar->solver_iter_max, caspar_default);

  const BundleAdjustmentOptions global_options =
      options.GlobalBundleAdjustment();
  ASSERT_TRUE(global_options.ceres);
  EXPECT_EQ(global_options.ceres->solver_options.max_num_iterations, 50);
  ASSERT_TRUE(global_options.caspar);
  EXPECT_EQ(global_options.caspar->solver_iter_max, caspar_default);
}

TEST(IncrementalPipelineOptions, EffBaMaxNumIterations) {
  IncrementalPipelineOptions options;
  const int caspar_default = CasparBundleAdjustmentOptions().solver_iter_max;

  // Sentinel resolves to the configured backend's default.
  EXPECT_EQ(options.EffBaLocalMaxNumIterations(), 25);
  EXPECT_EQ(options.EffBaGlobalMaxNumIterations(), 50);
  options.ba_local_backend = BundleAdjustmentBackend::CASPAR;
  options.ba_global_backend = BundleAdjustmentBackend::CASPAR;
  EXPECT_EQ(options.EffBaLocalMaxNumIterations(), caspar_default);
  EXPECT_EQ(options.EffBaGlobalMaxNumIterations(), caspar_default);

  // Explicit values win regardless of backend.
  options.ba_local_max_num_iterations = 7;
  options.ba_global_max_num_iterations = 13;
  EXPECT_EQ(options.EffBaLocalMaxNumIterations(), 7);
  EXPECT_EQ(options.EffBaGlobalMaxNumIterations(), 13);
}

}  // namespace
}  // namespace colmap
