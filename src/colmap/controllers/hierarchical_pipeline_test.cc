// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/hierarchical_pipeline.h"

#include "colmap/estimators/alignment.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

using CacheAdjacency = std::map<std::pair<image_t, point2D_t>,
                                std::vector<std::pair<image_t, point2D_t>>>;

CacheAdjacency CaptureCacheAdjacency(const DatabaseCache& cache) {
  CacheAdjacency result;
  const auto graph = cache.CorrespondenceGraph();
  for (const auto& [id, image] : cache.Images()) {
    if (!graph->ExistsImage(id)) continue;
    for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
      auto& neighbors = result[{id, idx}];
      const auto range = graph->FindCorrespondences(id, idx);
      for (auto corr = range.beg; corr != range.end; ++corr) {
        neighbors.emplace_back(corr->image_id, corr->point2D_idx);
      }
    }
  }
  return result;
}

void ExpectEqualCacheGeometry(const TwoViewGeometry& expected,
                              const TwoViewGeometry& actual) {
  EXPECT_EQ(actual.config, expected.config);
  EXPECT_EQ(actual.E, expected.E);
  EXPECT_EQ(actual.F, expected.F);
  EXPECT_EQ(actual.H, expected.H);
  EXPECT_EQ(actual.cam2_from_cam1, expected.cam2_from_cam1);
  EXPECT_EQ(actual.camera1, expected.camera1);
  EXPECT_EQ(actual.camera2, expected.camera2);
  EXPECT_EQ(actual.tri_angle, expected.tri_angle);
  // Preserve the order within each oriented pair, not just the match count.
  ASSERT_EQ(actual.inlier_matches.size(), expected.inlier_matches.size());
  for (size_t i = 0; i < expected.inlier_matches.size(); ++i) {
    EXPECT_EQ(actual.inlier_matches[i].point2D_idx1,
              expected.inlier_matches[i].point2D_idx1);
    EXPECT_EQ(actual.inlier_matches[i].point2D_idx2,
              expected.inlier_matches[i].point2D_idx2);
  }
}

void ExpectEqualClusterCaches(const DatabaseCache& expected,
                              const DatabaseCache& actual,
                              const std::string& order_property = "") {
  ASSERT_EQ(actual.NumRigs(), expected.NumRigs());
  ASSERT_EQ(actual.NumCameras(), expected.NumCameras());
  ASSERT_EQ(actual.NumFrames(), expected.NumFrames());
  ASSERT_EQ(actual.NumImages(), expected.NumImages());
  EXPECT_EQ(actual.NumPosePriors(), expected.NumPosePriors());
  EXPECT_EQ(actual.PosePriors(), expected.PosePriors());
  for (const auto& [id, rig] : expected.Rigs()) {
    ASSERT_TRUE(actual.ExistsRig(id));
    EXPECT_EQ(actual.Rig(id), rig);
  }
  for (const auto& [id, camera] : expected.Cameras()) {
    ASSERT_TRUE(actual.ExistsCamera(id));
    EXPECT_EQ(actual.Camera(id), camera);
  }
  for (const auto& [id, frame] : expected.Frames()) {
    ASSERT_TRUE(actual.ExistsFrame(id));
    EXPECT_EQ(actual.Frame(id), frame);  // Includes the full frame data roster.
  }
  const auto expected_graph = expected.CorrespondenceGraph();
  const auto actual_graph = actual.CorrespondenceGraph();
  EXPECT_EQ(actual_graph->NumImages(), expected_graph->NumImages());
  EXPECT_EQ(actual_graph->NumImagePairs(), expected_graph->NumImagePairs());
  for (const auto& [id, image] : expected.Images()) {
    ASSERT_TRUE(actual.ExistsImage(id));
    EXPECT_EQ(actual.Image(id), image);
    ASSERT_EQ(actual_graph->ExistsImage(id), expected_graph->ExistsImage(id));
    if (!expected_graph->ExistsImage(id)) continue;
    EXPECT_EQ(actual_graph->NumObservationsForImage(id),
              expected_graph->NumObservationsForImage(id));
    EXPECT_EQ(actual_graph->NumCorrespondencesForImage(id),
              expected_graph->NumCorrespondencesForImage(id));
  }
  auto expected_pairs = expected_graph->ImagePairs();
  auto actual_pairs = actual_graph->ImagePairs();
  std::sort(expected_pairs.begin(), expected_pairs.end());
  std::sort(actual_pairs.begin(), actual_pairs.end());
  ASSERT_EQ(actual_pairs, expected_pairs);
  for (const image_pair_t pair_id : expected_pairs) {
    auto [id1, id2] = PairIdToImagePair(pair_id);
    SCOPED_TRACE(pair_id);
    for (int orientation = 0; orientation < 2; ++orientation) {
      EXPECT_EQ(actual_graph->NumMatchesBetweenImages(id1, id2),
                expected_graph->NumMatchesBetweenImages(id1, id2));
      ExpectEqualCacheGeometry(
          expected_graph->ExtractTwoViewGeometry(id1, id2, true),
          actual_graph->ExtractTwoViewGeometry(id1, id2, true));
      std::swap(id1, id2);
    }
  }
  // Global neighbor order may change after one graph reconstruction instead
  // of two. Compare its full content, and record order differences separately.
  auto expected_adjacency = CaptureCacheAdjacency(expected);
  auto actual_adjacency = CaptureCacheAdjacency(actual);
  size_t order_differences = 0;
  for (auto& [point, neighbors] : expected_adjacency) {
    auto& actual_neighbors = actual_adjacency.at(point);
    order_differences += neighbors != actual_neighbors;
    std::sort(neighbors.begin(), neighbors.end());
    std::sort(actual_neighbors.begin(), actual_neighbors.end());
  }
  EXPECT_EQ(actual_adjacency, expected_adjacency);
  if (!order_property.empty()) {
    ::testing::Test::RecordProperty(order_property,
                                    std::to_string(order_differences));
  }
}

class HierarchicalCacheFilterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    database = Database::Open(kInMemorySqliteDatabasePath);
    SyntheticDatasetOptions options;
    options.num_rigs = 1;
    options.num_cameras_per_rig = 2;
    options.num_frames_per_rig = 3;
    options.num_points3D = 12;
    options.prior_position = true;
    options.prior_gravity = true;
    options.prior_position_coordinate_system =
        PosePrior::CoordinateSystem::WGS84;
    SynthesizeDataset(options, &reconstruction, database.get());
    std::vector<frame_t> frame_ids;
    for (const auto& [id, _] : reconstruction.Frames()) frame_ids.push_back(id);
    std::sort(frame_ids.begin(), frame_ids.end());
    for (const frame_t id : frame_ids) {
      for (const data_t& data : reconstruction.Frame(id).ImageIds()) {
        images.push_back(data.id);
      }
    }
    ASSERT_EQ(images.size(), 6);
    for (size_t i = 0; i < images.size(); ++i) {
      for (size_t j = i + 1; j < images.size(); ++j) {
        database->DeleteTwoViewGeometry(images[i], images[j]);
      }
    }
    TwoViewGeometry geometry;
    geometry.config = TwoViewGeometry::CALIBRATED;
    geometry.E = (Eigen::Matrix3d() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
    geometry.F = *geometry.E * 2;
    geometry.H =
        (Eigen::Matrix3d() << 1, 0.2, 0, 0, 2, 0.3, 0, 0, 1).finished();
    geometry.cam2_from_cam1 = Rigid3d(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX())),
        Eigen::Vector3d(1, 2, 3));
    geometry.camera1 =
        reconstruction.Camera(reconstruction.Image(images[0]).CameraId());
    geometry.camera2 =
        reconstruction.Camera(reconstruction.Image(images[3]).CameraId());
    geometry.camera2->params[0] += 11;
    geometry.inlier_matches = {{4, 1}, {0, 3}, {2, 0}};
    for (const auto& [i, j] : {std::pair{0, 2}, {2, 4}, {4, 5}}) {
      database->WriteTwoViewGeometry(images[i], images[j], geometry);
    }
    geometry.config = TwoViewGeometry::WATERMARK;
    database->WriteTwoViewGeometry(images[1], images[3], geometry);
    geometry.config = TwoViewGeometry::CALIBRATED;
    geometry.inlier_matches.resize(1);
    database->WriteTwoViewGeometry(images[0], images[1], geometry);
    geometry.inlier_matches.clear();
    database->WriteTwoViewGeometry(images[0], images[5], geometry);
  }

  FlatHashSet<std::string> ClusterNames() const {
    return {reconstruction.Image(images[0]).Name(),
            reconstruction.Image(images[2]).Name()};
  }

  std::shared_ptr<DatabaseCache> ComposedCache(
      const std::shared_ptr<DatabaseCache>& source,
      const FlatHashSet<std::string>& names,
      const IncrementalPipelineOptions& options,
      std::shared_ptr<ReconstructionManager> manager =
          std::make_shared<ReconstructionManager>()) {
    DatabaseCache::Options cluster_options;
    cluster_options.min_num_matches = options.min_num_matches;
    cluster_options.image_names = names;
    auto cluster = DatabaseCache::CreateFromCache(*source, cluster_options);
    IncrementalPipeline mapper(
        std::make_shared<IncrementalPipelineOptions>(options),
        cluster,
        std::move(manager));
    return mapper.DatabaseCache();
  }

  std::shared_ptr<DatabaseCache> DirectCache(
      const std::shared_ptr<DatabaseCache>& source,
      const IncrementalPipelineOptions& options,
      std::shared_ptr<ReconstructionManager> manager =
          std::make_shared<ReconstructionManager>()) {
    IncrementalPipeline mapper(
        std::make_shared<IncrementalPipelineOptions>(options),
        source,
        std::move(manager));
    return mapper.DatabaseCache();
  }

  std::shared_ptr<Database> database;
  Reconstruction reconstruction;
  std::vector<image_t> images;
};

TEST_F(HierarchicalCacheFilterTest, SingleFilterMatchesComposedCache) {
  for (const int min_num_matches : {1, 2, 4}) {
    for (const bool ignore_watermarks : {false, true}) {
      for (const bool use_prior_position : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << min_num_matches << ", " << ignore_watermarks << ", "
                     << use_prior_position);
        DatabaseCache::Options source_options;
        source_options.min_num_matches = min_num_matches;
        source_options.ignore_watermarks = ignore_watermarks;
        auto source = DatabaseCache::Create(*database, source_options);
        auto snapshot = DatabaseCache::Create(*database, source_options);
        const auto source_adjacency = CaptureCacheAdjacency(*source);
        const auto source_graph = source->CorrespondenceGraph();
        IncrementalPipelineOptions options;
        options.min_num_matches = min_num_matches;
        options.ignore_watermarks = ignore_watermarks;
        options.use_prior_position = use_prior_position;
        const auto names = ClusterNames();
        auto baseline = ComposedCache(source, names, options);
        options.image_names.assign(names.begin(), names.end());
        auto candidate = DirectCache(source, options);
        EXPECT_NE(baseline.get(), source.get());
        EXPECT_NE(candidate.get(), source.get());
        EXPECT_NE(candidate->CorrespondenceGraph().get(), source_graph.get());
        ExpectEqualClusterCaches(*baseline,
                                 *candidate,
                                 "adjacency_order_differences_" +
                                     std::to_string(min_num_matches) + "_" +
                                     std::to_string(ignore_watermarks) + "_" +
                                     std::to_string(use_prior_position));
        EXPECT_EQ(candidate->NumFrames(), min_num_matches > 3 ? 0 : 2);
        EXPECT_EQ(candidate->NumImages(), min_num_matches > 3 ? 0 : 4);
        for (size_t i = 0; i < images.size(); ++i) {
          EXPECT_EQ(candidate->ExistsImage(images[i]),
                    min_num_matches <= 3 && i < 4);
        }
        auto expected_priors = snapshot->PosePriors();
        if (use_prior_position) ConvertPosePriorsToENU(expected_priors);
        EXPECT_EQ(candidate->PosePriors(), expected_priors);
        ExpectEqualClusterCaches(*snapshot, *source);
        EXPECT_EQ(source->CorrespondenceGraph().get(), source_graph.get());
        EXPECT_EQ(CaptureCacheAdjacency(*source), source_adjacency);
      }
    }
  }
}

TEST_F(HierarchicalCacheFilterTest, EmptySourceAndUnconnectedCluster) {
  IncrementalPipelineOptions options;
  options.min_num_matches = 2;
  auto source = DatabaseCache::Create(*database, {});
  const FlatHashSet<std::string> names = {
      reconstruction.Image(images[0]).Name()};
  auto baseline = ComposedCache(source, names, options);
  options.image_names.assign(names.begin(), names.end());
  auto candidate = DirectCache(source, options);
  ExpectEqualClusterCaches(*baseline, *candidate);
  EXPECT_EQ(candidate->NumImages(), 0);

  auto empty_database = Database::Open(kInMemorySqliteDatabasePath);
  auto empty = DatabaseCache::Create(*empty_database, {});
  options.image_names.clear();
  baseline = ComposedCache(empty, names, options);
  options.image_names.assign(names.begin(), names.end());
  candidate = DirectCache(empty, options);
  ExpectEqualClusterCaches(*baseline, *candidate);
  EXPECT_EQ(candidate->NumImages(), 0);
}

// These counterexamples deliberately bypass the production guard. They witness
// why the original two-stage path must remain for each excluded option. The
// maintained HierarchicalPipeline tests below exercise the real call site.
TEST_F(HierarchicalCacheFilterTest, ExplicitNamesNeedComposedFilter) {
  auto source = DatabaseCache::Create(*database, {});
  IncrementalPipelineOptions options;
  options.min_num_matches = 2;
  options.image_names = {reconstruction.Image(images[4]).Name(),
                         reconstruction.Image(images[5]).Name()};
  auto baseline = ComposedCache(source, ClusterNames(), options);
  auto unguarded = DirectCache(source, options);
  EXPECT_EQ(baseline->NumImages(), 0);
  EXPECT_TRUE(unguarded->ExistsImage(images[4]));
  EXPECT_TRUE(unguarded->ExistsImage(images[5]));
}

TEST_F(HierarchicalCacheFilterTest, LoadAllImagesNeedsComposedFilter) {
  auto source = DatabaseCache::Create(*database, {});
  IncrementalPipelineOptions options;
  options.min_num_matches = 2;
  options.load_all_images = true;
  const FlatHashSet<std::string> names = {
      reconstruction.Image(images[0]).Name()};
  auto baseline = ComposedCache(source, names, options);
  options.image_names.assign(names.begin(), names.end());
  auto unguarded = DirectCache(source, options);
  EXPECT_EQ(baseline->NumImages(), 0);
  EXPECT_TRUE(unguarded->ExistsImage(images[0]));
  EXPECT_TRUE(unguarded->ExistsImage(images[1]));
}

TEST_F(HierarchicalCacheFilterTest, RegisteredExpansionNeedsComposedFilter) {
  auto source = DatabaseCache::Create(*database, {});
  auto manager = std::make_shared<ReconstructionManager>();
  manager->Get(manager->Add()) =
      std::make_shared<Reconstruction>(reconstruction);
  IncrementalPipelineOptions options;
  options.min_num_matches = 2;
  auto names = ClusterNames();
  auto baseline = ComposedCache(source, names, options, manager);
  options.image_names.assign(names.begin(), names.end());
  auto unguarded = DirectCache(source, options, manager);
  EXPECT_EQ(baseline->NumImages(), 4);
  EXPECT_FALSE(baseline->ExistsImage(images[4]));
  EXPECT_FALSE(baseline->ExistsImage(images[5]));
  EXPECT_TRUE(unguarded->ExistsImage(images[4]));
  EXPECT_TRUE(unguarded->ExistsImage(images[5]));
}

void ExpectEqualReconstructions(const Reconstruction& gt,
                                const Reconstruction& computed,
                                const double max_rotation_error_deg,
                                const double max_proj_center_error,
                                const double num_obs_tolerance) {
  EXPECT_EQ(computed.NumCameras(), gt.NumCameras());
  EXPECT_EQ(computed.NumImages(), gt.NumImages());
  EXPECT_EQ(computed.NumRegImages(), gt.NumRegImages());
  EXPECT_GE(computed.ComputeNumObservations(),
            (1 - num_obs_tolerance) * gt.ComputeNumObservations());

  Sim3d gt_from_computed;
  ASSERT_TRUE(AlignReconstructionsViaProjCenters(computed,
                                                 gt,
                                                 /*max_proj_center_error=*/0.1,
                                                 &gt_from_computed));

  const std::vector<ImageAlignmentError> errors =
      ComputeImageAlignmentError(computed, gt, gt_from_computed);
  EXPECT_EQ(errors.size(), gt.NumImages());
  for (const auto& error : errors) {
    EXPECT_LT(error.rotation_error_deg, max_rotation_error_deg);
    EXPECT_LT(error.proj_center_error, max_proj_center_error);
  }
}

// Automatic reconstruction forwards its image selection into these options.
// Exercise that contract at the real hierarchical child-construction boundary,
// alongside the empty-name control. A single leaf avoids unrelated empty-leaf
// behavior and makes overwriting the selected names admit the whole dataset.
TEST(HierarchicalPipeline, ImageNamesPreserveSelectedScope) {
  for (const bool select_subset : {false, true}) {
    SCOPED_TRACE(select_subset);
    SetPRNGSeed(1);
    auto database = Database::Open(kInMemorySqliteDatabasePath);
    Reconstruction expected;
    SyntheticDatasetOptions dataset_options;
    dataset_options.num_rigs = 2;
    dataset_options.num_cameras_per_rig = 1;
    dataset_options.num_frames_per_rig = 5;
    dataset_options.num_points3D = 100;
    dataset_options.camera_has_prior_focal_length = true;
    dataset_options.two_view_geometry_has_relative_pose = true;
    SynthesizeDataset(dataset_options, &expected, database.get());

    HierarchicalPipelineOptions options;
    options.num_threads = 1;
    options.num_workers = 1;
    options.clustering_options.leaf_max_num_images = 100;
    options.incremental_options.extract_colors = false;
    options.incremental_options.random_seed = 1;
    if (select_subset) {
      const rig_t selected_rig = expected.Rigs().begin()->first;
      std::vector<frame_t> excluded_frames;
      for (const auto& [id, frame] : expected.Frames()) {
        if (frame.RigId() == selected_rig) {
          for (const data_t& data : frame.ImageIds()) {
            options.incremental_options.image_names.push_back(
                expected.Image(data.id).Name());
          }
        } else {
          excluded_frames.push_back(id);
        }
      }
      for (const frame_t id : excluded_frames) expected.DeRegisterFrame(id);
      expected.TearDown();
    }
    ASSERT_EQ(expected.NumRegImages(), select_subset ? 5 : 10);
    auto expected_ids = expected.RegImageIds();
    std::sort(expected_ids.begin(), expected_ids.end());

    auto manager = std::make_shared<ReconstructionManager>();
    HierarchicalPipeline mapper(options, database, manager);
    mapper.Run();

    ASSERT_EQ(manager->Size(), 1);
    const Reconstruction& actual = *manager->Get(0);
    ASSERT_TRUE(actual.IsValid());
    auto actual_ids = actual.RegImageIds();
    std::sort(actual_ids.begin(), actual_ids.end());
    EXPECT_EQ(actual_ids, expected_ids);
    ExpectEqualReconstructions(expected,
                               actual,
                               /*max_rotation_error_deg=*/1e-2,
                               /*max_proj_center_error=*/1e-4,
                               /*num_obs_tolerance=*/0);
  }
}

TEST(HierarchicalPipeline, WithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 20;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalPipelineOptions mapper_options;
  mapper_options.clustering_options.leaf_max_num_images = 5;
  mapper_options.clustering_options.image_overlap = 3;
  HierarchicalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  auto reconstruction = reconstruction_manager->Get(0);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction,
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/5e-4,
                             /*num_obs_tolerance=*/0);

  // After the pipeline runs, point3D.error must be in pixel units, i.e.
  // equal to what UpdatePoint3DErrors would recompute.
  ASSERT_GT(reconstruction->NumPoints3D(), 0u);
  const double mean_after_run = reconstruction->ComputeMeanReprojectionError();
  reconstruction->UpdatePoint3DErrors();
  EXPECT_DOUBLE_EQ(mean_after_run,
                   reconstruction->ComputeMeanReprojectionError());
}

TEST(HierarchicalPipeline, WithoutNoiseAndNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.05;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalPipelineOptions mapper_options;
  mapper_options.clustering_options.leaf_max_num_images = 10;
  mapper_options.clustering_options.image_overlap = 3;
  // Note that the hierarchical mapper does not work well when the
  // sensor_from_rig poses are inconsistently refined in different clusters,
  // because then the merging does not work well.
  mapper_options.incremental_options.ba_refine_sensor_from_rig = false;
  HierarchicalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction_manager->Get(0),
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-3,
                             /*num_obs_tolerance=*/0);
}

TEST(HierarchicalPipeline, WithoutNoiseAndPanoramicNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalPipelineOptions mapper_options;
  mapper_options.clustering_options.leaf_max_num_images = 10;
  mapper_options.clustering_options.image_overlap = 3;
  // Note that the hierarchical mapper does not work well when the
  // sensor_from_rig poses are inconsistently refined in different clusters,
  // because then the merging does not work well.
  mapper_options.incremental_options.ba_refine_sensor_from_rig = false;
  HierarchicalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction_manager->Get(0),
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-3,
                             /*num_obs_tolerance=*/0);
}

TEST(HierarchicalPipeline, MultiReconstruction) {
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
  HierarchicalPipelineOptions mapper_options;
  mapper_options.clustering_options.leaf_max_num_images = 5;
  mapper_options.clustering_options.image_overlap = 3;
  HierarchicalPipeline mapper(mapper_options, database, reconstruction_manager);
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
  ExpectEqualReconstructions(gt_reconstruction1,
                             *computed_reconstruction1,
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-4,
                             /*num_obs_tolerance=*/0);
  ExpectEqualReconstructions(gt_reconstruction2,
                             *computed_reconstruction2,
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-4,
                             /*num_obs_tolerance=*/0);

  // After the pipeline runs, point3D.error must be in pixel units for every
  // reconstruction in the manager, i.e. equal to what UpdatePoint3DErrors
  // would recompute.
  for (Reconstruction* reconstruction :
       {computed_reconstruction1, computed_reconstruction2}) {
    ASSERT_GT(reconstruction->NumPoints3D(), 0u);
    const double mean_after_run =
        reconstruction->ComputeMeanReprojectionError();
    reconstruction->UpdatePoint3DErrors();
    EXPECT_DOUBLE_EQ(mean_after_run,
                     reconstruction->ComputeMeanReprojectionError());
  }
}

}  // namespace
}  // namespace colmap
