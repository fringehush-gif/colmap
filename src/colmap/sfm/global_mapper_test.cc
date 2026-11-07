// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sfm/global_mapper.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// TODO(jsch): Add tests for pose priors.

std::shared_ptr<DatabaseCache> CreateDatabaseCache(const Database& database) {
  DatabaseCache::Options options;
  return DatabaseCache::Create(database, options);
}

TEST(GlobalMapper, WithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialKnownRig) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialUnknownRig) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise

  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  // Set the rig sensors to be unknown
  for (const auto& [rig_id, rig] : reconstruction->Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction->Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithNoiseAndOutliers) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.7;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  const auto database_cache = CreateDatabaseCache(*database);
  for (const bool use_covariance : {false, true}) {
    SCOPED_TRACE(use_covariance);
    auto reconstruction = std::make_shared<Reconstruction>();
    GlobalMapper global_mapper(database_cache);
    global_mapper.BeginReconstruction(reconstruction);

    GlobalMapperOptions options;
    if (use_covariance) {
      options.rotation_averaging.backend = RotationAveragingBackend::CERES;
      options.rotation_averaging.reweighting =
          RotationAveragingReweighting::COVARIANCE;
    }
    global_mapper.Solve(options);

    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(*reconstruction,
                                   /*max_rotation_error_deg=*/1e-1,
                                   /*max_proj_center_error=*/1e-1,
                                   /*max_scale_error=*/std::nullopt,
                                   /*num_obs_tolerance=*/0.02));
  }
}

TEST(GlobalMapper, WithGivenPoseGraph) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  const auto database_cache = CreateDatabaseCache(*database);

  // The mapper uses the given pose graph instead of loading it from the
  // database: an image without valid edges is not reconstructed.
  PoseGraph pose_graph;
  pose_graph.Load(*database_cache->CorrespondenceGraph());
  constexpr image_t kIsolatedImageId = 1;
  for (const auto& [pair_id, edge] : pose_graph.Edges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (image_id1 == kIsolatedImageId || image_id2 == kIsolatedImageId) {
      pose_graph.SetInvalidEdge(pair_id);
    }
  }
  // Edges to images outside the database cache are dropped.
  constexpr image_t kUnknownImageId = 100;
  pose_graph.AddEdge(2, kUnknownImageId, PoseGraph::Edge(Rigid3d()));

  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper global_mapper(database_cache);
  global_mapper.BeginReconstruction(reconstruction, &pose_graph);
  ASSERT_TRUE(global_mapper.Solve(GlobalMapperOptions()));

  // The isolated image is not registered, and thus removed by the final
  // retriangulation, which tears down the reconstruction.
  EXPECT_FALSE(reconstruction->ExistsImage(kIsolatedImageId));
  EXPECT_EQ(reconstruction->NumRegImages(),
            gt_reconstruction.NumRegImages() - 1);
}

TEST(GlobalMapperOptions, RefineSensorFromRigPropagatesToSubOptions) {
  GlobalMapperOptions options;
  options.refine_sensor_from_rig = false;
  // Sub-options keep their own defaults (true) until accessed.
  EXPECT_TRUE(options.rotation_averaging.refine_sensor_from_rig);
  EXPECT_TRUE(options.global_positioning.refine_sensor_from_rig);
  EXPECT_TRUE(options.bundle_adjustment.refine_sensor_from_rig);
  // Accessors return resolved sub-options with the top-level flag applied.
  EXPECT_FALSE(options.RotationAveraging().refine_sensor_from_rig);
  EXPECT_FALSE(options.GlobalPositioning().refine_sensor_from_rig);
  EXPECT_FALSE(options.BundleAdjustment().refine_sensor_from_rig);
}

// Exercise the real track consumer using the maintained scene/database fixture.
// The explicit matches below define three tracks independently of union-find:
//   {(a,0), (a,1), (b,0), (c,0)}, {(a,2), (b,2), (c,2)}, {(a,3), (b,3)}.
class GlobalTrackEstablishmentTest : public testing::Test {
 protected:
  using Observation = std::pair<image_t, point2D_t>;
  using Observations = std::vector<Observation>;

  void SetUp() override {
    database_ = Database::Open(CreateTestDir() / "database.db");
    SyntheticDatasetOptions synthetic_options;
    synthetic_options.num_rigs = 1;
    synthetic_options.num_cameras_per_rig = 1;
    synthetic_options.num_frames_per_rig = 3;
    synthetic_options.num_points3D = 4;
    synthetic_options.two_view_geometry_has_relative_pose = true;
    SynthesizeDataset(synthetic_options, &ground_truth_, database_.get());
    DatabaseCache::Options cache_options;
    cache_options.load_all_images = true;
    cache_ = DatabaseCache::Create(*database_, cache_options);
    for (const auto& [image_id, image] : cache_->Images()) {
      image_ids_.push_back(image_id);
    }
    std::sort(image_ids_.begin(), image_ids_.end());
    ASSERT_EQ(image_ids_.size(), 3);

    CorrespondenceGraph graph;
    for (const image_t image_id : image_ids_) {
      graph.AddImage(image_id, cache_->Image(image_id).NumPoints2D());
    }
    const auto add_matches = [&](size_t i, size_t j, FeatureMatches matches) {
      TwoViewGeometry geometry;
      geometry.cam2_from_cam1 = Rigid3d();
      geometry.inlier_matches = std::move(matches);
      graph.AddTwoViewGeometry(
          image_ids_[i], image_ids_[j], std::move(geometry));
    };
    add_matches(0, 1, {{0, 0}, {2, 2}, {3, 3}});
    add_matches(1, 2, {{0, 0}, {2, 2}});
    add_matches(0, 2, {{1, 0}, {2, 2}});
    graph.Finalize();
    *cache_->CorrespondenceGraph() = std::move(graph);

    // A wrong cache-coordinate lookup sees distance 0, whereas the
    // reconstruction coordinates below have distance 10. Thus only the latter
    // must reject this track when the threshold is immediately below 10.
    cache_->Image(image_ids_[0]).Point2D(0).xy = Eigen::Vector2d::Zero();
    cache_->Image(image_ids_[0]).Point2D(1).xy = Eigen::Vector2d::Zero();

    BeginReconstruction();
  }

  void BeginReconstruction(const PoseGraph* pose_graph = nullptr) {
    mapper_ = std::make_unique<GlobalMapper>(cache_);
    reconstruction_ = std::make_shared<Reconstruction>();
    mapper_->BeginReconstruction(reconstruction_, pose_graph);
    for (const image_t image_id : image_ids_) {
      const frame_t frame_id = reconstruction_->Image(image_id).FrameId();
      reconstruction_->Frame(frame_id).SetRigFromWorld(
          ground_truth_.Frame(frame_id).RigFromWorld());
      reconstruction_->RegisterFrame(frame_id);
    }
    // Pixel-space distance 10, exactly the default consistency threshold.
    reconstruction_->Image(image_ids_[0]).Point2D(0).xy = Eigen::Vector2d(0, 0);
    reconstruction_->Image(image_ids_[0]).Point2D(1).xy = Eigen::Vector2d(6, 8);
  }

  Observations LongTrack() const {
    return {{image_ids_[0], 0},
            {image_ids_[0], 1},
            {image_ids_[1], 0},
            {image_ids_[2], 0}};
  }
  Observations ThreeViewTrack() const {
    return {{image_ids_[0], 2}, {image_ids_[1], 2}, {image_ids_[2], 2}};
  }
  Observations TwoViewTrack() const {
    return {{image_ids_[0], 3}, {image_ids_[1], 3}};
  }

  void ExpectTracks(std::vector<Observations> expected) {
    std::vector<Observations> actual;
    std::map<point3D_t, Observations> ordered_tracks;
    for (const auto& [point3D_id, point3D] : reconstruction_->Points3D()) {
      Observations observations;
      for (const auto& element : point3D.track.Elements()) {
        observations.emplace_back(element.image_id, element.point2D_idx);
        EXPECT_EQ(reconstruction_->Image(element.image_id)
                      .Point2D(element.point2D_idx)
                      .point3D_id,
                  point3D_id);
      }
      ordered_tracks.emplace(point3D_id, observations);
      std::sort(observations.begin(), observations.end());
      actual.push_back(std::move(observations));
    }
    for (auto& observations : expected) {
      std::sort(observations.begin(), observations.end());
    }
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(actual, expected);

    // Preserve point IDs and the unsorted per-track observation order for the
    // paired baseline/candidate XML comparison; the set oracle above is
    // primary.
    std::ostringstream roster;
    for (const auto& [point3D_id, observations] : ordered_tracks) {
      roster << point3D_id << ':';
      for (const auto& [image_id, point2D_idx] : observations) {
        roster << image_id << ',' << point2D_idx << ';';
      }
      roster << '|';
    }
    RecordProperty("ordered_track_roster", roster.str());
  }

  std::shared_ptr<Database> database_;
  Reconstruction ground_truth_;
  std::shared_ptr<DatabaseCache> cache_;
  std::unique_ptr<GlobalMapper> mapper_;
  std::shared_ptr<Reconstruction> reconstruction_;
  std::vector<image_t> image_ids_;
};

TEST_F(GlobalTrackEstablishmentTest, SuppliedGraphOutlivesItsSource) {
  {
    PoseGraph pose_graph;
    pose_graph.Load(*cache_->CorrespondenceGraph());
    BeginReconstruction(&pose_graph);
  }
  mapper_->EstablishTracks(GlobalMapperOptions());
  ExpectTracks({LongTrack(), ThreeViewTrack()});
}

TEST_F(GlobalTrackEstablishmentTest, SuppliedGraphPreservesInvalidEdges) {
  PoseGraph pose_graph;
  pose_graph.Load(*cache_->CorrespondenceGraph());
  const image_pair_t excluded_pair =
      ImagePairToPairId(image_ids_[0], image_ids_[2]);
  pose_graph.SetInvalidEdge(excluded_pair);
  BeginReconstruction(&pose_graph);
  mapper_->EstablishTracks(GlobalMapperOptions());
  ExpectTracks({{{image_ids_[0], 0}, {image_ids_[1], 0}, {image_ids_[2], 0}},
                ThreeViewTrack()});
  EXPECT_FALSE(pose_graph.IsValid(excluded_pair));
  EXPECT_EQ(pose_graph.NumEdges(), 3);
}

TEST_F(GlobalTrackEstablishmentTest, PixelThresholdAndSourceCoordinates) {
  std::map<image_t, std::vector<Eigen::Vector2d>> source_xy;
  for (const auto& [image_id, image] : reconstruction_->Images()) {
    for (const auto& point2D : image.Points2D()) {
      source_xy[image_id].push_back(point2D.xy);
    }
  }
  const auto cache_images = cache_->Images();
  mapper_->EstablishTracks(GlobalMapperOptions());
  ExpectTracks({LongTrack(), ThreeViewTrack()});
  for (const auto& [image_id, image] : reconstruction_->Images()) {
    for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
      EXPECT_EQ(image.Point2D(idx).xy.x(), source_xy.at(image_id).at(idx).x());
      EXPECT_EQ(image.Point2D(idx).xy.y(), source_xy.at(image_id).at(idx).y());
    }
    EXPECT_EQ(cache_->Image(image_id), cache_images.at(image_id));
  }
}

TEST_F(GlobalTrackEstablishmentTest, JustBelowPixelThresholdRejectsWholeTrack) {
  GlobalMapperOptions options;
  options.track_intra_image_consistency_threshold = std::nextafter(10., 0.);
  mapper_->EstablishTracks(options);
  ExpectTracks({ThreeViewTrack()});
}

TEST_F(GlobalTrackEstablishmentTest, ViewCountFilter) {
  GlobalMapperOptions options;
  options.track_min_num_views_per_track = 2;
  options.track_max_num_views_per_track = 2;
  mapper_->EstablishTracks(options);
  ExpectTracks({TwoViewTrack()});
}

TEST_F(GlobalTrackEstablishmentTest, LongestTrackBudget) {
  GlobalMapperOptions options;
  options.keep_max_num_tracks = 1;
  mapper_->EstablishTracks(options);
  ExpectTracks({LongTrack()});
}

TEST_F(GlobalTrackEstablishmentTest, ZeroTrackBudget) {
  GlobalMapperOptions options;
  options.keep_max_num_tracks = 0;
  mapper_->EstablishTracks(options);
  ExpectTracks({});
}

TEST_F(GlobalTrackEstablishmentTest, PerViewEarlyStopCountsImagesOnce) {
  GlobalMapperOptions options;
  options.track_required_tracks_per_view = 0;
  mapper_->EstablishTracks(options);
  ExpectTracks({LongTrack()});
}

TEST_F(GlobalTrackEstablishmentTest, PosedButUnregisteredImageRejected) {
  const frame_t frame_id = reconstruction_->Image(image_ids_[2]).FrameId();
  reconstruction_->DeRegisterFrame(frame_id);
  reconstruction_->Frame(frame_id).SetRigFromWorld(
      ground_truth_.Frame(frame_id).RigFromWorld());
  ASSERT_TRUE(reconstruction_->Image(image_ids_[2]).HasPose());
  ASSERT_EQ(reconstruction_->NumRegImages(), 2);
  EXPECT_ANY_THROW(mapper_->EstablishTracks(GlobalMapperOptions()));
  EXPECT_EQ(reconstruction_->NumPoints3D(), 0);
}

TEST_F(GlobalTrackEstablishmentTest, FeatureBoundsCheckedBeforePublication) {
  reconstruction_->Image(image_ids_[0])
      .SetPoints2D(std::vector<Eigen::Vector2d>(1, Eigen::Vector2d::Zero()));
  EXPECT_THROW(mapper_->EstablishTracks(GlobalMapperOptions()),
               std::out_of_range);
  EXPECT_EQ(reconstruction_->NumPoints3D(), 0);
}

}  // namespace
}  // namespace colmap
