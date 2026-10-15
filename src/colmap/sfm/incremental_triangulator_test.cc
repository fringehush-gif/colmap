// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sfm/incremental_triangulator.h"

#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/synthetic.h"

#include <limits>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void DeleteAllPoints3D(Reconstruction& reconstruction) {
  std::vector<point3D_t> point3D_ids_to_delete;
  for (const auto point3D_id : reconstruction.Point3DIds()) {
    point3D_ids_to_delete.push_back(point3D_id);
  }
  for (const auto point3D_id : point3D_ids_to_delete) {
    reconstruction.DeletePoint3D(point3D_id);
  }
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

void DeleteOneObservationFromEachTrack(Reconstruction& reconstruction) {
  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    ASSERT_GT(point3D.track.Length(), 0);
    reconstruction.DeleteObservation(point3D.track.Element(0).image_id,
                                     point3D.track.Element(0).point2D_idx);
  }
  EXPECT_EQ(reconstruction.ComputeNumObservations(),
            reconstruction.NumPoints3D() * (reconstruction.NumRegImages() - 1));
}

void SplitPoint3D(Reconstruction& reconstruction, point3D_t point3D_id) {
  auto& point3D = reconstruction.Point3D(point3D_id);
  ASSERT_GE(point3D.track.Length(), 4);
  Track split_track;
  for (size_t i = point3D.track.Length() / 2; i < point3D.track.Length(); ++i) {
    const auto& track_el = point3D.track.Element(i);
    split_track.AddElement(track_el);
    reconstruction.Image(track_el.image_id)
        .ResetPoint3DForPoint2D(track_el.point2D_idx);
  }
  point3D.track.Elements().resize(point3D.track.Length() / 2);
  const point3D_t new_point3D_id =
      reconstruction.AddPoint3D(point3D.xyz, split_track);
  for (const auto& track_el : split_track.Elements()) {
    reconstruction.Image(track_el.image_id)
        .SetPoint3DForPoint2D(track_el.point2D_idx, new_point3D_id);
  }
}

TEST(IncrementalTriangulator, Print) {
  Reconstruction reconstruction;
  IncrementalTriangulator triangulator(std::make_shared<CorrespondenceGraph>(),
                                       reconstruction);
  std::ostringstream stream;
  stream << triangulator;
  EXPECT_EQ(
      stream.str(),
      "IncrementalTriangulator(reconstruction=Reconstruction(num_rigs=0, "
      "num_cameras=0, num_frames=0, num_reg_frames=0, num_images=0, "
      "num_points3D=0), correspondence_graph=CorrespondenceGraph(num_images=0, "
      "num_image_pairs=0))");
}

TEST(IncrementalTriangulator, ModifiedPoints3D) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 50;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  EXPECT_THAT(triangulator.GetModifiedPoints3D(), testing::IsEmpty());

  auto points3D_it = reconstruction.Points3D().begin();
  const point3D_t point3D_id1 = (points3D_it++)->first;
  const point3D_t point3D_id2 = (points3D_it++)->first;

  triangulator.AddModifiedPoint3D(point3D_id1);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().size(), 1);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().count(point3D_id1), 1);

  triangulator.AddModifiedPoint3D(point3D_id2);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().size(), 2);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().count(point3D_id2), 1);

  triangulator.AddModifiedPoint3D(point3D_id1);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().size(), 2);

  triangulator.ClearModifiedPoints3D();
  EXPECT_TRUE(triangulator.GetModifiedPoints3D().empty());
}

TEST(IncrementalTriangulator, ModifiedPoints3DRemovesNonExistent) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 10;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  auto point3D_ids = reconstruction.Point3DIds();
  ASSERT_GE(point3D_ids.size(), 1);
  const point3D_t point3D_id = *point3D_ids.begin();

  triangulator.AddModifiedPoint3D(point3D_id);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().size(), 1);
  reconstruction.DeletePoint3D(point3D_id);
  EXPECT_TRUE(triangulator.GetModifiedPoints3D().empty());
}

TEST(IncrementalTriangulator, TriangulateImage) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  DeleteAllPoints3D(reconstruction);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);
  size_t total_tris = 0;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    total_tris += triangulator.TriangulateImage(
        IncrementalTriangulator::Options(), image_id);
  }

  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  EXPECT_EQ(total_tris,
            synthetic_options.num_points3D * reconstruction.NumRegImages());
}

TEST(IncrementalTriangulator, CompleteImage) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  DeleteOneObservationFromEachTrack(reconstruction);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  triangulator.CompleteImage(IncrementalTriangulator::Options(),
                             reconstruction.RegImageIds().at(0));
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  EXPECT_EQ(
      reconstruction.ComputeNumObservations(),
      synthetic_options.num_points3D * (reconstruction.NumRegImages() - 1));
}

TEST(IncrementalTriangulator, CompleteTracks) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  DeleteOneObservationFromEachTrack(reconstruction);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_completions =
      triangulator.CompleteTracks(IncrementalTriangulator::Options(),
                                  {reconstruction.Points3D().begin()->first});
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  EXPECT_EQ(num_completions, 1);
  EXPECT_EQ(reconstruction.ComputeNumObservations(),
            synthetic_options.num_points3D * reconstruction.NumRegImages() -
                synthetic_options.num_points3D + 1);
}

TEST(IncrementalTriangulator, CompleteAllTracks) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  DeleteOneObservationFromEachTrack(reconstruction);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_completions =
      triangulator.CompleteAllTracks(IncrementalTriangulator::Options());
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  EXPECT_EQ(num_completions, synthetic_options.num_points3D);
  EXPECT_EQ(reconstruction.ComputeNumObservations(),
            synthetic_options.num_points3D * reconstruction.NumRegImages());
}

TEST(IncrementalTriangulator, MergeTracks) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 10;
  synthetic_options.num_points3D = 5;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  auto points3D_it = reconstruction.Points3D().begin();
  const point3D_t point3D_id1 = (points3D_it++)->first;
  const point3D_t point3D_id2 = (points3D_it++)->first;

  SplitPoint3D(reconstruction, point3D_id1);
  SplitPoint3D(reconstruction, point3D_id2);

  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D + 2);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_merged = triangulator.MergeTracks(
      IncrementalTriangulator::Options(), {point3D_id1});
  EXPECT_EQ(num_merged, reconstruction.NumRegImages());
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D + 1);
}

TEST(IncrementalTriangulator, MergeAllTracks) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 10;
  synthetic_options.num_points3D = 5;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  auto points3D_it = reconstruction.Points3D().begin();
  const point3D_t point3D_id1 = (points3D_it++)->first;
  const point3D_t point3D_id2 = (points3D_it++)->first;

  SplitPoint3D(reconstruction, point3D_id1);
  SplitPoint3D(reconstruction, point3D_id2);

  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D + 2);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_merged =
      triangulator.MergeAllTracks(IncrementalTriangulator::Options());
  EXPECT_EQ(num_merged, 2 * reconstruction.NumRegImages());
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
}

TEST(IncrementalTriangulator, Retriangulate) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  DeleteAllPoints3D(reconstruction);

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_tris =
      triangulator.Retriangulate(IncrementalTriangulator::Options());
  EXPECT_EQ(num_tris,
            synthetic_options.num_points3D * reconstruction.NumRegImages());
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
}

struct FiniteTrackFixture {
  std::shared_ptr<Database> database =
      Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction reconstruction;
  std::shared_ptr<DatabaseCache> cache;
  FiniteTrackFixture() {
    SetPRNGSeed(0);
    SyntheticDatasetOptions options;
    options.num_rigs = 1;
    options.num_cameras_per_rig = 1;
    options.num_frames_per_rig = 5;
    options.num_points3D = 5;
    SynthesizeDataset(options, &reconstruction, database.get());
    cache = DatabaseCache::Create(*database, DatabaseCache::Options());
  }
  ~FiniteTrackFixture() { database->Close(); }
};

class NonfiniteTrackCoordinate
    : public testing::TestWithParam<std::tuple<int, double>> {};

TEST_P(NonfiniteTrackCoordinate, CompletionRejectsInvalidObservation) {
  FiniteTrackFixture fixture;
  auto& reconstruction = fixture.reconstruction;
  const auto id = reconstruction.Points3D().begin()->first;
  const auto row = reconstruction.Point3D(id).track.Element(0);
  reconstruction.DeleteObservation(row.image_id, row.point2D_idx);
  reconstruction.Image(row.image_id)
      .Point2D(row.point2D_idx)
      .xy[std::get<0>(GetParam())] = std::get<1>(GetParam());
  IncrementalTriangulator triangulator(fixture.cache->CorrespondenceGraph(),
                                       reconstruction);
  EXPECT_EQ(
      triangulator.CompleteTracks(IncrementalTriangulator::Options(), {id}), 0);
  EXPECT_FALSE(
      reconstruction.Image(row.image_id).Point2D(row.point2D_idx).HasPoint3D());
  EXPECT_EQ(reconstruction.NumPoints3D(), 5);
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 24);
}

TEST_P(NonfiniteTrackCoordinate, MergeRejectsInvalidObservation) {
  FiniteTrackFixture fixture;
  auto& reconstruction = fixture.reconstruction;
  const auto id = reconstruction.Points3D().begin()->first;
  const auto row = reconstruction.Point3D(id).track.Element(0);
  SplitPoint3D(reconstruction, id);
  reconstruction.Image(row.image_id)
      .Point2D(row.point2D_idx)
      .xy[std::get<0>(GetParam())] = std::get<1>(GetParam());
  IncrementalTriangulator triangulator(fixture.cache->CorrespondenceGraph(),
                                       reconstruction);
  EXPECT_EQ(triangulator.MergeTracks(IncrementalTriangulator::Options(), {id}),
            0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 6);
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 25);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidCoordinates,
    NonfiniteTrackCoordinate,
    testing::Combine(
        testing::Values(0, 1),
        testing::Values(std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity())));

class NonfiniteTrackPosition : public testing::TestWithParam<double> {};

TEST_P(NonfiniteTrackPosition, MergeRejectsInvalidPosition) {
  FiniteTrackFixture fixture;
  auto& reconstruction = fixture.reconstruction;
  const auto id = reconstruction.Points3D().begin()->first;
  SplitPoint3D(reconstruction, id);
  reconstruction.Point3D(id).xyz.x() = GetParam();
  IncrementalTriangulator triangulator(fixture.cache->CorrespondenceGraph(),
                                       reconstruction);
  EXPECT_EQ(triangulator.MergeTracks(IncrementalTriangulator::Options(), {id}),
            0);
  EXPECT_EQ(reconstruction.NumPoints3D(), 6);
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 25);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidPositions,
    NonfiniteTrackPosition,
    testing::Values(std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity()));

TEST(FiniteTrackBoundary, CompletionAcceptsExactThreshold) {
  FiniteTrackFixture fixture;
  auto& reconstruction = fixture.reconstruction;
  const auto id = reconstruction.Points3D().begin()->first;
  const auto row = reconstruction.Point3D(id).track.Element(0);
  reconstruction.DeleteObservation(row.image_id, row.point2D_idx);
  auto& image = reconstruction.Image(row.image_id);
  const auto projection = image.CameraPtr()->ImgFromCam(
      image.CamFromWorld() * reconstruction.Point3D(id).xyz);
  ASSERT_TRUE(projection.has_value());
  image.Point2D(row.point2D_idx).xy = *projection + Eigen::Vector2d(4.0, 0.0);
  IncrementalTriangulator::Options options;
  options.complete_max_reproj_error = 4.0;
  IncrementalTriangulator triangulator(fixture.cache->CorrespondenceGraph(),
                                       reconstruction);
  EXPECT_EQ(triangulator.CompleteTracks(options, {id}), 1);
  EXPECT_TRUE(image.Point2D(row.point2D_idx).HasPoint3D());
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 25);
}

TEST(FiniteTrackBoundary, MergeAcceptsExactThreshold) {
  FiniteTrackFixture fixture;
  auto& reconstruction = fixture.reconstruction;
  const auto id = reconstruction.Points3D().begin()->first;
  const auto row = reconstruction.Point3D(id).track.Element(0);
  const auto split_row = reconstruction.Point3D(id).track.Element(2);
  SplitPoint3D(reconstruction, id);
  const auto other_id = reconstruction.Image(split_row.image_id)
                            .Point2D(split_row.point2D_idx)
                            .point3D_id;
  const auto& first = reconstruction.Point3D(id);
  const auto& second = reconstruction.Point3D(other_id);
  const Eigen::Vector3d merged_xyz =
      (first.track.Length() * first.xyz + second.track.Length() * second.xyz) /
      (first.track.Length() + second.track.Length());
  auto& image = reconstruction.Image(row.image_id);
  const auto projection =
      image.CameraPtr()->ImgFromCam(image.CamFromWorld() * merged_xyz);
  ASSERT_TRUE(projection.has_value());
  image.Point2D(row.point2D_idx).xy = *projection + Eigen::Vector2d(4.0, 0.0);
  IncrementalTriangulator::Options options;
  options.merge_max_reproj_error = 4.0;
  IncrementalTriangulator triangulator(fixture.cache->CorrespondenceGraph(),
                                       reconstruction);
  EXPECT_EQ(triangulator.MergeTracks(options, {id}), 5);
  EXPECT_EQ(reconstruction.NumPoints3D(), 5);
  EXPECT_EQ(reconstruction.ComputeNumObservations(), 25);
}

}  // namespace
}  // namespace colmap
