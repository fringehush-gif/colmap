// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/workspace.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

struct FiveViewWorkspace {
  std::filesystem::path root = CreateTestDir();
  std::vector<std::string> names;

  FiveViewWorkspace() {
    CreateDirIfNotExists(root / "sparse");
    CreateDirIfNotExists(root / "images");
    CreateDirIfNotExists(root / "stereo");
    CreateDirIfNotExists(root / "stereo" / "depth_maps");
    CreateDirIfNotExists(root / "stereo" / "normal_maps");
    SyntheticDatasetOptions options;
    options.num_rigs = 1;
    options.num_cameras_per_rig = 1;
    options.num_frames_per_rig = 5;
    options.camera_width = 10;
    options.camera_height = 5;
    Reconstruction reconstruction;
    SynthesizeDataset(options, &reconstruction);
    reconstruction.Write(root / "sparse");
    for (int image_id = 1; image_id <= 5; ++image_id) {
      const std::string name = reconstruction.Image(image_id).Name();
      names.push_back(name);
      DepthMap depth(10, 5, 1.0f, 10.0f);
      depth.Fill(static_cast<float>(image_id + 1));
      depth.Set(0, 0, std::nextafter(static_cast<float>(image_id + 1), 9.0f));
      depth.Write(root / "stereo" / "depth_maps" / (name + ".geometric.bin"));
      NormalMap normal(10, 5);
      normal.Fill(1.0f);
      normal.Write(root / "stereo" / "normal_maps" / (name + ".geometric.bin"));
      Bitmap bitmap(10, 5, true);
      bitmap.Fill(BitmapColor<uint8_t>(image_id, image_id, image_id));
      bitmap.Write(root / "images" / name);
    }
  }

  Workspace::Options Options(const size_t cache_bytes) const {
    Workspace::Options options;
    options.workspace_path = root;
    options.workspace_format = "COLMAP";
    options.input_type = "geometric";
    options.cache_size =
        static_cast<double>(cache_bytes) / (1024.0 * 1024.0 * 1024.0);
    return options;
  }
};

struct PreparedLikeProblem {
  PatchMatch::Problem problem;
  std::vector<DepthMap> owned_depths;
  std::vector<std::shared_ptr<const DepthMap>> source_leases;

  PreparedLikeProblem(const size_t count, const int ref)
      : owned_depths(count), source_leases(count) {
    problem.ref_image_idx = ref;
    problem.depth_maps = &owned_depths;
    problem.source_depth_leases = &source_leases;
  }
};

TEST(PatchMatchSourceBitmapLease,
     AccessorConsumesLeaseAndKeepsReferencePrivate) {
  const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float translation[3] = {0, 0, 0};
  std::vector<Image> images;
  Bitmap original(10, 5, false);
  original.Fill(BitmapColor<uint8_t>(1, 1, 1));
  for (int i = 0; i < 2; ++i) {
    images.emplace_back("", 10, 5, identity, identity, translation);
    images.back().SetBitmap(original);
  }
  auto source = std::make_shared<Bitmap>(10, 5, false);
  source->Fill(BitmapColor<uint8_t>(7, 7, 7));
  std::vector<std::shared_ptr<const Bitmap>> leases(2, source);
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.images = &images;
  problem.source_bitmap_leases = &leases;
  EXPECT_EQ(&problem.InputBitmap(0), &images[0].GetBitmap());
  EXPECT_EQ(&problem.InputBitmap(1), source.get());
  EXPECT_NE(problem.InputBitmap(1).RowMajorData(),
            images[1].GetBitmap().RowMajorData());
  leases[1].reset();
  EXPECT_EQ(&problem.InputBitmap(1), &images[1].GetBitmap());
  leases[1] = std::shared_ptr<const Bitmap>(std::shared_ptr<const Bitmap>{},
                                            source.get());
  EXPECT_EQ(&problem.InputBitmap(1), &images[1].GetBitmap());
  std::weak_ptr<const Bitmap> weak = source;
  leases[0].reset();
  source.reset();
  ASSERT_TRUE(weak.expired());
  EXPECT_EQ(problem.InputBitmap(1).RowMajorData(), original.RowMajorData());
  leases[1] = std::make_shared<Bitmap>(9, 5, false);
  EXPECT_EQ(&problem.InputBitmap(1), &images[1].GetBitmap());
  leases[1] = std::make_shared<Bitmap>(10, 5, true);
  EXPECT_EQ(&problem.InputBitmap(1), &images[1].GetBitmap());
}

TEST(PatchMatchSourceDepthLease, NonOwningExpiredLeaseUsesRetainedCopy) {
  PreparedLikeProblem prepared(2, 0);
  prepared.owned_depths[1] = DepthMap(10, 5, 1.0f, 10.0f);
  prepared.owned_depths[1].Fill(2.0f);
  auto owner = std::make_shared<DepthMap>(10, 5, 1.0f, 10.0f);
  owner->Fill(7.0f);
  prepared.source_leases[1] = std::shared_ptr<const DepthMap>(
      std::shared_ptr<const DepthMap>{}, owner.get());
  std::weak_ptr<const DepthMap> weak = owner;
  owner.reset();
  ASSERT_TRUE(weak.expired());
  EXPECT_EQ(prepared.problem.InputDepthMap(1).GetPtr(),
            prepared.owned_depths[1].GetPtr());
}

TEST(PatchMatchSourceDepthLease, FivePreparedReferencesBorrowOneSource) {
  const FiveViewWorkspace fixture;
  CachedWorkspace workspace(fixture.Options(16 * 1024));
  workspace.Load(fixture.names);
  const auto& model = workspace.GetModel();
  ASSERT_EQ(model.images.size(), 5);
  std::vector<std::unique_ptr<PreparedLikeProblem>> prepared;
  for (size_t ref_position = 0; ref_position < fixture.names.size();
       ++ref_position) {
    const int ref = model.GetImageIdx(fixture.names[ref_position]);
    auto holder =
        std::make_unique<PreparedLikeProblem>(model.images.size(), ref);
    for (const auto& name : fixture.names) {
      const int id = model.GetImageIdx(name);
      const auto lease = workspace.GetDepthMapLease(id);
      if (id == ref) {
        holder->owned_depths.at(id) = *lease;
      } else {
        holder->source_leases.at(id) = lease;
        ASSERT_EQ(holder->problem.InputDepthMap(id).GetPtr(), lease->GetPtr());
      }
    }
    prepared.push_back(std::move(holder));
  }

  // For either a 3-view prefix or the full 5-view batch, the same source
  // retains one immutable workspace owner across different prepared problems.
  const int shared_id = model.GetImageIdx(fixture.names[4]);
  const float* shared = prepared[0]->problem.InputDepthMap(shared_id).GetPtr();
  for (size_t ref_position = 1; ref_position < 4; ++ref_position) {
    const auto& other =
        prepared[ref_position]->problem.InputDepthMap(shared_id);
    EXPECT_EQ(other.GetPtr(), shared);
    EXPECT_EQ(other.GetNumBytes(),
              prepared[0]->problem.InputDepthMap(shared_id).GetNumBytes());
    EXPECT_EQ(std::memcmp(other.GetPtr(), shared, other.GetNumBytes()), 0);
  }

  // The reference seed remains an independently mutable value copy even if
  // somebody accidentally supplies a lease at that index.
  prepared[4]->source_leases.at(shared_id) =
      workspace.GetDepthMapLease(shared_id);
  const float original =
      prepared[0]->problem.InputDepthMap(shared_id).Get(0, 0);
  EXPECT_NE(prepared[4]->problem.InputDepthMap(shared_id).GetPtr(), shared);
  prepared[4]->owned_depths.at(shared_id).Set(0, 0, 9.0f);
  EXPECT_FLOAT_EQ(prepared[4]->problem.InputDepthMap(shared_id).Get(0, 0),
                  9.0f);
  EXPECT_FLOAT_EQ(prepared[0]->problem.InputDepthMap(shared_id).Get(0, 0),
                  original);

  // A missing source lease selects the original owned-map path, including a
  // replacement of different dimensions. Check() remains responsible for
  // rejecting geometry that disagrees with the corresponding image.
  prepared[0]->source_leases.at(shared_id).reset();
  prepared[0]->owned_depths.at(shared_id) = DepthMap(8, 4, 1.0f, 10.0f);
  EXPECT_EQ(prepared[0]->problem.InputDepthMap(shared_id).GetWidth(), 8);
  EXPECT_EQ(prepared[0]->problem.InputDepthMap(shared_id).GetHeight(), 4);
  EXPECT_THROW(prepared[0]->problem.InputDepthMap(99), std::out_of_range);
  EXPECT_THROW(prepared[0]->problem.InputDepthMap(-1), std::out_of_range);
  EXPECT_EQ(prepared[1]->problem.InputDepthMap(shared_id).GetPtr(), shared);
}

TEST(PatchMatchSourceDepthLease, EvictedCacheEntryCannotInvalidateLease) {
  const FiveViewWorkspace fixture;
  CachedWorkspace workspace(fixture.Options(300));
  workspace.Load(fixture.names);
  const auto& model = workspace.GetModel();
  const int source = model.GetImageIdx(fixture.names[0]);
  const int other = model.GetImageIdx(fixture.names[1]);
  PreparedLikeProblem prepared(model.images.size(), other);
  prepared.source_leases.at(source) = workspace.GetDepthMapLease(source);
  const auto held = prepared.source_leases.at(source);
  const auto* old_ptr = held.get();
  const float old_value = held->Get(0, 0);
  (void)workspace.GetDepthMapLease(other);  // Exceeds one-map cache capacity.
  const auto reloaded = workspace.GetDepthMapLease(source);
  ASSERT_NE(reloaded.get(), old_ptr);
  EXPECT_EQ(prepared.problem.InputDepthMap(source).GetPtr(), held->GetPtr());
  EXPECT_FLOAT_EQ(prepared.problem.InputDepthMap(source).Get(0, 0), old_value);
  EXPECT_EQ(
      std::memcmp(reloaded->GetPtr(), held->GetPtr(), held->GetNumBytes()), 0);
  // The reload is safe but uses another allocation; the lease is a lifetime
  // guarantee, not an unconditional global deduplication guarantee.
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
