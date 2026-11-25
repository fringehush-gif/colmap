// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_input.h"

#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/workspace.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"
#include "colmap/util/threading.h"

#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

enum class Payload { Bitmap, Depth, Normal };

class InputWorkspace : public Workspace {
 public:
  explicit InputWorkspace(const Options& options) : Workspace(options) {
    for (int index = 0; index < 2; ++index) {
      bitmaps[index] = std::make_shared<Bitmap>(10, 5, true);
      bitmaps[index]->Fill(BitmapColor<uint8_t>(index + 1, 17, 93));
      depths[index] = std::make_shared<DepthMap>(10, 5, 1, 10);
      depths[index]->Fill(index + 2.0f);
      normals[index] = std::make_shared<NormalMap>(10, 5);
      normals[index]->Fill(index + 0.25f);
    }
  }

  // Exercise the default lease methods' dispatch to legacy virtual getters.
  const Bitmap& GetBitmap(const int index) override {
    ++reads.at(index)[0];
    if (before_read) before_read(index, Payload::Bitmap);
    return *bitmaps.at(index);
  }
  const DepthMap& GetDepthMap(const int index) override {
    ++reads.at(index)[1];
    if (before_read) before_read(index, Payload::Depth);
    return *depths.at(index);
  }
  std::shared_ptr<const DepthMap> GetDepthMapLease(const int index) override {
    if (!owning_depth) return Workspace::GetDepthMapLease(index);
    GetDepthMap(index);
    return depths.at(index);
  }
  const NormalMap& GetNormalMap(const int index) override {
    ++reads.at(index)[2];
    if (before_read) before_read(index, Payload::Normal);
    return *normals.at(index);
  }

  bool owning_depth = false;
  std::array<std::shared_ptr<Bitmap>, 2> bitmaps;
  std::array<std::shared_ptr<DepthMap>, 2> depths;
  std::array<std::shared_ptr<NormalMap>, 2> normals;
  std::array<std::array<int, 3>, 2> reads{};
  std::function<void(int, Payload)> before_read;
};

class PatchMatchInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto path = CreateTestDir();
    CreateDirIfNotExists(path / "sparse");
    SyntheticDatasetOptions synthetic;
    synthetic.num_rigs = 1;
    synthetic.num_cameras_per_rig = 1;
    synthetic.num_frames_per_rig = 2;
    synthetic.num_points3D = 4;
    synthetic.camera_width = 10;
    synthetic.camera_height = 5;
    Reconstruction reconstruction;
    SynthesizeDataset(synthetic, &reconstruction);
    reconstruction.Write(path / "sparse");
    Workspace::Options options;
    options.workspace_path = path;
    options.workspace_format = "COLMAP";
    workspace = std::make_unique<InputWorkspace>(options);
    ASSERT_EQ(workspace->GetModel().images.size(), 2);
  }

  std::unique_ptr<InputWorkspace> workspace;
};

TEST_F(PatchMatchInputTest, CopiesEveryRequestedPayloadAndReusesPool) {
  ThreadPool pool(2);
  for (const bool geometry : {false, true}) {
    SCOPED_TRACE(geometry);
    workspace->reads = {};
    auto images = workspace->GetModel().images;
    std::vector<DepthMap> depths(geometry ? images.size() : 0);
    std::vector<std::shared_ptr<const DepthMap>> leases(depths.size());
    std::vector<NormalMap> normals(geometry ? images.size() : 0);
    internal::StagePatchMatchInputs(
        *workspace, pool, {1, 0}, geometry, 0, images, depths, leases, normals);
    for (int index = 0; index < 2; ++index) {
      EXPECT_EQ(workspace->reads[index],
                (std::array<int, 3>{1, geometry ? 1 : 0, geometry ? 1 : 0}));
      EXPECT_EQ(images[index].GetBitmap().RowMajorData(),
                workspace->bitmaps[index]->RowMajorData());
      if (geometry) {
        EXPECT_EQ(depths[index].GetData(), workspace->depths[index]->GetData());
        EXPECT_EQ(normals[index].GetData(),
                  workspace->normals[index]->GetData());
      }
    }
    const auto reads = workspace->reads;
    internal::StagePatchMatchInputs(
        *workspace, pool, {}, geometry, 0, images, depths, leases, normals);
    EXPECT_EQ(workspace->reads, reads);
    EXPECT_NO_THROW(pool.Wait());
  }
}

TEST_F(PatchMatchInputTest, SharesOnlyOwningMatchingSourceDepths) {
  for (const int num_workers : {1, 2, 4}) {
    for (const bool owning : {false, true}) {
      SCOPED_TRACE(num_workers);
      SCOPED_TRACE(owning);
      workspace->owning_depth = owning;
      ThreadPool pool(num_workers);
      auto images = workspace->GetModel().images;
      std::vector<DepthMap> depths(images.size());
      std::vector<NormalMap> normals(images.size());
      std::vector<std::shared_ptr<const DepthMap>> leases(images.size());
      PatchMatch::Problem problem;
      problem.ref_image_idx = 0;
      problem.depth_maps = &depths;
      problem.source_depth_leases = &leases;

      workspace->depths[1] = std::make_shared<DepthMap>(10, 5, 1, 10);
      workspace->depths[1]->Fill(std::nextafter(2.0f, 3.0f));
      internal::StagePatchMatchInputs(
          *workspace, pool, {1, 0}, true, 0, images, depths, leases, normals);
      EXPECT_EQ(leases[0], nullptr);
      EXPECT_NE(depths[0].GetPtr(), workspace->depths[0]->GetPtr());
      const auto& source = problem.InputDepthMap(1);
      ASSERT_EQ(source.GetNumBytes(), workspace->depths[1]->GetNumBytes());
      EXPECT_EQ(std::memcmp(source.GetPtr(),
                            workspace->depths[1]->GetPtr(),
                            source.GetNumBytes()),
                0);
      if (owning) {
        EXPECT_EQ(leases[1].get(), workspace->depths[1].get());
        EXPECT_EQ(depths[1].GetNumBytes(), 0);
      } else {
        EXPECT_EQ(leases[1], nullptr);
        EXPECT_NE(source.GetPtr(), workspace->depths[1]->GetPtr());
      }
      depths[0].Fill(9.0f);
      EXPECT_FLOAT_EQ(workspace->depths[0]->Get(0, 0), 2.0f);

      // Reusing destinations must discard an old source lease on fallback.
      workspace->depths[1] = std::make_shared<DepthMap>(4, 3, 1, 10);
      workspace->depths[1]->Fill(7.0f);
      internal::StagePatchMatchInputs(
          *workspace, pool, {1}, true, 0, images, depths, leases, normals);
      EXPECT_EQ(leases[1], nullptr);
      EXPECT_EQ(problem.InputDepthMap(1).GetWidth(), 4);
      EXPECT_EQ(problem.InputDepthMap(1).GetHeight(), 3);
      EXPECT_NE(depths[1].GetPtr(), workspace->depths[1]->GetPtr());
      EXPECT_FLOAT_EQ(depths[1].Get(0, 0), 7.0f);
      EXPECT_NO_THROW(pool.Wait());
    }
  }
}

TEST(PatchMatchSourceDepth, ReferenceAndInvalidLeasesUseRetainedCopies) {
  std::vector<DepthMap> depths(2, DepthMap(2, 3, 1, 10));
  depths[0].Fill(2.0f);
  depths[1].Fill(3.0f);
  // Existing aggregate callers can omit the new optional lease field.
  PatchMatch::Problem problem{0, {1}, nullptr, &depths, nullptr};
  EXPECT_EQ(&problem.InputDepthMap(1), &depths[1]);

  std::vector<std::shared_ptr<const DepthMap>> leases(2);
  auto owner = std::make_shared<DepthMap>(2, 3, 1, 10);
  owner->Fill(4.0f);
  leases[0] = owner;
  leases[1] = owner;
  problem.source_depth_leases = &leases;
  EXPECT_EQ(&problem.InputDepthMap(0), &depths[0]);
  EXPECT_EQ(&problem.InputDepthMap(1), owner.get());
  leases[1].reset();
  EXPECT_EQ(&problem.InputDepthMap(1), &depths[1]);
  leases.resize(1);
  EXPECT_EQ(&problem.InputDepthMap(1), &depths[1]);
  leases.resize(2);
  leases[0].reset();
  leases[1] = std::shared_ptr<const DepthMap>(std::shared_ptr<const DepthMap>(),
                                              owner.get());
  ASSERT_NE(leases[1].get(), nullptr);
  ASSERT_EQ(leases[1].use_count(), 0);
  owner.reset();
  // The borrowed pointer is now expired and must never be dereferenced.
  EXPECT_EQ(&problem.InputDepthMap(1), &depths[1]);
  EXPECT_THROW(problem.InputDepthMap(-1), std::out_of_range);
  EXPECT_THROW(problem.InputDepthMap(2), std::out_of_range);
}

TEST(CachedPatchMatchInput, SharesSourcesAcrossReferencesAndCacheReloads) {
  const auto path = CreateTestDir();
  CreateDirIfNotExists(path / "sparse");
  CreateDirIfNotExists(path / "images");
  CreateDirIfNotExists(path / "stereo");
  CreateDirIfNotExists(path / "stereo" / "depth_maps");
  CreateDirIfNotExists(path / "stereo" / "normal_maps");
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 3;
  dataset.num_points3D = 4;
  dataset.camera_width = 10;
  dataset.camera_height = 5;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  reconstruction.Write(path / "sparse");
  for (const auto& [image_id, image] : reconstruction.Images()) {
    Bitmap bitmap(10, 5, true);
    bitmap.Fill(BitmapColor<uint8_t>(17, 31, 93));
    bitmap.Write(path / "images" / image.Name());
    DepthMap depth(10, 5, 1, 10);
    depth.Fill(std::nextafter(static_cast<float>(image_id), 10.0f));
    depth.Write(path / "stereo" / "depth_maps" /
                (image.Name() + ".geometric.bin"));
    NormalMap normal(10, 5);
    normal.Fill(0.25f);
    normal.Write(path / "stereo" / "normal_maps" /
                 (image.Name() + ".geometric.bin"));
  }
  for (const double cache_size : {1.0, 1e-7}) {
    for (const int num_workers : {1, 2, 4}) {
      SCOPED_TRACE(cache_size);
      SCOPED_TRACE(num_workers);
      Workspace::Options options;
      options.workspace_path = path;
      options.workspace_format = "COLMAP";
      options.input_type = "geometric";
      options.cache_size = cache_size;
      auto workspace = std::make_unique<CachedWorkspace>(options);
      ThreadPool pool(num_workers);
      std::array<std::vector<Image>, 2> images;
      std::array<std::vector<DepthMap>, 2> depths;
      std::array<std::vector<NormalMap>, 2> normals;
      std::array<std::vector<std::shared_ptr<const DepthMap>>, 2> leases;
      std::array<PatchMatch::Problem, 2> problems;
      for (int ref = 0; ref < 2; ++ref) {
        images[ref] = workspace->GetModel().images;
        depths[ref].resize(3);
        normals[ref].resize(3);
        leases[ref].resize(3);
        problems[ref].ref_image_idx = ref;
        problems[ref].depth_maps = &depths[ref];
        problems[ref].source_depth_leases = &leases[ref];
        internal::StagePatchMatchInputs(*workspace,
                                        pool,
                                        {0, 1, 2},
                                        true,
                                        ref,
                                        images[ref],
                                        depths[ref],
                                        leases[ref],
                                        normals[ref]);
      }
      ASSERT_NE(leases[0][2], nullptr);
      ASSERT_NE(leases[1][2], nullptr);
      const bool shared_owner = leases[0][2].get() == leases[1][2].get();
      if (cache_size == 1.0) {
        EXPECT_TRUE(shared_owner);
      }
      const DepthMap copied = *leases[0][2];
      const std::weak_ptr<const DepthMap> weak = leases[0][2];
      workspace->ClearCache();
      const auto reloaded = workspace->GetDepthMapLease(2);
      EXPECT_NE(reloaded.get(), leases[0][2].get());
      workspace.reset();
      for (const auto& problem : problems) {
        const auto& source = problem.InputDepthMap(2);
        ASSERT_EQ(source.GetNumBytes(), copied.GetNumBytes());
        EXPECT_EQ(
            std::memcmp(source.GetPtr(), copied.GetPtr(), copied.GetNumBytes()),
            0);
      }
      leases[0].clear();
      EXPECT_EQ(weak.expired(), !shared_owner);
      leases[1].clear();
      EXPECT_TRUE(weak.expired());
    }
  }
}

struct ReadGate {
  std::mutex mutex;
  std::condition_variable condition;
  bool delayed_entered = false;
  bool first_error_issued = false;
  bool released = false;
  bool delayed_completed = false;
  bool caller_returned = false;
  bool completed_before_return = false;

  void Signal(bool ReadGate::* field) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      this->*field = true;
    }
    condition.notify_all();
  }

  bool WaitFor(bool ReadGate::* field,
               std::chrono::milliseconds timeout = std::chrono::seconds(10)) {
    std::unique_lock<std::mutex> lock(mutex);
    return condition.wait_for(lock, timeout, [&] { return this->*field; });
  }
};

struct ReadCleanup {
  ReadGate& gate;
  ThreadPool& pool;
  std::thread caller;
  std::thread stopper;

  ~ReadCleanup() {
    gate.Signal(&ReadGate::released);
    if (caller.joinable()) caller.join();
    if (stopper.joinable()) stopper.join();
    try {
      pool.Wait();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Consume task errors if an assertion exits the test early.
      // The ordinary test path checks that staging already consumed them.
    }
  }
};

TEST_F(PatchMatchInputTest, ReadFailureDrainsAcceptedReadsBeforeReturning) {
  for (const Payload payload :
       {Payload::Bitmap, Payload::Depth, Payload::Normal}) {
    // Normal failures happen after the new source owner has been retained.
    workspace->owning_depth = payload == Payload::Normal;
    for (const bool secondary_failure : {false, true}) {
      SCOPED_TRACE(static_cast<int>(payload));
      SCOPED_TRACE(secondary_failure);
      ThreadPool pool(2);
      ReadGate gate;
      // Destinations survive the caller join and pool drain.
      // Check completion ordering while these payloads remain alive.
      auto images = workspace->GetModel().images;
      std::vector<DepthMap> depths(images.size());
      std::vector<std::shared_ptr<const DepthMap>> leases(depths.size());
      std::vector<NormalMap> normals(images.size());
      PatchMatch::Problem problem{0, {1}, &images, &depths, &normals, &leases};
      std::exception_ptr observed;
      workspace->before_read = [&](const int index, const Payload current) {
        if (current != payload) return;
        if (index == 0) {
          if (!gate.WaitFor(&ReadGate::delayed_entered)) {
            throw std::runtime_error("delayed input did not start");
          }
          gate.Signal(&ReadGate::first_error_issued);
          throw std::runtime_error("first input read failed");
        }
        gate.Signal(&ReadGate::delayed_entered);
        if (!gate.WaitFor(&ReadGate::released)) {
          throw std::runtime_error("delayed input was not released");
        }
        gate.Signal(&ReadGate::delayed_completed);
        if (secondary_failure) {
          throw std::runtime_error("second input read failed");
        }
      };
      ReadCleanup cleanup{gate, pool, {}, {}};
      cleanup.caller = std::thread([&] {
        try {
          internal::StagePatchMatchInputs(*workspace,
                                          pool,
                                          {0, 1},
                                          true,
                                          0,
                                          images,
                                          depths,
                                          leases,
                                          normals);
        } catch (...) {
          observed = std::current_exception();
        }
        {
          std::lock_guard<std::mutex> lock(gate.mutex);
          gate.completed_before_return = gate.delayed_completed;
        }
        gate.Signal(&ReadGate::caller_returned);
      });
      ASSERT_TRUE(gate.WaitFor(&ReadGate::delayed_entered));
      ASSERT_TRUE(gate.WaitFor(&ReadGate::first_error_issued));
      // Bound the observation while the accepted read is held. This is a
      // scheduling observation; completion order is checked again after join.
      EXPECT_FALSE(gate.WaitFor(&ReadGate::caller_returned,
                                std::chrono::milliseconds(500)));
      gate.Signal(&ReadGate::released);
      cleanup.caller.join();
      EXPECT_TRUE(gate.completed_before_return);
      ASSERT_NE(observed, nullptr);
      try {
        std::rethrow_exception(observed);
      } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "first input read failed");
      } catch (...) {
        ADD_FAILURE() << "Expected the first read's std::runtime_error";
      }
      EXPECT_NO_THROW(pool.Wait());

      // A consumed task error must not poison the shared pool's next problem.
      workspace->before_read = {};
      EXPECT_NO_THROW(internal::StagePatchMatchInputs(
          *workspace, pool, {0, 1}, true, 0, images, depths, leases, normals));
      for (int index = 0; index < 2; ++index) {
        EXPECT_EQ(images[index].GetBitmap().RowMajorData(),
                  workspace->bitmaps[index]->RowMajorData());
        EXPECT_EQ(problem.InputDepthMap(index).GetData(),
                  workspace->depths[index]->GetData());
        EXPECT_EQ(normals[index].GetData(),
                  workspace->normals[index]->GetData());
      }
      EXPECT_NO_THROW(pool.Wait());
    }
  }
}

TEST_F(PatchMatchInputTest, SubmissionFailurePreservesPoolDiagnostic) {
  ThreadPool pool(1);
  pool.Stop();
  auto images = workspace->GetModel().images;
  std::vector<DepthMap> depths;
  std::vector<std::shared_ptr<const DepthMap>> leases;
  std::vector<NormalMap> normals;
  try {
    internal::StagePatchMatchInputs(
        *workspace, pool, {0, 1}, false, 0, images, depths, leases, normals);
    FAIL() << "Expected submission to a stopped pool to fail";
  } catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()),
              "Cannot add task to stopped thread pool.");
  }
  EXPECT_EQ(workspace->reads, (std::array<std::array<int, 3>, 2>{}));
  EXPECT_NO_THROW(pool.Wait());
}

TEST_F(PatchMatchInputTest, SubmissionFailureDrainsPreviouslyAcceptedWork) {
  ThreadPool pool(1);
  ReadGate gate;
  auto images = workspace->GetModel().images;
  std::vector<DepthMap> depths;
  std::vector<std::shared_ptr<const DepthMap>> leases;
  std::vector<NormalMap> normals;
  std::exception_ptr observed;
  std::exception_ptr stop_error;
  ReadCleanup cleanup{gate, pool, {}, {}};

  pool.AddTask([&] {
    gate.Signal(&ReadGate::delayed_entered);
    if (!gate.WaitFor(&ReadGate::released)) {
      throw std::runtime_error("accepted input was not released");
    }
    images[0].SetBitmap(*workspace->GetBitmapLease(0));
    gate.Signal(&ReadGate::delayed_completed);
    throw std::runtime_error("previously accepted input failed");
  });
  ASSERT_TRUE(gate.WaitFor(&ReadGate::delayed_entered));
  const auto queued = pool.AddTask([] {});
  cleanup.stopper = std::thread([&] {
    try {
      pool.Stop();
    } catch (...) {
      stop_error = std::current_exception();
    }
  });

  // Stop sets its rejection state before discarding this queued task. The
  // active callback remains gated, so this observes Stop without polling.
  ASSERT_EQ(queued.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  try {
    queued.get();
    FAIL() << "Expected Stop to discard the queued task";
  } catch (const std::future_error& error) {
    EXPECT_EQ(error.code(),
              std::make_error_code(std::future_errc::broken_promise));
  }

  cleanup.caller = std::thread([&] {
    try {
      internal::StagePatchMatchInputs(
          *workspace, pool, {1}, false, 0, images, depths, leases, normals);
    } catch (...) {
      observed = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> lock(gate.mutex);
      gate.completed_before_return = gate.delayed_completed;
    }
    gate.Signal(&ReadGate::caller_returned);
  });
  EXPECT_FALSE(
      gate.WaitFor(&ReadGate::caller_returned, std::chrono::milliseconds(500)));
  gate.Signal(&ReadGate::released);
  cleanup.caller.join();
  cleanup.stopper.join();
  EXPECT_EQ(stop_error, nullptr);
  EXPECT_TRUE(gate.completed_before_return);
  ASSERT_NE(observed, nullptr);
  try {
    std::rethrow_exception(observed);
  } catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()),
              "Cannot add task to stopped thread pool.");
  } catch (...) {
    ADD_FAILURE() << "Expected the submission error";
  }
  EXPECT_NO_THROW(pool.Wait());
  EXPECT_EQ(workspace->reads[0], (std::array<int, 3>{1, 0, 0}));
  EXPECT_EQ(workspace->reads[1], (std::array<int, 3>{0, 0, 0}));
  EXPECT_EQ(images[0].GetBitmap().RowMajorData(),
            workspace->bitmaps[0]->RowMajorData());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
