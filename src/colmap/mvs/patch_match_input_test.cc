// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_input.h"

#include "colmap/mvs/workspace.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"
#include "colmap/util/threading.h"

#include <array>
#include <chrono>
#include <condition_variable>
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
  const NormalMap& GetNormalMap(const int index) override {
    ++reads.at(index)[2];
    if (before_read) before_read(index, Payload::Normal);
    return *normals.at(index);
  }

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
    std::vector<NormalMap> normals(geometry ? images.size() : 0);
    internal::StagePatchMatchInputs(
        *workspace, pool, {1, 0}, geometry, images, depths, normals);
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
        *workspace, pool, {}, geometry, images, depths, normals);
    EXPECT_EQ(workspace->reads, reads);
    EXPECT_NO_THROW(pool.Wait());
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

  void Signal(bool ReadGate::*field) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      this->*field = true;
    }
    condition.notify_all();
  }

  bool WaitFor(bool ReadGate::*field,
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
    for (const bool secondary_failure : {false, true}) {
      SCOPED_TRACE(static_cast<int>(payload));
      SCOPED_TRACE(secondary_failure);
      ThreadPool pool(2);
      ReadGate gate;
      // Destinations survive the caller join and pool drain.
      // Check completion ordering while these payloads remain alive.
      auto images = workspace->GetModel().images;
      std::vector<DepthMap> depths(images.size());
      std::vector<NormalMap> normals(images.size());
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
          internal::StagePatchMatchInputs(
              *workspace, pool, {0, 1}, true, images, depths, normals);
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
          *workspace, pool, {0, 1}, true, images, depths, normals));
      for (int index = 0; index < 2; ++index) {
        EXPECT_EQ(images[index].GetBitmap().RowMajorData(),
                  workspace->bitmaps[index]->RowMajorData());
        EXPECT_EQ(depths[index].GetData(), workspace->depths[index]->GetData());
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
  std::vector<NormalMap> normals;
  try {
    internal::StagePatchMatchInputs(
        *workspace, pool, {0, 1}, false, images, depths, normals);
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
          *workspace, pool, {1}, false, images, depths, normals);
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
