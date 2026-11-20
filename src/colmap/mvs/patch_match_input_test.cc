// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_input.h"

#include "colmap/mvs/workspace.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"
#include "colmap/util/threading.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

class InputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path = CreateTestDir();
    CreateDirIfNotExists(path / "sparse");
    CreateDirIfNotExists(path / "images");
    CreateDirIfNotExists(path / "stereo");
    CreateDirIfNotExists(path / "stereo" / "depth_maps");
    CreateDirIfNotExists(path / "stereo" / "normal_maps");
    SyntheticDatasetOptions synthetic;
    synthetic.num_rigs = 1;
    synthetic.num_cameras_per_rig = 1;
    synthetic.num_frames_per_rig = 3;
    synthetic.camera_width = 10;
    synthetic.camera_height = 5;
    Reconstruction reconstruction;
    SynthesizeDataset(synthetic, &reconstruction);
    reconstruction.Write(path / "sparse");
    int value = 1;
    for (const auto& [id, image] : reconstruction.Images()) {
      names.push_back(image.Name());
      Bitmap bitmap(10, 5, true);
      bitmap.Fill(BitmapColor<uint8_t>(value, value * 2, value * 3));
      bitmap.Write(path / "images" / image.Name());
      Mat<float> depth(10, 5, 1);
      depth.Fill(static_cast<float>(value));
      depth.Write(path / "stereo" / "depth_maps" /
                  (image.Name() + ".geometric.bin"));
      Mat<float> normal(10, 5, 3);
      normal.Fill(static_cast<float>(value) / 8);
      normal.Write(path / "stereo" / "normal_maps" /
                   (image.Name() + ".geometric.bin"));
      ++value;
    }
    ASSERT_EQ(names.size(), 3);
    options.workspace_path = path;
    options.workspace_format = "COLMAP";
    options.input_type = "geometric";
  }
  std::filesystem::path path;
  std::vector<std::string> names;
  Workspace::Options options;
};

// A test-only gate at the real copy boundary. Every wait is bounded; cleanup
// always releases the gate and joins the caller before removing the hooks.
struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool released = false;
  bool captured_error = false;
  bool caller_returned = false;
  bool timeout = false;
  void Block() {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    cv.notify_all();
    if (!cv.wait_for(
            lock, std::chrono::seconds(10), [&] { return released; })) {
      timeout = true;
    }
  }
  bool WaitFor(bool Gate::* flag,
               std::chrono::milliseconds duration = std::chrono::seconds(10)) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, duration, [&] { return this->*flag; });
  }
  void Release() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      released = true;
    }
    cv.notify_all();
  }
  void CapturedError() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      captured_error = true;
    }
    cv.notify_all();
  }
  void CallerReturned() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      caller_returned = true;
    }
    cv.notify_all();
  }
};

struct Cleanup {
  Gate& gate;
  ThreadPool& pool;
  std::thread caller;
  ~Cleanup() {
    gate.Release();
    if (caller.joinable()) caller.join();
    try {
      pool.Wait();
    } catch (...) {
      // Teardown must not throw; normal test paths check task errors earlier.
    }
    internal::SetInputCopyHookForTesting({});
    internal::SetInputSubmissionHookForTesting({});
  }
};

TEST_F(InputTest, FailedReadAndPartialSubmissionDrainBeforeUnwind) {
  for (const bool submission_failure : {false, true}) {
    SCOPED_TRACE(submission_failure);
    CachedWorkspace workspace(options);
    ThreadPool pool(2);
    const int bad = workspace.GetModel().GetImageIdx(names[0]);
    const int blocked = workspace.GetModel().GetImageIdx(names[1]);
    if (!submission_failure) {
      // Both accepted tasks fail; the first future's error must win, while
      // Wait consumes the checkers for both failures after the gate opens.
      for (const int idx : {bad, blocked}) {
        std::ofstream malformed(workspace.GetNormalMapPath(idx),
                                std::ios::trunc);
        malformed << "0&0&0&";
      }
    }
    Gate gate;
    std::mutex master;
    std::atomic<bool> destinations_alive{false};
    std::atomic<bool> returned{false};
    std::exception_ptr observed;
    internal::SetInputCopyHookForTesting([&](int idx, auto payload) {
      if (idx == blocked && payload == internal::InputPayload::Bitmap)
        gate.Block();
    });
    internal::SetInputSubmissionHookForTesting([&](size_t count) {
      if (count == 0) {
        gate.CapturedError();
      } else if (submission_failure && count == 1) {
        if (!gate.WaitFor(&Gate::entered))
          throw std::runtime_error("gate timeout");
        throw std::runtime_error("submission sentinel");
      }
    });
    Cleanup cleanup{gate, pool};
    cleanup.caller = std::thread([&] {
      {
        std::lock_guard<std::mutex> lock(master);
        auto images = workspace.GetModel().images;
        std::vector<DepthMap> depths(images.size());
        std::vector<NormalMap> normals(images.size());
        destinations_alive = true;
        try {
          const std::vector<int> indexes = submission_failure
                                               ? std::vector<int>{blocked, bad}
                                               : std::vector<int>{bad, blocked};
          internal::StagePatchMatchInputs(
              workspace, pool, indexes, true, bad, images, depths, normals);
        } catch (...) {
          observed = std::current_exception();
        }
        destinations_alive = false;
        returned = true;
      }
      gate.CallerReturned();
    });
    ASSERT_TRUE(gate.WaitFor(&Gate::entered));
    ASSERT_TRUE(gate.WaitFor(&Gate::captured_error));
    // The failing control unwinds promptly here; the real helper remains in
    // Wait until the accepted writer is released.
    EXPECT_FALSE(
        gate.WaitFor(&Gate::caller_returned, std::chrono::milliseconds(500)));
    EXPECT_TRUE(destinations_alive.load());
    EXPECT_FALSE(returned.load());
    const bool acquired_master = master.try_lock();
    if (acquired_master) master.unlock();
    EXPECT_FALSE(acquired_master);
    gate.Release();
    cleanup.caller.join();
    EXPECT_FALSE(gate.timeout);
    ASSERT_NE(observed, nullptr);
    try {
      std::rethrow_exception(observed);
    } catch (const std::exception& e) {
      const std::string message = e.what();
      if (submission_failure) {
        EXPECT_EQ(message, "submission sentinel");
      } else {
        EXPECT_NE(message.find(names[0]), std::string::npos) << message;
      }
    }
    EXPECT_NO_THROW(pool.Wait());
    internal::SetInputCopyHookForTesting({});
    internal::SetInputSubmissionHookForTesting({});
    auto images = workspace.GetModel().images;
    std::vector<DepthMap> depths(images.size());
    std::vector<NormalMap> normals(images.size());
    EXPECT_NO_THROW(internal::StagePatchMatchInputs(
        workspace, pool, {blocked}, false, blocked, images, depths, normals));
    EXPECT_FALSE(images.at(blocked).GetBitmap().IsEmpty());
    EXPECT_NO_THROW(pool.Wait());
  }
}

TEST_F(InputTest, ReferenceNormalOnlyButAllSourceNormalsAcquired) {
  CachedWorkspace workspace(options);
  ThreadPool pool(2);
  std::vector<int> indexes;
  indexes.reserve(names.size());
  for (const auto& name : names)
    indexes.push_back(workspace.GetModel().GetImageIdx(name));
  const int ref = indexes[0];
  auto images = workspace.GetModel().images;
  std::vector<DepthMap> depths(images.size());
  std::vector<NormalMap> normals(images.size());
  std::mutex mutex;
  std::vector<int> normal_acquisitions(images.size(), 0);
  internal::SetInputCopyHookForTesting([&](int idx, auto payload) {
    if (payload == internal::InputPayload::NormalMap) {
      std::lock_guard<std::mutex> lock(mutex);
      ++normal_acquisitions.at(idx);
    }
  });
  Gate gate;
  Cleanup cleanup{gate, pool};
  ASSERT_NO_THROW(internal::StagePatchMatchInputs(
      workspace, pool, indexes, true, ref, images, depths, normals));
  ASSERT_EQ(images.size(), 3);
  ASSERT_EQ(normals.size(), images.size());
  for (const int idx : indexes) {
    Bitmap expected_bitmap;
    expected_bitmap.Read(workspace.GetBitmapPath(idx), options.image_as_rgb);
    DepthMap expected_depth;
    expected_depth.Read(workspace.GetDepthMapPath(idx));
    EXPECT_EQ(images[idx].GetBitmap().RowMajorData(),
              expected_bitmap.RowMajorData());
    EXPECT_EQ(depths[idx].GetData(), expected_depth.GetData());
    EXPECT_EQ(normal_acquisitions[idx], 1);
    if (idx != ref) {
      EXPECT_TRUE(normals[idx].GetData().empty());
    }
  }
  NormalMap expected_normal;
  expected_normal.Read(workspace.GetNormalMapPath(ref));
  EXPECT_FALSE(expected_normal.GetData().empty());
  EXPECT_EQ(normals[ref].GetData(), expected_normal.GetData());

  // Cached inputs must still be acquired on a second pass, even after the
  // source file becomes malformed; a new workspace must observe the failure.
  {
    std::ofstream malformed(workspace.GetNormalMapPath(indexes[1]),
                            std::ios::trunc);
    malformed << "0&0&0&";
  }
  EXPECT_NO_THROW(internal::StagePatchMatchInputs(
      workspace, pool, indexes, true, ref, images, depths, normals));
  for (const int idx : indexes) EXPECT_EQ(normal_acquisitions[idx], 2);
  internal::SetInputCopyHookForTesting({});
  CachedWorkspace fresh(options);
  EXPECT_THROW(internal::StagePatchMatchInputs(
                   fresh, pool, indexes, true, ref, images, depths, normals),
               std::exception);
  EXPECT_NO_THROW(pool.Wait());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
