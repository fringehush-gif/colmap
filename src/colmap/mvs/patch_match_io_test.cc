// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_io.h"

#include "colmap/mvs/patch_match_options.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(PatchMatchOptions, WindowMatchesCompiledKernels) {
  PatchMatchOptions options;
  for (const int step : {1, 2}) {
    options.window_step = step;
    for (const int radius : {1, 20}) {
      options.window_radius = radius;
      EXPECT_TRUE(options.Check());
    }
    for (const int radius : {0, 21, 32}) {
      options.window_radius = radius;
      EXPECT_FALSE(options.Check());
    }
  }
  options.window_radius = 5;
  for (const int step : {0, 3}) {
    options.window_step = step;
    EXPECT_FALSE(options.Check());
  }
}

TEST(PatchMatchInputReads, SubmissionFailureJoinsBorrowedSibling) {
  ThreadPool pool(2);
  std::promise<void> sibling_started;
  auto started = sibling_started.get_future();
  std::promise<void> release_sibling;
  const auto release = release_sibling.get_future().share();
  std::atomic<bool> saw_live_owner{false};

  auto caller = std::async(std::launch::async, [&]() {
    auto local_owner = std::make_shared<int>(42);
    std::weak_ptr<int> borrowed = local_owner;
    try {
      RunPatchMatchTasks(pool, [&](auto& futures) {
        futures.push_back(pool.AddTask([&, borrowed]() {
          sibling_started.set_value();
          release.wait();
          const auto owner = borrowed.lock();
          saw_live_owner = owner && *owner == 42;
          throw std::runtime_error("sibling read failure");
        }));
        if (started.wait_for(std::chrono::seconds(5)) !=
            std::future_status::ready) {
          throw std::runtime_error("sibling did not start");
        }
        throw std::runtime_error("submission failure");
      });
    } catch (const std::runtime_error& error) {
      return std::string(error.what());
    }
    return std::string("no failure");
  });

  EXPECT_EQ(caller.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  release_sibling.set_value();
  EXPECT_EQ(caller.get(), "submission failure");
  EXPECT_TRUE(saw_live_owner);
  EXPECT_NO_THROW(pool.Wait());
}

TEST(PatchMatchInputReads, ReadFailureJoinsSiblingAndDrainsChecker) {
  ThreadPool pool(2);
  std::promise<void> sibling_started;
  auto started = sibling_started.get_future();
  std::promise<void> release_sibling;
  const auto release = release_sibling.get_future().share();
  std::atomic<bool> sibling_completed{false};

  auto caller = std::async(std::launch::async, [&]() {
    try {
      RunPatchMatchTasks(pool, [&](auto& futures) {
        futures.push_back(
            pool.AddTask([]() { throw std::runtime_error("read failure"); }));
        futures.push_back(pool.AddTask([&]() {
          sibling_started.set_value();
          release.wait();
          sibling_completed = true;
        }));
        if (started.wait_for(std::chrono::seconds(5)) !=
            std::future_status::ready) {
          throw std::runtime_error("sibling did not start");
        }
      });
    } catch (const std::runtime_error& error) {
      return std::string(error.what());
    }
    return std::string("no failure");
  });

  EXPECT_EQ(caller.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  release_sibling.set_value();
  EXPECT_EQ(caller.get(), "read failure");
  EXPECT_TRUE(sibling_completed);
  EXPECT_NO_THROW(pool.Wait());
}

TEST(PatchMatchTasks, UnretainedWorkerFailuresDrainCheckers) {
  ThreadPool pool(2);
  try {
    RunPatchMatchTasks(pool, [&](auto&) {
      pool.AddTask([]() { throw std::runtime_error("first worker failure"); });
      pool.AddTask([]() { throw std::runtime_error("second worker failure"); });
    });
    FAIL() << "Expected both worker failures";
  } catch (const AggregateException& errors) {
    EXPECT_EQ(errors.exceptions().size(), 2);
  }
  EXPECT_NO_THROW(pool.Wait());
  EXPECT_EQ(pool.AddTask([]() { return 42; }).get(), 42);
  EXPECT_NO_THROW(pool.Wait());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
