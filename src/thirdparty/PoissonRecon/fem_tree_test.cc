// SPDX-License-Identifier: BSD-3-Clause

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <stdexcept>
#include <tuple>
#include <vector>

// FEMTree requires the configuration declarations before its headers.
// clang-format off
#include "PreProcessor.h"
#include "FEMTree.h"
// clang-format on
#include <gtest/gtest.h>

namespace {

TEST(PoissonThreadPoolTest, AsyncExceptionsJoinOutstandingWorkers) {
  using Pool = PoissonRecon::ThreadPool;
  using namespace std::chrono_literals;
  for (const bool throw_on_caller : {true, false}) {
    SCOPED_TRACE(throw_on_caller);
    std::promise<void> blocked;
    std::promise<void> throwing;
    std::promise<void> release;
    auto blocked_future = blocked.get_future();
    auto throwing_future = throwing.get_future();
    auto release_future = release.get_future().share();
    std::atomic<bool> worker_finished{false};
    auto result = std::async(std::launch::async, [&] {
      Pool::ParallelFor(
          0,
          3,
          [&](unsigned int thread, size_t) {
            if (thread == 2) {
              blocked.set_value();
              if (release_future.wait_for(5s) != std::future_status::ready) {
                throw std::runtime_error("worker release timed out");
              }
              worker_finished = true;
            } else if (thread == (throw_on_caller ? 0u : 1u)) {
              if (blocked_future.wait_for(5s) != std::future_status::ready) {
                throw std::runtime_error("worker did not start");
              }
              throwing.set_value();
              throw std::runtime_error("injected worker failure");
            }
          },
          3,
          Pool::ASYNC,
          Pool::STATIC,
          1);
    });
    const auto started = throwing_future.wait_for(5s);
    const bool returned_early =
        started == std::future_status::ready &&
        result.wait_for(100ms) == std::future_status::ready;
    release.set_value();
    EXPECT_EQ(started, std::future_status::ready);
    EXPECT_FALSE(returned_early);
    EXPECT_THROW(result.get(), std::runtime_error);
    EXPECT_TRUE(worker_finished.load());

    std::atomic<size_t> completed{0};
    Pool::ParallelFor(
        0,
        3,
        [&](unsigned int, size_t) { ++completed; },
        3,
        Pool::ASYNC,
        Pool::STATIC,
        1);
    EXPECT_EQ(completed.load(), 3);
  }
}

TEST(PoissonFEMTreeTest, NeighborSelectionCanBeEmpty) {
  PoissonRecon::FEMTree<3, float> tree(1024);
  size_t visited = 0;
  const auto visit = [&](const auto*) { ++visited; };
  tree.processNeighbors<0, 0, false>(
      [](const auto*) { return false; }, std::tuple<>{}, visit);
  EXPECT_EQ(visited, 0);
  tree.processNeighbors<0, 0, false>(
      [](const auto*) { return true; }, std::tuple<>{}, visit);
  EXPECT_GT(visited, 0);
}

TEST(PoissonFEMTreeTest, NeighboringLeafSelectionCanBeEmpty) {
  PoissonRecon::FEMTree<3, float> tree(1024);
  std::atomic<size_t> visited{0};
  const auto visit = [&](const auto*) { ++visited; };
  tree.processNeighboringLeaves<0, 0>(
      [](const auto*) { return false; }, visit, false);
  EXPECT_EQ(visited.load(), 0);
  tree.processNeighboringLeaves<0, 0>(
      [](const auto*) { return true; }, visit, false);
  EXPECT_GT(visited.load(), 0);
}

TEST(PoissonFEMTreeTest, InitializesTriangleInsideOneGridCell) {
  using namespace PoissonRecon;
  FEMTree<3, float> tree(1024);
  const std::vector<Point<float, 3>> vertices = {
      Point<float, 3>(0.10f, 0.10f, 0.10f),
      Point<float, 3>(0.20f, 0.10f, 0.10f),
      Point<float, 3>(0.10f, 0.20f, 0.10f)};
  const std::vector<SimplexIndex<2, node_index_type>> triangles = {{0, 1, 2}};
  std::vector<NodeSimplices<3, float>> node_simplices;
  FEMTreeInitializer<3, float>::Initialize(tree.spaceRoot(),
                                           vertices,
                                           triangles,
                                           1,
                                           2,
                                           node_simplices,
                                           tree.nodeAllocators,
                                           tree.initializer());
  ASSERT_FALSE(node_simplices.empty());
  size_t count = 0;
  for (const auto& node : node_simplices) {
    ASSERT_NE(node.node, nullptr);
    for (const auto& simplex : node.data) {
      EXPECT_EQ(simplex.first, 0);
      EXPECT_GT(simplex.second.measure(), 0);
      ++count;
    }
  }
  EXPECT_GT(count, 0);
}

}  // namespace
