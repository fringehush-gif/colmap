// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// FEMTree requires the configuration declarations before its headers.
// clang-format off
#include "normal_capture.h"
#include "PreProcessor.h"
#include "FEMTree.h"
// clang-format on
#include <gtest/gtest.h>

namespace {

struct GridValue {
  inline static int alive = 0;
  int value = 0;
  GridValue() { ++alive; }
  ~GridValue() { --alive; }
  GridValue(const GridValue&) = delete;
  GridValue& operator=(const GridValue&) = delete;
};

// The move contract guarantees an empty, reusable source.
// NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
TEST(PoissonRegularGridTest, TransfersStorageAndReleasesPriorDestination) {
  using Grid = PoissonRecon::RegularGrid<GridValue, 2>;
  static_assert(!std::is_copy_constructible_v<Grid>);
  static_assert(!std::is_copy_assignable_v<Grid>);
  EXPECT_EQ(GridValue::alive, 0);
  {
    Grid source;
    source.resize(2, 3);
    source[5].value = 17;
    Grid destination(std::move(source));
    EXPECT_EQ(source.resolution(), 0);
    EXPECT_FALSE(source());
    EXPECT_EQ(destination.res(0), 2);
    EXPECT_EQ(destination.res(1), 3);
    EXPECT_EQ(destination[5].value, 17);
    EXPECT_EQ(GridValue::alive, 6);
    source.resize(1, 2);
    EXPECT_EQ(GridValue::alive, 8);
    source = std::move(destination);
    EXPECT_EQ(GridValue::alive, 6);
    EXPECT_EQ(source[5].value, 17);
    EXPECT_EQ(destination.resolution(), 0);
    EXPECT_FALSE(destination());
    auto* const alias = &source;
    source = std::move(*alias);
    EXPECT_EQ(source[5].value, 17);
    EXPECT_EQ(GridValue::alive, 6);
    source = Grid();
    EXPECT_EQ(source.resolution(), 0);
    EXPECT_FALSE(source());
    EXPECT_EQ(GridValue::alive, 0);
  }
  EXPECT_EQ(GridValue::alive, 0);
}
// NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)

TEST(PoissonRegularGridTest, ReturnsAndRelocatesGridsWithoutCopyingElements) {
  const auto make_grid = [] {
    PoissonRecon::RegularGrid<int, 2> grid;
    grid.resize(2, 3);
    grid[5] = 17;
    return grid;
  };
  const auto grid = make_grid();
  EXPECT_EQ(grid.resolution(), 6);
  EXPECT_EQ(grid[5], 17);
  std::vector<PoissonRecon::RegularGrid<std::mutex, 2>> grids(1);
  grids[0].resize(1, 2);
  const auto* const first_mutex = &grids[0][0];
  grids.resize(2);
  EXPECT_EQ(&grids[0][0], first_mutex);
  EXPECT_EQ(grids[0].resolution(), 2);
  EXPECT_EQ(grids[1].resolution(), 0);
}

struct ChildPublicationData {
  std::uint64_t marker = 0;
  std::uint64_t complement = 0;
};
using ChildPublicationNode =
    PoissonRecon::RegularTreeNode<3, ChildPublicationData, int>;
constexpr std::uint64_t kChildMarker = UINT64_C(0x7a39d4e12bcf5680);

bool CheckPublishedChildren(const ChildPublicationNode* children,
                            const ChildPublicationNode* parent) {
  if (!children) return false;
  for (int i = 0; i < 8; ++i) {
    if (children[i].parent != parent || children[i].depth() != 1 ||
        children[i].nodeData.marker != kChildMarker ||
        children[i].nodeData.complement != ~kChildMarker) {
      return false;
    }
  }
  return true;
}

TEST(PoissonRegularTreeTest, SerialChildLookupDoesNotReinitialize) {
  ChildPublicationNode root;
  EXPECT_EQ(root.getChildren<false>(), nullptr);
  EXPECT_EQ(root.getChildren<true>(), nullptr);
  size_t initialized = 0;
  auto initialize = [&](ChildPublicationNode& child) {
    ++initialized;
    child.nodeData.marker = kChildMarker;
    child.nodeData.complement = ~kChildMarker;
  };
  auto* const first = root.getOrInitChildren<false>(nullptr, initialize);
  auto* const second = root.getOrInitChildren<false>(nullptr, initialize);
  EXPECT_NE(first, nullptr);
  EXPECT_EQ(second, first);
  EXPECT_EQ(root.getChildren<false>(), first);
  EXPECT_EQ(root.getChildren<true>(), first);
  EXPECT_EQ(initialized, 8);
  EXPECT_TRUE(CheckPublishedChildren(first, &root));
  root.cleanChildren(true);
  EXPECT_EQ(root.getChildren<false>(), nullptr);
  EXPECT_EQ(root.getChildren<true>(), nullptr);
}

TEST(PoissonRegularTreeTest, ConcurrentCreatorsObservePublishedBrood) {
  using namespace std::chrono_literals;
  constexpr size_t kWorkers = 4;
  ChildPublicationNode root;
  std::atomic<bool> start{false}, abort{false}, timed_out{false};
  std::atomic<size_t> arrivals{0}, initialized{0};
  std::array<ChildPublicationNode*, kWorkers> results{};
  std::array<bool, kWorkers> valid{};
  std::array<std::exception_ptr, kWorkers> errors{};
  std::vector<std::thread> workers;
  workers.reserve(kWorkers);
  try {
    for (size_t i = 0; i < kWorkers; ++i) {
      workers.emplace_back([&, i] {
        while (!start.load(std::memory_order_acquire))
          std::this_thread::yield();
        if (abort.load(std::memory_order_acquire)) return;
        try {
          bool first_child = true;
          auto initialize = [&](ChildPublicationNode& child) {
            if (first_child) {
              first_child = false;
              arrivals.fetch_add(1, std::memory_order_acq_rel);
              const auto deadline = std::chrono::steady_clock::now() + 5s;
              while (arrivals.load(std::memory_order_acquire) != kWorkers &&
                     !abort.load(std::memory_order_acquire)) {
                if (std::chrono::steady_clock::now() >= deadline) {
                  timed_out.store(true, std::memory_order_relaxed);
                  abort.store(true, std::memory_order_release);
                  break;
                }
                std::this_thread::yield();
              }
            }
            child.nodeData.marker = kChildMarker;
            child.nodeData.complement = ~kChildMarker;
            initialized.fetch_add(1, std::memory_order_relaxed);
          };
          results[i] = root.getOrInitChildren<true>(nullptr, initialize);
          valid[i] = CheckPublishedChildren(results[i], &root);
        } catch (...) {
          errors[i] = std::current_exception();
          abort.store(true, std::memory_order_release);
        }
      });
    }
  } catch (...) {
    // Release and join workers even if only part of the team was created.
    abort.store(true, std::memory_order_release);
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    root.cleanChildren(true);
    FAIL() << "Could not create all child-publication workers";
  }
  start.store(true, std::memory_order_release);
  for (auto& worker : workers) worker.join();

  EXPECT_FALSE(timed_out.load());
  EXPECT_FALSE(abort.load());
  EXPECT_EQ(arrivals.load(), kWorkers);
  EXPECT_EQ(initialized.load(), kWorkers * 8);
  for (size_t i = 0; i < kWorkers; ++i) {
    EXPECT_EQ(errors[i], nullptr);
    EXPECT_TRUE(valid[i]);
    EXPECT_EQ(results[i], results[0]);
  }
  EXPECT_NE(results[0], nullptr);
  EXPECT_EQ(root.getChildren<true>(), results[0]);
  EXPECT_EQ(root.getChildren<false>(), results[0]);
  root.cleanChildren(true);
}

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

#ifdef _OPENMP
TEST(PoissonThreadPoolTest, OpenMpExceptionsJoinAndRestoreCaptureScope) {
  using Pool = PoissonRecon::ThreadPool;
  using namespace std::chrono_literals;
  const int old_dynamic = omp_get_dynamic();
  omp_set_dynamic(0);
  int team_size = 0;
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    team_size = omp_get_num_threads();
  }
  omp_set_dynamic(old_dynamic);
  if (team_size < 2) GTEST_SKIP() << "OpenMP needs two workers";

  for (const auto schedule : {Pool::STATIC, Pool::DYNAMIC}) {
    SCOPED_TRACE(static_cast<int>(schedule));
    std::promise<void> blocked, throwing, release;
    auto blocked_future = blocked.get_future();
    auto throwing_future = throwing.get_future();
    auto release_future = release.get_future().share();
    std::atomic<int> restored{0};
    std::atomic<bool> worker_finished{false};
    std::vector<std::vector<PoissonNormalGpu::Record>> records(2);
    auto result = std::async(std::launch::async, [&] {
      omp_set_dynamic(0);
      Pool::ParallelFor(
          0,
          2,
          [&](unsigned int thread, size_t chunk) {
            auto* const prior_target = PoissonNormalCapture::target;
            const bool prior_admission = PoissonNormalCapture::admitted;
            try {
              PoissonNormalCapture::Scope admission(true);
              PoissonNormalCapture::ThreadScope scope(&records[thread]);
              if (chunk == 0) {
                if (blocked_future.wait_for(5s) != std::future_status::ready)
                  throw std::runtime_error("worker did not start");
                throwing.set_value();
                throw std::runtime_error("injected worker failure");
              }
              blocked.set_value();
              if (release_future.wait_for(5s) != std::future_status::ready)
                throw std::runtime_error("worker release timed out");
              worker_finished = true;
            } catch (...) {
              if (PoissonNormalCapture::target == prior_target &&
                  PoissonNormalCapture::admitted == prior_admission)
                ++restored;
              throw;
            }
            if (PoissonNormalCapture::target == prior_target &&
                PoissonNormalCapture::admitted == prior_admission)
              ++restored;
          },
          2,
          Pool::OPEN_MP,
          schedule,
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
    EXPECT_EQ(restored.load(), 2);

    std::atomic<size_t> completed{0};
    Pool::ParallelFor(
        0,
        8,
        [&](unsigned int, size_t) { ++completed; },
        2,
        Pool::OPEN_MP,
        schedule,
        1);
    EXPECT_EQ(completed.load(), 8);
  }
}
#endif

#if defined(_OPENMP) && defined(POISSON_NORMAL_NATIVE_GPU)
TEST(PoissonFEMTreeTest, CapturedNormalCallbackFailureJoinsBeforeReturn) {
  using namespace PoissonRecon;
  using Pool = ThreadPool;
  using namespace std::chrono_literals;
  const int old_dynamic = omp_get_dynamic();
  omp_set_dynamic(0);
  int team_size = 0;
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    team_size = omp_get_num_threads();
  }
  if (team_size < 2) {
    omp_set_dynamic(old_dynamic);
    GTEST_SKIP() << "OpenMP needs two workers";
  }
  struct RestorePool {
    unsigned int threads = Pool::NumThreads();
    Pool::ParallelType parallel = Pool::ParallelizationType;
    Pool::ScheduleType schedule = Pool::Schedule;
    size_t chunk = Pool::ChunkSize;
    int dynamic;
    ~RestorePool() {
      Pool::SetNumThreads(threads);
      Pool::ParallelizationType = parallel;
      Pool::Schedule = schedule;
      Pool::ChunkSize = chunk;
      omp_set_dynamic(dynamic);
    }
  } restore{Pool::NumThreads(),
            Pool::ParallelizationType,
            Pool::Schedule,
            Pool::ChunkSize,
            old_dynamic};
  Pool::SetNumThreads(2);
  Pool::ParallelizationType = Pool::OPEN_MP;
  Pool::Schedule = Pool::DYNAMIC;
  Pool::ChunkSize = 128;

  using Tree = FEMTree<3, float>;
  using P = Point<float, 3>;
  constexpr size_t kSamples = 400000;
  Tree tree(1024);
  Tree::PointSample sample{};
  sample.node = &tree.spaceRoot();
  sample.sample = ProjectiveData<P, float>(P(0.5f, 0.5f, 0.5f), 1.f);
  std::vector<Tree::PointSample> samples(kSamples, sample);
  std::vector<P> normals(kSamples, P(1.f, 0.f, 0.f));
  // A valid estimator object is only an admission input: this callback never
  // returns true, so the estimator and splatting path are never read.
  Tree::DensityEstimator<2> density(0, 1, 1.f);
  ProjectiveData<Point<float, 2>, float> weights;
  std::atomic<size_t> calls{0};
  std::atomic<unsigned int> observed_threads{0};
  std::atomic<bool> admitted_target{false};
  std::promise<void> second_callback;
  auto second_ready = second_callback.get_future();
  std::function<bool(P, P&)> convert = [&](P, P&) {
    const auto index = calls.fetch_add(1);
    observed_threads.fetch_or(1u << omp_get_thread_num());
    if (PoissonNormalCapture::target) admitted_target = true;
    if (index == 0) {
      if (second_ready.wait_for(5s) != std::future_status::ready)
        throw std::runtime_error("second capture callback did not start");
      throw std::bad_alloc();
    }
    if (index == 1) second_callback.set_value();
    return false;
  };
  auto* const prior_target = PoissonNormalCapture::target;
  const bool prior_admitted = PoissonNormalCapture::admitted;
  {
    PoissonNormalCapture::Scope admission(true);
    using Sig = IsotropicUIntPack<3, FEMDegreeAndBType<2>::Signature>;
    EXPECT_THROW(tree.setInterpolatedDataField(P(),
                                               Sig(),
                                               samples,
                                               normals,
                                               &density,
                                               0,
                                               0,
                                               0.f,
                                               weights,
                                               convert),
                 std::bad_alloc);
    EXPECT_TRUE(PoissonNormalCapture::admitted);
  }
  EXPECT_TRUE(admitted_target.load());
  EXPECT_EQ(observed_threads.load() & 3u, 3u);
  EXPECT_EQ(PoissonNormalCapture::target, prior_target);
  EXPECT_EQ(PoissonNormalCapture::admitted, prior_admitted);
  const size_t count_after_return = calls.load();
  std::atomic<size_t> reused{0};
  Pool::ParallelFor(
      0,
      8,
      [&](unsigned int, size_t) { ++reused; },
      2,
      Pool::OPEN_MP,
      Pool::DYNAMIC,
      1);
  EXPECT_EQ(reused.load(), 8);
  EXPECT_EQ(calls.load(), count_after_return);
}
#endif

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
