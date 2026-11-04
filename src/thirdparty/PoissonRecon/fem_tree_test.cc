// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
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

TEST(PoissonThreadPoolTest, AsyncSchedulesPreserveTypedAndErasedCallbacks) {
  using Pool = PoissonRecon::ThreadPool;
  constexpr size_t kBegin = 7;
  constexpr size_t kEnd = 30;
  struct StatefulCallback {
    std::array<std::atomic<size_t>, 30> visits;
    std::atomic<size_t> calls{0};
    StatefulCallback() {
      for (auto& value : visits) value.store(0, std::memory_order_relaxed);
    }
    StatefulCallback(const StatefulCallback& other) {
      for (size_t i = 0; i < visits.size(); ++i)
        visits[i].store(other.visits[i].load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
      calls.store(other.calls.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
    }
    void operator()(unsigned int thread, size_t i) {
      EXPECT_LT(thread, 3u);
      ++visits.at(i);
      ++calls;
    }
  };
  for (const auto schedule : {Pool::STATIC, Pool::DYNAMIC}) {
    SCOPED_TRACE(static_cast<int>(schedule));
    std::array<std::atomic<size_t>, kEnd> visits;
    for (auto& value : visits) value.store(0, std::memory_order_relaxed);
    Pool::ParallelFor(
        kBegin,
        kEnd,
        [&](unsigned int thread, size_t i) {
          EXPECT_LT(thread, 3u);
          ++visits.at(i);
        },
        3,
        Pool::ASYNC,
        schedule,
        5);
    std::function<void(unsigned int, size_t)> erased = StatefulCallback{};
    Pool::ParallelFor(kBegin, kEnd, erased, 3, Pool::ASYNC, schedule, 5);
    const auto* state = erased.target<StatefulCallback>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->calls.load(), kEnd - kBegin);
    for (size_t i = 0; i < kEnd; ++i) {
      EXPECT_EQ(visits[i].load(), i >= kBegin ? 1u : 0u);
      EXPECT_EQ(state->visits[i].load(), visits[i].load());
    }
  }
}

TEST(PoissonThreadPoolTest, AsyncExceptionsJoinOutstandingWorkers) {
  using Pool = PoissonRecon::ThreadPool;
  using namespace std::chrono_literals;
  for (const auto schedule : {Pool::STATIC, Pool::DYNAMIC}) {
    SCOPED_TRACE(static_cast<int>(schedule));
    for (const bool throw_on_caller : {true, false}) {
      SCOPED_TRACE(throw_on_caller);
      std::promise<void> blocked;
      std::promise<void> throwing;
      std::promise<void> release;
      auto blocked_future = blocked.get_future().share();
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
              } else if (blocked_future.wait_for(5s) !=
                         std::future_status::ready) {
                throw std::runtime_error("blocked worker did not start");
              }
            },
            3,
            Pool::ASYNC,
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

      std::atomic<size_t> completed{0};
      Pool::ParallelFor(
          0,
          3,
          [&](unsigned int, size_t) { ++completed; },
          3,
          Pool::ASYNC,
          schedule,
          1);
      EXPECT_EQ(completed.load(), 3);
    }
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

// Expose existing protected owners only to this test; no production API change.
struct FractionalSplatTree : PoissonRecon::FEMTree<3, float> {
  using Base = PoissonRecon::FEMTree<3, float>;
  using Base::Base;
  using Base::_getSampleDepthAndWeight;
  using Base::_leaf;
  using Base::_localToGlobal;
  using Base::_setFullDepth;
  using Base::_splatPointData;
};

using SplatAddress = std::array<int, 4>;
using SplatCoefficients = std::map<SplatAddress, std::array<double, 3>>;
constexpr unsigned int kSplatSig = PoissonRecon::FEMDegreeAndBType<2>::Signature;
using SplatSigs = PoissonRecon::IsotropicUIntPack<3, kSplatSig>;
using SplatField =
    PoissonRecon::SparseNodeData<PoissonRecon::Point<float, 3>, SplatSigs>;

// Inspect the representation so finite checks survive finite-math assumptions.
bool IsFiniteSplatFloat(float value) {
  static_assert(sizeof(float) == 4 && sizeof(std::uint32_t) == 4);
  static_assert(std::numeric_limits<std::uint32_t>::digits == 32);
  static_assert(std::numeric_limits<float>::is_iec559 &&
                std::numeric_limits<float>::radix == 2 &&
                std::numeric_limits<float>::digits == 24 &&
                std::numeric_limits<float>::min_exponent == -125 &&
                std::numeric_limits<float>::max_exponent == 128);
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

// Closed-form centered quadratic cardinal basis, independent of FEM helpers.
double SplatB2(double x) {
  x = std::abs(x);
  if (x <= 0.5) return 0.75 - x * x;
  if (x < 1.5) return 0.5 * (1.5 - x) * (1.5 - x);
  return 0;
}

SplatCoefficients SplatSnapshot(const FractionalSplatTree& tree,
                                const SplatField& field) {
  SplatCoefficients result;
  tree.tree().processNodes([&](const FractionalSplatTree::FEMTreeNode* node) {
    if (node->nodeData.nodeIndex < 0) return;
    const auto* value = field(node);
    if (!value) return;
    int depth, offset[3];
    tree.depthAndOffset(node, depth, offset);
    result[{depth, offset[0], offset[1], offset[2]}] =
        {double((*value)[0]), double((*value)[1]), double((*value)[2])};
  });
  return result;
}

TEST(PoissonFEMTreeTest, FractionalSplatUsesEachLevelNodeAndWidth) {
  using namespace PoissonRecon;
  using Degrees = IsotropicUIntPack<3, 2>;
  FractionalSplatTree tree(1024);
  auto* allocator = tree.nodeAllocators.empty() ? nullptr : tree.nodeAllocators[0];
  tree._setFullDepth<false>(Degrees(), allocator, 4);
  const Point<float, 3> position(7.5f / 16, 8.5f / 16, 9.5f / 16);
  const Point<float, 3> value(1.f, 0.25f, -0.5f);
  FractionalSplatTree::PointSample sample{};
  sample.node = tree._leaf<true>(allocator, position, 4);
  sample.sample = ProjectiveData<Point<float, 3>, float>(position, 1.f);
  std::unique_ptr<FractionalSplatTree::DensityEstimator<2>> density(
      tree.setDensityEstimator<1, 2>({sample}, 3, 1.f));
  PointSupportKey<Degrees> weight_key, data_key;
  weight_key.set(tree._localToGlobal(4));
  data_key.set(tree._localToGlobal(4));
  float base_depth, base_weight;
  tree._getSampleDepthAndWeight(
      *density, position, weight_key, base_depth, base_weight);
  ASSERT_TRUE(IsFiniteSplatFloat(base_depth));
  ASSERT_TRUE(IsFiniteSplatFloat(base_weight));
  ASSERT_GT(base_weight, 0.f);
  const float bias = 3.5f - base_depth;

  Point<float, 2> fractional_metadata;
  // max=4 takes the fractional path; max=3 is an integer-depth clamp control.
  for (const int max_depth : {4, 3}) {
    SCOPED_TRACE(max_depth);
    SplatField fixed, functor;
    const auto fixed_metadata =
        tree._splatPointData<true, false, 2, Point<float, 3>,
                             kSplatSig, kSplatSig, kSplatSig>(
            Point<float, 3>(), allocator, *density, 0.f, position, value, fixed,
            weight_key, data_key, 0, max_depth, 3, bias);
    int callback_calls = 0;
    std::function<int(Point<float, 3>)> maximum = [&](Point<float, 3> p) {
      ++callback_calls;
      EXPECT_EQ(p[0], position[0]);
      EXPECT_EQ(p[1], position[1]);
      EXPECT_EQ(p[2], position[2]);
      return max_depth;
    };
    const auto functor_metadata =
        tree._splatPointData<true, false, 2, Point<float, 3>,
                             kSplatSig, kSplatSig, kSplatSig>(
            Point<float, 3>(), allocator, *density, 0.f, position, value, functor,
            weight_key, data_key, 0, maximum, 3, bias);
    EXPECT_EQ(callback_calls, 1);
    ASSERT_TRUE(IsFiniteSplatFloat(fixed_metadata[0]));
    ASSERT_TRUE(IsFiniteSplatFloat(fixed_metadata[1]));
    ASSERT_TRUE(IsFiniteSplatFloat(functor_metadata[0]));
    ASSERT_TRUE(IsFiniteSplatFloat(functor_metadata[1]));
    ASSERT_GT(fixed_metadata[0], 3.f);
    ASSERT_LT(fixed_metadata[0], 4.f);
    ASSERT_GT(fixed_metadata[1], 0.f);
    EXPECT_EQ(fixed_metadata[1], base_weight);
    EXPECT_EQ(fixed_metadata[0], functor_metadata[0]);
    EXPECT_EQ(fixed_metadata[1], functor_metadata[1]);
    if (max_depth == 4) {
      fractional_metadata = fixed_metadata;
    } else {
      EXPECT_EQ(fixed_metadata[0], fractional_metadata[0]);
      EXPECT_EQ(fixed_metadata[1], fractional_metadata[1]);
    }

    // Use ACTUAL raw metadata, not the requested depth: bias arithmetic can
    // round differently with compiler floating-point transformations.
    const double depth = std::min(double(max_depth), double(fixed_metadata[0]));
    const int fine = int(std::ceil(depth));
    const double alpha = 1.0 - (fine - depth);
    const bool has_coarse = std::abs(1.0 - alpha) > 1e-6;
    EXPECT_EQ(fine, max_depth);
    ASSERT_EQ(has_coarse, max_depth == 4);
    if (has_coarse) {
      ASSERT_GE(alpha, 0.1);
      ASSERT_LE(alpha, 0.9);
    } else {
      EXPECT_EQ(depth, 3.0);
      EXPECT_EQ(alpha, 1.0);
    }
    SplatCoefficients expected;
    for (int level = 0; level < (has_coarse ? 2 : 1); ++level) {
      const int d = fine - level;
      const double h = std::ldexp(1.0, -d);
      const double fraction = float(level ? 1.0 - alpha : alpha);
      int cell[3];
      for (int axis = 0; axis < 3; ++axis)
        cell[axis] = int(std::floor(double(position[axis]) / h));
      for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
          for (int x = -1; x <= 1; ++x) {
            const SplatAddress address{d, cell[0] + x, cell[1] + y, cell[2] + z};
            double basis = 1;
            for (int axis = 0; axis < 3; ++axis)
              basis *= SplatB2(double(position[axis]) / h -
                               (address[axis + 1] + 0.5));
            auto& coefficient = expected[address];
            for (int c = 0; c < 3; ++c)
              coefficient[c] = double(value[c]) * double(fixed_metadata[1]) *
                               fraction * basis / (h * h * h);
          }
    }
    const auto fixed_values = SplatSnapshot(tree, fixed);
    const auto functor_values = SplatSnapshot(tree, functor);
    // Snapshots only widen float coefficients; casting back is exact.
    for (const auto* values : {&fixed_values, &functor_values})
      for (const auto& item : *values)
        for (const double coefficient : item.second)
          EXPECT_TRUE(IsFiniteSplatFloat(float(coefficient)));
    EXPECT_EQ(fixed_values, functor_values);
    // Exact hex values retain metadata/control maps for a later A/B XML diff.
    std::ostringstream evidence;
    evidence << std::hexfloat << fixed_metadata[0] << ',' << fixed_metadata[1]
             << ";fine=" << fine << ";alpha=" << alpha
             << ";coarse=" << has_coarse;
    for (const auto& item : fixed_values) {
      for (const int address : item.first) evidence << ',' << address;
      for (const double coefficient : item.second) evidence << ',' << coefficient;
      evidence << ';';
    }
    RecordProperty(max_depth == 4 ? "fractional_contract" : "control_contract",
                   evidence.str());
    EXPECT_EQ(expected.size(), has_coarse ? 54u : 27u);
    EXPECT_EQ(fixed_values.size(), expected.size());
    for (const auto& item : expected) {
      const auto actual = fixed_values.find(item.first);
      EXPECT_NE(actual, fixed_values.end());
      if (actual == fixed_values.end()) continue;
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(actual->second[c], item.second[c],
                    64 * std::numeric_limits<float>::epsilon() *
                        std::max(1.0, std::abs(item.second[c])));
      }
    }
  }
}

}  // namespace
