// SPDX-License-Identifier: BSD-3-Clause

// FEMTree.h requires the vendor configuration before its declarations.
// clang-format off
#include "PreProcessor.h"
#include "FEMTree.h"
// clang-format on

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace pr = PoissonRecon;
using Point = pr::Point<float, 3>;
constexpr unsigned kDataSig =
    pr::FEMDegreeAndBType<0, pr::BOUNDARY_FREE>::Signature;
using DataSigs = pr::IsotropicUIntPack<3, kDataSig>;

struct ThreadSettings {
  ThreadSettings()
      : count(pr::ThreadPool::NumThreads()),
        parallel(pr::ThreadPool::ParallelizationType),
        schedule(pr::ThreadPool::Schedule),
        chunk(pr::ThreadPool::ChunkSize) {
    pr::ThreadPool::SetNumThreads(2);
    pr::ThreadPool::ParallelizationType = pr::ThreadPool::NONE;
    pr::ThreadPool::Schedule = pr::ThreadPool::STATIC;
    pr::ThreadPool::ChunkSize = 1;
  }
  ~ThreadSettings() {
    pr::ThreadPool::SetNumThreads(count);
    pr::ThreadPool::ParallelizationType = parallel;
    pr::ThreadPool::Schedule = schedule;
    pr::ThreadPool::ChunkSize = chunk;
  }
  unsigned count;
  pr::ThreadPool::ParallelType parallel;
  pr::ThreadPool::ScheduleType schedule;
  size_t chunk;
};

// The numeric payload is unchanged. Only its ordinary projective division is
// delayed, before GetIsoVertex returns and the real production CAS runs.
struct ScheduleGate {
  enum Event { kAEntered, kBProgress, kCPublished, kAReturned };
  explicit ScheduleGate(bool cross, bool enabled)
      : first(cross ? 26.0f : 14.0f),
        later(first + 1),
        sentinel(cross ? Point(1, 0, .375f) : Point(1, .375f, 0)),
        enabled(enabled) {}

  void Normalize(float value, float weight) {
    std::unique_lock<std::mutex> lock(mutex);
    weights_are_one = weights_are_one && weight == 1;
    normalized.push_back(value);
    if (!enabled) return;
    if (value == first && ++first_calls == 2) {
      entered = true;
      events.push_back(kAEntered);
      condition.notify_all();
      if (!condition.wait_for(
              lock, std::chrono::seconds(10), [&] { return published; })) {
        timed_out = true;
      }
      events.push_back(kAReturned);
    } else if (value == later) {
      if (!condition.wait_for(
              lock, std::chrono::seconds(10), [&] { return entered; })) {
        timed_out = true;
      }
      if (!later_progress) events.push_back(kBProgress);
      later_progress = true;
    }
  }

  void Publish(unsigned worker, const Point& point) {
    std::lock_guard<std::mutex> lock(mutex);
    workers.insert(worker);
    threads.insert(std::this_thread::get_id());
    if (enabled && point[0] == sentinel[0] && point[1] == sentinel[1] &&
        point[2] == sentinel[2]) {
      published = true;
      events.push_back(kCPublished);
      condition.notify_all();
    }
  }

  float first;
  float later;
  Point sentinel;
  bool enabled;
  std::mutex mutex;
  std::condition_variable condition;
  unsigned first_calls = 0;
  bool entered = false;
  bool published = false;
  bool later_progress = false;
  bool timed_out = false;
  bool weights_are_one = true;
  std::vector<Event> events;
  std::vector<float> normalized;
  std::set<unsigned> workers;
  std::set<std::thread::id> threads;
};

struct Payload {
  Payload() = default;
  explicit Payload(float value) : value(value) {}
  Payload& operator+=(Payload rhs) {
    value += rhs.value;
    return *this;
  }
  Payload& operator-=(Payload rhs) {
    value -= rhs.value;
    return *this;
  }
  Payload& operator*=(float rhs) {
    value *= rhs;
    return *this;
  }
  Payload& operator/=(float rhs) {
    value /= rhs;
    return *this;
  }
  friend Payload operator+(Payload lhs, Payload rhs) { return lhs += rhs; }
  friend Payload operator-(Payload lhs, Payload rhs) { return lhs -= rhs; }
  friend Payload operator*(Payload lhs, float rhs) { return lhs *= rhs; }
  friend Payload operator/(Payload lhs, float rhs) {
    if (gate != nullptr) gate->Normalize(lhs.value, rhs);
    return lhs /= rhs;
  }
  float value = 0;
  static inline ScheduleGate* gate = nullptr;
};

struct BindGate {
  explicit BindGate(ScheduleGate& value) : previous(Payload::gate) {
    Payload::gate = &value;
  }
  ~BindGate() { Payload::gate = previous; }
  ScheduleGate* previous;
};

using Extractor = pr::_LevelSetExtractor<true, float, 3, Payload>;
using Vertex = Extractor::Vertex;
using Data = pr::SparseNodeData<pr::ProjectiveData<Payload, float>, DataSigs>;

// Expose only setup information needed by the existing native index builders.
// The owner algorithm and its selection predicates remain production code.
struct FixtureTree : pr::FEMTree<3, float> {
  FixtureTree() : pr::FEMTree<3, float>(0) {
    spaceRoot().setFullDepth(2, nullptr, initializer());
    setSortedTreeNodes(std::make_tuple());
    _setFEM1ValidityFlags(DataSigs());
  }
  const pr::SortedTreeNodes<3>& Sorted() const { return _sNodes; }
  int GlobalDepth() const { return _localToGlobal(2); }
  int Inset() const { return _localInset(2); }
};

struct VertexStream {
  size_t write(unsigned worker, const Vertex& vertex) {
    size_t index;
    {
      std::lock_guard<std::mutex> lock(mutex);
      index = vertices.size();
      vertices.push_back(vertex);
    }
    if (Payload::gate != nullptr)
      Payload::gate->Publish(worker, vertex.get<0>());
    return index;
  }
  std::mutex mutex;
  std::vector<Vertex> vertices;
};

struct Stage {
  explicit Stage(bool cross) : cross(cross), slab(cross ? 1 : 0), slabs(3) {
    pr::LevelSetExtraction::SetHyperCubeTables<3>();
    pr::LevelSetExtraction::SetHyperCubeTables<2>();
    for (auto i = tree.nodesBegin(2); i < tree.nodesEnd(2); ++i) {
      int depth;
      int offset[3];
      tree.depthAndOffset(tree.node(i), depth, offset);
      const float value = 10 + offset[0] + 4 * offset[1] + 16 * offset[2];
      data[tree.node(i)] =
          pr::ProjectiveData<Payload, float>(Payload(value), 1);
    }
    InitSlice(slab);
    if (cross) {
      InitSlice(slab + 1);
      auto& values = slabs[2].xSliceValues(slab);
      values.cellIndices.set(
          tree.Sorted(), tree.GlobalDepth(), slab + tree.Inset());
      values.reset(slab);
      slabs[2].xSliceScratch(slab).reset(values.cellIndices);
    }
  }

  void InitSlice(int slice) {
    auto& values = slabs[2].sliceValues(slice);
    auto& scratch = slabs[2].sliceScratch(slice);
    values.cellIndices.set(
        tree.Sorted(), tree.GlobalDepth(), slice + tree.Inset());
    values.reset(slice, false);
    scratch.reset(values.cellIndices);
    std::vector<bool> initialized(values.cellIndices.counts[0], false);
    for (size_t j = 0; j < values.cellIndices.size(); ++j) {
      const auto i = values.cellIndices.nodeOffset + j;
      Point start;
      float width;
      tree.startAndWidth(i, start, width);
      const auto& indices = values.cellIndices.indices<0>(tree.node(i));
      float corners[4];
      for (unsigned c = 0; c < 4; ++c) {
        const auto* dirs =
            pr::LevelSetExtraction::HyperCubeTables<2, 0>::Directions[c];
        corners[c] = cross
                         ? static_cast<float>(slice) / 4
                         : start[1] + width * (dirs[1] == pr::HyperCube::FRONT);
        if (initialized[indices[c]] &&
            values.cornerValues[indices[c]] != corners[c]) {
          throw std::runtime_error("Analytic shared corner values disagree");
        }
        values.cornerValues[indices[c]] = corners[c];
        initialized[indices[c]] = true;
        scratch.cSet[indices[c]] = 1;
      }
      values.mcIndices[j] = pr::HyperCube::Cube<2>::MCIndex(corners, .375f);
    }
  }

  void Run(bool parallel, ScheduleGate& gate, bool with_data = true) {
    pr::ThreadPool::ParallelizationType =
        parallel ? pr::ThreadPool::ASYNC : pr::ThreadPool::NONE;
    pr::FEMIntegrator::PointEvaluator<DataSigs, pr::ZeroUIntPack<3>> evaluator(
        tree.maxDepth());
    pr::LevelSetExtraction::KeyGenerator<3> keys(tree.maxDepth());
    BindGate binding(gate);
    const Data* field = with_data ? &data : nullptr;
    if (cross) {
      Extractor::SetXSliceIsoVertices<2, kDataSig>(keys,
                                                   tree,
                                                   false,
                                                   false,
                                                   &evaluator,
                                                   nullptr,
                                                   field,
                                                   .375f,
                                                   2,
                                                   2,
                                                   slab,
                                                   .25f,
                                                   .5f,
                                                   stream,
                                                   slabs,
                                                   Payload());
    } else {
      Extractor::SetSliceIsoVertices<2, kDataSig>(keys,
                                                  tree,
                                                  false,
                                                  false,
                                                  &evaluator,
                                                  nullptr,
                                                  field,
                                                  .375f,
                                                  2,
                                                  2,
                                                  slab,
                                                  pr::HyperCube::BACK,
                                                  stream,
                                                  slabs,
                                                  Payload());
    }
    pr::ThreadPool::ParallelizationType = pr::ThreadPool::NONE;
  }

  bool cross;
  int slab;
  FixtureTree tree;
  Data data;
  std::vector<Extractor::SlabValues> slabs;
  VertexStream stream;
};

using Row = std::array<float, 4>;
std::vector<Row> Rows(const VertexStream& stream) {
  std::vector<Row> rows;
  for (const auto& vertex : stream.vertices) {
    const auto& p = vertex.get<0>();
    rows.push_back({p[0], p[1], p[2], vertex.get<3>().value});
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

class LevelSetColorOwner : public ::testing::TestWithParam<bool> {
 protected:
  ThreadSettings settings;
};

TEST_P(LevelSetColorOwner, ParallelClaimPreservesSerialCellValue) {
  const bool cross = GetParam();
  Stage serial(cross);
  ScheduleGate serial_gate(cross, false);
  serial.Run(false, serial_gate);
  ASSERT_EQ(serial.stream.vertices.size(), cross ? 25 : 5);
  ASSERT_EQ(serial_gate.normalized.size(), serial.stream.vertices.size());
  const Point target = cross ? Point(.25f, 0, .375f) : Point(.25f, .375f, 0);
  for (size_t i = 0; i < serial.stream.vertices.size(); ++i) {
    ASSERT_EQ(serial_gate.normalized[i],
              serial.stream.vertices[i].get<3>().value);
  }
  ASSERT_EQ(serial_gate.normalized[0], serial_gate.first);
  ASSERT_EQ(serial_gate.normalized[1], serial_gate.first);
  ASSERT_EQ(serial.stream.vertices[0].get<0>()[0], 0);
  for (unsigned axis = 0; axis < 3; ++axis) {
    ASSERT_EQ(serial.stream.vertices[1].get<0>()[axis], target[axis]);
  }

  // The native static scheduler assigns relative even/odd indices to workers
  // 0/1. A is first; B and independent C run on worker 1 in that order.
  const std::array<int, 3> indices =
      cross ? std::array<int, 3>{0, 1, 5} : std::array<int, 3>{2, 3, 7};
  ASSERT_EQ(serial.tree.nodesSize(2, serial.slab), 16);
  for (size_t j = 0; j < indices.size(); ++j) {
    int depth;
    int offset[3];
    const auto index = serial.tree.nodesBegin(2, serial.slab) + indices[j];
    serial.tree.depthAndOffset(serial.tree.node(index), depth, offset);
    ASSERT_EQ(depth, 2);
    ASSERT_EQ(offset[0], j == 2 ? 3 : j);
    ASSERT_EQ(offset[1], cross ? 0 : 1);
    ASSERT_EQ(offset[2], serial.slab);
    ASSERT_EQ(indices[j] % 2, j == 0 ? 0 : 1);
  }

  Stage parallel(cross);
  ScheduleGate gate(cross, true);
  parallel.Run(true, gate);
  ASSERT_FALSE(gate.timed_out)
      << "The schedule protocol failed, not the color assertion";
  EXPECT_TRUE(gate.weights_are_one);
  EXPECT_EQ(gate.workers, (std::set<unsigned>{0, 1}));
  EXPECT_EQ(gate.threads.size(), 2);
  EXPECT_EQ(gate.events,
            (std::vector<ScheduleGate::Event>{ScheduleGate::kAEntered,
                                              ScheduleGate::kBProgress,
                                              ScheduleGate::kCPublished,
                                              ScheduleGate::kAReturned}));
  size_t targets = 0;
  for (const auto& vertex : parallel.stream.vertices) {
    const auto& p = vertex.get<0>();
    if (p[0] == target[0] && p[1] == target[1] && p[2] == target[2]) {
      ++targets;
      EXPECT_EQ(vertex.get<3>().value, gate.first);
    }
  }
  EXPECT_EQ(targets, 1);
  EXPECT_EQ(Rows(parallel.stream), Rows(serial.stream));
  EXPECT_EQ(Payload::gate, nullptr);
}

TEST_P(LevelSetColorOwner, PreservesPreviouslyPublishedEdges) {
  Stage stage(GetParam());
  ScheduleGate first(GetParam(), false);
  stage.Run(false, first);
  ASSERT_EQ(stage.stream.vertices.size(), GetParam() ? 25 : 5);
  const auto before = Rows(stage.stream);
  for (size_t i = 0; i < stage.data.size(); ++i)
    stage.data[i].data.value += 100;
  ScheduleGate second(GetParam(), false);
  stage.Run(true, second);
  EXPECT_TRUE(second.normalized.empty());
  EXPECT_EQ(Rows(stage.stream), before);
}

TEST_P(LevelSetColorOwner, NullAuxiliaryFieldKeepsGeometry) {
  Stage serial(GetParam());
  ScheduleGate first(GetParam(), false);
  serial.Run(false, first, false);
  Stage parallel(GetParam());
  ScheduleGate second(GetParam(), false);
  parallel.Run(true, second, false);
  EXPECT_TRUE(first.normalized.empty());
  EXPECT_TRUE(second.normalized.empty());
  EXPECT_EQ(Rows(parallel.stream), Rows(serial.stream));
  EXPECT_EQ(parallel.stream.vertices.size(), GetParam() ? 25 : 5);
}

INSTANTIATE_TEST_SUITE_P(SliceAndCrossSlice,
                         LevelSetColorOwner,
                         ::testing::Values(false, true));
}  // namespace
