// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

// FEMTree requires configuration before its declarations.
// clang-format off
#include "PreProcessor.h"
#include "FEMTree.h"
// clang-format on
#include <gtest/gtest.h>

namespace {

using PoissonRecon::BOUNDARY_FREE;
using PoissonRecon::DirectSum;
using PoissonRecon::FEMDegreeAndBType;
using PoissonRecon::FEMTree;
using PoissonRecon::IsotropicUIntPack;
using PoissonRecon::Point;
using PoissonRecon::ProjectiveData;
using PoissonRecon::SparseNodeData;
using PoissonRecon::ThreadPool;
using Tree = FEMTree<3, float>;
using Node = Tree::FEMTreeNode;
using Color = Point<float, 3>;
using Data = DirectSum<float, Color>;
constexpr unsigned int kDataSig =
    FEMDegreeAndBType<0, BOUNDARY_FREE>::Signature;
using Field =
    SparseNodeData<ProjectiveData<Data, float>, IsotropicUIntPack<3, kDataSig>>;
constexpr size_t kSamples = 400000;

enum class Fault { kNone, kCapture, kVerification };

// Only this executable replaces allocation. The predicate observes the real
// destination, so it does not depend on STL allocation ordinals or sizes alone.
struct AllocationProbe {
  Field* field = nullptr;
  Fault fault = Fault::kNone;
  bool armed = false;
  unsigned int failures = 0;
};
thread_local AllocationProbe allocation;

void CheckAllocation(size_t size) {
  if (!allocation.armed || !allocation.field) return;
  const Field& field = *allocation.field;
  const bool first_written = field.size() >= 1 && field[0].weight == kSamples;
  const bool complete =
      first_written && field.size() == 2 && field[1].weight == 4 * kSamples;
  if ((allocation.fault == Fault::kCapture && complete &&
       size == 2 * sizeof(const void*)) ||
      (allocation.fault == Fault::kVerification && complete)) {
    allocation.armed = false;
    ++allocation.failures;
    throw std::bad_alloc();
  }
}

thread_local unsigned int reductions = 0;
thread_local bool invalid_output = false;

}  // namespace

void* operator new(std::size_t size) {
  CheckAllocation(size);
  if (void* result = std::malloc(size ? size : 1)) return result;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

namespace AuxNativeGpu {

// Test-only ABI shim: no CUDA implementation is linked into this executable.
bool Available(Stats&) { return true; }

bool Reduce(const View& input, Four* output, Stats&) {
  ++reductions;
  // A repeated sample in child zero has a closed-form two-node field. This
  // supplies known values, rather than copying the production reduction.
  if (input.nodes != 2 || input.seeds != kSamples || input.max_depth != 1 ||
      input.level_offsets[0] != 0 || input.level_offsets[1] != 1 ||
      input.level_offsets[2] != 2 || input.child_offsets[0] != 0 ||
      input.child_offsets[1] != 1 || input.child_offsets[2] != 1 ||
      input.children[0] != 1 || input.seed_offsets[0] != 0 ||
      input.seed_offsets[1] != 0 || input.seed_offsets[2] != kSamples) {
    throw std::runtime_error("unexpected two-node capture fixture");
  }
  for (unsigned int node = 0; node < 2; ++node) {
    const float weight =
        invalid_output ? 0.f : float(kSamples * (node ? 4 : 1));
    output[node] = {{weight, 2 * weight, 3 * weight, weight}};
  }
  allocation.armed = allocation.fault != Fault::kNone;
  return true;
}

}  // namespace AuxNativeGpu

namespace {

void SetMode(const char* value) {
#ifdef _WIN32
  if (_putenv_s("POISSON_GPU_AUX", value ? value : ""))
#else
  // This isolated test process runs serially without Poisson worker threads.
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  if (value ? setenv("POISSON_GPU_AUX", value, 1) : unsetenv("POISSON_GPU_AUX"))
#endif
    throw std::runtime_error("cannot set auxiliary test mode");
}

class AuxCaptureTest : public testing::Test {
 protected:
  void SetUp() override {
    // This isolated test process runs serially without Poisson worker threads.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (const char* mode = std::getenv("POISSON_GPU_AUX")) {
      had_mode_ = true;
      previous_mode_ = mode;
    }
    previous_threads_ = ThreadPool::NumThreads();
    previous_parallelization_ = ThreadPool::ParallelizationType;
    ThreadPool::SetNumThreads(1);
    ThreadPool::ParallelizationType = ThreadPool::NONE;
    SetMode("1");
    tree_ = std::make_unique<Tree>(1024);
    tree_->spaceRoot().template initChildren<false>(tree_->nodeAllocators[0],
                                                    tree_->initializer());
    sample_.node = tree_->spaceRoot().children;
    sample_.sample = ProjectiveData<Color, float>(Color(.25f, .25f, .25f), 1.f);
    datum_.template get<0>() = Color(1.f, 2.f, 3.f);
    zero_.template get<0>() = Color(0.f, 0.f, 0.f);
    reductions = 0;
    invalid_output = false;
  }

  void TearDown() override {
    allocation = {};
    invalid_output = false;
    tree_.reset();
    ThreadPool::SetNumThreads(previous_threads_);
    ThreadPool::ParallelizationType = previous_parallelization_;
    SetMode(had_mode_ ? previous_mode_.c_str() : nullptr);
  }

  void Update(Field& field, Fault fault = Fault::kNone, bool eligible = true) {
    allocation = {&field, fault, false, 0};
    struct Disarm {
      ~Disarm() { allocation.armed = false; }
    } disarm;
    AuxNativeInput::Scope input(eligible);
    sample_calls_ = 0;
    tree_->template updateExtrapolatedDataField<kDataSig, false, 2, Data>(
        zero_,
        field,
        kSamples,
        [&](size_t) -> const Tree::PointSample& {
          ++sample_calls_;
          return sample_;
        },
        [&](size_t) -> const Data& { return datum_; },
        nullptr,
        false);
  }

  bool Scale(Field& field, const AuxNativeScale::Scope& scope) {
    size_t scaled = 0;
    auto scale = [&](const Node* node) {
      if (auto* value = field(node)) {
        *value *= float(std::pow(32.f, tree_->depth(node)));
        ++scaled;
      }
    };
    const bool captured = scope.Apply<Node>(*tree_, field, scale);
    if (!captured) tree_->tree().processNodes(scale);
    EXPECT_EQ(scaled, 2);
    return captured;
  }

  void ExpectSameField(const Field& actual, const Field& expected) {
    ASSERT_EQ(actual.size(), 2);
    ASSERT_EQ(expected.size(), 2);
    for (const Node* node :
         {&tree_->spaceRoot(), tree_->spaceRoot().children}) {
      const auto* a = actual(node);
      const auto* b = expected(node);
      ASSERT_NE(a, nullptr);
      ASSERT_NE(b, nullptr);
      EXPECT_EQ(a->weight, b->weight);
      for (unsigned int c = 0; c < 3; ++c)
        EXPECT_EQ(a->data.template get<0>()[c], b->data.template get<0>()[c]);
    }
  }

  std::unique_ptr<Tree> tree_;
  Tree::PointSample sample_;
  Data zero_, datum_;
  size_t sample_calls_ = 0;

 private:
  unsigned int previous_threads_ = 0;
  decltype(ThreadPool::ParallelizationType) previous_parallelization_;
  bool had_mode_ = false;
  std::string previous_mode_;
};

TEST_F(AuxCaptureTest, CaptureRefusalPreservesFieldAndAllowsLaterReuse) {
  Field reference;
  AuxNativeScale::Scope scope(tree_.get(), true);
  Update(reference, Fault::kNone, false);
  ASSERT_EQ(reductions, 0);
  ASSERT_EQ(sample_calls_, kSamples);

  Field refused;
  ASSERT_NO_THROW(Update(refused, Fault::kCapture));
  ASSERT_EQ(allocation.failures, 1);
  ASSERT_EQ(reductions, 1);
  EXPECT_EQ(sample_calls_, kSamples);
  EXPECT_FALSE(scope.ready_);
  EXPECT_TRUE(scope.nodes_.empty());
  ExpectSameField(refused, reference);
  EXPECT_FALSE(Scale(refused, scope));
  EXPECT_FALSE(Scale(reference, scope));
  ExpectSameField(refused, reference);

  Field reused;
  ASSERT_NO_THROW(Update(reused));
  ASSERT_EQ(reductions, 2);
  EXPECT_EQ(sample_calls_, kSamples);
  ASSERT_TRUE(scope.ready_);
  ASSERT_EQ(scope.nodes_.size(), 2);
  EXPECT_TRUE(Scale(reused, scope));
  ExpectSameField(reused, reference);
}

TEST_F(AuxCaptureTest, VerificationAllocationFailureDoesNotReplayCpu) {
  AuxNativeScale::Scope scope(tree_.get(), true);
  SetMode("verify");
  Field partial;
  EXPECT_THROW(Update(partial, Fault::kVerification), std::bad_alloc);
  EXPECT_EQ(allocation.failures, 1);
  ASSERT_EQ(partial.size(), 2);
  EXPECT_EQ(partial[size_t(0)].weight, kSamples);
  EXPECT_EQ(partial[size_t(0)].data.template get<0>()[0], kSamples);
  EXPECT_EQ(sample_calls_, kSamples);
  EXPECT_EQ(reductions, 1);
  EXPECT_FALSE(scope.ready_);
  EXPECT_FALSE(AuxNativeGpu::native_reference);

  Field reused;
  ASSERT_NO_THROW(Update(reused));
  EXPECT_TRUE(scope.ready_);
  EXPECT_EQ(reused.size(), 2);
}

TEST_F(AuxCaptureTest, AlreadyReadyScopeRemainsTerminal) {
  AuxNativeScale::Scope scope(tree_.get(), true);
  Field first;
  ASSERT_NO_THROW(Update(first));
  ASSERT_TRUE(scope.ready_);
  const auto captured = scope.nodes_;
  Field second;
  EXPECT_THROW(Update(second), std::runtime_error);
  EXPECT_EQ(sample_calls_, kSamples);
  EXPECT_EQ(reductions, 2);
  EXPECT_TRUE(scope.ready_);
  EXPECT_EQ(scope.nodes_, captured);
  ExpectSameField(second, first);
}

TEST_F(AuxCaptureTest, VerificationFailureDoesNotReplayOrPublish) {
  SetMode("verify");
  AuxNativeScale::Scope scope(tree_.get(), true);
  invalid_output = true;
  Field field;
  EXPECT_THROW(Update(field), std::runtime_error);
  EXPECT_EQ(reductions, 1);
  // One packing pass and the explicit verification reference pass only.
  EXPECT_EQ(sample_calls_, 2 * kSamples);
  EXPECT_FALSE(AuxNativeGpu::native_reference);
  EXPECT_FALSE(scope.ready_);
  EXPECT_TRUE(scope.nodes_.empty());
  ASSERT_EQ(field.size(), 2);
  EXPECT_EQ(field[size_t(0)].weight, 0);
  EXPECT_EQ(field[1].weight, 0);
}

}  // namespace
