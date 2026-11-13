// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/packed_consistency_mask.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_hmm_checkpoint.h"
#include "colmap/mvs/patch_match_io.h"
#include "colmap/mvs/patch_match_rng_perimeter.h"
#include "colmap/mvs/patch_match_sampling.cuh"
#include "colmap/mvs/patch_match_source_array.h"
#include "colmap/mvs/workspace.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#ifndef COLMAP_HIP_ENABLED
#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/patch_match_cuda.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#ifdef __linux__
#include <cstdio>
#include <csignal>

#include <sys/resource.h>
#include <unistd.h>
#endif

// Selection is process-global because controller work runs on a pool thread.
// The production cleanup observer itself remains thread-local.
#ifdef COLMAP_TEST_CUDA_ALLOCATION_WRAPPING
#include <atomic>
namespace colmap::mvs::internal {
bool IsPatchMatchWorkStreamConstructionActive() noexcept __attribute__((weak));
}
namespace {
enum class StreamFault {
  kNone,
  kRefuse,
  kPrior,
  kOther,
  kMismatch,
  kFailedNonNull,
  kSuccessNull,
  kDestroy
};
enum class StreamEventKind { kCreated, kRefused, kDestroyed, kQuery };
struct StreamEvent {
  StreamEventKind kind;
  cudaStream_t stream;
  size_t ordinal;
};
// Fixed storage keeps wrapped cleanup calls allocation-free, including while
// an exception is unwinding. Tests inspect it after the worker has joined.
std::array<StreamEvent, 256> stream_events;
std::atomic<size_t> stream_event_count{0}, stream_attempts{0};
std::atomic<size_t> stream_fail_first{0}, stream_fail_second{0};
std::atomic<size_t> stream_fired{0}, stream_destroy_failures{0};
std::atomic<StreamFault> stream_fault{StreamFault::kNone};
std::atomic<bool> stream_capture{false};
thread_local bool stream_worker_seen = false;
bool InWorkStreamConstruction() {
  return colmap::mvs::internal::IsPatchMatchWorkStreamConstructionActive &&
         colmap::mvs::internal::IsPatchMatchWorkStreamConstructionActive();
}
void RecordStreamEvent(StreamEventKind kind,
                       cudaStream_t stream,
                       size_t ordinal) {
  const size_t index = stream_event_count.fetch_add(1);
  if (index < stream_events.size())
    stream_events[index] = {kind, stream, ordinal};
}
void ResetStreamFault(StreamFault fault,
                      size_t first,
                      size_t second = 0,
                      bool capture_only = false) {
  stream_capture = false;
  stream_event_count = 0;
  stream_attempts = 0;
  stream_fired = 0;
  stream_destroy_failures = 0;
  stream_fail_first = first;
  stream_fail_second = second;
  stream_fault = fault;
  stream_capture = capture_only || fault != StreamFault::kNone;
}
std::atomic<size_t> wrapped_eligible{0}, wrapped_fail_at{0};
std::atomic<bool> wrapped_fired{false}, wrapped_fail_cleanup{false};
std::atomic<size_t> wrapped_cleanup_injected{0};
std::atomic<bool> wrapped_arm_on_first_meminfo{false};
std::atomic<size_t> wrapped_meminfo_calls{0};
std::atomic<bool> wrapped_track_initial_dense{false};
std::atomic<size_t> wrapped_dense_captures{0},
    wrapped_dense_cleanup_injected{0};
thread_local bool wrapped_worker_armed = false;
thread_local cudaArray_t wrapped_initial_dense = nullptr;
std::atomic<int> wrapped_status{static_cast<int>(cudaErrorMemoryAllocation)};
bool InjectAllocation() {
  if (wrapped_arm_on_first_meminfo && !wrapped_worker_armed) return false;
  if (!colmap::CudaCleanupScope::ObserveCleanup("test probe", 0)) return false;
  const size_t ordinal = ++wrapped_eligible;
  const size_t selected = wrapped_fail_at.load();
  if (selected == 0 || ordinal != selected) return false;
  wrapped_fired = true;
  return true;
}
cudaError_t MaybeInjectCleanup(cudaError_t actual) {
  if (actual != cudaSuccess || !wrapped_fired || !wrapped_fail_cleanup ||
      !colmap::CudaCleanupScope::ObserveCleanup("test probe", 0))
    return actual;
  bool expected = true;
  if (!wrapped_fail_cleanup.compare_exchange_strong(expected, false))
    return actual;
  ++wrapped_cleanup_injected;
  // Real release already succeeded. Caller observes a terminal failure and
  // must not readmit; run this case in a fresh subprocess and exit promptly.
  return cudaErrorUnknown;
}
void ResetAllocationInjection(size_t ordinal,
                              cudaError_t status,
                              bool cleanup_failure = false,
                              bool arm_on_first_meminfo = false) {
  wrapped_arm_on_first_meminfo = arm_on_first_meminfo;
  wrapped_track_initial_dense = false;
  wrapped_dense_captures = 0;
  wrapped_dense_cleanup_injected = 0;
  wrapped_meminfo_calls = 0;
  wrapped_eligible = 0;
  wrapped_fired = false;
  wrapped_cleanup_injected = 0;
  wrapped_status = static_cast<int>(status);
  wrapped_fail_cleanup = cleanup_failure;
  wrapped_fail_at = ordinal;
}
}  // namespace
extern "C" cudaError_t __real_cudaMalloc(void**, size_t);
extern "C" cudaError_t __real_cudaMallocPitch(void**, size_t*, size_t, size_t);
extern "C" cudaError_t __real_cudaMalloc3DArray(cudaArray_t*,
                                                const cudaChannelFormatDesc*,
                                                cudaExtent,
                                                unsigned int);
extern "C" cudaError_t __real_cudaMemGetInfo(size_t*, size_t*);
extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __real_cudaFreeArray(cudaArray_t);
extern "C" cudaError_t __real_cudaStreamCreateWithFlags(cudaStream_t*,
                                                        unsigned int);
extern "C" cudaError_t __real_cudaStreamDestroy(cudaStream_t);
extern "C" cudaError_t __real_cudaPeekAtLastError();
extern "C" cudaError_t __real_cudaGetLastError();
extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* out,
                                                        unsigned int flags) {
  if (!stream_capture || !InWorkStreamConstruction())
    return __real_cudaStreamCreateWithFlags(out, flags);
  stream_worker_seen = true;
  const size_t ordinal = ++stream_attempts;
  const bool selected =
      ordinal == stream_fail_first || ordinal == stream_fail_second;
  if (!selected) {
    const auto status = __real_cudaStreamCreateWithFlags(out, flags);
    if (status == cudaSuccess)
      RecordStreamEvent(StreamEventKind::kCreated, *out, ordinal);
    return status;
  }
  ++stream_fired;
  const auto mode = stream_fault.load();
  if (mode == StreamFault::kFailedNonNull) {
    const auto status = __real_cudaStreamCreateWithFlags(out, flags);
    if (status != cudaSuccess) return status;
    RecordStreamEvent(StreamEventKind::kCreated, *out, ordinal);
  } else {
    *out = nullptr;
  }
  RecordStreamEvent(StreamEventKind::kRefused, *out, ordinal);
  if (mode == StreamFault::kSuccessNull) return cudaSuccess;
  return mode == StreamFault::kOther ? cudaErrorInvalidValue
                                     : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t __wrap_cudaStreamDestroy(cudaStream_t stream) {
  const bool partial = stream_capture && InWorkStreamConstruction();
  const auto status = __real_cudaStreamDestroy(stream);
  if (!partial) return status;
  RecordStreamEvent(
      StreamEventKind::kDestroyed, stream, stream_attempts.load());
  if (status == cudaSuccess && stream_fault == StreamFault::kDestroy &&
      stream_fired && stream_destroy_failures.fetch_add(1) == 0)
    return cudaErrorUnknown;  // The real handle was released before injection.
  return status;
}
extern "C" cudaError_t __wrap_cudaPeekAtLastError() {
  if (stream_capture && InWorkStreamConstruction() &&
      stream_fault == StreamFault::kPrior && !stream_fired &&
      stream_attempts.load() + 1 == stream_fail_first) {
    stream_worker_seen = true;
    ++stream_fired;
    RecordStreamEvent(StreamEventKind::kRefused, nullptr, stream_fail_first);
    return cudaErrorLaunchFailure;
  }
  return __real_cudaPeekAtLastError();
}
extern "C" cudaError_t __wrap_cudaGetLastError() {
  const auto actual = __real_cudaGetLastError();
  if (stream_capture && InWorkStreamConstruction() && stream_fired &&
      stream_fault == StreamFault::kMismatch && actual == cudaSuccess)
    return cudaErrorInvalidValue;
  return actual;
}
extern "C" cudaError_t __wrap_cudaMalloc(void** out, size_t size) {
  if (InjectAllocation()) {
    *out = nullptr;
    return static_cast<cudaError_t>(wrapped_status.load());
  }
  return __real_cudaMalloc(out, size);
}
extern "C" cudaError_t __wrap_cudaMallocPitch(void** out,
                                              size_t* pitch,
                                              size_t width,
                                              size_t height) {
  if (InjectAllocation()) {
    *out = nullptr;
    return static_cast<cudaError_t>(wrapped_status.load());
  }
  return __real_cudaMallocPitch(out, pitch, width, height);
}
extern "C" cudaError_t __wrap_cudaMalloc3DArray(
    cudaArray_t* out,
    const cudaChannelFormatDesc* desc,
    cudaExtent extent,
    unsigned int flags) {
  if (InjectAllocation()) {
    *out = nullptr;
    return static_cast<cudaError_t>(wrapped_status.load());
  }
  const cudaError_t status = __real_cudaMalloc3DArray(out, desc, extent, flags);
  // The five-view photo fixture has four source layers; its first such
  // allocation in the first constructor is the initial dense source array.
  if (status == cudaSuccess && wrapped_track_initial_dense &&
      !wrapped_worker_armed && wrapped_initial_dense == nullptr &&
      extent.depth == 4 && desc->x == 8 &&
      colmap::CudaCleanupScope::ObserveCleanup("test probe", 0)) {
    wrapped_initial_dense = *out;
    ++wrapped_dense_captures;
  }
  return status;
}
extern "C" cudaError_t __wrap_cudaMemGetInfo(size_t* free, size_t* total) {
  if (stream_capture && stream_worker_seen && stream_fired)
    RecordStreamEvent(StreamEventKind::kQuery, nullptr, stream_attempts.load());
  const cudaError_t status = __real_cudaMemGetInfo(free, total);
  if (status == cudaSuccess) {
    ++wrapped_meminfo_calls;
    if (wrapped_arm_on_first_meminfo) wrapped_worker_armed = true;
  }
  return status;
}
extern "C" cudaError_t __wrap_cudaFree(void* ptr) {
  const cudaError_t actual = __real_cudaFree(ptr);
  return ptr ? MaybeInjectCleanup(actual) : actual;
}
extern "C" cudaError_t __wrap_cudaFreeArray(cudaArray_t ptr) {
  const cudaError_t actual = __real_cudaFreeArray(ptr);
  if (actual == cudaSuccess && ptr && wrapped_track_initial_dense &&
      wrapped_worker_armed && ptr == wrapped_initial_dense) {
    wrapped_track_initial_dense = false;
    wrapped_initial_dense = nullptr;
    ++wrapped_dense_cleanup_injected;
    return cudaErrorUnknown;
  }
  return ptr ? MaybeInjectCleanup(actual) : actual;
}
#endif

namespace colmap {
namespace mvs {
#ifndef COLMAP_HIP_ENABLED
namespace internal {
float ProbePropagatedDepth(const std::array<float, 4>& K,
                           float depth,
                           const std::array<float, 3>& normal,
                           float row1,
                           float row2,
                           float col);
std::pair<std::vector<int>, std::vector<float>> ProbeSampledSources(
    const std::vector<int>& draws,
    const std::vector<float>& values,
    int sample_window_size,
    bool cooperative);
}  // namespace internal
#endif
TEST(PatchMatchCuda, SourceSamplingPreservesSupportAndStrides) {
  for (const int stride : {1, 8, 32}) {
    std::vector<float> probs(3 * stride, -7.0f);
    probs[0] = probs[stride] = probs[2 * stride] = 0.0f;
    EXPECT_FALSE(TransformPDFToCDF(probs.data(), 3, stride));
    for (const float draw : {std::numeric_limits<float>::denorm_min(),
                             FLT_EPSILON / 2,
                             FLT_EPSILON,
                             0.5f,
                             1.0f}) {
      EXPECT_EQ(SampleSource(probs.data(), 3, stride, draw), -1);
    }
    probs[stride] = 1.0f;
    probs[2 * stride] = 3.0f;
    EXPECT_TRUE(TransformPDFToCDF(probs.data(), 3, stride));
    EXPECT_EQ(probs[0], 0.0f);
    EXPECT_EQ(probs[stride], 0.25f);
    EXPECT_EQ(probs[2 * stride], 1.0f);
    EXPECT_EQ(SampleSource(probs.data(), 3, stride, FLT_EPSILON / 2), 1);
    EXPECT_EQ(SampleSource(probs.data(), 3, stride, 0.5f), 2);
    EXPECT_EQ(SampleSource(probs.data(), 3, stride, 1.0f), 2);
    for (size_t i = 0; i < probs.size(); ++i) {
      if (i % stride != 0) EXPECT_EQ(probs[i], -7.0f);
    }
  }
  // Invalid distributions keep the old normalization/selection policy; they
  // must not masquerade as validated all-zero support.
  for (const auto pdf : {std::array<float, 2>{-1.0f, 1.0f},
                         {std::numeric_limits<float>::quiet_NaN(), 0.0f},
                         {std::numeric_limits<float>::infinity(), 0.0f}}) {
    auto cdf = pdf;
    EXPECT_TRUE(TransformPDFToCDF(cdf.data(), 2, 1));
    EXPECT_EQ(SampleSource(cdf.data(), 2, 1, 0.5f), -1);
  }
}

TEST(PatchMatchCuda, SourceSamplingDirectedBoundaries) {
  EXPECT_FALSE(TransformPDFToCDF(nullptr, 0, 32));
  EXPECT_EQ(SampleSource(nullptr, 0, 32, 1.0f), -1);
  for (const int stride : {1, 8, 32}) {
    std::vector<float> probs(5 * stride, -7.0f);
    for (int i = 0; i < 5; ++i) probs[i * stride] = i % 2 ? -0.0f : 0.0f;
    const auto zeros = probs;
    EXPECT_FALSE(TransformPDFToCDF(probs.data(), 5, stride));
    EXPECT_EQ(
        std::memcmp(probs.data(), zeros.data(), probs.size() * sizeof(float)),
        0);
    probs[stride] = 1.0f;
    probs[3 * stride] = 3.0f;
    ASSERT_TRUE(TransformPDFToCDF(probs.data(), 5, stride));
    EXPECT_EQ(probs[0], 0.0f);
    EXPECT_EQ(probs[stride], 0.25f);
    EXPECT_EQ(probs[2 * stride], 0.25f);
    EXPECT_EQ(probs[3 * stride], 1.0f);
    EXPECT_EQ(probs[4 * stride], 1.0f);
    for (float u : {std::numeric_limits<float>::denorm_min(),
                    FLT_EPSILON / 2,
                    std::nextafter(FLT_EPSILON, 0.0f),
                    FLT_EPSILON,
                    std::nextafter(FLT_EPSILON, 1.0f)}) {
      EXPECT_EQ(SampleSource(probs.data(), 5, stride, u), 1);
    }
    EXPECT_EQ(SampleSource(probs.data(), 5, stride, .25f + FLT_EPSILON), 3);
    EXPECT_EQ(SampleSource(probs.data(), 5, stride, 1.0f), 3);
    for (int j = 0; j < 257; ++j) {
      const float u = float(j + 1) / 257.0f;
      const float old_threshold = u - FLT_EPSILON;
      int old_source = -1;
      for (int i = 0; i < 5; ++i) {
        if (probs[i * stride] > old_threshold) {
          old_source = i;
          break;
        }
      }
      EXPECT_EQ(SampleSource(probs.data(), 5, stride, u), old_source);
    }
    for (size_t i = 0; i < probs.size(); ++i) {
      if (i % stride != 0) EXPECT_EQ(probs[i], -7.0f);
    }
  }
  // Retain legacy classification and selection; these are not valid CDFs.
  float mixed[] = {-1.0f, 1.0f};
  ASSERT_TRUE(TransformPDFToCDF(mixed, 2, 1));
  EXPECT_TRUE(std::isinf(mixed[0]));
  EXPECT_TRUE(std::signbit(mixed[0]));
  EXPECT_TRUE(std::isnan(mixed[1]));
  EXPECT_EQ(SampleSource(mixed, 2, 1, .5f), -1);
  float reversed[] = {1.0f, -1.0f};
  ASSERT_TRUE(TransformPDFToCDF(reversed, 2, 1));
  EXPECT_TRUE(std::isinf(reversed[0]));
  EXPECT_FALSE(std::signbit(reversed[0]));
  EXPECT_TRUE(std::isnan(reversed[1]));
  EXPECT_EQ(SampleSource(reversed, 2, 1, .5f), 0);
  for (float first : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity()}) {
    float invalid[] = {first, 0.0f};
    ASSERT_TRUE(TransformPDFToCDF(invalid, 2, 1));
    EXPECT_TRUE(std::isnan(invalid[0]));
    EXPECT_TRUE(std::isnan(invalid[1]));
    EXPECT_EQ(SampleSource(invalid, 2, 1, .5f), -1);
  }
  float negative[] = {-1.0f, -3.0f};
  ASSERT_TRUE(TransformPDFToCDF(negative, 2, 1));
  EXPECT_EQ(negative[0], .25f);
  EXPECT_EQ(negative[1], 1.0f);
  EXPECT_EQ(SampleSource(negative, 2, 1, .5f), 1);
}

class PatchMatchControllerTestAccess {
 public:
  using Hooks = PatchMatchController::SparseTestHooks;
  using Prepared = PatchMatchController::PreparedProblem;
  using ConfigProblem = std::pair<std::string, std::vector<std::string>>;
  static std::vector<ConfigProblem> ReadConfig(PatchMatchController& controller) {
    controller.ReadWorkspace();
    controller.ReadProblems();
    const auto& model = controller.workspace_->GetModel();
    std::vector<ConfigProblem> result;
    for (const auto& problem : controller.problems_) {
      ConfigProblem entry{model.GetImageName(problem.ref_image_idx), {}};
      for (const auto source : problem.src_image_idxs)
        entry.second.push_back(model.GetImageName(source));
      result.push_back(std::move(entry));
    }
    return result;
  }
  static void Run(PatchMatchController& controller, Hooks& hooks) {
    controller.sparse_test_hooks_ = &hooks;
    controller.Run();
    controller.sparse_test_hooks_ = nullptr;
  }
#ifndef COLMAP_HIP_ENABLED
  static std::unique_ptr<Prepared> Prepare(PatchMatchController& controller,
                                           const PatchMatchOptions& options,
                                           size_t index = 0) {
    if (!controller.workspace_) {
      controller.ReadWorkspace();
      controller.ReadProblems();
      controller.ReadGpuIndices();
      controller.thread_pool_ = std::make_unique<ThreadPool>(1);
      controller.io_thread_pool_ = std::make_unique<ThreadPool>(2);
    }
    auto prepared = std::make_unique<Prepared>();
    bool ready = false;
    RunPatchMatchTasks(*controller.thread_pool_, [&](auto& futures) {
      futures.push_back(controller.thread_pool_->AddTask([&] {
        ready = controller.PrepareProblem(options, index, prepared.get());
      }));
    });
    return ready ? std::move(prepared) : nullptr;
  }
  static void Publish(
      const Prepared& prepared,
      const std::function<void(const std::filesystem::path&)>& write_output) {
    PatchMatchController::PublishOutput(prepared, write_output);
  }
  static ThreadPool& InitializeScheduler(PatchMatchController& controller) {
    controller.problems_.resize(1);
    controller.problems_.front().ref_image_idx = 42;
    controller.thread_pool_ = std::make_unique<ThreadPool>(2);
    return *controller.thread_pool_;
  }
  static int ScheduledReference(const PatchMatchController& controller) {
    return controller.problems_.front().ref_image_idx;
  }
  static void RunPair(PatchMatchController& controller,
                      Hooks& hooks,
                      const PatchMatchOptions& options,
                      size_t first,
                      size_t second) {
    controller.ReadWorkspace();
    controller.ReadProblems();
    controller.ReadGpuIndices();
    controller.thread_pool_ = std::make_unique<ThreadPool>(1);
    controller.io_thread_pool_ = std::make_unique<ThreadPool>(2);
    controller.sparse_test_hooks_ = &hooks;
    try {
      controller.thread_pool_->AddTask(
          &PatchMatchController::ProcessProblemPair,
          &controller,
          options,
          first,
          second);
      controller.thread_pool_->Wait();
    } catch (const AggregateException& errors) {
      controller.sparse_test_hooks_ = nullptr;
      if (errors.exceptions().size() == 1)
        std::rethrow_exception(errors.exceptions().front());
      throw;
    } catch (...) {
      controller.sparse_test_hooks_ = nullptr;
      throw;
    }
    controller.sparse_test_hooks_ = nullptr;
  }
  static void RunSingles(PatchMatchController& controller,
                         Hooks& hooks,
                         const PatchMatchOptions& options,
                         size_t count) {
    controller.ReadWorkspace();
    controller.ReadProblems();
    controller.ReadGpuIndices();
    controller.thread_pool_ = std::make_unique<ThreadPool>(1);
    controller.io_thread_pool_ = std::make_unique<ThreadPool>(2);
    controller.sparse_test_hooks_ = &hooks;
    try {
      for (size_t index = 0; index < count; ++index) {
        controller.thread_pool_->AddTask(
            &PatchMatchController::ProcessProblem, &controller, options, index);
        controller.thread_pool_->Wait();
      }
    } catch (...) {
      controller.sparse_test_hooks_ = nullptr;
      throw;
    }
    controller.sparse_test_hooks_ = nullptr;
  }
  struct PreparedSources {
    std::vector<int> local;
    std::vector<int> shared;
    int min_num_consistent;
  };
  static std::array<PreparedSources, 2> PrepareSourceOrders(
      PatchMatchController& controller, const int missing_source) {
    controller.ReadWorkspace();
    controller.ReadProblems();
    controller.gpu_indices_ = {-1};
    controller.thread_pool_ = std::make_unique<ThreadPool>(1);
    controller.io_thread_pool_ = std::make_unique<ThreadPool>(2);
    auto& problem = controller.problems_.at(0);
    problem.ref_image_idx = 0;
    problem.src_image_idxs = {3, 1, 3, 0, 4, 2, 1};
    if (missing_source >= 0) {
      const auto path =
          controller.workspace_path_ / "images" /
          controller.workspace_->GetModel().GetImageName(missing_source);
      THROW_CHECK(std::filesystem::remove(path));
    }
    std::array<PreparedSources, 2> results;
    for (auto& result : results) {
      result = controller.thread_pool_
                   ->AddTask([&]() {
                     PatchMatchController::PreparedProblem prepared;
                     THROW_CHECK(controller.PrepareProblem(
                         controller.options_, 0, &prepared));
                     return PreparedSources{
                         prepared.problem.src_image_idxs,
                         controller.problems_.at(0).src_image_idxs,
                         prepared.options.filter_min_num_consistent};
                   })
                   .get();
      controller.thread_pool_->Wait();
    }
    return results;
  }
  static std::array<size_t, 3> PreparedBitmapCounts(
      PatchMatchController& controller, const bool retain_sources) {
    controller.ReadWorkspace();
    controller.ReadProblems();
    controller.ReadGpuIndices();
    controller.thread_pool_ = std::make_unique<ThreadPool>(1);
    controller.io_thread_pool_ = std::make_unique<ThreadPool>(2);
    return controller.thread_pool_
        ->AddTask([&]() {
          PatchMatchController::PreparedProblem prepared;
          auto photo_options = controller.options_;
          photo_options.geom_consistency = false;
          photo_options.filter = false;
          if (!controller.PrepareProblem(
                  photo_options, 0, &prepared, retain_sources)) {
            return std::array<size_t, 3>{0, 0, 0};
          }
          std::array<size_t, 3> counts{0, 0, 0};
          counts[0] = !prepared.images.at(prepared.problem.ref_image_idx)
                           .GetBitmap()
                           .IsEmpty();
          for (const int source : prepared.problem.src_image_idxs) {
            if (!prepared.source_bitmap_leases.empty() &&
                prepared.source_bitmap_leases.at(source)) {
              ++counts[1];
            }
            if (prepared.images.at(source).GetBitmap().IsEmpty()) ++counts[2];
          }
          return counts;
        })
        .get();
  }
#endif
};
namespace {

#ifdef COLMAP_HIP_ENABLED
TEST(PatchMatchCuda, RotationBufferReuseRequiresCuda) {
  GTEST_SKIP() << "The rotation-buffer option is CUDA-only";
}
#else

TEST(PatchMatchController, SubmissionFailureDrainsBorrowingWorkers) {
  struct FailingSubmission {
    explicit FailingSubmission(std::promise<void>& fault) : fault(&fault) {}
    FailingSubmission(const FailingSubmission& other) : fault(other.fault) {
      fault->set_value();
      throw std::bad_alloc();
    }
    void operator()() const {}
    std::promise<void>* fault;
  };
  std::promise<void> borrower_started, worker_failed, submission_failed;
  auto started = borrower_started.get_future();
  auto failed_worker = worker_failed.get_future();
  auto failed_submission = submission_failed.get_future();
  std::promise<void> release_borrower;
  const auto release = release_borrower.get_future().share();
  std::atomic<bool> saw_live_controller{false}, borrower_completed{false};
  std::atomic<bool> completed_before_rethrow{false};

  auto caller = std::async(std::launch::async, [&]() {
    auto controller = std::make_shared<PatchMatchController>(
        PatchMatchOptions{}, "", "COLMAP", "");
    auto& pool = PatchMatchControllerTestAccess::InitializeScheduler(*controller);
    std::weak_ptr<PatchMatchController> borrowed = controller;
    try {
      // Exercise the real controller's borrowed state and the shared owning
      // drain, without invoking GPU work or Run's scheduling decisions.
      RunPatchMatchTasks(pool, [&](auto&) {
        pool.AddTask([&, borrowed]() {
          borrower_started.set_value();
          release.wait();
          const auto owner = borrowed.lock();
          saw_live_controller =
              owner &&
              PatchMatchControllerTestAccess::ScheduledReference(*owner) == 42;
          borrower_completed = true;
        });
        pool.AddTask([&]() {
          worker_failed.set_value();
          throw std::runtime_error("independent worker failure");
        });
        if (started.wait_for(std::chrono::seconds(5)) !=
                std::future_status::ready ||
            failed_worker.wait_for(std::chrono::seconds(5)) !=
                std::future_status::ready) {
          throw std::runtime_error("accepted workers did not start");
        }
        FailingSubmission task(submission_failed);
        pool.AddTask(task);
      });
    } catch (const std::bad_alloc&) {
      completed_before_rethrow = borrower_completed.load();
      // Also keep a regressed helper safe to test: record its premature
      // rethrow before draining anything the helper may have left behind.
      try {
        pool.Wait();
      } catch (...) {
        return false;
      }
      return true;
    }
    return false;
  });

  const auto fault_status =
      failed_submission.wait_for(std::chrono::seconds(5));
  EXPECT_EQ(fault_status, std::future_status::ready);
  if (fault_status == std::future_status::ready) {
    EXPECT_EQ(caller.wait_for(std::chrono::milliseconds(20)),
              std::future_status::timeout);
  }
  release_borrower.set_value();
  EXPECT_TRUE(caller.get());
  EXPECT_TRUE(saw_live_controller);
  EXPECT_TRUE(completed_before_rethrow);
}

TEST(PatchMatchCuda, PackedConsistencyMaskDepthCoversSourceTails) {
  for (const auto& [sources, expected] :
       std::array<std::pair<std::size_t, std::size_t>, 8>{{{0, 0},
                                                           {1, 1},
                                                           {7, 1},
                                                           {8, 1},
                                                           {9, 2},
                                                           {17, 3},
                                                           {32, 4},
                                                           {33, 5}}}) {
    EXPECT_EQ(PackedConsistencyMaskDepth(sources), expected) << sources;
  }
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(PackedConsistencyMaskDepth(kMax), kMax / 8 + 1);
}

TEST(PatchMatchCuda, HostSlicesRejectInvalidDimensionsBeforeCuda) {
  using Texture = CudaArrayLayeredTexture<float>;
  const cudaTextureDesc desc = {};
  const float value = 1.0f;
  std::vector<Texture::HostSlice> slices{{&value, 1, 1, sizeof(float)}};
  EXPECT_THROW(Texture::FromHostSlices(desc, 0, 1, slices), std::exception);
  EXPECT_THROW(Texture::FromHostSlices(desc, 1, 0, slices), std::exception);
  EXPECT_THROW(Texture::FromHostSlices(
                   desc, std::numeric_limits<size_t>::max(), 1, slices),
               std::exception);
  slices[0].pitch_bytes = 0;
  EXPECT_THROW(Texture::FromHostSlices(desc, 1, 1, slices), std::exception);
}

TEST(PatchMatchCuda, CheckpointHmmStripeRowsRoundsUpWithoutOverflow) {
  const std::array<std::size_t, 17> heights = {
      1,
      2,
      3,
      4,
      5,
      8,
      9,
      10,
      17,
      35,
      47,
      129,
      1024,
      1536,
      2048,
      3072,
      std::numeric_limits<std::size_t>::max()};
  for (const std::size_t height : heights) {
    SCOPED_TRACE(height);
    const std::size_t rows = PatchMatchHmmCheckpoint::StripeRows(height);
    ASSERT_GT(rows, 0);
    // Characterize ceil(sqrt(height)) without squaring and overflowing.
    EXPECT_GE(rows, height / rows + (height % rows != 0));
    if (rows > 1) {
      const std::size_t previous = rows - 1;
      EXPECT_LT(previous, height / previous + (height % previous != 0));
    }
  }
}

TEST(PatchMatchCuda, CheckpointHmmRejectsInvalidAllocationShapes) {
  // Every rejection occurs before attempting a device allocation.
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint::StripeRows(0));
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint(0, 1, 1));
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint(1, 0, 1));
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint(
      std::size_t(std::numeric_limits<int>::max()) + 1, 1, 1));
  EXPECT_ANY_THROW(PatchMatchHmmCheckpoint(
      128, 128, std::numeric_limits<std::size_t>::max()));
}

TEST(PatchMatchCuda, RngPerimeterLayoutPreservesRotatedRowZero) {
  const std::array<std::pair<std::size_t, std::size_t>, 7> shapes = {
      {{1, 1}, {1, 9}, {9, 1}, {2, 7}, {7, 2}, {33, 17}, {47, 35}}};
  for (const auto& [width, height] : shapes) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    const PatchMatchRngPerimeter layout{width, height};
    const std::size_t expected_count =
        width == 1 ? height
                   : (height == 1 ? width : 2 * width + 2 * height - 4);
    ASSERT_EQ(layout.Count(), expected_count);
    std::vector<bool> seen(expected_count, false);
    using Row = std::vector<std::size_t>;
    std::vector<Row> rotated(height, Row(width));
    for (std::size_t row = 0; row < height; ++row) {
      for (std::size_t col = 0; col < width; ++col) {
        rotated[row][col] = row * width + col;
        if (row != 0 && col != 0 && row + 1 != height && col + 1 != width) {
          continue;
        }
        const std::size_t index = layout.Index(row, col);
        ASSERT_LT(index, expected_count);
        EXPECT_FALSE(seen[index]);
        seen[index] = true;
        std::size_t recovered_row = 0, recovered_col = 0;
        layout.Coordinates(index, &recovered_row, &recovered_col);
        EXPECT_EQ(recovered_row, row);
        EXPECT_EQ(recovered_col, col);
      }
    }
    for (const bool visited : seen) EXPECT_TRUE(visited);

    for (unsigned rotation = 0; rotation < 4; ++rotation) {
      const std::size_t current_height = rotated.size();
      const std::size_t current_width = rotated.front().size();
      std::vector<bool> active(expected_count, false);
      for (std::size_t col = 0; col < current_width; ++col) {
        const std::size_t index = layout.SweepIndex(rotation, col);
        ASSERT_LT(index, expected_count);
        EXPECT_FALSE(active[index]);
        active[index] = true;
        std::size_t original_row = 0, original_col = 0;
        layout.Coordinates(index, &original_row, &original_col);
        EXPECT_EQ(original_row * width + original_col, rotated[0][col]);
      }
      for (std::size_t row = 0; row < current_height; ++row) {
        for (std::size_t col = 0; col < current_width; ++col) {
          std::size_t original_row = 0, original_col = 0;
          layout.OriginalCoordinates(
              rotation, row, col, &original_row, &original_col);
          ASSERT_LT(original_row, height);
          ASSERT_LT(original_col, width);
          EXPECT_EQ(original_row * width + original_col, rotated[row][col]);
        }
      }
      std::vector<Row> next(current_width, Row(current_height));
      for (std::size_t row = 0; row < current_height; ++row) {
        for (std::size_t col = 0; col < current_width; ++col) {
          next[current_width - 1 - col][row] = rotated[row][col];
        }
      }
      rotated.swap(next);
    }
    EXPECT_EQ(layout.SweepIndex(0, width - 1), layout.SweepIndex(1, 0));
    EXPECT_EQ(layout.SweepIndex(1, height - 1), layout.SweepIndex(2, 0));
    EXPECT_EQ(layout.SweepIndex(2, width - 1), layout.SweepIndex(3, 0));
    EXPECT_EQ(layout.SweepIndex(3, height - 1), layout.SweepIndex(0, 0));
  }
}

constexpr float kPlaneZ = 4.0f;
constexpr float kDepthMin = 3.0f;
constexpr float kDepthMax = 5.0f;

struct Inputs {
  std::vector<Image> images;
  std::vector<DepthMap> depths;
  std::vector<NormalMap> normals;
};

// Two independent reference problems, each with differently calibrated source
// views. All images are rendered from the same textured world plane. Sizes are
// deliberately non-square, odd, and not CUDA block multiples.
Inputs MakeInputs(const int variant,
                  const int num_source_images = 2,
                  const int ref_height = 0,
                  const int ref_width = 0) {
  Inputs inputs;
  for (int i = 0; i <= num_source_images; ++i) {
    const int width =
        (ref_width > 0 ? ref_width : (variant == 0 ? 47 : 39)) + 2 * i;
    const int height =
        (ref_height > 0 ? ref_height : (variant == 0 ? 35 : 53)) + 2 * i;
    const float fx = (variant == 0 ? 48.0f : 69.0f) + 3.0f * i;
    const float fy = (variant == 0 ? 57.0f : 44.0f) + 2.0f * i;
    const float cx = width * 0.5f + 0.7f + i * 0.2f;
    const float cy = height * 0.5f - 1.1f - i * 0.15f;
    const float yaw = (i - 1) * 0.035f;
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    const float camera_x = (i - 1) * 0.25f;
    const float camera_y = (i - 1) * 0.04f;
    const float K[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    const float R[9] = {c, 0, s, 0, 1, 0, -s, 0, c};
    const float T[3] = {-c * camera_x, -camera_y, s * camera_x};

    Image image(
        "synthetic-" + std::to_string(variant) + "-" + std::to_string(i),
        width,
        height,
        K,
        R,
        T);
    Bitmap bitmap(width, height, false);
    DepthMap depth(width, height, kDepthMin, kDepthMax);
    NormalMap normal(width, height);
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        // COLMAP's undistorted calibration uses pixel centers at (col+.5,
        // row+.5). Invert R to intersect the camera ray with world Z=4.
        const float ray_x = (col + 0.5f - cx) / fx;
        const float ray_y = (row + 0.5f - cy) / fy;
        const float world_ray_x = c * ray_x - s;
        const float world_ray_z = s * ray_x + c;
        const float z = kPlaneZ / world_ray_z;
        const float x = camera_x + z * world_ray_x;
        const float y = camera_y + z * ray_y;
        const float value = 128.0f + 42.0f * std::sin(8.0f * x + 3.0f * y) +
                            35.0f * std::cos(5.0f * x - 11.0f * y) +
                            24.0f * std::sin(17.0f * x + 13.0f * y);
        bitmap.SetPixel(col,
                        row,
                        BitmapColor<uint8_t>(static_cast<uint8_t>(
                            std::clamp(std::round(value), 0.0f, 255.0f))));
        depth.Set(row, col, z);
        normal.Set(row, col, 0, -s);
        normal.Set(row, col, 1, 0.0f);
        normal.Set(row, col, 2, -c);
      }
    }
    image.SetBitmap(std::move(bitmap));
    inputs.images.push_back(std::move(image));
    inputs.depths.push_back(std::move(depth));
    inputs.normals.push_back(std::move(normal));
  }
  return inputs;
}

PatchMatchOptions MakeOptions(const bool geometry,
                              const bool filter,
                              const bool reuse) {
  PatchMatchOptions options;
  options.gpu_index = "0";
  options.depth_min = kDepthMin;
  options.depth_max = kDepthMax;
  options.window_radius = 2;
  options.window_step = 1;
  options.num_samples = 3;
  // Two iterations cross both alternating orientations repeatedly.
  options.num_iterations = 2;
  options.geom_consistency = geometry;
  options.filter = filter;
  options.rotation_buffer_reuse = reuse;
  // Keep full-storage controls explicit when the public CUDA default changes.
  options.checkpoint_hmm = false;
  options.filter_min_ncc = -1.0;
  options.filter_min_triangulation_angle = 0.0;
  options.filter_min_num_consistent = 1;
  options.filter_geom_consistency_max_cost = 10.0;
  return options;
}

struct Result {
  DepthMap depths;
  NormalMap normals;
  Mat<float> selection;
  // Query graph through its public API, preserving each pixel and ordered IDs.
  std::vector<std::vector<int> > consistent_images;
};

Result RunPatchMatch(Inputs* inputs, const PatchMatchOptions& options) {
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  for (size_t i = 1; i < inputs->images.size(); ++i) {
    problem.src_image_idxs.push_back(static_cast<int>(i));
  }
  problem.images = &inputs->images;
  problem.depth_maps = &inputs->depths;
  problem.normal_maps = &inputs->normals;
  PatchMatch patch_match(options, problem);
  patch_match.Run();
  Result result{patch_match.GetDepthMap(),
                patch_match.GetNormalMap(),
                patch_match.GetSelProbMap(),
                {}};
  const ConsistencyGraph graph = patch_match.GetConsistencyGraph();
  for (size_t row = 0; row < result.depths.GetHeight(); ++row) {
    for (size_t col = 0; col < result.depths.GetWidth(); ++col) {
      int num_images = 0;
      const int* image_idxs = nullptr;
      graph.GetImageIdxs(row, col, &num_images, &image_idxs);
      result.consistent_images.emplace_back();
      if (num_images > 0) {
        result.consistent_images.back().assign(image_idxs,
                                               image_idxs + num_images);
      }
    }
  }
  return result;
}

void ExpectExactlyEqual(const Mat<float>& expected, const Mat<float>& actual) {
  ASSERT_EQ(expected.GetWidth(), actual.GetWidth());
  ASSERT_EQ(expected.GetHeight(), actual.GetHeight());
  ASSERT_EQ(expected.GetDepth(), actual.GetDepth());
  ASSERT_EQ(expected.GetData().size(), actual.GetData().size());
  for (size_t i = 0; i < expected.GetData().size(); ++i) {
    const float a = expected.GetData()[i];
    const float b = actual.GetData()[i];
    ASSERT_TRUE(std::isfinite(a)) << i;
    ASSERT_TRUE(std::isfinite(b)) << i;
    // These delivery/allocation/scheduling paths preserve the arithmetic.
    // Require byte equality within the same executable and CUDA device only.
    ASSERT_EQ(std::memcmp(&a, &b, sizeof(float)), 0)
        << "element " << i << ": " << a << " vs " << b;
  }
}

void ExpectEquivalent(const Result& expected, const Result& actual) {
  ExpectExactlyEqual(expected.depths, actual.depths);
  ExpectExactlyEqual(expected.normals, actual.normals);
  ExpectExactlyEqual(expected.selection, actual.selection);
  // Discrete source memberships must agree exactly, including empty pixels.
  ASSERT_EQ(expected.consistent_images.size(), actual.consistent_images.size());
  for (size_t i = 0; i < expected.consistent_images.size(); ++i) {
    ASSERT_EQ(expected.consistent_images[i], actual.consistent_images[i]) << i;
  }
}

void ExpectNonempty(const Result& result,
                    const Inputs& inputs,
                    const bool filter) {
  EXPECT_EQ(result.depths.GetWidth(), inputs.images[0].GetWidth());
  EXPECT_EQ(result.depths.GetHeight(), inputs.images[0].GetHeight());
  EXPECT_EQ(result.normals.GetDepth(), 3);
  EXPECT_EQ(result.selection.GetDepth(), inputs.images.size() - 1);
  EXPECT_EQ(result.depths.GetData().size(),
            result.depths.GetWidth() * result.depths.GetHeight());
  size_t valid_depths = 0;
  for (const float depth : result.depths.GetData()) {
    ASSERT_TRUE(std::isfinite(depth));
    if (depth > 0.0f) {
      ++valid_depths;
    }
  }
  // Propagation and perturbation can leave the initial sampling range.
  EXPECT_GT(valid_depths, 0);
  size_t consistent_pixels = 0;
  for (const auto& image_idxs : result.consistent_images) {
    consistent_pixels += !image_idxs.empty();
    for (const int image_idx : image_idxs) {
      EXPECT_GT(image_idx, 0);
      EXPECT_LT(image_idx, inputs.images.size());
    }
  }
  if (filter) {
    EXPECT_GT(consistent_pixels, 0);
  } else {
    EXPECT_EQ(consistent_pixels, 0);
  }
}

class CudaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int num_devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&num_devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && num_devices == 0)) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

TEST_F(CudaTest, EmptySamplingSupportRetainsInitializedParameters) {
  Inputs inputs = MakeInputs(0);
  for (auto& image : inputs.images) {
    Bitmap bitmap(image.GetWidth(), image.GetHeight(), false);
    // Match the zero texture border so edge windows also have zero variance.
    bitmap.Fill(BitmapColor<uint8_t>(0));
    image.SetBitmap(std::move(bitmap));
  }
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {1, 2};
  problem.images = &inputs.images;
  problem.depth_maps = &inputs.depths;
  problem.normal_maps = &inputs.normals;
  for (const int radius : {2, 5}) {
    for (const bool checkpoint : {false, true}) {
      for (const bool reuse : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << radius << " / checkpoint " << checkpoint << " / reuse "
                     << reuse);
        auto options = MakeOptions(false, false, reuse);
        options.window_radius = radius;
        options.checkpoint_hmm = checkpoint;
        options.ncc_sigma = 1.0 / 16;
        options.num_iterations = 1;
        // Constant patches produce max NCC cost; its likelihood exp(-512)
        // underflows to zero. No source can supply a candidate score.
        PatchMatchCuda owner(options, problem);
        const auto depth = owner.GetDepthMap();
        const auto normal = owner.GetNormalMap();
        ASSERT_GT(depth.GetData().front(), 0.0f);
        owner.Run();
        ExpectExactlyEqual(depth, owner.GetDepthMap());
        ExpectExactlyEqual(normal, owner.GetNormalMap());
      }
    }
  }
}

TEST_F(CudaTest, PropagationPreservesCompletePlaneAcrossSweepRotations) {
  std::array<float, 4> K{800, 640, 750, 480};
  std::array<float, 3> normal{0.6f, 0.4f, -std::sqrt(0.48f)};
  int width = 1280;
  int height = 960;
  for (int rotation = 0; rotation < 4; ++rotation) {
    SCOPED_TRACE(rotation);
    const Eigen::Vector3d n(normal[0], normal[1], normal[2]);
    for (const float col : {0.1f * width, K[1], 0.86f * width}) {
      const float row1 = 0.68f * height;
      const float row2 = row1 + 1;
      const Eigen::Vector3d ray1(
          (double(col) - K[1]) / K[0], (double(row1) - K[3]) / K[2], 1);
      const Eigen::Vector3d ray2(
          (double(col) - K[1]) / K[0], (double(row2) - K[3]) / K[2], 1);
      ASSERT_LT(n.dot(ray1), 0);
      ASSERT_LT(n.dot(ray2), 0);
      for (const float depth : {0.0001f, 10.0f, 1000000.0f}) {
        SCOPED_TRACE(depth);
        const Eigen::Vector3d point1 = double(depth) * ray1;
        const double expected = n.dot(point1) / n.dot(ray2);
        const float actual =
            internal::ProbePropagatedDepth(K, depth, normal, row1, row2, col);
        ASSERT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, expected, 2e-6 * std::abs(expected));
        // A propagated candidate must still lie on the original 3D plane.
        EXPECT_NEAR(n.dot(double(actual) * ray2),
                    n.dot(point1),
                    2e-6 * std::abs(n.dot(point1)));
      }
    }
    const float previous_cx = K[1];
    K = {K[2], K[3], K[0], width - 1.0f - previous_cx};
    normal = {normal[1], -normal[0], normal[2]};
    std::swap(width, height);
  }
}

TEST_F(CudaTest, PropagationRetainsDepthForNearParallelRay) {
  const std::array<float, 4> K{1, 0, 1, 0};
  const std::array<float, 3> normal{0, -1, 0};
  for (const float row2 : {0.0f, 0.000005f}) {
    EXPECT_FLOAT_EQ(internal::ProbePropagatedDepth(K, 3, normal, 1, row2, 0),
                    3);
  }
  EXPECT_NEAR(internal::ProbePropagatedDepth(K, 3, normal, 1, 0.00002f, 0),
              150000.0f,
              1.0f);
}

TEST_F(CudaTest, PropagationRetainsDepthAcrossPlaneHorizon) {
  const std::array<float, 4> K{1000, 0, 1000, 0};
  Eigen::Vector3f n(0.7f, 0.7f, -0.70035f);
  n.normalize();
  const std::array<float, 3> normal{n.x(), n.y(), n.z()};
  ASSERT_LT(n.cast<double>().dot(Eigen::Vector3d(1, 0, 1)), 0);
  ASSERT_GT(n.cast<double>().dot(Eigen::Vector3d(1, 0.001, 1)), 0);
  EXPECT_FLOAT_EQ(internal::ProbePropagatedDepth(K, 10, normal, 0, 1, 1000),
                  10);
}

TEST_F(CudaTest, PropagationAvoidsUnrepresentableDepth) {
  const std::array<float, 4> K{1, 0, 1, 0};
  const float component = std::sqrt(0.5f);
  const std::array<float, 3> normal{0, -component, -component};
  const float depth = std::numeric_limits<float>::max();
  EXPECT_FLOAT_EQ(internal::ProbePropagatedDepth(K, depth, normal, 1, 0, 0),
                  depth);
  // Dividing the plane terms before multiplying avoids an intermediate
  // overflow when the resulting depth is still representable.
  const double expected = double(depth) * 2 / 3;
  EXPECT_NEAR(internal::ProbePropagatedDepth(K, depth, normal, 1, 2, 0),
              expected,
              2e-6 * expected);
}

TEST_F(CudaTest, NondefaultFilteredPhotoPreservesRotatedStatistics) {
  Inputs inputs = MakeInputs(0, 4, 77, 95);
  auto options = MakeOptions(false, true, false);
  options.window_radius = 13;  // Full27/step2 uses the original float planes.
  options.window_step = 2;
  options.num_samples = PatchMatchOptions{}.num_samples;
  const Result native_rotation = RunPatchMatch(&inputs, options);
  options.rotation_buffer_reuse = true;
  const Result retained_rotation = RunPatchMatch(&inputs, options);
  ExpectNonempty(native_rotation, inputs, true);
  ExpectNonempty(retained_rotation, inputs, true);
  ExpectEquivalent(native_rotation, retained_rotation);
}

TEST_F(CudaTest, IntegerStatisticsArrayPreservesChannelBits) {
  // Exercise the existing typed-array wrapper with special FP32 representations
  // carried as integers. This is array/descriptor coverage; the compiled map
  // oracle must additionally cover production packing and device fetches.
  const std::array<uint2, 8> input = {{
      make_uint2(0x00000000u, 0x80000000u),
      make_uint2(0x00000001u, 0x80000001u),
      make_uint2(0x007fffffu, 0x807fffffu),
      make_uint2(0x00800000u, 0x80800000u),
      make_uint2(0x3f800000u, 0xbf800000u),
      make_uint2(0x7f7fffffu, 0xff7fffffu),
      make_uint2(0x7f800000u, 0xff800000u),
      make_uint2(0x7fc12345u, 0x7fa12345u),
  }};
  cudaTextureDesc descriptor = {};
  descriptor.addressMode[0] = cudaAddressModeClamp;
  descriptor.addressMode[1] = cudaAddressModeClamp;
  descriptor.filterMode = cudaFilterModePoint;
  descriptor.readMode = cudaReadModeElementType;
  descriptor.normalizedCoords = 0;
  auto texture = CudaArrayLayeredTexture<uint2>::FromHostArray(
      descriptor, 4, 2, 1, input.data());
  cudaTextureDesc actual = {};
  CUDA_SAFE_CALL(cudaGetTextureObjectTextureDesc(&actual, texture->GetObj()));
  EXPECT_EQ(actual.filterMode, cudaFilterModePoint);
  EXPECT_EQ(actual.readMode, cudaReadModeElementType);
  EXPECT_EQ(actual.normalizedCoords, 0);
  cudaResourceDesc resource = {};
  CUDA_SAFE_CALL(
      cudaGetTextureObjectResourceDesc(&resource, texture->GetObj()));
  ASSERT_EQ(resource.resType, cudaResourceTypeArray);
  cudaChannelFormatDesc channels = {};
  cudaExtent extent = {};
  unsigned flags = 0;
  CUDA_SAFE_CALL(
      cudaArrayGetInfo(&channels, &extent, &flags, resource.res.array.array));
  EXPECT_EQ(channels.x, 32);
  EXPECT_EQ(channels.y, 32);
  EXPECT_EQ(channels.z, 0);
  EXPECT_EQ(channels.w, 0);
  EXPECT_EQ(channels.f, cudaChannelFormatKindUnsigned);
  EXPECT_NE(flags & cudaArrayLayered, 0);
  EXPECT_EQ(extent.width, 4);
  EXPECT_EQ(extent.height, 2);
  EXPECT_EQ(extent.depth, 1);
  std::array<uint2, 8> output = {};
  cudaMemcpy3DParms copy = {};
  copy.srcArray = resource.res.array.array;
  copy.dstPtr = make_cudaPitchedPtr(
      output.data(), 4 * sizeof(uint2), 4 * sizeof(uint2), 2);
  copy.extent = make_cudaExtent(4, 2, 1);
  copy.kind = cudaMemcpyDeviceToHost;
  CUDA_SAFE_CALL(cudaMemcpy3D(&copy));
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(input[i].x, output[i].x);
    EXPECT_EQ(input[i].y, output[i].y);
  }
}

// Parameter order: geometric consistency, filtering.
class PatchMatchCudaTest
    : public CudaTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool> > {};

class SourceUploadTest : public CudaTest,
                         public ::testing::WithParamInterface<bool> {};

TEST_P(SourceUploadTest, DirectUploadPreservesAlgorithmOutputs) {
  Inputs inputs = MakeInputs(0);
  PatchMatchOptions options = MakeOptions(GetParam(), true, false);
  options.direct_source_upload = false;
  const Result padded = RunPatchMatch(&inputs, options);
  options.direct_source_upload = true;
  const Result direct = RunPatchMatch(&inputs, options);
  ExpectNonempty(padded, inputs, true);
  ExpectNonempty(direct, inputs, true);
  ExpectEquivalent(padded, direct);
}

void ExpectPairedDefaultSweepOutputs(const bool reuse,
                                     const bool tall = false) {
  const auto make_problem = [](Inputs& inputs) {
    PatchMatch::Problem problem;
    problem.ref_image_idx = 0;
    for (size_t i = 1; i < inputs.images.size(); ++i) {
      problem.src_image_idxs.push_back(static_cast<int>(i));
    }
    problem.images = &inputs.images;
    problem.depth_maps = &inputs.depths;
    problem.normal_maps = &inputs.normals;
    return problem;
  };
  const auto read = [](const PatchMatchCuda& cuda) {
    return std::make_tuple(cuda.GetDepthMap(),
                           cuda.GetNormalMap(),
                           cuda.GetSelProbMap(),
                           cuda.GetConsistentImageIdxs());
  };
  struct Streams {
    cudaStream_t first = nullptr, second = nullptr;
    ~Streams() {
      if (second) (void)cudaStreamDestroy(second);
      if (first) (void)cudaStreamDestroy(first);
    }
  };
  for (const bool geometry : {false, true}) {
    SCOPED_TRACE(::testing::Message()
                 << "geometry=" << geometry << " reuse=" << reuse);
    // Odd, differently calibrated references and nine sources exercise both
    // orientations and a final consistency-mask byte tail.
    Inputs first_inputs = MakeInputs(0, 9, tall ? 149 : 0);
    Inputs second_inputs = MakeInputs(1, 9, tall ? 153 : 0);
    auto options = MakeOptions(geometry, geometry, reuse);
    const PatchMatchOptions defaults;
    options.window_radius = defaults.window_radius;
    options.window_step = defaults.window_step;
    options.num_samples = defaults.num_samples;
    options.num_iterations = defaults.num_iterations;
    options.sigma_spatial = options.window_radius;
    auto first_problem = make_problem(first_inputs);
    auto second_problem = make_problem(second_inputs);
    const auto serial = [&](const PatchMatch::Problem& problem) {
      PatchMatchCuda cuda(options, problem);
      cuda.Run();
      return read(cuda);
    };
    const auto expected_first = serial(first_problem);
    const auto expected_second = serial(second_problem);
    if (reuse && geometry) {
      const auto& first_mask = std::get<3>(expected_first);
      const auto& second_mask = std::get<3>(expected_second);
      EXPECT_FALSE(first_mask.empty() && second_mask.empty());
    }
    PatchMatchCuda first(options, first_problem);
    PatchMatchCuda second(options, second_problem);
    Streams streams;
    ASSERT_EQ(cudaStreamCreateWithFlags(&streams.first, cudaStreamNonBlocking),
              cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&streams.second, cudaStreamNonBlocking),
              cudaSuccess);
    RunPatchMatchPair(first, second, streams.first, streams.second);
    const auto compare = [&](const auto& expected, const PatchMatchCuda& cuda) {
      const auto actual = read(cuda);
      ExpectExactlyEqual(std::get<0>(expected), std::get<0>(actual));
      ExpectExactlyEqual(std::get<1>(expected), std::get<1>(actual));
      ExpectExactlyEqual(std::get<2>(expected), std::get<2>(actual));
      EXPECT_EQ(std::get<3>(expected), std::get<3>(actual));
      const auto& depths = std::get<0>(actual).GetData();
      EXPECT_GT(std::count_if(depths.begin(),
                              depths.end(),
                              [](float depth) { return depth > 0.0f; }),
                0);
    };
    compare(expected_first, first);
    compare(expected_second, second);
  }
}

TEST_F(CudaTest, PairedDefaultSweepsPreserveIndependentOutputs) {
  ExpectPairedDefaultSweepOutputs(false);
}

TEST_F(CudaTest, PairedBankedRotationPreservesIndependentOutputs) {
  ExpectPairedDefaultSweepOutputs(true);
}

TEST_F(CudaTest, PairedTallInitialCostStripeTailsPreserveOutputs) {
  ExpectPairedDefaultSweepOutputs(false, true);
}

void ExpectReferenceBatchOutputs(const size_t count,
                                 const bool geometry,
                                 const bool reuse,
                                 const bool checkpoint = false) {
  const auto make_problem = [](Inputs& images) {
    PatchMatch::Problem problem;
    problem.ref_image_idx = 0;
    for (size_t i = 1; i < images.images.size(); ++i)
      problem.src_image_idxs.push_back(static_cast<int>(i));
    problem.images = &images.images;
    problem.depth_maps = &images.depths;
    problem.normal_maps = &images.normals;
    return problem;
  };
  auto options = MakeOptions(geometry, geometry, reuse);
  const PatchMatchOptions defaults;
  options.window_radius = defaults.window_radius;
  options.window_step = defaults.window_step;
  options.num_samples = defaults.num_samples;
  options.num_iterations = defaults.num_iterations;
  options.sigma_spatial = options.window_radius;
  std::vector<std::unique_ptr<Inputs> > inputs;
  std::vector<PatchMatch::Problem> problems;
  inputs.reserve(count);
  problems.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    inputs.push_back(std::make_unique<Inputs>(
        MakeInputs(static_cast<int>(i % 2),
                   9,
                   checkpoint ? 129 : (i % 2 ? 153 : 149),
                   checkpoint ? 257 : 0)));
    problems.push_back(make_problem(*inputs.back()));
  }
  const auto construct = [&]() {
    std::vector<std::unique_ptr<PatchMatchCuda> > owners;
    owners.reserve(problems.size());
    for (const auto& problem : problems)
      owners.push_back(std::make_unique<PatchMatchCuda>(options, problem));
    return owners;
  };
  const auto read = [](const PatchMatchCuda& cuda) {
    return std::make_tuple(cuda.GetDepthMap(),
                           cuda.GetNormalMap(),
                           cuda.GetSelProbMap(),
                           cuda.GetConsistentImageIdxs());
  };
  struct Streams {
    std::vector<cudaStream_t> values;
    explicit Streams(size_t count) : values(count, nullptr) {
      for (auto& stream : values)
        CUDA_SAFE_CALL(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Streams() {
      for (auto stream : values)
        if (stream) (void)cudaStreamDestroy(stream);
    }
  } streams(count);
  auto incumbent = construct();
  for (size_t i = 0; i < count; i += 2) {
    if (i + 1 < count) {
      RunPatchMatchPair(*incumbent[i],
                        *incumbent[i + 1],
                        streams.values[i],
                        streams.values[i + 1]);
    } else {
      incumbent[i]->Run();
    }
  }
  using Snapshot =
      std::tuple<DepthMap, NormalMap, Mat<float>, std::vector<int> >;
  std::vector<Snapshot> expected;
  expected.reserve(incumbent.size());
  for (const auto& owner : incumbent) expected.push_back(read(*owner));
  if (!checkpoint) incumbent.clear();
  options.checkpoint_hmm = checkpoint;
  auto batch = construct();
  if (checkpoint) {
    for (size_t i = 0; i < count; ++i)
      EXPECT_LT(batch[i]->PairPeakBytes(), incumbent[i]->PairPeakBytes());
  }
  std::vector<PatchMatchCuda*> views;
  views.reserve(batch.size());
  for (auto& owner : batch) views.push_back(owner.get());
  RunPatchMatchBatch(views, streams.values);
  for (size_t i = 0; i < count; ++i) {
    SCOPED_TRACE(::testing::Message() << "count=" << count << " view=" << i
                                      << " geometry=" << geometry << " reuse="
                                      << reuse << " checkpoint=" << checkpoint);
    const auto actual = read(*batch[i]);
    ExpectExactlyEqual(std::get<0>(expected[i]), std::get<0>(actual));
    ExpectExactlyEqual(std::get<1>(expected[i]), std::get<1>(actual));
    ExpectExactlyEqual(std::get<2>(expected[i]), std::get<2>(actual));
    EXPECT_EQ(std::get<3>(expected[i]), std::get<3>(actual));
  }
  if (checkpoint) {
    for (size_t i = 0; i < count; ++i) {
      incumbent[i]->Run();
      batch[i]->Run();
      const auto expected_next = read(*incumbent[i]);
      const auto actual_next = read(*batch[i]);
      ExpectExactlyEqual(std::get<0>(expected_next), std::get<0>(actual_next));
      ExpectExactlyEqual(std::get<1>(expected_next), std::get<1>(actual_next));
      ExpectExactlyEqual(std::get<2>(expected_next), std::get<2>(actual_next));
      EXPECT_EQ(std::get<3>(expected_next), std::get<3>(actual_next));
    }
  }
}

TEST_F(CudaTest, ThreeViewPhotoBatchPreservesOddTallOutputs) {
  ExpectReferenceBatchOutputs(3, false, false);
}

TEST_F(CudaTest, FiveViewPhotoBatchPreservesOddTallOutputs) {
  ExpectReferenceBatchOutputs(5, false, false);
}

TEST_F(CudaTest, FiveViewGeometricBankedBatchPreservesFinalMask) {
  ExpectReferenceBatchOutputs(5, true, true);
}

TEST_F(CudaTest, ThreeViewCheckpointPhotoBatchPreservesFullStorageOutputs) {
  ExpectReferenceBatchOutputs(3, false, true, true);
}

TEST_F(CudaTest, FiveViewCheckpointPhotoBatchPreservesFullStorageOutputs) {
  ExpectReferenceBatchOutputs(5, false, true, true);
}

TEST_F(CudaTest, ThreeViewCheckpointGeometricBatchPreservesFullStorageOutputs) {
  ExpectReferenceBatchOutputs(3, true, true, true);
}

TEST_F(CudaTest, FiveViewCheckpointGeometricBatchPreservesFullStorageOutputs) {
  ExpectReferenceBatchOutputs(5, true, true, true);
}

TEST_F(CudaTest, SharedGeometryBankPreservesRepeatedBatchAndStandaloneRuns) {
  for (const auto& [width, height] :
       std::array<std::pair<int, int>, 2>{{{128, 256}, {257, 129}}}) {
    for (const bool geometry : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << width << "x" << height << " geometry=" << geometry);
      std::vector<Inputs> inputs;
      for (int i = 0; i < 3; ++i)
        inputs.push_back(MakeInputs(i, 2, height, width));
      auto options = MakeOptions(geometry, geometry, true);
      const PatchMatchOptions defaults;
      options.window_radius = defaults.window_radius;
      options.window_step = defaults.window_step;
      options.num_samples = defaults.num_samples;
      options.num_iterations = defaults.num_iterations;
      options.sigma_spatial = options.window_radius;
      std::vector<PatchMatch::Problem> problems(3);
      std::vector<std::unique_ptr<PatchMatchCuda> > control, candidate;
      for (size_t i = 0; i < problems.size(); ++i) {
        auto& problem = problems[i];
        problem.ref_image_idx = 0;
        problem.src_image_idxs = {1, 2};
        problem.images = &inputs[i].images;
        problem.depth_maps = &inputs[i].depths;
        problem.normal_maps = &inputs[i].normals;
        control.push_back(std::make_unique<PatchMatchCuda>(options, problem));
        ASSERT_TRUE(control.back()->CanShareGeometryRotation());
        candidate.push_back(std::make_unique<PatchMatchCuda>(
            options,
            problem,
            std::shared_ptr<const PatchMatchSparseSources>{},
            i > 0 ? control[0].get() : nullptr));
        ASSERT_TRUE(candidate.back()->CanShareGeometryRotation());
        EXPECT_EQ(candidate.back()->PairPeakBytes(),
                  control.back()->PairPeakBytes());
      }
      std::vector<PatchMatchCuda*> views;
      std::vector<cudaStream_t> streams(3, nullptr);
      struct Cleanup {
        std::vector<cudaStream_t>& streams;
        ~Cleanup() {
          for (auto stream : streams)
            if (stream) (void)cudaStreamDestroy(stream);
        }
      } cleanup{streams};
      for (size_t i = 0; i < 3; ++i) {
        views.push_back(candidate[i].get());
        CUDA_SAFE_CALL(
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      }
      const auto compare = [](const PatchMatchCuda& expected,
                              const PatchMatchCuda& actual) {
        ExpectExactlyEqual(expected.GetDepthMap(), actual.GetDepthMap());
        ExpectExactlyEqual(expected.GetNormalMap(), actual.GetNormalMap());
        ExpectExactlyEqual(expected.GetSelProbMap(), actual.GetSelProbMap());
        EXPECT_EQ(expected.GetConsistentImageIdxs(),
                  actual.GetConsistentImageIdxs());
        EXPECT_GE(actual.PairPeakBytes(), expected.PairPeakBytes());
      };
      for (int repeat = 0; repeat < 2; ++repeat) {
        for (auto& owner : control) owner->Run();
        RunPatchMatchBatch(views, streams);
        for (size_t i = 0; i < 3; ++i) compare(*control[i], *candidate[i]);
      }
      // Non-first owners have no private alternate pair when leaving the batch.
      // Standalone Run must provision it and preserve logical views/readback.
      for (size_t i = 0; i < 3; ++i) {
        control[i]->Run();
        candidate[i]->Run();
        compare(*control[i], *candidate[i]);
      }
      control[0]->Run();
      control[1]->Run();
      RunPatchMatchPair(*candidate[0], *candidate[1], streams[0], streams[1]);
      compare(*control[0], *candidate[0]);
      compare(*control[1], *candidate[1]);
    }
  }
}

TEST_F(CudaTest, SparseSourceCapabilityAndUnavailableContract) {
  Inputs input = MakeInputs(0, 9, 257, 513);
  std::vector<std::shared_ptr<const Bitmap> > bitmap_leases;
  std::vector<std::shared_ptr<const DepthMap> > depth_leases;
  for (const auto& image : input.images)
    bitmap_leases.push_back(std::make_shared<Bitmap>(image.GetBitmap()));
  for (const auto& depth : input.depths)
    depth_leases.push_back(std::make_shared<DepthMap>(depth));
  std::vector<PatchMatch::Problem> problems(3);
  std::vector<const PatchMatch::Problem*> pointers;
  for (int ref = 0; ref < 3; ++ref) {
    auto& problem = problems[ref];
    problem.ref_image_idx = ref;
    problem.images = &input.images;
    problem.depth_maps = &input.depths;
    problem.source_bitmap_leases = &bitmap_leases;
    problem.source_depth_leases = &depth_leases;
    problem.normal_maps = &input.normals;
    for (int id = 0; id < static_cast<int>(input.images.size()); ++id) {
      if (id != ref) problem.src_image_idxs.push_back(id);
    }
    pointers.push_back(&problem);
  }
  const bool supported = PatchMatchSparseSources::IsSupported();
  if (!supported) {
    EXPECT_EQ(PatchMatchSparseSources::Budget(pointers, true), 0);
    EXPECT_FALSE(PatchMatchSparseSources::Create(pointers, true));
  } else {
    // A supported lane must still admit this known-profitable fixture;
    // Budget==0 must not turn a regression into a capability skip.
    EXPECT_GT(PatchMatchSparseSources::Budget(pointers, true), 0);
  }
}

TEST_F(CudaTest, InvalidBitmapLeasesUseBothIndependentDenseUploadPaths) {
  Inputs input = MakeInputs(0, 9, 257, 513);
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.images = &input.images;
  for (int id = 1; id < static_cast<int>(input.images.size()); ++id)
    problem.src_image_idxs.push_back(id);
  for (const bool direct : {false, true}) {
    auto options = MakeOptions(false, false, false);
    options.direct_source_upload = direct;
    problem.source_bitmap_leases = nullptr;
    PatchMatchCuda control(options, problem);
    control.Run();
    for (int invalid = 0; invalid < 3; ++invalid) {
      SCOPED_TRACE(::testing::Message()
                   << "direct " << direct << " invalid " << invalid);
      std::vector<std::shared_ptr<const Bitmap> > leases(input.images.size());
      const int source = 7;
      if (invalid == 0) {
        auto owner = std::make_shared<Bitmap>(input.images[source].GetBitmap());
        std::weak_ptr<const Bitmap> weak = owner;
        leases[source] = std::shared_ptr<const Bitmap>(
            std::shared_ptr<const Bitmap>{}, owner.get());
        owner.reset();
        ASSERT_TRUE(weak.expired());
      } else {
        leases[source] = std::make_shared<Bitmap>(
            input.images[source].GetWidth() + (invalid == 1 ? 1 : 0),
            input.images[source].GetHeight(),
            invalid == 2);
      }
      problem.source_bitmap_leases = &leases;
      ASSERT_EQ(&problem.InputBitmap(source),
                &input.images[source].GetBitmap());
      PatchMatchCuda fallback(options, problem);
      fallback.Run();
      ExpectExactlyEqual(control.GetDepthMap(), fallback.GetDepthMap());
      ExpectExactlyEqual(control.GetNormalMap(), fallback.GetNormalMap());
      ExpectExactlyEqual(control.GetSelProbMap(), fallback.GetSelProbMap());
      EXPECT_EQ(control.GetConsistentImageIdxs(),
                fallback.GetConsistentImageIdxs());
      problem.source_bitmap_leases = nullptr;
    }
  }
}

TEST_F(CudaTest, SparseSourceProvenanceRejectsUnknownOrChangedOwners) {
  Inputs input = MakeInputs(0, 9, 257, 513);
  std::vector<std::shared_ptr<const Bitmap> > bitmaps;
  std::vector<std::shared_ptr<const DepthMap> > depths;
  for (const auto& image : input.images)
    bitmaps.push_back(std::make_shared<Bitmap>(image.GetBitmap()));
  for (const auto& depth : input.depths)
    depths.push_back(std::make_shared<DepthMap>(depth));
  std::vector<std::vector<std::shared_ptr<const Bitmap> > > bitmap_leases(
      3, bitmaps);
  std::vector<std::vector<std::shared_ptr<const DepthMap> > > depth_leases(
      3, depths);
  std::vector<PatchMatch::Problem> problems(3);
  std::vector<const PatchMatch::Problem*> pointers;
  for (int ref = 0; ref < 3; ++ref) {
    auto& p = problems[ref];
    p.ref_image_idx = ref;
    p.images = &input.images;
    p.depth_maps = &input.depths;
    p.source_bitmap_leases = &bitmap_leases[ref];
    p.source_depth_leases = &depth_leases[ref];
    for (int id = 0; id < static_cast<int>(input.images.size()); ++id)
      if (id != ref) p.src_image_idxs.push_back(id);
    pointers.push_back(&p);
  }
  ASSERT_TRUE(PatchMatchSparseSources::HasConsistentInputs(pointers, true));
  const int source = 7;
  auto changed_bitmap = std::make_shared<Bitmap>(*bitmaps[source]);
  changed_bitmap->RowMajorData()[0] ^= 1;
  bitmap_leases[1][source] = changed_bitmap;
  EXPECT_FALSE(PatchMatchSparseSources::HasConsistentInputs(pointers, false));
  bitmap_leases[1][source] = std::shared_ptr<const Bitmap>(
      std::shared_ptr<const Bitmap>{}, bitmaps[source].get());
  EXPECT_FALSE(PatchMatchSparseSources::HasConsistentInputs(pointers, false));
  // Even equal contents after eviction/reload are not a common retained owner.
  bitmap_leases[1][source] = std::make_shared<Bitmap>(*bitmaps[source]);
  EXPECT_FALSE(PatchMatchSparseSources::HasConsistentInputs(pointers, false));
  bitmap_leases[1][source] = bitmaps[source];
  auto changed_depth = std::make_shared<DepthMap>(*depths[source]);
  changed_depth->Set(0, 0, 123.0f);
  depth_leases[1][source] = changed_depth;
  EXPECT_FALSE(PatchMatchSparseSources::HasConsistentInputs(pointers, true));
  depth_leases[1][source] = depths[source];
  problems[1].source_bitmap_leases = nullptr;
  EXPECT_FALSE(PatchMatchSparseSources::HasConsistentInputs(pointers, false));
  problems[1].source_bitmap_leases = &bitmap_leases[1];
  problems[1].source_depth_leases = nullptr;
  EXPECT_FALSE(PatchMatchSparseSources::HasConsistentInputs(pointers, true));
  problems[1].source_depth_leases = &depth_leases[1];
  EXPECT_TRUE(PatchMatchSparseSources::HasConsistentInputs(pointers, true));
}

TEST_F(CudaTest, SparseSourceCleanRefusalsReleasePartialSetup) {
  if (!PatchMatchSparseSources::IsSupported()) {
    GTEST_SKIP() << "Sparse source arrays are unavailable on this CUDA "
                 << "runtime/device";
  }
  Inputs input = MakeInputs(0, 9, 257, 513);
  std::vector<std::shared_ptr<const Bitmap> > bitmaps;
  std::vector<std::shared_ptr<const DepthMap> > depths;
  for (const auto& image : input.images)
    bitmaps.push_back(std::make_shared<Bitmap>(image.GetBitmap()));
  for (const auto& depth : input.depths)
    depths.push_back(std::make_shared<DepthMap>(depth));
  std::vector<PatchMatch::Problem> problems(3);
  std::vector<const PatchMatch::Problem*> pointers;
  for (int ref = 0; ref < 3; ++ref) {
    auto& p = problems[ref];
    p.ref_image_idx = ref;
    p.images = &input.images;
    p.depth_maps = &input.depths;
    p.normal_maps = &input.normals;
    p.source_bitmap_leases = &bitmaps;
    p.source_depth_leases = &depths;
    for (int id = 0; id < static_cast<int>(input.images.size()); ++id)
      if (id != ref) p.src_image_idxs.push_back(id);
    pointers.push_back(&p);
  }
  ASSERT_GT(PatchMatchSparseSources::Budget(pointers, true), 0);
  SparseSourceTestFailure probe{"cudaMalloc3DArray probe",
                                cudaErrorMemoryAllocation};
  EXPECT_THROW(PatchMatchSparseSources::Budget(pointers, true, &probe),
               SparseSourceResourceRefused);
  SparseSourceTestFailure unsupported{"cudaArrayGetSparseProperties",
                                      cudaErrorNotSupported};
  EXPECT_EQ(PatchMatchSparseSources::Budget(pointers, true, &unsupported), 0);
  struct Case {
    const char* operation;
    int code;
    size_t occurrence, arrays_released, pools_released;
  };
  std::vector<Case> cases{
      {"cudaStreamCreateWithFlags", cudaErrorMemoryAllocation, 1, 0, 0},
      {"cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY, 1, 0, 0},
      {"cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY, 2, 3, 1},
      {"host source staging", 1, 1, 3, 1},
      {"host source staging", 1, 2, 6, 2}};
  for (size_t i = 1; i <= 6; ++i)
    cases.push_back({"cudaMalloc3DArray source",
                     cudaErrorMemoryAllocation,
                     i,
                     i - 1,
                     i <= 3 ? 1u : 2u});
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << test.operation << " #" << test.occurrence);
    SparseSourceTestFailure released_pools{"cuMemRelease", 0};
    SparseSourceTestFailure released_arrays{"cudaFreeArray cleanup", 0};
    released_arrays.next = &released_pools;
    SparseSourceTestFailure failure{test.operation, test.code, test.occurrence};
    failure.next = &released_arrays;
    EXPECT_THROW(PatchMatchSparseSources::Create(pointers, true, &failure),
                 SparseSourceResourceRefused);
    EXPECT_EQ(released_arrays.calls, test.arrays_released);
    EXPECT_EQ(released_pools.calls, test.pools_released);
    // A successful fresh construction exercises the same device after cleanup.
    auto next = PatchMatchSparseSources::Create(pointers, true);
    ASSERT_TRUE(next);
    next->Close();
  }
  for (const auto& test : std::vector<Case>{
           {"cudaGetDevice", cudaErrorUnknown, 1, 0, 0},
           {"cudaGetDeviceProperties", cudaErrorUnknown, 1, 0, 0},
           {"cudaArrayGetSparseProperties", cudaErrorUnknown, 1, 0, 0},
           {"cuMemMapArrayAsync", CUDA_ERROR_INVALID_VALUE, 1, 3, 1},
           {"cudaStreamSynchronize map", cudaErrorUnknown, 1, 3, 1},
           {"cudaMemcpy3DAsync source", cudaErrorInvalidValue, 1, 3, 1}}) {
    SparseSourceTestFailure released_pools{"cuMemRelease", 0};
    SparseSourceTestFailure released_arrays{"cudaFreeArray cleanup", 0};
    released_arrays.next = &released_pools;
    SparseSourceTestFailure failure{test.operation, test.code, test.occurrence};
    failure.next = &released_arrays;
    try {
      (void)PatchMatchSparseSources::Create(pointers, true, &failure);
      FAIL() << "Expected terminal sparse operation failure";
    } catch (const SparseSourceResourceRefused&) {
      FAIL() << "Unexpected fallback permission for " << test.operation;
    } catch (const std::exception& error) {
      EXPECT_NE(std::string(error.what()).find(test.operation),
                std::string::npos);
    }
    EXPECT_EQ(released_arrays.calls, test.arrays_released);
    EXPECT_EQ(released_pools.calls, test.pools_released);
  }
  auto shared = PatchMatchSparseSources::Create(pointers, true);
  ASSERT_TRUE(shared);
  {
    auto live_view = PatchMatchSourceArray<uint8_t>::FromSparse(
        shared, problems[0].ref_image_idx);
    ASSERT_TRUE(live_view);
    EXPECT_THROW(shared->Close(), std::exception);
    EXPECT_TRUE(shared->Matches(problems[0], true));
    live_view->Close();
  }
  auto options = MakeOptions(true, true, false);
  PatchMatchCuda expected(options, problems[0]);
  expected.Run();
  for (size_t texture = 1; texture <= 2; ++texture) {
    PatchMatchCuda retained(options, problems[0]);
    SparseSourceTestFailure failure{
        "cudaCreateTextureObject sparse", cudaErrorMemoryAllocation, texture};
    EXPECT_THROW(retained.UseSparseSources(shared, &failure),
                 SparseSourceResourceRefused);
    retained.Run();
    ExpectExactlyEqual(expected.GetDepthMap(), retained.GetDepthMap());
    ExpectExactlyEqual(expected.GetNormalMap(), retained.GetNormalMap());
    ExpectExactlyEqual(expected.GetSelProbMap(), retained.GetSelProbMap());
    EXPECT_EQ(expected.GetConsistentImageIdxs(),
              retained.GetConsistentImageIdxs());
  }
  auto disabled_options = options;
  disabled_options.sparse_source_sharing = false;
  PatchMatchCuda disabled(disabled_options, problems[0]);
  SparseSourceTestFailure no_probe{"cudaCreateTextureObject sparse",
                                   cudaErrorUnknown};
  disabled.UseSparseSources(shared, &no_probe);
  EXPECT_EQ(no_probe.calls, 0);
  disabled.Run();
  ExpectExactlyEqual(expected.GetDepthMap(), disabled.GetDepthMap());
  std::weak_ptr<const Bitmap> retained_bitmap = bitmaps[7];
  bitmaps[7].reset();
  EXPECT_FALSE(retained_bitmap.expired());
  shared->Close();
  shared.reset();
  EXPECT_TRUE(retained_bitmap.expired());
}

class ControllerWorkspaceFixture {
 public:
  ControllerWorkspaceFixture(size_t width, size_t height)
      : width_(width), height_(height) {
    SyntheticDatasetOptions dataset;
    dataset.num_rigs = 1;
    dataset.num_cameras_per_rig = 1;
    dataset.num_frames_per_rig = 5;
    dataset.camera_width = width_;
    dataset.camera_height = height_;
    SynthesizeDataset(dataset, &reconstruction);
    for (int image = 1; image <= 5; ++image)
      names.push_back(reconstruction.Image(image).Name());
  }
  std::filesystem::path Workspace() {
    const auto root =
        CreateTestDir() / ("workspace_" + std::to_string(workspace_index_++));
    CreateDirIfNotExists(root);
    for (const auto* dir : {"images", "sparse", "stereo"})
      CreateDirIfNotExists(root / dir);
    for (const auto* dir : {"depth_maps", "normal_maps", "consistency_graphs"})
      CreateDirIfNotExists(root / "stereo" / dir);
    reconstruction.Write(root / "sparse");
    std::ofstream config(root / "stereo/patch-match.cfg");
    for (size_t i = 0; i < names.size(); ++i) {
      Bitmap bitmap(width_, height_, false);
      auto& pixels = bitmap.RowMajorData();
      for (size_t y = 0; y < height_; ++y)
        for (size_t x = 0; x < width_; ++x)
          pixels[y * width_ + x] = (17 * x + 31 * y + x * y + 19 * i) % 256;
      bitmap.Write(root / "images" / names[i]);
      config << names[i] << "\n";
      bool first = true;
      for (size_t j = 0; j < names.size(); ++j) {
        if (j == i) continue;
        if (!first) config << ",";
        config << names[j];
        first = false;
      }
      config << "\n";
    }
    config.close();
    return root;
  }
  const std::vector<std::string>& ImageNames() const { return names; }
  static auto Outputs(const std::filesystem::path& root) {
    std::map<std::filesystem::path, std::vector<char> > maps;
    for (const auto* family : {"depth_maps", "normal_maps"}) {
      for (const auto& file :
           std::filesystem::directory_iterator(root / "stereo" / family)) {
        if (!file.is_regular_file()) continue;
        std::ifstream stream(file.path(), std::ios::binary);
        EXPECT_TRUE(stream.good());
        maps[file.path().lexically_relative(root / "stereo")] =
            std::vector<char>(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
        EXPECT_FALSE(
            maps[file.path().lexically_relative(root / "stereo")].empty());
      }
    }
    return maps;
  }
  static PatchMatchOptions Options() {
    PatchMatchOptions options;
    options.gpu_index = "0";
    options.num_threads = 12;
    options.rotation_buffer_reuse = true;
    options.sparse_source_sharing = false;
    return options;
  }

 private:
  size_t width_, height_, workspace_index_ = 0;
  Reconstruction reconstruction;
  std::vector<std::string> names;
};

TEST(PatchMatchController, RejectsDuplicateConfiguredReferences) {
  ControllerWorkspaceFixture fixture(8, 8);
  const auto& names = fixture.ImageNames();
  for (const bool different_sources : {false, true}) {
    SCOPED_TRACE(different_sources);
    const auto root = fixture.Workspace();
    {
      std::ofstream config(root / "stereo/patch-match.cfg");
      config << names[0] << "\n" << names[1] << "\n"
             << names[0] << "\n" << names[different_sources ? 2 : 1]
             << "\n";
    }
    PatchMatchController controller(
        ControllerWorkspaceFixture::Options(), root, "COLMAP", "");
    try {
      PatchMatchControllerTestAccess::ReadConfig(controller);
      FAIL() << "Expected duplicate reference rejection";
    } catch (const std::exception& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find("Duplicate reference image"), std::string::npos);
      EXPECT_NE(message.find(names[0]), std::string::npos);
    }
  }
}

TEST(PatchMatchController, PreservesUniqueReferenceAndSourceOrder) {
  ControllerWorkspaceFixture fixture(8, 8);
  const auto& names = fixture.ImageNames();
  const auto root = fixture.Workspace();
  {
    std::ofstream config(root / "stereo/patch-match.cfg");
    config << names[2] << "\n" << names[4] << "," << names[0] << "\n"
           << names[0] << "\n" << names[3] << "," << names[2] << "\n";
  }
  PatchMatchController controller(
      ControllerWorkspaceFixture::Options(), root, "COLMAP", "");
  const std::vector<PatchMatchControllerTestAccess::ConfigProblem> expected = {
      {names[2], {names[4], names[0]}}, {names[0], {names[3], names[2]}}};
  EXPECT_EQ(PatchMatchControllerTestAccess::ReadConfig(controller), expected);
}

PatchMatchOptions PublicationOptions(const bool graph) {
  auto options = ControllerWorkspaceFixture::Options();
  options.geom_consistency = false;
  options.depth_min = 1;
  options.depth_max = 10;
  options.write_consistency_graph = graph;
  return options;
}

PatchMatchOptions PublicationGpuOptions() {
  auto options = PublicationOptions(true);
  options.num_iterations = 5;
  options.geom_consistency = true;
  options.filter = true;
  options.filter_min_ncc = -1.0;
  options.filter_min_num_consistent = 1;
  options.filter_min_triangulation_angle = 0.0;
  options.filter_geom_consistency_max_cost = 1000000.0;
  return options;
}

std::filesystem::path MakePublicationGpuWorkspace() {
  constexpr size_t width = 96, height = 64;
  ControllerWorkspaceFixture fixture(width, height);
  const auto root = fixture.Workspace();
  // Build a calibrated, common textured plane instead of the generic fixture's
  // unrelated views and 1024x768 camera intrinsics.
  Reconstruction model;
  model.Read(root / "sparse");
  for (const auto& [id, camera] : model.Cameras()) {
    model.Camera(id).params = {80.0, 48.0, 32.0, 0.0};
  }
  for (const auto& [id, frame] : model.Frames()) {
    Rigid3d pose;
    pose.translation() = Eigen::Vector3d(-0.1 * id, 0.0, 0.0);
    model.Frame(id).SetRigFromWorld(pose);
  }
  model.Write(root / "sparse");
  for (const auto& [id, image] : model.Images()) {
    const double camera_x = 0.1 * image.FrameId();
    Bitmap bitmap(width, height, false);
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        const double x = camera_x + 4.0 * (col + 0.5 - 48.0) / 80.0;
        const double y = 4.0 * (row + 0.5 - 32.0) / 80.0;
        const double value = 128.0 + 42.0 * std::sin(8.0 * x + 3.0 * y) +
                             35.0 * std::cos(5.0 * x - 11.0 * y) +
                             24.0 * std::sin(17.0 * x + 13.0 * y);
        bitmap.SetPixel(col, row, BitmapColor<uint8_t>(static_cast<uint8_t>(
                                      std::clamp(std::round(value), 0.0, 255.0))));
      }
    }
    bitmap.Write(root / "images" / image.Name());
    const auto name = image.Name() + ".photometric.bin";
    DepthMap depth(width, height, 1, 10);
    std::fill_n(depth.GetPtr(), width * height, 4.0f);
    depth.Write(root / "stereo" / "depth_maps" / name);
    NormalMap normal(width, height);
    std::fill_n(normal.GetPtr(), 3 * width * height, 0.0f);
    std::fill_n(normal.GetPtr() + 2 * width * height, width * height, -1.0f);
    normal.Write(root / "stereo" / "normal_maps" / name);
  }
  return root;
}

void ExpectPublicationGpuFiles(const std::filesystem::path& root,
                               const size_t expected) {
  constexpr size_t width = 96, height = 64;
  for (const auto* family : {"depth_maps", "normal_maps"}) {
    size_t count = 0;
    for (const auto& f : std::filesystem::directory_iterator(
             root / "stereo" / family)) {
      if (f.path().filename().string().find(".geometric.bin") ==
          std::string::npos) {
        continue;
      }
      EXPECT_GT(std::filesystem::file_size(f.path()), 0u);
      ++count;
    }
    EXPECT_EQ(count, expected);
  }
  size_t graphs = 0, nonempty = 0;
  for (const auto& f : std::filesystem::directory_iterator(
           root / "stereo" / "consistency_graphs")) {
    if (!f.is_regular_file()) continue;
    ++graphs;
    ConsistencyGraph graph;
    ASSERT_NO_THROW(graph.Read(f.path()));
    const auto header = std::to_string(width) + "&" +
                        std::to_string(height) + "&1&";
    if (std::filesystem::file_size(f.path()) > header.size()) ++nonempty;
  }
  EXPECT_EQ(graphs, expected);
  EXPECT_EQ(nonempty, expected);
}

TEST_F(CudaTest, PublicationRealCudaPair) {
  constexpr size_t width = 96, height = 64;
  const auto options = PublicationGpuOptions();
  ASSERT_TRUE(CanPairPatchMatchViews(
      width, height, width, height, true, options.checkpoint_hmm));
  const auto root = MakePublicationGpuWorkspace();
  PatchMatchController controller(options, root, "COLMAP", "");
  PatchMatchControllerTestAccess::Hooks hooks;
  ASSERT_NO_THROW(PatchMatchControllerTestAccess::RunPair(
      controller, hooks, options, 0, 1));
  ExpectPublicationGpuFiles(root, 2);
}

TEST_F(CudaTest, PublicationRealPatchMatchSingle) {
  const auto options = PublicationGpuOptions();
  const auto root = MakePublicationGpuWorkspace();
  PatchMatchController controller(options, root, "COLMAP", "");
  PatchMatchControllerTestAccess::Hooks hooks;
  ASSERT_NO_THROW(PatchMatchControllerTestAccess::RunSingles(
      controller, hooks, options, 1));
  ExpectPublicationGpuFiles(root, 1);
}

void WritePublicationFiles(
    const PatchMatchControllerTestAccess::Prepared& prepared,
    const std::filesystem::path& normal_path) {
  {
    DepthMap depth(8, 8, 1, 10);
    std::fill_n(depth.GetPtr(), 64, 4.0f);
    depth.Write(prepared.depth_path);
  }
  if (prepared.options.write_consistency_graph) {
    std::vector<int> graph;
    for (int row = 0; row < 8; ++row)
      for (int col = 0; col < 8; ++col)
        graph.insert(graph.end(), {col, row, 4, 1, 2, 3, 4});
    ConsistencyGraph(8, 8, graph).Write(prepared.consistency_path);
  }
  {
    NormalMap normal(8, 8);
    std::fill_n(normal.GetPtr() + 128, 64, -1.0f);
    normal.Write(normal_path);
  }
}

TEST(PatchMatchController, PublicationCompletesBeforeResumeSkips) {
  ControllerWorkspaceFixture fixture(8, 8);
  for (const bool graph : {false, true}) {
    const auto options = PublicationOptions(graph);
    PatchMatchController controller(options, fixture.Workspace(), "COLMAP", "");
    const auto prepared =
        PatchMatchControllerTestAccess::Prepare(controller, options);
    ASSERT_TRUE(prepared);
    if (!graph) std::filesystem::create_directory(prepared->consistency_path);
    PatchMatchControllerTestAccess::Publish(*prepared, [&](const auto& normal) {
      WritePublicationFiles(*prepared, normal);
      EXPECT_FALSE(ExistsFile(prepared->normal_path));
    });
    EXPECT_FALSE(PatchMatchControllerTestAccess::Prepare(controller, options));
    DepthMap depth;
    depth.Read(prepared->depth_path);
    EXPECT_EQ(depth.Get(7, 7), 4.0f);
    NormalMap normal;
    normal.Read(prepared->normal_path);
    EXPECT_EQ(normal.Get(7, 7, 2), -1.0f);
    EXPECT_EQ(ExistsFile(prepared->consistency_path), graph);
    if (!graph)
      EXPECT_TRUE(std::filesystem::is_directory(prepared->consistency_path));
  }
}

TEST(PatchMatchController, PublicationWithoutGraphInvalidatesPreviousGraph) {
  ControllerWorkspaceFixture fixture(8, 8);
  auto options = PublicationOptions(true);
  PatchMatchController controller(options, fixture.Workspace(), "COLMAP", "");
  const auto first =
      PatchMatchControllerTestAccess::Prepare(controller, options);
  ASSERT_TRUE(first);
  PatchMatchControllerTestAccess::Publish(*first, [&](const auto& normal) {
    WritePublicationFiles(*first, normal);
  });
  ASSERT_TRUE(ExistsFile(first->consistency_path));
  std::filesystem::remove(first->depth_path);
  options.write_consistency_graph = false;
  const auto next =
      PatchMatchControllerTestAccess::Prepare(controller, options);
  ASSERT_TRUE(next);
  PatchMatchControllerTestAccess::Publish(
      *next, [&](const auto& normal) { WritePublicationFiles(*next, normal); });
  EXPECT_FALSE(ExistsFile(next->consistency_path));
  EXPECT_FALSE(PatchMatchControllerTestAccess::Prepare(controller, options));
  options.write_consistency_graph = true;
  EXPECT_TRUE(PatchMatchControllerTestAccess::Prepare(controller, options));
}

TEST(PatchMatchController, PublicationPreservesNonRegularCompletionPath) {
  ControllerWorkspaceFixture fixture(8, 8);
  const auto options = PublicationOptions(false);
  PatchMatchController controller(options, fixture.Workspace(), "COLMAP", "");
  const auto prepared =
      PatchMatchControllerTestAccess::Prepare(controller, options);
  ASSERT_TRUE(prepared);
  std::filesystem::create_directory(prepared->normal_path);
  bool called = false;
  EXPECT_THROW(PatchMatchControllerTestAccess::Publish(
                   *prepared, [&](const auto&) { called = true; }),
               std::exception);
  EXPECT_FALSE(called);
  EXPECT_TRUE(std::filesystem::is_directory(prepared->normal_path));
  EXPECT_TRUE(PatchMatchControllerTestAccess::Prepare(controller, options));
#ifdef __linux__
  std::filesystem::remove(prepared->normal_path);
  const auto target = prepared->normal_path.parent_path() / "external.bin";
  NormalMap(8, 8).Write(target);
  std::filesystem::create_symlink(target, prepared->normal_path);
  EXPECT_THROW(PatchMatchControllerTestAccess::Publish(
                   *prepared, [&](const auto&) { called = true; }),
               std::exception);
  EXPECT_FALSE(called);
  EXPECT_TRUE(std::filesystem::is_symlink(prepared->normal_path));
  EXPECT_EQ(std::filesystem::file_size(target), 774);
#endif
}

TEST(PatchMatchController, PublicationCleanupKeepsOriginalErrorAndRetry) {
  ControllerWorkspaceFixture fixture(8, 8);
  const auto options = PublicationOptions(false);
  PatchMatchController controller(options, fixture.Workspace(), "COLMAP", "");
  const auto prepared =
      PatchMatchControllerTestAccess::Prepare(controller, options);
  ASSERT_TRUE(prepared);
  std::filesystem::path stage;
  try {
    PatchMatchControllerTestAccess::Publish(*prepared, [&](const auto& normal) {
      WritePublicationFiles(*prepared, normal);
      stage = normal.parent_path();
      // A nonempty directory makes cleanup fail portably, even as root. Only
      // owned temporary names may be removed; recursive cleanup is forbidden.
      std::filesystem::remove(normal);
      std::filesystem::create_directory(normal);
      std::ofstream(normal / "keep").put('x');
      throw std::runtime_error("original output failure");
    });
    FAIL() << "Expected the original output error";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "original output failure");
  }
  EXPECT_FALSE(ExistsFile(prepared->normal_path));
  EXPECT_TRUE(ExistsFile(stage / "normal.bin" / "keep"));
  const auto retry =
      PatchMatchControllerTestAccess::Prepare(controller, options);
  ASSERT_TRUE(retry);
  PatchMatchControllerTestAccess::Publish(*retry, [&](const auto& normal) {
    WritePublicationFiles(*retry, normal);
  });
  EXPECT_FALSE(PatchMatchControllerTestAccess::Prepare(controller, options));
  EXPECT_TRUE(ExistsFile(stage / "normal.bin" / "keep"));
}

#ifdef __linux__
TEST(PatchMatchController, PublicationRejectsDanglingGraphBeforeRetry) {
  ControllerWorkspaceFixture fixture(8, 8);
  auto options = PublicationOptions(false);
  PatchMatchController controller(options, fixture.Workspace(), "COLMAP", "");
  const auto a = PatchMatchControllerTestAccess::Prepare(controller, options, 0);
  const auto b = PatchMatchControllerTestAccess::Prepare(controller, options, 1);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  DepthMap(8, 8, 1, 10).Write(a->depth_path);
  const auto depth_size = std::filesystem::file_size(a->depth_path);
  ASSERT_FALSE(ExistsFile(a->normal_path));
  ASSERT_FALSE(std::filesystem::exists(b->consistency_path));
  std::filesystem::create_symlink(b->consistency_path, a->consistency_path);
  bool called = false;
  EXPECT_THROW(PatchMatchControllerTestAccess::Publish(
                   *a,
                   [&](const auto& normal) {
                     called = true;
                     WritePublicationFiles(*a, normal);
                   }),
               std::exception);
  EXPECT_FALSE(called);
  EXPECT_FALSE(ExistsFile(a->normal_path));
  EXPECT_EQ(std::filesystem::file_size(a->depth_path), depth_size);
  EXPECT_TRUE(std::filesystem::is_symlink(a->consistency_path));
  EXPECT_FALSE(std::filesystem::exists(b->consistency_path));
  // A later graph-enabled publication for B materializes A's old target.
  ConsistencyGraph(8, 8, {}).Write(b->consistency_path);
  EXPECT_EQ(std::filesystem::file_size(b->consistency_path), 6);
  options.write_consistency_graph = true;
  EXPECT_TRUE(PatchMatchControllerTestAccess::Prepare(controller, options, 0));
}

TEST(PatchMatchController, PublicationPreservesDepthAndGraphSymlinkTargets) {
  ControllerWorkspaceFixture fixture(8, 8);
  for (const bool graph : {false, true}) {
    for (const bool depth : {false, true}) {
      const auto options = PublicationOptions(graph);
      PatchMatchController controller(
          options, fixture.Workspace(), "COLMAP", "");
      const auto prepared =
          PatchMatchControllerTestAccess::Prepare(controller, options);
      ASSERT_TRUE(prepared);
      NormalMap(8, 8).Write(prepared->normal_path);
      const auto& link =
          depth ? prepared->depth_path : prepared->consistency_path;
      const auto target = link.parent_path() / "external.bin";
      ConsistencyGraph(8, 8, {}).Write(target);
      std::filesystem::create_symlink(target, link);
      const bool was_ready =
          bool(PatchMatchControllerTestAccess::Prepare(controller, options));
      bool called = false;
      EXPECT_THROW(PatchMatchControllerTestAccess::Publish(
                       *prepared, [&](const auto&) { called = true; }),
                   std::exception);
      EXPECT_FALSE(called);
      EXPECT_TRUE(ExistsFile(prepared->normal_path));
      EXPECT_TRUE(std::filesystem::is_symlink(link));
      EXPECT_EQ(std::filesystem::file_size(target), 6);
      EXPECT_EQ(bool(PatchMatchControllerTestAccess::Prepare(controller, options)),
                was_ready);
    }
  }
}

TEST(PatchMatchController, PublicationPartialWritesRemainEligibleForRetry) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  // The native 8x8 depth, normal and graph files are 262, 774 and 1798 bytes.
  // Exercise an empty depth, partial normal and partial graph with old siblings
  // present. Limit writes only in a child, restoring its limit before retry.
  for (const rlim_t limit : {0, 512, 1024}) {
    SCOPED_TRACE(limit);
    EXPECT_EXIT(
        {
          ControllerWorkspaceFixture fixture(8, 8);
          const auto options = PublicationOptions(limit != 512);
          PatchMatchController controller(
              options, fixture.Workspace(), "COLMAP", "");
          const auto prepared =
              PatchMatchControllerTestAccess::Prepare(controller, options);
          if (!prepared) _exit(2);
          NormalMap(8, 8).Write(prepared->normal_path);
          ConsistencyGraph(8, 8, {}).Write(prepared->consistency_path);
          struct rlimit previous;
          if (getrlimit(RLIMIT_FSIZE, &previous) != 0) _exit(3);
          auto restricted = previous;
          restricted.rlim_cur = limit;
          if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR ||
              setrlimit(RLIMIT_FSIZE, &restricted) != 0) {
            _exit(4);
          }
          bool failed = false;
          uintmax_t partial_bytes = std::numeric_limits<uintmax_t>::max();
          try {
            PatchMatchControllerTestAccess::Publish(
                *prepared, [&](const auto& normal) {
                  try {
                    WritePublicationFiles(*prepared, normal);
                  } catch (...) {
                    partial_bytes = std::filesystem::file_size(
                        limit == 0     ? prepared->depth_path
                        : limit == 512 ? normal
                                       : prepared->consistency_path);
                    throw;
                  }
                });
          } catch (const std::exception&) {
            failed = true;
          }
          if (setrlimit(RLIMIT_FSIZE, &previous) != 0) _exit(5);
          if (!failed || partial_bytes != limit) _exit(6);
          if (!PatchMatchControllerTestAccess::Prepare(controller, options)) {
            std::fputs("Partial output was accepted as completed on retry\n",
                       stderr);
            _exit(7);
          }
          if (ExistsFile(prepared->normal_path)) {
            std::fputs("Publication exposed completion after failure\n", stderr);
            _exit(8);
          }
          _exit(0);
        },
        testing::ExitedWithCode(0),
        "");
  }
}
#endif


TEST(PatchMatchController, PreparePreservesConfiguredSourceOrder) {
  for (const int missing_source : {-1, 4}) {
    ControllerWorkspaceFixture fixture(32, 24);
    auto options = ControllerWorkspaceFixture::Options();
    options.geom_consistency = false;
    options.filter = false;
    options.allow_missing_files = missing_source >= 0;
    options.depth_min = 0.1f;
    options.depth_max = 100.0f;
    options.filter_min_num_consistent = 10;
    PatchMatchController controller(options, fixture.Workspace(), "COLMAP", "");
    const auto results = PatchMatchControllerTestAccess::PrepareSourceOrders(
        controller, missing_source);
    const std::vector<int> expected = missing_source < 0
                                          ? std::vector<int>{3, 1, 4, 2}
                                          : std::vector<int>{3, 1, 2};
    for (const auto& result : results) {
      EXPECT_EQ(result.local, expected);
      EXPECT_EQ(result.shared, expected);
    }
    EXPECT_EQ(results[0].min_num_consistent, 4);
    EXPECT_EQ(results[1].min_num_consistent, expected.size());
  }
}

#ifdef COLMAP_TEST_CUDA_ALLOCATION_WRAPPING
std::pair<size_t, size_t> FindControllerFixtureShape(bool batch,
                                                     bool checkpoint) {
  for (const auto& shape : std::vector<std::pair<size_t, size_t> >{
           {513, 257}, {1025, 513}, {1536, 1024}}) {
    if (!CanPairPatchMatchViews(shape.first,
                                shape.second,
                                shape.first,
                                shape.second,
                                false,
                                checkpoint))
      continue;
    const std::vector<std::pair<size_t, size_t> > shapes(5, shape);
    if (!batch || (CanBatchPatchMatchViews(shapes, false, checkpoint) &&
                   CanBatchPatchMatchViews(shapes, true, checkpoint))) {
      return shape;
    }
  }
  return {0, 0};
}

// Validate every partially constructed handle before the first fresh query.
// Event capture itself is fixed-size and does not allocate in CUDA cleanup.
void ExpectStreamCleanupBeforeQueries(const std::vector<size_t>& ordinals) {
  const size_t count = stream_event_count.load();
  ASSERT_LE(count, stream_events.size());
  size_t begin = 0;
  size_t previous_ordinal = 0;
  for (const size_t ordinal : ordinals) {
    size_t failure = begin;
    while (failure < count &&
           stream_events[failure].kind != StreamEventKind::kRefused)
      ++failure;
    ASSERT_LT(failure, count);
    EXPECT_EQ(stream_events[failure].ordinal, ordinal);
    size_t query = failure + 1;
    while (query < count &&
           stream_events[query].kind != StreamEventKind::kQuery)
      ++query;
    ASSERT_LT(query, count) << "No fresh memory query after refusal";
    size_t created = 0;
    for (size_t i = begin; i < failure; ++i) {
      if (stream_events[i].kind != StreamEventKind::kCreated) continue;
      ++created;
      size_t releases = 0;
      for (size_t j = failure + 1; j < query; ++j)
        if (stream_events[j].kind == StreamEventKind::kDestroyed &&
            stream_events[j].stream == stream_events[i].stream)
          ++releases;
      EXPECT_EQ(releases, 1u);
    }
    EXPECT_EQ(created, ordinal - previous_ordinal - 1);
    previous_ordinal = ordinal;
    begin = failure + 1;
  }
}

void RunControllerWorkStreamRefusals(bool sparse) {
  ASSERT_TRUE(internal::IsPatchMatchWorkStreamConstructionActive != nullptr);
  auto options = ControllerWorkspaceFixture::Options();
  options.geom_consistency = false;
  options.filter = false;
  options.sparse_source_sharing = sparse;
  constexpr size_t width = 513, height = 257;
  const std::vector<std::pair<size_t, size_t>> shapes(5, {width, height});
  if (!CanPairPatchMatchViews(
          width, height, width, height, false, options.checkpoint_hmm) ||
      !CanBatchPatchMatchViews(shapes, false, options.checkpoint_hmm)) {
    GTEST_SKIP() << "Fixed controller pair/batch shape is not admitted";
  }
  if (sparse && !PatchMatchSparseSources::IsSupported())
    GTEST_SKIP() << "Sparse source sharing is unsupported";
  struct ResetOnExit {
    ~ResetOnExit() { ResetStreamFault(StreamFault::kNone, 0); }
  } reset;
  ControllerWorkspaceFixture fixture(width, height);
  const auto native_root = fixture.Workspace();
  PatchMatchControllerTestAccess::Hooks native_hooks;
  {
    PatchMatchController native(options, native_root, "COLMAP", "");
    ASSERT_NO_THROW(PatchMatchControllerTestAccess::RunSingles(
        native, native_hooks, options, 5));
  }
  const auto expected = ControllerWorkspaceFixture::Outputs(native_root);
  ASSERT_EQ(expected.size(), 10u);
  ASSERT_EQ(native_hooks.written_references.size(), 5u);
  // Direct pair uses the same first two images but has a two-reference output.
  const auto pair_root = fixture.Workspace();
  PatchMatchControllerTestAccess::Hooks pair_hooks;
  {
    PatchMatchController native(options, pair_root, "COLMAP", "");
    ASSERT_NO_THROW(PatchMatchControllerTestAccess::RunSingles(
        native, pair_hooks, options, 2));
  }
  const auto expected_pair = ControllerWorkspaceFixture::Outputs(pair_root);
  ASSERT_EQ(expected_pair.size(), 4u);
  // Capture-only controls establish creation and output parity. Normal stream
  // destruction occurs outside the construction scope and is not counted.
  for (const bool batch : {false, true}) {
    SCOPED_TRACE(batch);
    const auto root = fixture.Workspace();
    PatchMatchControllerTestAccess::Hooks hooks;
    ResetStreamFault(StreamFault::kNone, 0, 0, true);
    {
      PatchMatchController controller(options, root, "COLMAP", "");
      if (batch) {
        ASSERT_NO_THROW(PatchMatchControllerTestAccess::Run(controller, hooks));
      } else {
        ASSERT_NO_THROW(PatchMatchControllerTestAccess::RunPair(
            controller, hooks, options, 0, 1));
      }
    }
    const size_t expected_creations = batch ? 5u : 2u;
    ASSERT_EQ(stream_attempts.load(), expected_creations);
    ASSERT_EQ(stream_event_count.load(), expected_creations);
    ASSERT_EQ(stream_fired.load(), 0u);
    for (size_t i = 0; i < expected_creations; ++i) {
      EXPECT_EQ(stream_events[i].kind, StreamEventKind::kCreated);
      EXPECT_NE(stream_events[i].stream, nullptr);
      EXPECT_EQ(stream_events[i].ordinal, i + 1);
    }
    EXPECT_EQ(hooks.written_references,
              batch ? native_hooks.written_references
                    : pair_hooks.written_references);
    EXPECT_EQ(ControllerWorkspaceFixture::Outputs(root),
              batch ? expected : expected_pair);
    ResetStreamFault(StreamFault::kNone, 0);
  }
  for (const size_t ordinal : {1u, 2u}) {
    SCOPED_TRACE(ordinal);
    const auto root = fixture.Workspace();
    PatchMatchControllerTestAccess::Hooks hooks;
    ResetStreamFault(StreamFault::kRefuse, ordinal);
    {
      PatchMatchController controller(options, root, "COLMAP", "");
      ASSERT_NO_THROW(PatchMatchControllerTestAccess::RunPair(
          controller, hooks, options, 0, 1));
    }
    ASSERT_EQ(stream_fired.load(), 1u);
    ExpectStreamCleanupBeforeQueries({ordinal});
    EXPECT_EQ(hooks.written_references, pair_hooks.written_references);
    EXPECT_EQ(ControllerWorkspaceFixture::Outputs(root), expected_pair);
    ResetStreamFault(StreamFault::kNone, 0);
  }
  for (const auto failures :
       {std::pair<size_t, size_t>{1, 0}, {3, 0}, {5, 0}, {1, 2}}) {
    SCOPED_TRACE(failures.first);
    SCOPED_TRACE(failures.second);
    const auto root = fixture.Workspace();
    PatchMatchControllerTestAccess::Hooks hooks;
    ResetStreamFault(StreamFault::kRefuse, failures.first, failures.second);
    {
      PatchMatchController controller(options, root, "COLMAP", "");
      ASSERT_NO_THROW(PatchMatchControllerTestAccess::Run(controller, hooks));
    }
    ASSERT_EQ(stream_fired.load(), failures.second ? 2u : 1u);
    const auto has_phase = [&](const char* phase) {
      return std::any_of(
          hooks.memory_queries.begin(),
          hooks.memory_queries.end(),
          [&](const auto& q) { return std::string(q.phase) == phase; });
    };
    EXPECT_TRUE(has_phase("after-batch-stream-refusal"));
    EXPECT_EQ(hooks.sparse_bank_at_work_stream_refusal, sparse);
    if (failures.second) EXPECT_TRUE(has_phase("after-pair-stream-refusal"));
    ExpectStreamCleanupBeforeQueries(
        failures.second ? std::vector<size_t>{failures.first, failures.second}
                        : std::vector<size_t>{failures.first});
    EXPECT_EQ(hooks.written_references, native_hooks.written_references);
    EXPECT_EQ(ControllerWorkspaceFixture::Outputs(root), expected);
    ResetStreamFault(StreamFault::kNone, 0);
  }
}

TEST_F(CudaTest, ControllerWorkStreamRefusalsSparseOff) {
  RunControllerWorkStreamRefusals(false);
}

TEST_F(CudaTest, ControllerWorkStreamRefusalsSparseOptionOn) {
  // The refusal hook observes pointer presence, not bank contents or lifetime.
  RunControllerWorkStreamRefusals(true);
}

TEST_F(CudaTest, ControllerWorkStreamFatalFailuresNeverReadmit) {
  ASSERT_TRUE(internal::IsPatchMatchWorkStreamConstructionActive != nullptr);
  auto options = ControllerWorkspaceFixture::Options();
  options.geom_consistency = false;
  options.filter = false;
  constexpr size_t width = 513, height = 257;
  const std::vector<std::pair<size_t, size_t>> shapes(5, {width, height});
  if (!CanBatchPatchMatchViews(shapes, false, options.checkpoint_hmm))
    GTEST_SKIP() << "Fixed controller batch shape is not admitted";
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const auto& item : std::array<std::pair<StreamFault, const char*>, 6>{
           {{StreamFault::kPrior, "preexisting error"},
            {StreamFault::kOther, "runtime failure"},
            {StreamFault::kMismatch, "last-error mismatch"},
            {StreamFault::kFailedNonNull, "unresolved output"},
            {StreamFault::kSuccessNull, "Check failed: stream != nullptr"},
            {StreamFault::kDestroy, "cudaStreamDestroy"}}}) {
    const StreamFault fault = item.first;
    const char* const reason = item.second;
    SCOPED_TRACE(static_cast<int>(fault));
    EXPECT_EXIT(
        ([&, fault, reason]() {
          ControllerWorkspaceFixture fixture(width, height);
          const auto root = fixture.Workspace();
          PatchMatchControllerTestAccess::Hooks hooks;
          ResetStreamFault(fault, 3);
          bool threw = false;
          std::string message;
          {
            PatchMatchController controller(options, root, "COLMAP", "");
            try {
              PatchMatchControllerTestAccess::Run(controller, hooks);
            } catch (const std::exception& error) {
              threw = true;
              message = error.what();
            }
          }
          const size_t count = stream_event_count.load();
          if (!threw || message.find(reason) == std::string::npos ||
              stream_fired.load() != 1 || count > stream_events.size() ||
              !hooks.written_references.empty())
            std::_Exit(81);
          size_t failure = 0;
          while (failure < count &&
                 stream_events[failure].kind != StreamEventKind::kRefused)
            ++failure;
          if (failure == count || stream_events[failure].ordinal != 3)
            std::_Exit(82);
          size_t created = 0, destroyed = 0;
          for (size_t i = 0; i < count; ++i) {
            if (stream_events[i].kind == StreamEventKind::kQuery)
              std::_Exit(83);
            if (stream_events[i].kind == StreamEventKind::kCreated) {
              ++created;
              size_t releases = 0;
              for (size_t j = failure + 1; j < count; ++j)
                if (stream_events[j].kind == StreamEventKind::kDestroyed &&
                    stream_events[j].stream == stream_events[i].stream)
                  ++releases;
              if (releases != 1) std::_Exit(84);
            }
            if (stream_events[i].kind == StreamEventKind::kDestroyed)
              ++destroyed;
          }
          if (created != (fault == StreamFault::kFailedNonNull ? 3u : 2u) ||
              destroyed != created ||
              (fault == StreamFault::kDestroy &&
               stream_destroy_failures.load() == 0))
            std::_Exit(85);
          ResetStreamFault(StreamFault::kNone, 0);
          std::_Exit(0);
        }()),
        ::testing::ExitedWithCode(0),
        "");
  }
}

TEST_F(CudaTest, ControllerPairAllocationRefusals) {
  auto options = ControllerWorkspaceFixture::Options();
  const auto [width, height] =
      FindControllerFixtureShape(false, options.checkpoint_hmm);
  if (width == 0) GTEST_SKIP() << "Required controller admission unavailable";
  ControllerWorkspaceFixture fixture(width, height);
  const auto workspace = [&] { return fixture.Workspace(); };
  const auto outputs = ControllerWorkspaceFixture::Outputs;
  {
    struct ResetOnExit {
      ~ResetOnExit() { ResetAllocationInjection(0, cudaErrorMemoryAllocation); }
    } reset_on_exit;
    auto photo_options = options;
    photo_options.geom_consistency = false;
    photo_options.filter = false;
    photo_options.sparse_source_sharing = false;
    ASSERT_TRUE(CanPairPatchMatchViews(
        width, height, width, height, false, photo_options.checkpoint_hmm));

    const auto control_root = workspace();
    PatchMatchControllerTestAccess::Hooks control_hooks;
    ResetAllocationInjection(0, cudaErrorMemoryAllocation, false, true);
    {
      PatchMatchController control(photo_options, control_root, "COLMAP", "");
      ASSERT_NO_THROW(PatchMatchControllerTestAccess::RunPair(
          control, control_hooks, photo_options, 0, 1));
    }
    const auto pair_expected = outputs(control_root);
    ASSERT_EQ(pair_expected.size(), 4u);
    ASSERT_EQ(control_hooks.written_references.size(), 2u);
    ASSERT_NE(control_hooks.written_references[0],
              control_hooks.written_references[1]);
    ASSERT_GE(wrapped_meminfo_calls.load(), 2u)
        << "Pair was not admitted through second owner";
    ASSERT_GE(wrapped_eligible.load(), 2u);

    const auto refused_root = workspace();
    PatchMatchControllerTestAccess::Hooks refused_hooks;
    ResetAllocationInjection(2, cudaErrorMemoryAllocation, false, true);
    {
      PatchMatchController refused(photo_options, refused_root, "COLMAP", "");
      ASSERT_NO_THROW(PatchMatchControllerTestAccess::RunPair(
          refused, refused_hooks, photo_options, 0, 1));
    }
    ASSERT_TRUE(wrapped_fired.load());
    // The second query is in the optional refusal handler, after first-owner
    // admission. The native low-memory path would propagate this fake OOM.
    ASSERT_GE(wrapped_meminfo_calls.load(), 2u);
    EXPECT_EQ(refused_hooks.written_references,
              control_hooks.written_references);
    EXPECT_EQ(outputs(refused_root), pair_expected);

    // Arm before all queries: the second eligible allocation belongs to the
    // mandatory first constructor. It may not be converted to pair fallback.
    const auto mandatory_root = workspace();
    PatchMatchControllerTestAccess::Hooks mandatory_hooks;
    ResetAllocationInjection(2, cudaErrorMemoryAllocation);
    {
      PatchMatchController mandatory(
          photo_options, mandatory_root, "COLMAP", "");
      EXPECT_THROW(PatchMatchControllerTestAccess::RunPair(
                       mandatory, mandatory_hooks, photo_options, 0, 1),
                   colmap::CudaAllocationRefused);
    }
    EXPECT_TRUE(wrapped_fired.load());
    EXPECT_EQ(wrapped_meminfo_calls.load(), 0u);
    EXPECT_TRUE(mandatory_hooks.written_references.empty());
    EXPECT_TRUE(outputs(mandatory_root).empty());

    const auto invalid_root = workspace();
    PatchMatchControllerTestAccess::Hooks invalid_hooks;
    ResetAllocationInjection(2, cudaErrorInvalidValue, false, true);
    try {
      PatchMatchController invalid(photo_options, invalid_root, "COLMAP", "");
      PatchMatchControllerTestAccess::RunPair(
          invalid, invalid_hooks, photo_options, 0, 1);
      FAIL() << "Expected terminal non-allocation error";
    } catch (const colmap::CudaAllocationRefused&) {
      FAIL() << "Non-allocation error must not permit readmission";
    } catch (const std::exception&) {
      SUCCEED();
    }
    EXPECT_TRUE(wrapped_fired.load());
    EXPECT_EQ(wrapped_meminfo_calls.load(), 1u);
    EXPECT_TRUE(invalid_hooks.written_references.empty());
    EXPECT_TRUE(outputs(invalid_root).empty());
  }
}

TEST_F(CudaTest, ControllerPairCleanupFailureIsTerminal) {
  auto photo_options = ControllerWorkspaceFixture::Options();
  photo_options.geom_consistency = false;
  photo_options.filter = false;
  photo_options.sparse_source_sharing = false;
  const auto [width, height] =
      FindControllerFixtureShape(false, photo_options.checkpoint_hmm);
  if (width == 0) GTEST_SKIP() << "Required controller admission unavailable";
  ControllerWorkspaceFixture fixture(width, height);
  const auto workspace = [&] { return fixture.Workspace(); };
  // Re-exec so an unresolved cleanup status never resumes a CUDA context.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      ([&] {
        const auto root = workspace();
        PatchMatchControllerTestAccess::Hooks hooks;
        ResetAllocationInjection(2, cudaErrorMemoryAllocation, true, true);
        try {
          PatchMatchController failed(photo_options, root, "COLMAP", "");
          PatchMatchControllerTestAccess::RunPair(
              failed, hooks, photo_options, 0, 1);
        } catch (const colmap::CudaAllocationRefused&) {
          std::_Exit(21);  // Cleanup failure must override retryable refusal.
        } catch (const std::exception& error) {
          const bool correct =
              wrapped_fired.load() && wrapped_cleanup_injected.load() == 1 &&
              wrapped_meminfo_calls.load() == 1 &&
              hooks.written_references.empty() &&
              std::string(error.what()).find("cleanup") != std::string::npos;
          std::_Exit(correct ? 0 : 22);
        }
        std::_Exit(23);  // Unexpected successful retry.
      }()),
      ::testing::ExitedWithCode(0),
      "");
}

TEST_F(CudaTest, InitialSparseAttachmentCleanupFailureIsTerminal) {
  if (!PatchMatchSparseSources::IsSupported())
    GTEST_SKIP() << "Sparse source arrays are unavailable";
  auto options = ControllerWorkspaceFixture::Options();
  options.sparse_source_sharing = true;
  const auto [width, height] =
      FindControllerFixtureShape(true, options.checkpoint_hmm);
  if (width == 0) GTEST_SKIP() << "Required controller admission unavailable";
  ControllerWorkspaceFixture fixture(width, height);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      ([&] {
        const auto root = fixture.Workspace();
        PatchMatchControllerTestAccess::Hooks hooks;
        ResetAllocationInjection(0, cudaSuccess, false, true);
        wrapped_track_initial_dense = true;
        try {
          PatchMatchController controller(options, root, "COLMAP", "");
          PatchMatchControllerTestAccess::Run(controller, hooks);
        } catch (const AggregateException& errors) {
          bool cleanup_error = false;
          if (errors.exceptions().size() == 1) {
            try {
              std::rethrow_exception(errors.exceptions().front());
            } catch (const CudaAllocationRefused&) {
              std::_Exit(34);
            } catch (const SparseSourceResourceRefused&) {
              std::_Exit(35);
            } catch (const std::runtime_error& error) {
              cleanup_error = std::string(error.what()).find("cleanup") !=
                              std::string::npos;
            } catch (...) {
              std::_Exit(36);
            }
          }
          const bool no_readmission = std::none_of(
              hooks.memory_queries.begin(),
              hooks.memory_queries.end(),
              [](const auto& q) {
                return std::string(q.phase) == "after-initial-sparse-refusal" ||
                       std::string(q.phase) == "pair-or-single-selection";
              });
          const bool correct = cleanup_error && no_readmission &&
                               !wrapped_fired.load() &&
                               wrapped_dense_captures.load() == 1 &&
                               wrapped_dense_cleanup_injected.load() == 1 &&
                               hooks.written_references.empty();
          std::_Exit(correct ? 0 : 31);
        } catch (...) {
          std::_Exit(32);
        }
        std::_Exit(33);
      }()),
      ::testing::ExitedWithCode(0),
      "");
}

TEST_F(CudaTest, ControllerBatchAllocationRefusal) {
  auto options = ControllerWorkspaceFixture::Options();
  const auto [width, height] =
      FindControllerFixtureShape(true, options.checkpoint_hmm);
  if (width == 0) GTEST_SKIP() << "Required controller admission unavailable";
  ControllerWorkspaceFixture fixture(width, height);
  const auto workspace = [&] { return fixture.Workspace(); };
  const auto outputs = ControllerWorkspaceFixture::Outputs;
  const auto control_root = workspace();
  PatchMatchControllerTestAccess::Hooks control_hooks;
  struct ResetOnExit {
    ~ResetOnExit() { ResetAllocationInjection(0, cudaErrorMemoryAllocation); }
  } reset_on_exit;
  ResetAllocationInjection(0, cudaErrorMemoryAllocation, false, true);
  {
    PatchMatchController control(options, control_root, "COLMAP", "");
    PatchMatchControllerTestAccess::Run(control, control_hooks);
  }
  const auto expected = outputs(control_root);
  ASSERT_EQ(expected.size(), 20);
  ASSERT_EQ(control_hooks.written_references.size(), 10);
  std::map<int, size_t> control_writes;
  for (const int ref : control_hooks.written_references) ++control_writes[ref];
  ASSERT_EQ(control_writes.size(), 5);
  for (const auto& item : control_writes) EXPECT_EQ(item.second, 2);
  ASSERT_GT(wrapped_meminfo_calls.load(), 0u);
  ASSERT_GE(wrapped_eligible.load(), 2u)
      << "No second scoped allocation after the first memory query";
  ASSERT_FALSE(wrapped_fired.load());
  {
    const auto root = workspace();
    PatchMatchControllerTestAccess::Hooks hooks;
    // After the first owner and admission query, select the second eligible
    // allocation; a post-refusal marker below must confirm the actual path.
    ResetAllocationInjection(2, cudaErrorMemoryAllocation, false, true);
    PatchMatchController controller(options, root, "COLMAP", "");
    ASSERT_NO_THROW(PatchMatchControllerTestAccess::Run(controller, hooks));
    ASSERT_TRUE(wrapped_fired.load());
    const auto admission = std::find_if(
        hooks.memory_queries.begin(),
        hooks.memory_queries.end(),
        [](const auto& q) { return std::string(q.phase) == "admission"; });
    ASSERT_NE(admission, hooks.memory_queries.end());
    const auto refusal = std::find_if(
        admission + 1, hooks.memory_queries.end(), [](const auto& q) {
          return std::string(q.phase) == "after-optional-cuda-refusal";
        });
    ASSERT_NE(refusal, hooks.memory_queries.end());
    ASSERT_GT(refusal->free_bytes, 0);
    const auto smaller = std::find_if(
        refusal + 1, hooks.memory_queries.end(), [](const auto& q) {
          return std::string(q.phase) == "pair-or-single-selection";
        });
    ASSERT_NE(smaller, hooks.memory_queries.end());
    ASSERT_EQ(hooks.written_references, control_hooks.written_references);
    ASSERT_EQ(outputs(root), expected);
    std::map<int, size_t> writes;
    for (const int ref : hooks.written_references) ++writes[ref];
    EXPECT_EQ(writes, control_writes);
    ResetAllocationInjection(0, cudaErrorMemoryAllocation);
  }
}
#endif

TEST_F(CudaTest, ControllerReadmitsInitialLaterAndGeometryAttachmentRefusals) {
  const bool sparse_supported = PatchMatchSparseSources::IsSupported();
  size_t width = sparse_supported ? 0 : 513;
  size_t height = sparse_supported ? 0 : 257;
  if (sparse_supported) {
    for (const auto& shape : std::vector<std::pair<size_t, size_t> >{
             {513, 257}, {1025, 513}, {1536, 1024}}) {
      const std::vector<std::pair<size_t, size_t> > shapes(5, shape);
      if (CanBatchPatchMatchViews(shapes, false, true) &&
          CanBatchPatchMatchViews(shapes, true, true)) {
        width = shape.first;
        height = shape.second;
        break;
      }
    }
    ASSERT_GT(width, 0)
        << "Expected supported lane must admit a bounded fixture";
  }
  ControllerWorkspaceFixture fixture(width, height);
  const auto workspace = [&] { return fixture.Workspace(); };
  const auto outputs = ControllerWorkspaceFixture::Outputs;
  auto options = ControllerWorkspaceFixture::Options();
  {
    const auto root = workspace();
    options.sparse_source_sharing = true;
    PatchMatchController probe(options, root, "COLMAP", "");
    EXPECT_EQ(PatchMatchControllerTestAccess::PreparedBitmapCounts(probe, true),
              (std::array<size_t, 3>{1, 4, 4}));
    options.sparse_source_sharing = false;
    PatchMatchController disabled(options, root, "COLMAP", "");
    EXPECT_EQ(
        PatchMatchControllerTestAccess::PreparedBitmapCounts(disabled, true),
        (std::array<size_t, 3>{1, 0, 0}));
  }
  const auto control_root = workspace();
  PatchMatchControllerTestAccess::Hooks control_hooks;
  {
    PatchMatchController control(options, control_root, "COLMAP", "");
    PatchMatchControllerTestAccess::Run(control, control_hooks);
  }
  const auto expected = outputs(control_root);
  ASSERT_EQ(expected.size(), 20);
  ASSERT_EQ(control_hooks.written_references.size(), 10);
  std::map<int, size_t> control_writes;
  for (const int ref : control_hooks.written_references) ++control_writes[ref];
  ASSERT_EQ(control_writes.size(), 5);
  for (const auto& item : control_writes) EXPECT_EQ(item.second, 2);

  options.sparse_source_sharing = true;
  if (!sparse_supported) {
    const auto root = workspace();
    PatchMatchControllerTestAccess::Hooks hooks;
    PatchMatchController controller(options, root, "COLMAP", "");
    PatchMatchControllerTestAccess::Run(controller, hooks);
    EXPECT_EQ(outputs(root), expected);
    EXPECT_EQ(hooks.written_references, control_hooks.written_references);
    return;
  }
  struct Case {
    size_t texture;
    bool geometry;
  };
  // Photo has five image views; geometry then has two views per reference.
  // Thus9/13 target second/later depth, and16/17 target the reconstructed
  // geometry-fallback owner's image/depth after its one-shot admission refusal.
  for (const auto& test : std::vector<Case>{{1, false},
                                            {2, false},
                                            {4, false},
                                            {9, false},
                                            {13, false},
                                            {16, true},
                                            {17, true}}) {
    SCOPED_TRACE(::testing::Message() << "texture " << test.texture);
    const auto root = workspace();
    SparseSourceTestFailure failure{"cudaCreateTextureObject sparse",
                                    cudaErrorMemoryAllocation,
                                    test.texture};
    PatchMatchControllerTestAccess::Hooks hooks;
    hooks.failure = &failure;
    hooks.refuse_geometry_admission_once = test.geometry;
    {
      PatchMatchController controller(options, root, "COLMAP", "");
      PatchMatchControllerTestAccess::Run(controller, hooks);
    }
    EXPECT_GE(failure.calls, test.texture);
    EXPECT_EQ(outputs(root), expected);
    EXPECT_EQ(hooks.written_references, control_hooks.written_references);
    EXPECT_EQ(hooks.geometry_fallback_entries, test.geometry ? 1 : 0);
    const std::string expected_phase = test.texture == 1
                                           ? "after-initial-sparse-refusal"
                                           : "after-later-sparse-refusal";
    const auto refusal =
        std::find_if(hooks.memory_queries.begin(),
                     hooks.memory_queries.end(),
                     [&](const auto& q) { return q.phase == expected_phase; });
    ASSERT_NE(refusal, hooks.memory_queries.end());
    EXPECT_GT(refusal->free_bytes, 0);
    const auto smaller_queries = std::count_if(
        refusal + 1, hooks.memory_queries.end(), [](const auto& q) {
          return std::string(q.phase) == "pair-or-single-selection";
        });
    // Five references necessarily include a final singleton even if pairs fit.
    EXPECT_GE(smaller_queries, 3);
  }
}

TEST_F(CudaTest, SparseSourceCleanupFailureNeverPermitsReadmission) {
  if (!PatchMatchSparseSources::IsSupported()) {
    GTEST_SKIP() << "Sparse source arrays are unavailable on this CUDA "
                 << "runtime/device";
  }
  // Re-exec a fresh process: do not resume a CUDA context after fork or after
  // unresolved ownership. This test exits immediately on its terminal error.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      ([] {
        Inputs input = MakeInputs(0, 9, 257, 513);
        std::vector<std::shared_ptr<const Bitmap> > bitmaps;
        std::vector<std::shared_ptr<const DepthMap> > depths;
        for (const auto& image : input.images)
          bitmaps.push_back(std::make_shared<Bitmap>(image.GetBitmap()));
        for (const auto& depth : input.depths)
          depths.push_back(std::make_shared<DepthMap>(depth));
        std::vector<PatchMatch::Problem> problems(3);
        std::vector<const PatchMatch::Problem*> pointers;
        for (int ref = 0; ref < 3; ++ref) {
          auto& p = problems[ref];
          p.ref_image_idx = ref;
          p.images = &input.images;
          p.depth_maps = &input.depths;
          p.source_bitmap_leases = &bitmaps;
          p.source_depth_leases = &depths;
          for (int id = 0; id < static_cast<int>(input.images.size()); ++id)
            if (id != ref) p.src_image_idxs.push_back(id);
          pointers.push_back(&p);
        }
        SparseSourceTestFailure cleanup{"cudaFreeArray cleanup",
                                        cudaErrorUnknown};
        SparseSourceTestFailure staging{"host source staging", 1};
        staging.next = &cleanup;
        try {
          (void)PatchMatchSparseSources::Create(pointers, true, &staging);
        } catch (const SparseSourceResourceRefused&) {
          std::_Exit(3);
        } catch (const std::exception& error) {
          std::_Exit(std::string(error.what()).find("cleanup") !=
                             std::string::npos
                         ? 0
                         : 4);
        }
        std::_Exit(5);
      }()),
      ::testing::ExitedWithCode(0),
      "");
}

TEST_F(CudaTest, SparseSourceAliasesPreserveOrderedPaddedOutputs) {
  // One workspace identity; five references reuse differently ordered subsets.
  // Varying source dimensions exercise the original zero-padded layer extent.
  Inputs input = MakeInputs(0, 9, 257, 513);
  std::vector<std::shared_ptr<const Bitmap> > bitmap_leases;
  std::vector<std::shared_ptr<const DepthMap> > depth_leases;
  for (const auto& image : input.images)
    bitmap_leases.push_back(std::make_shared<Bitmap>(image.GetBitmap()));
  for (const auto& depth : input.depths)
    depth_leases.push_back(std::make_shared<DepthMap>(depth));
  for (const bool geometry : {false, true}) {
    for (const bool reuse : {false, true}) {
      auto options = MakeOptions(geometry, geometry, reuse);
      const PatchMatchOptions defaults;
      options.window_radius = defaults.window_radius;
      options.window_step = defaults.window_step;
      options.num_samples = defaults.num_samples;
      options.num_iterations = defaults.num_iterations;
      options.sigma_spatial = options.window_radius;
      // Native PrepareProblem owns different vectors for each reference.
      std::vector<std::vector<Image> > images(5, input.images);
      std::vector<std::vector<DepthMap> > depths(5, input.depths);
      std::vector<PatchMatch::Problem> problems(5);
      std::vector<const PatchMatch::Problem*> pointers;
      for (int ref = 0; ref < 5; ++ref) {
        auto& problem = problems[ref];
        problem.ref_image_idx = ref;
        problem.images = &images[ref];
        problem.depth_maps = &depths[ref];
        problem.source_bitmap_leases = &bitmap_leases;
        problem.source_depth_leases = &depth_leases;
        problem.normal_maps = &input.normals;
        for (int source = 0; source < 10; ++source)
          if (source != ref) problem.src_image_idxs.push_back(source);
        std::rotate(problem.src_image_idxs.begin(),
                    problem.src_image_idxs.begin() + ref,
                    problem.src_image_idxs.end());
        pointers.push_back(&problem);
      }
      EXPECT_EQ(PatchMatchSparseSources::Budget({nullptr, nullptr, nullptr},
                                                geometry),
                0);
      auto invalid = problems[0];
      invalid.images = nullptr;
      EXPECT_EQ(PatchMatchSparseSources::Budget(
                    {&invalid, &problems[1], &problems[2]}, geometry),
                0);
      invalid = problems[0];
      invalid.src_image_idxs.clear();
      EXPECT_EQ(PatchMatchSparseSources::Budget(
                    {&invalid, &problems[1], &problems[2]}, geometry),
                0);
      using Snapshot =
          std::tuple<DepthMap, NormalMap, Mat<float>, std::vector<int> >;
      const auto read = [](const PatchMatchCuda& owner) {
        return Snapshot(owner.GetDepthMap(),
                        owner.GetNormalMap(),
                        owner.GetSelProbMap(),
                        owner.GetConsistentImageIdxs());
      };
      std::vector<Snapshot> expected;
      for (const auto& problem : problems) {
        PatchMatchCuda independent(options, problem);
        independent.Run();
        expected.push_back(read(independent));
        const auto& depth = std::get<0>(expected.back()).GetData();
        EXPECT_TRUE(std::any_of(depth.begin(), depth.end(), [](float value) {
          return value > 0.0f;
        }));
      }
      if (!PatchMatchSparseSources::IsSupported()) {
        EXPECT_EQ(PatchMatchSparseSources::Budget(pointers, geometry), 0);
        EXPECT_FALSE(PatchMatchSparseSources::Create(pointers, geometry));
        continue;
      }
      ASSERT_GT(PatchMatchSparseSources::Budget(pointers, geometry), 0);
      auto shared = PatchMatchSparseSources::Create(pointers, geometry);
      ASSERT_TRUE(shared);
      const auto bitmap_generation = bitmap_leases[7];
      bitmap_leases[7] = std::make_shared<Bitmap>(*bitmap_generation);
      EXPECT_FALSE(shared->Matches(problems[0], geometry));
      bitmap_leases[7] = bitmap_generation;
      EXPECT_TRUE(shared->Matches(problems[0], geometry));
      if (geometry) {
        const auto depth_generation = depth_leases[7];
        depth_leases[7] = std::make_shared<DepthMap>(*depth_generation);
        EXPECT_FALSE(shared->Matches(problems[0], true));
        depth_leases[7] = depth_generation;
      }
      std::swap(problems[0].src_image_idxs[0], problems[0].src_image_idxs[1]);
      EXPECT_FALSE(shared->Matches(problems[0], geometry));
      std::swap(problems[0].src_image_idxs[0], problems[0].src_image_idxs[1]);
      std::vector<std::unique_ptr<PatchMatchCuda> > owners;
      owners.push_back(std::make_unique<PatchMatchCuda>(options, problems[0]));
      owners[0]->UseSparseSources(shared);
      for (size_t i = 1; i < problems.size(); ++i)
        owners.push_back(
            std::make_unique<PatchMatchCuda>(options, problems[i], shared));
      shared.reset();  // Owners, not the caller, must retain the bank.
      std::vector<PatchMatchCuda*> views;
      std::vector<cudaStream_t> streams(owners.size(), nullptr);
      struct StreamCleanup {
        std::vector<cudaStream_t>& values;
        ~StreamCleanup() {
          for (auto stream : values) {
            if (stream) {
              (void)cudaStreamSynchronize(stream);
              (void)cudaStreamDestroy(stream);
            }
          }
        }
      } cleanup{streams};
      for (auto& stream : streams)
        CUDA_SAFE_CALL(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      for (auto& owner : owners) views.push_back(owner.get());
      RunPatchMatchBatch(views, streams);
      for (size_t i = 0; i < owners.size(); ++i) {
        const auto actual = read(*owners[i]);
        ExpectExactlyEqual(std::get<0>(expected[i]), std::get<0>(actual));
        ExpectExactlyEqual(std::get<1>(expected[i]), std::get<1>(actual));
        ExpectExactlyEqual(std::get<2>(expected[i]), std::get<2>(actual));
        EXPECT_EQ(std::get<3>(expected[i]), std::get<3>(actual));
      }
    }
  }
}

TEST_F(CudaTest, ImmutableSourceDepthLeasesPreserveThreeViewGeoOutput) {
  Inputs input = MakeInputs(0, 9, 257, 513);
  auto options = MakeOptions(true, true, false);
  // The production batch path admits the current native PatchMatch defaults.
  // This fixture compares copied and leased source maps under that same path.
  options.window_radius = 5;
  options.num_samples = 15;
  options.num_iterations = 5;
  constexpr int kReferences = 3;
  std::vector<std::shared_ptr<const DepthMap> > canonical;
  std::vector<std::shared_ptr<const Bitmap> > bitmap_leases;
  for (const auto& image : input.images)
    bitmap_leases.push_back(std::make_shared<Bitmap>(image.GetBitmap()));
  for (const auto& depth : input.depths)
    canonical.push_back(std::make_shared<DepthMap>(depth));
  std::vector<std::vector<DepthMap> > old_depths(kReferences, input.depths);
  std::vector<std::vector<DepthMap> > owned_depths(
      kReferences, std::vector<DepthMap>(input.depths.size()));
  std::vector<std::vector<std::shared_ptr<const DepthMap> > > leases(
      kReferences,
      std::vector<std::shared_ptr<const DepthMap> >(input.depths.size()));
  std::vector<PatchMatch::Problem> old_problems(kReferences),
      new_problems(kReferences);
  std::vector<const PatchMatch::Problem*> new_pointers;
  for (int ref = 0; ref < kReferences; ++ref) {
    auto& old = old_problems[ref];
    auto& candidate = new_problems[ref];
    old.ref_image_idx = candidate.ref_image_idx = ref;
    old.images = candidate.images = &input.images;
    candidate.source_bitmap_leases = &bitmap_leases;
    old.normal_maps = candidate.normal_maps = &input.normals;
    old.depth_maps = &old_depths[ref];
    candidate.depth_maps = &owned_depths[ref];
    candidate.source_depth_leases = &leases[ref];
    owned_depths[ref][ref] = input.depths[ref];
    for (int id = 0; id < static_cast<int>(input.depths.size()); ++id) {
      if (id == ref) continue;
      old.src_image_idxs.push_back(id);
      candidate.src_image_idxs.push_back(id);
      leases[ref][id] = canonical[id];
      ASSERT_EQ(candidate.InputDepthMap(id).GetPtr(), canonical[id]->GetPtr());
    }
    new_pointers.push_back(&candidate);
  }

  using Snapshot =
      std::tuple<DepthMap, NormalMap, Mat<float>, std::vector<int> >;
  const auto read = [](const PatchMatchCuda& owner) {
    return Snapshot(owner.GetDepthMap(),
                    owner.GetNormalMap(),
                    owner.GetSelProbMap(),
                    owner.GetConsistentImageIdxs());
  };
  std::vector<Snapshot> expected;
  for (const auto& problem : old_problems) {
    PatchMatchCuda owner(options, problem);
    owner.Run();
    expected.push_back(read(owner));
  }

  // Lease semantics are independent of sparse-array capability. Run this
  // comparison on older toolkits/devices too, before the sparse-only section.
  for (int ref = 0; ref < kReferences; ++ref) {
    PatchMatchCuda leased(options, new_problems[ref]);
    leased.Run();
    const auto actual = read(leased);
    ExpectExactlyEqual(std::get<0>(expected[ref]), std::get<0>(actual));
    ExpectExactlyEqual(std::get<1>(expected[ref]), std::get<1>(actual));
    ExpectExactlyEqual(std::get<2>(expected[ref]), std::get<2>(actual));
    EXPECT_EQ(std::get<3>(expected[ref]), std::get<3>(actual));
  }
  if (!PatchMatchSparseSources::IsSupported()) {
    EXPECT_EQ(PatchMatchSparseSources::Budget(new_pointers, true), 0);
    EXPECT_FALSE(PatchMatchSparseSources::Create(new_pointers, true));
    return;
  }
  ASSERT_GT(PatchMatchSparseSources::Budget(new_pointers, true), 0);
  auto shared = PatchMatchSparseSources::Create(new_pointers, true);
  ASSERT_TRUE(shared);
  std::vector<std::unique_ptr<PatchMatchCuda> > owners;
  owners.push_back(std::make_unique<PatchMatchCuda>(options, new_problems[0]));
  owners[0]->UseSparseSources(shared);
  for (int ref = 1; ref < kReferences; ++ref)
    owners.push_back(
        std::make_unique<PatchMatchCuda>(options, new_problems[ref], shared));
  shared.reset();
  std::vector<PatchMatchCuda*> views;
  std::vector<cudaStream_t> streams(kReferences, nullptr);
  struct StreamCleanup {
    std::vector<cudaStream_t>& values;
    ~StreamCleanup() {
      for (auto stream : values) {
        if (stream) {
          (void)cudaStreamSynchronize(stream);
          (void)cudaStreamDestroy(stream);
        }
      }
    }
  } cleanup{streams};
  for (auto& stream : streams)
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (auto& owner : owners) views.push_back(owner.get());
  RunPatchMatchBatch(views, streams);
  for (int ref = 0; ref < kReferences; ++ref) {
    const auto actual = read(*owners[ref]);
    ExpectExactlyEqual(std::get<0>(expected[ref]), std::get<0>(actual));
    ExpectExactlyEqual(std::get<1>(expected[ref]), std::get<1>(actual));
    ExpectExactlyEqual(std::get<2>(expected[ref]), std::get<2>(actual));
    EXPECT_EQ(std::get<3>(expected[ref]), std::get<3>(actual));
  }
}

TEST_F(CudaTest, BatchAdmissionCanBeatIneligiblePairs) {
  // Search grid widths against the queried device rather than baking in an
  // sm86/SM-count cutoff. This exercises the large-grid path where the
  // incumbent schedules singletons but the five-view aggregate saves waves.
  bool found = false;
  for (size_t width = 64; width <= 65536; width += 64) {
    const std::vector<std::pair<size_t, size_t> > shapes(5, {width, 3});
    if (!CanPairPatchMatchViews(width, 3, width, 3, false) &&
        CanBatchPatchMatchViews(shapes, false)) {
      found = true;
      break;
    }
  }
  if (!found) GTEST_SKIP() << "No beneficial five-grid case on this device";
  EXPECT_TRUE(found);
}

TEST_F(CudaTest, BatchAdmissionRejectsUnsupportedGroupSizes) {
  EXPECT_FALSE(CanBatchPatchMatchViews({{47, 149}, {39, 153}}, false));
  EXPECT_FALSE(CanBatchPatchMatchViews(
      std::vector<std::pair<size_t, size_t> >(6, {47, 149}), false));
  EXPECT_FALSE(
      CanBatchPatchMatchViews({{47, 149}, {0, 153}, {47, 149}}, false));
}

TEST_F(CudaTest, ExplicitAndDefaultStreamHelpers) {
  // Preserve cleanup even when a CUDA allocation or operation assertion fails.
  struct Resources {
    cudaStream_t stream = nullptr;
    uint8_t* device = nullptr;
    uint8_t* host = nullptr;
    ~Resources() {
      if (stream != nullptr) {
        cudaStreamSynchronize(stream);
      }
      cudaStreamSynchronize(CudaDefaultStream());
      if (host != nullptr) {
        cudaFreeHost(host);
      }
      if (device != nullptr) {
        cudaFree(device);
      }
      if (stream != nullptr) {
        cudaStreamDestroy(stream);
      }
    }
  } resources;
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&resources.device), 1),
            cudaSuccess);
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void**>(&resources.host), 1),
            cudaSuccess);
  {
    CudaTimer timer(resources.stream);
    ASSERT_EQ(cudaMemsetAsync(resources.device, 0x5a, 1, resources.stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(resources.host,
                              resources.device,
                              1,
                              cudaMemcpyDeviceToHost,
                              resources.stream),
              cudaSuccess);
    CudaSyncAndCheck(__FILE__, __LINE__, resources.stream);
    EXPECT_EQ(resources.host[0], 0x5a);
    timer.Print("Explicit nonblocking stream helper test");
  }
  EXPECT_EQ(CudaDefaultStream(), cudaStreamPerThread);
  {
    CudaTimer timer;
    ASSERT_EQ(cudaMemsetAsync(resources.device, 0xa5, 1, CudaDefaultStream()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(resources.host,
                              resources.device,
                              1,
                              cudaMemcpyDeviceToHost,
                              CudaDefaultStream()),
              cudaSuccess);
    CudaSyncAndCheck(__FILE__, __LINE__);
    EXPECT_EQ(resources.host[0], 0xa5);
    timer.Print("Per-thread default stream helper test");
  }
}

TEST_F(CudaTest, WideWindowsPreserveOutputs) {
  Inputs inputs = MakeInputs(0);
  // Cover both boundaries between the 8-, 16-, and 32-column tiles.
  for (const int radius : {8, 9, 16, 17}) {
    SCOPED_TRACE(radius);
    PatchMatchOptions options = MakeOptions(false, false, false);
    options.window_radius = radius;
    options.num_iterations = 1;
    options.num_samples = 1;
    const Result original = RunPatchMatch(&inputs, options);
    options.rotation_buffer_reuse = true;
    const Result reused = RunPatchMatch(&inputs, options);
    ExpectNonempty(original, inputs, false);
    ExpectNonempty(reused, inputs, false);
    ExpectEquivalent(original, reused);
  }
}

TEST_F(CudaTest, TallInitialCostStripeTailsPreserveOutputs) {
  // The two odd reference heights leave 21- and 25-row final stripes. Odd
  // widths also retain the existing pitched-storage and border-lane coverage.
  for (const int variant : {0, 1}) {
    Inputs inputs = MakeInputs(variant, 2, variant == 0 ? 149 : 153);
    for (const int radius : {2, 5, 9, 17}) {
      SCOPED_TRACE(::testing::Message()
                   << "variant=" << variant << " radius=" << radius);
      PatchMatchOptions options = MakeOptions(false, false, false);
      options.window_radius = radius;
      options.sigma_spatial = radius;
      options.num_iterations = 1;
      options.num_samples = 1;
      const Result original = RunPatchMatch(&inputs, options);
      options.rotation_buffer_reuse = true;
      const Result reused = RunPatchMatch(&inputs, options);
      ExpectNonempty(original, inputs, false);
      ExpectNonempty(reused, inputs, false);
      ExpectEquivalent(original, reused);
    }
  }
}

TEST_F(CudaTest, GeometricCostGroupsPreserveOutputs) {
  Inputs inputs = MakeInputs(0, 5);
  // Cover shared geometric costs in both narrow and wide cooperative tiles.
  for (const int radius : {5, 17}) {
    for (const bool filter : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "radius=" << radius << ", filter=" << filter);
      PatchMatchOptions options = MakeOptions(true, filter, false);
      options.window_radius = radius;
      options.num_iterations = 1;
      options.num_samples = 1;
      // The wide case targets shared-result safety rather than patch workload.
      if (radius == 17) options.window_step = 2;
      const Result original = RunPatchMatch(&inputs, options);
      options.rotation_buffer_reuse = true;
      const Result reused = RunPatchMatch(&inputs, options);
      ExpectNonempty(original, inputs, filter);
      ExpectNonempty(reused, inputs, filter);
      ExpectEquivalent(original, reused);
    }
  }
}

TEST_F(CudaTest, EvenSampleGroupsPreserveOutputs) {
  // One source forces duplicate picks; seventeen cover all groups and a wrap.
  for (const int num_sources : {1, 17}) {
    SCOPED_TRACE(num_sources);
    Inputs inputs = MakeInputs(0, num_sources);
    for (const int num_samples : {2, 4}) {
      SCOPED_TRACE(num_samples);
      PatchMatchOptions options = MakeOptions(true, true, false);
      options.num_iterations = 1;
      options.num_samples = num_samples;
      const Result original = RunPatchMatch(&inputs, options);
      options.rotation_buffer_reuse = true;
      const Result reused = RunPatchMatch(&inputs, options);
      ExpectNonempty(original, inputs, true);
      ExpectNonempty(reused, inputs, true);
      ExpectEquivalent(original, reused);
    }
  }
}

TEST_F(CudaTest, SourcePackingPreservesDrawIdentityAndOrderedFold) {
  constexpr int kLanes = 8;
  for (const int sources : {32, 33}) {
    std::vector<float> values(sources);
    for (int source = 0; source < sources; ++source) {
      values[source] = (source % 7 + 1) * 0.125f;
    }
    values[0] = 1.0f;
    values[31] = std::ldexp(1.0f, -24);
    values[17] = 0.125f;
    values[1] = 0.75f;
    const auto check = [&](const std::vector<int>& pattern) {
      const size_t count = pattern.size();
      std::vector<int> draws(count * kLanes);
      std::vector<float> expected(kLanes, 0.0f);
      for (int lane = 0; lane < kLanes; ++lane) {
        for (size_t sample = 0; sample < count; ++sample) {
          const int original = pattern[sample];
          const int source = original < 0 ? -1 : (original + lane) % sources;
          draws[lane * count + sample] = source;
          // Independent scalar oracle: no packing, deduplication, ranking,
          // masks or alias mapping, and no production fold helper.
          if (source >= 0) expected[lane] += values[source];
        }
      }
      for (const auto& config :
           {std::pair{false, 1}, std::pair{true, 1}, std::pair{true, 16}}) {
        SCOPED_TRACE(::testing::Message()
                     << "sources=" << sources << " samples=" << count
                     << " cooperative=" << config.first
                     << " window=" << config.second);
        const auto actual = internal::ProbeSampledSources(
            draws, values, config.second, config.first);
        ASSERT_EQ(actual.first, draws);
        ASSERT_EQ(actual.second.size(), expected.size());
        for (int lane = 0; lane < kLanes; ++lane) {
          EXPECT_EQ(actual.second[lane], expected[lane]) << lane;
        }
      }
    };
    const std::array<int, 6> seed = {31, 0, 31, 17, -1, 1};
    for (const size_t samples : {15, 16, 17}) {
      std::vector<int> pattern(samples);
      for (size_t i = 0; i < samples; ++i) pattern[i] = seed[i % seed.size()];
      if (sources == 33) pattern[7] = 32;
      pattern.back() = -1;
      check(pattern);
    }
    check(std::vector<int>(17, -1));
    // Confirm that this fixture detects addition reordering in the valid,
    // finite nonnegative cost range instead of using only exact sums.
    float ordered = 0;
    for (const int source : {0, 31, 31}) ordered += values[source];
    float reordered = 0;
    for (const int source : {31, 31, 0}) reordered += values[source];
    ASSERT_NE(ordered, reordered);
    check({0, 31, 31});
  }
}

TEST_F(CudaTest, SampleWindowBoundariesPreserveOutputs) {
  // Cover source-mask capacity and its larger-source fallback.
  for (const int num_sources : {1, 17, 32, 33}) {
    SCOPED_TRACE(num_sources);
    Inputs inputs = MakeInputs(0, num_sources);
    for (const int num_samples : {15, 16, 17}) {
      SCOPED_TRACE(num_samples);
      PatchMatchOptions options = MakeOptions(true, true, false);
      options.num_iterations = 1;
      options.num_samples = num_samples;
      const Result original = RunPatchMatch(&inputs, options);
      options.rotation_buffer_reuse = true;
      const Result reused = RunPatchMatch(&inputs, options);
      ExpectNonempty(original, inputs, true);
      ExpectNonempty(reused, inputs, true);
      ExpectEquivalent(original, reused);
    }
  }
}

TEST_F(CudaTest, DuplicatedSourcesPreservePhotoGeometryAcrossCacheBoundary) {
  // The same source bitmap, calibration, and pose occupy every source layer.
  // Sampling can choose different IDs as the CDF length changes, but each
  // selected source contributes the same photo cost. Compare geometry only:
  // selection maps necessarily have different source dimensions.
  const Inputs seed = MakeInputs(0, 1);
  const auto duplicate_source = [&seed](const int num_sources) {
    Inputs inputs;
    inputs.images.push_back(seed.images[0]);
    inputs.depths.push_back(seed.depths[0]);
    inputs.normals.push_back(seed.normals[0]);
    for (int i = 0; i < num_sources; ++i) {
      inputs.images.push_back(seed.images[1]);
      inputs.depths.push_back(seed.depths[1]);
      inputs.normals.push_back(seed.normals[1]);
    }
    return inputs;
  };

  PatchMatchOptions options = MakeOptions(false, false, false);
  options.window_radius = 5;
  options.window_step = 1;
  options.num_samples = 15;
  options.num_iterations = 5;
  options.sigma_spatial = options.window_radius;

  Inputs nine_sources = duplicate_source(9);
  Inputs thirty_three_sources = duplicate_source(33);
  const Result nine = RunPatchMatch(&nine_sources, options);
  const Result thirty_three = RunPatchMatch(&thirty_three_sources, options);
  ExpectNonempty(nine, nine_sources, false);
  ExpectNonempty(thirty_three, thirty_three_sources, false);
  EXPECT_EQ(nine.selection.GetDepth(), 9);
  EXPECT_EQ(thirty_three.selection.GetDepth(), 33);
  ExpectExactlyEqual(nine.depths, thirty_three.depths);
  ExpectExactlyEqual(nine.normals, thirty_three.normals);
}

TEST_P(PatchMatchCudaTest, CheckpointHmmPreservesAlgorithmOutputs) {
  const auto [geometry, filter] = GetParam();
  for (const int num_sources : {1, 9, 17}) {
    for (const bool reuse : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "sources=" << num_sources << ", reuse=" << reuse);
      // The 47x35 reference has stripe tails in both orientations, unequal
      // plane strides, and odd sizes; source counts also cross packed tails.
      Inputs inputs = MakeInputs(0, num_sources);
      PatchMatchOptions options = MakeOptions(geometry, filter, reuse);
      options.window_radius = 5;
      options.checkpoint_hmm = false;
      const Result original = RunPatchMatch(&inputs, options);
      options.checkpoint_hmm = true;
      const Result checkpointed = RunPatchMatch(&inputs, options);
      ExpectNonempty(original, inputs, filter);
      ExpectNonempty(checkpointed, inputs, filter);
      ExpectEquivalent(original, checkpointed);
    }
  }
}

TEST_P(PatchMatchCudaTest, UnevenSourceGroupsPreserveOutputs) {
  const auto [geometry, filter] = GetParam();
  // Five sources exercise every cooperative group and a second source in one
  // group. Odd image sizes also leave invalid lanes at the tile boundary.
  Inputs inputs = MakeInputs(0, 5);
  PatchMatchOptions options = MakeOptions(geometry, filter, false);
  const Result original = RunPatchMatch(&inputs, options);
  options.rotation_buffer_reuse = true;
  const Result reused = RunPatchMatch(&inputs, options);
  ExpectNonempty(original, inputs, filter);
  ExpectNonempty(reused, inputs, filter);
  ExpectEquivalent(original, reused);
  // Allow FP32 rounding at the probability bounds; parity above stays exact.
  constexpr float kProbabilityTolerance = 1e-6f;
  for (const float probability : original.selection.GetData()) {
    EXPECT_GE(probability, -kProbabilityTolerance);
    EXPECT_LE(probability, 1.0f + kProbabilityTolerance);
  }
}

TEST_P(PatchMatchCudaTest, RotationBufferReusePreservesAlgorithmOutputs) {
  const auto [geometry, filter] = GetParam();
  Inputs inputs = MakeInputs(0);
  const Result legacy =
      RunPatchMatch(&inputs, MakeOptions(geometry, filter, false));
  const Result reused =
      RunPatchMatch(&inputs, MakeOptions(geometry, filter, true));
  ExpectNonempty(legacy, inputs, filter);
  ExpectNonempty(reused, inputs, filter);
  ExpectEquivalent(legacy, reused);
}

TEST_P(PatchMatchCudaTest, SeparateCalibrationsMatchSerialOnSameDevice) {
  const auto [geometry, filter] = GetParam();
  for (const bool reuse : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "rotation_buffer_reuse=" << reuse);
    const PatchMatchOptions options = MakeOptions(geometry, filter, reuse);
    Inputs serial_a = MakeInputs(0);
    Inputs serial_b = MakeInputs(1);
    ASSERT_NE(serial_a.images[0].GetK()[0], serial_b.images[0].GetK()[0]);
    ASSERT_NE(serial_a.images[0].GetK()[4], serial_b.images[0].GetK()[4]);
    const Result expected_a = RunPatchMatch(&serial_a, options);
    const Result expected_b = RunPatchMatch(&serial_b, options);
    ExpectNonempty(expected_a, serial_a, filter);
    ExpectNonempty(expected_b, serial_b, filter);
    // A bounded repetition increases the opportunity for interleaving without
    // sleeps, external datasets, hooks into kernels, or a stress-test workload.
    for (int repetition = 0; repetition < 2; ++repetition) {
      SCOPED_TRACE(repetition);
      Inputs concurrent_a = MakeInputs(0);
      Inputs concurrent_b = MakeInputs(1);
      std::promise<void> start;
      std::promise<void> entered_a;
      std::promise<void> entered_b;
      const std::shared_future<void> ready = start.get_future().share();
      auto a = std::async(std::launch::async, [&]() {
        entered_a.set_value();
        ready.wait();
        return RunPatchMatch(&concurrent_a, options);
      });
      auto b = std::async(std::launch::async, [&]() {
        entered_b.set_value();
        ready.wait();
        return RunPatchMatch(&concurrent_b, options);
      });
      entered_a.get_future().wait();
      entered_b.get_future().wait();
      start.set_value();
      const Result actual_a = a.get();
      const Result actual_b = b.get();
      ExpectNonempty(actual_a, concurrent_a, filter);
      ExpectNonempty(actual_b, concurrent_b, filter);
      ExpectEquivalent(expected_a, actual_a);
      ExpectEquivalent(expected_b, actual_b);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(PhotoAndGeometry,
                         PatchMatchCudaTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));
INSTANTIATE_TEST_SUITE_P(PhotoAndGeometry, SourceUploadTest, ::testing::Bool());
#endif

}  // namespace
}  // namespace mvs
}  // namespace colmap
