// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#ifdef CASPAR_USE_DOUBLE
#include "thirdparty/Symforce-Caspar/generated/f64/shared_indices.h"
#else
#include "thirdparty/Symforce-Caspar/generated/f32/shared_indices.h"
#endif

#if defined(COLMAP_TEST_CASPAR_READBACK_WRAPPING)
#ifdef CASPAR_USE_DOUBLE
#include "thirdparty/Symforce-Caspar/generated/f64/solver_tools.h"
#else
#include "thirdparty/Symforce-Caspar/generated/f32/solver_tools.h"
#endif

namespace {
enum class CudaCall { kNone, kSelect, kAllocate, kFree, kMap, kCopy };
struct CudaTrace {
  std::vector<CudaCall> calls;
  CudaCall fault = CudaCall::kNone;
  bool fired = false;
  cudaError_t real_mapping_status = cudaSuccess;
  cudaMemcpyKind copy_kind = cudaMemcpyDeviceToHost;
  bool host_allocation = false;
  std::vector<uint8_t> allocation;
  bool freed_expected_pointer = false;
};
thread_local CudaTrace* cuda_trace = nullptr;
constexpr cudaError_t kInjectedCudaError = cudaErrorInvalidValue;

class ScopedCudaFault {
 public:
  ScopedCudaFault(CudaTrace& trace, CudaCall fault) {
    if (cuda_trace != nullptr) throw std::logic_error("nested fault arm");
    trace.fault = fault;
    cuda_trace = &trace;
  }
  ~ScopedCudaFault() { cuda_trace = nullptr; }
  ScopedCudaFault(const ScopedCudaFault&) = delete;
  ScopedCudaFault& operator=(const ScopedCudaFault&) = delete;
};
}  // namespace

extern "C" cudaError_t CUDARTAPI __real_cudaSetDevice(int);
extern "C" cudaError_t CUDARTAPI __real_cudaGetLastError();
extern "C" cudaError_t CUDARTAPI __real_cudaMalloc(void**, size_t);
extern "C" cudaError_t CUDARTAPI __real_cudaFree(void*);
extern "C" cudaError_t CUDARTAPI __real_cudaMemcpy(void*,
                                                   const void*,
                                                   size_t,
                                                   cudaMemcpyKind);

extern "C" cudaError_t CUDARTAPI __wrap_cudaSetDevice(int device) {
  if (cuda_trace != nullptr) {
    cuda_trace->calls.push_back(CudaCall::kSelect);
    if (cuda_trace->fault == CudaCall::kSelect) {
      cuda_trace->fired = true;
      return kInjectedCudaError;
    }
    if (cuda_trace->host_allocation) return cudaSuccess;
  }
  return __real_cudaSetDevice(device);
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaMalloc(void** pointer,
                                                   size_t bytes) {
  if (cuda_trace != nullptr && cuda_trace->host_allocation) {
    cuda_trace->calls.push_back(CudaCall::kAllocate);
    // Even the failing control gets valid backing storage: the old unchecked
    // constructor can form its pointers without relying on undefined behavior.
    cuda_trace->allocation.resize(bytes);
    *pointer = cuda_trace->allocation.data();
    if (cuda_trace->fault == CudaCall::kAllocate) {
      cuda_trace->fired = true;
      return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
  }
  return __real_cudaMalloc(pointer, bytes);
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaFree(void* pointer) {
  if (cuda_trace != nullptr && cuda_trace->host_allocation) {
    cuda_trace->calls.push_back(CudaCall::kFree);
    cuda_trace->freed_expected_pointer =
        pointer == cuda_trace->allocation.data();
    return cudaSuccess;
  }
  return __real_cudaFree(pointer);
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaGetLastError() {
  const cudaError_t actual = __real_cudaGetLastError();
  if (cuda_trace != nullptr) {
    cuda_trace->calls.push_back(CudaCall::kMap);
    cuda_trace->real_mapping_status = actual;
    if (cuda_trace->fault == CudaCall::kMap && actual == cudaSuccess) {
      cuda_trace->fired = true;
      return kInjectedCudaError;
    }
  }
  return actual;
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaMemcpy(void* destination,
                                                   const void* source,
                                                   size_t bytes,
                                                   cudaMemcpyKind kind) {
  if (cuda_trace != nullptr) {
    cuda_trace->calls.push_back(CudaCall::kCopy);
    if (cuda_trace->fault == CudaCall::kCopy && kind == cuda_trace->copy_kind) {
      cuda_trace->fired = true;
      return kInjectedCudaError;
    }
  }
  return __real_cudaMemcpy(destination, source, bytes, kind);
}
#endif

namespace caspar {
struct GraphSolverTestAccess {
  using Real = colmap::StorageType;
  static const auto& Params(const GraphSolver& solver) {
    return solver.params_;
  }
  static size_t InputScratchBytes(const GraphSolver& solver) {
    const auto offset =
        reinterpret_cast<uintptr_t>(solver.marker__scratch_inout_) -
        reinterpret_cast<uintptr_t>(solver.origin_ptr_);
    return solver.allocation_size_ - offset;
  }
  static void SetPointReadbackSize(GraphSolver& solver,
                                   size_t count,
                                   size_t capacity) {
    solver.Point_num_ = count;
    solver.Point_num_max_ = capacity;
  }
  static Real* InitialNorm(GraphSolver& solver) {
    return solver.solver__r_0_norm2_tot_;
  }
  static Real* UpdatedNorm(GraphSolver& solver) {
    return solver.solver__r_kp1_norm2_tot_;
  }
  static Real* NegAlpha(GraphSolver& solver) {
    return solver.solver__neg_alpha_;
  }
  static Real* PrincipalResidual(GraphSolver& solver) {
    return solver.nodes__PinholePrincipalPoint__r_k_;
  }
  static Real* PrincipalW(GraphSolver& solver) {
    return solver.nodes__PinholePrincipalPoint__w_;
  }
  static Real* PointResidual(GraphSolver& solver) {
    return solver.nodes__Point__r_k_;
  }
  static Real* PointW(GraphSolver& solver) { return solver.nodes__Point__w_; }
  static void Update(GraphSolver& solver) { solver.DoUpdateRFirst(); }
  static Real* PointPreconditionedResidual(GraphSolver& solver) {
    return solver.nodes__Point__z_;
  }
  static Real* PreviousResidualProduct(GraphSolver& solver) {
    return solver.solver__alpha_numerator_;
  }
  static Real* Beta(GraphSolver& solver) { return solver.solver__beta_; }
  static void ComputeBeta(GraphSolver& solver) { solver.DoBeta(); }
  static Real InitialHost(const GraphSolver& solver) {
    return solver.pcg_r_0_norm2_;
  }
  static Real UpdatedHost(const GraphSolver& solver) {
    return solver.pcg_r_kp1_norm2_;
  }
};
}  // namespace caspar

// Due to pose normalization operations, constant variables may not be perfectly
// fixed during bundle adjustment.
constexpr double kConstantPoseVarEps = 1e-9;

#define CheckVariableCamera(camera, orig_camera)       \
  {                                                    \
    const size_t focal_length_idx =                    \
        SimpleRadialCameraModel::focal_length_idxs[0]; \
    const size_t extra_param_idx =                     \
        SimpleRadialCameraModel::extra_params_idxs[0]; \
    EXPECT_NE((camera).params[focal_length_idx],       \
              (orig_camera).params[focal_length_idx]); \
    EXPECT_NE((camera).params[extra_param_idx],        \
              (orig_camera).params[extra_param_idx]);  \
  }

#define CheckConstantCamera(camera, orig_camera)       \
  {                                                    \
    const size_t focal_length_idx =                    \
        SimpleRadialCameraModel::focal_length_idxs[0]; \
    const size_t extra_param_idx =                     \
        SimpleRadialCameraModel::extra_params_idxs[0]; \
    EXPECT_EQ((camera).params[focal_length_idx],       \
              (orig_camera).params[focal_length_idx]); \
    EXPECT_EQ((camera).params[extra_param_idx],        \
              (orig_camera).params[extra_param_idx]);  \
  }

#define CheckVariableCamFromWorld(image, orig_image)                   \
  {                                                                    \
    EXPECT_THAT((image).CamFromWorld(),                                \
                testing::Not(Rigid3dEq((orig_image).CamFromWorld()))); \
  }

#define CheckConstantCamFromWorld(image, orig_image)     \
  {                                                      \
    EXPECT_THAT((image).CamFromWorld(),                  \
                Rigid3dNear((orig_image).CamFromWorld(), \
                            kConstantPoseVarEps,         \
                            kConstantPoseVarEps));       \
  }

#define CheckConstantPoint(point, orig_point) \
  {                                           \
    EXPECT_EQ((point).xyz, (orig_point).xyz); \
  }

namespace colmap {
namespace {

// Match the canonical generator's single-owner allocation contract.
static_assert(!std::is_copy_constructible_v<caspar::GraphSolver>);
static_assert(!std::is_copy_assignable_v<caspar::GraphSolver>);
static_assert(!std::is_move_constructible_v<caspar::GraphSolver>);
static_assert(!std::is_move_assignable_v<caspar::GraphSolver>);

struct CudaDeviceBuffer {
  void* ptr = nullptr;
  ~CudaDeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer() = default;
};

TEST(CasparSharedIndices, MapsPartialBlocksAndDuplicateTargets) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  constexpr size_t kCapacity = 2049;
  CudaDeviceBuffer device_indices;
  CudaDeviceBuffer device_shared_indices;
  ASSERT_EQ(cudaMalloc(&device_indices.ptr, kCapacity * sizeof(unsigned int)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_shared_indices.ptr,
                       kCapacity * sizeof(caspar::SharedIndex)),
            cudaSuccess);
  for (const size_t size : {1, 31, 32, 33, 1023, 1024, 1025, 2049}) {
    for (int pattern = 0; pattern < 4; ++pattern) {
      SCOPED_TRACE(::testing::Message()
                   << "size=" << size << " pattern=" << pattern);
      std::vector<unsigned int> input(size);
      for (size_t i = 0; i < size; ++i) {
        // Sorted, reversed, interleaved duplicates, and one shared target.
        input[i] = static_cast<unsigned int>(pattern == 0   ? i
                                             : pattern == 1 ? size - i
                                             : pattern == 2 ? (size - i) % 7
                                                            : 7);
      }
      ASSERT_EQ(cudaMemcpy(device_indices.ptr,
                           input.data(),
                           size * sizeof(unsigned int),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      caspar::SharedIndices(
          static_cast<const unsigned int*>(device_indices.ptr),
          static_cast<caspar::SharedIndex*>(device_shared_indices.ptr),
          static_cast<unsigned int>(size));
      ASSERT_EQ(cudaGetLastError(), cudaSuccess);
      std::vector<caspar::SharedIndex> output(size);
      ASSERT_EQ(cudaMemcpy(output.data(),
                           device_shared_indices.ptr,
                           size * sizeof(caspar::SharedIndex),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (size_t start = 0; start < size; start += 1024) {
        const size_t count = std::min(size_t{1024}, size - start);
        std::vector<unsigned int> unique(input.begin() + start,
                                         input.begin() + start + count);
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        std::vector<bool> seen(count, false);
        for (size_t i = 0; i < count; ++i) {
          const auto& index = output[start + i];
          EXPECT_EQ(index.unique,
                    i < unique.size()
                        ? unique[i]
                        : std::numeric_limits<unsigned int>::max());
          ASSERT_LT(index.target, unique.size());
          EXPECT_EQ(unique[index.target], input[start + i]);
          ASSERT_LT(index.argsort, count);
          EXPECT_FALSE(seen[index.argsort]);
          seen[index.argsort] = true;
          if (i != 0) {
            EXPECT_LE(input[start + output[start + i - 1].argsort],
                      input[start + index.argsort]);
          }
        }
      }
    }
  }
}

TEST(CasparTransfers, ReservesFactorScratchWithoutPointNodes) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }

  constexpr size_t kNumFactors = 200;
  CasparSolverSizing sizing;
  sizing.num_simple_radial_calibs = 1;
  sizing.num_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point =
      kNumFactors;
  auto solver = CreateSolver(caspar::SolverParams<StorageType>{}, sizing, 0);
  // A fixed pose needs seven stacked scalars per factor, even without any
  // variable Point nodes. Check capacity before attempting the transfer so the
  // old allocation fails safely instead of writing beyond the device buffer.
  std::vector<StorageType> sensor_from_rig(7 * kNumFactors, 0);
  for (size_t i = 0; i < kNumFactors; ++i) {
    sensor_from_rig[7 * i + 3] = 1;
  }
  ASSERT_GE(caspar::GraphSolverTestAccess::InputScratchBytes(solver),
            sensor_from_rig.size() * sizeof(StorageType));
  solver
      .SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
          sensor_from_rig.data(), 0, kNumFactors);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(CasparParameters, RejectsInvalidUpdatesWithoutChangingParameters) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }

  using Access = caspar::GraphSolverTestAccess;
  const caspar::SolverParams<StorageType> valid;
  auto solver = CreateSolver(valid, {}, 0);
  const auto original = Access::Params(solver);
  const auto expect_error = [](const auto& action, const char* message) {
    try {
      action();
      ADD_FAILURE() << "Expected parameter error: " << message;
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), message);
    }
  };
  const auto expect_invalid = [&](auto invalid, const char* message) {
    invalid.solver_iter_max = 13;
    expect_error([&] { solver.set_params(invalid); }, message);
    const auto& stored = Access::Params(solver);
    EXPECT_EQ(stored.pcg_iter_max, original.pcg_iter_max);
    EXPECT_EQ(stored.pcg_rel_error_exit, original.pcg_rel_error_exit);
    EXPECT_EQ(stored.diag_init, original.diag_init);
    EXPECT_EQ(stored.solver_iter_max, original.solver_iter_max);
    expect_error([&] { CreateSolver(invalid, {}, device_count); }, message);
  };
  auto invalid = valid;
  for (const int count : {0, -1}) {
    invalid = valid;
    invalid.pcg_iter_max = count;
    expect_invalid(invalid, "params.pcg_iter_max must be at least 1");
  }
  for (const double threshold : {0.0, -1.0}) {
    invalid = valid;
    invalid.pcg_rel_error_exit = threshold;
    expect_invalid(invalid, "params.pcg_rel_error_exit must be positive");
  }
  invalid = valid;
  invalid.diag_init = -1.0;
  expect_invalid(invalid, "params.diag_init must be positive");

  auto boundary = valid;
  boundary.diag_init = 0.0;
  boundary.pcg_iter_max = 1;
  EXPECT_NO_THROW(solver.set_params(boundary));
  EXPECT_EQ(Access::Params(solver).diag_init, 0.0);
  EXPECT_EQ(Access::Params(solver).pcg_iter_max, 1);
}

TEST(CasparOptions, NonfiniteInitialObjectiveReturnsFailureWithoutWriteback) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  SetPRNGSeed(0);
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 3;
  dataset.num_points3D = 8;
  Reconstruction reconstruction;
  SynthesizeDataset(dataset, &reconstruction);
  ASSERT_FALSE(reconstruction.Points3D().empty());
  const auto& track =
      reconstruction.Points3D().begin()->second.track.Elements();
  ASSERT_FALSE(track.empty());
  const auto observation = track.front();
  reconstruction.Image(observation.image_id)
      .Point2D(observation.point2D_idx)
      .xy.x() = std::numeric_limits<double>::quiet_NaN();
  const Reconstruction before = reconstruction;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  BundleAdjustmentOptions options;
  options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
  options.caspar->gpu_index = "0";
  options.caspar->solver_iter_max = 2;
  options.caspar->pcg_iter_max = 1;
  // One rejection would exceed the damping limit in the old solver.
  options.caspar->diag_init = 1.0;
  options.caspar->diag_exit_value = 1.5;
  options.caspar->diag_scaling_up = 2.0;
  options.caspar->collect_iteration_data = true;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  auto summary =
      std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(adjuster->Solve());
  ASSERT_NE(summary, nullptr);
  EXPECT_FALSE(std::isfinite(summary->initial_score));
  EXPECT_FALSE(summary->IsSolutionUsable());
  EXPECT_EQ(summary->termination_type, BundleAdjustmentTerminationType::FAILURE);
  EXPECT_EQ(summary->iteration_count, 0);
  EXPECT_TRUE(summary->iterations.empty());
  EXPECT_GT(summary->num_residuals, 0);

  for (const auto& [id, point] : before.Points3D()) {
    EXPECT_EQ(reconstruction.Point3D(id).xyz, point.xyz);
  }
  for (const auto& [id, camera] : before.Cameras()) {
    EXPECT_EQ(reconstruction.Camera(id).params, camera.params);
  }
  for (const auto& [id, frame] : before.Frames()) {
    EXPECT_EQ(reconstruction.Frame(id).RigFromWorld().params,
              frame.RigFromWorld().params);
  }
}

TEST(CasparOptions, ReportsCompletedAndTerminalAttempts) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  SetPRNGSeed(0);
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 3;
  dataset.num_points3D = 8;
  Reconstruction initial;
  SynthesizeDataset(dataset, &initial);
  SyntheticNoiseOptions noise;
  noise.point3D_stddev = 0.02;
  SynthesizeNoise(noise, &initial);
  BundleAdjustmentConfig config;
  for (const image_t image_id : initial.RegImageIds()) {
    config.AddImage(image_id);
  }

  struct TestCase {
    const char* name;
    int iteration_limit;
    double score_limit;
    double damping_limit;
    int completed;
  };
  constexpr TestCase cases[] = {
      {"exhaustion", 3, 0.0, 1000.0, 3},
      {"score stop", 3, 1e6, 1000.0, 1},
      {"damping stop", 3, 0.0, 1.5, 1},
      {"zero attempts", 0, 0.0, 1000.0, 0},
  };
  for (const TestCase& test : cases) {
    for (const bool collect : {true, false}) {
      SCOPED_TRACE(::testing::Message() << test.name << " collect=" << collect);
      BundleAdjustmentOptions options;
      options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
      options.caspar->gpu_index = "0";
      options.caspar->solver_iter_max = test.iteration_limit;
      options.caspar->pcg_iter_max = 1;
      options.caspar->solver_rel_decrease_min = 0;
      options.caspar->score_exit_value = test.score_limit;
      options.caspar->diag_init = 1.0;
      options.caspar->diag_scaling_up = 2.0;
      options.caspar->diag_exit_value = test.damping_limit;
      options.caspar->collect_iteration_data = collect;
      Reconstruction reconstruction = initial;
      const auto adjuster =
          CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
      const auto summary =
          std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
              adjuster->Solve());
      ASSERT_NE(summary, nullptr);
      ASSERT_GT(summary->initial_score, 0.0);
      ASSERT_LT(summary->initial_score, 1e6);
      const bool exhausted = test.completed == test.iteration_limit;
      EXPECT_EQ(summary->termination_type,
                exhausted ? BundleAdjustmentTerminationType::NO_CONVERGENCE
                          : BundleAdjustmentTerminationType::CONVERGENCE);
      EXPECT_EQ(summary->iteration_count, test.completed);
      ASSERT_EQ(summary->iterations.size(),
                collect ? static_cast<size_t>(test.completed) : 0u);
      for (size_t i = 0; i < summary->iterations.size(); ++i) {
        EXPECT_EQ(summary->iterations[i].solver_iter, static_cast<int>(i));
        EXPECT_FALSE(summary->iterations[i].step_accepted);
        const auto& row = summary->iterations[i];
        const double previous_time =
            i == 0 ? 0.0 : summary->iterations[i - 1].dt_tot;
        EXPECT_NEAR(row.dt_inc,
                    row.dt_tot - previous_time,
                    8 * std::numeric_limits<double>::epsilon() *
                        std::max(1.0, row.dt_tot));
      }
      if (!summary->iterations.empty()) {
        EXPECT_EQ(summary->iterations.front().diag, 1.0);
      }
    }
  }
}

TEST(CasparPcg, CarriesResidualProductBetweenBetaUpdates) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  using Access = caspar::GraphSolverTestAccess;
  using Real = StorageType;
  CasparSolverSizing sizing;
  sizing.num_points = 1;
  auto solver = CreateSolver(caspar::SolverParams<Real>{}, sizing, 0);
  solver.SetPointNum(1);

  const Real initial_product = 12;
  const Real residual[] = {2, 1, -1, 0};
  // Float kernels read a padded float4. Double kernels read three values.
  constexpr size_t point_bytes =
      sizeof(Real) * (sizeof(Real) == sizeof(float) ? 4 : 3);
  ASSERT_EQ(cudaMemcpy(Access::PreviousResidualProduct(solver),
                       &initial_product,
                       sizeof(initial_product),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(Access::PointResidual(solver),
                       residual,
                       point_bytes,
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  const Real preconditioned[][4] = {{2, 3, 1, 0}, {3, 4, 1, 0}};
  const Real expected_beta[] = {0.5, 1.5};  // 6 / 12, then 9 / 6.
  for (int update = 0; update < 2; ++update) {
    SCOPED_TRACE(update);
    ASSERT_EQ(cudaMemcpy(Access::PointPreconditionedResidual(solver),
                         preconditioned[update],
                         point_bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    Access::ComputeBeta(solver);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    Real beta = 0;
    ASSERT_EQ(cudaMemcpy(
                  &beta, Access::Beta(solver), sizeof(beta), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(beta, expected_beta[update]);
  }
}

TEST(CasparFirstNorm, ResetsBothAccumulatorsBeforeRealKernelUpdates) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }

  using Access = caspar::GraphSolverTestAccess;
  using Real = StorageType;
  CasparSolverSizing sizing;
  sizing.num_pinhole_calibs = 1;
  sizing.num_points = 1;
  auto solver = CreateSolver(caspar::SolverParams<StorageType>{}, sizing, 0);
  // The calibration pool also allocates these two uninitialized node types.
  solver.SetPinholeCalibNum(0);
  solver.SetPinholeFocalNum(0);
  const Real neg_alpha = -0.5;
  const Real principal_r[] = {1, 2}, principal_w[] = {2, 0};
  const Real point_r[] = {3, 0, 0, 0}, point_w[] = {2, 0, 0, 0};
  // Float kernels use a padded float4 load; double kernels access three values.
  constexpr size_t point_bytes =
      sizeof(Real) * (sizeof(Real) == sizeof(float) ? 4 : 3);
  ASSERT_EQ(cudaMemcpy(Access::NegAlpha(solver),
                       &neg_alpha,
                       sizeof(neg_alpha),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  int call = 0;
  const auto run = [&](size_t principal_count,
                       size_t point_count,
                       Real expected_initial,
                       Real expected_updated) {
    SCOPED_TRACE(::testing::Message() << "principal=" << principal_count
                                      << " point=" << point_count);
    solver.SetPinholePrincipalPointNum(principal_count);
    solver.SetPointNum(point_count);
    const Real initial_seed = Real(39 + 2 * call);
    const Real updated_seed = Real(62 + 3 * call++);
    ASSERT_EQ(cudaMemcpy(Access::InitialNorm(solver),
                         &initial_seed,
                         sizeof(Real),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(Access::UpdatedNorm(solver),
                         &updated_seed,
                         sizeof(Real),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(Access::PrincipalResidual(solver),
                         principal_r,
                         sizeof(principal_r),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(Access::PrincipalW(solver),
                         principal_w,
                         sizeof(principal_w),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(Access::PointResidual(solver),
                         point_r,
                         point_bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(Access::PointW(solver),
                         point_w,
                         point_bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    Access::Update(solver);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_NEAR(Access::InitialHost(solver), expected_initial, 1e-6);
    EXPECT_NEAR(Access::UpdatedHost(solver), expected_updated, 1e-6);
    if (principal_count) {
      Real values[2];
      ASSERT_EQ(cudaMemcpy(values,
                           Access::PrincipalResidual(solver),
                           sizeof(values),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      EXPECT_NEAR(values[0], 0, 1e-6);
      EXPECT_NEAR(values[1], 2, 1e-6);
    }
    if (point_count) {
      Real values[4];
      ASSERT_EQ(cudaMemcpy(values,
                           Access::PointResidual(solver),
                           point_bytes,
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      EXPECT_NEAR(values[0], 2, 1e-6);
      EXPECT_NEAR(values[1], 0, 1e-6);
      EXPECT_NEAR(values[2], 0, 1e-6);
    }
  };
  run(1, 1, 14, 8);
  run(1, 1, 14, 8);
  run(1, 0, 5, 4);
  run(0, 1, 9, 4);
  run(0, 0, 0, 0);
}

TEST(CasparScoredPcg, IndependentCriteriaAcceptTheSameSingleStep) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  Reconstruction initial;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = 3;
  dataset_options.num_points3D = 64;
  SynthesizeDataset(dataset_options, &initial);
  SyntheticNoiseOptions noise_options;
  noise_options.point3D_stddev = 0.02;
  SynthesizeNoise(noise_options, &initial);

  BundleAdjustmentConfig config;
  for (const image_t image_id : initial.RegImageIds()) {
    const auto& image = initial.Image(image_id);
    config.AddImage(image_id);
    config.SetConstantRigFromWorldPose(image.FrameId());
    config.SetConstantCamIntrinsics(image.CameraId());
  }
  double reference_score = 0;
  double reference_quality = 0;
#if defined(COLMAP_TEST_CASPAR_READBACK_WRAPPING)
  std::vector<CudaCall> reference_calls;
#endif
  for (const double decrease : {-1.0, -0.5, -2.0, 1.0}) {
    for (const double score_exit : {-1.0, -0.5, -2.0, 0.5}) {
      SCOPED_TRACE(::testing::Message()
                   << "decrease=" << decrease << " score_exit=" << score_exit);
      Reconstruction reconstruction = initial;
      BundleAdjustmentOptions options;
      options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
      options.caspar->solver_iter_max = 1;
      options.caspar->pcg_iter_max = 1;
      options.caspar->pcg_rel_decrease_min = decrease;
      options.caspar->pcg_rel_score_exit = score_exit;
      options.caspar->collect_iteration_data = true;
      auto adjuster =
          CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
#if defined(COLMAP_TEST_CASPAR_READBACK_WRAPPING)
      CudaTrace trace;
      ScopedCudaFault scope(trace, CudaCall::kNone);
#endif
      const auto summary =
          std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
              adjuster->Solve());
#if defined(COLMAP_TEST_CASPAR_READBACK_WRAPPING)
      if (decrease == -1.0 && score_exit == -1.0) {
        reference_calls = trace.calls;
      } else if (decrease < 0.0 && score_exit < 0.0) {
        EXPECT_EQ(trace.calls, reference_calls);
      }
#endif
      ASSERT_NE(summary, nullptr);
      ASSERT_EQ(summary->iterations.size(), 1u);
      const auto& iteration = summary->iterations.front();
      ASSERT_GT(summary->initial_score, 0);
      EXPECT_TRUE(iteration.step_accepted);
      EXPECT_EQ(iteration.pcg_iter, 0);
      EXPECT_LT(iteration.score_best, summary->initial_score);
      EXPECT_TRUE(std::isfinite(iteration.step_quality));
      if (decrease == -1.0 && score_exit == -1.0) {
        reference_score = iteration.score_best;
        reference_quality = iteration.step_quality;
      } else {
        EXPECT_NEAR(iteration.score_best,
                    reference_score,
                    1e-4 * std::max(1.0, std::abs(reference_score)));
        EXPECT_NEAR(iteration.step_quality,
                    reference_quality,
                    1e-4 * std::max(1.0, std::abs(reference_quality)));
      }
    }
  }
}

TEST(CasparScoredPcg, RetainedPredictionMatchesTruncatedSolve) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  SetPRNGSeed(0);
  Reconstruction initial;
  SyntheticDatasetOptions data;
  data.num_rigs = 1;
  data.num_cameras_per_rig = 1;
  data.num_frames_per_rig = 3;
  data.num_points3D = 64;
  SynthesizeDataset(data, &initial);
  SyntheticNoiseOptions noise;
  noise.point3D_stddev = 0.05;
  noise.rig_from_world_rotation_stddev = 0.5;
  noise.rig_from_world_translation_stddev = 0.02;
  SynthesizeNoise(noise, &initial);
  BundleAdjustmentConfig config;
  for (const image_t id : initial.RegImageIds()) {
    config.AddImage(id);
  }
  // Require a retained step followed by rejection on a fixed synthetic scene.
  for (double diag : {1.0, 1e-3, 1e-6}) {
    for (double decrease : {0.5, 0.8, 0.95}) {
      SCOPED_TRACE(::testing::Message()
                   << "diag=" << diag << " decrease=" << decrease);
      constexpr double kDiagMin = 1e-12;
      const auto solve = [&](int pcg_limit,
                             int solver_limit,
                             double criterion,
                             Reconstruction& reconstruction) {
        BundleAdjustmentOptions options;
        options.refine_principal_point = true;
        options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
        options.caspar->solver_iter_max = solver_limit;
        options.caspar->pcg_iter_max = pcg_limit;
        options.caspar->diag_init = diag;
        options.caspar->diag_min = kDiagMin;
        options.caspar->diag_scaling_down = 0;
        options.caspar->pcg_rel_decrease_min = criterion;
        options.caspar->pcg_rel_score_exit = -1;
        options.caspar->pcg_rel_error_exit = 1e-12;
        options.caspar->collect_iteration_data = true;
        options.caspar->gpu_index = "0";
        auto adjuster =
            CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
        return std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
            adjuster->Solve());
      };
      Reconstruction first_reconstruction = initial;
      Reconstruction second_reconstruction = initial;
      const auto first = solve(1, 2, decrease, first_reconstruction);
      const auto second = solve(2, 2, decrease, second_reconstruction);
      ASSERT_NE(first, nullptr);
      ASSERT_NE(second, nullptr);
      if (first->iterations.size() < 2 || second->iterations.size() < 2) {
        continue;
      }
      const auto& a = first->iterations[0];
      const auto& b = second->iterations[0];
      if (!a.step_accepted || !b.step_accepted || b.pcg_iter != 1 ||
          std::abs(a.score_best - b.score_best) >
              1e-6 * std::max(1.0, std::abs(a.score_best))) {
        continue;
      }
      ASSERT_TRUE(std::isfinite(a.step_quality));
      ASSERT_TRUE(std::isfinite(b.step_quality));
      EXPECT_NEAR(a.step_quality,
                  b.step_quality,
                  1e-5 * std::max(1.0, std::abs(a.step_quality)));
      for (const auto& summary : {first, second}) {
        const auto& step = summary->iterations[0];
        ASSERT_GE(step.step_quality, 0.5);
        ASSERT_LE(step.step_quality, 1.0);
        const double quality_term = 2.0 * step.step_quality - 1.0;
        const double expected = std::max(
            double(StorageType(kDiagMin)),
            step.diag *
                std::max(0.0, 1.0 - quality_term * quality_term * quality_term));
        // In this interval 2*q and 2*q-1 are exact; the remaining operations
        // and clamps are bounded by four storage-type rounding errors.
        const double rounding_bound =
            4.0 * std::numeric_limits<StorageType>::epsilon() * step.diag;
        EXPECT_NEAR(summary->iterations[1].diag, expected, rounding_bound);
      }

      // Compare accepted state after one outer step, before later damping can
      // produce legitimately different trajectories.
      Reconstruction short_state = initial;
      Reconstruction strict_state = initial;
      Reconstruction relaxed_state = initial;
      const auto short_run = solve(1, 1, decrease, short_state);
      const auto strict_run = solve(2, 1, decrease, strict_state);
      const auto relaxed_run = solve(2, 1, 1.0, relaxed_state);
      ASSERT_NE(short_run, nullptr);
      ASSERT_NE(strict_run, nullptr);
      ASSERT_NE(relaxed_run, nullptr);
      ASSERT_EQ(short_run->iterations.size(), 1u);
      ASSERT_EQ(strict_run->iterations.size(), 1u);
      ASSERT_EQ(relaxed_run->iterations.size(), 1u);
      ASSERT_TRUE(short_run->iterations[0].step_accepted);
      ASSERT_TRUE(strict_run->iterations[0].step_accepted);
      ASSERT_TRUE(relaxed_run->iterations[0].step_accepted);
      ASSERT_EQ(strict_run->iterations[0].pcg_iter, 1);
      ASSERT_EQ(relaxed_run->iterations[0].pcg_iter, 1);
      for (const auto& pair : {std::make_pair(short_run, first),
                               std::make_pair(strict_run, second)}) {
        EXPECT_NEAR(pair.first->iterations[0].score_best,
                    pair.second->iterations[0].score_best,
                    1e-6 *
                        std::max(1.0, std::abs(pair.second->iterations[0].score_best)));
        EXPECT_NEAR(pair.first->iterations[0].step_quality,
                    pair.second->iterations[0].step_quality,
                    1e-5 * std::max(
                               1.0, std::abs(pair.second->iterations[0].step_quality)));
      }
      // The relaxed second candidate improves but fails the strict criterion.
      EXPECT_LT(relaxed_run->iterations[0].score_best,
                strict_run->iterations[0].score_best);
      EXPECT_GT(relaxed_run->iterations[0].score_best,
                strict_run->iterations[0].score_best * decrease);
      double max_state_ratio = 0;
      const auto near_state = [&](double lhs, double rhs) {
        ASSERT_TRUE(std::isfinite(lhs));
        ASSERT_TRUE(std::isfinite(rhs));
        // A prospective fixture budget, not a solver-wide error bound.
        const double budget =
            64.0 * std::numeric_limits<StorageType>::epsilon() *
            std::max({1.0, std::abs(lhs), std::abs(rhs)});
        max_state_ratio =
            std::max(max_state_ratio, std::abs(lhs - rhs) / budget);
        EXPECT_NEAR(lhs, rhs, budget);
      };
      for (const auto& [id, camera] : short_state.Cameras()) {
        const auto& other = strict_state.Camera(id);
        ASSERT_EQ(camera.params.size(), other.params.size());
        for (size_t i = 0; i < camera.params.size(); ++i) {
          near_state(camera.params[i], other.params[i]);
        }
      }
      for (const auto& [id, image] : short_state.Images()) {
        const auto& a_pose = image.CamFromWorld();
        const auto& b_pose = strict_state.Image(id).CamFromWorld();
        const double sign = a_pose.rotation().coeffs().dot(
                                b_pose.rotation().coeffs()) < 0
                                ? -1.0
                                : 1.0;
        for (int i = 0; i < 4; ++i) {
          near_state(a_pose.rotation().coeffs()[i],
                     sign * b_pose.rotation().coeffs()[i]);
        }
        for (int i = 0; i < 3; ++i) {
          near_state(a_pose.translation()[i], b_pose.translation()[i]);
        }
      }
      bool changed_from_initial = false;
      bool relaxed_outside_budget = false;
      for (const auto& [id, point] : short_state.Points3D()) {
        for (int i = 0; i < 3; ++i) {
          const double strict = strict_state.Point3D(id).xyz[i];
          const double relaxed = relaxed_state.Point3D(id).xyz[i];
          ASSERT_TRUE(std::isfinite(strict));
          ASSERT_TRUE(std::isfinite(relaxed));
          const double budget =
              64.0 * std::numeric_limits<StorageType>::epsilon() *
              std::max({1.0, std::abs(strict), std::abs(relaxed)});
          changed_from_initial |= point.xyz[i] != initial.Point3D(id).xyz[i];
          relaxed_outside_budget |= std::abs(strict - relaxed) > budget;
          near_state(point.xyz[i], strict);
        }
      }
      EXPECT_TRUE(changed_from_initial);
      EXPECT_TRUE(relaxed_outside_budget);
      RecordProperty("max_state_error_budget_ratio", max_state_ratio);
      return;
    }
  }
  FAIL() << "No retained-then-rejected witness in the fixed fixture grid";
}

TEST(CasparRejectedStep, RetriesMatchFreshLinearizations) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  SetPRNGSeed(0);
  Reconstruction initial;
  SyntheticDatasetOptions data;
  data.num_rigs = 1;
  data.num_cameras_per_rig = 1;
  data.num_frames_per_rig = 3;
  data.num_points3D = 67;
  SynthesizeDataset(data, &initial);
  SyntheticNoiseOptions noise;
  noise.point3D_stddev = 0.02;
  noise.rig_from_world_translation_stddev = 0.01;
  SynthesizeNoise(noise, &initial);
  BundleAdjustmentConfig config;
  for (const image_t id : initial.RegImageIds()) {
    config.AddImage(id);
    config.SetConstantCamIntrinsics(initial.Image(id).CameraId());
  }
  config.SetConstantRigFromWorldPose(
      initial.Image(initial.RegImageIds().front()).FrameId());
  config.AddConstantPoint(initial.Points3D().begin()->first);

  constexpr double kDiagMin = 0.125;

  // Reject every proposal while keeping a coupled pose/point linear system.
  // Each retry must match a fresh solve at the same state and damping.
  for (const int pcg_limit : {1, 3}) {
    for (const bool score_pcg : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "pcg_limit=" << pcg_limit << " score_pcg=" << score_pcg);
      const auto solve = [&](int iterations,
                             double diag,
                             Reconstruction& reconstruction) {
        BundleAdjustmentOptions options;
        options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
        options.caspar->solver_iter_max = iterations;
        options.caspar->pcg_iter_max = pcg_limit;
        options.caspar->diag_init = diag;
        options.caspar->diag_min = kDiagMin;
        options.caspar->solver_rel_decrease_min = 0;
        options.caspar->pcg_rel_decrease_min = score_pcg ? 1 : -1;
        options.caspar->pcg_rel_score_exit = -1;
        options.caspar->pcg_rel_error_exit = 1e-20;
        options.caspar->collect_iteration_data = true;
        options.caspar->gpu_index = "0";
        auto adjuster =
            CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
        return std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
            adjuster->Solve());
      };
      Reconstruction retried = initial;
      const auto retries = solve(3, 1, retried);
      ASSERT_NE(retries, nullptr);
      ASSERT_GT(retries->initial_score, 0);
      ASSERT_EQ(retries->iterations.size(), 3u);
      EXPECT_EQ(retries->iterations[0].diag, 1.0);
      EXPECT_EQ(retries->iterations[1].diag, 2.0);
      EXPECT_EQ(retries->iterations[2].diag, 8.0);
      for (const auto& retry : retries->iterations) {
        SCOPED_TRACE(retry.solver_iter);
        Reconstruction fresh = initial;
        const auto reference = solve(1, retry.diag, fresh);
        ASSERT_NE(reference, nullptr);
        ASSERT_EQ(reference->iterations.size(), 1u);
        const auto& expected = reference->iterations.front();
        EXPECT_FALSE(retry.step_accepted);
        EXPECT_FALSE(expected.step_accepted);
        EXPECT_EQ(retry.score_best, retries->initial_score);
        EXPECT_EQ(retry.step_quality, 0);
        EXPECT_EQ(retry.pcg_iter, expected.pcg_iter);
        if (!score_pcg) {
          EXPECT_EQ(retry.pcg_iter, pcg_limit - 1);
        }
        ASSERT_TRUE(std::isfinite(retry.score_current));
        ASSERT_TRUE(std::isfinite(expected.score_current));
        // GPU reductions can sum identical inputs in a different order.
        const double tolerance =
            std::is_same_v<StorageType, double> ? 1e-8 : 5e-4;
        EXPECT_NEAR(retry.score_current,
                    expected.score_current,
                    tolerance * std::max(1.0, std::abs(expected.score_current)));
        for (const auto& [point_id, point] : retried.Points3D()) {
          EXPECT_EQ(point.xyz, fresh.Point3D(point_id).xyz);
        }
      }

      for (const double initial_diag : {0.0, kDiagMin / 2}) {
        SCOPED_TRACE(initial_diag);
        Reconstruction from_floor = initial;
        const auto summary = solve(3, initial_diag, from_floor);
        ASSERT_NE(summary, nullptr);
        ASSERT_EQ(summary->iterations.size(), 3u);
        EXPECT_EQ(summary->iterations[0].diag, kDiagMin);
        EXPECT_EQ(summary->iterations[1].diag, 2 * kDiagMin);
        EXPECT_EQ(summary->iterations[2].diag, 8 * kDiagMin);
        for (const auto& step : summary->iterations) {
          EXPECT_FALSE(step.step_accepted);
          EXPECT_TRUE(std::isfinite(step.score_current));
        }
      }
    }
  }
}

TEST(CasparOptions, RigThrowsErrorOnVariableSensorFromRig) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  BundleAdjustmentOptions options;
  BundleAdjustmentConfig config;
  options.refine_sensor_from_rig = true;  // Not supported yet
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  EXPECT_THROW(
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction),
      std::invalid_argument);
}

class DefaultBundleAdjuster : public ::testing::Test {
 protected:
  void SetUp() override {
    if (GetNumCudaDevices() == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
  }
};

TEST_F(DefaultBundleAdjuster, MultiCameraRigLargeConstantSensorFromRig) {
  // Real-world multi-camera rigs (stereo, surround-view) have large
  // sensor_from_rig offsets — typically 20–90 degrees and 0.1–1 m baseline.
  // This test uses a 30-degree Z-rotation and 0.3 m translation to exercise
  // the non-identity sensor_from_rig path with realistic values.
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30.0;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.3;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  BundleAdjustmentOptions options;
  options.refine_sensor_from_rig = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST_F(DefaultBundleAdjuster, MergedCalibConvergence) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.2,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST_F(DefaultBundleAdjuster, MergedCalibFixedPose) {
  // Verifies that all four intrinsic parameters change.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t principal_point_idx_x =
      SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t principal_point_idx_y =
      SimpleRadialCameraModel::principal_point_idxs[1];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  for (const camera_t cam_id : {camera_t{1}, camera_t{2}}) {
    const auto& cam = reconstruction.Camera(cam_id);
    const auto& orig_cam = orig_reconstruction.Camera(cam_id);
    EXPECT_NE(cam.params[focal_length_idx], orig_cam.params[focal_length_idx]);
    EXPECT_NE(cam.params[extra_param_idx], orig_cam.params[extra_param_idx]);
    EXPECT_NE(cam.params[principal_point_idx_x],
              orig_cam.params[principal_point_idx_x]);
    EXPECT_NE(cam.params[principal_point_idx_y],
              orig_cam.params[principal_point_idx_y]);
  }

  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
}

TEST_F(DefaultBundleAdjuster, MergedCalibFixedPoint) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  // Fix all 3D points; only pose and calib are free.
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddConstantPoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    CheckConstantPoint(point3D, orig_reconstruction.Point3D(point3D_id));
  }
}

TEST_F(DefaultBundleAdjuster, ExternalImagePoseIsInvariant) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 3;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddVariablePoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
}

TEST_F(DefaultBundleAdjuster, ExternalCameraIntrinsicsOrderingIsConsistent) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 3;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddVariablePoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamera(reconstruction.Camera(3), orig_reconstruction.Camera(3));
}

TEST_F(DefaultBundleAdjuster, ExternalImageViaConstantPointsIsInvariant) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 3;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddConstantPoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
}

TEST_F(DefaultBundleAdjuster, MultipleExternalImagesAreInvariant) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 4;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  opts.num_points2D_without_point3D = 0;
  SynthesizeDataset(opts, &reconstruction);

  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddVariablePoint(point3D_id);
  }

  BundleAdjustmentOptions options;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  CheckConstantCamFromWorld(reconstruction.Image(3),
                            orig_reconstruction.Image(3));
  CheckConstantCamFromWorld(reconstruction.Image(4),
                            orig_reconstruction.Image(4));
}

TEST_F(DefaultBundleAdjuster, MergedCalibMatchesCeres) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.3;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;

  Reconstruction reconstruction_ceres = reconstruction;
  Reconstruction reconstruction_caspar = reconstruction;

  std::unique_ptr<BundleAdjuster> ceres_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction_ceres);
  ASSERT_NE(ceres_adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  std::unique_ptr<BundleAdjuster> caspar_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction_caspar);
  ASSERT_NE(caspar_adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Ceres uses exact dense Schur elimination with 7-DOF gauge fixing (scale
  // pinned), whereas Caspar uses iterative PCG with 6-DOF gauge fixing (scale
  // unpinned). Tolerances accommodate this gauge and solver difference while
  // ensuring convergence to the same calibration solution.
#ifdef CASPAR_USE_DOUBLE
  constexpr double kFocalTol = 1.0;
  constexpr double kPPTol = 2.0;
  constexpr double kExtraTol = 5e-4;
#else
  constexpr double kFocalTol = 20.0;
  constexpr double kPPTol = 10.0;
  constexpr double kExtraTol = 1.5e-2;
#endif

  const size_t f_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t cx_idx = SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t cy_idx = SimpleRadialCameraModel::principal_point_idxs[1];
  const size_t k_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  for (const auto& [cam_id, _] : reconstruction.Cameras()) {
    const auto& cam_ceres = reconstruction_ceres.Camera(cam_id);
    const auto& cam_caspar = reconstruction_caspar.Camera(cam_id);
    EXPECT_NEAR(cam_caspar.params[f_idx], cam_ceres.params[f_idx], kFocalTol)
        << "focal length mismatch for camera " << cam_id;
    EXPECT_NEAR(cam_caspar.params[cx_idx], cam_ceres.params[cx_idx], kPPTol)
        << "cx mismatch for camera " << cam_id;
    EXPECT_NEAR(cam_caspar.params[cy_idx], cam_ceres.params[cy_idx], kPPTol)
        << "cy mismatch for camera " << cam_id;
    EXPECT_NEAR(cam_caspar.params[k_idx], cam_ceres.params[k_idx], kExtraTol)
        << "radial distortion mismatch for camera " << cam_id;
  }
}

bool PoseExactlyUnchanged(const Image& a, const Image& b) {
  return a.CamFromWorld().rotation().coeffs() ==
             b.CamFromWorld().rotation().coeffs() &&
         a.CamFromWorld().translation() == b.CamFromWorld().translation();
}

TEST_F(DefaultBundleAdjuster, GaugeFixingWithOneFrameFromWorld) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Exactly one of the two frames must be pinned by gauge fixing.
  const int n_fixed =
      static_cast<int>(PoseExactlyUnchanged(reconstruction.Image(1),
                                            orig_reconstruction.Image(1))) +
      static_cast<int>(PoseExactlyUnchanged(reconstruction.Image(2),
                                            orig_reconstruction.Image(2)));
  EXPECT_EQ(n_fixed, 1);
}

TEST_F(DefaultBundleAdjuster,
       GaugeFixingWithOneFrameFromWorld_SkipsWhenAlreadyFixed) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantRigFromWorldPose(1);  // frame 1 explicitly constant
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Frame 1 is explicitly constant — must be unchanged.
  CheckConstantCamFromWorld(reconstruction.Image(1),
                            orig_reconstruction.Image(1));
  // Frame 2 is not gauge-fixed (gauge fixer saw frame 1 already fixed) — must
  // change.
  CheckVariableCamFromWorld(reconstruction.Image(2),
                            orig_reconstruction.Image(2));
}

TEST_F(DefaultBundleAdjuster,
       GaugeFixingWithThreePoints_PinsExactlyThreePoints) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  int n_unchanged = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.xyz == orig_reconstruction.Point3D(point3D_id).xyz) {
      ++n_unchanged;
    }
  }
  EXPECT_EQ(n_unchanged, 3);

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
}

TEST_F(DefaultBundleAdjuster,
       GaugeFixingWithThreePoints_CountsExistingConstantPoints) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 2;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.AddConstantPoint(1);  // 1 existing constant; gauge fixer adds 2 more
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

  BundleAdjustmentOptions options;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // Total unchanged = 1 config-constant + 2 gauge-fixed = 3.
  int n_unchanged = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.xyz == orig_reconstruction.Point3D(point3D_id).xyz) {
      ++n_unchanged;
    }
  }
  EXPECT_EQ(n_unchanged, 3);

  CheckConstantPoint(reconstruction.Point3D(1), orig_reconstruction.Point3D(1));
}

TEST_F(DefaultBundleAdjuster,
       MultiCameraRigResidualCountConstantSensorFromRig) {
  // All sensor observations (ref and non-ref) must contribute residuals.
  // The old code skipped non-ref sensor observations when the pose was
  // variable, which would halve the residual count for a 2-camera rig.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_sensor_from_rig = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  // 100 points × 4 images (2 sensors × 2 frames) × 2 residuals per obs
  EXPECT_EQ(summary->num_residuals, 800);
}

TEST_F(DefaultBundleAdjuster, MultiCameraRigConstantRigPoseHoldsAllSensors) {
  // When a frame's rig_from_world is held constant, and Caspar always holds
  // sensor_from_rig constant, ALL sensors in that frame (ref and non-ref)
  // must have invariant cam_from_world. Sensors in the variable frame must
  // change. This differs from the Ceres behaviour where non-ref sensors can
  // still move via a variable sensor_from_rig.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.3;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  const frame_t constant_frame_id = 1;

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.SetConstantRigFromWorldPose(constant_frame_id);

  BundleAdjustmentOptions options;
  options.refine_sensor_from_rig = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  ASSERT_NE(bundle_adjuster->Solve()->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    if (image.FrameId() == constant_frame_id) {
      CheckConstantCamFromWorld(image, orig_reconstruction.Image(image_id));
    } else {
      CheckVariableCamFromWorld(image, orig_reconstruction.Image(image_id));
    }
  }
}

TEST_F(DefaultBundleAdjuster,
       MultiCameraRigLargeConvergenceConstantSensorFromRig) {
  // 2 rigs × 3 cameras × 5 frames = 30 images. Mirrors the Ceres
  // NominalMultiCameraRig test to verify Caspar converges to GT at the same
  // scale as the single-camera nominal test.
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.refine_sensor_from_rig = false;
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

#if defined(COLMAP_TEST_CASPAR_READBACK_WRAPPING)
TEST(CasparConstruction, StopsOnCudaFailure) {
  // Only constructor/destructor host control flow runs here. No device calls
  // or kernels are simulated beyond the three explicitly wrapped APIs.
  using C = CudaCall;
  for (const auto fault : {C::kNone, C::kSelect, C::kAllocate}) {
    SCOPED_TRACE(static_cast<int>(fault));
    CudaTrace trace;
    trace.host_allocation = true;
    bool constructed = false;
    {
      ScopedCudaFault scope(trace, fault);
      try {
        auto solver = CreateSolver(caspar::SolverParams<StorageType>{}, {}, 0);
        constructed = true;
        EXPECT_EQ(trace.allocation.size(), solver.get_allocation_size());
      } catch (const std::runtime_error& error) {
        const char* operation =
            fault == C::kSelect ? "cudaSetDevice" : "cudaMalloc";
        const cudaError_t status = fault == C::kSelect
                                       ? kInjectedCudaError
                                       : cudaErrorMemoryAllocation;
        EXPECT_EQ(
            std::string(error.what()),
            std::string(operation) + " failed: " + cudaGetErrorString(status));
      }
    }
    EXPECT_EQ(constructed, fault == C::kNone);
    EXPECT_EQ(trace.fired, fault != C::kNone);
    EXPECT_EQ(trace.freed_expected_pointer, fault == C::kNone);
    const std::vector<C> expected =
        fault == C::kSelect ? std::vector<C>{C::kSelect}
        : fault == C::kAllocate
            ? std::vector<C>{C::kSelect, C::kAllocate}
            : std::vector<C>{C::kSelect, C::kAllocate, C::kSelect, C::kFree};
    EXPECT_EQ(trace.calls, expected);
  }
}

template <typename F>
void ExpectCudaFailure(CudaCall fault,
                       const char* operation,
                       const std::vector<CudaCall>& calls,
                       F&& action,
                       cudaMemcpyKind copy_kind = cudaMemcpyDeviceToHost) {
  CudaTrace trace;
  trace.copy_kind = copy_kind;
  bool threw = false;
  {
    ScopedCudaFault scope(trace, fault);
    try {
      action();
    } catch (const std::runtime_error& error) {
      threw = true;
      EXPECT_NE(std::string(error.what()).find(operation), std::string::npos);
      EXPECT_NE(std::string(error.what())
                    .find(cudaGetErrorString(kInjectedCudaError)),
                std::string::npos);
    }
  }
  EXPECT_TRUE(threw);
  EXPECT_TRUE(trace.fired);
  EXPECT_EQ(trace.calls, calls);
  if (fault == CudaCall::kMap) {
    EXPECT_EQ(trace.real_mapping_status, cudaSuccess);
  }
}

TEST(CasparUploadFault, StopsOnCudaFailure) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "No CUDA devices available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  using Real = StorageType;
  using C = CudaCall;
  CasparSolverSizing sizing;
  sizing.num_points = 1;
  sizing.num_simple_radial_poses = 1;
  sizing.num_simple_radial_calibs = 1;
  sizing.num_simple_radial = 1;
  sizing.num_simple_radial_fixed_pose = 1;
  auto solver = CreateSolver(caspar::SolverParams<Real>{}, sizing, 0);
  const std::array<Real, 7> data{
      Real(1), Real(2), Real(3), Real(1), Real(0), Real(0), Real(0)};
  const std::array<unsigned, 1> indices{0};
  CudaDeviceBuffer device_data;
  CudaDeviceBuffer device_indices;
  ASSERT_EQ(cudaMalloc(&device_data.ptr, sizeof(data)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_indices.ptr, sizeof(indices)), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(
          device_data.ptr, data.data(), sizeof(data), cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_indices.ptr,
                       indices.data(),
                       sizeof(indices),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  solver.SetPointNodesFromStackedHost(data.data(), 0, 1);
  CudaTrace probe;
  {
    ScopedCudaFault scope(probe, C::kNone);
    std::array<Real, 3> readback{};
    solver.GetPointNodesToStackedHost(readback.data(), 0, 1);
  }
  if (probe.calls != std::vector<C>{C::kSelect, C::kMap, C::kCopy}) {
    GTEST_SKIP() << "Generated CUDA calls are not fully intercepted";
  }

  // Node, factor constant, and factor index uploads have distinct generators.
  for (const int family : {0, 1, 2}) {
    SCOPED_TRACE(family);
    const auto host_upload = [&] {
      if (family == 0) {
        solver.SetPointNodesFromStackedHost(data.data(), 0, 1);
      } else if (family == 1) {
        solver.SetSimpleRadialFixedPosePoseDataFromStackedHost(
            data.data(), 0, 1);
      } else {
        solver.SetSimpleRadialPoseIndicesFromHost(indices.data(), 1);
      }
    };
    const auto device_upload = [&] {
      if (family == 0) {
        solver.SetPointNodesFromStackedDevice(
            static_cast<Real*>(device_data.ptr), 0, 1);
      } else if (family == 1) {
        solver.SetSimpleRadialFixedPosePoseDataFromStackedDevice(
            static_cast<Real*>(device_data.ptr), 0, 1);
      } else {
        solver.SetSimpleRadialPoseIndicesFromDevice(
            static_cast<unsigned*>(device_indices.ptr), 1);
      }
    };
    // Prime the scratch with valid data before any copy-failure control.
    host_upload();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (const auto fault : {C::kSelect, C::kCopy, C::kMap}) {
      const char* operation = fault == C::kSelect ? "cudaSetDevice"
                              : fault == C::kCopy ? "cudaMemcpy host-to-device"
                              : family == 0       ? "PointStackedToCaspar"
                              : family == 1
                                  ? "ConstSimpleRadialPoseStackedToCaspar"
                                  : "SharedIndices";
      const std::vector<C> expected =
          fault == C::kSelect ? std::vector<C>{C::kSelect}
          : fault == C::kCopy ? std::vector<C>{C::kSelect, C::kCopy}
          : family == 2
              ? std::vector<C>{C::kSelect, C::kCopy, C::kSelect, C::kMap}
              : std::vector<C>{C::kSelect, C::kCopy, C::kMap};
      ExpectCudaFailure(
          fault, operation, expected, host_upload, cudaMemcpyHostToDevice);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    for (const auto fault : {C::kSelect, C::kMap}) {
      const char* operation = fault == C::kSelect ? "cudaSetDevice"
                              : family == 0       ? "PointStackedToCaspar"
                              : family == 1
                                  ? "ConstSimpleRadialPoseStackedToCaspar"
                                  : "SharedIndices";
      const std::vector<C> expected = fault == C::kSelect
                                          ? std::vector<C>{C::kSelect}
                                          : std::vector<C>{C::kSelect, C::kMap};
      ExpectCudaFailure(fault, operation, expected, device_upload);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
  }

  CudaTrace host_only;
  {
    ScopedCudaFault scope(host_only, C::kNone);
    solver.SetPointNum(1);
    solver.SetPointNodesFromStackedHost(data.data(), 1, 0);
    solver.SetPointNodesFromStackedDevice(
        static_cast<Real*>(device_data.ptr), 1, 0);
    solver.SetSimpleRadialFixedPosePoseDataFromStackedHost(data.data(), 0, 0);
    solver.SetSimpleRadialFixedPosePoseDataFromStackedDevice(
        static_cast<Real*>(device_data.ptr), 0, 0);
    solver.SetSimpleRadialNum(0);
    solver.SetSimpleRadialPoseIndicesFromHost(indices.data(), 0);
    solver.SetSimpleRadialPoseIndicesFromDevice(
        static_cast<unsigned*>(device_indices.ptr), 0);
    solver.SetSimpleRadialNum(1);
    EXPECT_THROW(solver.SetPointNodesFromStackedHost(data.data(), 2, 0),
                 std::runtime_error);
    EXPECT_THROW(solver.SetSimpleRadialFixedPosePoseDataFromStackedHost(
                     data.data(), std::numeric_limits<size_t>::max(), 0),
                 std::runtime_error);
    EXPECT_THROW(solver.SetSimpleRadialPoseIndicesFromHost(indices.data(), 0),
                 std::runtime_error);
  }
  EXPECT_TRUE(host_only.calls.empty());
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  // A zero-factor, zero-iteration control remains safe if an unchecked owner
  // continues past an injected solve-entry error.
  caspar::SolverParams<Real> params;
  params.solver_iter_max = 0;
  auto empty_solver = CreateSolver(params, {}, 0);
  EXPECT_NO_THROW(empty_solver.solve(false, false));
  for (const auto fault : {C::kSelect, C::kCopy}) {
    const std::vector<C> expected = fault == C::kSelect
                                        ? std::vector<C>{C::kSelect}
                                        : std::vector<C>{C::kSelect, C::kCopy};
    ExpectCudaFailure(
        fault,
        fault == C::kSelect ? "cudaSetDevice" : "cudaMemcpy host-to-device",
        expected,
        [&] { empty_solver.solve(false, false); },
        cudaMemcpyHostToDevice);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
}

TEST(CasparReadbackFault, GetterStagesAndRealRoundTrips) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  using Real = StorageType;
  CasparSolverSizing sizing;
  sizing.num_points = 1;
  auto solver = CreateSolver(caspar::SolverParams<StorageType>{}, sizing, 0);
  const std::array<Real, 3> input{Real(1), Real(2), Real(3)};
  solver.SetPointNodesFromStackedHost(input.data(), 0, 1);
  std::array<Real, 3> host{};
  CudaTrace probe;
  {
    ScopedCudaFault scope(probe, CudaCall::kNone);
    solver.GetPointNodesToStackedHost(host.data(), 0, 1);
  }
  EXPECT_EQ(host, input);
  // Verify interception reaches the generated getter, including CUDA's actual
  // copy entry point. Per-thread-default-stream aliases may bypass this wrap.
  const std::vector<CudaCall> expected_probe{
      CudaCall::kSelect, CudaCall::kMap, CudaCall::kCopy};
  if (probe.calls != expected_probe) {
    GTEST_SKIP() << "Generated getter CUDA calls are not fully intercepted";
  }
  CudaDeviceBuffer device;
  ASSERT_EQ(cudaMalloc(&device.ptr, sizeof(input)), cudaSuccess);
  const std::array<Real, 3> sentinel{Real(-11), Real(-12), Real(-13)};
  ASSERT_EQ(cudaMemcpy(device.ptr,
                       sentinel.data(),
                       sizeof(sentinel),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  host.fill(Real(-99));
  solver.GetPointNodesToStackedDevice(static_cast<Real*>(device.ptr), 0, 1);
  // Test-side copy happens after the getter and may synchronize; no claim of
  // asynchronous completion or instrumentation of all synchronization APIs.
  ASSERT_EQ(cudaMemcpy(
                host.data(), device.ptr, sizeof(input), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(host, input);

  using C = CudaCall;
  for (const auto fault : {C::kSelect, C::kMap, C::kCopy}) {
    const std::vector<C> expected =
        fault == C::kSelect ? std::vector<C>{C::kSelect}
        : fault == C::kMap  ? std::vector<C>{C::kSelect, C::kMap}
                            : std::vector<C>{C::kSelect, C::kMap, C::kCopy};
    const char* operation = fault == C::kSelect ? "cudaSetDevice"
                            : fault == C::kMap  ? "PointCasparToStacked"
                                                : "cudaMemcpy";
    ExpectCudaFailure(fault, operation, expected, [&] {
      solver.GetPointNodesToStackedHost(host.data(), 0, 1);
    });
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  for (const auto fault : {C::kSelect, C::kMap}) {
    const std::vector<C> expected = fault == C::kSelect
                                        ? std::vector<C>{C::kSelect}
                                        : std::vector<C>{C::kSelect, C::kMap};
    ExpectCudaFailure(
        fault,
        fault == C::kSelect ? "cudaSetDevice" : "PointCasparToStacked",
        expected,
        [&] {
          solver.GetPointNodesToStackedDevice(
              static_cast<Real*>(device.ptr), 0, 1);
        });
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
}

TEST(CasparReadbackFault, ScalarCopyFailure) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  using Real = StorageType;
  CudaDeviceBuffer buffer;
  ASSERT_EQ(cudaMalloc(&buffer.ptr, sizeof(Real)), cudaSuccess);
  const Real input = Real(0.25);
  ASSERT_EQ(
      cudaMemcpy(buffer.ptr, &input, sizeof(input), cudaMemcpyHostToDevice),
      cudaSuccess);
  CudaTrace probe;
  Real result;
  {
    ScopedCudaFault scope(probe, CudaCall::kNone);
    result = caspar::ReadCuMem(static_cast<const Real*>(buffer.ptr));
  }
  EXPECT_EQ(result, input);
  if (probe.calls != std::vector<CudaCall>{CudaCall::kCopy}) {
    GTEST_SKIP() << "Generated scalar CUDA copy is not intercepted";
  }
  ExpectCudaFailure(CudaCall::kCopy, "ReadCuMem", {CudaCall::kCopy}, [&] {
    (void)caspar::ReadCuMem(static_cast<const Real*>(buffer.ptr));
  });
}

TEST(CasparReadbackFault, EmptyAndOverflowRange) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  CasparSolverSizing sizing;
  sizing.num_points = 1;
  auto solver = CreateSolver(caspar::SolverParams<StorageType>{}, sizing, 0);
  using Getter = void (caspar::GraphSolver::*)(StorageType*, size_t, size_t);
  for (const Getter getter :
       {&caspar::GraphSolver::GetPointNodesToStackedHost,
        &caspar::GraphSolver::GetPointNodesToStackedDevice}) {
    CudaTrace trace;
    {
      ScopedCudaFault scope(trace, CudaCall::kSelect);
      EXPECT_NO_THROW((solver.*getter)(nullptr, 1, 0));
      EXPECT_THROW((solver.*getter)(nullptr, 2, 0), std::runtime_error);
      EXPECT_THROW(
          (solver.*getter)(nullptr, std::numeric_limits<size_t>::max(), 2),
          std::runtime_error);
      // Reject capacities outside the mapper's supported unsigned-index
      // range before touching CUDA, including a small read within that pool.
      caspar::GraphSolverTestAccess::SetPointReadbackSize(
          solver, 1, std::numeric_limits<unsigned int>::max() / 4 + 1);
      EXPECT_THROW((solver.*getter)(nullptr, 0, 1), std::runtime_error);
      caspar::GraphSolverTestAccess::SetPointReadbackSize(solver, 1, 1);
    }
    EXPECT_TRUE(trace.calls.empty());
    EXPECT_FALSE(trace.fired);
  }
}

#else
TEST(CasparConstruction, WrappingUnavailable) {
  GTEST_SKIP()
      << "Constructor fault injection requires a GNU/Linux static test link";
}
TEST(CasparUploadFault, WrappingUnavailable) {
  GTEST_SKIP() << "CUDA fault wrapping unavailable";
}

TEST(CasparReadbackFault, WrappingUnavailable) {
  GTEST_SKIP()
      << "Readback fault injection requires a GNU/Linux static test link";
}
#endif
}  // namespace
}  // namespace colmap
