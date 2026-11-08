// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#if defined(COLMAP_TEST_CASPAR_READBACK_WRAPPING)
#ifdef CASPAR_USE_DOUBLE
#include "thirdparty/Symforce-Caspar/generated/f64/solver_tools.h"
#else
#include "thirdparty/Symforce-Caspar/generated/f32/solver_tools.h"
#endif

namespace {
enum class ReadbackCall { kNone, kSelect, kMap, kCopy };
struct NodeReadbackTrace {
  struct Event {
    size_t bytes = 0;
    cudaError_t status = cudaSuccess;
    bool actual_copy = false;
  };
  std::array<size_t, 9> expected{};
  std::array<Event, 9> events{};
  size_t expected_count = 0;
  size_t count = 0;
  size_t scalar_bytes = 0;
  size_t scalar_copies = 0;
  size_t fail_at = std::numeric_limits<size_t>::max();
  bool started = false;
  bool mismatch = false;
  bool fired = false;
  bool multi_model = false;
  bool merged = false;
  bool order_known = false;
  bool simple_first = false;
  size_t simple_pose_bytes = 0;
  size_t pinhole_pose_bytes = 0;
};
struct ReadbackTrace {
  std::array<ReadbackCall, 16> calls{};
  size_t count = 0;
  bool overflow = false;
  bool fired = false;
  cudaError_t real_mapping_status = cudaSuccess;
  ReadbackCall fault = ReadbackCall::kSelect;
  NodeReadbackTrace* node_reads = nullptr;
};
thread_local ReadbackTrace* readback_trace = nullptr;
constexpr cudaError_t kInjectedReadbackError = cudaErrorInvalidValue;

void TraceReadbackCall(ReadbackCall call) {
  if (readback_trace == nullptr || readback_trace->node_reads != nullptr)
    return;
  if (readback_trace->count == readback_trace->calls.size()) {
    readback_trace->overflow = true;
    return;
  }
  readback_trace->calls[readback_trace->count++] = call;
}

class ScopedReadbackFault {
 public:
  ScopedReadbackFault(ReadbackTrace& trace, ReadbackCall fault)
      : trace_(trace) {
    if (readback_trace != nullptr) throw std::logic_error("nested fault arm");
    trace_.fault = fault;
    readback_trace = &trace_;
  }
  ~ScopedReadbackFault() { readback_trace = nullptr; }
  ScopedReadbackFault(const ScopedReadbackFault&) = delete;
  ScopedReadbackFault& operator=(const ScopedReadbackFault&) = delete;

 private:
  ReadbackTrace& trace_;
};
}  // namespace

extern "C" cudaError_t CUDARTAPI __real_cudaSetDevice(int);
extern "C" cudaError_t CUDARTAPI __real_cudaGetLastError();
extern "C" cudaError_t CUDARTAPI __real_cudaMemcpy(void*,
                                                   const void*,
                                                   size_t,
                                                   cudaMemcpyKind);

namespace {
cudaError_t CopyWithNodeTrace(void* destination,
                              const void* source,
                              size_t bytes,
                              cudaMemcpyKind kind,
                              NodeReadbackTrace& trace) {
  if (kind != cudaMemcpyDeviceToHost || trace.mismatch) {
    return __real_cudaMemcpy(destination, source, bytes, kind);
  }
  if (!trace.started && bytes != trace.expected[0]) {
    // Current solver score/norm reads contain one scalar or the paired score
    // and prediction. Anything else before the unique point read is ambiguous.
    if (bytes != trace.scalar_bytes && bytes != 2 * trace.scalar_bytes) {
      trace.mismatch = true;
    }
    ++trace.scalar_copies;
    const cudaError_t status =
        __real_cudaMemcpy(destination, source, bytes, kind);
    trace.mismatch |= status != cudaSuccess;
    return status;
  }
  trace.started = true;
  // The two model-pose sizes differ. Identify this solver's adapter order
  // from its own first pose read, then require the calibration groups in the
  // same order used by the unchanged adapter container in ReadSolverResults.
  if (trace.multi_model && trace.count == 1 && !trace.order_known) {
    if (bytes != trace.simple_pose_bytes && bytes != trace.pinhole_pose_bytes) {
      trace.mismatch = true;
      return __real_cudaMemcpy(destination, source, bytes, kind);
    }
    trace.simple_first = bytes == trace.simple_pose_bytes;
    trace.order_known = true;
    trace.expected[1] = bytes;
    trace.expected[2] =
        trace.simple_first ? trace.pinhole_pose_bytes : trace.simple_pose_bytes;
    size_t next = 3;
    for (const bool simple : {trace.simple_first, !trace.simple_first}) {
      const size_t cameras = simple ? 1 : 2;
      trace.expected[next++] = 2 * cameras * trace.scalar_bytes;
      trace.expected[next++] = 2 * cameras * trace.scalar_bytes;
      if (trace.merged) {
        trace.expected[next++] = 4 * cameras * trace.scalar_bytes;
      }
    }
    if (next != trace.expected_count) {
      trace.mismatch = true;
      return __real_cudaMemcpy(destination, source, bytes, kind);
    }
  }
  if (trace.count >= trace.expected_count ||
      trace.count >= trace.events.size() ||
      bytes != trace.expected[trace.count]) {
    trace.mismatch = true;
    return __real_cudaMemcpy(destination, source, bytes, kind);
  }
  auto& event = trace.events[trace.count++];
  event.bytes = bytes;
  if (trace.count - 1 == trace.fail_at) {
    trace.fired = true;
    event.status = kInjectedReadbackError;
    return event.status;
  }
  event.actual_copy = true;
  event.status = __real_cudaMemcpy(destination, source, bytes, kind);
  trace.mismatch |= event.status != cudaSuccess;
  return event.status;
}
}  // namespace

extern "C" cudaError_t CUDARTAPI __wrap_cudaSetDevice(int device) {
  if (readback_trace == nullptr) return __real_cudaSetDevice(device);
  TraceReadbackCall(ReadbackCall::kSelect);
  if (readback_trace->fault == ReadbackCall::kSelect) {
    readback_trace->fired = true;
    return kInjectedReadbackError;
  }
  return __real_cudaSetDevice(device);
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaGetLastError() {
  const cudaError_t actual = __real_cudaGetLastError();
  if (readback_trace == nullptr) return actual;
  TraceReadbackCall(ReadbackCall::kMap);
  readback_trace->real_mapping_status = actual;
  if (readback_trace->fault == ReadbackCall::kMap && actual == cudaSuccess) {
    readback_trace->fired = true;
    return kInjectedReadbackError;
  }
  return actual;
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaMemcpy(void* destination,
                                                   const void* source,
                                                   size_t size,
                                                   cudaMemcpyKind kind) {
  if (readback_trace == nullptr)
    return __real_cudaMemcpy(destination, source, size, kind);
  if (readback_trace->node_reads != nullptr) {
    return CopyWithNodeTrace(
        destination, source, size, kind, *readback_trace->node_reads);
  }
  TraceReadbackCall(ReadbackCall::kCopy);
  if (readback_trace->fault == ReadbackCall::kCopy &&
      kind == cudaMemcpyDeviceToHost) {
    readback_trace->fired = true;
    return kInjectedReadbackError;
  }
  return __real_cudaMemcpy(destination, source, size, kind);
}
#endif

namespace caspar {
struct GraphSolverTestAccess {
  using Real = colmap::StorageType;
  static size_t InputScratchBytes(const GraphSolver& solver) {
    const auto offset =
        reinterpret_cast<uintptr_t>(solver.marker__scratch_inout_) -
        reinterpret_cast<uintptr_t>(solver.origin_ptr_);
    return solver.allocation_size_ - offset;
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

caspar::GraphSolver MakeParameterTestSolver(
    const caspar::SolverParams<double>& params, const int device) {
  return caspar::GraphSolver(params,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             device);
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

TEST(CasparFirstNorm, ResetsBothAccumulatorsBeforeRealKernelUpdates) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }

  using Access = caspar::GraphSolverTestAccess;
  using Real = StorageType;
  auto solver = MakeParameterTestSolver(caspar::SolverParams<double>(), 0);
  // The constructor allocates one of every node; only these two are
  // initialized.
  solver.SetPinholeCalibNum(0);
  solver.SetPinholeFocalNum(0);
  solver.SetPinholePoseNum(0);
  solver.SetSimpleRadialCalibNum(0);
  solver.SetSimpleRadialFocalAndExtraNum(0);
  solver.SetSimpleRadialPoseNum(0);
  solver.SetSimpleRadialPrincipalPointNum(0);
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

TEST(CasparParameters, RejectsInvalidUpdates) {
  caspar::SolverParams<double> valid;
  auto solver = MakeParameterTestSolver(valid, 0);
  const auto expect_constructor_error = [](const auto& params,
                                           const char* message) {
    try {
      MakeParameterTestSolver(params, -1);
      ADD_FAILURE() << "Expected constructor parameter error";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), message);
    }
  };
  auto invalid = valid;
  for (const int count : {0, -1}) {
    invalid = valid;
    invalid.pcg_iter_max = count;
    EXPECT_THROW(solver.set_params(invalid), std::runtime_error);
    expect_constructor_error(invalid, "params.pcg_iter_max must be at least 1");
  }
  for (const double threshold : {0.0, -1.0}) {
    invalid = valid;
    invalid.pcg_rel_error_exit = threshold;
    EXPECT_THROW(solver.set_params(invalid), std::runtime_error);
    expect_constructor_error(invalid,
                             "params.pcg_rel_error_exit must be positive");
  }
  invalid = valid;
  invalid.diag_init = -1.0;
  EXPECT_THROW(solver.set_params(invalid), std::runtime_error);
  expect_constructor_error(invalid, "params.diag_init must be positive");
  valid.diag_init = 0.0;
  valid.pcg_iter_max = 1;
  EXPECT_NO_THROW(solver.set_params(valid));
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
  for (const bool decrease_enabled : {false, true}) {
    for (const bool score_exit_enabled : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "decrease=" << decrease_enabled
                   << " score_exit=" << score_exit_enabled);
      Reconstruction reconstruction = initial;
      BundleAdjustmentOptions options;
      options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
      options.caspar->solver_iter_max = 1;
      options.caspar->pcg_iter_max = 1;
      options.caspar->pcg_rel_decrease_min = decrease_enabled ? 1.0 : -1.0;
      options.caspar->pcg_rel_score_exit = score_exit_enabled ? 0.5 : -1.0;
      options.caspar->collect_iteration_data = true;
      auto adjuster =
          CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
      const auto summary =
          std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
              adjuster->Solve());
      ASSERT_NE(summary, nullptr);
      ASSERT_EQ(summary->iterations.size(), 1);
      const auto& iteration = summary->iterations.front();
      ASSERT_GT(summary->initial_score, 0);
      EXPECT_TRUE(iteration.step_accepted);
      EXPECT_EQ(iteration.pcg_iter, 0);
      EXPECT_LT(iteration.score_best, summary->initial_score);
      EXPECT_TRUE(std::isfinite(iteration.step_quality));
      if (!decrease_enabled && !score_exit_enabled) {
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
      const auto solve = [&](int pcg_limit) {
        Reconstruction reconstruction = initial;
        BundleAdjustmentOptions options;
        options.refine_principal_point = true;
        options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
        options.caspar->solver_iter_max = 2;
        options.caspar->pcg_iter_max = pcg_limit;
        options.caspar->diag_init = diag;
        options.caspar->diag_scaling_down = 0;
        options.caspar->pcg_rel_decrease_min = decrease;
        options.caspar->pcg_rel_score_exit = -1;
        options.caspar->pcg_rel_error_exit = 1e-12;
        options.caspar->collect_iteration_data = true;
        options.caspar->gpu_index = "0";
        auto adjuster =
            CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
        return std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
            adjuster->Solve());
      };
      const auto first = solve(1);
      const auto second = solve(2);
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
      EXPECT_NEAR(first->iterations[1].diag,
                  second->iterations[1].diag,
                  2e-5 * std::max(1e-9, std::abs(first->iterations[1].diag)));
      return;
    }
  }
  FAIL() << "No retained-then-rejected witness in the fixed fixture grid";
}

TEST(DefaultBundleAdjuster, RigThrowsErrorOnVariableSensorFromRig) {
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

TEST(DefaultBundleAdjuster, MixedCameraModelsRespectFramePoseCapability) {
  enum class Case {
    kVariable,
    kFixed,
    kGloballyFixed,
    kSeparate,
    kUnsupported
  };
  for (const Case test : {Case::kVariable,
                          Case::kFixed,
                          Case::kGloballyFixed,
                          Case::kSeparate,
                          Case::kUnsupported}) {
    SCOPED_TRACE(static_cast<int>(test));
    Reconstruction reconstruction;
    SyntheticDatasetOptions dataset;
    dataset.num_rigs = 1;
    dataset.num_cameras_per_rig = 2;
    dataset.num_frames_per_rig = 2;
    SynthesizeDataset(dataset, &reconstruction);

    const camera_t camera_id = 2;
    const Camera old_camera = reconstruction.Camera(camera_id);
    reconstruction.Camera(camera_id) = Camera::CreateFromModelId(
        camera_id,
        test == Case::kUnsupported ? CameraModelId::kOpenCV
                                   : CameraModelId::kPinhole,
        old_camera.MeanFocalLength(),
        old_camera.width,
        old_camera.height);
    const Reconstruction original = reconstruction;

    BundleAdjustmentConfig config;
    for (const image_t image_id : reconstruction.RegImageIds()) {
      const Image& image = reconstruction.Image(image_id);
      if (test != Case::kSeparate ||
          (image.FrameId() == 1 && image.CameraId() == 1) ||
          (image.FrameId() == 2 && image.CameraId() == 2)) {
        config.AddImage(image_id);
      }
    }
    ASSERT_EQ(config.Images().size(), test == Case::kSeparate ? 2 : 4);
    if (test == Case::kVariable) {
      // Gauge fixing pins frame 1, leaving the mixed frame 2 variable.
      config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
      ASSERT_EQ(reconstruction.Image(1).FrameId(), 1);
      ASSERT_TRUE(reconstruction.Image(1).IsRefInFrame());
      ASSERT_EQ(reconstruction.Image(3).FrameId(), 2);
      ASSERT_EQ(reconstruction.Image(4).FrameId(), 2);
      ASSERT_NE(reconstruction.Image(3).CameraId(),
                reconstruction.Image(4).CameraId());
    } else if (test == Case::kFixed) {
      config.SetConstantRigFromWorldPose(1);
      config.SetConstantRigFromWorldPose(2);
    }
    BundleAdjustmentOptions options;
    options.refine_sensor_from_rig = false;
    options.refine_rig_from_world = test != Case::kGloballyFixed;

    if (test == Case::kVariable) {
      try {
        CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
        FAIL() << "Expected a mixed-model variable-frame refusal";
      } catch (const std::invalid_argument& error) {
        EXPECT_NE(
            std::string(error.what())
                .find(
                    "Caspar cannot refine a frame with multiple camera models"),
            std::string::npos);
      }
    } else {
      EXPECT_NO_THROW({
        const auto adjuster =
            CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
        EXPECT_NE(adjuster, nullptr);
      });
    }
    for (const frame_t frame_id : {frame_t{1}, frame_t{2}}) {
      EXPECT_THAT(reconstruction.Frame(frame_id).RigFromWorld(),
                  Rigid3dEq(original.Frame(frame_id).RigFromWorld()));
    }
    EXPECT_EQ(reconstruction.Camera(1), original.Camera(1));
    EXPECT_EQ(reconstruction.Camera(2), original.Camera(2));
  }
}

TEST(DefaultBundleAdjuster, MultiCameraRigLargeConstantSensorFromRig) {
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

TEST(DefaultBundleAdjuster, MergedCalibConvergence) {
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

TEST(DefaultBundleAdjuster, MergedCalibFixedPose) {
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

TEST(DefaultBundleAdjuster, MergedCalibFixedPoint) {
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

TEST(DefaultBundleAdjuster, ExternalImagePoseIsInvariant) {
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

// Evaluate the selected observations from the published reconstruction, not
// the solver's internal cost or the cached Point3D error.
double SelectedMeanSquaredReprojectionError(
    const Reconstruction& reconstruction,
    const std::vector<point3D_t>& point_ids,
    const std::vector<image_t>& image_ids) {
  double sum = 0;
  size_t count = 0;
  for (const point3D_t point_id : point_ids) {
    const Point3D& point = reconstruction.Point3D(point_id);
    for (const auto& element : point.track.Elements()) {
      if (std::find(image_ids.begin(), image_ids.end(), element.image_id) ==
          image_ids.end()) {
        continue;
      }
      const Image& image = reconstruction.Image(element.image_id);
      const auto projected =
          image.CameraPtr()->ImgFromCam(image.CamFromWorld() * point.xyz);
      if (!projected || !projected->allFinite()) {
        return std::numeric_limits<double>::infinity();
      }
      const Eigen::Vector2d residual =
          *projected - image.Point2D(element.point2D_idx).xy;
      if (!residual.allFinite()) {
        return std::numeric_limits<double>::infinity();
      }
      sum += residual.squaredNorm();
      ++count;
    }
  }
  return count == 0 ? std::numeric_limits<double>::infinity() : sum / count;
}

std::vector<point3D_t> SelectCommonPoints(const Reconstruction& reconstruction,
                                          BundleAdjustmentConfig& config,
                                          bool variable) {
  std::vector<point3D_t> selected;
  for (const auto& [id, point] : reconstruction.Points3D()) {
    bool has_first = false;
    bool has_second = false;
    for (const auto& element : point.track.Elements()) {
      has_first |= element.image_id == 1;
      has_second |= element.image_id == 2;
    }
    if (has_first && has_second && point.track.Length() == 2) {
      selected.push_back(id);
      if (variable) {
        config.AddVariablePoint(id);
      } else {
        config.AddConstantPoint(id);
      }
    } else {
      config.IgnorePoint(id);
    }
  }
  return selected;
}

TEST(DefaultBundleAdjuster, SharedCameraWithExternalObservations) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  enum class Case { kPrincipalPoint, kMerged, kConstantCamera };
  for (const Case test :
       {Case::kPrincipalPoint, Case::kMerged, Case::kConstantCamera}) {
    SCOPED_TRACE(static_cast<int>(test));
    SetPRNGSeed(0);
    Reconstruction reconstruction;
    SyntheticDatasetOptions dataset;
    dataset.num_rigs = 1;
    dataset.num_cameras_per_rig = 1;
    dataset.num_frames_per_rig = 2;
    dataset.num_points3D = 100;
    dataset.num_points2D_without_point3D = 0;
    SynthesizeDataset(dataset, &reconstruction);
    ASSERT_EQ(reconstruction.Image(1).CameraId(),
              reconstruction.Image(2).CameraId());
    ASSERT_NE(reconstruction.Image(1).FrameId(),
              reconstruction.Image(2).FrameId());
    const Reconstruction truth = reconstruction;
    Camera& camera = reconstruction.Camera(reconstruction.Image(1).CameraId());
    camera.params[1] += 5;
    camera.params[2] -= 7;
    if (test == Case::kMerged) {
      camera.params[0] += 30;
      camera.params[3] += 0.03;
    }
    BundleAdjustmentConfig config;
    config.AddImage(1);
    const bool constant_camera = test == Case::kConstantCamera;
    const auto selected =
        SelectCommonPoints(reconstruction, config, constant_camera);
    ASSERT_GE(selected.size(), 20);
    if (constant_camera) {
      config.SetConstantCamIntrinsics(camera.camera_id);
      for (const point3D_t id : selected) {
        reconstruction.Point3D(id).xyz.x() += 0.03;
      }
    }
    const Reconstruction before = reconstruction;
    const double error_before =
        SelectedMeanSquaredReprojectionError(before, selected, {1, 2});
    ASSERT_TRUE(std::isfinite(error_before));
    ASSERT_GT(error_before, 0.0);
    BundleAdjustmentOptions options;
    options.refine_focal_length = test != Case::kPrincipalPoint;
    options.refine_extra_params = test != Case::kPrincipalPoint;
    options.refine_principal_point = true;
    options.refine_rig_from_world = false;
    options.refine_sensor_from_rig = false;
    options.refine_points3D = constant_camera;
    auto adjuster =
        CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
    const auto summary = adjuster->Solve();
    ASSERT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    ASSERT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::USER_FAILURE);
    EXPECT_EQ(summary->num_residuals, 4 * selected.size());
    for (const double parameter : camera.params) {
      ASSERT_TRUE(std::isfinite(parameter));
    }
    const double error_after =
        SelectedMeanSquaredReprojectionError(reconstruction, selected, {1, 2});
    ASSERT_TRUE(std::isfinite(error_after));
    if (constant_camera) {
      EXPECT_EQ(camera, before.Camera(camera.camera_id));
    } else {
      // Demand a substantial published reprojection improvement and improvement
      // of every enabled calibration component toward the synthetic truth.
      EXPECT_LT(error_after, 0.5 * error_before);
      const auto& true_params = truth.Camera(camera.camera_id).params;
      const auto& initial_params = before.Camera(camera.camera_id).params;
      for (const size_t idx : {size_t{1}, size_t{2}}) {
        EXPECT_LT(std::abs(camera.params[idx] - true_params[idx]),
                  0.9 * std::abs(initial_params[idx] - true_params[idx]));
      }
      // Match the precision-specific tolerances of MergedCalibMatchesCeres.
#ifdef CASPAR_USE_DOUBLE
      constexpr double kFocalTol = 1.0;
      constexpr double kPPTol = 2.0;
      constexpr double kExtraTol = 5e-4;
#else
      constexpr double kFocalTol = 20.0;
      constexpr double kPPTol = 10.0;
      constexpr double kExtraTol = 1.5e-2;
#endif
      EXPECT_NEAR(camera.params[1], true_params[1], kPPTol);
      EXPECT_NEAR(camera.params[2], true_params[2], kPPTol);
      if (test == Case::kMerged) {
        EXPECT_NEAR(camera.params[0], true_params[0], kFocalTol);
        EXPECT_NEAR(camera.params[3], true_params[3], kExtraTol);
        for (const size_t idx : {size_t{0}, size_t{3}}) {
          EXPECT_LT(std::abs(camera.params[idx] - true_params[idx]),
                    0.9 * std::abs(initial_params[idx] - true_params[idx]));
        }
      } else {
        EXPECT_EQ(camera.params[0], initial_params[0]);
        EXPECT_EQ(camera.params[3], initial_params[3]);
      }
      for (const point3D_t id : selected) {
        CheckConstantPoint(reconstruction.Point3D(id), before.Point3D(id));
      }
    }
    for (const image_t image_id : reconstruction.RegImageIds()) {
      CheckConstantCamFromWorld(reconstruction.Image(image_id),
                                before.Image(image_id));
    }
  }
}

TEST(DefaultBundleAdjuster, ExternalSensorDoesNotFreezeConfiguredFrame) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 2;
  dataset.num_frames_per_rig = 1;
  dataset.num_points3D = 100;
  dataset.num_points2D_without_point3D = 0;
  dataset.sensor_from_rig_rotation_stddev = 0;
  dataset.sensor_from_rig_translation_stddev = 0;
  SynthesizeDataset(dataset, &reconstruction);
  ASSERT_EQ(reconstruction.Image(1).FrameId(),
            reconstruction.Image(2).FrameId());
  ASSERT_TRUE(reconstruction.Image(1).IsRefInFrame());
  ASSERT_FALSE(reconstruction.Image(2).IsRefInFrame());
  const Reconstruction truth = reconstruction;
  const frame_t frame_id = reconstruction.Image(1).FrameId();
  reconstruction.Frame(frame_id).RigFromWorld().translation().x() += 0.1;
  BundleAdjustmentConfig config;
  config.AddImage(1);
  const auto selected = SelectCommonPoints(reconstruction, config, false);
  ASSERT_GE(selected.size(), 20);
  const Reconstruction before = reconstruction;
  const double error_before =
      SelectedMeanSquaredReprojectionError(before, selected, {1});
  ASSERT_TRUE(std::isfinite(error_before));
  ASSERT_GT(error_before, 0.0);
  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  options.refine_principal_point = false;
  options.refine_sensor_from_rig = false;
  options.refine_points3D = false;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::USER_FAILURE);
  EXPECT_EQ(summary->num_residuals, 2 * selected.size());
  const Rigid3d& pose = reconstruction.Frame(frame_id).RigFromWorld();
  ASSERT_TRUE(pose.rotation().coeffs().allFinite());
  ASSERT_TRUE(pose.translation().allFinite());
  const double error_after =
      SelectedMeanSquaredReprojectionError(reconstruction, selected, {1});
  ASSERT_TRUE(std::isfinite(error_after));
  EXPECT_LT(error_after, 0.5 * error_before);
  const Eigen::Vector3d true_translation =
      truth.Frame(frame_id).RigFromWorld().translation();
  const double translation_error =
      (pose.translation() - true_translation).norm();
  const double initial_translation_error =
      (before.Frame(frame_id).RigFromWorld().translation() - true_translation)
          .norm();
  EXPECT_LT(translation_error, 0.9 * initial_translation_error);
  ASSERT_TRUE(std::isfinite(
      SelectedMeanSquaredReprojectionError(reconstruction, selected, {1, 2})));
  for (const point3D_t id : selected) {
    CheckConstantPoint(reconstruction.Point3D(id), before.Point3D(id));
  }
  EXPECT_EQ(reconstruction.Camera(1), before.Camera(1));
  EXPECT_EQ(reconstruction.Camera(2), before.Camera(2));
}

TEST(DefaultBundleAdjuster, ExternalCameraIntrinsicsOrderingIsConsistent) {
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

TEST(DefaultBundleAdjuster, ExternalImageViaConstantPointsIsInvariant) {
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

TEST(DefaultBundleAdjuster, MultipleExternalImagesAreInvariant) {
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

TEST(DefaultBundleAdjuster, MergedCalibMatchesCeres) {
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

TEST(DefaultBundleAdjuster, GaugeFixingWithOneFrameFromWorld) {
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

TEST(DefaultBundleAdjuster,
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

TEST(DefaultBundleAdjuster, GaugeFixingWithThreePoints_PinsExactlyThreePoints) {
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

std::vector<image_t> GaugeTrackImageIds(const Point3D& point) {
  std::vector<image_t> ids;
  for (const auto& element : point.track.Elements())
    ids.push_back(element.image_id);
  std::sort(ids.begin(), ids.end());
  return ids;
}

TEST(DefaultBundleAdjuster,
     GaugeFixingWithThreePoints_CountsExistingConstantPoints) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  struct AnchorCase {
    int implicit_points;
    int explicit_points;
    bool refine_points;
    bool fixed_pose_only = false;
  };
  for (const AnchorCase test : {AnchorCase{0, 1, true},
                                AnchorCase{0, 0, true},
                                AnchorCase{1, 0, true},
                                AnchorCase{2, 0, true},
                                AnchorCase{3, 0, true},
                                AnchorCase{3, 0, false},
                                AnchorCase{3, 0, true, true}}) {
    SCOPED_TRACE(::testing::Message()
                 << "implicit=" << test.implicit_points
                 << " explicit=" << test.explicit_points
                 << " refine_points=" << test.refine_points
                 << " fixed_pose_only=" << test.fixed_pose_only);
    SetPRNGSeed(0);
    Reconstruction reconstruction;
    SyntheticDatasetOptions opts;
    opts.num_rigs = 3;
    opts.num_cameras_per_rig = 1;
    opts.num_frames_per_rig = 1;
    opts.num_points3D = 100;
    SynthesizeDataset(opts, &reconstruction);
    ASSERT_EQ(reconstruction.NumPoints3D(), opts.num_points3D);
    for (const auto& [id, point] : reconstruction.Points3D()) {
      ASSERT_GE(id, 1);
      ASSERT_LE(id, opts.num_points3D);
      ASSERT_EQ(GaugeTrackImageIds(point), (std::vector<image_t>{1, 2, 3}));
    }
    // Only the selected anchors retain an observation outside the local BA.
    // Their full tracks therefore exceed the two included observations.
    const Image& external_image = reconstruction.Image(3);
    for (point2D_t idx = 0; idx < external_image.NumPoints2D(); ++idx) {
      const Point2D& point = external_image.Point2D(idx);
      if (point.HasPoint3D() &&
          point.point3D_id > static_cast<point3D_t>(test.implicit_points)) {
        reconstruction.DeleteObservation(3, idx);
      }
    }
    if (test.fixed_pose_only) {
      const Image& image = reconstruction.Image(2);
      for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
        const Point2D& point = image.Point2D(idx);
        if (point.HasPoint3D() &&
            point.point3D_id <= static_cast<point3D_t>(test.implicit_points)) {
          reconstruction.DeleteObservation(2, idx);
        }
      }
    }
    ASSERT_EQ(reconstruction.NumPoints3D(), opts.num_points3D);
    for (const auto& [id, point] : reconstruction.Points3D()) {
      const bool implicit = id <= static_cast<point3D_t>(test.implicit_points);
      const std::vector<image_t> expected =
          !implicit              ? std::vector<image_t>{1, 2}
          : test.fixed_pose_only ? std::vector<image_t>{1, 3}
                                 : std::vector<image_t>{1, 2, 3};
      ASSERT_EQ(GaugeTrackImageIds(point), expected);
    }
    SyntheticNoiseOptions noise_opts;
    noise_opts.point2D_stddev = 1;
    noise_opts.point3D_stddev = 0.1;
    noise_opts.rig_from_world_rotation_stddev = 0.3;
    noise_opts.rig_from_world_translation_stddev = 0.05;
    SynthesizeNoise(noise_opts, &reconstruction);
    const Reconstruction original = reconstruction;
    ASSERT_GT((original.Point3D(2).xyz - original.Point3D(1).xyz)
                  .cross(original.Point3D(3).xyz - original.Point3D(1).xyz)
                  .norm(),
              1e-3);
    BundleAdjustmentConfig config;
    config.AddImage(1);
    config.AddImage(2);
    for (int idx = 1; idx <= test.explicit_points; ++idx) {
      config.AddConstantPoint(idx);
    }
    config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
    if (test.fixed_pose_only) {
      config.SetConstantRigFromWorldPose(reconstruction.Image(1).FrameId());
      config.SetConstantCamIntrinsics(reconstruction.Image(2).CameraId());
    }
    BundleAdjustmentOptions options;
    options.refine_focal_length = test.fixed_pose_only;
    options.refine_extra_params = test.fixed_pose_only;
    options.refine_principal_point = false;
    options.refine_points3D = test.refine_points;
    options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
    options.caspar->collect_iteration_data = true;
    auto adjuster =
        CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
    const auto summary =
        std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
            adjuster->Solve());
    ASSERT_NE(summary, nullptr);
    ASSERT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    EXPECT_EQ(summary->num_residuals,
              400 - (test.fixed_pose_only ? 2 * test.implicit_points : 0));
    ASSERT_TRUE(std::isfinite(summary->initial_score));
    ASSERT_GT(summary->initial_score, 0);
    EXPECT_TRUE(std::any_of(summary->iterations.begin(),
                            summary->iterations.end(),
                            [&](const auto& iteration) {
                              return iteration.step_accepted &&
                                     std::isfinite(iteration.score_best) &&
                                     iteration.score_best <
                                         summary->initial_score;
                            }));
    int unchanged = 0;
    double largest_change = 0;
    for (const auto& [id, point] : reconstruction.Points3D()) {
      EXPECT_TRUE(point.xyz.allFinite());
      if (point.xyz == original.Point3D(id).xyz) ++unchanged;
      largest_change = std::max(largest_change,
                                (point.xyz - original.Point3D(id).xyz).norm());
    }
    EXPECT_EQ(unchanged,
              test.refine_points
                  ? 3 + (test.fixed_pose_only ? test.implicit_points : 0)
                  : opts.num_points3D);
    if (test.refine_points) {
      EXPECT_GT(largest_change, 1e-3);
    }
    for (int idx = 1; idx <= test.implicit_points; ++idx) {
      CheckConstantPoint(reconstruction.Point3D(idx), original.Point3D(idx));
    }
    for (int idx = 1; idx <= test.explicit_points; ++idx) {
      CheckConstantPoint(reconstruction.Point3D(idx), original.Point3D(idx));
    }
  }
}

TEST(DefaultBundleAdjuster,
     GaugeFixingWithThreePoints_IgnoresInactiveImplicitExternalPoints) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  SetPRNGSeed(0);
  Reconstruction reconstruction;
  SyntheticDatasetOptions opts;
  opts.num_rigs = 4;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 1;
  opts.num_points3D = 100;
  SynthesizeDataset(opts, &reconstruction);
  ASSERT_EQ(reconstruction.NumPoints3D(), opts.num_points3D);
  for (const auto& [id, point] : reconstruction.Points3D()) {
    ASSERT_EQ(GaugeTrackImageIds(point), (std::vector<image_t>{1, 2, 3, 4}));
  }
  for (const image_t image_id : {image_t{3}, image_t{4}}) {
    const Image& image = reconstruction.Image(image_id);
    for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
      if (image.Point2D(idx).HasPoint3D()) {
        reconstruction.DeleteObservation(image_id, idx);
      }
    }
  }
  ASSERT_EQ(reconstruction.NumPoints3D(), opts.num_points3D);
  for (const auto& [id, point] : reconstruction.Points3D()) {
    ASSERT_EQ(GaugeTrackImageIds(point), (std::vector<image_t>{1, 2}));
  }
  SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 1;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.3;
  noise_opts.rig_from_world_translation_stddev = 0.05;
  SynthesizeNoise(noise_opts, &reconstruction);
  const camera_t unsupported_id = reconstruction.Image(4).CameraId();
  const Camera old_camera = reconstruction.Camera(unsupported_id);
  reconstruction.Camera(unsupported_id) =
      Camera::CreateFromModelId(unsupported_id,
                                CameraModelId::kOpenCV,
                                old_camera.MeanFocalLength(),
                                old_camera.width,
                                old_camera.height);
  BundleAdjustmentConfig config;
  config.AddImage(1);
  config.AddImage(2);
  config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
  std::vector<point3D_t> external_ids;
  for (point2D_t idx = 0; idx < 3; ++idx) {
    ASSERT_FALSE(reconstruction.Image(3).Point2D(idx).HasPoint3D());
    ASSERT_FALSE(reconstruction.Image(4).Point2D(idx).HasPoint3D());
    Track track;
    track.AddElement(3, idx);
    track.AddElement(4, idx);
    const Eigen::Vector3d xyz(idx == 1 ? 1.0 : 0.0, idx == 2 ? 1.0 : 0.0, 4.0);
    const point3D_t id = reconstruction.AddPoint3D(xyz, track);
    external_ids.push_back(id);
    ASSERT_EQ(GaugeTrackImageIds(reconstruction.Point3D(id)),
              (std::vector<image_t>{3, 4}));
    config.AddVariablePoint(id);
  }
  const Reconstruction original = reconstruction;
  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  options.refine_principal_point = false;
  options.caspar = std::make_shared<CasparBundleAdjustmentOptions>();
  options.caspar->collect_iteration_data = true;
  auto adjuster =
      CreateDefaultCasparBundleAdjuster(options, config, reconstruction);
  const auto summary = std::dynamic_pointer_cast<CasparBundleAdjustmentSummary>(
      adjuster->Solve());
  ASSERT_NE(summary, nullptr);
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);
  EXPECT_EQ(summary->num_residuals, 400);
  ASSERT_TRUE(std::isfinite(summary->initial_score));
  ASSERT_GT(summary->initial_score, 0);
  EXPECT_TRUE(std::any_of(summary->iterations.begin(),
                          summary->iterations.end(),
                          [&](const auto& iteration) {
                            return iteration.step_accepted &&
                                   std::isfinite(iteration.score_best) &&
                                   iteration.score_best <
                                       summary->initial_score;
                          }));
  int unchanged_local = 0;
  double largest_change = 0;
  for (const auto& [id, point] : reconstruction.Points3D()) {
    EXPECT_TRUE(point.xyz.allFinite());
    if (std::find(external_ids.begin(), external_ids.end(), id) !=
        external_ids.end()) {
      CheckConstantPoint(point, original.Point3D(id));
    } else {
      if (point.xyz == original.Point3D(id).xyz) ++unchanged_local;
      largest_change = std::max(largest_change,
                                (point.xyz - original.Point3D(id).xyz).norm());
    }
  }
  EXPECT_EQ(unchanged_local, 3);
  EXPECT_GT(largest_change, 1e-3);
}

TEST(DefaultBundleAdjuster, MultiCameraRigResidualCountConstantSensorFromRig) {
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

TEST(DefaultBundleAdjuster, MultiCameraRigConstantRigPoseHoldsAllSensors) {
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

TEST(DefaultBundleAdjuster,
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

}  // namespace
}  // namespace colmap

namespace colmap {
namespace {
#if defined(COLMAP_TEST_CASPAR_READBACK_WRAPPING)
struct ReadbackDeviceBuffer {
  void* ptr = nullptr;
  ~ReadbackDeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
  ReadbackDeviceBuffer(const ReadbackDeviceBuffer&) = delete;
  ReadbackDeviceBuffer& operator=(const ReadbackDeviceBuffer&) = delete;
  ReadbackDeviceBuffer() = default;
};

template <typename F>
void ExpectReadbackFailure(ReadbackCall fault,
                           const char* operation,
                           const std::vector<ReadbackCall>& calls,
                           F&& action) {
  ReadbackTrace trace;
  bool threw = false;
  {
    ScopedReadbackFault scope(trace, fault);
    try {
      action();
    } catch (const std::runtime_error& error) {
      threw = true;
      EXPECT_NE(std::string(error.what()).find(operation), std::string::npos);
      EXPECT_NE(std::string(error.what())
                    .find(cudaGetErrorString(kInjectedReadbackError)),
                std::string::npos);
    }
  }
  EXPECT_TRUE(threw);
  EXPECT_TRUE(trace.fired);
  EXPECT_FALSE(trace.overflow);
  ASSERT_EQ(trace.count, calls.size());
  for (size_t i = 0; i < calls.size(); ++i) EXPECT_EQ(trace.calls[i], calls[i]);
  if (fault == ReadbackCall::kMap) {
    EXPECT_EQ(trace.real_mapping_status, cudaSuccess);
  }
}

TEST(CasparReadbackFault, GetterStagesAndRealRoundTrips) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  using Real = StorageType;
  auto solver = MakeParameterTestSolver(caspar::SolverParams<double>(), 0);
  solver.SetPointNum(1);
  const std::array<Real, 3> input{Real(1), Real(2), Real(3)};
  solver.SetPointNodesFromStackedHost(input.data(), 0, 1);
  std::array<Real, 3> host{};
  solver.GetPointNodesToStackedHost(host.data(), 0, 1);
  EXPECT_EQ(host, input);
  ReadbackDeviceBuffer device;
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

  using C = ReadbackCall;
  for (const auto fault : {C::kSelect, C::kMap, C::kCopy}) {
    const std::vector<C> expected =
        fault == C::kSelect ? std::vector<C>{C::kSelect}
        : fault == C::kMap  ? std::vector<C>{C::kSelect, C::kMap}
                            : std::vector<C>{C::kSelect, C::kMap, C::kCopy};
    const char* operation = fault == C::kSelect ? "cudaSetDevice"
                            : fault == C::kMap  ? "PointCasparToStacked"
                                                : "cudaMemcpy";
    ExpectReadbackFailure(fault, operation, expected, [&] {
      solver.GetPointNodesToStackedHost(host.data(), 0, 1);
    });
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  for (const auto fault : {C::kSelect, C::kMap}) {
    const std::vector<C> expected = fault == C::kSelect
                                        ? std::vector<C>{C::kSelect}
                                        : std::vector<C>{C::kSelect, C::kMap};
    ExpectReadbackFailure(
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

template <typename Real>
void CheckScalarReadback() {
  ReadbackDeviceBuffer buffer;
  ASSERT_EQ(cudaMalloc(&buffer.ptr, sizeof(Real)), cudaSuccess);
  const Real input = Real(0.25);
  ASSERT_EQ(
      cudaMemcpy(buffer.ptr, &input, sizeof(input), cudaMemcpyHostToDevice),
      cudaSuccess);
  EXPECT_EQ(caspar::ReadCuMem(static_cast<const Real*>(buffer.ptr)), input);
  ExpectReadbackFailure(
      ReadbackCall::kCopy, "ReadCuMem", {ReadbackCall::kCopy}, [&] {
        (void)caspar::ReadCuMem(static_cast<const Real*>(buffer.ptr));
      });
}

TEST(CasparReadbackFault, ScalarCopyFailure) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    GTEST_SKIP() << "CUDA unavailable";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  CheckScalarReadback<float>();
  CheckScalarReadback<double>();
}

TEST(CasparReadbackFault, EmptyAndOverflowRange) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    GTEST_SKIP() << "CUDA unavailable";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto solver = MakeParameterTestSolver(caspar::SolverParams<double>(), 0);
  solver.SetPointNum(1);
  // Verify wrapping reaches this generated getter, not merely this test TU.
  using Real = StorageType;
  const std::array<Real, 3> input{Real(1), Real(2), Real(3)};
  solver.SetPointNodesFromStackedHost(input.data(), 0, 1);
  std::array<Real, 3> observed{};
  ReadbackTrace probe;
  {
    ScopedReadbackFault scope(probe, ReadbackCall::kNone);
    solver.GetPointNodesToStackedHost(observed.data(), 0, 1);
  }
  ASSERT_EQ(observed, input);
  ASSERT_FALSE(probe.overflow);
  ASSERT_EQ(probe.real_mapping_status, cudaSuccess);
  ASSERT_EQ(probe.count, 3);
  ASSERT_EQ(probe.calls[0], ReadbackCall::kSelect);
  ASSERT_EQ(probe.calls[1], ReadbackCall::kMap);
  ASSERT_EQ(probe.calls[2], ReadbackCall::kCopy);
  ReadbackTrace trace;
  {
    ScopedReadbackFault scope(trace, ReadbackCall::kNone);
    EXPECT_NO_THROW(solver.GetPointNodesToStackedHost(nullptr, 1, 0));
    EXPECT_NO_THROW(solver.GetPointNodesToStackedDevice(nullptr, 1, 0));
    EXPECT_THROW(solver.GetPointNodesToStackedHost(nullptr, 2, 0),
                 std::runtime_error);
    EXPECT_THROW(solver.GetPointNodesToStackedDevice(nullptr, 2, 0),
                 std::runtime_error);
    EXPECT_THROW(solver.GetPointNodesToStackedHost(
                     nullptr, std::numeric_limits<size_t>::max(), 2),
                 std::runtime_error);
    EXPECT_THROW(solver.GetPointNodesToStackedDevice(
                     nullptr, std::numeric_limits<size_t>::max(), 2),
                 std::runtime_error);
  }
  EXPECT_EQ(trace.count, 0);
  EXPECT_FALSE(trace.overflow);
}
struct PublicationFixture {
  Reconstruction reconstruction;
  BundleAdjustmentConfig config;
  BundleAdjustmentOptions options;
  NodeReadbackTrace node_reads;
};

void PreparePublicationFixture(bool merged, PublicationFixture* fixture) {
  SetPRNGSeed(0);
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 2;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 2;
  dataset.num_points3D = 64;
  dataset.num_points2D_without_point3D = 0;
  SynthesizeDataset(dataset, &fixture->reconstruction);
  auto& reconstruction = fixture->reconstruction;
  std::set<point3D_t> points;
  std::set<frame_t> frames;
  std::set<camera_t> cameras;
  for (const image_t id : reconstruction.RegImageIds()) {
    fixture->config.AddImage(id);
    const auto& image = reconstruction.Image(id);
    frames.insert(image.FrameId());
    cameras.insert(image.CameraId());
    ASSERT_FALSE(image.Points2D().empty());
    for (const auto& point : image.Points2D()) {
      if (point.HasPoint3D()) points.insert(point.point3D_id);
    }
  }
  ASSERT_EQ(frames.size(), 4);
  ASSERT_EQ(cameras.size(), 2);
  ASSERT_GE(points.size(), 20);
  for (const camera_t id : cameras) {
    auto& camera = reconstruction.Camera(id);
    ASSERT_EQ(camera.model_id, CameraModelId::kSimpleRadial);
    ASSERT_EQ(camera.params.size(), 4);
    camera.params[0] += 30;
    camera.params[3] += 0.03;
    if (merged) {
      camera.params[1] += 5;
      camera.params[2] -= 7;
    }
  }
  auto& options = fixture->options;
  options.refine_rig_from_world = false;
  options.refine_sensor_from_rig = false;
  options.refine_points3D = false;
  options.refine_focal_length = true;
  options.refine_extra_params = true;
  options.refine_principal_point = merged;
  ASSERT_NE(options.caspar, nullptr);
  options.min_track_length = 0;
  options.caspar->solver_iter_max = 5;
  options.caspar->pcg_iter_max = 30;
  auto& trace = fixture->node_reads;
  trace.scalar_bytes = sizeof(StorageType);
  trace.expected = {3 * points.size() * sizeof(StorageType),
                    7 * frames.size() * sizeof(StorageType),
                    2 * cameras.size() * sizeof(StorageType),
                    2 * cameras.size() * sizeof(StorageType),
                    4 * cameras.size() * sizeof(StorageType)};
  trace.expected_count = merged ? 5 : 4;
  // Point is the first readback group and its distinct size identifies the
  // phase within this Solve, without importing a pointer/ordinal from a peer.
  for (size_t i = 1; i < trace.expected_count; ++i) {
    ASSERT_NE(trace.expected[0], trace.expected[i]);
  }
  ASSERT_GT(trace.expected[0], 2 * trace.scalar_bytes);
}

std::string DescribeNodeReadback(const NodeReadbackTrace& trace) {
  std::ostringstream stream;
  stream << "scalar=" << trace.scalar_copies << ";nodes=" << trace.count
         << ";target=" << trace.fail_at << ";fired=" << trace.fired
         << ";mismatch=" << trace.mismatch
         << ";order_known=" << trace.order_known
         << ";simple_first=" << trace.simple_first;
  for (size_t i = 0; i < trace.count && i < trace.events.size(); ++i) {
    stream << ";" << trace.events[i].bytes << ":"
           << static_cast<int>(trace.events[i].status) << ":"
           << trace.events[i].actual_copy;
  }
  return stream.str();
}

TEST(CasparReadbackPublicationControl, BothCalibrationRoutesWriteResults) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  for (const bool merged : {false, true}) {
    SCOPED_TRACE(merged);
    PublicationFixture fixture;
    ASSERT_NO_FATAL_FAILURE(PreparePublicationFixture(merged, &fixture));
    const Reconstruction before = fixture.reconstruction;
    auto adjuster = CreateDefaultCasparBundleAdjuster(
        fixture.options, fixture.config, fixture.reconstruction);
    ASSERT_THAT(fixture.reconstruction, ReconstructionEq(before));
    ReadbackTrace calls;
    calls.node_reads = &fixture.node_reads;
    std::shared_ptr<BundleAdjustmentSummary> summary;
    {
      ScopedReadbackFault scope(calls, ReadbackCall::kNone);
      ASSERT_NO_THROW(summary = adjuster->Solve());
    }
    ASSERT_NE(summary, nullptr);
    EXPECT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    EXPECT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::USER_FAILURE);
    const auto& trace = fixture.node_reads;
    RecordProperty(merged ? "merged_node_trace" : "split_node_trace",
                   DescribeNodeReadback(trace));
    ASSERT_FALSE(trace.mismatch);
    ASSERT_TRUE(trace.started);
    ASSERT_EQ(trace.count, trace.expected_count);
    ASSERT_GT(trace.scalar_copies, 0);
    EXPECT_FALSE(trace.fired);
    for (size_t i = 0; i < trace.count; ++i) {
      EXPECT_TRUE(trace.events[i].actual_copy);
      EXPECT_EQ(trace.events[i].status, cudaSuccess);
    }
    EXPECT_THAT(fixture.reconstruction, testing::Not(ReconstructionEq(before)));
    for (const auto& item : fixture.reconstruction.Cameras()) {
      for (double parameter : item.second.params)
        EXPECT_TRUE(std::isfinite(parameter));
    }
  }
}

class CasparReadbackPublication
    : public testing::TestWithParam<std::pair<bool, size_t>> {};

TEST_P(CasparReadbackPublication, FailedNodeReadDoesNotPublish) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  const auto [merged, fail_at] = GetParam();
  PublicationFixture fixture;
  ASSERT_NO_FATAL_FAILURE(PreparePublicationFixture(merged, &fixture));
  auto& trace = fixture.node_reads;
  ASSERT_LT(fail_at, trace.expected_count);
  trace.fail_at = fail_at;
  const Reconstruction before = fixture.reconstruction;
  auto adjuster = CreateDefaultCasparBundleAdjuster(
      fixture.options, fixture.config, fixture.reconstruction);
  ASSERT_THAT(fixture.reconstruction, ReconstructionEq(before));
  ReadbackTrace calls;
  calls.node_reads = &trace;
  bool threw = false;
  {
    ScopedReadbackFault scope(calls, ReadbackCall::kNone);
    try {
      adjuster->Solve();
    } catch (const std::runtime_error& error) {
      threw = true;
      EXPECT_NE(std::string(error.what()).find("cudaMemcpy device-to-host"),
                std::string::npos);
      EXPECT_NE(std::string(error.what())
                    .find(cudaGetErrorString(kInjectedReadbackError)),
                std::string::npos);
    }
  }
  RecordProperty("node_trace", DescribeNodeReadback(trace));
  EXPECT_TRUE(threw);
  ASSERT_FALSE(trace.mismatch);
  ASSERT_TRUE(trace.started);
  ASSERT_TRUE(trace.fired);
  ASSERT_EQ(trace.count, fail_at + 1);
  ASSERT_GT(trace.scalar_copies, 0);
  for (size_t i = 0; i < fail_at; ++i) {
    EXPECT_TRUE(trace.events[i].actual_copy);
    EXPECT_EQ(trace.events[i].status, cudaSuccess);
  }
  EXPECT_FALSE(trace.events[fail_at].actual_copy);
  EXPECT_EQ(trace.events[fail_at].status, kInjectedReadbackError);
  EXPECT_THAT(fixture.reconstruction, ReconstructionEq(before));
  EXPECT_EQ(fixture.reconstruction.RegImageIds(), before.RegImageIds());
}

INSTANTIATE_TEST_SUITE_P(NodeDownloads,
                         CasparReadbackPublication,
                         testing::Values(std::make_pair(true, size_t{0}),
                                         std::make_pair(true, size_t{2}),
                                         std::make_pair(true, size_t{4}),
                                         std::make_pair(false, size_t{0}),
                                         std::make_pair(false, size_t{2}),
                                         std::make_pair(false, size_t{3})));

void PrepareMultiModelPublicationFixture(bool merged,
                                         PublicationFixture* fixture) {
  SetPRNGSeed(0);
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 3;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 2;
  dataset.num_points3D = 64;
  dataset.num_points2D_without_point3D = 0;
  SynthesizeDataset(dataset, &fixture->reconstruction);
  auto& reconstruction = fixture->reconstruction;
  ASSERT_EQ(reconstruction.NumCameras(), 3);
  for (camera_t id : {camera_t{2}, camera_t{3}}) {
    auto& camera = reconstruction.Camera(id);
    ASSERT_EQ(camera.model_id, CameraModelId::kSimpleRadial);
    ASSERT_EQ(camera.params.size(), 4);
    const auto original = camera.params;
    camera.model_id = CameraModelId::kPinhole;
    camera.params = {original[0], original[0], original[1], original[2]};
    ASSERT_TRUE(camera.VerifyParams());
  }
  // Reproject the truth through the converted cameras before perturbing
  // intrinsics. Every retained observation remains geometrically consistent.
  for (const image_t id : reconstruction.RegImageIds()) {
    auto& image = reconstruction.Image(id);
    const auto& camera = *image.CameraPtr();
    for (point2D_t i = 0; i < image.NumPoints2D(); ++i) {
      auto& observation = image.Point2D(i);
      if (!observation.HasPoint3D()) continue;
      const Eigen::Vector3d point =
          image.CamFromWorld() *
          reconstruction.Point3D(observation.point3D_id).xyz;
      ASSERT_TRUE(point.allFinite());
      ASSERT_GT(point.z(), 0);
      const auto projected = camera.ImgFromCam(point);
      ASSERT_TRUE(projected.has_value());
      ASSERT_TRUE(projected->allFinite());
      ASSERT_GE(projected->x(), 0);
      ASSERT_LT(projected->x(), camera.width);
      ASSERT_GE(projected->y(), 0);
      ASSERT_LT(projected->y(), camera.height);
      observation.xy = *projected;
    }
  }
  std::set<point3D_t> points;
  std::set<frame_t> simple_frames, pinhole_frames;
  std::set<camera_t> simple_cameras, pinhole_cameras;
  for (const image_t id : reconstruction.RegImageIds()) {
    fixture->config.AddImage(id);
    const auto& image = reconstruction.Image(id);
    const auto model = image.CameraPtr()->model_id;
    if (model == CameraModelId::kSimpleRadial) {
      simple_frames.insert(image.FrameId());
      simple_cameras.insert(image.CameraId());
    } else {
      ASSERT_EQ(model, CameraModelId::kPinhole);
      pinhole_frames.insert(image.FrameId());
      pinhole_cameras.insert(image.CameraId());
    }
    ASSERT_FALSE(image.Points2D().empty());
    for (const auto& point : image.Points2D()) {
      if (point.HasPoint3D()) points.insert(point.point3D_id);
    }
  }
  ASSERT_EQ(simple_cameras.size(), 1);
  ASSERT_EQ(pinhole_cameras.size(), 2);
  ASSERT_EQ(simple_frames.size(), 2);
  ASSERT_EQ(pinhole_frames.size(), 4);
  for (const frame_t id : simple_frames) ASSERT_EQ(pinhole_frames.count(id), 0);
  ASSERT_GE(points.size(), 20);
  for (const camera_t id : simple_cameras) {
    auto& camera = reconstruction.Camera(id);
    camera.params[0] += 30;
    camera.params[3] += 0.03;
    if (merged) {
      camera.params[1] += 5;
      camera.params[2] -= 7;
    }
    ASSERT_TRUE(camera.VerifyParams());
  }
  for (const camera_t id : pinhole_cameras) {
    auto& camera = reconstruction.Camera(id);
    camera.params[0] += 30;
    camera.params[1] -= 25;
    if (merged) {
      camera.params[2] += 5;
      camera.params[3] -= 7;
    }
    ASSERT_TRUE(camera.VerifyParams());
  }
  auto& options = fixture->options;
  options.refine_rig_from_world = false;
  options.refine_sensor_from_rig = false;
  options.refine_points3D = false;
  options.refine_focal_length = true;
  options.refine_extra_params = true;
  options.refine_principal_point = merged;
  ASSERT_NE(options.caspar, nullptr);
  options.min_track_length = 0;
  options.caspar->solver_iter_max = 5;
  options.caspar->pcg_iter_max = 30;
  auto& trace = fixture->node_reads;
  trace.scalar_bytes = sizeof(StorageType);
  trace.multi_model = true;
  trace.merged = merged;
  trace.expected[0] = 3 * points.size() * trace.scalar_bytes;
  trace.simple_pose_bytes = 7 * simple_frames.size() * trace.scalar_bytes;
  trace.pinhole_pose_bytes = 7 * pinhole_frames.size() * trace.scalar_bytes;
  trace.expected_count = merged ? 9 : 7;
  ASSERT_NE(trace.simple_pose_bytes, trace.pinhole_pose_bytes);
  ASSERT_GT(trace.expected[0], trace.simple_pose_bytes);
  ASSERT_GT(trace.expected[0], trace.pinhole_pose_bytes);
  ASSERT_GT(trace.simple_pose_bytes, 2 * trace.scalar_bytes);
}

TEST(CasparReadbackPublicationControl,
     MultiModelCalibrationRoutesWriteBothModels) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  for (const bool merged : {false, true}) {
    SCOPED_TRACE(merged);
    PublicationFixture fixture;
    ASSERT_NO_FATAL_FAILURE(
        PrepareMultiModelPublicationFixture(merged, &fixture));
    const Reconstruction before = fixture.reconstruction;
    auto adjuster = CreateDefaultCasparBundleAdjuster(
        fixture.options, fixture.config, fixture.reconstruction);
    ASSERT_THAT(fixture.reconstruction, ReconstructionEq(before));
    ReadbackTrace calls;
    calls.node_reads = &fixture.node_reads;
    std::shared_ptr<BundleAdjustmentSummary> summary;
    {
      ScopedReadbackFault scope(calls, ReadbackCall::kNone);
      ASSERT_NO_THROW(summary = adjuster->Solve());
    }
    ASSERT_NE(summary, nullptr);
    EXPECT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::FAILURE);
    EXPECT_NE(summary->termination_type,
              BundleAdjustmentTerminationType::USER_FAILURE);
    const auto& trace = fixture.node_reads;
    RecordProperty(merged ? "multi_merged_trace" : "multi_split_trace",
                   DescribeNodeReadback(trace));
    ASSERT_FALSE(trace.mismatch);
    ASSERT_TRUE(trace.started);
    ASSERT_TRUE(trace.order_known);
    ASSERT_EQ(trace.count, trace.expected_count);
    ASSERT_GT(trace.scalar_copies, 0);
    EXPECT_FALSE(trace.fired);
    for (size_t i = 0; i < trace.count; ++i) {
      EXPECT_TRUE(trace.events[i].actual_copy);
      EXPECT_EQ(trace.events[i].status, cudaSuccess);
    }
    bool simple_changed = false, pinhole_changed = false;
    for (const auto& item : fixture.reconstruction.Cameras()) {
      const auto& camera = item.second;
      for (const double value : camera.params)
        EXPECT_TRUE(std::isfinite(value));
      const bool changed = camera.params != before.Camera(item.first).params;
      if (camera.model_id == CameraModelId::kSimpleRadial) {
        simple_changed |= changed;
      } else if (camera.model_id == CameraModelId::kPinhole) {
        pinhole_changed |= changed;
      }
    }
    EXPECT_TRUE(simple_changed);
    EXPECT_TRUE(pinhole_changed);
  }
}

class CasparReadbackPublicationMultiModel
    : public testing::TestWithParam<bool> {};

TEST_P(CasparReadbackPublicationMultiModel, FailedLaterModelDoesNotPublish) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  const bool merged = GetParam();
  PublicationFixture fixture;
  ASSERT_NO_FATAL_FAILURE(
      PrepareMultiModelPublicationFixture(merged, &fixture));
  auto& trace = fixture.node_reads;
  trace.fail_at = trace.expected_count - 1;
  const Reconstruction before = fixture.reconstruction;
  auto adjuster = CreateDefaultCasparBundleAdjuster(
      fixture.options, fixture.config, fixture.reconstruction);
  ASSERT_THAT(fixture.reconstruction, ReconstructionEq(before));
  ReadbackTrace calls;
  calls.node_reads = &trace;
  bool threw = false;
  {
    ScopedReadbackFault scope(calls, ReadbackCall::kNone);
    try {
      adjuster->Solve();
    } catch (const std::runtime_error& error) {
      threw = true;
      EXPECT_NE(std::string(error.what()).find("cudaMemcpy device-to-host"),
                std::string::npos);
      EXPECT_NE(std::string(error.what())
                    .find(cudaGetErrorString(kInjectedReadbackError)),
                std::string::npos);
    }
  }
  RecordProperty("multi_node_trace", DescribeNodeReadback(trace));
  EXPECT_TRUE(threw);
  ASSERT_FALSE(trace.mismatch);
  ASSERT_TRUE(trace.started);
  ASSERT_TRUE(trace.order_known);
  ASSERT_TRUE(trace.fired);
  ASSERT_EQ(trace.count, trace.expected_count);
  ASSERT_GT(trace.scalar_copies, 0);
  // The first model's complete calibration group and the later model's prior
  // reads must have returned successfully before its final download fails.
  for (size_t i = 0; i < trace.fail_at; ++i) {
    EXPECT_TRUE(trace.events[i].actual_copy);
    EXPECT_EQ(trace.events[i].status, cudaSuccess);
  }
  EXPECT_FALSE(trace.events[trace.fail_at].actual_copy);
  EXPECT_EQ(trace.events[trace.fail_at].status, kInjectedReadbackError);
  EXPECT_THAT(fixture.reconstruction, ReconstructionEq(before));
  EXPECT_EQ(fixture.reconstruction.RegImageIds(), before.RegImageIds());
}

INSTANTIATE_TEST_SUITE_P(BothRoutes,
                         CasparReadbackPublicationMultiModel,
                         testing::Bool());
#else
TEST(CasparReadbackFault, WrappingUnavailable) {
  GTEST_SKIP()
      << "requires Linux GNU static test link and verified --wrap interception";
}
#endif
}  // namespace
}  // namespace colmap
