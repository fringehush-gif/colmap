// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/loransac.h"

#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_alignment.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(OptimisticMSAC, PreservesStrictScoreTies) {
  const std::vector<double> residuals{1.0, 4.0, 9.0};
  const double score = 9.0;
  EXPECT_TRUE(internal::CannotImproveOptimisticMSAC(residuals, 4.0, score));
  EXPECT_TRUE(internal::CannotImproveOptimisticMSAC(
      residuals, 4.0, std::nextafter(score, 0.0)));
  EXPECT_FALSE(internal::CannotImproveOptimisticMSAC(
      residuals, 4.0, std::nextafter(score, 10.0)));
}

TEST(OptimisticMSAC, InvalidBoundsRequireFullScoring) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double max = std::numeric_limits<double>::max();
  for (const double invalid : {-1.0, inf, -inf, nan}) {
    EXPECT_FALSE(internal::CannotImproveOptimisticMSAC({invalid}, 4.0, 1.0));
    EXPECT_FALSE(internal::CannotImproveOptimisticMSAC({1.0}, invalid, 1.0));
  }
  for (const double invalid : {inf, -inf, nan, max}) {
    EXPECT_FALSE(internal::CannotImproveOptimisticMSAC({1.0}, 4.0, invalid));
  }
  EXPECT_FALSE(internal::CannotImproveOptimisticMSAC(
      {max / 2.0, max / 2.0, max / 2.0}, max, 1.0));
}

TEST(OptimisticMSAC, RejectedBoundCannotBecomeBetterAfterCheirality) {
  MEstimatorSupportMeasurer measurer;
  const std::vector<double> lower{0.0, 1.0, 4.0};
  for (const double threshold : {0.0, 1.0, 4.0}) {
    const auto bound = measurer.Evaluate(lower, threshold);
    for (const double incumbent : {0.0, bound.score, bound.score + 1.0}) {
      MEstimatorSupportMeasurer::Support best;
      best.score = incumbent;
      for (int mask = 0; mask < 8; ++mask) {
        auto full = lower;
        for (size_t i = 0; i < full.size(); ++i) {
          if (mask & (1 << i)) full[i] = std::numeric_limits<double>::max();
        }
        if (internal::CannotImproveOptimisticMSAC(
                lower, threshold, incumbent)) {
          EXPECT_FALSE(
              measurer.IsLeftBetter(measurer.Evaluate(full, threshold), best));
        }
      }
    }
  }
}

struct SimilarityTransformTestData {
  Sim3d expected_tgt_from_src;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> tgt;
  size_t num_samples;
  size_t num_outliers;
};

SimilarityTransformTestData GenerateTestData(const size_t num_samples = 1000,
                                             const size_t num_outliers = 400) {
  SimilarityTransformTestData data;
  data.num_samples = num_samples;
  data.num_outliers = num_outliers;

  data.expected_tgt_from_src =
      Sim3d(2, RandomEigenQuaterniond(), Eigen::Vector3d(100, 10, 10));

  for (size_t i = 0; i < num_samples; ++i) {
    data.src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    data.tgt.push_back(data.expected_tgt_from_src * data.src.back());
  }

  for (size_t i = 0; i < num_outliers; ++i) {
    data.tgt[i] = Eigen::Vector3d(RandomUniformReal(-3000.0, -2000.0),
                                  RandomUniformReal(-4000.0, -3000.0),
                                  RandomUniformReal(-5000.0, -4000.0));
  }

  return data;
}

class ShrinkingInlierEstimator {
 public:
  using X_t = double;
  using Y_t = double;
  using M_t = int;

  static const int kMinNumSamples = 5;

  static void Estimate(const std::vector<X_t>& X,
                       const std::vector<Y_t>& Y,
                       std::vector<M_t>* models) {
    THROW_CHECK_EQ(X.size(), Y.size());
    THROW_CHECK_GE(X.size(), kMinNumSamples);
    models->assign(1, 0);
  }

  static bool Refine(const std::vector<X_t>& X,
                     const std::vector<Y_t>& Y,
                     M_t* model) {
    THROW_CHECK_EQ(X.size(), Y.size());
    THROW_CHECK_GE(X.size(), kMinNumSamples);
    *model = 1;
    return true;
  }

  static void Residuals(const std::vector<X_t>& X,
                        const std::vector<Y_t>& Y,
                        const M_t model,
                        std::vector<double>* residuals) {
    THROW_CHECK_EQ(X.size(), Y.size());
    residuals->assign(X.size(), 0.9);
    if (model == 1) {
      residuals->assign(X.size(), 2.0);
      std::fill_n(residuals->begin(), 4, 0.0);
    }
  }
};

// Each model appears in every trial, independent of the sampler's subset.
// The shared counters remain valid across LORANSAC's per-thread estimator
// copies.
struct ScoringCalls {
  std::atomic<int> estimates{0};
  std::array<std::atomic<int>, 4> optimistic{};
  std::array<std::atomic<int>, 4> exact{};

  ScoringCalls() {
    for (auto& count : optimistic) count.store(0, std::memory_order_relaxed);
    for (auto& count : exact) count.store(0, std::memory_order_relaxed);
  }
};

class InstrumentedOptimisticEstimator {
 public:
  using X_t = double;
  using Y_t = double;
  using M_t = int;
  static const int kMinNumSamples = 2;

  InstrumentedOptimisticEstimator(bool enabled,
                                  std::shared_ptr<ScoringCalls> calls)
      : enabled_(enabled), calls_(std::move(calls)) {}

  bool UseOptimisticScoring() const { return enabled_; }

  void Estimate(const std::vector<X_t>&,
                const std::vector<Y_t>&,
                std::vector<M_t>* models) const {
    calls_->estimates.fetch_add(1, std::memory_order_relaxed);
    *models = {0, 1, 2, 3};
  }

  void OptimisticResiduals(const std::vector<X_t>& X,
                           const std::vector<Y_t>&,
                           M_t model,
                           std::vector<double>* residuals) const {
    calls_->optimistic.at(model).fetch_add(1, std::memory_order_relaxed);
    if (model == 3) {
      residuals->assign(X.size(), std::numeric_limits<double>::quiet_NaN());
    } else {
      // Model 1 ties the winner's lower bound; model 2 remains competitive
      // optimistically but loses after exact scoring.
      residuals->assign(X.size(), model == 2 ? 0.1 : 0.2);
    }
  }

  void Residuals(const std::vector<X_t>& X,
                 const std::vector<Y_t>&,
                 M_t model,
                 std::vector<double>* residuals) const {
    calls_->exact.at(model).fetch_add(1, std::memory_order_relaxed);
    residuals->assign(X.size(), 0.2 + 0.1 * model);
  }

 private:
  bool enabled_;
  std::shared_ptr<ScoringCalls> calls_;
};

class NoLocalScoringEstimator {
 public:
  using X_t = double;
  using Y_t = double;
  using M_t = int;
  static const int kMinNumSamples = 100;

  static void Estimate(const std::vector<X_t>&,
                       const std::vector<Y_t>&,
                       std::vector<M_t>* models) {
    models->clear();
    ADD_FAILURE() << "Local refinement should not run in this fixture";
  }

  static void Residuals(const std::vector<X_t>&,
                        const std::vector<Y_t>&,
                        M_t,
                        std::vector<double>*) {
    ADD_FAILURE() << "Local scoring should not run in this fixture";
  }
};

template <typename Report>
void ValidateReport(const Report& report,
                    const SimilarityTransformTestData& data) {
  EXPECT_TRUE(report.success);
  EXPECT_GT(report.num_trials, 0);

  EXPECT_EQ(report.support.num_inliers, data.num_samples - data.num_outliers);
  for (size_t i = 0; i < data.num_samples; ++i) {
    if (i < data.num_outliers) {
      EXPECT_FALSE(report.inlier_mask[i]);
    } else {
      EXPECT_TRUE(report.inlier_mask[i]);
    }
  }

  const double matrix_diff =
      (data.expected_tgt_from_src.ToMatrix() - report.model).norm();
  EXPECT_LT(matrix_diff, 1e-6);
}

TEST(LORANSAC, Report) {
  LORANSAC<SimilarityTransformEstimator<3>,
           SimilarityTransformEstimator<3>>::Report report;
  EXPECT_FALSE(report.success);
  EXPECT_EQ(report.num_trials, 0);
  EXPECT_EQ(report.support.num_inliers, 0);
  EXPECT_EQ(report.support.residual_sum, std::numeric_limits<double>::max());
  EXPECT_EQ(report.inlier_mask.size(), 0);
}

TEST(LORANSAC, SimilarityTransform) {
  const auto data = GenerateTestData();

  RANSACOptions options;
  options.max_error = 10;
  options.random_seed = kDefaultPRNGSeed;
  LORANSAC<SimilarityTransformEstimator<3>, SimilarityTransformEstimator<3>>
      loransac(options);
  const auto report = loransac.Estimate(data.src, data.tgt);

  ValidateReport(report, data);
}

TEST(LORANSAC, ParallelSimilarityTransform) {
  const auto data = GenerateTestData();

  RANSACOptions options;
  options.max_error = 10;
  options.random_seed = kDefaultPRNGSeed;
  options.num_threads = 4;
  LORANSAC<SimilarityTransformEstimator<3>, SimilarityTransformEstimator<3>>
      loransac(options);
  const auto report = loransac.Estimate(data.src, data.tgt);

  ValidateReport(report, data);
}

TEST(LORANSAC, RecursiveRefinementStopsBelowMinimumSampleCount) {
  RANSACOptions options;
  options.max_error = 1.0;
  options.min_num_trials = 1;
  options.max_num_trials = 1;
  options.random_seed = kDefaultPRNGSeed;

  LORANSAC<ShrinkingInlierEstimator,
           ShrinkingInlierEstimator,
           MEstimatorSupportMeasurer>
      loransac(options);
  const std::vector<double> samples(6, 0.0);

  LORANSAC<ShrinkingInlierEstimator,
           ShrinkingInlierEstimator,
           MEstimatorSupportMeasurer>::Report report;
  EXPECT_NO_THROW(report = loransac.Estimate(samples, samples));
  EXPECT_FALSE(report.success);
  EXPECT_EQ(report.support.num_inliers, 4);
}

TEST(LORANSAC, CannotImproveOptimisticMSACConservativeDecisions) {
  using internal::CannotImproveOptimisticMSAC;
  const std::vector<double> residuals{0.25, 0.25};
  EXPECT_TRUE(CannotImproveOptimisticMSAC(residuals, 1.0, 0.5));
  EXPECT_TRUE(CannotImproveOptimisticMSAC(residuals, 1.0, 0.49));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(residuals, 1.0, 0.51));

  // Scores cap each outlier at the squared threshold, in native order.
  EXPECT_TRUE(CannotImproveOptimisticMSAC({0.25, 4.0}, 1.0, 1.25));
  EXPECT_FALSE(CannotImproveOptimisticMSAC({0.25, 4.0}, 1.0, 1.26));

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(CannotImproveOptimisticMSAC({nan}, 1.0, 1.0));
  EXPECT_FALSE(CannotImproveOptimisticMSAC({inf}, 1.0, 1.0));
  EXPECT_FALSE(CannotImproveOptimisticMSAC({-inf}, 1.0, 1.0));
  EXPECT_FALSE(CannotImproveOptimisticMSAC({-0.1}, 1.0, 1.0));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(residuals, nan, 0.5));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(residuals, inf, 0.5));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(residuals, -inf, 0.5));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(residuals, -1.0, 0.5));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(residuals, 1.0, nan));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(residuals, 1.0, inf));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(residuals, 1.0, -inf));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(residuals, 1.0, -1.0));
  EXPECT_FALSE(CannotImproveOptimisticMSAC(
      residuals, 1.0, std::numeric_limits<double>::max()));
  // A finite per-residual input can still overflow the accumulated score.
  const double huge = std::numeric_limits<double>::max();
  EXPECT_FALSE(CannotImproveOptimisticMSAC({huge, huge, huge}, huge / 2, 1.0));
}

TEST(LORANSAC, OptimisticMSACPrunesOnlyModelsThatCannotWin) {
  const std::vector<double> samples(6, 0.0);
  for (const int num_threads : {1, 2}) {
    double exact_score = 0.0;
    for (const bool enabled : {false, true}) {
      auto calls = std::make_shared<ScoringCalls>();
      RANSACOptions options;
      options.max_error = 1.0;
      options.min_num_trials = 3;
      options.max_num_trials = 3;
      options.random_seed = kDefaultPRNGSeed;
      options.num_threads = num_threads;
      LORANSAC<InstrumentedOptimisticEstimator,
               NoLocalScoringEstimator,
               MEstimatorSupportMeasurer>
          loransac(options,
                   InstrumentedOptimisticEstimator(enabled, calls),
                   NoLocalScoringEstimator());
      const auto report = loransac.Estimate(samples, samples);
      ASSERT_TRUE(report.success);
      EXPECT_EQ(report.model, 0);
      EXPECT_EQ(report.support.num_inliers, samples.size());
      EXPECT_TRUE(std::all_of(report.inlier_mask.begin(),
                              report.inlier_mask.end(),
                              [](const char inlier) { return inlier; }));
      if (enabled) {
        EXPECT_DOUBLE_EQ(report.support.score, exact_score);
      } else {
        exact_score = report.support.score;
      }

      const int trials = calls->estimates.load(std::memory_order_relaxed);
      EXPECT_EQ(trials, 3);
      EXPECT_EQ(calls->exact[1].load(), enabled ? 0 : trials);
      EXPECT_EQ(calls->exact[2].load(), trials);
      EXPECT_EQ(calls->exact[3].load(), trials);
      EXPECT_EQ(calls->optimistic[1].load(), enabled ? trials : 0);
      EXPECT_EQ(calls->optimistic[2].load(), enabled ? trials : 0);
      EXPECT_EQ(calls->optimistic[3].load(), enabled ? trials : 0);
    }
  }
}

}  // namespace
}  // namespace colmap
