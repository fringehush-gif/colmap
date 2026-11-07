// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/support_measurement.h"

#include <cmath>
#include <future>
#include <random>
#include <set>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(InlierSupportMeasurer, Nominal) {
  InlierSupportMeasurer::Support support1;
  EXPECT_EQ(support1.num_inliers, 0);
  EXPECT_EQ(support1.residual_sum, std::numeric_limits<double>::max());
  InlierSupportMeasurer measurer;
  std::vector<double> residuals = {-1.0, 0.0, 1.0, 2.0};
  support1 = measurer.Evaluate(residuals, 1.0);
  EXPECT_EQ(support1.num_inliers, 3);
  EXPECT_EQ(support1.residual_sum, 0.0);
  InlierSupportMeasurer::Support support2;
  support2.num_inliers = 2;
  EXPECT_TRUE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.residual_sum = support1.residual_sum;
  EXPECT_TRUE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.num_inliers = support1.num_inliers;
  support2.residual_sum += 0.01;
  EXPECT_TRUE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.residual_sum -= 0.01;
  EXPECT_FALSE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.residual_sum -= 0.01;
  EXPECT_FALSE(measurer.IsLeftBetter(support1, support2));
  EXPECT_TRUE(measurer.IsLeftBetter(support2, support1));
}

TEST(UniqueInlierSupportMeasurer, Nominal) {
  UniqueInlierSupportMeasurer::Support support1;
  EXPECT_EQ(support1.num_inliers, 0);
  EXPECT_EQ(support1.num_unique_inliers, 0);
  EXPECT_EQ(support1.residual_sum, std::numeric_limits<double>::max());

  UniqueInlierSupportMeasurer measurer({1, 2, 2, 3});
  const std::vector<double> residuals = {-1.0, 0.0, 1.0, 2.0};
  support1 = measurer.Evaluate(residuals, 1.0);
  EXPECT_EQ(support1.num_inliers, 3);
  EXPECT_EQ(support1.num_unique_inliers, 2);
  EXPECT_EQ(support1.residual_sum, 0.0);

  UniqueInlierSupportMeasurer::Support support2;
  support2.num_unique_inliers = support1.num_unique_inliers - 1;
  EXPECT_TRUE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.num_inliers = support1.num_inliers + 1;
  EXPECT_TRUE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.num_inliers = support1.num_inliers;
  EXPECT_TRUE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.residual_sum = support1.residual_sum - 0.01;
  EXPECT_TRUE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.residual_sum = support1.residual_sum;
  EXPECT_TRUE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.num_unique_inliers = support1.num_unique_inliers;
  EXPECT_FALSE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.residual_sum = support1.residual_sum - 0.01;
  EXPECT_FALSE(measurer.IsLeftBetter(support1, support2));
  EXPECT_TRUE(measurer.IsLeftBetter(support2, support1));
  support2.num_inliers = support1.num_inliers + 1;
  support2.residual_sum = support1.residual_sum + 0.01;
  EXPECT_FALSE(measurer.IsLeftBetter(support1, support2));
  EXPECT_TRUE(measurer.IsLeftBetter(support2, support1));
  support2.num_unique_inliers = support1.num_unique_inliers + 1;
  support2.num_inliers = support1.num_inliers - 1;
  support2.residual_sum = support1.residual_sum + 0.01;
  EXPECT_FALSE(measurer.IsLeftBetter(support1, support2));
  EXPECT_TRUE(measurer.IsLeftBetter(support2, support1));
}

TEST(UniqueInlierSupportMeasurer, SparseIdsAndIndependentEvaluations) {
  const size_t max_id = std::numeric_limits<size_t>::max();
  UniqueInlierSupportMeasurer measurer({max_id, 7, max_id, 42});
  const auto all = measurer.Evaluate({0, 0, 0, 0}, 1);
  EXPECT_EQ(all.num_unique_inliers, 3);
  EXPECT_EQ(all.num_inliers, 4);
  const auto none = measurer.Evaluate({2, 2, 2, 2}, 1);
  EXPECT_EQ(none.num_unique_inliers, 0);
  EXPECT_EQ(none.num_inliers, 0);
  EXPECT_ANY_THROW(measurer.Evaluate({0}, 1));
  const auto duplicate = measurer.Evaluate({0, 2, 0, 2}, 1);
  EXPECT_EQ(duplicate.num_unique_inliers, 1);
  EXPECT_EQ(duplicate.num_inliers, 2);
  const auto repeated = measurer.Evaluate({0, 0, 0, 0}, 1);
  EXPECT_EQ(repeated.num_unique_inliers, all.num_unique_inliers);
  EXPECT_EQ(repeated.num_inliers, all.num_inliers);
  EXPECT_EQ(repeated.residual_sum, all.residual_sum);
  // Labels must not reorder residual accumulation by equivalence class.
  const auto ordered = measurer.Evaluate(
      {1e16, 1, -1e16, 2}, std::numeric_limits<double>::infinity());
  EXPECT_EQ(ordered.residual_sum, 2);
}

TEST(UniqueInlierSupportMeasurer, MatchesOrderedSetOracle) {
  std::mt19937 rng(19);
  const double infinity = std::numeric_limits<double>::infinity();
  for (const size_t num_samples :
       {size_t{0}, size_t{1}, size_t{8}, size_t{257}}) {
    std::vector<size_t> ids(num_samples);
    for (size_t& id : ids) {
      id = std::numeric_limits<size_t>::max() - (rng() % 17);
    }
    UniqueInlierSupportMeasurer measurer(ids);
    for (const double threshold : {-infinity, -1.0, 0.0, 1.0, infinity}) {
      for (size_t trial = 0; trial < 8; ++trial) {
        std::vector<double> residuals(num_samples);
        for (double& residual : residuals) {
          residual = static_cast<int>(rng() % 9) - 4;
        }
        if (num_samples > 3) {
          residuals[0] = std::numeric_limits<double>::quiet_NaN();
          residuals[1] = infinity;
          residuals[2] = -infinity;
        }
        std::set<size_t> seen;
        size_t num_inliers = 0;
        double residual_sum = 0;
        for (size_t i = 0; i < num_samples; ++i) {
          if (residuals[i] <= threshold) {
            seen.insert(ids[i]);
            ++num_inliers;
            residual_sum += residuals[i];
          }
        }
        const auto actual = measurer.Evaluate(residuals, threshold);
        EXPECT_EQ(actual.num_unique_inliers, seen.size());
        EXPECT_EQ(actual.num_inliers, num_inliers);
        if (std::isnan(residual_sum)) {
          EXPECT_TRUE(std::isnan(actual.residual_sum));
        } else {
          EXPECT_EQ(actual.residual_sum, residual_sum);
        }
      }
    }
  }
}

TEST(UniqueInlierSupportMeasurer, ConcurrentEvaluationsAreIndependent) {
  UniqueInlierSupportMeasurer measurer({0, 0, 3, 3});
  std::vector<std::future<void>> workers;
  for (size_t i = 0; i < 4; ++i) {
    workers.push_back(std::async(std::launch::async, [&measurer, i]() {
      const std::vector<double> residuals =
          i % 2 == 0 ? std::vector<double>{0, 0, 2, 2}
                     : std::vector<double>{0, 2, 0, 2};
      for (size_t trial = 0; trial < 100; ++trial) {
        const auto result = measurer.Evaluate(residuals, 1);
        EXPECT_EQ(result.num_inliers, 2);
        EXPECT_EQ(result.num_unique_inliers, i % 2 == 0 ? 1 : 2);
        EXPECT_EQ(result.residual_sum, 0);
      }
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }
}

TEST(UniqueInlierSupportMeasurer, CopyAndMovePreserveLabels) {
  for (const auto& ids :
       {std::vector<size_t>{std::numeric_limits<size_t>::max(), 7, 7, 42},
        std::vector<size_t>{0, 0, 0, 3}}) {
    UniqueInlierSupportMeasurer original(ids);
    auto check = [&ids](UniqueInlierSupportMeasurer& measurer) {
      const auto result = measurer.Evaluate({0, 0, 0, 0}, 1);
      EXPECT_EQ(result.num_unique_inliers,
                std::set<size_t>(ids.begin(), ids.end()).size());
      EXPECT_EQ(result.num_inliers, 4);
      EXPECT_EQ(result.residual_sum, 0);
    };
    auto copied = original;
    check(copied);
    auto moved = std::move(copied);
    check(moved);
    UniqueInlierSupportMeasurer assigned({1});
    assigned = original;
    check(assigned);
    UniqueInlierSupportMeasurer move_assigned({1, 1});
    move_assigned = std::move(assigned);
    check(move_assigned);
    check(original);
  }
}

TEST(UniqueInlierSupportMeasurer, BoundedLabelsWithGaps) {
  for (const auto& ids : {std::vector<size_t>{},
                          std::vector<size_t>{0},
                          std::vector<size_t>{3, 3, 3, 3},
                          std::vector<size_t>{0, 0, 2, 2},
                          std::vector<size_t>{0, 1, 3, 3}}) {
    UniqueInlierSupportMeasurer measurer(ids);
    const std::vector<double> residuals(ids.size(), 0);
    const auto all = measurer.Evaluate(residuals, 0);
    EXPECT_EQ(all.num_inliers, ids.size());
    EXPECT_EQ(all.num_unique_inliers,
              std::set<size_t>(ids.begin(), ids.end()).size());
    const auto none = measurer.Evaluate(residuals, -1);
    EXPECT_EQ(none.num_inliers, 0);
    EXPECT_EQ(none.num_unique_inliers, 0);
    EXPECT_EQ(none.residual_sum, 0);
    if (ids.empty()) {
      continue;
    }
    for (const size_t first_inlier :
         {size_t{0}, ids.size() / 2, ids.size() - 1}) {
      std::vector<double> sparse_residuals(
          ids.size(), std::numeric_limits<double>::quiet_NaN());
      sparse_residuals[first_inlier] = 0.25;
      const auto sparse = measurer.Evaluate(sparse_residuals, 1);
      EXPECT_EQ(sparse.num_inliers, 1);
      EXPECT_EQ(sparse.num_unique_inliers, 1);
      EXPECT_EQ(sparse.residual_sum, 0.25);
      const auto unordered = measurer.Evaluate(
          sparse_residuals, std::numeric_limits<double>::quiet_NaN());
      EXPECT_EQ(unordered.num_inliers, 0);
      EXPECT_EQ(unordered.num_unique_inliers, 0);
      EXPECT_EQ(unordered.residual_sum, 0);
    }
  }
}

TEST(MEstimatorSupportMeasurer, Nominal) {
  MEstimatorSupportMeasurer::Support support1;
  EXPECT_EQ(support1.num_inliers, 0);
  EXPECT_EQ(support1.score, std::numeric_limits<double>::max());
  MEstimatorSupportMeasurer measurer;
  std::vector<double> residuals = {-1.0, 0.0, 1.0, 2.0};
  support1 = measurer.Evaluate(residuals, 1.0);
  EXPECT_EQ(support1.num_inliers, 3);
  EXPECT_EQ(support1.score, 1.0);
  MEstimatorSupportMeasurer::Support support2 = support1;
  EXPECT_FALSE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.num_inliers -= 1;
  support2.score += 0.01;
  EXPECT_TRUE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.score -= 0.01;
  EXPECT_FALSE(measurer.IsLeftBetter(support1, support2));
  EXPECT_FALSE(measurer.IsLeftBetter(support2, support1));
  support2.score -= 0.01;
  EXPECT_FALSE(measurer.IsLeftBetter(support1, support2));
  EXPECT_TRUE(measurer.IsLeftBetter(support2, support1));
}

}  // namespace
}  // namespace colmap
