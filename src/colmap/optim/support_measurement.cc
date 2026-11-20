// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/support_measurement.h"

#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

#include <algorithm>

namespace colmap {

InlierSupportMeasurer::Support InlierSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
  Support support;
  support.num_inliers = 0;
  support.residual_sum = 0;

  for (const auto residual : residuals) {
    if (residual <= max_residual) {
      support.num_inliers += 1;
      support.residual_sum += residual;
    }
  }

  return support;
}

bool InlierSupportMeasurer::IsLeftBetter(const Support& left,
                                         const Support& right) {
  if (left.num_inliers > right.num_inliers) {
    return true;
  } else {
    return left.num_inliers == right.num_inliers &&
           left.residual_sum < right.residual_sum;
  }
}

UniqueInlierSupportMeasurer::UniqueInlierSupportMeasurer(
    std::vector<size_t> unique_sample_ids)
    : unique_sample_ids_(std::move(unique_sample_ids)) {
  if (unique_sample_ids_.empty()) {
    return;
  }
  const size_t max_id =
      *std::max_element(unique_sample_ids_.begin(), unique_sample_ids_.end());
  if (max_id < unique_sample_ids_.size()) {
    // Existing labels already fit at most one scratch slot per sample.
    num_label_slots_ = max_id + 1;
    return;
  }
  FlatHashMap<size_t, size_t> dense_ids;
  for (size_t& id : unique_sample_ids_) {
    const auto [it, inserted] = dense_ids.try_emplace(id, dense_ids.size());
    id = it->second;
  }
  num_label_slots_ = dense_ids.size();
}

UniqueInlierSupportMeasurer::Support UniqueInlierSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
  THROW_CHECK_EQ(residuals.size(), unique_sample_ids_.size());
  Support support;
  support.num_inliers = 0;
  support.num_unique_inliers = 0;
  support.residual_sum = 0;

  const auto is_inlier = [max_residual](const double residual) {
    return residual <= max_residual;
  };
  auto residual_it =
      std::find_if(residuals.begin(), residuals.end(), is_inlier);
  if (residual_it == residuals.end()) {
    return support;
  }

  // Keep scratch local: independent Evaluate calls may share a measurer.
  std::vector<char> inlier_point_ids(num_label_slots_, false);
  for (; residual_it != residuals.end();
       residual_it =
           std::find_if(residual_it + 1, residuals.end(), is_inlier)) {
    const size_t idx = residual_it - residuals.begin();
    support.num_inliers += 1;
    const size_t id = unique_sample_ids_[idx];
    if (!inlier_point_ids[id]) {
      inlier_point_ids[id] = true;
      ++support.num_unique_inliers;
    }
    support.residual_sum += *residual_it;
  }
  return support;
}

bool UniqueInlierSupportMeasurer::IsLeftBetter(const Support& left,
                                               const Support& right) {
  if (left.num_unique_inliers > right.num_unique_inliers) {
    return true;
  } else if (left.num_unique_inliers == right.num_unique_inliers) {
    if (left.num_inliers > right.num_inliers) {
      return true;
    } else {
      return left.num_inliers == right.num_inliers &&
             left.residual_sum < right.residual_sum;
    }
  } else {
    return false;
  }
}

MEstimatorSupportMeasurer::Support MEstimatorSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
  Support support;
  support.num_inliers = 0;
  support.score = 0;

  for (const auto residual : residuals) {
    if (residual <= max_residual) {
      support.num_inliers += 1;
      support.score += residual;
    } else {
      support.score += max_residual;
    }
  }

  return support;
}

bool MEstimatorSupportMeasurer::IsLeftBetter(const Support& left,
                                             const Support& right) {
  return left.score < right.score;
}

}  // namespace colmap
