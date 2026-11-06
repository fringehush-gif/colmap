// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Eigen/Core>
#include <Eigen/QR>

namespace colmap::internal {

// Three non-collinear points fix a similarity gauge. Test their affine span,
// which is independent of the world-coordinate origin. This only selects the
// geometry of the anchors; callers determine which points are fixed or
// variable.
struct FixedGaugeWithThreePoints {
  Eigen::Index num_fixed_points = 0;
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 3, 2> directions = Eigen::Matrix<double, 3, 2>::Zero();

  bool MaybeAddFixedPoint(const Eigen::Vector3d& point) {
    if (num_fixed_points >= 3) {
      return false;
    }
    if (num_fixed_points == 0) {
      origin = point;
      ++num_fixed_points;
      return true;
    }
    directions.col(num_fixed_points - 1) = point - origin;
    // Use Eigen's default relative rank threshold rather than an absolute
    // world-unit tolerance.
    if (directions.colPivHouseholderQr().rank() == num_fixed_points) {
      ++num_fixed_points;
      return true;
    }
    directions.col(num_fixed_points - 1).setZero();
    return false;
  }
};

}  // namespace colmap::internal
