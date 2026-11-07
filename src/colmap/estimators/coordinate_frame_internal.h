// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/image/line.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Geometry>

namespace colmap {

struct VanishingPointEstimator {
  // The line segments.
  using X_t = LineSegment;
  // The line representation of the segments.
  using Y_t = Eigen::Vector3d;
  // The vanishing point.
  using M_t = Eigen::Vector3d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 2;

  // Estimate the vanishing point from at least two line segments.
  static void Estimate(const std::vector<X_t>& line_segments,
                       const std::vector<Y_t>& lines,
                       std::vector<M_t>* models) {
    THROW_CHECK_EQ(line_segments.size(), 2);
    THROW_CHECK_EQ(lines.size(), 2);
    THROW_CHECK(models != nullptr);
    models->resize(1);
    (*models)[0] = lines[0].cross(lines[1]);
  }

  // Calculate the squared distance of each line segment's end point to the line
  // connecting the vanishing point and the midpoint of the line segment.
  static void Residuals(const std::vector<X_t>& line_segments,
                        const std::vector<Y_t>& lines,
                        const M_t& vanishing_point,
                        std::vector<double>* residuals) {
    residuals->resize(line_segments.size());

    // Homogeneous points at infinity are valid. Only invalid or zero
    // homogeneous vectors fail to define a vanishing point.
    if (!vanishing_point.allFinite() ||
        (vanishing_point.array() == 0).all()) {
      std::fill(residuals->begin(),
                residuals->end(),
                std::numeric_limits<double>::max());
      return;
    }

    for (size_t i = 0; i < lines.size(); ++i) {
      const Eigen::Vector3d midpoint =
          (0.5 * (line_segments[i].start + line_segments[i].end)).homogeneous();
      const Eigen::Vector3d connecting_line = midpoint.cross(vanishing_point);
      const double line_norm = connecting_line.head<2>().norm();
      if (!(line_norm > 0) || !std::isfinite(line_norm)) {
        (*residuals)[i] = std::numeric_limits<double>::max();
        continue;
      }
      const double signed_distance =
          connecting_line.dot(line_segments[i].end.homogeneous()) / line_norm;
      (*residuals)[i] = signed_distance * signed_distance;
    }
  }
};

}  // namespace colmap
