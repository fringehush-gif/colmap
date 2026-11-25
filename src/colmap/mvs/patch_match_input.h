// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

namespace colmap {
class ThreadPool;
namespace mvs {
class Workspace;
class Image;
class DepthMap;
class NormalMap;
namespace internal {

// The caller retains its workspace lock and destination vectors until this
// returns or throws. Every accepted input read finishes before either occurs.
// Image indices must be unique and valid for the destination vectors.
// Geometric staging reads each requested depth map and the reference normal.
void StagePatchMatchInputs(Workspace& workspace,
                           ThreadPool& io_thread_pool,
                           const std::vector<int>& image_idxs,
                           const int ref_image_idx,
                           bool geom_consistency,
                           std::vector<Image>& images,
                           std::vector<DepthMap>& depth_maps,
                           std::vector<NormalMap>& normal_maps);

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
