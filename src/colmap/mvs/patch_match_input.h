// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
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
// Owning source-depth leases are retained when their dimensions match the
// images. Reference maps and borrowed leases keep the independent copy path.
void StagePatchMatchInputs(
    Workspace& workspace,
    ThreadPool& io_thread_pool,
    const std::vector<int>& image_idxs,
    bool geom_consistency,
    int ref_image_idx,
    std::vector<Image>& images,
    std::vector<DepthMap>& depth_maps,
    std::vector<std::shared_ptr<const DepthMap>>& source_depth_leases,
    std::vector<NormalMap>& normal_maps);

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
