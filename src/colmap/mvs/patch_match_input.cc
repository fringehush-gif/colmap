// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_input.h"

#include "colmap/mvs/image.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/threading.h"

#include <exception>
#include <future>

namespace colmap {
namespace mvs {
namespace internal {

void StagePatchMatchInputs(
    Workspace& workspace,
    ThreadPool& io_thread_pool,
    const std::vector<int>& image_idxs,
    const bool geom_consistency,
    const int ref_image_idx,
    std::vector<Image>& images,
    std::vector<DepthMap>& depth_maps,
    std::vector<std::shared_ptr<const DepthMap>>& source_depth_leases,
    std::vector<NormalMap>& normal_maps) {
  // Reserve before submitting so growth cannot lose an accepted task.
  std::vector<std::shared_future<void>> futures;
  futures.reserve(image_idxs.size());
  std::exception_ptr error;
  try {
    for (const int image_idx : image_idxs) {
      futures.push_back(io_thread_pool.AddTask([&, image_idx]() {
        images.at(image_idx).SetBitmap(*workspace.GetBitmapLease(image_idx));
        if (geom_consistency) {
          const auto depth_lease = workspace.GetDepthMapLease(image_idx);
          auto& source_lease = source_depth_leases.at(image_idx);
          // Workspace's borrowed leases can be non-null without owning data.
          // Only retain a cache owner; mutable reference seeds remain copies.
          if (image_idx != ref_image_idx && depth_lease.use_count() > 0 &&
              depth_lease->GetWidth() == images.at(image_idx).GetWidth() &&
              depth_lease->GetHeight() == images.at(image_idx).GetHeight()) {
            source_lease = depth_lease;
            depth_maps.at(image_idx) = DepthMap();
          } else {
            source_lease.reset();
            depth_maps.at(image_idx) = *depth_lease;
          }
          normal_maps.at(image_idx) = *workspace.GetNormalMapLease(image_idx);
        }
      }));
    }
    for (auto& future : futures) {
      future.get();
    }
  } catch (...) {
    error = std::current_exception();
  }

  // Wait also consumes the pool's retained task errors. Preserve the first
  // submission/read error if draining observes additional failures.
  try {
    io_thread_pool.Wait();
  } catch (...) {
    if (!error) {
      error = std::current_exception();
    }
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
