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

void StagePatchMatchInputs(Workspace& workspace,
                           ThreadPool& io_thread_pool,
                           const std::vector<int>& image_idxs,
                           const bool geom_consistency,
                           std::vector<Image>& images,
                           std::vector<DepthMap>& depth_maps,
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
          depth_maps.at(image_idx) = *workspace.GetDepthMapLease(image_idx);
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
