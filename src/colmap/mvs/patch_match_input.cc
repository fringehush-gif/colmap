// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_input.h"

#include "colmap/mvs/image.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/threading.h"

#include <exception>
#include <future>
#include <utility>

namespace colmap {
namespace mvs {
namespace internal {

#ifdef COLMAP_MVS_INPUT_TESTING
namespace {
InputCopyHook copy_hook;
InputSubmissionHook submission_hook;
}  // namespace

void SetInputCopyHookForTesting(InputCopyHook hook) {
  copy_hook = std::move(hook);
}
void SetInputSubmissionHookForTesting(InputSubmissionHook hook) {
  submission_hook = std::move(hook);
}
void NotifyInputCopyForTesting(const int image_idx,
                               const InputPayload payload) {
  if (copy_hook) copy_hook(image_idx, payload);
}
#endif

void StagePatchMatchInputs(Workspace& workspace,
                           ThreadPool& io_thread_pool,
                           const std::vector<int>& image_idxs,
                           const bool geom_consistency,
                           const int ref_image_idx,
                           std::vector<Image>& images,
                           std::vector<DepthMap>& depth_maps,
                           std::vector<NormalMap>& normal_maps) {
  // Allocate before submission so vector growth cannot lose an accepted task.
  std::vector<std::shared_future<void>> futures;
  futures.reserve(image_idxs.size());
  std::exception_ptr error;
  try {
    for (const int image_idx : image_idxs) {
      futures.push_back(io_thread_pool.AddTask([&, image_idx]() {
        images.at(image_idx).SetBitmap(workspace.CopyBitmap(image_idx));
        if (geom_consistency) {
          depth_maps.at(image_idx) = workspace.CopyDepthMap(image_idx);
          if (image_idx == ref_image_idx) {
            normal_maps.at(image_idx) = workspace.CopyNormalMap(image_idx);
          } else {
            // Source normals remain an input dependency and are still loaded.
            workspace.EnsureNormalLoaded(image_idx);
          }
        }
      }));
#ifdef COLMAP_MVS_INPUT_TESTING
      if (submission_hook) submission_hook(futures.size());
#endif
    }
    for (auto& future : futures) future.get();
  } catch (...) {
    error = std::current_exception();
#ifdef COLMAP_MVS_INPUT_TESTING
    // Zero signals that the original error is captured and drain is next.
    try {
      if (submission_hook) submission_hook(0);
    } catch (...) {
      // The original input error takes priority over this test-only failure.
      // Continue to drain accepted tasks before rethrowing that error.
    }
#endif
  }

  // Wait also consumes the pool's retained error checkers, including when a
  // future was already observed. Preserve the first error observed above.
  try {
    io_thread_pool.Wait();
  } catch (...) {
    if (!error) error = std::current_exception();
  }
  if (error) std::rethrow_exception(error);
}

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
