// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <vector>

#ifdef COLMAP_MVS_INPUT_TESTING
#include <functional>
#endif

namespace colmap {
class ThreadPool;
namespace mvs {
class Workspace;
class Image;
class DepthMap;
class NormalMap;
namespace internal {

// The caller must retain its workspace lock and destination vectors until
// this returns or throws. Accepted tasks are drained before either occurs.
void StagePatchMatchInputs(Workspace& workspace,
                           ThreadPool& io_thread_pool,
                           const std::vector<int>& image_idxs,
                           bool geom_consistency,
                           int ref_image_idx,
                           std::vector<Image>& images,
                           std::vector<DepthMap>& depth_maps,
                           std::vector<NormalMap>& normal_maps);

#ifdef COLMAP_MVS_INPUT_TESTING
// Private test-build seams. Install before starting a batch, clear after drain.
enum class InputPayload { Bitmap, DepthMap, NormalMap };
using InputCopyHook = std::function<void(int, InputPayload)>;
// Positive counts follow accepted submissions; zero signals error capture.
using InputSubmissionHook = std::function<void(std::size_t)>;
void SetInputCopyHookForTesting(InputCopyHook hook);
void SetInputSubmissionHookForTesting(InputSubmissionHook hook);
void NotifyInputCopyForTesting(int image_idx, InputPayload payload);
#endif

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
