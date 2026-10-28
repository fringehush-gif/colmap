// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <memory>

#include <onnxruntime_cxx_api.h>

namespace colmap {
// The returned owner must outlive the options and session using its domain.
std::shared_ptr<void> RegisterAlikedNativeOps(Ort::SessionOptions& options);
}  // namespace colmap
