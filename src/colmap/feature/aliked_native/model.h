// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace colmap {

// Prepared only by an explicitly requested native N16ROT caller. Both model
// snapshots are returned together after their fixed checksums pass.
struct AlikedNativeModelData {
  std::shared_ptr<const std::vector<char>> original;
  std::shared_ptr<const std::vector<char>> native;
  std::string fallback_reason;
};

AlikedNativeModelData PrepareAlikedNativeModelData(
    const std::string& original_model_path,
    const std::string& native_model_path,
    const std::string& gpu_index);

}  // namespace colmap
