// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>

namespace colmap {

// Existing Bitmap::CloneAsGrey arithmetic for packed RGB8. Source and output
// must cover 3*num_pixels and num_pixels bytes respectively and not overlap.
// Neither buffer is retained. The caller owns their lifetime through the call.
void ConvertPackedRGB8ToGrey(const uint8_t* rgb,
                             size_t num_pixels,
                             uint8_t* gray);

}  // namespace colmap
