// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/mat.h"

#include <cstddef>
namespace colmap::mvs::normal_pack {
enum class Backend { kAutomatic, kCpp, kBoundedCpp };
// ISPC build, AVX2/OS and exact FP-state gate for automatic selection.
bool Available();
// Exact FP-state gate for explicit bounded C++; independent of ISPC and AVX2.
// False selects the original C++ codec. Non-SSE FP controls remain unqualified.
bool BoundedCppAvailable();
// These functions never allocate or retain a pointer. Validation must precede
// output allocation/publication; unchecked encoding requires validated input.
bool Validate(const float* source,
              size_t count,
              Backend backend = Backend::kAutomatic);
void SnormUnchecked(const float* source,
                    size_t count,
                    int16_t* output,
                    Backend backend = Backend::kAutomatic);
void OctUnchecked(const float* source,
                  size_t plane,
                  OctNormalPair* output,
                  uint16_t* magnitude = nullptr,
                  Backend backend = Backend::kAutomatic);
}  // namespace colmap::mvs::normal_pack
