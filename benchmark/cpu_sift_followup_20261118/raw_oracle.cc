// SPDX-License-Identifier: BSD-3-Clause
// Compile this driver independently against the pristine raw owner, the
// candidate ordinary API, and the candidate prepared API. Compare full bytes.
#include "thirdparty/VLFeat/mathop.h"
#include "thirdparty/VLFeat/sift.h"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Geometry {
  int width, height;
  double x, y, sigma, angle;
};

void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

std::vector<float> Gradient(int width, int height, int phase) {
  std::vector<float> result(2 * width * height, 0.0f);
  if (phase == 0) return result;
  if (phase == 1) {
    result[2 * ((height / 2) * width + width / 2)] = 1.0f;
    return result;
  }
  for (size_t i = 0; i < result.size() / 2; ++i) {
    result[2 * i] = 0.1f + static_cast<float>((17 * i + 13 * phase) % 101) / 101;
    result[2 * i + 1] = static_cast<float>((29 * i + 97 * phase) % 1009) *
                        static_cast<float>(2 * VL_PI) / 1009;
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Check(argc == 2, "usage: raw_oracle WITNESS");
    std::ofstream out(argv[1], std::ios::binary);
    Check(static_cast<bool>(out), "open witness");
    const int initial_rounding = std::fegetround();
    std::unique_ptr<VlSiftFilt, decltype(&vl_sift_delete)> filter(
        vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
    Check(filter != nullptr, "create filter");
    const std::array<Geometry, 8> cases = {{
        {31, 31, 15, 15, 2, 0},
        {23, 19, 7.25, 8.1, 0.2, 0.7},
        {23, 19, 0, 0, 2, -1.4},
        {23, 19, 22, 17, 7.5, 0.7},
        {2, 2, 0.37, 0, 0.2, 0},
        {31, 31, 15.5, 14.25, 7.5, 0.37},
        {17, 13, 3.7, 10.4, 1.3, 6.7},
        {9, 7, 8.0, 0.4, 0.7, -0.37},
    }};
    uint64_t descriptors = 0;
    uint64_t nonzero = 0;
    for (const int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      Check(std::fesetround(rounding) == 0, "unsupported rounding mode");
      for (const auto& geometry : cases) {
        for (int settings = 0; settings < 3; ++settings) {
          vl_sift_set_magnif(filter.get(), settings == 1 ? 4.5 : 3.0);
          vl_sift_set_window_size(filter.get(), settings == 2 ? 3.0 : 2.0);
#ifdef PREPARED_ORACLE
          std::unique_ptr<VlSiftRawDescriptorGeometry,
                          decltype(&vl_sift_delete_raw_descriptor_geometry)> prepared(
              vl_sift_prepare_raw_descriptor(filter.get(), geometry.width, geometry.height,
                                              geometry.x, geometry.y, geometry.sigma, geometry.angle),
              &vl_sift_delete_raw_descriptor_geometry);
          Check(prepared != nullptr, "preparation unexpectedly refused fixture");
#endif
          for (int phase = 0; phase < 9; ++phase) {
            const auto gradient = Gradient(geometry.width, geometry.height, phase);
            for (const double norm_threshold : {0.0, 0.001, 100.0}) {
              vl_sift_set_norm_thresh(filter.get(), norm_threshold);
              std::array<float, 128> descriptor;
              descriptor.fill(13.25f);
#ifdef PREPARED_ORACLE
              vl_sift_calc_raw_descriptor_prepared(
                  filter.get(), prepared.get(), gradient.data(), descriptor.data());
#else
              vl_sift_calc_raw_descriptor(filter.get(), gradient.data(), descriptor.data(),
                                          geometry.width, geometry.height, geometry.x,
                                          geometry.y, geometry.sigma, geometry.angle);
#endif
              bool any_nonzero = false;
              for (const auto value : descriptor) {
                Check(std::isfinite(value), "nonfinite descriptor");
                Check(value >= 0 && value <= 1, "descriptor value out of range");
                any_nonzero |= value != 0;
                uint32_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                for (int i = 0; i < 4; ++i) out.put(static_cast<char>(bits >> (8 * i)));
              }
              ++descriptors;
              nonzero += any_nonzero;
            }
          }
        }
      }
    }
    Check(std::fesetround(initial_rounding) == 0, "restore rounding");
    Check(descriptors == 2592 && nonzero > 0, "oracle fixture not fully exercised");
    out.flush();
    Check(static_cast<bool>(out), "write witness");
    std::cerr << "ORACLE {\"descriptors\":" << descriptors
              << ",\"nonzero\":" << nonzero << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
