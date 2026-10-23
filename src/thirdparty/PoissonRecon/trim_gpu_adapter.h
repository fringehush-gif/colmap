// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "trim_gpu.h"

namespace PoissonTrimGpu {

inline bool Verify() {
  const char* mode = std::getenv("POISSON_GPU_TRIM");
  return mode && std::strcmp(mode, "verify") == 0;
}

inline bool Requested(std::size_t polygons) {
  const char* mode = std::getenv("POISSON_GPU_TRIM");
  if (mode && std::strcmp(mode, "1") != 0 && std::strcmp(mode, "verify") != 0) {
    return false;
  }
  // Conservative starting floor; tiny synthetic correctness cases use verify.
  constexpr std::size_t kMinimumPolygons = 1000000;
  return polygons >= kMinimumPolygons || Verify();
}

struct UnsupportedSize {};

template <typename Index>
struct FlatPolygons {
  std::vector<std::uint64_t> offsets{0};
  std::vector<std::uint32_t> vertices;

  explicit FlatPolygons(const std::vector<std::vector<Index>>& polygons) {
    constexpr auto kMaximum =
        std::size_t(std::numeric_limits<std::int32_t>::max());
    if (polygons.size() > kMaximum) throw UnsupportedSize{};
    std::size_t entries = 0;
    for (const auto& polygon : polygons) {
      if (polygon.size() > kMaximum - entries) throw UnsupportedSize{};
      entries += polygon.size();
    }
    offsets.reserve(polygons.size() + 1);
    vertices.reserve(entries);
    for (const auto& polygon : polygons) {
      for (const Index vertex : polygon) {
        if constexpr (std::is_signed_v<Index>) {
          if (vertex < 0) throw std::invalid_argument("negative mesh index");
        }
        if (std::uint64_t(vertex) > std::numeric_limits<std::uint32_t>::max()) {
          throw std::invalid_argument("mesh index exceeds uint32");
        }
        vertices.push_back(std::uint32_t(vertex));
      }
      offsets.push_back(vertices.size());
    }
  }
};

template <typename Index>
bool TryConnectedComponents(const std::vector<std::vector<Index>>& polygons,
                            std::vector<std::vector<Index>>& components,
                            Stats& stats) {
  if constexpr (sizeof(Index) != sizeof(std::uint32_t)) {
    return false;
  } else {
    if (!Requested(polygons.size()) || !Available(stats)) return false;
    try {
      FlatPolygons<Index> flat(polygons);
      const std::size_t count = polygons.size();
      std::vector<std::uint32_t> roots(count);
      if (!ConnectedComponents(count,
                               flat.vertices.size(),
                               flat.offsets.data(),
                               flat.vertices.data(),
                               roots.data(),
                               stats)) {
        return false;
      }
      constexpr auto kMissing = std::numeric_limits<std::uint32_t>::max();
      std::vector<std::uint32_t> component_ids(count, kMissing);
      std::size_t component_count = 0;
      for (std::size_t face = 0; face < count; ++face) {
        if (roots[face] >= count || roots[face] < face) {
          throw std::runtime_error("invalid GPU component representative");
        }
        if (roots[face] == face) {
          component_ids[face] = std::uint32_t(component_count++);
        }
      }
      std::vector<std::vector<Index>> result(component_count);
      for (std::size_t face = 0; face < count; ++face) {
        const auto component = component_ids[roots[face]];
        if (component >= component_count) {
          throw std::runtime_error("GPU component root is not representative");
        }
        result[component].push_back(Index(face));
      }
      components.swap(result);
      return true;
    } catch (const UnsupportedSize&) {
      stats.reason = "host-index-cap";
      return false;
    } catch (const std::bad_alloc&) {
      stats.reason = "host-allocation";
      return false;
    }
  }
}

}  // namespace PoissonTrimGpu
