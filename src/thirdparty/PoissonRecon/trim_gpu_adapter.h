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

inline bool FiniteNonnegative(double value) {
  // The vendored trimmer is compiled with fast-math, which may fold isfinite.
  static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                std::numeric_limits<double>::is_iec559);
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return !(bits >> 63) &&
         (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

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

template <typename Index>
bool TryBoundaryGraph(const std::vector<std::vector<Index>>& polygons,
                      const std::vector<std::vector<Index>>& components,
                      std::vector<std::uint64_t>& graph,
                      Stats& stats) {
  if constexpr (sizeof(Index) != sizeof(std::uint32_t)) {
    return false;
  } else {
    if (!Requested(polygons.size()) || !Available(stats)) return false;
    try {
      FlatPolygons<Index> flat(polygons);
      constexpr auto kMissing = std::numeric_limits<std::uint32_t>::max();
      std::vector<std::uint32_t> labels(polygons.size(), kMissing);
      for (std::size_t component = 0; component < components.size();
           ++component) {
        if (component >= kMissing) throw UnsupportedSize{};
        for (const Index index : components[component]) {
          if constexpr (std::is_signed_v<Index>) {
            if (index < 0)
              throw std::invalid_argument("negative polygon index");
          }
          const auto face = std::size_t(index);
          if (face >= labels.size() || labels[face] != kMissing) {
            throw std::invalid_argument("invalid component polygon partition");
          }
          labels[face] = std::uint32_t(component);
        }
      }
      if (std::find(labels.begin(), labels.end(), kMissing) != labels.end()) {
        throw std::invalid_argument("incomplete component polygon partition");
      }
      std::vector<std::uint64_t> result;
      if (!BoundaryGraph(polygons.size(),
                         flat.vertices.size(),
                         flat.offsets.data(),
                         flat.vertices.data(),
                         labels.data(),
                         result,
                         stats)) {
        return false;
      }
      for (const auto edge : result) {
        const auto left = std::uint32_t(edge >> 32);
        const auto right = std::uint32_t(edge);
        if (left >= right || right >= components.size()) {
          throw std::runtime_error("invalid GPU boundary component edge");
        }
      }
      if (!std::is_sorted(result.begin(), result.end()) ||
          std::adjacent_find(result.begin(), result.end()) != result.end()) {
        throw std::runtime_error(
            "GPU boundary edges are not unique and sorted");
      }
      graph.swap(result);
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

// Replay native minimum-area/index selection on a small graph copy. Every
// executed merge must have a unique recipient. Leaf deletion can change
// adjacency order but cannot affect a later degree-one merge. Thus all area
// additions and polygon-list splices occur in the native order. Reject before
// touching real nodes if an actual branching merge would be required.
template <typename Node>
bool HasOrderIndependentMerges(const std::vector<Node>& nodes,
                               double threshold,
                               std::size_t& completed_merges) {
  completed_merges = 0;
  if (!FiniteNonnegative(threshold)) return false;
  try {
    const std::size_t count = nodes.size();
    std::vector<double> areas(count);
    std::vector<std::vector<std::size_t>> neighbors(count);
    std::vector<bool> active(count);
    for (std::size_t i = 0; i < count; ++i) {
      areas[i] = nodes[i].area;
      if (!FiniteNonnegative(areas[i])) return false;
      active[i] = !nodes[i].polygonIndices.empty();
      for (const auto* neighbor : nodes[i].neighbors) {
        neighbors[i].push_back(std::size_t(neighbor - nodes.data()));
      }
    }
    for (;;) {
      std::size_t selected = count;
      for (std::size_t i = 0; i < count; ++i) {
        if (active[i] && !neighbors[i].empty() &&
            (selected == count || areas[i] < areas[selected])) {
          selected = i;
        }
      }
      if (selected == count || !(areas[selected] < threshold)) return true;
      if (neighbors[selected].size() != 1) return false;
      const std::size_t recipient = neighbors[selected][0];
      if (recipient >= count || !active[recipient] || recipient == selected) {
        return false;
      }
      auto& adjacent = neighbors[recipient];
      const auto position =
          std::find(adjacent.begin(), adjacent.end(), selected);
      if (position == adjacent.end()) return false;
      *position = adjacent.back();
      adjacent.pop_back();
      areas[recipient] += areas[selected];
      if (!FiniteNonnegative(areas[recipient])) return false;
      areas[selected] = 0;
      active[selected] = false;
      neighbors[selected].clear();
      ++completed_merges;
    }
  } catch (const std::bad_alloc&) {
    return false;
  }
}

}  // namespace PoissonTrimGpu
