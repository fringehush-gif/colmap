// SPDX-License-Identifier: BSD-3-Clause
// Auxiliary-field CUDA integration for FEMTree.inl. Include twice:
//   #define POISSON_AUX_NATIVE_DECL
//   #include "native_aux.h.inc"             // FEMTree.h, BEFORE namespace
//   PoissonRecon #undef POISSON_AUX_NATIVE_DECL #define POISSON_AUX_NATIVE_BODY
//   #include "native_aux.h.inc"             // first statement in
//   #undef POISSON_AUX_NATIVE_BODY            // updateExtrapolatedDataField,
//   line 1087
// Compile only with POISSON_AUX_NATIVE_GPU. The body either returns after
// complete materialization, or falls through to the original method untouched.
#if defined(POISSON_AUX_NATIVE_DECL) && defined(POISSON_AUX_NATIVE_GPU)
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
namespace AuxNativeInput {
inline thread_local bool eligible = false;
struct Scope {
  bool previous_;
  explicit Scope(bool standard_rgb) : previous_(eligible) {
    eligible = standard_rgb;
  }
  ~Scope() { eligible = previous_; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};
}  // namespace AuxNativeInput
#include <functional>

#include "aux_gpu.h"  // Internal CUDA ABI, namespace AuxNativeGpu.
// One lexical setExtrapolatedDataField + native scale operation, never a cache.
namespace AuxNativeScale {
struct Scope;
inline thread_local Scope* active = nullptr;
struct Scope {
  const void* const owner_;
  Scope* const previous_;
  std::vector<const void*> nodes_;
  const bool compatible_scale_;
  bool ready_ = false, verify_ = false;
  explicit Scope(const void* tree_owner, bool default_scale) noexcept
      : owner_(tree_owner),
        previous_(active),
        compatible_scale_(default_scale) {
    active = this;
  }
  ~Scope() noexcept { active = previous_; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  Scope(Scope&&) = delete;
  Scope& operator=(Scope&&) = delete;
  template <class Node, class Tree, class Field, class Functor>
  bool Apply(Tree& tree, const Field& field, Functor& node_functor) const {
    if (!ready_ || owner_ != static_cast<const void*>(&tree)) return false;
    if (verify_) {
      std::vector<const void*> native, captured = nodes_;
      std::uint64_t native_visited = 0;
      const auto started = std::chrono::steady_clock::now();
      tree.tree().processNodes([&](const Node* n) {
        ++native_visited;
        if (field(n)) native.push_back(n);
      });
      const double scan_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        started)
              .count();
      const std::less<const void*> less;
      std::sort(native.begin(), native.end(), less);
      std::sort(captured.begin(), captured.end(), less);
      if (native != captured ||
          std::adjacent_find(captured.begin(), captured.end()) !=
              captured.end())
        throw std::runtime_error(
            "aux scale capture differs from full native slot traversal");
      std::fprintf(
          stderr,
          "# AuxScaleVerify nativeVisited=%llu captured=%zu scanSeconds=%.6f\n",
          (unsigned long long)native_visited,
          nodes_.size(),
          scan_seconds);
    }
    const auto scale_started = std::chrono::steady_clock::now();
    for (const void* n : nodes_) node_functor(static_cast<const Node*>(n));
    const double scale_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      scale_started)
            .count();
    std::fprintf(stderr,
                 "# AuxScaleTraversal nodes=%zu seconds=%.6f\n",
                 nodes_.size(),
                 scale_seconds);
    return true;
  }
};
}  // namespace AuxNativeScale
// This declaration block is included at GLOBAL scope before namespace
// PoissonRecon. The containing Poisson TU uses -ffast-math; classification must
// inspect object representations BEFORE floating comparisons/conversions.
namespace AuxNativeGuard {
static_assert(sizeof(float) == 4 && sizeof(double) == 8 &&
                  std::numeric_limits<float>::radix == 2 &&
                  std::numeric_limits<float>::digits == 24 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53,
              "aux guard requires IEEE binary32/binary64");
inline std::uint32_t Bits(const float& x) noexcept {
  std::uint32_t b;
  std::memcpy(&b, &x, 4);
  return b;
}
inline std::uint64_t Bits(const double& x) noexcept {
  std::uint64_t b;
  std::memcpy(&b, &x, 8);
  return b;
}
inline bool IsZero(const float& x) noexcept {
  return (Bits(x) & 0x7fffffffu) == 0;
}
inline bool IsZero(const double& x) noexcept {
  return (Bits(x) & 0x7fffffffffffffffull) == 0;
}
inline bool IsFinite(const float& x) noexcept {
  return (Bits(x) & 0x7fffffffu) < 0x7f800000u;
}
inline bool IsFinite(const double& x) noexcept {
  return (Bits(x) & 0x7fffffffffffffffull) < 0x7ff0000000000000ull;
}
inline bool NormalOrZero(const float& x) noexcept {
  const auto m = Bits(x) & 0x7fffffffu;
  return m == 0 || (m >= 0x00800000u && m < 0x7f800000u);
}
inline bool NormalOrZero(const double& x) noexcept {
  const auto m = Bits(x) & 0x7fffffffffffffffull;
  return m == 0 || (m >= 0x0010000000000000ull && m < 0x7ff0000000000000ull);
}
inline bool NonnegativeNormalOrZero(const float& x) noexcept {
  const auto b = Bits(x), m = b & 0x7fffffffu;
  return m == 0 || (!(b & 0x80000000u) && m >= 0x00800000u && m < 0x7f800000u);
}
inline bool NonnegativeNormalOrZero(const double& x) noexcept {
  const auto b = Bits(x);
  const auto m = b & 0x7fffffffffffffffull;
  return m == 0 || (!(b & 0x8000000000000000ull) &&
                    m >= 0x0010000000000000ull && m < 0x7ff0000000000000ull);
}
template <class Real>
bool DefaultScale(const Real& value) noexcept {
  if constexpr (std::is_same_v<Real, float> || std::is_same_v<Real, double>)
    return Bits(value) == Bits(Real(32));
  else
    return false;
}
}  // namespace AuxNativeGuard
#endif

#if defined(POISSON_AUX_NATIVE_BODY) && defined(POISSON_AUX_NATIVE_GPU)
if constexpr (Dim == 3 && std::is_same_v<Real, float> &&
              DataSig == FEMDegreeAndBType<0, BOUNDARY_FREE>::Signature &&
              DensityDegree == 2 && !CreateNodes &&
              (std::is_same_v<Data, DirectSum<float, Point<float, 3>>> ||
               std::is_same_v<Data, DirectSum<float, Point<float>>>)) {
  // updateExtrapolatedDataField can also add to a nonempty field. The
  // default Reconstructors path passes an empty one; preserve all others.
  const char* aux_mode = std::getenv("POISSON_GPU_AUX");
  if (!aux_mode) aux_mode = "1";
  ::AuxNativeGpu::Stats admission{};
  const bool rgb_shape = [&]() {
    if constexpr (std::is_same_v<Data, DirectSum<float, Point<float>>>)
      return zero.template get<0>().dim() == 3;
    else
      return true;
  }();
  if (std::strcmp(aux_mode, "verify") == 0 && !::AuxNativeGpu::native_reference)
    std::fprintf(stderr,
                 "# AuxNativeAdmission rawNodes=%lld samples=%zu shape=%d "
                 "density=%d nearest=%d existing=%zu\n",
                 (long long)_nodeCount,
                 sampleNum,
                 (int)rgb_shape,
                 (int)(density != nullptr),
                 (int)nearest,
                 dataField.size());
  if (::AuxNativeInput::eligible && ::AuxNativeScale::active &&
      ::AuxNativeScale::active->owner_ == static_cast<const void*>(this) &&
      ::AuxNativeScale::active->compatible_scale_ &&
      (std::strcmp(aux_mode, "1") == 0 ||
       std::strcmp(aux_mode, "verify") == 0) &&
      !::AuxNativeGpu::native_reference && rgb_shape && !density && !nearest &&
      !dataField.size() && sampleNum &&
      ::AuxNativeGuard::IsZero(zero.template get<0>()[0]) &&
      ::AuxNativeGuard::IsZero(zero.template get<0>()[1]) &&
      ::AuxNativeGuard::IsZero(zero.template get<0>()[2]) && _nodeCount > 0 &&
      sampleNum >= 400000 && sampleNum <= 4000000 &&
      ::AuxNativeGpu::Available(admission)) {
    bool destination_touched = false;
    try {
      const auto pack_start = std::chrono::steady_clock::now();
      constexpr uint32_t MaxCompactNodes = 16000000;
      constexpr size_t MaxActualIndexSpan = (size_t(1) << 28);
      using Node = FEMTreeNode;
      struct Tmp {
        Node* node;
        uint32_t parent;
        uint8_t depth;
      };
      std::vector<int32_t> by_index;
      const auto include_index = [&](node_index_type index) -> bool {
        if (index < 0 || index >= _nodeCount ||
            (size_t)index >= MaxActualIndexSpan)
          return false;
        const size_t required = (size_t)index + 1;
        if (required <= by_index.size()) return true;
        size_t next = by_index.empty() ? size_t(4096) : by_index.size();
        while (next < required) next = std::min(MaxActualIndexSpan, next * 2);
        by_index.resize(next, -1);
        return true;
      };
      std::vector<Tmp> tmp;
      std::vector<uint32_t> seed_owner;
      std::vector<AuxNativeGpu::Four> raw;
      seed_owner.reserve(sampleNum);
      raw.reserve(sampleNum);
      bool supported = true;
      uint64_t new_ancestors = 0, pack_ancestor_visits = 0,
               native_ancestor_jobs = 0, out_of_bounds = 0;
      for (size_t i = 0; i < sampleNum && supported; ++i) {
        const PointSample& sampleAndNode = sampleFunctor(i);
        const auto& sample = sampleAndNode.sample;
        // Check stored binary32 values before division: fast-math may
        // assume finite values, and DAZ may turn a subnormal into zero.
        if (!::AuxNativeGuard::NonnegativeNormalOrZero(sample.weight)) {
          supported = false;
          break;
        }
        for (unsigned int k = 0; k < Dim; ++k)
          if (!::AuxNativeGuard::NormalOrZero(sample.data[k]))
            supported = false;
        if (!supported) break;
        Point<Real, Dim> p = ::AuxNativeGuard::IsZero(sample.weight)
                                 ? sample.data
                                 : sample.data / sample.weight;
        for (unsigned int k = 0; k < Dim; ++k)
          if (!::AuxNativeGuard::NormalOrZero(p[k])) supported = false;
        if (!supported) break;
        if (!_InBounds(p)) {
          ++out_of_bounds;
          continue;
        }
        const Data& datum = sampleDataFunctor(i);
        const auto& rgb = datum.template get<0>();
        if constexpr (std::is_same_v<Data, DirectSum<float, Point<float>>>)
          if (rgb.dim() != 3) {
            supported = false;
            break;
          }
        const AuxNativeGpu::Four seed = {
            {rgb[0], rgb[1], rgb[2], sample.weight}};
        // The native bounds predicate admits NaN. Never silently drop
        // such a sample or change FTZ behavior in the private GPU path.
        for (const float& v : seed.v)
          if (!::AuxNativeGuard::NonnegativeNormalOrZero(v)) supported = false;
        if (!supported) break;
        Node* first = (Node*)sampleAndNode.node;
        if (!first) {
          supported = false;
          break;
        }
        if (_localDepth(first) < 0) continue;  // Native ancestor loop is empty.
        native_ancestor_jobs += (uint64_t)_localDepth(first) + 1;
        for (Node* n = first; n && _localDepth(n) >= 0; n = n->parent) {
          ++pack_ancestor_visits;
          const node_index_type idx = n->nodeData.nodeIndex;
          const LocalDepth depth = _localDepth(n);
          if (depth > 18 || depth < 0 || tmp.size() >= MaxCompactNodes ||
              !include_index(idx)) {
            supported = false;
            break;
          }
          if (by_index[(size_t)idx] != -1)
            break;  // Earlier sample supplied ancestor closure.
          const uint32_t id = (uint32_t)tmp.size();
          by_index[(size_t)idx] = (int32_t)id;
          tmp.push_back({n, UINT32_MAX, (uint8_t)depth});
          ++new_ancestors;
        }
        if (!supported) break;
        // Starting node may have been seen by an earlier sample.
        const node_index_type first_index = first->nodeData.nodeIndex;
        if (first_index < 0 || first_index >= _nodeCount ||
            (size_t)first_index >= by_index.size() ||
            by_index[(size_t)first_index] < 0) {
          supported = false;
          break;
        }
        seed_owner.push_back((uint32_t)by_index[(size_t)first_index]);
        raw.push_back(seed);
      }
      if (supported && !tmp.empty() && tmp.size() <= UINT32_MAX) {
        // Every seed must terminate at this single local-depth-zero
        // root. No padded sibling root may be omitted accidentally.
        uint32_t root_count = 0;
        for (Tmp& t : tmp) {
          if (t.depth == 0) {
            if (t.node != _spaceRoot) supported = false;
            ++root_count;
          } else {
            Node* parent = t.node->parent;
            const node_index_type idx =
                parent ? parent->nodeData.nodeIndex : -1;
            if (idx < 0 || idx >= _nodeCount ||
                (size_t)idx >= by_index.size() || by_index[(size_t)idx] < 0)
              supported = false;
            else
              t.parent = (uint32_t)by_index[(size_t)idx];
          }
        }
        if (root_count != 1) supported = false;
      }
      if (supported && !tmp.empty()) {
        const uint32_t N = (uint32_t)tmp.size();
        std::vector<uint32_t> order(N), remap(N), level_offsets(20, 0);
        uint32_t max_depth = 0;
        for (uint32_t n = 0; n < N; ++n) {
          ++level_offsets[tmp[n].depth + 1];
          max_depth = std::max(max_depth, uint32_t(tmp[n].depth));
        }
        for (uint32_t d = 1; d <= max_depth + 1; ++d)
          level_offsets[d] += level_offsets[d - 1];
        level_offsets.resize(max_depth + 2);
        // Bounded depth is a small integer key: stable counting order,
        // avoiding N log N comparisons and a second sort workspace.
        auto depth_cursor = level_offsets;
        for (uint32_t n = 0; n < N; ++n)
          order[depth_cursor[tmp[n].depth]++] = n;
        for (uint32_t n = 0; n < N; ++n) remap[order[n]] = n;
        std::vector<uint32_t> child_offsets(1, 0), children,
            seed_offsets(N + 1, 0), seed_ids(raw.size());
        std::vector<Node*> output_nodes(N);
        for (uint32_t n = 0; n < N; ++n) {
          Node* node = tmp[order[n]].node;
          output_nodes[n] = node;
          if (node->children)
            for (unsigned int slot = 0; slot < 8; ++slot) {
              Node* child = node->children + slot;
              const node_index_type idx = child->nodeData.nodeIndex;
              if (idx >= 0 && idx < _nodeCount &&
                  (size_t)idx < by_index.size() && by_index[(size_t)idx] >= 0)
                children.push_back(remap[(uint32_t)by_index[(size_t)idx]]);
            }
          child_offsets.push_back((uint32_t)children.size());
        }
        for (uint32_t owner : seed_owner) ++seed_offsets[remap[owner] + 1];
        for (uint32_t n = 1; n <= N; ++n)
          seed_offsets[n] += seed_offsets[n - 1];
        std::vector<uint32_t> cursor = seed_offsets;
        for (uint32_t s = 0; s < seed_owner.size(); ++s)
          seed_ids[cursor[remap[seed_owner[s]]]++] = s;
        std::vector<AuxNativeGpu::Four> output(N);
        const AuxNativeGpu::View view = {N,
                                         (uint32_t)raw.size(),
                                         max_depth,
                                         level_offsets.data(),
                                         child_offsets.data(),
                                         children.data(),
                                         seed_offsets.data(),
                                         seed_ids.data(),
                                         raw.data()};
        const auto pack_end = std::chrono::steady_clock::now();
        AuxNativeGpu::Stats gpu_stats{};
        if (AuxNativeGpu::Reduce(view, output.data(), gpu_stats)) {
          // No CPU replay after the first possible destination write.
          destination_touched = true;
          const auto materialize_start = std::chrono::steady_clock::now();
          for (uint32_t n = 0; n < N; ++n)
            if (IsActiveNode<Dim>(output_nodes[n])) {
              auto& value = dataField.at(output_nodes[n],
                                         ProjectiveData<Data, Real>(zero));
              for (unsigned int c = 0; c < 3; ++c)
                value.data.template get<0>()[c] = output[n].v[c];
              value.weight = output[n].v[3];
            }
          for (uint64_t w = 0; w < out_of_bounds; ++w)
            MK_WARN("Point is out of bounds");
          const auto materialize_end = std::chrono::steady_clock::now();
          if (std::strcmp(aux_mode, "verify") == 0) {
            // Both fields see the SAME post-normal native tree and
            // sample payload. The recursive call skips this hook.
            SparseNodeData<ProjectiveData<Data, Real>,
                           IsotropicUIntPack<Dim, DataSig>>
                reference;
            const auto reference_start = std::chrono::steady_clock::now();
            ::AuxNativeGpu::native_reference = true;
            try {
              this->template updateExtrapolatedDataField<DataSig,
                                                         CreateNodes,
                                                         DensityDegree,
                                                         Data>(
                  zero,
                  reference,
                  sampleNum,
                  sampleFunctor,
                  sampleDataFunctor,
                  density,
                  nearest);
            } catch (...) {
              ::AuxNativeGpu::native_reference = false;
              throw;
            }
            ::AuxNativeGpu::native_reference = false;
            double max_rgb = 0, max_weight_relative = 0, max_raw_relative = 0;
            uint64_t active_count = 0, zero_weight = 0;
            for (uint32_t n = 0; n < N; ++n) {
              const auto* cpu = reference(output_nodes[n]);
              const auto* gpu = dataField(output_nodes[n]);
              const bool active = IsActiveNode<Dim>(output_nodes[n]);
              if (bool(cpu) != active || bool(gpu) != active)
                throw std::runtime_error(
                    "aux native/GPU sparse membership differs");
              if (!active) continue;
              ++active_count;
              // Classify source floats before promoting them: a
              // conversion can quiet NaNs or flush subnormals.
              const bool cpu_zero = ::AuxNativeGuard::IsZero(cpu->weight),
                         gpu_zero = ::AuxNativeGuard::IsZero(gpu->weight);
              if (!::AuxNativeGuard::NonnegativeNormalOrZero(cpu->weight) ||
                  !::AuxNativeGuard::NonnegativeNormalOrZero(gpu->weight) ||
                  cpu_zero != gpu_zero)
                throw std::runtime_error("aux weight class differs");
              const double cw = cpu->weight, gw = gpu->weight;
              if (cpu_zero) ++zero_weight;
              const double weight_relative =
                  std::abs(cw - gw) / std::max(1., std::abs(cw));
              if (!::AuxNativeGuard::NonnegativeNormalOrZero(weight_relative))
                throw std::runtime_error("aux nonfinite weight error");
              max_weight_relative =
                  std::max(max_weight_relative, weight_relative);
              for (unsigned c = 0; c < 3; ++c) {
                const float& af = cpu->data.template get<0>()[c];
                const float& bf = gpu->data.template get<0>()[c];
                if (!::AuxNativeGuard::NonnegativeNormalOrZero(af) ||
                    !::AuxNativeGuard::NonnegativeNormalOrZero(bf))
                  throw std::runtime_error("aux nonfinite field");
                const double a = af, b = bf;
                const double raw_relative =
                    std::abs(a - b) / std::max(1., std::abs(a));
                if (!::AuxNativeGuard::NonnegativeNormalOrZero(raw_relative))
                  throw std::runtime_error("aux nonfinite raw error");
                max_raw_relative = std::max(max_raw_relative, raw_relative);
                if (!cpu_zero && !gpu_zero) {
                  const double rgb_error = std::abs(a / cw - b / gw);
                  if (!::AuxNativeGuard::NonnegativeNormalOrZero(rgb_error))
                    throw std::runtime_error("aux nonfinite RGB error");
                  max_rgb = std::max(max_rgb, rgb_error);
                }
              }
            }
            if (reference.size() != active_count ||
                dataField.size() != active_count ||
                !::AuxNativeGuard::NonnegativeNormalOrZero(max_rgb) ||
                !::AuxNativeGuard::NonnegativeNormalOrZero(
                    max_weight_relative) ||
                !::AuxNativeGuard::NonnegativeNormalOrZero(max_raw_relative) ||
                max_rgb > 0.05)
              throw std::runtime_error("aux native field validation failed");
            std::fprintf(stderr,
                         "# AuxNativeVerify slots=%llu zeroWeight=%llu "
                         "maxRgb=%.12g maxWeightRelative=%.12g "
                         "maxRawRelative=%.12g referenceSeconds=%.6f\n",
                         (unsigned long long)active_count,
                         (unsigned long long)zero_weight,
                         max_rgb,
                         max_weight_relative,
                         max_raw_relative,
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - reference_start)
                             .count());
          }
          const double pack_seconds =
              std::chrono::duration<double>(pack_end - pack_start).count();
          const double materialize_seconds =
              std::chrono::duration<double>(materialize_end - materialize_start)
                  .count();
          const uint64_t host_bytes =
              (uint64_t)by_index.capacity() * sizeof(int32_t) +
              (uint64_t)tmp.capacity() * sizeof(Tmp) +
              (uint64_t)seed_owner.capacity() * sizeof(uint32_t) +
              (uint64_t)raw.capacity() * sizeof(AuxNativeGpu::Four) +
              (uint64_t)order.capacity() * sizeof(uint32_t) +
              (uint64_t)remap.capacity() * sizeof(uint32_t) +
              (uint64_t)level_offsets.capacity() * sizeof(uint32_t) +
              (uint64_t)child_offsets.capacity() * sizeof(uint32_t) +
              (uint64_t)children.capacity() * sizeof(uint32_t) +
              (uint64_t)seed_offsets.capacity() * sizeof(uint32_t) +
              (uint64_t)seed_ids.capacity() * sizeof(uint32_t) +
              (uint64_t)output_nodes.capacity() * sizeof(Node*) +
              (uint64_t)output.capacity() * sizeof(AuxNativeGpu::Four);
          std::fprintf(stderr,
                       "# AuxNativeGPU nodes=%u seeds=%zu newAncestors=%llu "
                       "packAncestorVisits=%llu nativeAncestorJobs=%llu "
                       "outOfBounds=%llu hostBytes=%llu pack=%.6f gpu=%.6f "
                       "upload=%.6f kernel=%.6f download=%.6f materialize=%.6f "
                       "deviceBytes=%llu integralChannels=%llu\n",
                       N,
                       raw.size(),
                       (unsigned long long)new_ancestors,
                       (unsigned long long)pack_ancestor_visits,
                       (unsigned long long)native_ancestor_jobs,
                       (unsigned long long)out_of_bounds,
                       (unsigned long long)host_bytes,
                       pack_seconds,
                       gpu_stats.total_seconds,
                       gpu_stats.upload_seconds,
                       gpu_stats.kernel_seconds,
                       gpu_stats.download_seconds,
                       materialize_seconds,
                       (unsigned long long)gpu_stats.device_bytes,
                       (unsigned long long)gpu_stats.integral_channels);
          // Publish only a complete successful field, after optional
          // nativeReference verification. Recursive reference calls
          // never reset or publish the caller's lexical scope.
          if (auto* scope = ::AuxNativeScale::active;
              scope && scope->owner_ == static_cast<const void*>(this) &&
              !::AuxNativeGpu::native_reference) {
            if (scope->ready_)
              throw std::runtime_error("aux scale scope already filled");
            std::vector<const void*> captured;
            captured.reserve(dataField.size());
            for (Node* node : output_nodes)
              if (IsActiveNode<Dim>(node)) captured.push_back(node);
            if (captured.size() != dataField.size())
              throw std::runtime_error("aux scale capture size differs");
            scope->nodes_.swap(captured);
            scope->verify_ = std::strcmp(aux_mode, "verify") == 0;
            scope->ready_ = true;
          }
          return;  // Reconstructors retains its original 32^depth scale pass.
        }
        std::fprintf(
            stderr, "# AuxNativeFallback reason=%s\n", gpu_stats.reason);
      }
    } catch (const std::bad_alloc&) {
      if (destination_touched) throw;
      // Packing only read the native tree and samples. Its temporary
      // vectors have unwound; the original empty field is untouched.
      std::fprintf(stderr, "# AuxNativeFallback reason=host-allocation\n");
    }
  }
}
#endif
