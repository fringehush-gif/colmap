// SPDX-License-Identifier: BSD-3-Clause
// Match the existing NVCC --default-stream per-thread dense helper semantics.
#if !defined(COLMAP_HIP_ENABLED) && !defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "colmap/mvs/patch_match_source_array.h"

#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

#if !defined(COLMAP_HIP_ENABLED) && CUDART_VERSION >= 13020
#include <cuda.h>
#include <cudaTypedefs.h>
#define COLMAP_SPARSE_SOURCE_RUNTIME 1
#endif

namespace colmap {
namespace mvs {
namespace {
bool InjectSparseFailure(const SparseSourceTestFailure* failure,
                         const char* operation,
                         int* code) {
  for (auto* f = failure; f != nullptr; f = f->next) {
    if (std::strcmp(f->operation, operation) == 0 &&
        ++f->calls == f->fail_on_call) {
      if (f->code != 0) {
        *code = f->code;
        return true;
      }
    }
  }
  return false;
}
template <typename Call>
auto SparseOperation(const SparseSourceTestFailure* failure,
                     const char* operation,
                     Call&& call) -> decltype(call()) {
  int code = 0;
  if (InjectSparseFailure(failure, operation, &code))
    return static_cast<decltype(call())>(code);
  return call();
}
enum class SparseFailureKind { kUnsupported, kResource, kFatal };
struct SparseCallFailure : std::exception {
  SparseCallFailure(const char* operation,
                    int code,
                    SparseFailureKind kind,
                    bool driver = false) noexcept
      : operation(operation), code(code), kind(kind), driver(driver) {
    std::snprintf(message,
                  sizeof(message),
                  "%s failed (%s code %d)",
                  operation,
                  driver ? "driver" : "runtime",
                  code);
  }
  const char* what() const noexcept override { return message; }
  char message[256] = {};
  const char* operation;
  int code;
  SparseFailureKind kind;
  bool driver;
};
void SparseRuntime(cudaError_t status,
                   const char* operation,
                   bool allow_resource = false) {
  if (status == cudaSuccess) return;
  const auto kind = allow_resource && status == cudaErrorNotSupported
                        ? SparseFailureKind::kUnsupported
                    : allow_resource && status == cudaErrorMemoryAllocation
                        ? SparseFailureKind::kResource
                        : SparseFailureKind::kFatal;
  throw SparseCallFailure(operation, static_cast<int>(status), kind);
}
void ConsumeOwnRefusal(cudaError_t status) {
  const auto recorded = cudaGetLastError();
  if (recorded != cudaSuccess && recorded != status)
    throw SparseCallFailure("unexpected CUDA error after sparse refusal",
                            static_cast<int>(recorded),
                            SparseFailureKind::kFatal);
}
template <typename Allocate>
void SparseAllocation(const char* operation,
                      Allocate&& allocate,
                      const SparseSourceTestFailure* failure = nullptr) {
  // Never reinterpret an earlier async/device error as this allocation's OOM.
  SparseRuntime(cudaPeekAtLastError(),
                "preexisting CUDA error before sparse allocation");
  const auto status = SparseOperation(failure, operation, allocate);
  if (status == cudaErrorMemoryAllocation || status == cudaErrorNotSupported)
    ConsumeOwnRefusal(status);
  SparseRuntime(status, operation, true);
}
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
void SparseDriver(CUresult status,
                  const char* operation,
                  bool allow_resource = false) {
  if (status == CUDA_SUCCESS) return;
  const auto kind = allow_resource && status == CUDA_ERROR_NOT_SUPPORTED
                        ? SparseFailureKind::kUnsupported
                    : allow_resource && status == CUDA_ERROR_OUT_OF_MEMORY
                        ? SparseFailureKind::kResource
                        : SparseFailureKind::kFatal;
  throw SparseCallFailure(operation, static_cast<int>(status), kind, true);
}
#endif
template <typename T>
bool SameOwnedPayload(const std::shared_ptr<const T>& a,
                      const std::shared_ptr<const T>& b) {
  return a && b && a.use_count() > 0 && b.use_count() > 0 &&
         a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
}
std::shared_ptr<const Bitmap> BitmapOwner(const PatchMatch::Problem& p,
                                          int id) {
  if (!p.images || id < 0 || id == p.ref_image_idx ||
      size_t(id) >= p.images->size() || !p.source_bitmap_leases ||
      size_t(id) >= p.source_bitmap_leases->size())
    return {};
  const auto& lease = p.source_bitmap_leases->at(id);
  const auto& image = p.images->at(id);
  if (!lease || lease.use_count() == 0 || &p.InputBitmap(id) != lease.get() ||
      !lease->IsGrey() || lease->Width() != image.GetWidth() ||
      lease->Height() != image.GetHeight())
    return {};
  return lease;
}
std::shared_ptr<const DepthMap> DepthOwner(const PatchMatch::Problem& p,
                                           int id) {
  if (!p.images || id < 0 || id == p.ref_image_idx ||
      size_t(id) >= p.images->size() || !p.source_depth_leases ||
      size_t(id) >= p.source_depth_leases->size())
    return {};
  const auto& lease = p.source_depth_leases->at(id);
  if (!lease || lease.use_count() == 0 || &p.InputDepthMap(id) != lease.get() ||
      lease->GetWidth() != p.images->at(id).GetWidth() ||
      lease->GetHeight() != p.images->at(id).GetHeight())
    return {};
  return lease;
}
class SourceDeviceScope {
 public:
  explicit SourceDeviceScope(int device) {
    SparseRuntime(cudaGetDevice(&previous_), "cudaGetDevice source view");
    changed_ = device != previous_;
    if (changed_)
      SparseRuntime(cudaSetDevice(device), "cudaSetDevice source view");
  }
  void Restore() {
    if (restored_) return;
    restored_ = true;
    if (changed_)
      SparseRuntime(cudaSetDevice(previous_),
                    "cudaSetDevice source view restore");
  }
  ~SourceDeviceScope() {
    if (changed_ && !restored_) {
      const auto status = cudaSetDevice(previous_);
      if (status != cudaSuccess)
        LOG(FATAL) << "Terminal source-view device restore failure: "
                   << static_cast<int>(status);
    }
  }

 private:
  int previous_;
  bool changed_;
  bool restored_ = false;
};

#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
struct SourceApi {
  PFN_cuMemCreate_v10020 create = nullptr;
  PFN_cuMemRelease_v10020 release = nullptr;
  PFN_cuMemGetAllocationGranularity_v10020 granularity = nullptr;
  PFN_cuMemMapArrayAsync_v11010 map = nullptr;
  bool Load() {
    const auto entry = [](const char* name, unsigned version, void** pointer) {
      cudaDriverEntryPointQueryResult query =
          cudaDriverEntryPointSymbolNotFound;
      SparseRuntime(cudaPeekAtLastError(),
                    "preexisting CUDA error before sparse entrypoint");
      const auto status = cudaGetDriverEntryPointByVersion(
          name, pointer, version, cudaEnableLegacyStream, &query);
      if (status == cudaErrorNotSupported) {
        ConsumeOwnRefusal(status);
        return false;
      }
      SparseRuntime(status, "cudaGetDriverEntryPointByVersion");
      return query == cudaDriverEntryPointSuccess && *pointer;
    };
    void *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
    if (!entry("cuMemCreate", 10020, &a) || !entry("cuMemRelease", 10020, &b) ||
        !entry("cuMemGetAllocationGranularity", 10020, &c) ||
        !entry("cuMemMapArrayAsync", 11010, &d))
      return false;
    create = reinterpret_cast<PFN_cuMemCreate_v10020>(a);
    release = reinterpret_cast<PFN_cuMemRelease_v10020>(b);
    granularity = reinterpret_cast<PFN_cuMemGetAllocationGranularity_v10020>(c);
    map = reinterpret_cast<PFN_cuMemMapArrayAsync_v11010>(d);
    return true;
  }
};

size_t Product(size_t a, size_t b) {
  THROW_CHECK(a == 0 || b <= std::numeric_limits<size_t>::max() / a);
  return a * b;
}
size_t Sum(size_t a, size_t b) {
  THROW_CHECK_LE(b, std::numeric_limits<size_t>::max() - a);
  return a + b;
}
size_t Round(size_t value, size_t alignment) {
  THROW_CHECK_GT(alignment, 0);
  const size_t extra = (alignment - value % alignment) % alignment;
  THROW_CHECK_LE(extra, std::numeric_limits<size_t>::max() - value);
  return value + extra;
}
#endif
struct SourcePlan {
  size_t width = 0, height = 0, slots = 0;
  std::vector<int> ids;
  std::vector<const PatchMatch::Problem*> providers;
  std::vector<const PatchMatch::Problem*> problems;
};
SourcePlan Plan(const std::vector<const PatchMatch::Problem*>& problems,
                bool geometric) {
  SourcePlan plan;
  if (problems.size() < 3 || problems.size() > 5) return plan;
  std::vector<int> references;
  for (const auto* p : problems) {
    if (!p || !p->images || p->src_image_idxs.empty() || p->ref_image_idx < 0 ||
        size_t(p->ref_image_idx) >= p->images->size() ||
        (geometric && !p->depth_maps) ||
        std::find(references.begin(), references.end(), p->ref_image_idx) !=
            references.end())
      return {};
    references.push_back(p->ref_image_idx);
    size_t width = 0, height = 0;
    for (int id : p->src_image_idxs) {
      if (id < 0 || size_t(id) >= p->images->size() || id == p->ref_image_idx)
        return {};
      const auto& image = p->images->at(id);
      const auto bitmap = BitmapOwner(*p, id);
      if (!image.GetWidth() || !image.GetHeight() || !bitmap) return {};
      if (geometric &&
          (!DepthOwner(*p, id) || size_t(id) >= p->depth_maps->size() ||
           p->InputDepthMap(id).GetWidth() != image.GetWidth() ||
           p->InputDepthMap(id).GetHeight() != image.GetHeight()))
        return {};
      width = std::max(width, image.GetWidth());
      height = std::max(height, image.GetHeight());
      const auto found = std::find(plan.ids.begin(), plan.ids.end(), id);
      if (found == plan.ids.end()) {
        plan.ids.push_back(id);
        plan.providers.push_back(p);
      } else {
        const auto& old =
            plan.providers[found - plan.ids.begin()]->images->at(id);
        const auto* provider = plan.providers[found - plan.ids.begin()];
        if (!SameOwnedPayload(BitmapOwner(*provider, id), bitmap) ||
            (geometric && !SameOwnedPayload(DepthOwner(*provider, id),
                                            DepthOwner(*p, id))) ||
            old.GetPath() != image.GetPath() ||
            old.GetWidth() != image.GetWidth() ||
            old.GetHeight() != image.GetHeight())
          return {};
      }
    }
    if (plan.slots && (plan.width != width || plan.height != height)) return {};
    plan.width = width;
    plan.height = height;
    if (p->src_image_idxs.size() >
        std::numeric_limits<size_t>::max() - plan.slots)
      return {};
    plan.slots += p->src_image_idxs.size();
  }
  if (plan.ids.size() >= plan.slots) return {};
  plan.problems = problems;
  return plan;
}
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
struct SourceLayout {
  size_t width = 0, height = 0, stride = 0, pool = 0;
};
size_t SourcePoolBudget(const SourcePlan& plan,
                        const SourceLayout& image,
                        const SourceLayout& depth,
                        bool geometric) {
  if (!image.pool || (geometric && !depth.pool)) return 0;
  const size_t total = Sum(image.pool, depth.pool);
  const size_t independent = Product(
      Product(Product(plan.width, plan.height), plan.slots), geometric ? 5 : 1);
  if (total >= independent ||
      total > (std::numeric_limits<size_t>::max() - 32 * 1024 * 1024) / 2)
    return 0;
  return 2 * total +
         32 * 1024 * 1024;  // Pools + conservative array/page metadata.
}
struct SourceSupport {
  int device = -1;
  SourceApi api;
  CUmemAllocationProp property = {};
  size_t granularity = 0;
  cudaDeviceProp device_property = {};
  bool Query(const SparseSourceTestFailure* failure = nullptr) {
    SparseRuntime(
        SparseOperation(
            failure, "cudaGetDevice", [&] { return cudaGetDevice(&device); }),
        "cudaGetDevice");
    SparseRuntime(
        SparseOperation(
            failure,
            "cudaGetDeviceProperties",
            [&] { return cudaGetDeviceProperties(&device_property, device); }),
        "cudaGetDeviceProperties");
    if (device < 0 || device >= 32 ||
        !device_property.sparseCudaArraySupported || !api.Load())
      return false;
    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device;
    property.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
    const auto status = api.granularity(
        &granularity, &property, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (status == CUDA_ERROR_NOT_SUPPORTED) return false;
    SparseDriver(status, "cuMemGetAllocationGranularity");
    return granularity != 0;
  }
  template <typename T>
  SourceLayout Layout(const SourcePlan& plan,
                      const SparseSourceTestFailure* failure = nullptr) const {
    if (!plan.width || !plan.height) return {};
    cudaArray_t probe = nullptr;
    const auto channel = cudaCreateChannelDesc<T>();
    SparseAllocation(
        "cudaMalloc3DArray probe",
        [&] {
          const auto status =
              cudaMalloc3DArray(&probe,
                                &channel,
                                make_cudaExtent(1024, 1024, 1),
                                cudaArrayLayered | cudaArraySparse);
          if (status != cudaSuccess && probe != nullptr)
            throw SparseCallFailure("unresolved cudaMalloc3DArray probe output",
                                    static_cast<int>(status),
                                    SparseFailureKind::kFatal);
          return status;
        },
        failure);
    cudaArraySparseProperties sparse = {};
    const auto status =
        SparseOperation(failure, "cudaArrayGetSparseProperties", [&] {
          return cudaArrayGetSparseProperties(&sparse, probe);
        });
    // Release the successful probe before classifying its query result.
    const auto cleanup = cudaFreeArray(probe);
    if (cleanup != cudaSuccess)
      throw SparseCallFailure("cudaFreeArray probe",
                              static_cast<int>(cleanup),
                              SparseFailureKind::kFatal);
    if (status == cudaErrorNotSupported) {
      ConsumeOwnRefusal(status);
      return {};
    }
    SparseRuntime(status, "cudaArrayGetSparseProperties");
    if (!sparse.tileExtent.width || !sparse.tileExtent.height ||
        sparse.tileExtent.depth != 1 || sparse.miptailSize)
      return {};
    SourceLayout out;
    out.width = Round(plan.width, sparse.tileExtent.width);
    out.height = Round(plan.height, sparse.tileExtent.height);
    if (out.width > size_t(device_property.maxTexture2DLayered[0]) ||
        out.height > size_t(device_property.maxTexture2DLayered[1]))
      return {};
    for (const auto* p : plan.problems)
      if (p->src_image_idxs.size() >
          size_t(device_property.maxTexture2DLayered[2]))
        return {};
    out.stride = Product(Product(out.width, out.height), sizeof(T));
    out.pool = Round(Product(out.stride, plan.ids.size()), granularity);
    return out;
  }
};
#endif
}  // namespace

struct PatchMatchSparseSources::Impl {
  int device = -1;
  bool geometric = false;
  size_t width = 0, height = 0;
  struct Binding {
    int reference;
    std::vector<int> sources;
    const std::vector<Image>* images;
    const std::vector<DepthMap>* depths;
    std::vector<std::shared_ptr<const Bitmap>> bitmap_owners;
    std::vector<std::shared_ptr<const DepthMap>> depth_owners;
  };
  std::vector<Binding> bindings;
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  SourceApi api;
  cudaStream_t stream = nullptr;
  std::vector<cudaArray_t> images, depths;
  std::vector<CUarrayMapInfo> maps;
  std::vector<size_t> map_batches;
  CUmemGenericAllocationHandle image_pool = 0, depth_pool = 0;
  const SparseSourceTestFailure* test_failure = nullptr;
  std::atomic<size_t> active_views{0};
  // Failure cleanup can mark the bank while its view-creation scope owns this
  // lock. Serialize that reentrant terminal transition with Close/Matches.
  std::recursive_mutex views_mutex;
  bool released = false, abandoned = false;
  CUresult uncertain_map_status = CUDA_SUCCESS;
  struct CleanupResult {
    const char* operation = nullptr;
    int code = 0;
    bool driver = false;
  };
  CleanupResult cleanup_failure;
  CleanupResult Cleanup() noexcept {
    if (abandoned) return cleanup_failure;
    if (released) return {};
    if (device < 0) {
      released = true;
      return {};
    }
    int previous = -1;
    auto status = cudaGetDevice(&previous);
    if (status != cudaSuccess) {
      abandoned = true;
      cleanup_failure = {
          "cudaGetDevice cleanup", static_cast<int>(status), false};
      return cleanup_failure;
    }
    const bool changed = previous != device;
    const auto fail =
        [&](const char* operation, int code, bool driver = false) {
          abandoned = true;
          if (changed) {
            const auto restore = cudaSetDevice(previous);
            if (restore != cudaSuccess) {
              cleanup_failure = {
                  "cudaSetDevice restore", static_cast<int>(restore), false};
              return cleanup_failure;
            }
          }
          cleanup_failure = {operation, code, driver};
          return cleanup_failure;
        };
    if (changed) {
      status = cudaSetDevice(device);
      if (status != cudaSuccess) {
        abandoned = true;
        cleanup_failure = {
            "cudaSetDevice cleanup", static_cast<int>(status), false};
        return cleanup_failure;
      }
    }
    if (uncertain_map_status != CUDA_SUCCESS)
      return fail("unresolved cuMemMapArrayAsync submission",
                  static_cast<int>(uncertain_map_status),
                  true);
    if (stream) {
      status = cudaStreamSynchronize(stream);
      if (status != cudaSuccess)
        return fail("cudaStreamSynchronize before unmap",
                    static_cast<int>(status));
    }
    size_t end = maps.size();
    for (auto batch = map_batches.rbegin(); batch != map_batches.rend();
         ++batch) {
      auto* records = maps.data() + end - *batch;
      for (size_t i = 0; i < *batch; ++i) {
        records[i].memOperationType = CU_MEM_OPERATION_TYPE_UNMAP;
        records[i].memHandle.memHandle = 0;
      }
      const auto unmap = api.map(records,
                                 static_cast<unsigned>(*batch),
                                 reinterpret_cast<CUstream>(stream));
      if (unmap != CUDA_SUCCESS)
        return fail("cuMemMapArrayAsync unmap", static_cast<int>(unmap), true);
      end -= *batch;
    }
    if (stream) {
      status = cudaStreamSynchronize(stream);
      if (status != cudaSuccess)
        return fail("cudaStreamSynchronize after unmap",
                    static_cast<int>(status));
    }
    maps.clear();
    map_batches.clear();
    for (auto* arrays : {&depths, &images}) {
      while (!arrays->empty()) {
        status = SparseOperation(test_failure, "cudaFreeArray cleanup", [&] {
          return cudaFreeArray(arrays->back());
        });
        if (status != cudaSuccess)
          return fail("cudaFreeArray cleanup", static_cast<int>(status));
        arrays->pop_back();
      }
    }
    for (auto* pool : {&depth_pool, &image_pool}) {
      if (!*pool) continue;
      const auto release = SparseOperation(
          test_failure, "cuMemRelease", [&] { return api.release(*pool); });
      if (release != CUDA_SUCCESS)
        return fail("cuMemRelease", static_cast<int>(release), true);
      *pool = 0;
    }
    if (stream) {
      status = cudaStreamDestroy(stream);
      if (status != cudaSuccess)
        return fail("cudaStreamDestroy", static_cast<int>(status));
      stream = nullptr;
    }
    if (changed) {
      status = cudaSetDevice(previous);
      if (status != cudaSuccess) {
        abandoned = true;
        cleanup_failure = {
            "cudaSetDevice restore", static_cast<int>(status), false};
        return cleanup_failure;
      }
    }
    released = true;
    return {};
  }
  void Close() {
    const auto result = Cleanup();
    if (result.operation)
      throw SparseCallFailure(result.operation,
                              result.code,
                              SparseFailureKind::kFatal,
                              result.driver);
  }
  ~Impl() {
    if (abandoned) return;  // Explicit failed cleanup was already surfaced.
    if (active_views.load() != 0)
      LOG(FATAL) << "Sparse bank destroyed with live texture views";
    const auto result = Cleanup();
    if (result.operation)
      LOG(FATAL) << "Terminal sparse cleanup failure: " << result.operation
                 << " code " << result.code;
  }
  template <typename T>
  void Populate(const SourcePlan& plan,
                const SourceLayout& l,
                CUmemGenericAllocationHandle& pool,
                std::vector<cudaArray_t>& arrays) {
    CUmemGenericAllocationHandle created_pool = 0;
    const auto pool_status = SparseOperation(test_failure, "cuMemCreate", [&] {
      return api.create(&created_pool, l.pool, &property_, 0);
    });
    if (pool_status == CUDA_SUCCESS)
      pool = created_pool;
    else if (created_pool != 0)
      throw SparseCallFailure("unresolved cuMemCreate output",
                              static_cast<int>(pool_status),
                              SparseFailureKind::kFatal,
                              true);
    SparseDriver(pool_status, "cuMemCreate", true);
    const auto channel = cudaCreateChannelDesc<T>();
    std::vector<CUarrayMapInfo> pending;
    pending.reserve(plan.slots);
    for (const auto* p : plan.problems) {
      cudaArray_t array = nullptr;
      SparseAllocation(
          "cudaMalloc3DArray source",
          [&] {
            const auto status = cudaMalloc3DArray(
                &array,
                &channel,
                make_cudaExtent(l.width, l.height, p->src_image_idxs.size()),
                cudaArrayLayered | cudaArraySparse);
            if (status != cudaSuccess && array != nullptr)
              throw SparseCallFailure(
                  "unresolved cudaMalloc3DArray source output",
                  static_cast<int>(status),
                  SparseFailureKind::kFatal);
            return status;
          },
          test_failure);
      arrays.push_back(array);
      for (size_t slot = 0; slot < p->src_image_idxs.size(); ++slot) {
        const size_t index =
            std::find(
                plan.ids.begin(), plan.ids.end(), p->src_image_idxs[slot]) -
            plan.ids.begin();
        CUarrayMapInfo map = {};
        map.resourceType = CU_RESOURCE_TYPE_ARRAY;
        map.resource.array = reinterpret_cast<CUarray>(array);
        map.subresourceType = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
        map.subresource.sparseLevel.layer = slot;
        map.subresource.sparseLevel.extentWidth = l.width;
        map.subresource.sparseLevel.extentHeight = l.height;
        map.subresource.sparseLevel.extentDepth = 1;
        map.memOperationType = CU_MEM_OPERATION_TYPE_MAP;
        map.memHandleType = CU_MEM_HANDLE_TYPE_GENERIC;
        map.memHandle.memHandle = pool;
        map.offset = Product(index, l.stride);
        map.deviceBitMask = 1u << device;
        pending.push_back(map);
      }
    }
    THROW_CHECK_LE(pending.size(), std::numeric_limits<unsigned>::max());
    THROW_CHECK_GE(maps.capacity() - maps.size(), pending.size());
    THROW_CHECK_LT(map_batches.size(), map_batches.capacity());
    int injected = 0;
    const bool injected_map =
        InjectSparseFailure(test_failure, "cuMemMapArrayAsync", &injected);
    uncertain_map_status = injected_map
                               ? static_cast<CUresult>(injected)
                               : api.map(pending.data(),
                                         static_cast<unsigned>(pending.size()),
                                         reinterpret_cast<CUstream>(stream));
    if (injected_map) {
      // No submission occurred at this injected boundary. Do not leak test
      // resources or pretend this proves rollback after a real async failure.
      const auto status = uncertain_map_status;
      uncertain_map_status = CUDA_SUCCESS;
      SparseDriver(status, "cuMemMapArrayAsync");
    }
    SparseDriver(uncertain_map_status, "cuMemMapArrayAsync");
    // Capacity was reserved before submission; recording cannot allocate.
    maps.insert(maps.end(), pending.begin(), pending.end());
    map_batches.push_back(pending.size());
    SparseRuntime(
        SparseOperation(test_failure,
                        "cudaStreamSynchronize map",
                        [&] { return cudaStreamSynchronize(stream); }),
        "cudaStreamSynchronize map");
    // All aliases exist before one upload per canonical source. Fill the
    // complete virtual plane, including original stack padding and
    // tile-rounding tails.
    int staging_failure = 0;
    if (InjectSparseFailure(
            test_failure, "host source staging", &staging_failure))
      throw std::bad_alloc();
    std::vector<T> pixels(Product(l.width, l.height));
    for (size_t index = 0; index < plan.ids.size(); ++index) {
      std::fill(pixels.begin(), pixels.end(), T(0));
      const auto* p = plan.providers[index];
      const int id = plan.ids[index];
      const auto& image = p->images->at(id);
      const T* source;
      if constexpr (sizeof(T) == 1)
        source = p->InputBitmap(id).RowMajorData().data();
      else
        source = p->InputDepthMap(id).GetPtr();
      for (size_t row = 0; row < image.GetHeight(); ++row)
        std::memcpy(pixels.data() + row * l.width,
                    source + row * image.GetWidth(),
                    image.GetWidth() * sizeof(T));
      const size_t view =
          std::find(plan.problems.begin(), plan.problems.end(), p) -
          plan.problems.begin();
      const size_t slot =
          std::find(p->src_image_idxs.begin(), p->src_image_idxs.end(), id) -
          p->src_image_idxs.begin();
      cudaMemcpy3DParms copy = {};
      copy.srcPtr = make_cudaPitchedPtr(
          pixels.data(), l.width * sizeof(T), l.width * sizeof(T), l.height);
      copy.dstArray = arrays[view];
      copy.dstPos = make_cudaPos(0, 0, slot);
      copy.extent = make_cudaExtent(l.width, l.height, 1);
      copy.kind = cudaMemcpyHostToDevice;
      SparseRuntime(
          SparseOperation(test_failure,
                          "cudaMemcpy3DAsync source",
                          [&] { return cudaMemcpy3DAsync(&copy, stream); }),
          "cudaMemcpy3DAsync source");
      SparseRuntime(cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize upload");
    }
  }
  CUmemAllocationProp property_ = {};
#endif
};

PatchMatchSparseSources::PatchMatchSparseSources(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PatchMatchSparseSources::~PatchMatchSparseSources() = default;
void PatchMatchSparseSources::Close() const {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  std::lock_guard<std::recursive_mutex> lock(impl_->views_mutex);
  if (impl_->active_views.load() != 0)
    throw SparseCallFailure("sparse bank still has live texture views",
                            0,
                            SparseFailureKind::kFatal);
  impl_->Close();
#endif
}
void PatchMatchSparseSources::AbandonAfterTerminalViewFailure(
    const char* operation, int code) const {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  std::lock_guard<std::recursive_mutex> lock(impl_->views_mutex);
  impl_->abandoned = true;
  impl_->cleanup_failure = {operation, code, false};
#endif
}
bool PatchMatchSparseSources::HasConsistentInputs(
    const std::vector<const PatchMatch::Problem*>& problems, bool geometric) {
  return Plan(problems, geometric).width != 0;
}
bool PatchMatchSparseSources::IsSupported() {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  SourceSupport support;
  return support.Query();
#else
  return false;
#endif
}
size_t PatchMatchSparseSources::Budget(
    const std::vector<const PatchMatch::Problem*>& problems,
    bool geometric,
    const SparseSourceTestFailure* failure) {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  try {
    const auto plan = Plan(problems, geometric);
    SourceSupport support;
    if (!plan.width || !support.Query(failure)) return 0;
    const auto image = support.Layout<uint8_t>(plan, failure);
    const auto depth =
        geometric ? support.Layout<float>(plan, failure) : SourceLayout{};
    return SourcePoolBudget(plan, image, depth, geometric);
  } catch (const SparseCallFailure& error) {
    if (error.kind == SparseFailureKind::kUnsupported) return 0;
    if (error.kind == SparseFailureKind::kResource)
      throw SparseSourceResourceRefused(error.operation, error.code);
    throw;
  } catch (const std::bad_alloc&) {
    throw SparseSourceResourceRefused("host sparse planning", 0);
  }
#else
  return 0;
#endif
}
std::shared_ptr<const PatchMatchSparseSources> PatchMatchSparseSources::Create(
    const std::vector<const PatchMatch::Problem*>& problems,
    bool geometric,
    const SparseSourceTestFailure* failure) {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  std::unique_ptr<Impl> impl;
  try {
    // Allocate the shared owner/control block before owning any GPU resources.
    auto owner = std::shared_ptr<PatchMatchSparseSources>(
        new PatchMatchSparseSources(nullptr));
    const auto plan = Plan(problems, geometric);
    SourceSupport support;
    if (!plan.width || !support.Query(failure)) return {};
    const auto image = support.Layout<uint8_t>(plan, failure);
    const auto depth =
        geometric ? support.Layout<float>(plan, failure) : SourceLayout{};
    if (!SourcePoolBudget(plan, image, depth, geometric)) return {};
    impl = std::make_unique<Impl>();
    impl->device = support.device;
    impl->test_failure = failure;
    impl->api = support.api;
    impl->property_ = support.property;
    impl->geometric = geometric;
    impl->width = plan.width;
    impl->height = plan.height;
    for (const auto* p : problems) {
      Impl::Binding binding{p->ref_image_idx,
                            p->src_image_idxs,
                            p->images,
                            geometric ? p->depth_maps : nullptr,
                            {},
                            {}};
      for (int id : p->src_image_idxs) {
        binding.bitmap_owners.push_back(BitmapOwner(*p, id));
        if (geometric) binding.depth_owners.push_back(DepthOwner(*p, id));
      }
      impl->bindings.push_back(std::move(binding));
    }
    impl->images.reserve(problems.size());
    impl->depths.reserve(problems.size());
    impl->maps.reserve(Product(plan.slots, geometric ? 2 : 1));
    impl->map_batches.reserve(geometric ? 2 : 1);
    cudaStream_t created_stream = nullptr;
    SparseAllocation(
        "cudaStreamCreateWithFlags",
        [&] {
          const auto status =
              cudaStreamCreateWithFlags(&created_stream, cudaStreamNonBlocking);
          if (status != cudaSuccess && created_stream != nullptr)
            throw SparseCallFailure(
                "unresolved cudaStreamCreateWithFlags output",
                static_cast<int>(status),
                SparseFailureKind::kFatal);
          return status;
        },
        failure);
    impl->stream = created_stream;
    impl->Populate<uint8_t>(plan, image, impl->image_pool, impl->images);
    if (geometric)
      impl->Populate<float>(plan, depth, impl->depth_pool, impl->depths);
    LOG(INFO) << "Sparse source pools: references=" << problems.size()
              << ", source_slots=" << plan.slots
              << ", unique_sources=" << plan.ids.size()
              << ", physical_bytes=" << Sum(image.pool, depth.pool);
    impl->test_failure = nullptr;
    owner->impl_ = std::move(impl);
    return owner;
  } catch (const SparseCallFailure& error) {
    if (impl) impl->Close();
    if (error.kind == SparseFailureKind::kUnsupported) return {};
    if (error.kind == SparseFailureKind::kResource)
      throw SparseSourceResourceRefused(error.operation, error.code);
    throw;
  } catch (const std::bad_alloc&) {
    if (impl) impl->Close();
    throw SparseSourceResourceRefused("host sparse setup", 0);
  }
#else
  return {};
#endif
}
bool PatchMatchSparseSources::Matches(const PatchMatch::Problem& p,
                                      bool geometric) const {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  std::lock_guard<std::recursive_mutex> lock(impl_->views_mutex);
  if (impl_->released || impl_->abandoned) return false;
#endif
  if (geometric != impl_->geometric) return false;
  for (const auto& binding : impl_->bindings)
    if (binding.reference == p.ref_image_idx) {
      if (binding.sources != p.src_image_idxs || binding.images != p.images ||
          (geometric && binding.depths != p.depth_maps))
        return false;
      for (size_t slot = 0; slot < binding.sources.size(); ++slot) {
        const int id = binding.sources[slot];
        if (!SameOwnedPayload(binding.bitmap_owners[slot],
                              BitmapOwner(p, id)) ||
            (geometric &&
             !SameOwnedPayload(binding.depth_owners[slot], DepthOwner(p, id))))
          return false;
      }
      return true;
    }
  return false;
}

template <typename T>
struct PatchMatchSourceArray<T>::Impl {
  int device = -1;
  size_t width = 0, height = 0, depth = 0;
  cudaTextureObject_t texture = 0;
  std::unique_ptr<CudaArrayLayeredTexture<T>> dense;
  std::shared_ptr<const PatchMatchSparseSources> bank;
  bool abandoned = false;
  bool registered_view = false;
  const char* failure_operation = nullptr;
  int failure_code = 0;
  void Close() {
    if (abandoned)
      throw SparseCallFailure(
          failure_operation, failure_code, SparseFailureKind::kFatal);
    if (device < 0) return;
    try {
      SourceDeviceScope scope(device);
      if (texture) {
        SparseRuntime(cudaDestroyTextureObject(texture),
                      "cudaDestroyTextureObject");
        texture = 0;
      }
      if (registered_view) {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
        bank->impl_->active_views.fetch_sub(1);
#endif
        registered_view = false;
      }
      dense.reset();
      bank.reset();  // View is destroyed before its bank may unmap storage.
      scope.Restore();
      device = -1;
    } catch (const SparseCallFailure& error) {
      abandoned = true;
      failure_operation = error.operation;
      failure_code = error.code;
      if (bank)
        bank->AbandonAfterTerminalViewFailure(error.operation, error.code);
      throw;
    }
  }
  ~Impl() {
    if (abandoned) return;
    try {
      Close();
    } catch (const SparseCallFailure& error) {
      LOG(FATAL) << "Terminal source-view cleanup failure: " << error.what();
    }
  }
};
template <typename T>
PatchMatchSourceArray<T>::PatchMatchSourceArray(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
template <typename T>
PatchMatchSourceArray<T>::~PatchMatchSourceArray() = default;
template <typename T>
void PatchMatchSourceArray<T>::Close() {
  impl_->Close();
}
template <typename T>
std::unique_ptr<PatchMatchSourceArray<T>>
PatchMatchSourceArray<T>::FromHostArray(
    const cudaTextureDesc& desc, size_t w, size_t h, size_t n, const T* data) {
  auto impl = std::make_unique<Impl>();
  CUDA_SAFE_CALL(cudaGetDevice(&impl->device));
  impl->dense = CudaArrayLayeredTexture<T>::FromHostArray(desc, w, h, n, data);
  return std::unique_ptr<PatchMatchSourceArray>(
      new PatchMatchSourceArray(std::move(impl)));
}
template <typename T>
std::unique_ptr<PatchMatchSourceArray<T>>
PatchMatchSourceArray<T>::FromHostSlices(const cudaTextureDesc& desc,
                                         size_t w,
                                         size_t h,
                                         const std::vector<HostSlice>& slices) {
  auto impl = std::make_unique<Impl>();
  CUDA_SAFE_CALL(cudaGetDevice(&impl->device));
  impl->dense = CudaArrayLayeredTexture<T>::FromHostSlices(desc, w, h, slices);
  return std::unique_ptr<PatchMatchSourceArray>(
      new PatchMatchSourceArray(std::move(impl)));
}
template <typename T>
std::unique_ptr<PatchMatchSourceArray<T>> PatchMatchSourceArray<T>::FromSparse(
    std::shared_ptr<const PatchMatchSparseSources> bank,
    int reference,
    const SparseSourceTestFailure* failure) {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  if (!bank) return {};
  auto& b = *bank->impl_;
  std::unique_lock<std::recursive_mutex> lock(b.views_mutex);
  if (b.released || b.abandoned) return {};
  auto found = std::find_if(
      b.bindings.begin(), b.bindings.end(), [reference](const auto& v) {
        return v.reference == reference;
      });
  if (found == b.bindings.end() || (sizeof(T) == 4 && !b.geometric)) return {};
  const size_t index = found - b.bindings.begin();
  SourceDeviceScope scope(b.device);
  std::unique_ptr<Impl> impl;
  try {
    impl = std::make_unique<Impl>();
    impl->device = b.device;
    impl->width = b.width;
    impl->height = b.height;
    impl->depth = found->sources.size();
    cudaResourceDesc resource = {};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array =
        sizeof(T) == 1 ? b.images[index] : b.depths[index];
    cudaTextureDesc desc = {};
    for (int axis = 0; axis < 3; ++axis)
      desc.addressMode[axis] = cudaAddressModeBorder;
    desc.filterMode =
        sizeof(T) == 1 ? cudaFilterModeLinear : cudaFilterModePoint;
    desc.readMode =
        sizeof(T) == 1 ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
    impl->bank = bank;
    cudaTextureObject_t created_texture = 0;
    SparseAllocation(
        "cudaCreateTextureObject sparse",
        [&] {
          const auto status = cudaCreateTextureObject(
              &created_texture, &resource, &desc, nullptr);
          if (status != cudaSuccess && created_texture != 0) {
            bank->AbandonAfterTerminalViewFailure(
                "unresolved cudaCreateTextureObject output",
                static_cast<int>(status));
            throw SparseCallFailure("unresolved cudaCreateTextureObject output",
                                    static_cast<int>(status),
                                    SparseFailureKind::kFatal);
          }
          return status;
        },
        failure);
    impl->texture = created_texture;
    b.active_views.fetch_add(1);
    impl->registered_view = true;
    auto owner = std::unique_ptr<PatchMatchSourceArray>(
        new PatchMatchSourceArray(std::move(impl)));
    scope.Restore();
    return owner;
  } catch (const SparseCallFailure& error) {
    if (impl) impl->Close();
    scope.Restore();
    lock.unlock();
    if (bank.use_count() == 1) bank->Close();
    if (error.kind != SparseFailureKind::kFatal)
      throw SparseSourceResourceRefused(error.operation, error.code);
    throw;
  } catch (const std::bad_alloc&) {
    if (impl) impl->Close();
    scope.Restore();
    lock.unlock();
    if (bank.use_count() == 1) bank->Close();
    throw SparseSourceResourceRefused("host sparse view", 0);
  }
#else
  return {};
#endif
}
template <typename T>
cudaTextureObject_t PatchMatchSourceArray<T>::GetObj() const {
  return impl_->dense ? impl_->dense->GetObj() : impl_->texture;
}
template <typename T>
size_t PatchMatchSourceArray<T>::GetWidth() const {
  return impl_->dense ? impl_->dense->GetWidth() : impl_->width;
}
template <typename T>
size_t PatchMatchSourceArray<T>::GetHeight() const {
  return impl_->dense ? impl_->dense->GetHeight() : impl_->height;
}
template <typename T>
size_t PatchMatchSourceArray<T>::GetDepth() const {
  return impl_->dense ? impl_->dense->GetDepth() : impl_->depth;
}
template <typename T>
bool PatchMatchSourceArray<T>::IsSparse() const {
  return !!impl_->bank;
}
template class PatchMatchSourceArray<uint8_t>;
template class PatchMatchSourceArray<float>;

}  // namespace mvs
}  // namespace colmap
