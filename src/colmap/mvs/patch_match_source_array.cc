// SPDX-License-Identifier: BSD-3-Clause
// Match the existing NVCC --default-stream per-thread dense helper semantics.
#if !defined(COLMAP_HIP_ENABLED) && !defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "colmap/mvs/patch_match_source_array.h"

#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#if !defined(COLMAP_HIP_ENABLED) && CUDART_VERSION >= 13020
#include <cuda.h>
#include <cudaTypedefs.h>
#define COLMAP_SPARSE_SOURCE_RUNTIME 1
#endif

namespace colmap {
namespace mvs {
namespace {
class SourceDeviceScope {
 public:
  explicit SourceDeviceScope(int device) {
    CHECK_EQ(cudaGetDevice(&previous_), cudaSuccess);
    changed_ = device != previous_;
    if (changed_) CHECK_EQ(cudaSetDevice(device), cudaSuccess);
  }
  ~SourceDeviceScope() {
    if (changed_) CHECK_EQ(cudaSetDevice(previous_), cudaSuccess);
  }

 private:
  int previous_;
  bool changed_;
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
      const auto status = cudaGetDriverEntryPointByVersion(
          name, pointer, version, cudaEnableLegacyStream, &query);
      if (status == cudaErrorNotSupported) return false;
      CUDA_SAFE_CALL(status);
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
      if (!image.GetWidth() || !image.GetHeight() ||
          !image.GetBitmap().IsGrey() ||
          image.GetWidth() != image.GetBitmap().Width() ||
          image.GetHeight() != image.GetBitmap().Height())
        return {};
      if (geometric && (size_t(id) >= p->depth_maps->size() ||
                        p->depth_maps->at(id).GetWidth() != image.GetWidth() ||
                        p->depth_maps->at(id).GetHeight() != image.GetHeight()))
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
        if (old.GetPath() != image.GetPath() ||
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
  bool Query() {
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_property, device));
    if (device < 0 || device >= 32 ||
        !device_property.sparseCudaArraySupported || !api.Load())
      return false;
    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device;
    property.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
    THROW_CHECK_EQ(
        api.granularity(
            &granularity, &property, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        CUDA_SUCCESS);
    return granularity != 0;
  }
  template <typename T>
  SourceLayout Layout(const SourcePlan& plan) const {
    if (!plan.width || !plan.height) return {};
    cudaArray_t probe = nullptr;
    const auto channel = cudaCreateChannelDesc<T>();
    CUDA_SAFE_CALL(cudaMalloc3DArray(&probe,
                                     &channel,
                                     make_cudaExtent(1024, 1024, 1),
                                     cudaArrayLayered | cudaArraySparse));
    cudaArraySparseProperties sparse = {};
    const auto status = cudaArrayGetSparseProperties(&sparse, probe);
    CUDA_SAFE_CALL(cudaFreeArray(probe));
    CUDA_SAFE_CALL(status);
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
  };
  std::vector<Binding> bindings;
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  SourceApi api;
  cudaStream_t stream = nullptr;
  std::vector<cudaArray_t> images, depths;
  std::vector<CUarrayMapInfo> maps;
  std::vector<size_t> map_batches;
  CUmemGenericAllocationHandle image_pool = 0, depth_pool = 0;
  ~Impl() {
    if (device < 0) return;
    SourceDeviceScope scope(device);
    if (stream) CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    size_t end = maps.size();
    for (auto batch = map_batches.rbegin(); batch != map_batches.rend();
         ++batch) {
      auto* records = maps.data() + end - *batch;
      for (size_t i = 0; i < *batch; ++i) {
        records[i].memOperationType = CU_MEM_OPERATION_TYPE_UNMAP;
        records[i].memHandle.memHandle = 0;
      }
      CHECK_EQ(api.map(records,
                       static_cast<unsigned>(*batch),
                       reinterpret_cast<CUstream>(stream)),
               CUDA_SUCCESS);
      end -= *batch;
    }
    if (stream) CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (auto array : depths) CHECK_EQ(cudaFreeArray(array), cudaSuccess);
    for (auto array : images) CHECK_EQ(cudaFreeArray(array), cudaSuccess);
    if (depth_pool) CHECK_EQ(api.release(depth_pool), CUDA_SUCCESS);
    if (image_pool) CHECK_EQ(api.release(image_pool), CUDA_SUCCESS);
    if (stream) CHECK_EQ(cudaStreamDestroy(stream), cudaSuccess);
  }
  template <typename T>
  void Populate(const SourcePlan& plan,
                const SourceLayout& l,
                CUmemGenericAllocationHandle& pool,
                std::vector<cudaArray_t>& arrays) {
    THROW_CHECK_EQ(api.create(&pool, l.pool, &property_, 0), CUDA_SUCCESS);
    const auto channel = cudaCreateChannelDesc<T>();
    std::vector<CUarrayMapInfo> pending;
    pending.reserve(plan.slots);
    for (const auto* p : plan.problems) {
      cudaArray_t array = nullptr;
      CUDA_SAFE_CALL(cudaMalloc3DArray(
          &array,
          &channel,
          make_cudaExtent(l.width, l.height, p->src_image_idxs.size()),
          cudaArrayLayered | cudaArraySparse));
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
    THROW_CHECK_EQ(api.map(pending.data(),
                           static_cast<unsigned>(pending.size()),
                           reinterpret_cast<CUstream>(stream)),
                   CUDA_SUCCESS);
    // Capacity was reserved before submission; recording cannot allocate.
    maps.insert(maps.end(), pending.begin(), pending.end());
    map_batches.push_back(pending.size());
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    // All aliases exist before one upload per canonical source. Fill the
    // complete virtual plane, including original stack padding and
    // tile-rounding tails.
    std::vector<T> pixels(Product(l.width, l.height));
    for (size_t index = 0; index < plan.ids.size(); ++index) {
      std::fill(pixels.begin(), pixels.end(), T(0));
      const auto* p = plan.providers[index];
      const int id = plan.ids[index];
      const auto& image = p->images->at(id);
      const T* source;
      if constexpr (sizeof(T) == 1)
        source = image.GetBitmap().RowMajorData().data();
      else
        source = p->depth_maps->at(id).GetPtr();
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
      CUDA_SAFE_CALL(cudaMemcpy3DAsync(&copy, stream));
      CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    }
  }
  CUmemAllocationProp property_ = {};
#endif
};

PatchMatchSparseSources::PatchMatchSparseSources(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PatchMatchSparseSources::~PatchMatchSparseSources() = default;
size_t PatchMatchSparseSources::Budget(
    const std::vector<const PatchMatch::Problem*>& problems, bool geometric) {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  const auto plan = Plan(problems, geometric);
  SourceSupport support;
  if (!plan.width || !support.Query()) return 0;
  const auto image = support.Layout<uint8_t>(plan);
  const auto depth = geometric ? support.Layout<float>(plan) : SourceLayout{};
  return SourcePoolBudget(plan, image, depth, geometric);
#else
  return 0;
#endif
}
std::shared_ptr<const PatchMatchSparseSources> PatchMatchSparseSources::Create(
    const std::vector<const PatchMatch::Problem*>& problems, bool geometric) {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  const auto plan = Plan(problems, geometric);
  SourceSupport support;
  if (!plan.width || !support.Query()) return {};
  const auto image = support.Layout<uint8_t>(plan);
  const auto depth = geometric ? support.Layout<float>(plan) : SourceLayout{};
  if (!SourcePoolBudget(plan, image, depth, geometric)) return {};
  auto impl = std::make_unique<Impl>();
  impl->device = support.device;
  impl->api = support.api;
  impl->property_ = support.property;
  impl->geometric = geometric;
  impl->width = plan.width;
  impl->height = plan.height;
  for (const auto* p : problems)
    impl->bindings.push_back({p->ref_image_idx,
                              p->src_image_idxs,
                              p->images,
                              geometric ? p->depth_maps : nullptr});
  impl->images.reserve(problems.size());
  impl->depths.reserve(problems.size());
  impl->maps.reserve(Product(plan.slots, geometric ? 2 : 1));
  impl->map_batches.reserve(geometric ? 2 : 1);
  CUDA_SAFE_CALL(
      cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking));
  impl->Populate<uint8_t>(plan, image, impl->image_pool, impl->images);
  if (geometric)
    impl->Populate<float>(plan, depth, impl->depth_pool, impl->depths);
  LOG(INFO) << "Sparse source pools: references=" << problems.size()
            << ", source_slots=" << plan.slots
            << ", unique_sources=" << plan.ids.size()
            << ", physical_bytes=" << Sum(image.pool, depth.pool);
  return std::shared_ptr<const PatchMatchSparseSources>(
      new PatchMatchSparseSources(std::move(impl)));
#else
  return {};
#endif
}
bool PatchMatchSparseSources::Matches(const PatchMatch::Problem& p,
                                      bool geometric) const {
  if (geometric != impl_->geometric) return false;
  for (const auto& binding : impl_->bindings)
    if (binding.reference == p.ref_image_idx)
      return binding.sources == p.src_image_idxs &&
             binding.images == p.images &&
             (!geometric || binding.depths == p.depth_maps);
  return false;
}

template <typename T>
struct PatchMatchSourceArray<T>::Impl {
  int device = -1;
  size_t width = 0, height = 0, depth = 0;
  cudaTextureObject_t texture = 0;
  std::unique_ptr<CudaArrayLayeredTexture<T>> dense;
  std::shared_ptr<const PatchMatchSparseSources> bank;
  ~Impl() {
    if (device < 0) return;
    SourceDeviceScope scope(device);
    if (texture) CHECK_EQ(cudaDestroyTextureObject(texture), cudaSuccess);
    dense.reset();
    bank.reset();  // Last view dies before bank unmaps/releases physical
                   // storage.
  }
};
template <typename T>
PatchMatchSourceArray<T>::PatchMatchSourceArray(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
template <typename T>
PatchMatchSourceArray<T>::~PatchMatchSourceArray() = default;
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
    std::shared_ptr<const PatchMatchSparseSources> bank, int reference) {
#ifdef COLMAP_SPARSE_SOURCE_RUNTIME
  if (!bank) return {};
  const auto& b = *bank->impl_;
  auto found = std::find_if(
      b.bindings.begin(), b.bindings.end(), [reference](const auto& v) {
        return v.reference == reference;
      });
  if (found == b.bindings.end() || (sizeof(T) == 4 && !b.geometric)) return {};
  const size_t index = found - b.bindings.begin();
  SourceDeviceScope scope(b.device);
  auto impl = std::make_unique<Impl>();
  impl->device = b.device;
  impl->width = b.width;
  impl->height = b.height;
  impl->depth = found->sources.size();
  cudaResourceDesc resource = {};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = sizeof(T) == 1 ? b.images[index] : b.depths[index];
  cudaTextureDesc desc = {};
  for (int axis = 0; axis < 3; ++axis)
    desc.addressMode[axis] = cudaAddressModeBorder;
  desc.filterMode = sizeof(T) == 1 ? cudaFilterModeLinear : cudaFilterModePoint;
  desc.readMode =
      sizeof(T) == 1 ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
  CUDA_SAFE_CALL(
      cudaCreateTextureObject(&impl->texture, &resource, &desc, nullptr));
  impl->bank = std::move(bank);
  return std::unique_ptr<PatchMatchSourceArray>(
      new PatchMatchSourceArray(std::move(impl)));
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
