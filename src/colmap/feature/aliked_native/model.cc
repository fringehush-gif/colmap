// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/aliked_native/model.h"

#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/onnx.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include <cuda_runtime_api.h>

namespace colmap {
namespace {

#ifdef COLMAP_DOWNLOAD_ENABLED
struct PinnedModelRead {
  std::shared_ptr<const std::vector<char>> data;
  std::string reason;
};

PinnedModelRead ReadPinnedModel(const std::string& path,
                                const size_t expected_bytes,
                                const std::string_view expected_sha256,
                                const bool missing_is_unavailable) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (missing_is_unavailable && !error && !exists) {
      return {nullptr, "native model file is missing"};
    }
    throw std::runtime_error("ALIKED model snapshot could not be opened");
  }
  const auto size = input.tellg();
  if (size < 0) {
    throw std::runtime_error("ALIKED model snapshot size could not be read");
  }
  if (size != static_cast<std::streamoff>(expected_bytes)) {
    return {nullptr, "native model pair has an unsupported size"};
  }
  auto data = std::make_shared<std::vector<char>>(expected_bytes);
  input.seekg(0);
  if (!input.read(data->data(), static_cast<std::streamsize>(data->size())) ||
      input.peek() != std::ifstream::traits_type::eof() || input.bad()) {
    throw std::runtime_error("Native ALIKED model snapshot read failed");
  }
  if (ComputeSHA256({data->data(), data->size()}) != expected_sha256) {
    return {nullptr, "native model pair checksum differs"};
  }
  return {std::move(data), {}};
}
#endif

}  // namespace

AlikedNativeModelData PrepareAlikedNativeModelData(
    const std::string& original_model_path,
    const std::string& native_model_path,
    const std::string& gpu_index) {
  AlikedNativeModelData data;
#ifdef COLMAP_DOWNLOAD_ENABLED
  if (std::string_view(OrtGetApiBase()->GetVersionString()) != "1.31.1") {
    data.fallback_reason = "native operators require ONNX Runtime 1.31.1";
    return data;
  }
  const auto devices = CSVToVector<int>(gpu_index);
  if (devices.size() != 1) {
    data.fallback_reason = "native operators require one CUDA device";
    return data;
  }
  const int device = devices[0] < 0 ? 0 : devices[0];
  int runtime = 0;
  cudaDeviceProp properties{};
  const auto runtime_status = cudaRuntimeGetVersion(&runtime);
  if (runtime_status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA runtime query failed: ") +
                             cudaGetErrorString(runtime_status));
  }
  if (runtime < 13020 || runtime >= 13030) {
    data.fallback_reason =
        "native operators require CUDA 13.2 on an sm86 device";
    return data;
  }
  const auto device_status = cudaGetDeviceProperties(&properties, device);
  if (device_status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA device query failed: ") +
                             cudaGetErrorString(device_status));
  }
  if (properties.major != 8 || properties.minor != 6) {
    data.fallback_reason = "native operators require an sm86 device";
    return data;
  }
  if (IsURI(native_model_path)) {
    data.fallback_reason = "native ALIKED graph must be a local file";
    return data;
  }
  auto native = ReadPinnedModel(
      native_model_path,
      2995274,
      "66c1b167d88c5e5e6b3b55683f9dc2159362dc1c228d29a790ef027f057aa300",
      true);
  if (!native.data) {
    data.fallback_reason = std::move(native.reason);
    return data;
  }
  auto original = ReadPinnedModel(
      ResolveONNXModelPath(original_model_path),
      2997054,
      "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
      false);
  if (!original.data) {
    data.fallback_reason = std::move(original.reason);
    return data;
  }
  data.original = std::move(original.data);
  data.native = std::move(native.data);
#else
  data.fallback_reason = "native model checksum support is not built";
#endif
  return data;
}

}  // namespace colmap
