// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/feature/aliked_native/custom_ops.h"

#include "colmap/feature/aliked_native/test_models.h"
#include "colmap/util/file.h"
#include "colmap/util/onnx.h"
#include "colmap/util/testing.h"

#include <cstdlib>
#include <filesystem>
#include <future>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <onnxruntime_lite_custom_op.h>

std::unique_ptr<Ort::Custom::OrtLiteCustomOp> CreateFusedConcatCustomOp();
std::unique_ptr<Ort::Custom::OrtLiteCustomOp>
CreateNormalizedGridSampleCustomOp();

namespace colmap {
TEST(AlikedNativeOps, SessionDefaultsStayUnchanged) {
  const ONNXSessionConfig config;
  EXPECT_FALSE(config.require_cuda);
  EXPECT_FALSE(config.register_custom_ops);
  EXPECT_TRUE(config.log_initialization_errors);
  EXPECT_FALSE(config.model_data);
}
TEST(AlikedNativeOps, DescriptorsUseCheckedStatusAndNoShapeCallback) {
  auto concat = CreateFusedConcatCustomOp();
  auto sample = CreateNormalizedGridSampleCustomOp();
  for (const OrtCustomOp* op :
       {static_cast<const OrtCustomOp*>(concat.get()),
        static_cast<const OrtCustomOp*>(sample.get())}) {
    EXPECT_EQ(op->CreateKernel, nullptr);
    EXPECT_NE(op->CreateKernelV2, nullptr);
    EXPECT_NE(op->KernelComputeV2, nullptr);
    EXPECT_EQ(op->InferOutputShapeFn, nullptr);
  }
}
TEST(AlikedNativeOps, ParallelDomainRegistration) {
  auto make_owner = [] {
    std::shared_ptr<void> owner;
    Ort::SessionOptions options;
    owner = RegisterAlikedNativeOps(options);
    return static_cast<bool>(owner);
  };
  auto a = std::async(std::launch::async, make_owner);
  auto b = std::async(std::launch::async, make_owner);
  EXPECT_TRUE(a.get());
  EXPECT_TRUE(b.get());
}
TEST(AlikedNativeOps, SeparateOwnersRetainSeparateDomains) {
  struct OwnedOptions {
    // Options are destroyed before the retained domain owner.
    std::shared_ptr<void> owner;
    Ort::SessionOptions options;
  };
  OwnedOptions first;
  OwnedOptions second;
  first.owner = RegisterAlikedNativeOps(first.options);
  second.owner = RegisterAlikedNativeOps(second.options);
  EXPECT_TRUE(first.owner);
  EXPECT_TRUE(second.owner);
  EXPECT_NE(first.owner.get(), second.owner.get());
}

namespace {
std::filesystem::path Fixture(const char* name) {
  using namespace aliked_test;
  const struct {
    const char* name;
    const unsigned char* data;
    size_t size;
  } models[] = {
      {"concat_valid.onnx", kConcatValid, sizeof(kConcatValid)},
      {"concat_bad_channels.onnx",
       kConcatBadChannels,
       sizeof(kConcatBadChannels)},
      {"sample_valid.onnx", kSampleValid, sizeof(kSampleValid)},
      {"sample_bad_norm.onnx", kSampleBadNorm, sizeof(kSampleBadNorm)}};
  for (const auto& model : models) {
    if (std::string_view(name) == model.name) {
      const auto path = CreateTestDir() / name;
      WriteBinaryBlob(path,
                      {reinterpret_cast<const char*>(model.data), model.size});
      return path;
    }
  }
  throw std::runtime_error("Unknown synthetic native-op fixture");
}
ONNXSessionConfig NativeConfig() {
  ONNXSessionConfig config;
  config.require_cuda = true;
  config.register_custom_ops = RegisterAlikedNativeOps;
  return config;
}
bool HasQualifiedDevice() {
  int device = -1;
  cudaDeviceProp prop{};
  int runtime = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess ||
      cudaRuntimeGetVersion(&runtime) != cudaSuccess || prop.major != 8 ||
      prop.minor != 6 || runtime < 13020 || runtime >= 13030) {
    return false;
  }
  return true;
}
std::vector<Ort::Value> ConcatInputs(int b_channels) {
  static std::vector<float> a(32 * 32 * 32, 1.0f);
  static std::vector<float> b(32 * 16 * 16, 2.0f);
  static std::vector<float> c(32 * 4 * 4, 3.0f);
  static std::vector<float> d(32, 4.0f);
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(CreateONNXTensor(a, {1, 32, 32, 32}));
  inputs.emplace_back(CreateONNXTensor(
      b.data(), b_channels * 16 * 16, {1, b_channels, 16, 16}));
  inputs.emplace_back(CreateONNXTensor(c, {1, 32, 4, 4}));
  inputs.emplace_back(CreateONNXTensor(d, {1, 32, 1, 1}));
  return inputs;
}
}  // namespace

TEST(AlikedNativeOps, RequiresCudaBeforeLoadingCustomGraph) {
  EXPECT_ANY_THROW(ONNXModel(Fixture("concat_valid.onnx").string(),
                             1,
                             false,
                             "0",
                             false,
                             NativeConfig()));
}
TEST(AlikedNativeOps, SyntheticConcatSuccessAndRuntimeRefusal) {
  if (!HasQualifiedDevice()) GTEST_SKIP() << "Requires CUDA 13.2/sm86 device";
  ONNXModel valid(Fixture("concat_valid.onnx").string(),
                  1,
                  true,
                  "0",
                  false,
                  NativeConfig());
  auto outputs = valid.Run(ConcatInputs(32));
  ASSERT_EQ(outputs.size(), 1);
  const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  EXPECT_EQ(shape, (std::vector<int64_t>{1, 128, 32, 32}));
  ASSERT_EQ(outputs[0].GetTensorMemoryInfo().GetDeviceType(),
            OrtMemoryInfoDeviceType_CPU);
  const float* values = outputs[0].GetTensorData<float>();
  for (size_t channel = 0; channel < 128; ++channel)
    for (size_t pixel = 0; pixel < 32 * 32; ++pixel)
      EXPECT_FLOAT_EQ(values[channel * 32 * 32 + pixel],
                      static_cast<float>(channel / 32 + 1));
  ONNXModel bad(Fixture("concat_bad_channels.onnx").string(),
                1,
                true,
                "0",
                false,
                NativeConfig());
  EXPECT_ANY_THROW(bad.Run(ConcatInputs(31)));
}
TEST(AlikedNativeOps, SyntheticSampleAndEmptyGrid) {
  if (!HasQualifiedDevice()) GTEST_SKIP() << "Requires CUDA 13.2/sm86 device";
  ONNXModel model(Fixture("sample_valid.onnx").string(),
                  1,
                  true,
                  "0",
                  false,
                  NativeConfig());
  std::vector<float> raw(128 * 4 * 4, 2.0f), norm(4 * 4, 2.0f), grid(2, 0.0f);
  for (int count : {1, 0}) {
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(CreateONNXTensor(raw, {1, 128, 4, 4}));
    inputs.emplace_back(CreateONNXTensor(norm, {1, 1, 4, 4}));
    inputs.emplace_back(
        CreateONNXTensor(grid.data(), count * 2, {1, count, 1, 2}));
    auto output = model.Run(inputs);
    ASSERT_EQ(output.size(), 1);
    EXPECT_EQ(output[0].GetTensorTypeAndShapeInfo().GetShape(),
              (std::vector<int64_t>{1, 128, count, 1}));
    if (count) {
      ASSERT_EQ(output[0].GetTensorMemoryInfo().GetDeviceType(),
                OrtMemoryInfoDeviceType_CPU);
      const float* values = output[0].GetTensorData<float>();
      for (int channel = 0; channel < 128; ++channel)
        EXPECT_FLOAT_EQ(values[channel], 1.0f);
    }
  }
  ONNXModel bad(Fixture("sample_bad_norm.onnx").string(),
                1,
                true,
                "0",
                false,
                NativeConfig());
  std::vector<float> bad_norm(2 * 4 * 4, 2.0f);
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(CreateONNXTensor(raw, {1, 128, 4, 4}));
  inputs.emplace_back(CreateONNXTensor(bad_norm, {1, 2, 4, 4}));
  inputs.emplace_back(CreateONNXTensor(grid, {1, 1, 1, 2}));
  EXPECT_ANY_THROW(bad.Run(inputs));
}
}  // namespace colmap
