// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/feature/aliked_native/custom_ops.h"

#include <onnxruntime_lite_custom_op.h>
#if ORT_API_VERSION != 31
#error "ALIKED native ops require ONNX Runtime API 31"
#endif
#include <memory>

std::unique_ptr<Ort::Custom::OrtLiteCustomOp> CreateFusedConcatCustomOp();
std::unique_ptr<Ort::Custom::OrtLiteCustomOp>
CreateNormalizedGridSampleCustomOp();

namespace colmap {
namespace {
struct NativeDomain {
  // Declaration order makes the domain die before its descriptors.
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> concat =
      CreateFusedConcatCustomOp();
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> sample =
      CreateNormalizedGridSampleCustomOp();
  Ort::CustomOpDomain domain{"com.colmap"};
  NativeDomain() {
    domain.Add(concat.get());
    domain.Add(sample.get());
  }
};
}  // namespace

std::shared_ptr<void> RegisterAlikedNativeOps(Ort::SessionOptions& options) {
  auto owner = std::make_shared<NativeDomain>();
  options.Add(owner->domain);
  return owner;
}
}  // namespace colmap
