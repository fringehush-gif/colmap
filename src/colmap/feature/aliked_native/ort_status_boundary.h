// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <exception>
#include <mutex>
#include <stdexcept>

#include "onnxruntime_c_api.h"

namespace colmap::aliked_native {

// Decorate a Lite descriptor, without replacing its identity or kernel storage.
// Apply once during its thread-safe static initialization, before registration.
// Each descriptor needs a distinct Tag because these callbacks carry no
// closure.
template <typename Tag>
struct StatusBoundary {
  static void Apply(OrtCustomOp& op, const OrtApi& api) {
    if (!op.CreateKernel || op.CreateKernelV2 || !op.KernelComputeV2) {
      throw std::invalid_argument("Unsupported custom-op callback layout");
    }
    std::call_once(initialized_, [&] {
      api_ = &api;
      create_ = op.CreateKernel;
      compute_ = op.KernelComputeV2;
    });
    if (api_ != &api || create_ != op.CreateKernel ||
        compute_ != op.KernelComputeV2) {
      throw std::invalid_argument("Mismatched custom-op callback or ORT API");
    }
    op.CreateKernel = nullptr;
    op.CreateKernelV2 = Create;
    op.KernelComputeV2 = Compute;
    // ORT 1.31.1 does not consume/release a returned shape-callback status.
    op.InferOutputShapeFn = nullptr;
  }

 private:
  static OrtStatusPtr ORT_API_CALL Create(const OrtCustomOp* op,
                                          const OrtApi* api,
                                          const OrtKernelInfo* info,
                                          void** kernel) noexcept {
    *kernel = nullptr;
    try {
      *kernel = create_(op, api, info);
      if (!*kernel)
        return api_->CreateStatus(ORT_FAIL,
                                  "Custom-op kernel creation returned null");
      return nullptr;
    } catch (const std::exception& error) {
      return api_->CreateStatus(ORT_FAIL, error.what());
    } catch (...) {
      return api_->CreateStatus(ORT_FAIL,
                                "Unknown custom-op creation exception");
    }
  }

  static OrtStatusPtr ORT_API_CALL Compute(void* kernel,
                                           OrtKernelContext* context) noexcept {
    try {
      return compute_(kernel, context);
    } catch (const std::exception& error) {
      return api_->CreateStatus(ORT_FAIL, error.what());
    } catch (...) {
      return api_->CreateStatus(ORT_FAIL,
                                "Unknown custom-op compute exception");
    }
  }

  inline static std::once_flag initialized_;
  inline static const OrtApi* api_ = nullptr;
  inline static decltype(OrtCustomOp::CreateKernel) create_ = nullptr;
  inline static decltype(OrtCustomOp::KernelComputeV2) compute_ = nullptr;
};

}  // namespace colmap::aliked_native
