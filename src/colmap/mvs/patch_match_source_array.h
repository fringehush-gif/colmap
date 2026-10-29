// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/patch_match.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {

// Per-call test-only operation outcome; no global runtime/driver interposition.
// code uses the selected operation's actual CUDA runtime or driver domain.
// A nonzero host-staging failure models std::bad_alloc. Code zero observes
// call counts without replacing an operation. Chaining allows a setup and
// cleanup failure in one isolated call. Nothing retains this pointer on
// success.
struct SparseSourceTestFailure {
  const char* operation;
  int code;
  size_t fail_on_call = 1;
  mutable size_t calls = 0;
  const SparseSourceTestFailure* next = nullptr;
};

// Only this exception permits sparse-only resource readmission. It is emitted
// after all partial sparse setup has been explicitly and successfully released.
// Ordinary PatchMatchCuda/dense allocation failures are outside this contract.
class SparseSourceResourceRefused : public std::runtime_error {
 public:
  SparseSourceResourceRefused(const std::string& operation, int code)
      : std::runtime_error(operation + " refused (code " +
                           std::to_string(code) + ")"),
        operation(operation),
        code(code) {}
  std::string operation;
  int code;
};

// Batch only: one workspace and immutable source-map generation. Inputs stay
// unchanged through Create; source IDs from different workspaces never alias.
// No cross-batch cache. Attach views before Run and join consumers before
// release. No GPU LUT or changed shader sampling interface.
class PatchMatchSparseSources {
 public:
  // Build, device and required entrypoint capability, independent of a
  // particular batch's profitability or source provenance.
  static bool IsSupported();
  static bool HasConsistentInputs(
      const std::vector<const PatchMatch::Problem*>& problems, bool geometric);
  static size_t Budget(const std::vector<const PatchMatch::Problem*>& problems,
                       bool geometric,
                       const SparseSourceTestFailure* failure = nullptr);
  static std::shared_ptr<const PatchMatchSparseSources> Create(
      const std::vector<const PatchMatch::Problem*>& problems,
      bool geometric,
      const SparseSourceTestFailure* failure = nullptr);
  ~PatchMatchSparseSources();
  // Only after all consumer views are released/joined. Cleanup failure is
  // terminal and must not be caught as a resource refusal.
  void Close() const;
  bool Matches(const PatchMatch::Problem& problem, bool geometric) const;

 private:
  template <typename T>
  friend class PatchMatchSourceArray;
  struct Impl;
  explicit PatchMatchSparseSources(std::unique_ptr<Impl> impl);
  void AbandonAfterTerminalViewFailure(const char* operation, int code) const;
  std::unique_ptr<Impl> impl_;
};

// Owns either the original dense helper or a texture view plus a bank token.
// Its raw GetObj ABI and logical dimensions match the original source owner.
template <typename T>
class PatchMatchSourceArray {
 public:
  using HostSlice = typename CudaArrayLayeredTexture<T>::HostSlice;
  static std::unique_ptr<PatchMatchSourceArray> FromHostArray(
      const cudaTextureDesc& desc,
      size_t width,
      size_t height,
      size_t depth,
      const T* data);
  static std::unique_ptr<PatchMatchSourceArray> FromHostSlices(
      const cudaTextureDesc& desc,
      size_t width,
      size_t height,
      const std::vector<HostSlice>& slices);
  static std::unique_ptr<PatchMatchSourceArray> FromSparse(
      std::shared_ptr<const PatchMatchSparseSources> bank,
      int reference,
      const SparseSourceTestFailure* failure = nullptr);
  ~PatchMatchSourceArray();
  void Close();
  cudaTextureObject_t GetObj() const;
  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;
  bool IsSparse() const;

 private:
  struct Impl;
  explicit PatchMatchSourceArray(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

extern template class PatchMatchSourceArray<uint8_t>;
extern template class PatchMatchSourceArray<float>;

}  // namespace mvs
}  // namespace colmap
