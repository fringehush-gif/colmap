// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

namespace custom_caller {
int calls = 0;
void CudaSafeCall(cudaError_t, const std::string&, int) { ++calls; }
void Invoke() { CUDA_SAFE_CALL(cudaSuccess); }
}  // namespace custom_caller

using colmap::CudaSafeCall;

namespace imported_caller {
using colmap::CudaSafeCall;
void Invoke() { imported_caller::CUDA_SAFE_CALL(cudaSuccess); }
}  // namespace imported_caller

namespace {
// This executable isolates the replacement allocator from other test targets.
thread_local bool observe_allocations = false;
thread_local bool refuse_allocations = false;
thread_local size_t allocation_attempts = 0;
// Exercise the runtime success branch even in optimized builds.
volatile cudaError_t success_status = cudaSuccess;

void ImportedCall(cudaError_t status, int* line) {
  using colmap::CudaSafeCall;
  *line = __LINE__ + 1;
  CUDA_SAFE_CALL(status);
}
void QualifiedCall(cudaError_t status, int* line) {
  *line = __LINE__ + 1;
  colmap::CUDA_SAFE_CALL(status);
}
void AbsoluteQualifiedCall(cudaError_t status, int* line) {
  *line = __LINE__ + 1;
  ::colmap::CUDA_SAFE_CALL(status);
}
void GlobalQualifiedCall(cudaError_t status, int* line) {
  *line = __LINE__ + 1;
  ::CUDA_SAFE_CALL(status);
}
}  // namespace

void* operator new(std::size_t size) {
  if (observe_allocations) {
    ++allocation_attempts;
    if (refuse_allocations) throw std::bad_alloc();
  }
  while (true) {
    if (void* ptr = std::malloc(size == 0 ? 1 : size)) return ptr;
    const auto handler = std::get_new_handler();
    if (handler == nullptr) throw std::bad_alloc();
    handler();
  }
}

void operator delete(void* ptr) noexcept { std::free(ptr); }
#ifdef __cpp_sized_deallocation
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
#endif

namespace colmap {
namespace {

using LegacyCall = void (*)(cudaError_t, const std::string&, int);
static_assert(std::is_same_v<decltype(&CudaSafeCall), LegacyCall>);

// The negative-control build routes the same checks to the legacy function.
void CheckOptIn(cudaError_t error, const char* file, int line) {
#ifdef COLMAP_TEST_LEGACY_SAFE_CALL
  ::colmap::CudaSafeCall(error, file, line);
#else
  ::colmap::internal::CudaSafeCallWithLocation(error, file, line);
#endif
}

constexpr size_t kRepetitions = 32;
struct Observation {
  size_t attempts;
  bool bad_alloc;
};
template <typename Call>
Observation Observe(Call&& call, bool refuse) {
  allocation_attempts = 0;
  refuse_allocations = refuse;
  observe_allocations = true;
  bool bad_alloc = false;
  try {
    for (size_t i = 0; i < kRepetitions; ++i) call();
  } catch (const std::bad_alloc&) {
    bad_alloc = true;
  } catch (...) {
    observe_allocations = false;
    refuse_allocations = false;
    throw;
  }
  observe_allocations = false;
  refuse_allocations = false;
  return {allocation_attempts, bad_alloc};
}

const char* RealFilename() { return __FILE__; }
void RealFilenameSuccess() { CheckOptIn(success_status, __FILE__, __LINE__); }
const char* LongFilename();
void LongFilenameSuccess();

void ExpectNoAllocations(void (*call)(), const char* filename, bool refuse) {
  testing::Test::RecordProperty("filename_bytes",
                                std::to_string(std::strlen(filename)));
  const auto result = Observe(call, refuse);
  testing::Test::RecordProperty("allocation_attempts",
                                std::to_string(result.attempts));
  testing::Test::RecordProperty("bad_alloc", result.bad_alloc ? 1 : 0);
  EXPECT_EQ(result.attempts, 0);
  EXPECT_FALSE(result.bad_alloc);
}

TEST(CudaSafeCallAllocation, RealFilenameSuccess) {
  ExpectNoAllocations(RealFilenameSuccess, RealFilename(), false);
}
TEST(CudaSafeCallAllocation, LongFilenameSuccess) {
  ExpectNoAllocations(LongFilenameSuccess, LongFilename(), false);
}
TEST(CudaSafeCallAllocation, RealFilenameSuccessWithAllocationRefused) {
  ExpectNoAllocations(RealFilenameSuccess, RealFilename(), true);
}
TEST(CudaSafeCallAllocation, LongFilenameSuccessWithAllocationRefused) {
  ExpectNoAllocations(LongFilenameSuccess, LongFilename(), true);
}

TEST(CudaSafeCallAllocation, ExistingMacroSpellingsKeepLegacyAllocation) {
  for (const auto call : {ImportedCall,
                          QualifiedCall,
                          AbsoluteQualifiedCall,
                          GlobalQualifiedCall}) {
    SCOPED_TRACE(call == ImportedCall            ? "imported"
                 : call == QualifiedCall         ? "colmap::"
                 : call == AbsoluteQualifiedCall ? "::colmap::"
                                                 : "::");
    for (const bool refuse : {false, true}) {
      SCOPED_TRACE(refuse);
      int line = 0;
      const auto result = Observe([&] { call(success_status, &line); }, refuse);
      EXPECT_EQ(result.attempts, refuse ? 1 : kRepetitions);
      EXPECT_EQ(result.bad_alloc, refuse);
    }
  }
}

TEST(CudaSafeCallAllocation, CounterDetectsLegacyFilenameConversion) {
  const auto result =
      Observe([] { CudaSafeCall(success_status, LongFilename(), 17); }, false);
  EXPECT_GE(result.attempts, kRepetitions);
  EXPECT_FALSE(result.bad_alloc);
}

TEST(CudaSafeCallAllocation, LegacyInterfaceAndExistingString) {
  const std::string filename(1024, 'x');
  const LegacyCall legacy = &CudaSafeCall;
  const auto result = Observe([&] { legacy(cudaSuccess, filename, 17); }, true);
  EXPECT_EQ(result.attempts, 0);
  EXPECT_FALSE(result.bad_alloc);
}

TEST(CudaSafeCallAllocation, LegacyLocalLookupAndQualifiedImport) {
  custom_caller::calls = 0;
  custom_caller::Invoke();
  EXPECT_EQ(custom_caller::calls, 1);
  EXPECT_NO_THROW(imported_caller::Invoke());
}

TEST(CudaSafeCallAllocation, SingleEvaluationAndExpressionUse) {
  int evaluations = 0;
  const auto status = [&] {
    ++evaluations;
    return cudaSuccess;
  };
  if (evaluations == 0)
    CheckOptIn(status(), __FILE__, __LINE__);
  else
    FAIL() << "Unexpected branch";
  EXPECT_EQ(evaluations, 1);
  const bool result = (CheckOptIn(status(), __FILE__, __LINE__), true);
  EXPECT_TRUE(result);
  EXPECT_EQ(evaluations, 2);
  colmap::CUDA_SAFE_CALL(status());
  ::colmap::CUDA_SAFE_CALL(status());
  EXPECT_EQ(evaluations, 4);
}

template <typename Call>
std::string CaptureError(Call&& call) {
  try {
    call();
  } catch (const std::invalid_argument& error) {
    return error.what();
  }
  ADD_FAILURE() << "Expected the existing exception type";
  return {};
}

TEST(CudaSafeCallAllocation, FailureMessageMatchesLegacyExactly) {
  for (const auto status :
       {cudaErrorInsufficientDriver, static_cast<cudaError_t>(123456)}) {
    SCOPED_TRACE(static_cast<int>(status));
    int line = 0;
    const auto actual = CaptureError([&] {
      line = __LINE__ + 1;
      CheckOptIn(status, __FILE__, __LINE__);
    });
    const auto expected =
        CaptureError([&] { CudaSafeCall(status, __FILE__, line); });
    EXPECT_EQ(actual, expected);
  }
}

TEST(CudaSafeCallAllocation, ExistingMacroSpellingsKeepDiagnostics) {
  for (const auto call : {ImportedCall,
                          QualifiedCall,
                          AbsoluteQualifiedCall,
                          GlobalQualifiedCall}) {
    SCOPED_TRACE(call == ImportedCall            ? "imported"
                 : call == QualifiedCall         ? "colmap::"
                 : call == AbsoluteQualifiedCall ? "::colmap::"
                                                 : "::");
    for (const auto status :
         {cudaErrorInsufficientDriver, static_cast<cudaError_t>(123456)}) {
      int line = 0;
      const auto actual = CaptureError([&] { call(status, &line); });
      const auto expected =
          CaptureError([&] { CudaSafeCall(status, __FILE__, line); });
      EXPECT_EQ(actual, expected);
    }
  }
}

}  // namespace
}  // namespace colmap

// A second literal path exercises lengths independent of checkout placement.
#line 1 "/cuda/safe-call/allocation-regression/long-source-filename/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/cudacc_allocation_test.cc"
namespace colmap {
namespace {
const char* LongFilename() { return __FILE__; }
void LongFilenameSuccess() { CheckOptIn(success_status, __FILE__, __LINE__); }
}  // namespace
}  // namespace colmap
