// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/oct_normal_state.cuh"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace colmap::mvs {
namespace {
using oct_magnitude::Encoded;
using oct_magnitude::Status;

struct Observation {
  Status status;
  Encoded code;
  float3 decoded;
  bool decoded_ok;
};
__global__ void Observe(const float3* input, Observation* output, size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Observation result{};
  result.status = oct_magnitude::Encode<2>(input[i], result.code, 0x1p-23f);
  result.decoded_ok = oct_magnitude::Decode(result.code, result.decoded);
  output[i] = result;
}
struct PreparedObservation {
  Status status;
  Status reference_status;
  Encoded code;
  Encoded reference_code;
  unsigned prepared_bits[3];
  unsigned decoded_bits[3];
  unsigned owner_bits[3];
  unsigned owner_decoded_bits[3];
  Encoded owner_code;
  Encoded owner_reference_code;
  unsigned error;
  unsigned expected_error;
};
__device__ void SaveBits(float3 value, unsigned* bits) {
  bits[0] = __float_as_uint(value.x);
  bits[1] = __float_as_uint(value.y);
  bits[2] = __float_as_uint(value.z);
}
__global__ void ObservePrepared(const float3* input,
                                PreparedObservation* output,
                                unsigned* errors,
                                size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * 4) return;
  const unsigned frame = i % 4;
  const float3 value = input[i / 4];
  PreparedObservation result{};
  float3 prepared, decoded;
  const float limit = i / 4 == n - 1 ? 0.0f : 0x1p-23f;
  result.status =
      oct_magnitude::Encode<2>(value, result.code, limit, &prepared);
  result.reference_status =
      oct_magnitude::Encode<2>(value, result.reference_code, limit);
  oct_magnitude::Decode(result.reference_code, decoded);
  SaveBits(prepared, result.prepared_bits);
  SaveBits(decoded, result.decoded_bits);

  OctNormalView view{nullptr, nullptr, 0, 0, 0, &errors[i], frame};
  float normal[3] = {value.x, value.y, value.z};
  result.owner_code = view.Canonicalize(normal);
  SaveBits(make_float3(normal[0], normal[1], normal[2]), result.owner_bits);
  float3 canonical;
  switch (frame) {
    case 0:
      canonical = value;
      break;
    case 1:
      canonical = make_float3(-value.y, value.x, value.z);
      break;
    case 2:
      canonical = make_float3(-value.x, -value.y, value.z);
      break;
    default:
      canonical = make_float3(value.y, -value.x, value.z);
      break;
  }
  const Status status = oct_magnitude::Encode<2>(
      canonical, result.owner_reference_code, 0x1p-23f);
  view.DecodeCode(result.owner_reference_code, normal);
  SaveBits(make_float3(normal[0], normal[1], normal[2]),
           result.owner_decoded_bits);
  result.error = errors[i];
  result.expected_error = status == Status::kOk || status == Status::kZero
                              ? 0u
                              : 1u << static_cast<unsigned>(status);
  output[i] = result;
}
__global__ void DecodeMalformed(bool* ok, float3* values) {
  Encoded c = oct_magnitude::Invalid();
  c.magnitude = 127;
  ok[0] = oct_magnitude::Decode(c, values[0]);
  c = oct_magnitude::Invalid();
  c.direction.x = 0;
  ok[1] = oct_magnitude::Decode(c, values[1]);
}
__global__ void InvalidOwnerOperations(OctNormalView view, int which) {
  if (which == 0) {
    const float invalid[3] = {2.0f, 0.0f, 0.0f};
    view.SetSlice(0, 0, invalid);
  } else if (which == 1) {
    Encoded malformed = oct_magnitude::Invalid();
    malformed.magnitude = 127;
    view.StoreCode(0, 0, malformed);
  } else if (which == 2) {
    view.Set(0, 0, 1, 1.0f);
  } else {
    view.direction[0] = {oct_revisit::kInvalid, 0};
    view.magnitude[0] = 127;
    float n[3];
    view.GetSlice(0, 0, n);
  }
}
__global__ void ReadAndCarry(OctNormalView view,
                             Encoded* codes,
                             float3* values,
                             unsigned* changed) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= view.width * view.height) return;
  const size_t row = i / view.width, col = i % view.width;
  float n[3];
  Encoded code;
  const ActiveNormalView active{{}, view, true};
  LoadNormal(active, row, col, n, code);
  StoreNormal(active, row, col, n, code);  // winner carry must not re-encode
  codes[i] = view.ReadCode(row, col);
  if (codes[i].direction.x != code.direction.x ||
      codes[i].direction.y != code.direction.y ||
      codes[i].magnitude != code.magnitude)
    atomicAdd(changed, 1u);
  values[i] = make_float3(n[0], n[1], n[2]);
}
bool SameCode(Encoded a, Encoded b) {
  return a.direction.x == b.direction.x && a.direction.y == b.direction.y &&
         a.magnitude == b.magnitude;
}
class OctNormalCodec : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count == 0)) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

TEST_F(OctNormalCodec, IndependentDoubleAngularAndNormOracle) {
  std::vector<float3> input;
  // Fibonacci sphere, both hemispheres and folds; each input is FP32, while
  // comparison below uses double precision on those exact stored values.
  for (int i = 0; i < 4096; ++i) {
    const double z = 1.0 - 2.0 * (i + 0.5) / 4096.0;
    const double phi = i * 2.39996322972865332;
    const double r = std::sqrt(1.0 - z * z);
    input.push_back(make_float3(
        float(r * std::cos(phi)), float(r * std::sin(phi)), float(z)));
  }
  float3* device_input = nullptr;
  Observation* device_output = nullptr;
  CUDA_SAFE_CALL(cudaMalloc(&device_input, input.size() * sizeof(float3)));
  CUDA_SAFE_CALL(
      cudaMalloc(&device_output, input.size() * sizeof(Observation)));
  CUDA_SAFE_CALL(cudaMemcpy(device_input,
                            input.data(),
                            input.size() * sizeof(float3),
                            cudaMemcpyHostToDevice));
  Observe<<<(input.size() + 127) / 128, 128>>>(
      device_input, device_output, input.size());
  CUDA_SYNC_AND_CHECK();
  std::vector<Observation> output(input.size());
  CUDA_SAFE_CALL(cudaMemcpy(output.data(),
                            device_output,
                            output.size() * sizeof(Observation),
                            cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(device_output));
  CUDA_SAFE_CALL(cudaFree(device_input));
  for (size_t i = 0; i < input.size(); ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(output[i].status, Status::kOk);
    ASSERT_TRUE(output[i].decoded_ok);
    const double ax = input[i].x, ay = input[i].y, az = input[i].z;
    const double bx = output[i].decoded.x, by = output[i].decoded.y,
                 bz = output[i].decoded.z;
    const double an = std::sqrt(ax * ax + ay * ay + az * az),
                 bn = std::sqrt(bx * bx + by * by + bz * bz);
    const double cx = ay * bz - az * by, cy = az * bx - ax * bz,
                 cz = ax * by - ay * bx;
    const double angle = std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz),
                                    ax * bx + ay * by + az * bz);
    EXPECT_LE(angle, 0.00015);  // 16+16 oct direction bound, radians
    EXPECT_LE(std::abs(an - bn), std::ldexp(1.0, -23));
  }
}

TEST_F(OctNormalCodec, PreparedNormalMatchesSelectedCodeInAllFrames) {
  std::vector<float3> input;
  for (int i = 0; i < 4096; ++i) {
    const double z = 1.0 - 2.0 * (i + 0.5) / 4096.0;
    const double phi = i * 2.39996322972865332;
    const double r = std::sqrt(1.0 - z * z);
    input.push_back(make_float3(
        float(r * std::cos(phi)), float(r * std::sin(phi)), float(z)));
  }
  input.push_back(make_float3(0, 0, 0));
  input.push_back(make_float3(2, 0, 0));
  input.push_back(make_float3(std::numeric_limits<float>::infinity(), 0, 0));
  input.push_back(make_float3(std::numeric_limits<float>::quiet_NaN(), 0, 0));
  input.push_back(make_float3(1, 0, 0));  // Invalid certificate limit.
  const size_t count = input.size() * 4;
  float3* device_input = nullptr;
  PreparedObservation* device_output = nullptr;
  unsigned* errors = nullptr;
  CUDA_SAFE_CALL(cudaMalloc(&device_input, input.size() * sizeof(float3)));
  CUDA_SAFE_CALL(
      cudaMalloc(&device_output, count * sizeof(PreparedObservation)));
  CUDA_SAFE_CALL(cudaMalloc(&errors, count * sizeof(unsigned)));
  CUDA_SAFE_CALL(cudaMemset(errors, 0, count * sizeof(unsigned)));
  CUDA_SAFE_CALL(cudaMemcpy(device_input,
                            input.data(),
                            input.size() * sizeof(float3),
                            cudaMemcpyHostToDevice));
  ObservePrepared<<<(count + 127) / 128, 128>>>(
      device_input, device_output, errors, input.size());
  CUDA_SYNC_AND_CHECK();
  std::vector<PreparedObservation> output(count);
  CUDA_SAFE_CALL(cudaMemcpy(output.data(),
                            device_output,
                            count * sizeof(PreparedObservation),
                            cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(errors));
  CUDA_SAFE_CALL(cudaFree(device_output));
  CUDA_SAFE_CALL(cudaFree(device_input));
  EXPECT_EQ(output[0].status, Status::kOk);
  EXPECT_EQ(output[4096 * 4].status, Status::kZero);
  EXPECT_EQ(output[4097 * 4].status, Status::kOutOfEnvelope);
  EXPECT_EQ(output[4098 * 4].status, Status::kNonfinite);
  EXPECT_EQ(output[(input.size() - 1) * 4].status,
            Status::kNoCertifiedCandidate);
  for (size_t i = 0; i < count; ++i) {
    SCOPED_TRACE(i);
    const auto& got = output[i];
    EXPECT_EQ(got.status, got.reference_status);
    EXPECT_TRUE(SameCode(got.code, got.reference_code));
    EXPECT_TRUE(SameCode(got.owner_code, got.owner_reference_code));
    EXPECT_EQ(got.error, got.expected_error);
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_EQ(got.prepared_bits[axis], got.decoded_bits[axis]);
      EXPECT_EQ(got.owner_bits[axis], got.owner_decoded_bits[axis]);
    }
  }
}

TEST_F(OctNormalCodec, RefusalAndSentinels) {
  const float inf = std::numeric_limits<float>::infinity(),
              nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float3> input = {make_float3(0, 0, 0),
                                     make_float3(inf, 0, 0),
                                     make_float3(nan, 0, 0),
                                     make_float3(2, 0, 0)};
  float3* d = nullptr;
  Observation* o = nullptr;
  CUDA_SAFE_CALL(cudaMalloc(&d, input.size() * sizeof(float3)));
  CUDA_SAFE_CALL(cudaMalloc(&o, input.size() * sizeof(Observation)));
  CUDA_SAFE_CALL(cudaMemcpy(
      d, input.data(), input.size() * sizeof(float3), cudaMemcpyHostToDevice));
  Observe<<<1, 32>>>(d, o, input.size());
  CUDA_SYNC_AND_CHECK();
  std::vector<Observation> got(input.size());
  CUDA_SAFE_CALL(cudaMemcpy(
      got.data(), o, got.size() * sizeof(Observation), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(o));
  CUDA_SAFE_CALL(cudaFree(d));
  EXPECT_EQ(got[0].status, Status::kZero);
  EXPECT_EQ(got[1].status, Status::kNonfinite);
  EXPECT_EQ(got[2].status, Status::kNonfinite);
  EXPECT_EQ(got[3].status, Status::kOutOfEnvelope);
  for (auto& x : got) {
    EXPECT_TRUE(x.decoded_ok);
    EXPECT_EQ(x.code.direction.x, oct_revisit::kInvalid);
    EXPECT_EQ(x.code.direction.y, oct_revisit::kInvalid);
    EXPECT_EQ(x.code.magnitude, oct_magnitude::kInvalidMagnitude);
    EXPECT_EQ(x.decoded.x, 0);
    EXPECT_EQ(x.decoded.y, 0);
    EXPECT_EQ(x.decoded.z, 0);
  }
  bool* decoded_ok = nullptr;
  float3* decoded_values = nullptr;
  CUDA_SAFE_CALL(cudaMalloc(&decoded_ok, 2 * sizeof(bool)));
  CUDA_SAFE_CALL(cudaMalloc(&decoded_values, 2 * sizeof(float3)));
  DecodeMalformed<<<1, 1>>>(decoded_ok, decoded_values);
  CUDA_SYNC_AND_CHECK();
  bool oks[2];
  float3 zeros[2];
  CUDA_SAFE_CALL(
      cudaMemcpy(oks, decoded_ok, sizeof(oks), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(
      cudaMemcpy(zeros, decoded_values, sizeof(zeros), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(decoded_values));
  CUDA_SAFE_CALL(cudaFree(decoded_ok));
  for (int i = 0; i < 2; ++i) {
    EXPECT_FALSE(oks[i]);
    EXPECT_EQ(zeros[i].x, 0);
    EXPECT_EQ(zeros[i].y, 0);
    EXPECT_EQ(zeros[i].z, 0);
  }
  for (int mode = 0; mode < 4; ++mode) {
    OctNormalMap map(1, 1);
    InvalidOwnerOperations<<<1, 1>>>(map.View(), mode);
    CUDA_SYNC_AND_CHECK();
    EXPECT_ANY_THROW(map.CheckDomain());
    EXPECT_ANY_THROW(map.CopyToMat());
  }
}

TEST_F(OctNormalCodec, RotationRejectsEitherPoisonedOwnerBeforeCopy) {
  constexpr size_t width = 7, height = 5, count = width * height;
  std::vector<float> xyz(3 * count, 0.0f);
  std::fill(xyz.begin() + 2 * count, xyz.end(), -1.0f);
  for (const bool poison_source : {false, true}) {
    SCOPED_TRACE(poison_source);
    constexpr size_t target_width = 9, target_height = 11;
    OctNormalMap source(width, height), target(target_width, target_height);
    source.CopyFromCPU(xyz.data(), width * sizeof(float));
    std::vector<float> target_xyz(3 * target_width * target_height, 0.0f);
    std::fill(target_xyz.begin() + 2 * target_width * target_height,
              target_xyz.end(),
              -1.0f);
    target.CopyFromCPU(target_xyz.data(), target_width * sizeof(float));
    auto& poisoned = poison_source ? source : target;
    auto& healthy = poison_source ? target : source;
    const auto before = healthy.CopyToMat().GetData();
    InvalidOwnerOperations<<<1, 1>>>(poisoned.View(), 2);
    CUDA_SYNC_AND_CHECK();
    EXPECT_THROW(source.RotateTo(target), std::invalid_argument);
    EXPECT_EQ(source.GetWidth(), width);
    EXPECT_EQ(source.GetHeight(), height);
    EXPECT_EQ(target.GetWidth(), target_width);
    EXPECT_EQ(target.GetHeight(), target_height);
    EXPECT_EQ(source.GetFrame(), 0u);
    EXPECT_EQ(target.GetFrame(), 0u);
    EXPECT_EQ(healthy.CopyToMat().GetData(), before);
    EXPECT_NO_THROW(healthy.CheckDomain());
    EXPECT_THROW(poisoned.CheckDomain(), std::invalid_argument);
  }
}

TEST_F(OctNormalCodec, FourFrameOwnerRotationAndWinnerCodeCarry) {
  constexpr size_t w = 7, h = 5, n = w * h;
  std::vector<float> xyz(3 * n);
  for (size_t i = 0; i < n; ++i) {
    const double z = -0.9 + 1.8 * (i + 0.5) / n, p = 2.39996322972865332 * i,
                 r = std::sqrt(1 - z * z);
    xyz[i] = float(r * std::cos(p));
    xyz[n + i] = float(r * std::sin(p));
    xyz[2 * n + i] = float(z);
  }
  OctNormalMap a(w, h, true), b(h, w, true);
  a.CopyFromCPU(xyz.data(), w * sizeof(float));
  Encoded* codes = nullptr;
  float3* values = nullptr;
  unsigned* changed = nullptr;
  CUDA_SAFE_CALL(cudaMalloc(&changed, sizeof(unsigned)));
  CUDA_SAFE_CALL(cudaMemset(changed, 0, sizeof(unsigned)));
  CUDA_SAFE_CALL(cudaMalloc(&codes, n * sizeof(Encoded)));
  CUDA_SAFE_CALL(cudaMalloc(&values, n * sizeof(float3)));
  auto capture = [&](OctNormalMap& owner,
                     std::vector<Encoded>& c,
                     std::vector<float3>& v) {
    ReadAndCarry<<<1, 128>>>(owner.View(), codes, values, changed);
    CUDA_SYNC_AND_CHECK();
    owner.CheckDomain();
    c.resize(n);
    v.resize(n);
    CUDA_SAFE_CALL(cudaMemcpy(
        c.data(), codes, n * sizeof(Encoded), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(
        v.data(), values, n * sizeof(float3), cudaMemcpyDeviceToHost));
  };
  std::vector<Encoded> expected, actual;
  std::vector<float3> expected_values, actual_values;
  capture(a, expected, expected_values);
  OctNormalMap* in = &a;
  OctNormalMap* out = &b;
  size_t width = w, height = h;
  for (unsigned frame = 1; frame <= 4; ++frame) {
    std::vector<Encoded> next(n);
    std::vector<float3> nv(n);
    for (size_t row = 0; row < width; ++row)
      for (size_t col = 0; col < height; ++col) {
        const size_t dst = row * height + col,
                     src = col * width + (width - 1 - row);
        next[dst] = expected[src];
        const auto v = expected_values[src];
        nv[dst] = make_float3(v.y, -v.x, v.z);
      }
    in->RotateTo(*out);
    EXPECT_EQ(out->GetFrame(), frame % 4);
    capture(*out, actual, actual_values);
    for (size_t i = 0; i < n; ++i) {
      EXPECT_TRUE(SameCode(actual[i], next[i]));
      EXPECT_FLOAT_EQ(actual_values[i].x, nv[i].x);
      EXPECT_FLOAT_EQ(actual_values[i].y, nv[i].y);
      EXPECT_FLOAT_EQ(actual_values[i].z, nv[i].z);
    }
    expected.swap(next);
    expected_values.swap(nv);
    std::swap(width, height);
    std::swap(in, out);
  }
  unsigned changed_count = 0;
  CUDA_SAFE_CALL(cudaMemcpy(
      &changed_count, changed, sizeof(unsigned), cudaMemcpyDeviceToHost));
  EXPECT_EQ(changed_count, 0u);
  CUDA_SAFE_CALL(cudaFree(changed));
  CUDA_SAFE_CALL(cudaFree(values));
  CUDA_SAFE_CALL(cudaFree(codes));
}
}  // namespace
}  // namespace colmap::mvs
