// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/testing.h"

#include "thirdparty/PoissonRecon/VertexFactory.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace PoissonRecon::VertexFactory;
using File = std::unique_ptr<FILE, int (*)(FILE*)>;

File OpenFile(const char* name) {
  return File(
      std::fopen((colmap::CreateTestDir() / name).string().c_str(), "w+b"),
      &std::fclose);
}

template <typename T>
std::string Bytes(const T* values, const size_t count) {
  if (count == 0) return {};
  return std::string(reinterpret_cast<const char*>(values), sizeof(T) * count);
}

void ExpectBytes(FILE* file, const std::string& expected) {
  ASSERT_EQ(std::ftell(file), expected.size());
  std::rewind(file);
  std::string actual(expected.size(), '\0');
  ASSERT_EQ(std::fread(actual.data(), 1, actual.size(), file), actual.size());
  EXPECT_EQ(actual, expected);
}

template <typename Real, typename Disk>
void CheckVertexIO(const TypeOnDisk type) {
  SCOPED_TRACE(type);
  const std::array<Real, 3> values = {Real(7), Real(8), Real(9)};
  const std::array<Disk, 3> on_disk = {Disk(7), Disk(8), Disk(9)};
  const std::string expected = Bytes(on_disk.data(), on_disk.size());
  {
    const File file = OpenFile("vertex-output.bin");
    ASSERT_NE(file, nullptr);
    VertexIO<Real>::WriteBinary(file.get(), type, values[0]);
    VertexIO<Real>::WriteBinary(file.get(), type, 2, &values[1]);
    ExpectBytes(file.get(), expected);
  }
  {
    // Decode independently constructed bytes, so paired I/O bugs cannot hide.
    const File file = OpenFile("vertex-input.bin");
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(std::fwrite(expected.data(), 1, expected.size(), file.get()),
              expected.size());
    std::rewind(file.get());
    std::array<Real, 3> loaded{};
    ASSERT_TRUE(VertexIO<Real>::ReadBinary(file.get(), type, loaded[0]));
    ASSERT_TRUE(VertexIO<Real>::ReadBinary(file.get(), type, 2, &loaded[1]));
    EXPECT_EQ(loaded, values);
    EXPECT_EQ(std::ftell(file.get()), expected.size());
    EXPECT_FALSE(VertexIO<Real>::ReadBinary(file.get(), type, loaded[0]));
  }
}

template <typename Real, typename Factory>
void CheckFactoryIO(const Factory& factory,
                    const std::vector<Real>& values,
                    const std::string& expected) {
  {
    const File file = OpenFile("factory-output.bin");
    ASSERT_NE(file, nullptr);
    auto vertex = factory();
    for (size_t i = 0; i < values.size(); ++i) vertex[i] = values[i];
    factory.writeBinary(file.get(), vertex);
    ExpectBytes(file.get(), expected);
  }
  {
    const File file = OpenFile("factory-input.bin");
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(std::fwrite(expected.data(), 1, expected.size(), file.get()),
              expected.size());
    std::rewind(file.get());
    auto vertex = factory();
    ASSERT_TRUE(factory.readBinary(file.get(), vertex));
    for (size_t i = 0; i < values.size(); ++i) EXPECT_EQ(vertex[i], values[i]);
    EXPECT_EQ(std::ftell(file.get()), expected.size());
  }
}

template <typename Real>
class VertexFactoryBinaryTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(VertexFactoryBinaryTest, RealTypes);

TYPED_TEST(VertexFactoryBinaryTest, ScalarAndArrayUseDeclaredWidth) {
  using Real = TypeParam;
  CheckVertexIO<Real, char>(TypeOnDisk::CHAR);
  CheckVertexIO<Real, unsigned char>(TypeOnDisk::UCHAR);
  CheckVertexIO<Real, int>(TypeOnDisk::INT);
  CheckVertexIO<Real, unsigned int>(TypeOnDisk::UINT);
  CheckVertexIO<Real, float>(TypeOnDisk::FLOAT);
  CheckVertexIO<Real, double>(TypeOnDisk::DOUBLE);
  CheckVertexIO<Real, int8_t>(TypeOnDisk::INT_8);
  CheckVertexIO<Real, uint8_t>(TypeOnDisk::UINT_8);
  CheckVertexIO<Real, int16_t>(TypeOnDisk::INT_16);
  CheckVertexIO<Real, uint16_t>(TypeOnDisk::UINT_16);
  CheckVertexIO<Real, int32_t>(TypeOnDisk::INT_32);
  CheckVertexIO<Real, uint32_t>(TypeOnDisk::UINT_32);
  CheckVertexIO<Real, int64_t>(TypeOnDisk::INT_64);
  CheckVertexIO<Real, uint64_t>(TypeOnDisk::UINT_64);
}

TYPED_TEST(VertexFactoryBinaryTest, FixedFactoriesWriteEachValueOnce) {
  using Real = TypeParam;
  const TypeOnDisk native_type = GetTypeOnDisk<Real>();
  const std::vector<Real> texture = {Real(7), Real(8)};
  const std::vector<Real> rgb = {Real(7), Real(8), Real(9)};
  const std::vector<Real> rgba = {Real(7), Real(8), Real(9), Real(10)};
  CheckFactoryIO(TextureFactory<Real, 2>(native_type),
                 texture,
                 Bytes(texture.data(), texture.size()));
  CheckFactoryIO(
      RGBColorFactory<Real>(native_type), rgb, Bytes(rgb.data(), rgb.size()));
  CheckFactoryIO(RGBAColorFactory<Real>(native_type),
                 rgba,
                 Bytes(rgba.data(), rgba.size()));
  const std::array<unsigned char, 3> on_disk = {7, 8, 9};
  CheckFactoryIO(
      RGBColorFactory<Real>(), rgb, Bytes(on_disk.data(), on_disk.size()));
}

TYPED_TEST(VertexFactoryBinaryTest, DynamicFactoriesUseEveryDeclaredType) {
  using Real = TypeParam;
  const std::vector<Real> values = {Real(7), Real(8)};
  const TypeOnDisk native_type = GetTypeOnDisk<Real>();
  const Real native = Real(7);
  const unsigned char first = 7;
  const unsigned int second = 8;
  const unsigned char last = 8;
  struct Layout {
    std::array<TypeOnDisk, 2> types;
    std::string bytes;
  };
  const std::array<Layout, 3> layouts = {
      Layout{{native_type, native_type}, Bytes(values.data(), values.size())},
      Layout{{TypeOnDisk::UCHAR, TypeOnDisk::UINT},
             Bytes(&first, 1) + Bytes(&second, 1)},
      Layout{{native_type, TypeOnDisk::UCHAR},
             Bytes(&native, 1) + Bytes(&last, 1)}};
  for (const auto& layout : layouts) {
    SCOPED_TRACE(::testing::PrintToString(layout.types));
    std::vector<std::pair<std::string, TypeOnDisk>> names_and_types;
    std::vector<PoissonRecon::PlyProperty> properties;
    for (size_t i = 0; i < layout.types.size(); ++i) {
      const std::string name = "property" + std::to_string(i);
      names_and_types.emplace_back(name, layout.types[i]);
      properties.emplace_back(name, ToPlyType(layout.types[i]), 0, 0);
    }
    CheckFactoryIO(DynamicFactory<Real>(names_and_types), values, layout.bytes);
    CheckFactoryIO(DynamicFactory<Real>(properties), values, layout.bytes);
  }
}

TYPED_TEST(VertexFactoryBinaryTest, EmptyDynamicFactoryLeavesStreamUntouched) {
  using Real = TypeParam;
  const std::vector<std::pair<std::string, TypeOnDisk>> names_and_types;
  const std::vector<PoissonRecon::PlyProperty> properties;
  CheckFactoryIO(
      DynamicFactory<Real>(names_and_types), std::vector<Real>{}, {});
  CheckFactoryIO(DynamicFactory<Real>(properties), std::vector<Real>{}, {});
}

}  // namespace
