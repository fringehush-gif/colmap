// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/endian.h"
#include "colmap/util/testing.h"

#include "thirdparty/PoissonRecon/PlyFile.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace {

using PoissonRecon::PlyFile;
using PoissonRecon::PlyOtherProp;
using PoissonRecon::PlyProperty;

struct Encoding {
  int format;
  bool use_double;
};

enum class Storage { kRequested, kDiscarded, kOther };

template <typename T>
struct Record {
  T value = 0;
  uint8_t count = 0;
  T* values = nullptr;
  uint32_t marker = 0;
  void* other = nullptr;
};

const char* FormatName(const int format) {
  if (format == PLY_ASCII) return "ascii";
  return format == PLY_BINARY_LE ? "binary_little_endian" : "binary_big_endian";
}

const char* TypeName(const int type) {
  switch (type) {
    case PLY_FLOAT:
      return "float";
    case PLY_FLOAT_32:
      return "float32";
    case PLY_DOUBLE:
      return "double";
    default:
      return "float64";
  }
}

template <typename T>
std::array<int, 2> Types() {
  return std::is_same_v<T, float>
             ? std::array<int, 2>{PLY_FLOAT, PLY_FLOAT_32}
             : std::array<int, 2>{PLY_DOUBLE, PLY_FLOAT_64};
}

template <typename T>
std::array<T, 9> Values() {
  const T large = T(std::is_same_v<T, float> ? 1e20 : 1e100);
  return {T(-1.25),
          T(-0.0),
          T(0),
          T(1.25),
          large,
          -large,
          std::numeric_limits<T>::infinity(),
          -std::numeric_limits<T>::infinity(),
          std::numeric_limits<T>::quiet_NaN()};
}

template <typename T>
void ExpectValue(const T actual, const T expected) {
  // The vendor target uses -ffast-math. Classify representations instead of
  // relying on isnan/isfinite comparisons under finite-math assumptions.
  using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
  Bits actual_bits, expected_bits;
  std::memcpy(&actual_bits, &actual, sizeof(T));
  std::memcpy(&expected_bits, &expected, sizeof(T));
  const Bits exponent =
      static_cast<Bits>(sizeof(T) == 4 ? 0x7f800000ULL : 0x7ff0000000000000ULL);
  const Bits fraction =
      static_cast<Bits>(sizeof(T) == 4 ? 0x007fffffULL : 0x000fffffffffffffULL);
  if ((expected_bits & exponent) == exponent && (expected_bits & fraction)) {
    EXPECT_EQ(actual_bits & exponent, exponent);
    EXPECT_NE(actual_bits & fraction, 0);
  } else {
    EXPECT_EQ(actual_bits, expected_bits);
  }
}

template <typename T>
void Append(std::string& bytes, const int format, T value) {
  if (format == PLY_ASCII) {
    std::ostringstream stream;
    // Default precision matches the public writer's %g for these fixtures.
    stream << +value << ' ';
    bytes += stream.str();
  } else {
    value = format == PLY_BINARY_LE ? colmap::NativeToLittleEndian(value)
                                    : colmap::NativeToBigEndian(value);
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
  }
}

std::string Header(const int format,
                   const char* type,
                   const bool list,
                   const size_t count) {
  std::ostringstream stream;
  stream << "ply\nformat " << FormatName(format) << " 1.0\nelement vertex "
         << count << "\nproperty ";
  if (list) stream << "list uchar ";
  stream << type << (list ? " values\n" : " value\n")
         << "property uint marker\nend_header\n";
  return stream.str();
}

template <typename T>
std::string Payload(const int format, const bool list) {
  const auto values = Values<T>();
  const size_t count = list ? 2 : values.size();
  std::string bytes;
  for (size_t i = 0; i < count; ++i) {
    if (list) {
      Append(bytes, format, uint8_t(i ? values.size() : 0));
      if (i) {
        for (const T value : values) Append(bytes, format, value);
      }
    } else {
      Append(bytes, format, values[i]);
    }
    Append(bytes, format, uint32_t(17 + i));
    if (format == PLY_ASCII) bytes += '\n';
  }
  return bytes;
}

void WriteBytes(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(bytes.data(), bytes.size());
  stream.close();
  ASSERT_TRUE(stream.good());
}

std::string ReadBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), {}};
}

template <typename T>
std::array<PlyProperty, 2> Properties(const int type, const bool list) {
  return {
      list ? PlyProperty("values",
                         type,
                         type,
                         offsetof(Record<T>, values),
                         PLY_LIST,
                         PLY_UCHAR,
                         PLY_UCHAR,
                         offsetof(Record<T>, count))
           : PlyProperty("value", type, type, offsetof(Record<T>, value)),
      PlyProperty("marker", PLY_UINT, PLY_UINT, offsetof(Record<T>, marker))};
}

template <typename T>
void CheckRead(const int format, const bool list, const Storage storage) {
  const auto path = colmap::CreateTestDir() / "input.ply";
  const auto values = Values<T>();
  const size_t count = list ? 2 : values.size();
  for (const int type : Types<T>()) {
    SCOPED_TRACE(TypeName(type));
    WriteBytes(
        path,
        Header(format, TypeName(type), list, count) + Payload<T>(format, list));
    std::vector<std::string> names;
    int actual_format;
    float version;
    const std::unique_ptr<PlyFile> reader(
        PlyFile::Read(path.string(), names, actual_format, version));
    ASSERT_NE(reader, nullptr);
    EXPECT_EQ(actual_format, format);
    EXPECT_EQ(version, 1.0f);
    ASSERT_EQ(names, std::vector<std::string>{"vertex"});
    auto properties = Properties<T>(type, list);
    if (storage == Storage::kRequested) {
      reader->get_element_setup("vertex", properties.size(), properties.data());
    } else {
      reader->get_element_setup("vertex", 1, &properties[1]);
    }
    PlyOtherProp other;
    if (storage == Storage::kOther) {
      ASSERT_TRUE(reader->set_other_properties(
          "vertex", offsetof(Record<T>, other), other));
      ASSERT_EQ(other.props.size(), 1);
      EXPECT_EQ(other.props[0].internal_type, type);
    }
    for (size_t i = 0; i < count; ++i) {
      SCOPED_TRACE(i);
      Record<T> record;
      reader->get_element(&record);
      const std::unique_ptr<void, decltype(&std::free)> other_data(record.other,
                                                                   &std::free);
      EXPECT_EQ(record.marker, 17 + i);
      if (storage == Storage::kDiscarded) {
        EXPECT_EQ(record.value, 0);
        EXPECT_EQ(record.count, 0);
        EXPECT_EQ(record.values, nullptr);
        EXPECT_EQ(record.other, nullptr);
        continue;
      }
      if (storage == Storage::kOther) {
        ASSERT_NE(record.other, nullptr);
        const auto* data = static_cast<const char*>(record.other);
        if (list) {
          std::memcpy(&record.count,
                      data + other.props[0].count_offset,
                      sizeof(record.count));
          std::memcpy(static_cast<void*>(&record.values),
                      data + other.props[0].offset,
                      sizeof(record.values));
        } else {
          std::memcpy(&record.value,
                      data + other.props[0].offset,
                      sizeof(record.value));
        }
      }
      const std::unique_ptr<T, decltype(&std::free)> list_data(record.values,
                                                               &std::free);
      if (list) {
        ASSERT_EQ(record.count, i ? values.size() : 0);
        if (i == 0) {
          EXPECT_EQ(record.values, nullptr);
        } else {
          ASSERT_NE(record.values, nullptr);
          for (size_t j = 0; j < values.size(); ++j) {
            ExpectValue(record.values[j], values[j]);
          }
        }
      } else {
        ExpectValue(record.value, values[i]);
      }
    }
  }
}

template <typename T>
void CheckWrite(const int format, const bool list) {
  const auto path = colmap::CreateTestDir() / "output.ply";
  auto values = Values<T>();
  const size_t count = list ? 2 : values.size();
  for (const int type : Types<T>()) {
    SCOPED_TRACE(TypeName(type));
    float version;
    auto properties = Properties<T>(type, list);
    {
      const std::unique_ptr<PlyFile> writer(
          PlyFile::Write(path.string(), {"vertex"}, format, version));
      ASSERT_NE(writer, nullptr);
      writer->describe_element(
          "vertex", count, properties.size(), properties.data());
      writer->header_complete();
      writer->put_element_setup("vertex");
      for (size_t i = 0; i < count; ++i) {
        Record<T> record;
        record.value = values[i];
        record.count = list && i ? values.size() : 0;
        record.values = values.data();
        record.marker = 17 + i;
        writer->put_element(&record);
      }
    }
    EXPECT_EQ(
        ReadBytes(path),
        Header(format, TypeName(type), list, count) + Payload<T>(format, list));
  }
}

class PoissonPlyFloatTest : public ::testing::TestWithParam<Encoding> {
 protected:
  void Read(const bool list, const Storage storage) {
    if (GetParam().use_double) {
      CheckRead<double>(GetParam().format, list, storage);
    } else {
      CheckRead<float>(GetParam().format, list, storage);
    }
  }

  void Write(const bool list) {
    if (GetParam().use_double) {
      CheckWrite<double>(GetParam().format, list);
    } else {
      CheckWrite<float>(GetParam().format, list);
    }
  }
};

TEST_P(PoissonPlyFloatTest, ReadsScalars) { Read(false, Storage::kRequested); }
TEST_P(PoissonPlyFloatTest, ReadsLists) { Read(true, Storage::kRequested); }
TEST_P(PoissonPlyFloatTest, DiscardsScalars) {
  Read(false, Storage::kDiscarded);
}
TEST_P(PoissonPlyFloatTest, DiscardsLists) { Read(true, Storage::kDiscarded); }
TEST_P(PoissonPlyFloatTest, PreservesOtherScalars) {
  Read(false, Storage::kOther);
}
TEST_P(PoissonPlyFloatTest, PreservesOtherLists) {
  Read(true, Storage::kOther);
}
TEST_P(PoissonPlyFloatTest, WritesScalars) { Write(false); }
TEST_P(PoissonPlyFloatTest, WritesLists) { Write(true); }

INSTANTIATE_TEST_SUITE_P(Encodings,
                         PoissonPlyFloatTest,
                         ::testing::Values(Encoding{PLY_ASCII, false},
                                           Encoding{PLY_ASCII, true},
                                           Encoding{PLY_BINARY_LE, false},
                                           Encoding{PLY_BINARY_LE, true},
                                           Encoding{PLY_BINARY_BE, false},
                                           Encoding{PLY_BINARY_BE, true}),
                         [](const ::testing::TestParamInfo<Encoding>& info) {
                           const std::string format =
                               info.param.format == PLY_ASCII       ? "Ascii"
                               : info.param.format == PLY_BINARY_LE ? "Little"
                                                                    : "Big";
                           return format +
                                  (info.param.use_double ? "Double" : "Float");
                         });

class PoissonPlyIntegerTest : public ::testing::TestWithParam<int> {};

TEST_P(PoissonPlyIntegerTest, PreservesIntegerPayloadsAndCounts) {
  struct IntegerRecord {
    int32_t signed_value = 0;
    uint32_t unsigned_value = 0;
    int32_t fraction = 0;
    int16_t count = 0;
    int32_t* values = nullptr;
  };
  const int format = GetParam();
  const auto path = colmap::CreateTestDir() / "integers.ply";
  std::ostringstream header;
  header << "ply\nformat " << FormatName(format)
         << " 1.0\nelement vertex 1\nproperty int signed_value\n"
            "property uint unsigned_value\nproperty float fraction\n"
            "property list short int values\nend_header\n";
  const std::array<int32_t, 3> values = {-23, 0, 37};
  std::string payload;
  Append(payload, format, int32_t(-17));
  Append(payload, format, uint32_t(4000000000));
  Append(payload, format, 12.75f);
  Append(payload, format, int16_t(values.size()));
  for (const int32_t value : values) Append(payload, format, value);
  if (format == PLY_ASCII) payload += '\n';
  WriteBytes(path, header.str() + payload);
  std::array<PlyProperty, 4> properties = {
      PlyProperty("signed_value",
                  PLY_INT,
                  PLY_INT_32,
                  offsetof(IntegerRecord, signed_value)),
      PlyProperty("unsigned_value",
                  PLY_UINT,
                  PLY_UINT_32,
                  offsetof(IntegerRecord, unsigned_value)),
      PlyProperty(
          "fraction", PLY_FLOAT, PLY_INT_32, offsetof(IntegerRecord, fraction)),
      PlyProperty("values",
                  PLY_INT,
                  PLY_INT_32,
                  offsetof(IntegerRecord, values),
                  PLY_LIST,
                  PLY_SHORT,
                  PLY_INT_16,
                  offsetof(IntegerRecord, count))};
  std::vector<std::string> names;
  int actual_format;
  float version;
  const std::unique_ptr<PlyFile> reader(
      PlyFile::Read(path.string(), names, actual_format, version));
  ASSERT_NE(reader, nullptr);
  reader->get_element_setup("vertex", properties.size(), properties.data());
  IntegerRecord loaded;
  reader->get_element(&loaded);
  const std::unique_ptr<int32_t, decltype(&std::free)> list_data(loaded.values,
                                                                 &std::free);
  EXPECT_EQ(loaded.signed_value, -17);
  EXPECT_EQ(loaded.unsigned_value, 4000000000U);
  EXPECT_EQ(loaded.fraction, 12);
  ASSERT_EQ(loaded.count, values.size());
  ASSERT_NE(loaded.values, nullptr);
  for (size_t i = 0; i < values.size(); ++i)
    EXPECT_EQ(loaded.values[i], values[i]);

  // Real integer destinations still request conversion on scalar/list writes.
  for (const bool list : {false, true}) {
    const auto output = colmap::CreateTestDir() / "converted.ply";
    auto props = Properties<double>(PLY_DOUBLE, list);
    props[0].external_type = PLY_INT;
    std::array<double, 3> input_values = {12.75, 0.5, 23};
    Record<double> record;
    record.value = input_values[0];
    record.count = input_values.size();
    record.values = input_values.data();
    record.marker = 17;
    {
      const std::unique_ptr<PlyFile> writer(
          PlyFile::Write(output.string(), {"vertex"}, format, version));
      ASSERT_NE(writer, nullptr);
      writer->describe_element("vertex", 1, props.size(), props.data());
      writer->header_complete();
      writer->put_element_setup("vertex");
      writer->put_element(&record);
    }
    std::string expected;
    if (list) {
      Append(expected, format, uint8_t(input_values.size()));
      for (const int32_t value : {12, 0, 23}) Append(expected, format, value);
    } else {
      Append(expected, format, int32_t(12));
    }
    Append(expected, format, uint32_t(17));
    if (format == PLY_ASCII) expected += '\n';
    EXPECT_EQ(ReadBytes(output), Header(format, "int", list, 1) + expected);
  }
}

INSTANTIATE_TEST_SUITE_P(Encodings,
                         PoissonPlyIntegerTest,
                         ::testing::Values(PLY_ASCII,
                                           PLY_BINARY_LE,
                                           PLY_BINARY_BE));

TEST(PlyFloatingProperties, RoundTripsFloatingScalarAndListDestinations) {
  const auto path = colmap::CreateTestDir() / "precision.ply";
  struct Element {
    double scalar;
    int count;
    double* list;
  };
  for (int type : {PLY_FLOAT, PLY_FLOAT_32, PLY_DOUBLE, PLY_FLOAT_64}) {
    for (int format : {PLY_ASCII, PLY_BINARY_LE, PLY_BINARY_BE}) {
      SCOPED_TRACE(std::to_string(type) + "/" + std::to_string(format));
      std::vector<double> values{-2.0, 0.1, 1e20};
      if (type == PLY_DOUBLE || type == PLY_FLOAT_64) {
        // These observed world positions collapse with six ASCII digits.
        values.push_back(-0.8250000342726711);
        values.push_back(-0.8250000178813934);
      }
      Element input{values[0], static_cast<int>(values.size()), values.data()};
      PlyProperty properties[] = {
          PlyProperty("x", type, PLY_DOUBLE, offsetof(Element, scalar)),
          PlyProperty("values",
                      type,
                      PLY_DOUBLE,
                      offsetof(Element, list),
                      PLY_LIST,
                      PLY_UCHAR,
                      PLY_INT,
                      offsetof(Element, count))};
      float version;
      std::vector<std::string> names{"vertex"};
      {
        std::unique_ptr<PlyFile> file(
            PlyFile::Write(path.string(), names, format, version));
        ASSERT_NE(file, nullptr);
        file->describe_element("vertex", 1, 2, properties);
        file->header_complete();
        file->put_element_setup("vertex");
        file->put_element(&input);
      }
      Element output{};
      {
        std::unique_ptr<PlyFile> file(
            PlyFile::Read(path.string(), names, format, version));
        ASSERT_NE(file, nullptr);
        file->get_element_setup("vertex", 2, properties);
        file->get_element(&output);
      }
      std::unique_ptr<double, decltype(&std::free)> allocation(output.list,
                                                               &std::free);
      ASSERT_EQ(output.count, values.size());
      EXPECT_EQ(output.scalar, -2.0);
      for (int i = 0; i < output.count; ++i) {
        const double expected =
            type == PLY_FLOAT || type == PLY_FLOAT_32
                ? static_cast<double>(static_cast<float>(values[i]))
                : values[i];
        EXPECT_EQ(output.list[i], expected);
      }
    }
  }
}

TEST(PlyFloatingProperties, HonorsAsciiSourceWidthAndIgnoresUnusedFloats) {
  const auto path = colmap::CreateTestDir() / "precision.ply";
  {
    std::ofstream file(path.string());
    file << "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty double y\nproperty float32 z\n"
            "property double unused\nproperty list uchar double unused_list\n"
            "end_header\n0.1 0.2 0.3 -1e20 2 -2 1e20\n";
  }
  struct Position {
    double x, y, z;
  } position{};
  PlyProperty properties[] = {
      PlyProperty("x", PLY_FLOAT, PLY_DOUBLE, offsetof(Position, x)),
      PlyProperty("y", PLY_DOUBLE, PLY_DOUBLE, offsetof(Position, y)),
      PlyProperty("z", PLY_FLOAT_32, PLY_DOUBLE, offsetof(Position, z))};
  int format;
  float version;
  std::vector<std::string> names;
  std::unique_ptr<PlyFile> file(
      PlyFile::Read(path.string(), names, format, version));
  ASSERT_NE(file, nullptr);
  file->get_element_setup("vertex", 3, properties);
  file->get_element(&position);
  EXPECT_EQ(position.x, static_cast<double>(0.1f));
  EXPECT_EQ(position.y, 0.2);
  EXPECT_EQ(position.z, static_cast<double>(0.3f));
}

TEST(PlyFloatingProperties, PreservesExistingIntegerRequestedConversion) {
  const auto path = colmap::CreateTestDir() / "precision.ply";
  for (const std::string type : {"float", "float32", "double", "float64"}) {
    {
      std::ofstream file(path.string());
      file << "ply\nformat ascii 1.0\nelement vertex 1\nproperty " << type
           << " x\nend_header\n16777217\n";
    }
    // Preserve the prior integer-requested interpretation of this ASCII token.
    int result = 0;
    PlyProperty property("x", PLY_FLOAT, PLY_INT, 0);
    int format;
    float version;
    std::vector<std::string> names;
    std::unique_ptr<PlyFile> file(
        PlyFile::Read(path.string(), names, format, version));
    ASSERT_NE(file, nullptr);
    file->get_element_setup("vertex", 1, &property);
    file->get_element(&result);
    EXPECT_EQ(result, 16777217);
  }
}
}  // namespace
