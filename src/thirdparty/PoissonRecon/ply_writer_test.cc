// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/testing.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Ply.h"
#include "PreProcessor.h"
#include "VertexFactory.h"
#include <gtest/gtest.h>

namespace {

struct Vertex {
  float x, y, z;
};

struct VertexFactory {
  using VertexType = Vertex;
  static constexpr bool IsStaticallyAllocated() { return true; }
  unsigned int plyWriteNum() const { return 3; }
  unsigned int plyReadNum() const { return 3; }
  size_t bufferSize() const { return sizeof(Vertex); }
  Vertex operator()() const { return {}; }
  PoissonRecon::PlyProperty plyStaticWriteProperty(unsigned int i) const {
    const PoissonRecon::PlyProperty props[] = {
        {"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex, x)},
        {"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex, y)},
        {"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex, z)}};
    return props[i];
  }
  PoissonRecon::PlyProperty plyStaticReadProperty(unsigned int i) const {
    return plyStaticWriteProperty(i);
  }
};

template <typename T>
struct Input : PoissonRecon::InputDataStream<T> {
  explicit Input(std::vector<T> values) : values(std::move(values)) {}
  void reset() override { index = 0; }
  bool read(T& value) override {
    if (fail) throw std::runtime_error("injected stream error");
    if (index == values.size()) return false;
    value = values[index++];
    return true;
  }
  std::vector<T> values;
  size_t index = 0;
  bool fail = false;
};

TEST(PlyWriterTest, VectorRoundTripBinaryAndAscii) {
  const auto dir = colmap::CreateTestDir();
  const std::vector<Vertex> vertices = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const std::vector<std::vector<int>> faces = {{0, 1, 2}};
  for (const int type : {PLY_ASCII, PLY_BINARY_LE, PLY_BINARY_BE}) {
    const auto path = dir / (std::to_string(type) + ".ply");
    PoissonRecon::PLY::WritePolygons<VertexFactory, int>(
        path.string(), VertexFactory{}, vertices, faces, type, {});
    std::vector<Vertex> decoded;
    std::vector<std::vector<int>> decoded_faces;
    std::vector<std::string> comments;
    int read_type = 0;
    PoissonRecon::PLY::ReadPolygons<VertexFactory, int>(path.string(),
                                                        VertexFactory{},
                                                        decoded,
                                                        decoded_faces,
                                                        read_type,
                                                        comments);
    EXPECT_EQ(read_type, type);
    ASSERT_EQ(decoded.size(), vertices.size());
    EXPECT_EQ(decoded_faces, faces);
    for (size_t i = 0; i < vertices.size(); ++i) {
      EXPECT_FLOAT_EQ(decoded[i].x, vertices[i].x);
      EXPECT_FLOAT_EQ(decoded[i].y, vertices[i].y);
      EXPECT_FLOAT_EQ(decoded[i].z, vertices[i].z);
    }
  }
}

template <typename OutputIndex>
void CompareStreamingAndVectorWriter(const std::filesystem::path& dir,
                                     const int type,
                                     const bool empty) {
  const std::vector<Vertex> vertices = {
      {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  const std::vector<std::vector<int>> faces =
      empty ? std::vector<std::vector<int>>{}
            : std::vector<std::vector<int>>{
                  {0, 1, 2}, {}, {0, 1, 2, 3}, {3, 2, 1}};
  std::vector<std::vector<OutputIndex>> converted;
  converted.reserve(faces.size());
  for (const auto& face : faces)
    converted.emplace_back(face.begin(), face.end());
  Input<Vertex> vertex_stream(vertices);
  Input<std::vector<int>> face_stream(faces);
  const std::string suffix = std::to_string(type) + "-" +
                             std::to_string(empty) + "-" +
                             std::to_string(sizeof(OutputIndex)) + "-" +
                             std::to_string(std::is_signed_v<OutputIndex>);
  const auto stream_path = dir / ("stream-" + suffix + ".ply");
  const auto vector_path = dir / ("vector-" + suffix + ".ply");
  PoissonRecon::PLY::Write<VertexFactory, int, float, 3, OutputIndex>(
      stream_path.string(),
      VertexFactory{},
      vertices.size(),
      faces.size(),
      vertex_stream,
      face_stream,
      type,
      {});
  PoissonRecon::PLY::WritePolygons<VertexFactory, OutputIndex>(
      vector_path.string(), VertexFactory{}, vertices, converted, type, {});
  std::ifstream stream(stream_path, std::ios::binary);
  std::ifstream vector(vector_path, std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  ASSERT_TRUE(vector.is_open());
  const std::vector<char> stream_bytes((std::istreambuf_iterator<char>(stream)),
                                       std::istreambuf_iterator<char>());
  const std::vector<char> vector_bytes((std::istreambuf_iterator<char>(vector)),
                                       std::istreambuf_iterator<char>());
  EXPECT_EQ(stream_bytes, vector_bytes);
  std::vector<Vertex> decoded;
  std::vector<std::vector<OutputIndex>> decoded_faces;
  std::vector<std::string> comments;
  int decoded_type = 0;
  PoissonRecon::PLY::ReadPolygons<VertexFactory, OutputIndex>(
      stream_path.string(),
      VertexFactory{},
      decoded,
      decoded_faces,
      decoded_type,
      comments);
  EXPECT_EQ(decoded_type, type);
  EXPECT_EQ(decoded_faces, converted);
  ASSERT_EQ(decoded.size(), vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i) {
    EXPECT_FLOAT_EQ(decoded[i].x, vertices[i].x);
    EXPECT_FLOAT_EQ(decoded[i].y, vertices[i].y);
    EXPECT_FLOAT_EQ(decoded[i].z, vertices[i].z);
  }
}

TEST(PlyWriterTest, StreamingFacesMatchVectorWireFormat) {
  const auto dir = colmap::CreateTestDir();
  for (const int type : {PLY_ASCII, PLY_BINARY_LE, PLY_BINARY_BE}) {
    for (const bool empty : {false, true}) {
      SCOPED_TRACE(type);
      SCOPED_TRACE(empty);
      CompareStreamingAndVectorWriter<int>(dir, type, empty);
      CompareStreamingAndVectorWriter<unsigned int>(dir, type, empty);
      CompareStreamingAndVectorWriter<long long>(dir, type, empty);
    }
  }
}

TEST(PlyReaderTest, RejectsInvalidHeadersAndReadsValidFile) {
  const auto dir = colmap::CreateTestDir();
  const std::vector<std::string> headers = {
      "not-ply\n",
      "ply\nformat unsupported 1.0\nend_header\n",
      "ply\nelement vertex 0\nend_header\n",
      "ply\nformat ascii 1.0\nelement vertex 0\n"};
  for (size_t i = 0; i < headers.size(); ++i) {
    const auto path = dir / ("invalid-" + std::to_string(i) + ".ply");
    {
      std::ofstream output(path, std::ios::binary);
      ASSERT_TRUE(output.is_open());
      output << headers[i];
      ASSERT_TRUE(output.good());
    }
    int type = -7;
    float version = -3;
    std::vector<std::string> names;
    std::unique_ptr<PoissonRecon::PlyFile> reader(
        PoissonRecon::PlyFile::Read(path.string(), names, type, version));
    EXPECT_EQ(reader, nullptr);
    EXPECT_EQ(type, -7);
    EXPECT_FLOAT_EQ(version, -3);
  }

  const auto valid = dir / "valid.ply";
  {
    std::ofstream output(valid, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";
    ASSERT_TRUE(output.good());
  }
  int type = -7;
  float version = -3;
  std::vector<std::string> names;
  std::unique_ptr<PoissonRecon::PlyFile> reader(
      PoissonRecon::PlyFile::Read(valid.string(), names, type, version));
  ASSERT_NE(reader, nullptr);
  EXPECT_EQ(type, PLY_ASCII);
  EXPECT_FLOAT_EQ(version, 1.0f);
  EXPECT_EQ(names, std::vector<std::string>{"vertex"});
}

TEST(PlyReaderTest, RejectsScalarFacePropertyBeforeListDuplicate) {
  const auto dir = colmap::CreateTestDir();
  for (const std::string name : {"vertex_indices", "vertex_index"}) {
    const auto path = dir / (name + ".ply");
    {
      std::ofstream output(path);
      ASSERT_TRUE(output.is_open());
      output << "ply\nformat ascii 1.0\n"
                "element vertex 0\nproperty float x\nproperty float y\n"
                "property float z\nelement face 1\nproperty int "
             << name << "\nproperty list uchar int " << name
             << "\nend_header\n";
      ASSERT_TRUE(output.good());
    }
    std::vector<Vertex> vertices;
    std::vector<std::vector<int>> polygons;
    std::vector<std::string> comments;
    int type = 0;
    try {
      PoissonRecon::PLY::ReadPolygons<VertexFactory, int>(
          path.string(), VertexFactory{}, vertices, polygons, type, comments);
      FAIL() << "Expected a face-list validation error";
    } catch (const std::exception& error) {
      EXPECT_NE(
          std::string(error.what()).find("Could not read face vertex indices"),
          std::string::npos);
    }
  }
}

TEST(PlyReaderTest, RejectsTruncatedFaceListAndReadsValidList) {
  const auto dir = colmap::CreateTestDir();
  const std::vector<Vertex> vertices = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const std::vector<std::vector<int>> faces = {{0, 1, 2}};
  for (const int format : {PLY_BINARY_LE, PLY_BINARY_BE}) {
    SCOPED_TRACE(format);
    const auto valid = dir / (std::to_string(format) + "-valid.ply");
    const auto truncated = dir / (std::to_string(format) + "-truncated.ply");
    PoissonRecon::PLY::WritePolygons<VertexFactory, int>(
        valid.string(), VertexFactory{}, vertices, faces, format, {});
    std::filesystem::copy_file(valid, truncated);
    // Leave one index after the list count. LeakSanitizer must also verify
    // cleanup when the reader throws after allocating the complete list.
    std::filesystem::resize_file(
        truncated, std::filesystem::file_size(truncated) - 2 * sizeof(int));
    std::vector<Vertex> decoded;
    std::vector<std::vector<int>> decoded_faces;
    std::vector<std::string> comments;
    int read_format = 0;
    try {
      PoissonRecon::PLY::ReadPolygons<VertexFactory, int>(truncated.string(),
                                                          VertexFactory{},
                                                          decoded,
                                                          decoded_faces,
                                                          read_format,
                                                          comments);
      FAIL() << "Expected a truncated face-list error";
    } catch (const std::exception& error) {
      EXPECT_NE(std::string(error.what()).find("fread() failed"),
                std::string::npos);
    }
    PoissonRecon::PLY::ReadPolygons<VertexFactory, int>(valid.string(),
                                                        VertexFactory{},
                                                        decoded,
                                                        decoded_faces,
                                                        read_format,
                                                        comments);
    EXPECT_EQ(read_format, format);
    EXPECT_EQ(decoded_faces, faces);
    ASSERT_EQ(decoded.size(), vertices.size());
  }
}

TEST(PlyReaderTest, ClosesFileAfterTruncatedBinaryBody) {
  const auto path = colmap::CreateTestDir() / "truncated.ply";
  {
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << "ply\nformat binary_little_endian 1.0\n"
              "element vertex 1\nproperty float x\nproperty float y\n"
              "property float z\nend_header\n";
    ASSERT_TRUE(output.good());
  }
#if defined(__linux__)
  const auto expected = std::filesystem::canonical(path);
  const auto count_open = [&]() {
    size_t count = 0;
    for (const auto& fd :
         std::filesystem::directory_iterator("/proc/self/fd")) {
      std::error_code error;
      const auto target = std::filesystem::canonical(fd.path(), error);
      if (!error && target == expected) ++count;
    }
    return count;
  };
#endif
  for (bool read_edges : {false, true}) {
#if defined(__linux__)
    const size_t before = count_open();
#endif
    std::vector<Vertex> vertices;
    std::vector<std::vector<int>> polygons;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::string> comments;
    int type = 0;
    if (read_edges) {
      EXPECT_ANY_THROW((PoissonRecon::PLY::ReadEdges<VertexFactory, int>(
          path.string(), VertexFactory{}, vertices, edges, type, comments)));
    } else {
      EXPECT_ANY_THROW((PoissonRecon::PLY::ReadPolygons<VertexFactory, int>(
          path.string(), VertexFactory{}, vertices, polygons, type, comments)));
    }
#if defined(__linux__)
    EXPECT_EQ(count_open(), before);
#endif
  }
}

#if defined(__linux__)
TEST(PlyWriterTest, ReportsDeferredWriteError) {
  const auto path = colmap::CreateTestDir() / "full.ply";
  std::error_code error;
  std::filesystem::create_symlink("/dev/full", path, error);
  if (error)
    GTEST_SKIP() << "Cannot create /dev/full symlink: " << error.message();
  EXPECT_ANY_THROW((PoissonRecon::PLY::WritePolygons<VertexFactory, int>(
      path.string(),
      VertexFactory{},
      std::vector<Vertex>{{1, 2, 3}},
      std::vector<std::vector<int>>{},
      PLY_BINARY_NATIVE,
      {})));
}

TEST(PlyWriterTest, ClosesFileAfterSourceStreamThrows) {
  const auto path = colmap::CreateTestDir() / "throw.ply";
  Input<Vertex> vertices(std::vector<Vertex>{{1, 2, 3}});
  Input<std::vector<int>> faces(std::vector<std::vector<int>>{});
  vertices.fail = true;
  EXPECT_ANY_THROW(
      (PoissonRecon::PLY::Write<VertexFactory, int, float, 3>(path.string(),
                                                              VertexFactory{},
                                                              1,
                                                              0,
                                                              vertices,
                                                              faces,
                                                              PLY_BINARY_NATIVE,
                                                              {})));
  // A descriptor to the output remaining open would retain this exact target.
  const auto expected = std::filesystem::canonical(path);
  for (const auto& fd : std::filesystem::directory_iterator("/proc/self/fd")) {
    std::error_code error;
    const auto target = std::filesystem::canonical(fd.path(), error);
    if (!error) EXPECT_NE(target, expected);
  }
}
#endif

TEST(PlyScalarProperties, PreservesDynamicFactoryScalarTypes) {
  struct Scalar {
    const char* name;
    int input_type;
    int output_type;
    float value;
  };
  // Keep values representable in every legacy intermediate conversion.
  // Property descriptors independently check signedness and scalar widths.
  // Existing canonical output spellings remain unchanged for working types.
  const std::array<Scalar, 20> scalars{{
      {"char", PLY_CHAR, PLY_CHAR, 7},
      {"short", PLY_SHORT, PLY_INT_16, 7},
      {"int", PLY_INT, PLY_INT, 7},
      {"longlong", PLY_LONGLONG, PLY_INT_64, 7},
      {"uchar", PLY_UCHAR, PLY_UCHAR, 7},
      {"ushort", PLY_USHORT, PLY_UINT_16, 7},
      {"uint", PLY_UINT, PLY_UINT, 7},
      {"ulonglong", PLY_ULONGLONG, PLY_UINT_64, 7},
      {"float", PLY_FLOAT, PLY_FLOAT, 1.25f},
      {"double", PLY_DOUBLE, PLY_DOUBLE, 1.25f},
      {"int8", PLY_INT_8, PLY_INT_8, 7},
      {"uint8", PLY_UINT_8, PLY_UCHAR, 7},
      {"int16", PLY_INT_16, PLY_INT_16, 7},
      {"uint16", PLY_UINT_16, PLY_UINT_16, 7},
      {"int32", PLY_INT_32, PLY_INT, 7},
      {"uint32", PLY_UINT_32, PLY_UINT, 7},
      {"int64", PLY_INT_64, PLY_INT_64, 7},
      {"uint64", PLY_UINT_64, PLY_UINT_64, 7},
      {"float32", PLY_FLOAT_32, PLY_FLOAT, 1.25f},
      {"float64", PLY_FLOAT_64, PLY_DOUBLE, 1.25f},
  }};
  const auto dir = colmap::CreateTestDir();
  const auto input = dir / "scalar-types.ply";
  {
    std::ofstream file(input);
    ASSERT_TRUE(file.is_open());
    file << "ply\nformat ascii 1.0\nelement vertex 1\n";
    for (size_t i = 0; i < scalars.size(); ++i) {
      file << "property " << scalars[i].name << " metadata_" << i << '\n';
    }
    file << "element face 0\nproperty list uchar int vertex_indices\nend_header\n";
    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto& scalar : scalars) {
      if (scalar.input_type == PLY_FLOAT || scalar.input_type == PLY_DOUBLE ||
          scalar.input_type == PLY_FLOAT_32 || scalar.input_type == PLY_FLOAT_64) {
        file << scalar.value;
      } else {
        file << static_cast<int64_t>(scalar.value);
      }
      file << ' ';
    }
    file << '\n';
    ASSERT_TRUE(file.good());
  }
  std::vector<PoissonRecon::PlyProperty> properties;
  PoissonRecon::PLY::ReadVertexHeader(input.string(), properties);
  ASSERT_EQ(properties.size(), scalars.size());
  using Factory = PoissonRecon::VertexFactory::DynamicFactory<float>;
  const Factory factory(properties);
  for (size_t i = 0; i < scalars.size(); ++i) {
    SCOPED_TRACE(scalars[i].name);
    EXPECT_EQ(properties[i].external_type, scalars[i].input_type);
    EXPECT_EQ(factory.plyReadProperty(i).external_type, scalars[i].output_type);
    EXPECT_EQ(factory.plyWriteProperty(i).external_type, scalars[i].output_type);
  }
  std::vector<Factory::VertexType> vertices;
  std::vector<std::vector<int>> faces;
  std::vector<std::string> comments;
  int input_format = 0;
  PoissonRecon::PLY::ReadPolygons<Factory, int>(
      input.string(), factory, vertices, faces, input_format, comments);
  ASSERT_EQ(vertices.size(), 1);
  EXPECT_TRUE(faces.empty());
  for (size_t i = 0; i < scalars.size(); ++i) {
    SCOPED_TRACE(scalars[i].name);
    EXPECT_EQ(vertices[0][i], scalars[i].value);
  }
  for (const int format : {PLY_ASCII, PLY_BINARY_LE, PLY_BINARY_BE}) {
    SCOPED_TRACE(format);
    const auto output = dir / ("roundtrip-" + std::to_string(format) + ".ply");
    PoissonRecon::PLY::WritePolygons<Factory, int>(
        output.string(), factory, vertices, faces, format, {});
    std::vector<PoissonRecon::PlyProperty> written_properties;
    PoissonRecon::PLY::ReadVertexHeader(output.string(), written_properties);
    ASSERT_EQ(written_properties.size(), scalars.size());
    std::vector<Factory::VertexType> decoded;
    std::vector<std::vector<int>> decoded_faces;
    int read_format = 0;
    PoissonRecon::PLY::ReadPolygons<Factory, int>(
        output.string(), factory, decoded, decoded_faces, read_format, comments);
    EXPECT_EQ(read_format, format);
    EXPECT_TRUE(decoded_faces.empty());
    ASSERT_EQ(decoded.size(), 1);
    for (size_t i = 0; i < scalars.size(); ++i) {
      SCOPED_TRACE(scalars[i].name);
      EXPECT_EQ(written_properties[i].external_type, scalars[i].output_type);
      EXPECT_EQ(decoded[0][i], scalars[i].value);
    }
  }
}

TEST(PlyScalarProperties, WritesOnlySelectedFloatingRepresentations) {
  const auto dir = colmap::CreateTestDir();
  int case_index = 0;
  const auto check_source = [&](auto source_zero, int source_type) {
    using Source = decltype(source_zero);
    const auto check = [&](double value, auto expected, int destination_type) {
      using Destination = decltype(expected);
      struct Input {
        Source scalar;
        unsigned char count;
        Source* list;
      };
      struct Output {
        Destination scalar = 0;
        unsigned char count = 0;
        Destination* list = nullptr;
      };
      std::array<Source, 2> values{{static_cast<Source>(value), Source(7)}};
      Input input{values[0], 2, values.data()};
      PoissonRecon::PlyProperty write_properties[] = {
          {"scalar", destination_type, source_type, offsetof(Input, scalar)},
          {"list", destination_type, source_type, offsetof(Input, list),
           1, PLY_UCHAR, PLY_UCHAR, offsetof(Input, count)}};
      PoissonRecon::PlyProperty read_properties[] = {
          {"scalar", destination_type, destination_type, offsetof(Output, scalar)},
          {"list", destination_type, destination_type, offsetof(Output, list),
           1, PLY_UCHAR, PLY_UCHAR, offsetof(Output, count)}};
      for (int format : {PLY_ASCII, PLY_BINARY_LE, PLY_BINARY_BE}) {
        SCOPED_TRACE(source_type);
        SCOPED_TRACE(destination_type);
        SCOPED_TRACE(format);
        const auto path = dir / ("selected-" + std::to_string(case_index++) + ".ply");
        float version = 1;
        std::vector<std::string> names{"vertex"};
        {
          std::unique_ptr<PoissonRecon::PlyFile> file(
              PoissonRecon::PlyFile::Write(path.string(), names, format, version));
          ASSERT_NE(file, nullptr);
          file->describe_element("vertex", 1, 2, write_properties);
          file->header_complete();
          file->put_element_setup("vertex");
          file->put_element(&input);
        }
        Output output;
        {
          std::unique_ptr<PoissonRecon::PlyFile> file(
              PoissonRecon::PlyFile::Read(path.string(), names, format, version));
          ASSERT_NE(file, nullptr);
          file->get_element_setup("vertex", 2, read_properties);
          file->get_element(&output);
        }
        std::unique_ptr<Destination, decltype(&std::free)> allocation(output.list,
                                                                    &std::free);
        EXPECT_EQ(output.scalar, expected);
        ASSERT_EQ(output.count, 2);
        ASSERT_NE(output.list, nullptr);
        EXPECT_EQ(output.list[0], expected);
        EXPECT_EQ(output.list[1], Destination(7));
      }
    };
    // Each value fits its selected destination and both floating source types.
    // Wide values are written as integer payloads, then decoded as integers.
    check(-7.75, -7, PLY_INT);
    check(2147483648.0, 2147483648u, PLY_UINT);
    check(-1099511627776.0, -(1LL << 40), PLY_INT_64);
    check(1099511627776.0, 1ULL << 40, PLY_UINT_64);
    check(-1.25, -1.25f, PLY_FLOAT);
    check(-1.25, -1.25, PLY_DOUBLE);
  };
  check_source(float{}, PLY_FLOAT);
  check_source(double{}, PLY_DOUBLE);
}

TEST(PlyScalarProperties, ReadsOnlySelectedAsciiFloatingRepresentations) {
  const auto dir = colmap::CreateTestDir();
  int case_index = 0;
  for (const auto* source_type : {"float", "float32", "double", "float64"}) {
    const auto check = [&](const char* token, auto expected, int destination_type) {
      using Destination = decltype(expected);
      struct Output {
        Destination scalar = 0;
        unsigned char count = 0;
        Destination* list = nullptr;
      };
      SCOPED_TRACE(source_type);
      SCOPED_TRACE(destination_type);
      const auto path = dir / ("selected-" + std::to_string(case_index++) + ".ply");
      {
        std::ofstream file(path);
        ASSERT_TRUE(file.is_open());
        file << "ply\nformat ascii 1.0\nelement vertex 1\nproperty "
             << source_type << " scalar\nproperty list uchar " << source_type
             << " list\nend_header\n" << token << " 2 " << token << " 7\n";
        ASSERT_TRUE(file.good());
      }
      PoissonRecon::PlyProperty properties[] = {
          {"scalar", PLY_DOUBLE, destination_type, offsetof(Output, scalar)},
          {"list", PLY_DOUBLE, destination_type, offsetof(Output, list),
           1, PLY_UCHAR, PLY_UCHAR, offsetof(Output, count)}};
      Output output;
      int format = 0;
      float version = 1;
      std::vector<std::string> names;
      {
        std::unique_ptr<PoissonRecon::PlyFile> file(
            PoissonRecon::PlyFile::Read(path.string(), names, format, version));
        ASSERT_NE(file, nullptr);
        file->get_element_setup("vertex", 2, properties);
        file->get_element(&output);
      }
      std::unique_ptr<Destination, decltype(&std::free)> allocation(output.list,
                                                                  &std::free);
      EXPECT_EQ(output.scalar, expected);
      ASSERT_EQ(output.count, 2);
      ASSERT_NE(output.list, nullptr);
      EXPECT_EQ(output.list[0], expected);
      EXPECT_EQ(output.list[1], Destination(7));
    };
    check("-7.75", -7, PLY_INT);
    check("2147483648", 2147483648u, PLY_UINT);
    check("-1099511627776", -(1LL << 40), PLY_INT_64);
    check("1099511627776", 1ULL << 40, PLY_UINT_64);
    check("-1.25", -1.25f, PLY_FLOAT);
    check("-1.25", -1.25, PLY_DOUBLE);
  }
}

TEST(PlyScalarProperties, WidensUnsigned32ScalarsAndLists) {
  const auto dir = colmap::CreateTestDir();
  const std::array<unsigned int, 3> values{
      0x7fffffffU, 0x80000000U, 0xffffffffU};
  for (const int source_type : {PLY_UINT, PLY_UINT_32}) {
    for (const int format : {PLY_ASCII, PLY_BINARY_LE, PLY_BINARY_BE}) {
      for (const bool wide_count : {false, true}) {
        SCOPED_TRACE(source_type);
        SCOPED_TRACE(format);
        SCOPED_TRACE(wide_count);
        const auto path = dir / "unsigned32.ply";
        {
          std::ofstream file(path, std::ios::binary);
          ASSERT_TRUE(file.is_open());
          const char* format_name = format == PLY_ASCII ? "ascii"
                                    : format == PLY_BINARY_LE
                                        ? "binary_little_endian"
                                        : "binary_big_endian";
          const char* source_name = source_type == PLY_UINT ? "uint" : "uint32";
          file << "ply\nformat " << format_name << " 1.0\nelement vertex "
               << values.size() << "\nproperty " << source_name
               << " scalar\nproperty list " << (wide_count ? "int" : "uchar")
               << ' ' << source_name << " list\nend_header\n";
          // Encode bytes independently of the PLY writer and host endianness.
          const auto append = [&](unsigned int value, int width) {
            if (format == PLY_ASCII) {
              file << value << ' ';
            } else {
              std::array<unsigned char, 4> bytes{};
              for (int i = 0; i < width; ++i) {
                const int byte = format == PLY_BINARY_BE ? width - i - 1 : i;
                bytes[i] = static_cast<unsigned char>(value >> (8 * byte));
              }
              file.write(reinterpret_cast<const char*>(bytes.data()), width);
            }
          };
          for (const unsigned int value : values) {
            append(value, 4);
            append(static_cast<unsigned int>(values.size()),
                   wide_count ? 4 : 1);
            for (const unsigned int item : values) append(item, 4);
            if (format == PLY_ASCII) file << '\n';
          }
          ASSERT_TRUE(file.good());
        }
        const auto check = [&](auto zero, int destination_type) {
          using Destination = decltype(zero);
          struct Output {
            Destination scalar = 0;
            int count = 0;
            Destination* list = nullptr;
          };
          SCOPED_TRACE(destination_type);
          PoissonRecon::PlyProperty properties[] = {
              {"scalar",
               source_type,
               destination_type,
               offsetof(Output, scalar)},
              {"list",
               source_type,
               destination_type,
               offsetof(Output, list),
               PLY_LIST,
               wide_count ? PLY_INT : PLY_UCHAR,
               PLY_INT,
               offsetof(Output, count)}};
          int read_format = 0;
          float version = 0;
          std::vector<std::string> names;
          std::unique_ptr<PoissonRecon::PlyFile> reader(
              PoissonRecon::PlyFile::Read(
                  path.string(), names, read_format, version));
          ASSERT_NE(reader, nullptr);
          EXPECT_EQ(read_format, format);
          reader->get_element_setup("vertex", 2, properties);
          for (const unsigned int value : values) {
            Output output;
            reader->get_element(&output);
            std::unique_ptr<Destination, decltype(&std::free)> allocation(
                output.list, &std::free);
            EXPECT_EQ(output.scalar, static_cast<Destination>(value));
            ASSERT_EQ(output.count, values.size());
            ASSERT_NE(output.list, nullptr);
            for (size_t i = 0; i < values.size(); ++i) {
              EXPECT_EQ(output.list[i], static_cast<Destination>(values[i]));
            }
          }
        };
        check(0LL, PLY_LONGLONG);
        check(0LL, PLY_INT_64);
        check(0ULL, PLY_ULONGLONG);
        check(0ULL, PLY_UINT_64);
      }
    }
  }
}

}  // namespace
