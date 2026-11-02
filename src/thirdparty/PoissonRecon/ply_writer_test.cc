// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/testing.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Ply.h"
#include "PreProcessor.h"
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

}  // namespace
