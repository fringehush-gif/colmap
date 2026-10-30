// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/testing.h"

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
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
  for (const int type : {PLY_BINARY_NATIVE, PLY_ASCII}) {
    const auto path = dir / (type == PLY_ASCII ? "ascii.ply" : "binary.ply");
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
    ASSERT_EQ(decoded.size(), vertices.size());
    EXPECT_EQ(decoded_faces, faces);
    for (size_t i = 0; i < vertices.size(); ++i) {
      EXPECT_FLOAT_EQ(decoded[i].x, vertices[i].x);
      EXPECT_FLOAT_EQ(decoded[i].y, vertices[i].y);
      EXPECT_FLOAT_EQ(decoded[i].z, vertices[i].z);
    }
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
