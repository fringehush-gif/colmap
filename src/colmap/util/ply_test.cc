// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/ply.h"

#include "colmap/util/endian.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <array>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef __linux__
#include <sys/resource.h>
#endif  // __linux__

namespace colmap {
namespace {

// Write a binary PLY mesh with the given vertex properties (e.g. "short x")
// and triangular faces. The payload callback receives a Write function that
// encodes one scalar in the file's endianness; it must write exactly
// num_vertices vertices, followed by the raw faces if face_properties is set.
// Otherwise, faces are appended using the default uchar/int layout.
// Pass explicitly typed values (e.g. int16_t(-300))
// so that scalars get the intended width.
template <typename WritePayload>
std::filesystem::path WriteBinaryPlyMesh(
    const std::string& file_name,
    const std::string& format,
    size_t num_vertices,
    const std::vector<std::string>& properties,
    WritePayload&& write_payload,
    const std::vector<std::array<int, 3>>& faces = {},
    const std::string& face_properties = {},
    size_t raw_face_count = 0) {
  const auto test_file = CreateTestDir() / file_name;
  {
    std::ofstream text_file(test_file);
    text_file << "ply\n";
    text_file << "format " << format << " 1.0\n";
    text_file << "element vertex " << num_vertices << "\n";
    for (const auto& property : properties) {
      text_file << "property " << property << "\n";
    }
    text_file << "element face "
              << (face_properties.empty() ? faces.size() : raw_face_count)
              << "\n";
    text_file << (face_properties.empty()
                      ? "property list uchar int vertex_index\n"
                      : face_properties);
    text_file << "end_header\n";
    text_file.close();

    std::ofstream binary_file(test_file,
                              std::ios::out | std::ios::binary | std::ios::app);
    const bool little_endian = (format == "binary_little_endian");
    const auto Write = [&](auto value) {
      const auto converted = little_endian ? NativeToLittleEndian(value)
                                           : NativeToBigEndian(value);
      binary_file.write(reinterpret_cast<const char*>(&converted),
                        sizeof(converted));
    };
    write_payload(Write);
    for (const auto& face : faces) {
      Write(uint8_t(3));
      for (int idx : face) {
        Write(int32_t(idx));
      }
    }
  }
  return test_file;
}

// Write an ASCII PLY mesh with the given vertex properties, vertex lines,
// and face lines.
std::filesystem::path WriteTextPlyMesh(
    const std::string& file_name,
    const std::vector<std::string>& properties,
    const std::vector<std::string>& vertex_lines,
    const std::vector<std::string>& face_lines = {}) {
  const auto test_file = CreateTestDir() / file_name;
  std::ofstream file(test_file);
  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << vertex_lines.size() << "\n";
  for (const auto& property : properties) {
    file << "property " << property << "\n";
  }
  file << "element face " << face_lines.size() << "\n";
  file << "property list uchar int vertex_index\n";
  file << "end_header\n";
  for (const auto& vertex_line : vertex_lines) {
    file << vertex_line << "\n";
  }
  for (const auto& face_line : face_lines) {
    file << face_line << "\n";
  }
  return test_file;
}

void ExpectPlyMeshError(const std::filesystem::path& path,
                        const std::string& expected_message) {
  try {
    ReadPlyMesh(path);
    ADD_FAILURE() << "Expected PLY mesh read to fail: " << path;
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find(expected_message),
              std::string::npos)
        << error.what();
  }
}

// Check a mesh vertex or point position.
template <typename Vertex>
void ExpectPositionEq(const Vertex& vertex, float x, float y, float z) {
  EXPECT_EQ(vertex.x, x);
  EXPECT_EQ(vertex.y, y);
  EXPECT_EQ(vertex.z, z);
}

// Check a point normal.
void ExpectNormalEq(const PlyPoint& point, float nx, float ny, float nz) {
  EXPECT_EQ(point.nx, nx);
  EXPECT_EQ(point.ny, ny);
  EXPECT_EQ(point.nz, nz);
}

// Check a mesh vertex or point color.
template <typename Vertex>
void ExpectColorEq(const Vertex& vertex, int r, int g, int b) {
  EXPECT_EQ(vertex.r, r);
  EXPECT_EQ(vertex.g, g);
  EXPECT_EQ(vertex.b, b);
}

TEST(PlyPoint, DefaultConstructor) {
  PlyPoint point;
  ExpectPositionEq(point, 0.0f, 0.0f, 0.0f);
  ExpectNormalEq(point, 0.0f, 0.0f, 0.0f);
  ExpectColorEq(point, 0, 0, 0);
}

TEST(PlyMeshVertex, DefaultConstructor) {
  PlyMeshVertex vertex;
  ExpectPositionEq(vertex, 0.0f, 0.0f, 0.0f);
  ExpectColorEq(vertex, 200, 200, 200);
}

TEST(PlyMeshVertex, ParameterizedConstructor) {
  PlyMeshVertex vertex(1.5f, 2.5f, 3.5f);
  ExpectPositionEq(vertex, 1.5f, 2.5f, 3.5f);
  ExpectColorEq(vertex, 200, 200, 200);
}

TEST(PlyMeshVertex, ColorConstructor) {
  PlyMeshVertex vertex(1.0f, 2.0f, 3.0f, 10, 20, 30);
  ExpectPositionEq(vertex, 1.0f, 2.0f, 3.0f);
  ExpectColorEq(vertex, 10, 20, 30);
}

TEST(PlyMeshFace, DefaultConstructor) {
  PlyMeshFace face;
  EXPECT_EQ(face.vertex_idx1, 0);
  EXPECT_EQ(face.vertex_idx2, 0);
  EXPECT_EQ(face.vertex_idx3, 0);
}

TEST(PlyMeshFace, ParameterizedConstructor) {
  PlyMeshFace face(10, 20, 30);
  EXPECT_EQ(face.vertex_idx1, 10);
  EXPECT_EQ(face.vertex_idx2, 20);
  EXPECT_EQ(face.vertex_idx3, 30);
}

#ifdef __linux__
TEST(Ply, FailedOutputIsReported) {
  const std::filesystem::path path = "/dev/full";
  const std::vector<PlyPoint> points(1);
  const PlyTexturedMesh mesh;
  const auto throws_write_error = testing::ThrowsMessage<std::invalid_argument>(
      testing::HasSubstr("Failed to write PLY file"));
  for (const bool normals : {false, true}) {
    for (const bool colors : {false, true}) {
      EXPECT_THAT([&] { WriteTextPlyPoints(path, points, normals, colors); },
                  throws_write_error);
      EXPECT_THAT([&] { WriteBinaryPlyPoints(path, points, normals, colors); },
                  throws_write_error);
    }
  }
  EXPECT_THAT([&] { WriteTextPlyMesh(path, mesh); }, throws_write_error);
  EXPECT_THAT([&] { WriteBinaryPlyMesh(path, mesh); }, throws_write_error);
}
#endif  // __linux__

TEST(Ply, RoundTripTextPlyPointsFullData) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "test.ply";

  std::vector<PlyPoint> original_points;

  // Create test points with full data
  for (int i = 0; i < 3; ++i) {
    PlyPoint p;
    p.x = i * 1.0f;
    p.y = i * 2.0f;
    p.z = i * 3.0f;
    p.nx = i * 0.1f;
    p.ny = i * 0.2f;
    p.nz = i * 0.3f;
    p.r = i * 10;
    p.g = i * 20;
    p.b = i * 30;
    original_points.push_back(p);
  }

  WriteTextPlyPoints(test_file, original_points, true, true);

  std::vector<PlyPoint> loaded_points = ReadPly(test_file);

  ASSERT_EQ(loaded_points.size(), original_points.size());

  for (size_t i = 0; i < original_points.size(); ++i) {
    ExpectPositionEq(loaded_points[i],
                     original_points[i].x,
                     original_points[i].y,
                     original_points[i].z);
    ExpectNormalEq(loaded_points[i],
                   original_points[i].nx,
                   original_points[i].ny,
                   original_points[i].nz);
    ExpectColorEq(loaded_points[i],
                  original_points[i].r,
                  original_points[i].g,
                  original_points[i].b);
  }
}

TEST(Ply, RoundTripTextPlyPointsXYZOnly) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "test.ply";

  std::vector<PlyPoint> original_points;

  PlyPoint p;
  p.x = 1.5f;
  p.y = 2.5f;
  p.z = 3.5f;
  p.nx = 0.15f;
  p.ny = 0.25f;
  p.nz = 0.35f;
  p.r = 15;
  p.g = 25;
  p.b = 35;
  original_points.push_back(p);

  WriteTextPlyPoints(test_file, original_points, false, false);

  std::vector<PlyPoint> loaded_points = ReadPly(test_file);

  ASSERT_EQ(loaded_points.size(), 1);
  ExpectPositionEq(loaded_points[0], 1.5f, 2.5f, 3.5f);
  // Normals and colors should be default (0)
  ExpectNormalEq(loaded_points[0], 0.0f, 0.0f, 0.0f);
  ExpectColorEq(loaded_points[0], 0, 0, 0);
}

TEST(Ply, RoundTripBinaryPlyPointsFullData) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "test.ply";

  std::vector<PlyPoint> original_points;

  // Create test points
  for (int i = 0; i < 5; ++i) {
    PlyPoint p;
    p.x = i * 1.5f;
    p.y = i * 2.5f;
    p.z = i * 3.5f;
    p.nx = i * 0.15f;
    p.ny = i * 0.25f;
    p.nz = i * 0.35f;
    p.r = i * 15;
    p.g = i * 25;
    p.b = i * 35;
    original_points.push_back(p);
  }

  WriteBinaryPlyPoints(test_file, original_points, true, true);

  std::vector<PlyPoint> loaded_points = ReadPly(test_file);

  ASSERT_EQ(loaded_points.size(), original_points.size());

  for (size_t i = 0; i < original_points.size(); ++i) {
    ExpectPositionEq(loaded_points[i],
                     original_points[i].x,
                     original_points[i].y,
                     original_points[i].z);
    ExpectNormalEq(loaded_points[i],
                   original_points[i].nx,
                   original_points[i].ny,
                   original_points[i].nz);
    ExpectColorEq(loaded_points[i],
                  original_points[i].r,
                  original_points[i].g,
                  original_points[i].b);
  }
}

TEST(Ply, RoundTripBinaryPlyPointsXYZOnly) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "test.ply";

  std::vector<PlyPoint> original_points;

  PlyPoint p;
  p.x = 10.5f;
  p.y = 20.5f;
  p.z = 30.5f;
  p.nx = 0.15f;
  p.ny = 0.25f;
  p.nz = 0.35f;
  p.r = 15;
  p.g = 25;
  p.b = 35;
  original_points.push_back(p);

  WriteBinaryPlyPoints(test_file, original_points, false, false);

  std::vector<PlyPoint> loaded_points = ReadPly(test_file);

  ASSERT_EQ(loaded_points.size(), 1);
  ExpectPositionEq(loaded_points[0], 10.5f, 20.5f, 30.5f);
  // Normals and colors should be default (0)
  ExpectNormalEq(loaded_points[0], 0.0f, 0.0f, 0.0f);
  ExpectColorEq(loaded_points[0], 0, 0, 0);
}

TEST(Ply, ReadMeshVerticesAsPoints) {
  const auto test_dir = CreateTestDir();
  PlyMesh mesh;
  mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  mesh.faces.emplace_back(0, 1, 2);

  for (const bool binary : {false, true}) {
    SCOPED_TRACE(binary);
    const auto test_file = test_dir / (binary ? "binary.ply" : "text.ply");
    if (binary) {
      WriteBinaryPlyMesh(test_file, PlyTexturedMesh{mesh});
    } else {
      WriteTextPlyMesh(test_file, PlyTexturedMesh{mesh});
    }
    const auto points = ReadPly(test_file);
    ASSERT_EQ(points.size(), mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
      ExpectPositionEq(points[i],
                       mesh.vertices[i].x,
                       mesh.vertices[i].y,
                       mesh.vertices[i].z);
    }
  }
}

TEST(Ply, ReadRejectsNonemptyElementsBeforeVertices) {
  for (const std::string format :
       {"ascii", "binary_little_endian", "binary_big_endian"}) {
    SCOPED_TRACE(format);
    const auto path = CreateTestDir() / "preceding.ply";
    std::ofstream file(path, std::ios::binary);
    file << "ply\nformat " << format
         << " 1.0\n"
            "element junk 1\nproperty float a\nproperty float b\nproperty "
            "float c\n"
            "element vertex 1\nproperty float x\nproperty float y\nproperty "
            "float z\nend_header\n";
    if (format == "ascii") {
      file << "99 98 97\n1 2 3\n";
    } else {
      for (const float value : {99.0f, 98.0f, 97.0f, 1.0f, 2.0f, 3.0f}) {
        if (format == "binary_little_endian")
          WriteBinaryLittleEndian<float>(&file, value);
        else {
          const float encoded = NativeToBigEndian(value);
          file.write(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
        }
      }
    }
    file.close();
    EXPECT_THAT([&] { ReadPly(path); },
                testing::ThrowsMessage<std::invalid_argument>(
                    testing::HasSubstr("nonempty elements before vertices")));
  }
}

TEST(Ply, ReadAllowsUnconsumedElementsBeforeVertices) {
  for (const std::string format :
       {"ascii", "binary_little_endian", "binary_big_endian"}) {
    for (const bool empty_vertices : {false, true}) {
      SCOPED_TRACE(format + " " + std::to_string(empty_vertices));
      const auto path = CreateTestDir() / "preceding-unconsumed.ply";
      std::ofstream file(path, std::ios::binary);
      file << "ply\nformat " << format << " 1.0\nelement junk "
           << (empty_vertices ? 1 : 0)
           << "\nproperty float a\nproperty float b\nproperty float c\n"
              "element vertex "
           << (empty_vertices ? 0 : 1)
           << "\nproperty float x\nproperty float y\nproperty float z\n"
              "end_header\n";
      for (const float value : {1.0f, 2.0f, 3.0f}) {
        if (format == "ascii") {
          file << value << ' ';
        } else {
          const float encoded = format == "binary_little_endian"
                                    ? NativeToLittleEndian(value)
                                    : NativeToBigEndian(value);
          file.write(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
        }
      }
      if (format == "ascii") file << '\n';
      file.close();
      const auto points = ReadPly(path);
      ASSERT_EQ(points.size(), empty_vertices ? 0 : 1);
      if (!empty_vertices) {
        ExpectPositionEq(points.front(), 1.0f, 2.0f, 3.0f);
      }
    }
  }
}

TEST(Ply, ReadRejectsRepeatedVertices) {
  for (const std::string format :
       {"ascii", "binary_little_endian", "binary_big_endian"}) {
    SCOPED_TRACE(format);
    const auto path = CreateTestDir() / "repeated-vertices.ply";
    std::ofstream file(path, std::ios::binary);
    file << "ply\nformat " << format
         << " 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
            "property float z\nelement vertex 0\nend_header\n";
    if (format == "ascii") {
      file << "1 2 3\n";
    } else {
      for (const float value : {1.0f, 2.0f, 3.0f}) {
        const float encoded = format == "binary_little_endian"
                                  ? NativeToLittleEndian(value)
                                  : NativeToBigEndian(value);
        file.write(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
      }
    }
    file.close();
    EXPECT_THAT([&] { ReadPly(path); },
                testing::ThrowsMessage<std::invalid_argument>(
                    testing::HasSubstr("Duplicate PLY vertex element")));
  }
}

TEST(Ply, ReadTextZeroVerticesIgnoresFollowingElements) {
  const auto test_file = CreateTestDir() / "empty.ply";
  std::ofstream file(test_file);
  file << "ply\nformat ascii 1.0\nelement vertex 0\n"
          "property float x\nproperty float y\nproperty float z\n"
          "element color 1\nproperty uchar red\nproperty uchar green\n"
          "property uchar blue\nend_header\n1 2 3\n";
  file.close();
  EXPECT_TRUE(ReadPly(test_file).empty());
}

TEST(Ply, ReadTextTruncatedVertices) {
  const auto test_file = CreateTestDir() / "truncated.ply";
  std::ofstream file(test_file);
  file << "ply\nformat ascii 1.0\nelement vertex 2\n"
          "property float x\nproperty float y\nproperty float z\n"
          "end_header\n1 2 3\n";
  file.close();
  EXPECT_THROW(ReadPly(test_file), std::invalid_argument);
}

TEST(Ply, RoundTripTextPlyMesh) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh.ply";

  PlyMesh original_mesh;
  original_mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  original_mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  original_mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  original_mesh.vertices.emplace_back(1.0f, 1.0f, 1.0f);

  original_mesh.faces.emplace_back(0, 1, 2);
  original_mesh.faces.emplace_back(1, 3, 2);

  WriteTextPlyMesh(test_file, PlyTexturedMesh{original_mesh});

  PlyMesh loaded_mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(loaded_mesh.vertices.size(), original_mesh.vertices.size());
  ASSERT_EQ(loaded_mesh.faces.size(), original_mesh.faces.size());

  for (size_t i = 0; i < original_mesh.vertices.size(); ++i) {
    ExpectPositionEq(loaded_mesh.vertices[i],
                     original_mesh.vertices[i].x,
                     original_mesh.vertices[i].y,
                     original_mesh.vertices[i].z);
    // No colors in file, should get defaults.
    ExpectColorEq(loaded_mesh.vertices[i], 200, 200, 200);
  }

  for (size_t i = 0; i < original_mesh.faces.size(); ++i) {
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx1,
              original_mesh.faces[i].vertex_idx1);
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx2,
              original_mesh.faces[i].vertex_idx2);
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx3,
              original_mesh.faces[i].vertex_idx3);
  }
}

TEST(Ply, RoundTripBinaryPlyMesh) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh.ply";

  PlyMesh original_mesh;
  original_mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  original_mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  original_mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  original_mesh.vertices.emplace_back(1.0f, 1.0f, 1.0f);

  original_mesh.faces.emplace_back(0, 1, 2);
  original_mesh.faces.emplace_back(1, 3, 2);

  WriteBinaryPlyMesh(test_file, PlyTexturedMesh{original_mesh});

  PlyMesh loaded_mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(loaded_mesh.vertices.size(), original_mesh.vertices.size());
  ASSERT_EQ(loaded_mesh.faces.size(), original_mesh.faces.size());

  for (size_t i = 0; i < original_mesh.vertices.size(); ++i) {
    ExpectPositionEq(loaded_mesh.vertices[i],
                     original_mesh.vertices[i].x,
                     original_mesh.vertices[i].y,
                     original_mesh.vertices[i].z);
    // No colors in file, should get defaults.
    ExpectColorEq(loaded_mesh.vertices[i], 200, 200, 200);
  }

  for (size_t i = 0; i < original_mesh.faces.size(); ++i) {
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx1,
              original_mesh.faces[i].vertex_idx1);
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx2,
              original_mesh.faces[i].vertex_idx2);
    EXPECT_EQ(loaded_mesh.faces[i].vertex_idx3,
              original_mesh.faces[i].vertex_idx3);
  }
}

PlyTexturedMesh CreateTestTexturedMesh() {
  PlyTexturedMesh textured_mesh;
  textured_mesh.texture_file = "texture.png";

  textured_mesh.mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  textured_mesh.mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  textured_mesh.mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  textured_mesh.mesh.vertices.emplace_back(1.0f, 1.0f, 1.0f);

  textured_mesh.mesh.faces.emplace_back(0, 1, 2);
  textured_mesh.mesh.faces.emplace_back(1, 3, 2);

  // 6 UV floats per face (u1,v1, u2,v2, u3,v3)
  // Face 0
  textured_mesh.face_uvs.push_back(0.0f);
  textured_mesh.face_uvs.push_back(0.0f);
  textured_mesh.face_uvs.push_back(1.0f);
  textured_mesh.face_uvs.push_back(0.0f);
  textured_mesh.face_uvs.push_back(0.0f);
  textured_mesh.face_uvs.push_back(1.0f);
  // Face 1
  textured_mesh.face_uvs.push_back(0.5f);
  textured_mesh.face_uvs.push_back(0.5f);
  textured_mesh.face_uvs.push_back(1.0f);
  textured_mesh.face_uvs.push_back(1.0f);
  textured_mesh.face_uvs.push_back(0.25f);
  textured_mesh.face_uvs.push_back(0.75f);

  return textured_mesh;
}

void VerifyTexturedMesh(const PlyTexturedMesh& loaded,
                        const PlyTexturedMesh& original) {
  EXPECT_EQ(loaded.texture_file, original.texture_file);

  ASSERT_EQ(loaded.mesh.vertices.size(), original.mesh.vertices.size());
  ASSERT_EQ(loaded.mesh.faces.size(), original.mesh.faces.size());
  ASSERT_EQ(loaded.face_uvs.size(), original.face_uvs.size());

  for (size_t i = 0; i < original.mesh.vertices.size(); ++i) {
    ExpectPositionEq(loaded.mesh.vertices[i],
                     original.mesh.vertices[i].x,
                     original.mesh.vertices[i].y,
                     original.mesh.vertices[i].z);
  }

  for (size_t i = 0; i < original.mesh.faces.size(); ++i) {
    EXPECT_EQ(loaded.mesh.faces[i].vertex_idx1,
              original.mesh.faces[i].vertex_idx1);
    EXPECT_EQ(loaded.mesh.faces[i].vertex_idx2,
              original.mesh.faces[i].vertex_idx2);
    EXPECT_EQ(loaded.mesh.faces[i].vertex_idx3,
              original.mesh.faces[i].vertex_idx3);
  }

  for (size_t i = 0; i < original.face_uvs.size(); ++i) {
    EXPECT_EQ(loaded.face_uvs[i], original.face_uvs[i]);
  }
}

TEST(Ply, RoundTripTextTexturedPlyMesh) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "textured_mesh.ply";

  PlyTexturedMesh original = CreateTestTexturedMesh();
  WriteTextPlyMesh(test_file, original);
  PlyTexturedMesh loaded = ReadPlyMesh(test_file);
  VerifyTexturedMesh(loaded, original);
}

TEST(Ply, RoundTripBinaryTexturedPlyMesh) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "textured_mesh.ply";

  PlyTexturedMesh original = CreateTestTexturedMesh();
  WriteBinaryPlyMesh(test_file, original);
  PlyTexturedMesh loaded = ReadPlyMesh(test_file);
  VerifyTexturedMesh(loaded, original);
}

#ifdef __linux__
TEST(Ply, BinaryPayloadFailureIsReported) {
  const auto path = CreateTestDir() / "limited.ply";
  const std::vector<PlyPoint> points(64);
  auto mesh = CreateTestTexturedMesh();
  mesh.face_uvs.clear();
  mesh.mesh.faces.resize(128, mesh.mesh.faces.front());
  for (const bool write_mesh : {false, true}) {
    SCOPED_TRACE(write_mesh);
    const auto Write = [&] {
      if (write_mesh) {
        WriteBinaryPlyMesh(path, mesh);
      } else {
        WriteBinaryPlyPoints(path, points);
      }
    };
    Write();
    std::ifstream input(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(input)), {});
    input.close();
    const size_t header_end = bytes.find("end_header\n");
    ASSERT_NE(header_end, std::string::npos);
    ASSERT_LT(header_end + std::string("end_header\n").size(), 1024);
    ASSERT_GT(bytes.size(), 1024);
    ASSERT_LT(bytes.size(), 4096);
    ASSERT_TRUE(std::filesystem::remove(path));
    // Restrict only the isolated child. The header fits; the buffered binary
    // payload reaches the file-size limit and must be reported on completion.
    EXPECT_EXIT(
        {
          struct rlimit limit;
          if (getrlimit(RLIMIT_FSIZE, &limit) != 0 ||
              std::signal(SIGXFSZ, SIG_IGN) == SIG_ERR) {
            std::_Exit(2);
          }
          limit.rlim_cur = 1024;
          if (setrlimit(RLIMIT_FSIZE, &limit) != 0) std::_Exit(2);
          try {
            Write();
          } catch (const std::invalid_argument& error) {
            std::_Exit(
                std::string(error.what()).find("Failed to write PLY file") !=
                        std::string::npos
                    ? 0
                    : 1);
          }
          std::_Exit(1);
        },
        testing::ExitedWithCode(0),
        "");
    EXPECT_EQ(std::filesystem::file_size(path), 1024);
  }
}
#endif  // __linux__

TEST(Ply, BinaryMeshRejectsInvalidInputBeforeOpeningOutput) {
  const auto path = CreateTestDir() / "existing_mesh.ply";
  const std::string previous = "Retain existing output on invalid mesh";
  const auto ExpectRefusal = [&](const PlyTexturedMesh& mesh,
                                 const std::string& message) {
    {
      std::ofstream output(path, std::ios::binary);
      output << previous;
    }
    try {
      WriteBinaryPlyMesh(path, mesh);
      ADD_FAILURE() << "Expected invalid binary PLY mesh refusal";
    } catch (const std::invalid_argument& error) {
      if (!message.empty()) {
        EXPECT_NE(std::string(error.what()).find(message), std::string::npos);
      }
    }
    std::ifstream input(path, std::ios::binary);
    const std::string retained((std::istreambuf_iterator<char>(input)), {});
    EXPECT_EQ(retained, previous);
  };
  for (const bool textured : {false, true}) {
    for (const size_t index :
         {static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1,
          std::numeric_limits<size_t>::max()}) {
      for (const size_t face_index : {0, 1}) {
        for (int slot = 0; slot < 3; ++slot) {
          auto mesh = CreateTestTexturedMesh();
          if (!textured) mesh.face_uvs.clear();
          auto& face = mesh.mesh.faces[face_index];
          const std::array<size_t*, 3> indices = {
              &face.vertex_idx1, &face.vertex_idx2, &face.vertex_idx3};
          *indices[slot] = index;
          ExpectRefusal(mesh, "binary PLY int range");
        }
      }
    }
    auto mesh = CreateTestTexturedMesh();
    if (!textured) mesh.face_uvs.clear();
    mesh.mesh.faces.back().vertex_idx3 = mesh.mesh.vertices.size();
    ExpectRefusal(mesh, "");
  }
  auto mesh = CreateTestTexturedMesh();
  mesh.face_uvs.pop_back();
  ExpectRefusal(mesh, "Expected 6 UV coordinates");
}

TEST(Ply, ReadPlyMeshWithoutTexcoords) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "mesh.ply";

  // Write a plain mesh (no texcoords) and read it with ReadPlyMesh.
  PlyMesh plain_mesh;
  plain_mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  plain_mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  plain_mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  plain_mesh.faces.emplace_back(0, 1, 2);

  WriteTextPlyMesh(test_file, PlyTexturedMesh{plain_mesh});
  PlyTexturedMesh loaded = ReadPlyMesh(test_file);

  ASSERT_EQ(loaded.mesh.vertices.size(), 3);
  ASSERT_EQ(loaded.mesh.faces.size(), 1);
  EXPECT_TRUE(loaded.face_uvs.empty());
  EXPECT_TRUE(loaded.texture_file.empty());
}

TEST(Ply, ReadTextPlyMeshWithVertexColors) {
  const auto test_file =
      WriteTextPlyMesh("mesh_colors.ply",
                       {"float x",
                        "float y",
                        "float z",
                        "uchar red",
                        "uchar green",
                        "uchar blue"},
                       {"0 0 0 255 0 0", "1 0 0 0 255 0", "0 1 0 0 0 255"},
                       {"3 0 1 2"});

  PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 3);
  ASSERT_EQ(mesh.faces.size(), 1);

  ExpectColorEq(mesh.vertices[0], 255, 0, 0);
  ExpectColorEq(mesh.vertices[1], 0, 255, 0);
  ExpectColorEq(mesh.vertices[2], 0, 0, 255);

  EXPECT_EQ(mesh.faces[0].vertex_idx1, 0);
  EXPECT_EQ(mesh.faces[0].vertex_idx2, 1);
  EXPECT_EQ(mesh.faces[0].vertex_idx3, 2);
}

TEST(Ply, ReadBinaryPlyMeshWithVertexColors) {
  const auto test_file = WriteBinaryPlyMesh("mesh_colors_bin.ply",
                                            "binary_little_endian",
                                            3,
                                            {"float x",
                                             "float y",
                                             "float z",
                                             "uchar red",
                                             "uchar green",
                                             "uchar blue"},
                                            [](const auto& Write) {
                                              Write(1.0f);
                                              Write(2.0f);
                                              Write(3.0f);
                                              Write(uint8_t(100));
                                              Write(uint8_t(150));
                                              Write(uint8_t(200));
                                              Write(4.0f);
                                              Write(5.0f);
                                              Write(6.0f);
                                              Write(uint8_t(10));
                                              Write(uint8_t(20));
                                              Write(uint8_t(30));
                                              Write(7.0f);
                                              Write(8.0f);
                                              Write(9.0f);
                                              Write(uint8_t(50));
                                              Write(uint8_t(60));
                                              Write(uint8_t(70));
                                            },
                                            {{0, 1, 2}});

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 3);
  ASSERT_EQ(mesh.faces.size(), 1);

  ExpectPositionEq(mesh.vertices[0], 1.0f, 2.0f, 3.0f);
  ExpectColorEq(mesh.vertices[0], 100, 150, 200);

  ExpectPositionEq(mesh.vertices[1], 4.0f, 5.0f, 6.0f);
  ExpectColorEq(mesh.vertices[1], 10, 20, 30);

  ExpectPositionEq(mesh.vertices[2], 7.0f, 8.0f, 9.0f);
  ExpectColorEq(mesh.vertices[2], 50, 60, 70);
}

TEST(Ply, ReadTextPlyMeshWithExtraProperties) {
  // Normals should be skipped.
  const auto test_file = WriteTextPlyMesh("mesh_extra.ply",
                                          {"float x",
                                           "float y",
                                           "float z",
                                           "float nx",
                                           "float ny",
                                           "float nz",
                                           "uchar red",
                                           "uchar green",
                                           "uchar blue"},
                                          {"1 2 3 0.1 0.2 0.3 255 128 64",
                                           "4 5 6 0.4 0.5 0.6 32 64 96",
                                           "7 8 9 0.7 0.8 0.9 10 20 30"},
                                          {"3 0 1 2"});

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 3);

  // Positions should be read correctly.
  ExpectPositionEq(mesh.vertices[0], 1.0f, 2.0f, 3.0f);
  ExpectPositionEq(mesh.vertices[1], 4.0f, 5.0f, 6.0f);
  ExpectPositionEq(mesh.vertices[2], 7.0f, 8.0f, 9.0f);

  // Colors should be read correctly despite extra normal properties.
  ExpectColorEq(mesh.vertices[0], 255, 128, 64);
  ExpectColorEq(mesh.vertices[1], 32, 64, 96);
  ExpectColorEq(mesh.vertices[2], 10, 20, 30);
}

TEST(Ply, ReadBinaryPlyMeshWithExtraProperties) {
  const auto test_file = WriteBinaryPlyMesh("mesh_extra_bin.ply",
                                            "binary_little_endian",
                                            3,
                                            {"float x",
                                             "float y",
                                             "float z",
                                             "float nx",
                                             "float ny",
                                             "float nz",
                                             "uchar red",
                                             "uchar green",
                                             "uchar blue"},
                                            [](const auto& Write) {
                                              Write(1.0f);
                                              Write(2.0f);
                                              Write(3.0f);
                                              Write(0.1f);
                                              Write(0.2f);
                                              Write(0.3f);
                                              Write(uint8_t(255));
                                              Write(uint8_t(128));
                                              Write(uint8_t(64));
                                              Write(4.0f);
                                              Write(5.0f);
                                              Write(6.0f);
                                              Write(0.4f);
                                              Write(0.5f);
                                              Write(0.6f);
                                              Write(uint8_t(32));
                                              Write(uint8_t(64));
                                              Write(uint8_t(96));
                                              Write(7.0f);
                                              Write(8.0f);
                                              Write(9.0f);
                                              Write(0.7f);
                                              Write(0.8f);
                                              Write(0.9f);
                                              Write(uint8_t(10));
                                              Write(uint8_t(20));
                                              Write(uint8_t(30));
                                            },
                                            {{0, 1, 2}});

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 3);

  ExpectPositionEq(mesh.vertices[0], 1.0f, 2.0f, 3.0f);
  ExpectColorEq(mesh.vertices[0], 255, 128, 64);

  ExpectPositionEq(mesh.vertices[1], 4.0f, 5.0f, 6.0f);
  ExpectColorEq(mesh.vertices[1], 32, 64, 96);

  ExpectPositionEq(mesh.vertices[2], 7.0f, 8.0f, 9.0f);
  ExpectColorEq(mesh.vertices[2], 10, 20, 30);
}

TEST(Ply, ReadBinaryPlyMeshWithIntegerCoordinates) {
  const auto test_file = WriteBinaryPlyMesh("mesh_integer_coords.ply",
                                            "binary_little_endian",
                                            2,
                                            {"short x", "ushort y", "int z"},
                                            [](const auto& Write) {
                                              Write(int16_t(-2));
                                              Write(uint16_t(400));
                                              Write(int32_t(100000));
                                              Write(int16_t(30000));
                                              Write(uint16_t(50000));
                                              Write(int32_t(-500000));
                                            });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  ExpectPositionEq(mesh.vertices[0], -2.0f, 400.0f, 100000.0f);
  ExpectPositionEq(mesh.vertices[1], 30000.0f, 50000.0f, -500000.0f);
}

TEST(Ply, ReadBinaryPlyMeshWithDoubleCoordinates) {
  const auto test_file =
      WriteBinaryPlyMesh("mesh_double_coords.ply",
                         "binary_little_endian",
                         2,
                         {"double x", "double y", "double z"},
                         [](const auto& Write) {
                           Write(1.5);
                           Write(-2.25);
                           Write(100000.0);
                           Write(-0.5);
                           Write(0.0);
                           Write(-30000.0);
                         });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  ExpectPositionEq(mesh.vertices[0], 1.5f, -2.25f, 100000.0f);
  ExpectPositionEq(mesh.vertices[1], -0.5f, 0.0f, -30000.0f);
}

TEST(Ply, ReadBinaryPlyMeshWith8And32BitCoordinates) {
  const auto test_file = WriteBinaryPlyMesh("mesh_small_coords.ply",
                                            "binary_little_endian",
                                            2,
                                            {"char x", "uchar y", "uint z"},
                                            [](const auto& Write) {
                                              Write(int8_t(-100));
                                              Write(uint8_t(0));
                                              Write(uint32_t(12345678));
                                              Write(int8_t(100));
                                              Write(uint8_t(255));
                                              Write(uint32_t(100));
                                            });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  ExpectPositionEq(mesh.vertices[0], -100.0f, 0.0f, 12345678.0f);
  ExpectPositionEq(mesh.vertices[1], 100.0f, 255.0f, 100.0f);
}

TEST(Ply, ReadBinaryPlyMeshWithBigEndianIntegerCoordinates) {
  const auto test_file = WriteBinaryPlyMesh("mesh_big_endian_coords.ply",
                                            "binary_big_endian",
                                            2,
                                            {"short x", "int y", "float z"},
                                            [](const auto& Write) {
                                              Write(int16_t(-300));
                                              Write(int32_t(-500000));
                                              Write(1.5f);
                                              Write(int16_t(30000));
                                              Write(int32_t(500000));
                                              Write(-2.5f);
                                            });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  ExpectPositionEq(mesh.vertices[0], -300.0f, -500000.0f, 1.5f);
  ExpectPositionEq(mesh.vertices[1], 30000.0f, 500000.0f, -2.5f);
}

TEST(Ply, ReadBinaryPlyMeshWithMixedColorTypes) {
  const auto test_file = WriteBinaryPlyMesh("mesh_mixed_colors.ply",
                                            "binary_little_endian",
                                            2,
                                            {"float x",
                                             "float y",
                                             "float z",
                                             "uchar red",
                                             "ushort green",
                                             "float blue"},
                                            [](const auto& Write) {
                                              Write(1.0f);
                                              Write(2.0f);
                                              Write(3.0f);
                                              Write(uint8_t(200));
                                              Write(uint16_t(100));
                                              Write(1.0f);
                                              Write(4.0f);
                                              Write(5.0f);
                                              Write(6.0f);
                                              Write(uint8_t(10));
                                              Write(uint16_t(300));
                                              Write(0.5f);
                                            });

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  // Out-of-range integer channels clamp to 255; 0-1 float channels scale.
  ExpectColorEq(mesh.vertices[0], 200, 100, 255);
  ExpectColorEq(mesh.vertices[1], 10, 255, 127);
}

TEST(Ply, ReadMeshElementOrder) {
  const std::array<std::array<float, 3>, 3> vertices = {
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
  for (const std::string format :
       {"ascii", "binary_little_endian", "binary_big_endian"}) {
    for (const int position : {0, 1, 2}) {
      for (const int metadata_count : {0, 1}) {
        SCOPED_TRACE(format + " " + std::to_string(position) + " " +
                     std::to_string(metadata_count));
        const auto path = CreateTestDir() / "element_order.ply";
        std::ofstream file(path, std::ios::binary);
        file << "ply\nformat " << format << " 1.0\n";
        std::vector<std::string> elements = {"vertex", "face"};
        elements.insert(elements.begin() + position, "metadata");
        for (const auto& element : elements) {
          if (element == "vertex") {
            file << "element vertex 3\nproperty float x\nproperty float y\n"
                    "property float z\n";
          } else {
            file << "element " << element << " "
                 << (element == "face" ? 1 : metadata_count)
                 << "\nproperty list uchar int "
                 << (element == "face" ? "vertex_index" : "indices") << "\n";
          }
        }
        file << "end_header\n";
        const auto Write = [&](auto value) {
          if (format == "ascii") {
            file << +value << ' ';
          } else {
            const auto converted = format == "binary_little_endian"
                                       ? NativeToLittleEndian(value)
                                       : NativeToBigEndian(value);
            file.write(reinterpret_cast<const char*>(&converted),
                       sizeof(converted));
          }
        };
        for (const auto& element : elements) {
          if (element == "vertex") {
            for (const auto& vertex : vertices) {
              for (const float value : vertex) Write(value);
              if (format == "ascii") file << '\n';
            }
          } else if (element == "face" || metadata_count > 0) {
            Write(uint8_t(3));
            for (int32_t i = 0; i < 3; ++i) {
              Write(element == "face" ? i : 2 - i);
            }
            if (format == "ascii") file << '\n';
          }
        }
        file.close();
        if (metadata_count > 0 && position < 2) {
          ExpectPlyMeshError(
              path,
              position == 0 ? "Unsupported nonempty PLY element before vertices"
                            : "Unsupported nonempty PLY element before faces");
        } else {
          const auto mesh = ReadPlyMesh(path).mesh;
          ASSERT_EQ(mesh.vertices.size(), 3);
          for (size_t i = 0; i < vertices.size(); ++i) {
            ExpectPositionEq(mesh.vertices[i],
                             vertices[i][0],
                             vertices[i][1],
                             vertices[i][2]);
          }
          ASSERT_EQ(mesh.faces.size(), 1);
          EXPECT_EQ(mesh.faces[0].vertex_idx1, 0);
          EXPECT_EQ(mesh.faces[0].vertex_idx2, 1);
          EXPECT_EQ(mesh.faces[0].vertex_idx3, 2);
        }
      }
    }
  }
}

TEST(Ply, ReadMeshRejectsRepeatedAndReorderedElements) {
  const std::string vertices =
      "element vertex 3\nproperty float x\nproperty float y\nproperty float "
      "z\n";
  const std::string faces =
      "element face 1\nproperty list uchar int vertex_index\n";
  const std::array<std::array<std::string, 3>, 3> cases = {{
      {faces + vertices,
       "3 0 1 2\n0 0 0\n1 0 0\n0 1 0\n",
       "Unsupported nonempty PLY element before vertices"},
      {vertices + "element vertex 0\n" + faces,
       "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
       "Duplicate PLY vertex element"},
      {vertices + faces + "element face 0\n",
       "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
       "Duplicate PLY face element"},
  }};
  for (const auto& values : cases) {
    const auto path = CreateTestDir() / "invalid_elements.ply";
    std::ofstream file(path);
    file << "ply\nformat ascii 1.0\n"
         << values[0] << "end_header\n"
         << values[1];
    file.close();
    ExpectPlyMeshError(path, values[2]);
  }
}

TEST(Ply, ReadMeshFacePropertyOrder) {
  struct Case {
    std::vector<std::string> properties;
    std::string ascii_error;
    std::string binary_error;
  };
  const std::string indices_error = "Vertex indices must be the first PLY face";
  const std::string uv_error = "texture coordinates must immediately follow";
  const std::string binary_error = "Unsupported binary PLY face property";
  const std::vector<Case> cases = {
      {{"indices"}, "", ""},
      {{"indices", "uvs"}, "", ""},
      {{"metadata-list", "indices"}, indices_error, binary_error},
      {{"metadata-scalar", "indices"}, indices_error, binary_error},
      {{"indices", "metadata-list"}, "", binary_error},
      {{"indices", "metadata-scalar"}, "", binary_error},
      {{"indices", "uvs", "metadata-list"}, "", binary_error},
      {{"indices", "uvs", "metadata-scalar"}, "", binary_error},
      {{"indices", "indices"}, indices_error, indices_error},
      {{"uvs", "indices"}, uv_error, uv_error},
      {{"indices", "uvs", "uvs"}, uv_error, uv_error},
      {{"indices", "metadata-list", "uvs"}, uv_error, binary_error},
      {{"metadata-list"}, "face vertex indices missing", binary_error}};
  const std::array<std::array<uint8_t, 3>, 4> vertices = {
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}};
  const std::vector<float> uvs = {0, 0, 1, 0, 0, 1};
  for (const std::string format :
       {"ascii", "binary_little_endian", "binary_big_endian"}) {
    for (size_t case_idx = 0; case_idx < cases.size(); ++case_idx) {
      SCOPED_TRACE(format + " " + std::to_string(case_idx));
      const auto& test_case = cases[case_idx];
      const auto path = CreateTestDir() / "face-property-order.ply";
      std::ofstream file(path, std::ios::binary);
      file << "ply\nformat " << format
           << " 1.0\nelement vertex 4\nproperty uchar x\nproperty uchar y\n"
              "property uchar z\nelement face 2\n";
      for (const auto& property : test_case.properties) {
        if (property == "indices")
          file << "property list uchar uchar vertex_indices\n";
        else if (property == "uvs")
          file << "property list uchar float texcoord\n";
        else if (property == "metadata-list")
          file << "property list uchar uchar metadata\n";
        else
          file << "property uchar metadata\n";
      }
      file << "end_header\n";
      const auto Write = [&](auto value) {
        if (format == "ascii") {
          file << +value << ' ';
        } else {
          const auto encoded = format == "binary_little_endian"
                                   ? NativeToLittleEndian(value)
                                   : NativeToBigEndian(value);
          file.write(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
        }
      };
      for (const auto& vertex : vertices) {
        for (const uint8_t value : vertex) Write(value);
        if (format == "ascii") file << '\n';
      }
      for (int face = 0; face < 2; ++face) {
        for (const auto& property : test_case.properties) {
          Write(uint8_t(property == "uvs" ? 6 : 3));
          if (property == "indices" || property == "metadata-list") {
            for (uint8_t i = 0; i < 3; ++i) {
              Write(uint8_t(property == "indices" ? i : 2 - i));
            }
          } else if (property == "uvs") {
            for (const float uv : uvs) Write(uv);
          }
        }
        if (format == "ascii") file << '\n';
      }
      file.close();
      const std::string& error =
          format == "ascii" ? test_case.ascii_error : test_case.binary_error;
      if (!error.empty()) {
        ExpectPlyMeshError(path, error);
      } else {
        const auto mesh = ReadPlyMesh(path);
        ASSERT_EQ(mesh.mesh.vertices.size(), vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
          ExpectPositionEq(mesh.mesh.vertices[i],
                           vertices[i][0],
                           vertices[i][1],
                           vertices[i][2]);
        }
        ASSERT_EQ(mesh.mesh.faces.size(), 2);
        for (const auto& face : mesh.mesh.faces) {
          EXPECT_EQ(face.vertex_idx1, 0);
          EXPECT_EQ(face.vertex_idx2, 1);
          EXPECT_EQ(face.vertex_idx3, 2);
        }
        std::vector<float> expected_uvs;
        if (std::find(test_case.properties.begin(),
                      test_case.properties.end(),
                      "uvs") != test_case.properties.end()) {
          expected_uvs = uvs;
          expected_uvs.insert(expected_uvs.end(), uvs.begin(), uvs.end());
        }
        EXPECT_EQ(mesh.face_uvs, expected_uvs);
      }
    }
  }
}

TEST(Ply, ReadBinaryMeshFaceIntegerTypes) {
  for (const std::string format :
       {"binary_little_endian", "binary_big_endian"}) {
    for (const std::string type :
         {"ushort", "uint16", "uchar", "uint8", "uint", "uint32"}) {
      const size_t num_vertices =
          (type == "ushort" || type == "uint16") ? 32769 : 256;
      const uint32_t index = static_cast<uint32_t>(num_vertices - 1);
      const auto path = WriteBinaryPlyMesh(
          "face_types.ply",
          format,
          num_vertices,
          {"uchar x", "uchar y", "uchar z"},
          [&](const auto& Write) {
            for (size_t i = 0; i < num_vertices * 3; ++i) Write(uint8_t(0));
            Write(uint8_t(3));
            for (int i = 0; i < 3; ++i) {
              if (type == "ushort" || type == "uint16")
                Write(uint16_t(index));
              else if (type == "uchar" || type == "uint8")
                Write(uint8_t(index));
              else
                Write(uint32_t(index));
            }
          },
          {},
          "property list uchar " + type + " vertex_index\n",
          1);
      const auto mesh = ReadPlyMesh(path).mesh;
      ASSERT_EQ(mesh.faces.size(), 1);
      EXPECT_EQ(mesh.faces[0].vertex_idx1, index) << format << " " << type;
    }
    for (const std::string type :
         {"char", "int8", "short", "int16", "int", "int32"}) {
      const auto path = WriteBinaryPlyMesh(
          "negative_index.ply",
          format,
          1,
          {"uchar x", "uchar y", "uchar z"},
          [&](const auto& Write) {
            Write(uint8_t(0));
            Write(uint8_t(0));
            Write(uint8_t(0));
            Write(uint8_t(3));
            if (type == "char" || type == "int8")
              Write(int8_t(-1));
            else if (type == "short" || type == "int16")
              Write(int16_t(-1));
            else
              Write(int32_t(-1));
          },
          {},
          "property list uchar " + type + " vertex_index\n",
          1);
      ExpectPlyMeshError(path, "Negative face vertex index");
    }
    for (const uint32_t index : {uint32_t(1),
                                 uint32_t(0x7fffffff),
                                 uint32_t(0x80000000),
                                 uint32_t(0xffffffff)}) {
      const auto path = WriteBinaryPlyMesh(
          "large_index.ply",
          format,
          1,
          {"uchar x", "uchar y", "uchar z"},
          [&](const auto& Write) {
            Write(uint8_t(0));
            Write(uint8_t(0));
            Write(uint8_t(0));
            Write(uint8_t(3));
            Write(index);
          },
          {},
          "property list uchar uint32 vertex_index\n",
          1);
      ExpectPlyMeshError(path, "Face vertex index out of bounds");
    }
    for (const std::string type : {"char",
                                   "int8",
                                   "uchar",
                                   "uint8",
                                   "short",
                                   "int16",
                                   "ushort",
                                   "uint16",
                                   "int",
                                   "int32",
                                   "uint",
                                   "uint32"}) {
      const auto path = WriteBinaryPlyMesh(
          "count_type.ply",
          format,
          1,
          {"uchar x", "uchar y", "uchar z"},
          [&](const auto& Write) {
            Write(uint8_t(0));
            Write(uint8_t(0));
            Write(uint8_t(0));
            if (type == "char" || type == "int8")
              Write(int8_t(3));
            else if (type == "uchar" || type == "uint8")
              Write(uint8_t(3));
            else if (type == "short" || type == "int16")
              Write(int16_t(3));
            else if (type == "ushort" || type == "uint16")
              Write(uint16_t(3));
            else if (type == "int" || type == "int32")
              Write(int32_t(3));
            else
              Write(uint32_t(3));
            Write(uint8_t(0));
            Write(uint8_t(0));
            Write(uint8_t(0));
          },
          {},
          "property list " + type + " uchar vertex_index\n",
          1);
      EXPECT_EQ(ReadPlyMesh(path).mesh.faces.size(), 1)
          << format << " " << type;
      if (type == "char" || type == "int8" || type == "short" ||
          type == "int16" || type == "int" || type == "int32") {
        const auto negative_path = WriteBinaryPlyMesh(
            "negative_count.ply",
            format,
            1,
            {"uchar x", "uchar y", "uchar z"},
            [&](const auto& Write) {
              for (int i = 0; i < 3; ++i) Write(uint8_t(0));
              if (type == "char" || type == "int8")
                Write(int8_t(-1));
              else if (type == "short" || type == "int16")
                Write(int16_t(-1));
              else
                Write(int32_t(-1));
            },
            {},
            "property list " + type + " uchar vertex_index\n",
            1);
        ExpectPlyMeshError(negative_path,
                           "Only triangular faces are supported");
      }
    }
    for (const std::string declaration :
         {"float uchar", "uchar float", "double uchar", "uchar float64"}) {
      const auto path = WriteBinaryPlyMesh(
          "float_face.ply",
          format,
          1,
          {"uchar x", "uchar y", "uchar z"},
          [](const auto& Write) {
            Write(uint8_t(0));
            Write(uint8_t(0));
            Write(uint8_t(0));
          },
          {},
          "property list " + declaration + " vertex_index\n",
          1);
      ExpectPlyMeshError(path, "Unsupported PLY face data type");
    }
  }
}

TEST(Ply, ReadMeshTexcoordTypes) {
  const std::array<std::array<std::string, 2>, 6> types = {
      {{"uchar", "float"},
       {"uint8", "float32"},
       {"ushort", "float"},
       {"uint16", "float32"},
       {"uchar", "double"},
       {"uint8", "float64"}}};
  const std::vector<float> uvs = {0, 0, 1, 0, 0, 1};
  for (const std::string format :
       {"binary_little_endian", "binary_big_endian"}) {
    for (const auto& type : types) {
      SCOPED_TRACE(format + " " + type[0] + " " + type[1]);
      const bool wide_count = type[0] == "ushort" || type[0] == "uint16";
      const bool double_uv = type[1] == "double" || type[1] == "float64";
      const auto path = WriteBinaryPlyMesh(
          "mesh_uv_types.ply",
          format,
          3,
          {"uchar x", "uchar y", "uchar z"},
          [&](const auto& Write) {
            for (const uint8_t value : {0, 0, 0, 1, 0, 0, 0, 1, 0})
              Write(value);
            Write(uint8_t(3));
            for (int32_t i = 0; i < 3; ++i) Write(i);
            if (wide_count)
              Write(uint16_t(6));
            else
              Write(uint8_t(6));
            for (const float uv : uvs) {
              if (double_uv)
                Write(double{uv});
              else
                Write(uv);
            }
          },
          {},
          "property list uchar int vertex_index\nproperty list " + type[0] +
              " " + type[1] + " texcoord\n",
          1);
      if (wide_count || double_uv) {
        ExpectPlyMeshError(path, "Unsupported binary PLY texcoord data types");
      } else {
        EXPECT_EQ(ReadPlyMesh(path).face_uvs, uvs);
      }
    }
    const auto empty_path = WriteBinaryPlyMesh(
        "empty_uv_types.ply",
        format,
        3,
        {"uchar x", "uchar y", "uchar z"},
        [](const auto& Write) {
          for (const uint8_t value : {0, 0, 0, 1, 0, 0, 0, 1, 0}) Write(value);
        },
        {},
        "property list uchar int vertex_index\n"
        "property list ushort double texcoord\n",
        0);
    const auto empty = ReadPlyMesh(empty_path);
    EXPECT_EQ(empty.mesh.vertices.size(), 3);
    EXPECT_TRUE(empty.mesh.faces.empty());
    EXPECT_TRUE(empty.face_uvs.empty());
  }
  // ASCII conversion is independent of the binary list representation.
  const auto path = CreateTestDir() / "text_uv_types.ply";
  std::ofstream file(path);
  file << "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n"
          "property float y\nproperty float z\nelement face 1\n"
          "property list uchar int vertex_index\n"
          "property list ushort double texcoord\nend_header\n"
          "0 0 0\n1 0 0\n0 1 0\n3 0 1 2 6 0 0 1 0 0 1\n";
  file.close();
  EXPECT_EQ(ReadPlyMesh(path).face_uvs, uvs);
  file.open(path, std::ios::trunc);
  file << "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n"
          "property float y\nproperty float z\nelement face 0\n"
          "property list uchar int vertex_index\n"
          "property list ushort double texcoord\nend_header\n"
          "0 0 0\n1 0 0\n0 1 0\n";
  file.close();
  const auto empty = ReadPlyMesh(path);
  EXPECT_EQ(empty.mesh.vertices.size(), 3);
  EXPECT_TRUE(empty.mesh.faces.empty());
  EXPECT_TRUE(empty.face_uvs.empty());
}

TEST(Ply, ReadBinaryMeshTexcoordTruncation) {
  const std::array<uint32_t, 12> uv_bits = {0x00000000,
                                            0x80000000,
                                            0x00000001,
                                            0x3e800000,
                                            0x3f800000,
                                            0x7f7fffff,
                                            0x00800000,
                                            0xbf800000,
                                            0x7f800000,
                                            0xff800000,
                                            0x7fc12345,
                                            0x3f012345};
  for (const std::string format :
       {"binary_little_endian", "binary_big_endian"}) {
    const auto path = WriteBinaryPlyMesh(
        "mesh_uv.ply",
        format,
        3,
        {"uchar x", "uchar y", "uchar z"},
        [&](const auto& Write) {
          for (int i = 0; i < 9; ++i) Write(uint8_t(0));
          for (int face = 0; face < 2; ++face) {
            Write(uint8_t(3));
            for (int i = 0; i < 3; ++i) Write(int32_t(i));
            Write(uint8_t(6));
            for (int i = 0; i < 6; ++i) Write(uv_bits[face * 6 + i]);
          }
        },
        {},
        "property list uchar int vertex_index\nproperty list uchar float "
        "texcoord\n",
        2);
    const auto mesh = ReadPlyMesh(path);
    ASSERT_EQ(mesh.face_uvs.size(), 12);
    for (size_t i = 0; i < uv_bits.size(); ++i) {
      uint32_t actual;
      std::memcpy(&actual, &mesh.face_uvs[i], sizeof(actual));
      EXPECT_EQ(actual, uv_bits[i]) << format << " " << i;
    }
    std::ifstream input(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    input.close();
    // Cut at every byte of each face's UV count and payload, including the
    // first face (which leaves the second face unread).
    for (int face = 0; face < 2; ++face) {
      const size_t uv_offset = bytes.size() - (2 - face) * 38 + 13;
      for (size_t length = 0; length < 25; ++length) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(bytes.data(), uv_offset + length);
        output.close();
        const std::string suffix =
            length == 0
                ? "texcoord count"
                : "texcoord " + std::to_string((length - 1) / sizeof(float));
        ExpectPlyMeshError(path, "face " + std::to_string(face) + " " + suffix);
      }
      std::vector<char> wrong_count = bytes;
      wrong_count[uv_offset] = 5;
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      output.write(wrong_count.data(), wrong_count.size());
      output.close();
      ExpectPlyMeshError(path, "Expected 6 texture coordinates");
    }
  }
}

TEST(Ply, ReadTextPlyMeshWithFloatColors) {
  const auto test_file =
      WriteTextPlyMesh("mesh_float_colors.ply",
                       {"float x",
                        "float y",
                        "float z",
                        "float red",
                        "float green",
                        "float blue"},
                       {"0 0 0 1.0 0.5 0.0", "1 0 0 0.0 1.0 0.25"});

  const PlyMesh mesh = ReadPlyMesh(test_file).mesh;

  ASSERT_EQ(mesh.vertices.size(), 2);
  ExpectColorEq(mesh.vertices[0], 255, 127, 0);
  ExpectColorEq(mesh.vertices[1], 0, 255, 63);
}

TEST(Ply, HasPlyMeshFacesWithMesh) {
  const auto test_file = CreateTestDir() / "mesh.ply";

  PlyMesh mesh;
  mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f);
  mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f);
  mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f);
  mesh.faces.emplace_back(0, 1, 2);

  WriteTextPlyMesh(test_file, PlyTexturedMesh{mesh});
  EXPECT_TRUE(HasPlyMeshFaces(test_file));
}

TEST(Ply, HasPlyMeshFacesWithPointCloud) {
  const auto test_file = CreateTestDir() / "points.ply";

  std::vector<PlyPoint> points;
  points.emplace_back();

  WriteTextPlyPoints(test_file, points, false, false);
  EXPECT_FALSE(HasPlyMeshFaces(test_file));
}

TEST(Ply, HasPlyMeshFacesWithZeroFaces) {
  const auto test_file = CreateTestDir() / "mesh_no_faces.ply";

  PlyMesh mesh;
  WriteTextPlyMesh(test_file, PlyTexturedMesh{mesh});

  EXPECT_FALSE(HasPlyMeshFaces(test_file));
}

}  // namespace
}  // namespace colmap
