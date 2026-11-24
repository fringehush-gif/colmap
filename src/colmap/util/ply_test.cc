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
#include <sstream>
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
// Otherwise, faces are appended using the default uchar/int layout,
// with optional face_uvs containing six coordinates per face.
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
    const std::vector<float>& face_uvs = {},
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
    if (!face_uvs.empty()) {
      text_file << "property list uchar float texcoord\n";
    }
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
    for (size_t i = 0; i < faces.size(); ++i) {
      Write(uint8_t(3));
      for (int idx : faces[i]) {
        Write(int32_t(idx));
      }
      if (!face_uvs.empty()) {
        Write(uint8_t(6));
        for (size_t j = 0; j < 6; ++j) {
          Write(face_uvs.at(6 * i + j));
        }
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
  original_mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f, 0, 17, 255);
  original_mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f, 109, 233, 1);
  original_mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f, 255, 0, 128);
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
    ExpectColorEq(loaded_mesh.vertices[i],
                  original_mesh.vertices[i].r,
                  original_mesh.vertices[i].g,
                  original_mesh.vertices[i].b);
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
  original_mesh.vertices.emplace_back(0.0f, 0.0f, 0.0f, 0, 17, 255);
  original_mesh.vertices.emplace_back(1.0f, 0.0f, 0.0f, 109, 233, 1);
  original_mesh.vertices.emplace_back(0.0f, 1.0f, 0.0f, 255, 0, 128);
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
    ExpectColorEq(loaded_mesh.vertices[i],
                  original_mesh.vertices[i].r,
                  original_mesh.vertices[i].g,
                  original_mesh.vertices[i].b);
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
    ExpectColorEq(loaded.mesh.vertices[i],
                  original.mesh.vertices[i].r,
                  original.mesh.vertices[i].g,
                  original.mesh.vertices[i].b);
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
  original.mesh.vertices[0].r = 0;
  original.mesh.vertices[0].g = 17;
  original.mesh.vertices[0].b = 255;
  WriteTextPlyMesh(test_file, original);
  PlyTexturedMesh loaded = ReadPlyMesh(test_file);
  VerifyTexturedMesh(loaded, original);
}

TEST(Ply, RoundTripBinaryTexturedPlyMesh) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "textured_mesh.ply";

  PlyTexturedMesh original = CreateTestTexturedMesh();
  original.mesh.vertices[0].r = 0;
  original.mesh.vertices[0].g = 17;
  original.mesh.vertices[0].b = 255;
  WriteBinaryPlyMesh(test_file, original);
  PlyTexturedMesh loaded = ReadPlyMesh(test_file);
  VerifyTexturedMesh(loaded, original);
}

TEST(Ply, RejectTruncatedBinaryTextureCoordinates) {
  PlyTexturedMesh original = CreateTestTexturedMesh();
  original.texture_file.clear();
  constexpr size_t kFaceIndexBytes = sizeof(uint8_t) + 3 * sizeof(int32_t);
  constexpr size_t kFaceTexcoordBytes = sizeof(uint8_t) + 6 * sizeof(float);
  constexpr size_t kFaceBytes = kFaceIndexBytes + kFaceTexcoordBytes;

  for (const std::string format :
       {"binary_little_endian", "binary_big_endian"}) {
    SCOPED_TRACE(format);
    const auto path = WriteBinaryPlyMesh(
        "truncated.ply",
        format,
        original.mesh.vertices.size(),
        {"float x", "float y", "float z"},
        [&](const auto& Write) {
          for (const auto& vertex : original.mesh.vertices) {
            Write(vertex.x);
            Write(vertex.y);
            Write(vertex.z);
          }
        },
        {{0, 1, 2}, {1, 3, 2}},
        original.face_uvs);
    VerifyTexturedMesh(ReadPlyMesh(path), original);

    std::ifstream input(path, std::ios::binary);
    const std::vector<char> bytes(std::istreambuf_iterator<char>{input}, {});
    input.close();
    ASSERT_GE(bytes.size(), original.mesh.faces.size() * kFaceBytes);

    for (size_t face = 0; face < original.mesh.faces.size(); ++face) {
      SCOPED_TRACE(face);
      const size_t uv_offset =
          bytes.size() - (original.mesh.faces.size() - face) * kFaceBytes +
          kFaceIndexBytes;
      // Keep zero through 24 bytes: omit the count, or truncate any UV float.
      for (size_t length = 0; length < kFaceTexcoordBytes; ++length) {
        SCOPED_TRACE(length);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(bytes.data(), uv_offset + length);
        output.close();
        ASSERT_TRUE(output.good());

        const std::string expected_message =
            (length == 0 ? "Missing texture coordinate count at face "
                         : "Unexpected end of texture coordinates at face ") +
            std::to_string(face);
        try {
          ReadPlyMesh(path);
          FAIL() << "Expected rejection of truncated texture coordinates";
        } catch (const std::invalid_argument& error) {
          EXPECT_NE(std::string(error.what()).find(expected_message),
                    std::string::npos);
        }
      }
    }
  }
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

void WriteTestFile(const std::filesystem::path& path,
                   const std::string& bytes) {
  std::ofstream file(path, std::ios::binary);
  file << bytes;
  file.close();
  ASSERT_TRUE(file.good());
}

std::string ReadTestFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
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
#endif

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
#endif

TEST(Ply, MeshRejectsInvalidInputBeforeOpeningOutput) {
  const auto path = CreateTestDir() / "existing_mesh.ply";
  const std::string previous = "Retain existing output on invalid mesh";
  const auto ExpectRefusal = [&](const PlyTexturedMesh& mesh,
                                 const std::string& message) {
    for (const bool binary : {false, true}) {
      SCOPED_TRACE(binary);
      const auto Write = [&] {
        binary ? WriteBinaryPlyMesh(path, mesh) : WriteTextPlyMesh(path, mesh);
      };
      const auto expected_error = testing::ThrowsMessage<std::invalid_argument>(
          testing::HasSubstr(message));
      WriteTestFile(path, previous);
      EXPECT_THAT(Write, expected_error);
      EXPECT_EQ(ReadTestFile(path), previous);
      ASSERT_TRUE(std::filesystem::remove(path));
      EXPECT_THAT(Write, expected_error);
      EXPECT_FALSE(std::filesystem::exists(path));
    }
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
          ExpectRefusal(mesh, "PLY int range");
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

TEST(Ply, MeshWriterPayloadAndColorRetention) {
  const auto p = CreateTestDir() / "mesh.ply";
  PlyTexturedMesh mesh;
  mesh.mesh.vertices = {
      {1, 2, 3, 17, 109, 233}, {-4, 5, 6, 0, 128, 255}, {7, -8, 9, 255, 1, 2}};
  mesh.mesh.faces = {{2, 0, 1}, {1, 0, 2}};
  mesh.face_uvs = {0.125f, 0.25f, 0.5f, 0.75f, 1, 0, 1, 1, 0, 1, 0.5f, 0.25f};
  mesh.texture_file = "texture.png";
  for (const bool binary : {false, true}) {
    binary ? WriteBinaryPlyMesh(p, mesh) : WriteTextPlyMesh(p, mesh);
    const auto loaded = ReadPlyMesh(p);
    ASSERT_EQ(loaded.mesh.vertices.size(), mesh.mesh.vertices.size());
    ASSERT_EQ(loaded.mesh.faces.size(), mesh.mesh.faces.size());
    EXPECT_EQ(loaded.face_uvs, mesh.face_uvs);
    EXPECT_EQ(loaded.texture_file, mesh.texture_file);
    for (size_t i = 0; i < mesh.mesh.vertices.size(); ++i) {
      EXPECT_EQ(loaded.mesh.vertices[i].x, mesh.mesh.vertices[i].x);
      EXPECT_EQ(loaded.mesh.vertices[i].y, mesh.mesh.vertices[i].y);
      EXPECT_EQ(loaded.mesh.vertices[i].z, mesh.mesh.vertices[i].z);
      ExpectColorEq(loaded.mesh.vertices[i],
                    mesh.mesh.vertices[i].r,
                    mesh.mesh.vertices[i].g,
                    mesh.mesh.vertices[i].b);
    }
    for (size_t i = 0; i < mesh.mesh.faces.size(); ++i) {
      EXPECT_EQ(loaded.mesh.faces[i].vertex_idx1,
                mesh.mesh.faces[i].vertex_idx1);
      EXPECT_EQ(loaded.mesh.faces[i].vertex_idx2,
                mesh.mesh.faces[i].vertex_idx2);
      EXPECT_EQ(loaded.mesh.faces[i].vertex_idx3,
                mesh.mesh.faces[i].vertex_idx3);
    }
    EXPECT_TRUE(HasPlyMeshFaces(p));
    const auto points = ReadPly(p);
    ASSERT_EQ(points.size(), mesh.mesh.vertices.size());
    for (size_t i = 0; i < points.size(); ++i) {
      EXPECT_EQ(points[i].x, mesh.mesh.vertices[i].x);
      EXPECT_EQ(points[i].y, mesh.mesh.vertices[i].y);
      EXPECT_EQ(points[i].z, mesh.mesh.vertices[i].z);
      EXPECT_EQ(points[i].r, mesh.mesh.vertices[i].r);
      EXPECT_EQ(points[i].g, mesh.mesh.vertices[i].g);
      EXPECT_EQ(points[i].b, mesh.mesh.vertices[i].b);
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

TEST(Ply, ReadBinaryMeshTexcoordBitsAndCount) {
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
    const std::string bytes = ReadTestFile(path);
    for (int face = 0; face < 2; ++face) {
      const size_t uv_offset = bytes.size() - (2 - face) * 38 + 13;
      std::string wrong_count = bytes;
      wrong_count[uv_offset] = 5;
      WriteTestFile(path, wrong_count);
      ExpectPlyMeshError(path, "Expected 6 texture coordinates");
    }
  }
}

TEST(Ply, HeaderAndCountFailures) {
  const auto p = CreateTestDir() / "input.ply";
  const std::string empty_vertex =
      "element vertex 0\nproperty float x\nproperty float y\nproperty float "
      "z\n";
  for (const std::string& header :
       {empty_vertex,
        "ply\nformat ascii 1.0\n" + empty_vertex,
        "wrong\nformat ascii 1.0\n" + empty_vertex + "end_header\n",
        "ply\n" + empty_vertex + "end_header\n",
        "ply\nformat unsupported 1.0\n" + empty_vertex + "end_header\n",
        "ply\nformat ascii 1.0\nformat binary_little_endian 1.0\n" +
            empty_vertex + "end_header\n"}) {
    SCOPED_TRACE(header);
    WriteTestFile(p, header);
    EXPECT_THROW(ReadPly(p), std::invalid_argument);
    EXPECT_THROW(ReadPlyMesh(p), std::invalid_argument);
    EXPECT_EQ(ReadTestFile(p), header);
  }
  for (const std::string count :
       {"-1", "0garbage", "+", "18446744073709551616"}) {
    SCOPED_TRACE(count);
    const std::string bytes =
        "ply\nformat ascii 1.0\nelement vertex " + count +
        "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    WriteTestFile(p, bytes);
    EXPECT_ANY_THROW(ReadPly(p));
    EXPECT_ANY_THROW(ReadPlyMesh(p));
    EXPECT_EQ(ReadTestFile(p), bytes);
  }
  for (const std::string count : {"+0", "+1"}) {
    SCOPED_TRACE(count);
    WriteTestFile(p,
                  "ply\nformat ascii 1.0\nelement vertex " + count +
                      "\nproperty float x\nproperty float y\nproperty float z\n"
                      "end_header\n1 2 3\n");
    const size_t expected = count == "+0" ? 0 : 1;
    const auto points = ReadPly(p);
    const auto mesh = ReadPlyMesh(p);
    ASSERT_EQ(points.size(), expected);
    ASSERT_EQ(mesh.mesh.vertices.size(), expected);
    if (expected != 0) {
      ExpectPositionEq(points.front(), 1, 2, 3);
      ExpectPositionEq(mesh.mesh.vertices.front(), 1, 2, 3);
    }
  }
}

TEST(Ply, MeshColorParity) {
  const auto p = CreateTestDir() / "color.ply";
  for (const std::string format :
       {"ascii", "binary_little_endian", "binary_big_endian"}) {
    SCOPED_TRACE(format);
    const bool binary = format != "ascii";
    const bool little = format == "binary_little_endian";
    for (const std::string type : {"short", "float"}) {
      std::ostringstream header;
      header << "ply\nformat " << format
             << " 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
                "property float z\n";
      for (const char* channel : {"red", "green", "blue"}) {
        header << "property " << type << ' ' << channel << '\n';
      }
      header << "end_header\n";
      std::string bytes = header.str();
      const auto append = [&](auto value) {
        value = little ? NativeToLittleEndian(value) : NativeToBigEndian(value);
        bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
      };
      if (binary) {
        append(1.0f);
        append(-2.0f);
        append(3.0f);
        if (type == "short") {
          append(int16_t(-1));
          append(int16_t(128));
          append(int16_t(256));
        } else {
          append(-0.5f);
          append(0.5f);
          append(1.5f);
        }
      } else
        bytes +=
            type == "short" ? "1 -2 3 -1 128 256\n" : "1 -2 3 -0.5 0.5 1.5\n";
      WriteTestFile(p, bytes);
      const auto mesh = ReadPlyMesh(p).mesh;
      ASSERT_EQ(mesh.vertices.size(), 1);
      const auto& v = mesh.vertices[0];
      EXPECT_EQ(v.x, 1);
      EXPECT_EQ(v.y, -2);
      EXPECT_EQ(v.z, 3);
      EXPECT_EQ(v.r, 0);
      EXPECT_EQ(v.g, type == "short" ? 128 : 127);
      EXPECT_EQ(v.b, 255);
      EXPECT_EQ(ReadTestFile(p), bytes);
    }
  }
}

TEST(Ply, ReadBinaryMeshColorsRejectNaNAndSaturateInfinity) {
  for (const std::string format :
       {"binary_little_endian", "binary_big_endian"}) {
    for (const std::string type : {"float", "double"}) {
      SCOPED_TRACE(testing::Message() << format << ' ' << type);
      const auto WriteColors = [&](const std::array<double, 3>& colors) {
        return WriteBinaryPlyMesh(
            "nonfinite-colors.ply",
            format,
            1,
            {"float x",
             "float y",
             "float z",
             type + " red",
             type + " green",
             type + " blue"},
            [&](const auto& Write) {
              Write(std::numeric_limits<float>::quiet_NaN());
              Write(std::numeric_limits<float>::infinity());
              Write(-std::numeric_limits<float>::infinity());
              for (const double value : colors) {
                if (type == "float") {
                  Write(static_cast<float>(value));
                } else {
                  Write(value);
                }
              }
            });
      };
      for (size_t channel = 0; channel < 3; ++channel) {
        SCOPED_TRACE(channel);
        std::array<double, 3> colors = {0.5, 0.5, 0.5};
        colors[channel] = std::numeric_limits<double>::quiet_NaN();
        ExpectPlyMeshError(WriteColors(colors), "NaN PLY color");
      }
      const auto mesh =
          ReadPlyMesh(WriteColors({std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity(),
                                   0.5}));
      ASSERT_EQ(mesh.mesh.vertices.size(), 1);
      const auto& vertex = mesh.mesh.vertices.front();
      ExpectColorEq(vertex, 255, 0, 127);
      EXPECT_TRUE(std::isnan(vertex.x));
      EXPECT_EQ(vertex.y, std::numeric_limits<float>::infinity());
      EXPECT_EQ(vertex.z, -std::numeric_limits<float>::infinity());
    }
  }
}

TEST(Ply, EveryConsumedBinaryByteIsRequired) {
  const auto p = CreateTestDir() / "payload.ply";
  PlyTexturedMesh mesh;
  mesh.mesh.vertices = {{1, -2, 3}, {4, 5, -6}, {-7, 8, 9}};
  mesh.mesh.faces = {{2, 1, 0}, {0, 1, 2}};
  mesh.face_uvs = {0.125f, 0.25f, 0.5f, 0.75f, 1, 0, 1, 1, 0, 1, 0.5f, 0.25f};
  WriteBinaryPlyMesh(p, mesh);
  const auto bytes = ReadTestFile(p);
  const size_t header_end = bytes.find("end_header");
  ASSERT_NE(header_end, std::string::npos);
  const size_t header_newline = bytes.find('\n', header_end);
  ASSERT_NE(header_newline, std::string::npos);
  const size_t payload = header_newline + 1;
  ASSERT_LT(payload, bytes.size());
  for (size_t end = payload; end < bytes.size(); ++end) {
    SCOPED_TRACE(end - payload);
    WriteTestFile(p, bytes.substr(0, end));
    EXPECT_ANY_THROW(ReadPlyMesh(p));
    EXPECT_EQ(ReadTestFile(p), bytes.substr(0, end));
  }
  // The point reader deliberately consumes only vertices and ignores faces.
  for (size_t end = payload; end < payload + mesh.mesh.vertices.size() * 12;
       ++end) {
    WriteTestFile(p, bytes.substr(0, end));
    EXPECT_ANY_THROW(ReadPly(p));
  }
  WriteTestFile(p, bytes);
  EXPECT_EQ(ReadPly(p).size(), mesh.mesh.vertices.size());
  EXPECT_EQ(ReadPlyMesh(p).face_uvs, mesh.face_uvs);
}

TEST(Ply, IncompleteAsciiConsumedFieldsAreRejected) {
  const auto p = CreateTestDir() / "partial.ply";
  const std::string header =
      "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty "
      "float y\nproperty float z\nproperty uchar red\nproperty uchar "
      "green\nproperty uchar blue\nend_header\n";
  for (const std::string body :
       {"", "1", "1 2", "1 2 3", "1 2 3 17", "1 2 3 17 109"}) {
    WriteTestFile(p, header + body);
    EXPECT_ANY_THROW(ReadPly(p));
    EXPECT_ANY_THROW(ReadPlyMesh(p));
  }
  WriteTestFile(p, header + "1 2 3 17 109 233");
  EXPECT_EQ(ReadPly(p)[0].b, 233);
  EXPECT_EQ(ReadPlyMesh(p).mesh.vertices[0].b, 233);
}

TEST(Ply, MissingAsciiFaceRecordsAreRejected) {
  const auto path = CreateTestDir() / "missing-faces.ply";
  const std::string vertex_header =
      "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty "
      "float y\nproperty float z\nproperty uchar red\nproperty uchar "
      "green\nproperty uchar blue\n";
  const std::vector<float> uvs = {0, 0.25f, 0.5f, 0.75f, 1, 0};
  for (const bool has_texcoord : {false, true}) {
    const std::string face =
        has_texcoord ? "3 0 0 0 6 0 0.25 0.5 0.75 1 0" : "3 0 0 0";
    for (const size_t present_faces : {0, 1}) {
      const std::string header =
          vertex_header + "element face " + std::to_string(present_faces + 1) +
          "\nproperty list uchar int vertex_indices\n" +
          (has_texcoord ? "property list uchar float texcoord\n" : "") +
          "end_header\n";
      // This legal vertex record can also be parsed as a triangle if a failed
      // getline leaves it in the buffer. A prior complete face can be reused
      // too.
      std::string payload = "3 0 0 0 0 0";
      for (size_t i = 0; i < present_faces; ++i) payload += "\n" + face;
      PlyTexturedMesh expected;
      expected.mesh.vertices.emplace_back(3, 0, 0, 0, 0, 0);
      for (size_t i = 0; i <= present_faces; ++i) {
        expected.mesh.faces.emplace_back(0, 0, 0);
        if (has_texcoord) {
          expected.face_uvs.insert(
              expected.face_uvs.end(), uvs.begin(), uvs.end());
        }
      }
      const std::string incomplete = header + payload;
      std::string complete = incomplete;
      complete += '\n';
      complete += face;
      for (const bool has_newline : {false, true}) {
        SCOPED_TRACE(testing::Message() << has_texcoord << ' ' << present_faces
                                        << ' ' << has_newline);
        const std::string ending = has_newline ? "\n" : "";
        WriteTestFile(path, incomplete + ending);
        ExpectPlyMeshError(path,
                           "Unexpected end of PLY file at face " +
                               std::to_string(present_faces));

        WriteTestFile(path, complete + ending);
        const auto loaded = ReadPlyMesh(path);
        VerifyTexturedMesh(loaded, expected);
        ExpectColorEq(loaded.mesh.vertices.at(0), 0, 0, 0);
      }
    }
  }
}

TEST(Ply, UnsupportedFaceTypePrecedesPayloadReads) {
  const auto p = CreateTestDir() / "missing-payload.ply";
  for (const std::string format : {"ascii", "binary_little_endian"}) {
    for (const std::string list : {"float int", "uchar double"}) {
      std::ostringstream header;
      header << "ply\nformat " << format
             << " 1.0\nelement vertex 3\nproperty float x\nproperty float "
                "y\nproperty float z\nelement face 1\nproperty list "
             << list << " vertex_indices\nend_header\n";
      WriteTestFile(p, header.str());
      try {
        ReadPlyMesh(p);
        FAIL() << "Expected unsupported declaration";
      } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("Unsupported PLY face data type"),
                  std::string::npos);
      }
    }
  }
}

TEST(Ply, DoubleColorRangeIsCheckedBeforeNarrowing) {
  const auto p = CreateTestDir() / "double-colors.ply";
  for (const std::string format :
       {"ascii", "binary_little_endian", "binary_big_endian"}) {
    std::string bytes =
        "ply\nformat " + format +
        " 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty "
        "float z\nproperty double red\nproperty double green\nproperty double "
        "blue\nend_header\n";
    if (format == "ascii")
      bytes += "1 2 3 -1e100 0.5019607843137254 1e100\n";
    else {
      const auto append = [&](auto value) {
        value = format == "binary_little_endian" ? NativeToLittleEndian(value)
                                                 : NativeToBigEndian(value);
        bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
      };
      append(1.0f);
      append(2.0f);
      append(3.0f);
      append(-1e100);
      // Narrowing this normalized channel to float would round it up to 128.
      append(0.5019607843137254);
      append(1e100);
    }
    WriteTestFile(p, bytes);
    const auto mesh = ReadPlyMesh(p);
    ASSERT_EQ(mesh.mesh.vertices.size(), 1);
    EXPECT_EQ(mesh.mesh.vertices[0].r, 0);
    EXPECT_EQ(mesh.mesh.vertices[0].g, 127);
    EXPECT_EQ(mesh.mesh.vertices[0].b, 255);
  }
}

}  // namespace
}  // namespace colmap
