// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/ply.h"

#include "colmap/util/endian.h"
#include "colmap/util/testing.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Write a binary PLY mesh with the given vertex properties (e.g. "short x")
// and triangular faces. The payload callback receives a Write function that
// encodes one scalar in the file's endianness; it must write exactly
// num_vertices vertices. Pass explicitly typed values (e.g. int16_t(-300))
// so that scalars get the intended width.
template <typename WritePayload>
std::filesystem::path WriteBinaryPlyMesh(
    const std::string& file_name,
    const std::string& format,
    size_t num_vertices,
    const std::vector<std::string>& properties,
    WritePayload&& write_payload,
    const std::vector<std::array<int, 3>>& faces = {}) {
  const auto test_file = CreateTestDir() / file_name;
  {
    std::ofstream text_file(test_file);
    text_file << "ply\n";
    text_file << "format " << format << " 1.0\n";
    text_file << "element vertex " << num_vertices << "\n";
    for (const auto& property : properties) {
      text_file << "property " << property << "\n";
    }
    text_file << "element face " << faces.size() << "\n";
    text_file << "property list uchar int vertex_index\n";
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

TEST(Ply, BinaryPointWireBytes) {
  const auto test_dir = CreateTestDir();
  const auto test_file = test_dir / "wire.ply";
  const std::array<float, 7> values = {0.0f,
                                       -0.0f,
                                       std::numeric_limits<float>::denorm_min(),
                                       std::numeric_limits<float>::max(),
                                       std::numeric_limits<float>::lowest(),
                                       std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::quiet_NaN()};
  for (const size_t count : {size_t{0}, size_t{1}, size_t{4097}}) {
    std::vector<PlyPoint> points(count);
    for (size_t i = 0; i < count; ++i) {
      auto& point = points[i];
      point.x = values[i % values.size()];
      point.y = values[(i + 1) % values.size()];
      point.z = values[(i + 2) % values.size()];
      point.nx = values[(i + 3) % values.size()];
      point.ny = values[(i + 4) % values.size()];
      point.nz = values[(i + 5) % values.size()];
      point.r = static_cast<uint8_t>(i);
      point.g = static_cast<uint8_t>(17 * i + 3);
      point.b = static_cast<uint8_t>(31 * i + 7);
    }
    for (const bool write_normal : {false, true}) {
      for (const bool write_rgb : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << count << " points, normal=" << write_normal
                     << ", rgb=" << write_rgb);
        WriteBinaryPlyPoints(test_file, points, write_normal, write_rgb);
        std::ifstream file(test_file, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        const std::string actual((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
        const size_t marker = actual.find("end_header");
        ASSERT_NE(marker, std::string::npos);
        const size_t header_end = actual.find('\n', marker);
        ASSERT_NE(header_end, std::string::npos);
        // The scalar wire writer independently defines field order and endian
        // conversion, including signed zero and non-finite float bits.
        std::ostringstream expected(std::ios::out | std::ios::binary);
        for (const auto& point : points) {
          WriteBinaryLittleEndian<float>(&expected, point.x);
          WriteBinaryLittleEndian<float>(&expected, point.y);
          WriteBinaryLittleEndian<float>(&expected, point.z);
          if (write_normal) {
            WriteBinaryLittleEndian<float>(&expected, point.nx);
            WriteBinaryLittleEndian<float>(&expected, point.ny);
            WriteBinaryLittleEndian<float>(&expected, point.nz);
          }
          if (write_rgb) {
            WriteBinaryLittleEndian<uint8_t>(&expected, point.r);
            WriteBinaryLittleEndian<uint8_t>(&expected, point.g);
            WriteBinaryLittleEndian<uint8_t>(&expected, point.b);
          }
        }
        EXPECT_EQ(actual.substr(header_end + 1), expected.str());
        const auto decoded = ReadPly(test_file);
        ASSERT_EQ(decoded.size(), points.size());
        const auto bits = [](float value) {
          uint32_t result;
          std::memcpy(&result, &value, sizeof(result));
          return result;
        };
        for (size_t i = 0; i < points.size(); ++i) {
          const auto& a = decoded[i];
          const auto& b = points[i];
          const std::array<float, 6> actual_values = {
              a.x, a.y, a.z, a.nx, a.ny, a.nz};
          const std::array<float, 6> expected_values = {
              b.x,
              b.y,
              b.z,
              write_normal ? b.nx : 0.0f,
              write_normal ? b.ny : 0.0f,
              write_normal ? b.nz : 0.0f};
          for (size_t field = 0; field < actual_values.size(); ++field) {
            EXPECT_EQ(bits(actual_values[field]), bits(expected_values[field]));
          }
          EXPECT_EQ(a.r, write_rgb ? b.r : 0);
          EXPECT_EQ(a.g, write_rgb ? b.g : 0);
          EXPECT_EQ(a.b, write_rgb ? b.b : 0);
        }
      }
    }
  }
}

TEST(Ply, ReadBinaryMixedLayoutAndOversizedRecords) {
  const auto path = CreateTestDir() / "mixed.ply";
  for (const bool little_endian : {false, true}) {
    for (const size_t extra_doubles : {size_t{0}, size_t{4096}}) {
      SCOPED_TRACE(::testing::Message() << "little=" << little_endian
                                        << ", extra=" << extra_doubles);
      const size_t count = extra_doubles == 0 ? 2049 : 3;
      std::ofstream file(path, std::ios::binary);
      file << "ply\nformat binary_" << (little_endian ? "little" : "big")
           << "_endian 1.0\nelement vertex " << count
           << "\nproperty uchar green\nproperty double z\nproperty float nx\n"
              "property float x\n";
      for (size_t i = 0; i < extra_doubles; ++i) {
        file << "property double ignored" << i << '\n';
      }
      file << "property uchar blue\nproperty double ny\nproperty float y\n"
              "property float nz\nproperty uchar red\nend_header\n";
      const auto write = [&](const auto value) {
        if (little_endian) {
          WriteBinaryLittleEndian(&file, value);
        } else {
          const auto encoded = NativeToBigEndian(value);
          file.write(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
        }
      };
      for (size_t i = 0; i < count; ++i) {
        write(static_cast<uint8_t>(i));
        write(-2.75 - 0.25 * i);
        write(0.125f);
        write(i == 0 ? -0.0f : static_cast<float>(i) + 0.25f);
        for (size_t extra = 0; extra < extra_doubles; ++extra) write(123.5);
        write(static_cast<uint8_t>(31 * i + 7));
        write(-0.5);
        write(static_cast<float>(i) - 0.75f);
        write(1.0f);
        write(static_cast<uint8_t>(17 * i + 3));
      }
      file.close();
      const auto points = ReadPly(path);
      ASSERT_EQ(points.size(), count);
      EXPECT_TRUE(std::signbit(points.front().x));
      for (size_t i = 0; i < count; ++i) {
        const auto& point = points[i];
        EXPECT_FLOAT_EQ(point.x,
                        i == 0 ? -0.0f : static_cast<float>(i) + 0.25f);
        EXPECT_FLOAT_EQ(point.y, static_cast<float>(i) - 0.75f);
        EXPECT_FLOAT_EQ(point.z, static_cast<float>(-2.75 - 0.25 * i));
        EXPECT_FLOAT_EQ(point.nx, 0.125f);
        EXPECT_FLOAT_EQ(point.ny, -0.5f);
        EXPECT_FLOAT_EQ(point.nz, 1.0f);
        EXPECT_EQ(point.r, static_cast<uint8_t>(17 * i + 3));
        EXPECT_EQ(point.g, static_cast<uint8_t>(i));
        EXPECT_EQ(point.b, static_cast<uint8_t>(31 * i + 7));
      }
    }
  }
}

TEST(Ply, ReadBinaryTruncationReportsFirstIncompleteVertex) {
  const auto path = CreateTestDir() / "truncated-binary.ply";
  for (const size_t complete_vertices :
       {size_t{0}, size_t{2730}, size_t{2731}}) {
    for (const size_t trailing_bytes : {size_t{0}, size_t{7}}) {
      SCOPED_TRACE(::testing::Message() << complete_vertices << " complete, "
                                        << trailing_bytes << " trailing bytes");
      std::ofstream file(path, std::ios::binary);
      file << "ply\nformat binary_little_endian 1.0\nelement vertex 4097\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n";
      for (size_t i = 0; i < complete_vertices; ++i) {
        WriteBinaryLittleEndian<float>(&file, 1.0f);
        WriteBinaryLittleEndian<float>(&file, 2.0f);
        WriteBinaryLittleEndian<float>(&file, 3.0f);
      }
      const std::array<char, 7> partial{};
      file.write(partial.data(), trailing_bytes);
      file.close();
      try {
        ReadPly(path);
        FAIL() << "Expected truncated PLY input to be rejected";
      } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what())
                      .find("vertex " + std::to_string(complete_vertices)),
                  std::string::npos);
      }
    }
  }
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

TEST(Ply, RoundTripLargeBinaryPlyPointsFullData) {
  const auto test_file = CreateTestDir() / "large.ply";
  // Full records exceed 1 MiB and leave a partial final output buffer.
  std::vector<PlyPoint> original_points(40001);
  for (size_t i = 0; i < original_points.size(); ++i) {
    const float value = static_cast<float>(static_cast<int>(i % 2048) - 1024);
    auto& point = original_points[i];
    point.x = value / 8.0f;
    point.y = -value / 16.0f;
    point.z = static_cast<float>(i) / 32.0f;
    point.nx = value / 1024.0f;
    point.ny = -value / 2048.0f;
    point.nz = static_cast<float>(i % 513) / 512.0f;
    point.r = static_cast<uint8_t>(i % 256);
    point.g = static_cast<uint8_t>((i * 7 + 31) % 256);
    point.b = static_cast<uint8_t>((i * 13 + 127) % 256);
  }

  WriteBinaryPlyPoints(test_file, original_points, true, true);
  ASSERT_GT(std::filesystem::file_size(test_file), 1024 * 1024);
  const auto loaded_points = ReadPly(test_file);
  ASSERT_EQ(loaded_points.size(), original_points.size());
  for (size_t i = 0; i < original_points.size(); ++i) {
    const auto& expected = original_points[i];
    const auto& actual = loaded_points[i];
    EXPECT_EQ(actual.x, expected.x);
    EXPECT_EQ(actual.y, expected.y);
    EXPECT_EQ(actual.z, expected.z);
    EXPECT_EQ(actual.nx, expected.nx);
    EXPECT_EQ(actual.ny, expected.ny);
    EXPECT_EQ(actual.nz, expected.nz);
    EXPECT_EQ(actual.r, expected.r);
    EXPECT_EQ(actual.g, expected.g);
    EXPECT_EQ(actual.b, expected.b);
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

}  // namespace
}  // namespace colmap
