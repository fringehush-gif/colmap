// SPDX-License-Identifier: BSD-3-Clause
// The vendor configuration must precede its declarations.
// clang-format off
#include "PreProcessor.h"
#include "PoissonRecon.h"
#include "Reconstructors.h"
#include "SurfaceTrimmer.h"
// clang-format on

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace PoissonRecon {
namespace {

using FloatPoint = Point<float, 3>;
using DoublePoint = Point<double, 3>;
using ColorFactory = VertexFactory::RGBColorFactory<float>;
using ReadFactory =
    VertexFactory::Factory<float,
                           VertexFactory::PositionFactory<double, 3>,
                           VertexFactory::NormalFactory<float, 3>,
                           VertexFactory::ValueFactory<float>,
                           ColorFactory>;
using Vertex = ReadFactory::VertexType;

// The external negative control changes only the selected production types.
#ifdef LEGACY_POSITION_OUTPUT
using PositionReal = float;
using VInfo =
    Reconstructor::OutputVertexInfo<float, 3, true, true, ColorFactory>;
template <bool InCore>
using Buffer = Reconstructor::OutputInputFactoryTypeStream<float,
                                                           3,
                                                           VInfo::Factory,
                                                           InCore,
                                                           true,
                                                           FloatPoint>;
using Mapper =
    Reconstructor::TransformedOutputLevelSetVertexStream<float, 3, FloatPoint>;
#else
using PositionReal = double;
using VInfo = Reconstructor::
    OutputVertexInfoWithPosition<double, float, 3, true, true, ColorFactory>;
template <bool InCore>
using Buffer =
    Reconstructor::OutputInputFactoryTypeStreamWithPosition<double,
                                                            float,
                                                            3,
                                                            VInfo::Factory,
                                                            InCore,
                                                            true,
                                                            FloatPoint>;
using Mapper = Reconstructor::TransformedOutputLevelSetVertexStreamWithPosition<
    double,
    float,
    3,
    FloatPoint>;
#endif

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::atomic<unsigned int> sequence{0};
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (unsigned int attempt = 0; attempt < 100; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("poisson-position-" + std::to_string(stamp) + "-" +
               std::to_string(sequence++));
      if (std::filesystem::create_directory(path_)) return;
    }
    throw std::runtime_error("Could not create a unique test directory");
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  std::string File(const std::string& name) const {
    return (path_ / name).string();
  }

 private:
  std::filesystem::path path_;
};

class SerialThreads {
 public:
  SerialThreads()
      : count_(ThreadPool::NumThreads()),
        mode_(ThreadPool::ParallelizationType) {
    ThreadPool::SetNumThreads(1);
    ThreadPool::ParallelizationType = ThreadPool::NONE;
  }
  ~SerialThreads() {
    ThreadPool::SetNumThreads(count_);
    ThreadPool::ParallelizationType = mode_;
  }

 private:
  unsigned int count_;
  ThreadPool::ParallelType mode_;
};

ReadFactory GetReadFactory() {
  return ReadFactory(VertexFactory::PositionFactory<double, 3>(),
                     VertexFactory::NormalFactory<float, 3>(),
                     VertexFactory::ValueFactory<float>(),
                     ColorFactory());
}

float FromBits(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

const std::array<std::array<std::uint32_t, 3>, 4> kInternalBits{{
    {{1054081024u, 1061027840u, 1040187391u}},
    {{1054081023u, 1061027840u, 1040187392u}},
    {{1054081024u, 1061027840u, 1040187392u}},
    {{1054212096u, 1061039147u, 1040127284u}},
}};
const std::array<std::array<double, 3>, 4> kMapped{{
    {{-0.18906250409781933, 0.5328125115483999, -0.8250000342726711}},
    {{-0.18906256966293, 0.5328125115483999, -0.8250000178813934}},
    {{-0.18906250409781933, 0.5328125115483999, -0.8250000178813934}},
    {{-0.1804687539115548, 0.5342952009611821, -0.8259852647994919}},
}};
constexpr std::array<std::array<int, 3>, 2> kFaces{{{{2, 0, 1}}, {{2, 3, 0}}}};

template <bool InCore>
void WriteRetainedWitness(const std::string& filename, int format) {
  XForm<float, 4> transform;
  for (unsigned int c = 0; c < 4; ++c)
    for (unsigned int r = 0; r < 4; ++r) transform(c, r) = 0;
  for (unsigned int d = 0; d < 3; ++d) {
    transform(d, d) = FromBits(0x400ccccdu);
    transform(3, d) = FromBits(0xbf8ccccdu);
  }
  transform(3, 3) = 1;
  auto factory = VInfo::GetFactory(ColorFactory());
  Buffer<InCore> vertices(factory, VInfo::Convert);
  Mapper mapped(transform, vertices);
  Reconstructor::OutputInputFaceStream<2, InCore, true> faces;
  for (const auto& bits : kInternalBits)
    mapped.write(
        0,
        FloatPoint(FromBits(bits[0]), FromBits(bits[1]), FromBits(bits[2])),
        FloatPoint(1, 2, 3),
        4.0f,
        FloatPoint(12, 34, 56));
  for (const auto& face : kFaces)
    faces.write(0, std::vector<node_index_type>(face.begin(), face.end()));
  PLY::Write<typename VInfo::Factory, node_index_type, PositionReal, 3>(
      filename,
      factory,
      vertices.size(),
      faces.size(),
      vertices,
      faces,
      format,
      {});
}

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<std::vector<int>> faces;
};

Mesh ReadMesh(const std::string& filename) {
  Mesh mesh;
  int format;
  std::vector<std::string> comments;
  PLY::ReadPolygons<ReadFactory, int>(
      filename, GetReadFactory(), mesh.vertices, mesh.faces, format, comments);
  return mesh;
}

void ExpectRetainedWitness(const Mesh& mesh) {
  ASSERT_GE(mesh.vertices.size(), 4);
  ASSERT_GE(mesh.faces.size(), 2);
  for (size_t i = 0; i < 4; ++i) {
    for (unsigned int d = 0; d < 3; ++d) {
      EXPECT_EQ(mesh.vertices[i].get<0>()[d], kMapped[i][d]);
      // Uniform-scale gradient normalization uses unchanged float pow/inverse.
      EXPECT_NEAR(mesh.vertices[i].get<1>()[d],
                  float(d + 1),
                  8 * std::numeric_limits<float>::epsilon() * float(d + 1));
      EXPECT_EQ(mesh.vertices[i].get<3>()[d], float(12 + 22 * d));
    }
    EXPECT_EQ(mesh.vertices[i].get<2>(), 4.0f);
  }
  EXPECT_NE(mesh.vertices[0].get<0>()[2], mesh.vertices[2].get<0>()[2]);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(mesh.faces[i],
              std::vector<int>(kFaces[i].begin(), kFaces[i].end()));
    const auto a = mesh.vertices[kFaces[i][1]].get<0>() -
                   mesh.vertices[kFaces[i][0]].get<0>();
    const auto b = mesh.vertices[kFaces[i][2]].get<0>() -
                   mesh.vertices[kFaces[i][0]].get<0>();
    const auto cross = DoublePoint::CrossProduct(a, b);
    EXPECT_GT(DoublePoint::SquareNorm(cross), 0.0);
  }
}

int RunEntry(int (*entry)(int, char**), std::vector<std::string> arguments) {
  std::vector<char*> raw;
  raw.reserve(arguments.size());
  for (auto& argument : arguments) raw.push_back(argument.data());
  return entry(static_cast<int>(raw.size()), raw.data());
}

TEST(OutputPositionPrecision, PreservesDistinctMappedVerticesInBothPlyModes) {
  SerialThreads threads;
  TemporaryDirectory directory;
  for (int format : {PLY_BINARY_NATIVE, PLY_ASCII}) {
    const auto in_core = directory.File("in-core.ply");
    const auto file_backed = directory.File("file-backed.ply");
    WriteRetainedWitness<true>(in_core, format);
    WriteRetainedWitness<false>(file_backed, format);
    for (const auto& path : {in_core, file_backed}) {
      const auto mesh = ReadMesh(path);
      ASSERT_EQ(mesh.vertices.size(), 4);
      ASSERT_EQ(mesh.faces.size(), 2);
      ExpectRetainedWitness(mesh);
    }
  }
}

TEST(OutputPositionPrecision, TrimsWidePositionsAndTranslatedIntersections) {
  SerialThreads threads;
  TemporaryDirectory directory;
  const auto input = directory.File("input.ply");
  const auto output = directory.File("output.ply");
  WriteRetainedWitness<true>(input, PLY_BINARY_NATIVE);
  auto mesh = ReadMesh(input);
  constexpr double translation = 1048576;
  const std::array<DoublePoint, 3> triangle{
      {DoublePoint(translation, translation, translation),
       DoublePoint(translation + 3, translation, translation),
       DoublePoint(translation, translation + 3, translation)}};
  for (size_t i = 0; i < triangle.size(); ++i) {
    Vertex vertex;
    vertex.get<0>() = triangle[i];
    vertex.get<1>() = FloatPoint(1, 2, 3);
    vertex.get<2>() = i == 0 ? 1.0f : 4.0f;
    vertex.get<3>() = FloatPoint(5 + 12 * float(i), 0, 0);
    mesh.vertices.push_back(vertex);
  }
  mesh.faces.push_back({4, 5, 6});
  PLY::WritePolygons<ReadFactory, int>(input,
                                       GetReadFactory(),
                                       mesh.vertices,
                                       mesh.faces,
                                       PLY_BINARY_NATIVE,
                                       {});
  // One trimmer invocation per process: its command-line options are static.
  ASSERT_EQ(RunEntry(RunSurfaceTrimmer,
                     {"SurfaceTrimmer",
                      "--in",
                      input,
                      "--out",
                      output,
                      "--trim",
                      "2",
                      "--aRatio",
                      "0",
                      "--polygonMesh",
                      "--ascii"}),
            0);
  const auto trimmed = ReadMesh(output);
  ASSERT_EQ(trimmed.vertices.size(), 8);
  ASSERT_EQ(trimmed.faces.size(), 3);
  ExpectRetainedWitness(trimmed);
  EXPECT_EQ(trimmed.faces[2], (std::vector<int>{6, 4, 5, 7}));
  const double dx = static_cast<double>(1.0f / 3.0f);
  EXPECT_EQ(trimmed.vertices[6].get<0>()[0], translation + 3 * dx);
  EXPECT_EQ(trimmed.vertices[6].get<0>()[1], translation);
  EXPECT_EQ(trimmed.vertices[6].get<0>()[2], translation);
  EXPECT_EQ(trimmed.vertices[6].get<2>(), 2.0f);
  EXPECT_EQ(trimmed.vertices[6].get<3>()[0], 9.0f);
}

// The test inherits the vendor's fast-math flags. Inspect the representation
// instead of calling a floating predicate that those flags may optimize away.
bool IsFiniteBinary64(const double& value) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr std::uint64_t kExponentMask = 0x7ff0000000000000ULL;
  return (bits & kExponentMask) != kExponentMask;
}

TEST(OutputPositionPrecision,
     DefaultPublicReconstructionWritesDoublePositions) {
  // These controls ensure that fast-math has not made the check vacuous.
  for (std::uint64_t pattern : {0x7ff0000000000000ULL, 0x7ff8000000000001ULL}) {
    double nonfinite;
    std::memcpy(&nonfinite, &pattern, sizeof(nonfinite));
    EXPECT_FALSE(IsFiniteBinary64(nonfinite));
  }
  SerialThreads threads;
  TemporaryDirectory directory;
  const auto input = directory.File("sphere.ply");
  const auto output = directory.File("surface.ply");
  {
    std::ofstream points(input);
    points << "ply\nformat ascii 1.0\nelement vertex 128\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar "
              "blue\nend_header\n";
    points << std::setprecision(std::numeric_limits<float>::max_digits10);
    const double pi = std::acos(-1.0);
    for (int t = 0; t < 8; ++t)
      for (int p = 0; p < 16; ++p) {
        const double theta = pi * (t + 0.5) / 8;
        const double phi = 2 * pi * p / 16;
        const FloatPoint point(float(std::sin(theta) * std::cos(phi)),
                               float(std::sin(theta) * std::sin(phi)),
                               float(std::cos(theta)));
        for (int repeat = 0; repeat < 2; ++repeat)
          for (unsigned int d = 0; d < 3; ++d) points << point[d] << ' ';
        points << "12 34 56\n";
      }
  }
  ASSERT_EQ(RunEntry(RunPoissonRecon,
                     {"PoissonRecon",
                      "--in",
                      input,
                      "--out",
                      output,
                      "--depth",
                      "4",
                      "--fullDepth",
                      "2",
                      "--parallel",
                      std::to_string(int(ThreadPool::NONE)),
                      "--colors",
                      "--density",
                      "--gradients"}),
            0);
  std::vector<PlyProperty> properties;
  PLY::ReadVertexHeader(output, properties);
  for (const std::string name : {"x", "y", "z"}) {
    const auto property = std::find_if(
        properties.begin(), properties.end(), [&](const PlyProperty& p) {
          return p.name == name;
        });
    ASSERT_NE(property, properties.end());
    EXPECT_EQ(property->external_type, PLY_DOUBLE);
  }
  const auto mesh = ReadMesh(output);
  ASSERT_FALSE(mesh.vertices.empty());
  ASSERT_FALSE(mesh.faces.empty());
  for (const auto& vertex : mesh.vertices)
    for (unsigned int d = 0; d < 3; ++d)
      EXPECT_TRUE(IsFiniteBinary64(vertex.get<0>()[d]));
}

}  // namespace
}  // namespace PoissonRecon
