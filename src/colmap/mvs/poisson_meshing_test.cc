// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/poisson_meshing.h"

#include "colmap/math/random_eigen.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

// clang-format off
#include "thirdparty/PoissonRecon/PreProcessor.h"
// clang-format on

#include "thirdparty/PoissonRecon/MultiThreading.h"
#include "thirdparty/PoissonRecon/Ply.h"
#include "thirdparty/PoissonRecon/PoissonRecon.h"
#include "thirdparty/PoissonRecon/VertexFactory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

void WriteRandomPlyPoints(const std::filesystem::path& path,
                          int num_points = 100) {
  std::vector<PlyPoint> ply_points;
  ply_points.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    const Eigen::Vector3d point3D = RandomEigenVectord<3>();
    PlyPoint ply_point;
    ply_point.x = point3D.x();
    ply_point.y = point3D.y();
    ply_point.z = point3D.z();
    ply_point.nx = 0.0f;
    ply_point.ny = 0.0f;
    ply_point.nz = 1.0f;
    ply_point.r = 0;
    ply_point.g = 64;
    ply_point.b = 128;
    ply_points.push_back(ply_point);
  }
  WriteBinaryPlyPoints(
      path, ply_points, /*write_normal=*/true, /*write_rgb=*/true);
}

TEST(PoissonMeshing, Integration) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "points.ply";
  const auto output_path = test_dir / "mesh.ply";
  WriteRandomPlyPoints(input_path);

  PoissonMeshingOptions options;
  options.point_weight = 1.0;
  options.depth = 3;   // Use smaller depth for faster test
  options.trim = 0.0;  // Disable trimming
  options.num_threads = 1;

  EXPECT_TRUE(PoissonMeshing(options, input_path, output_path));

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST(PoissonMeshing, WithTrimming) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "points.ply";
  const auto output_path = test_dir / "mesh.ply";
  WriteRandomPlyPoints(input_path);

  PoissonMeshingOptions options;
  options.point_weight = 1.0;
  options.depth = 3;
  options.trim = 5.0;
  options.num_threads = 1;

  EXPECT_TRUE(PoissonMeshing(options, input_path, output_path));
  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  // With random data and trimming, we can't make strong assumptions about
  // the number of vertices, but reading the file ensures valid PLY format.
  EXPECT_GE(mesh_vertices.size(), 0);
}

std::vector<PlyPoint> MakeDeterministicColoredCubePoints() {
  std::vector<PlyPoint> points;
  points.reserve(96);
  for (int axis = 0; axis < 3; ++axis) {
    for (int sign : std::array<int, 2>{-1, 1}) {
      for (int u = 0; u < 4; ++u) {
        for (int v = 0; v < 4; ++v) {
          std::array<float, 3> position{};
          std::array<float, 3> normal{};
          position[axis] = static_cast<float>(sign);
          position[(axis + 1) % 3] = static_cast<float>(2 * u - 3) * 0.25f;
          position[(axis + 2) % 3] = static_cast<float>(2 * v - 3) * 0.25f;
          normal[axis] = static_cast<float>(sign);
          PlyPoint point;
          point.x = position[0];
          point.y = position[1];
          point.z = position[2];
          point.nx = normal[0];
          point.ny = normal[1];
          point.nz = normal[2];
          point.r = 17;
          point.g = 109;
          point.b = 233;
          points.push_back(point);
        }
      }
    }
  }
  return points;
}

bool WriteAuxiliaryAsciiPly(const std::filesystem::path& path,
                            const std::vector<PlyPoint>& points,
                            const std::vector<std::string>& color_properties,
                            const bool write_quality) {
  std::ofstream output(path);
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  output << "ply\nformat ascii 1.0\nelement vertex " << points.size()
         << "\nproperty float x\nproperty float y\nproperty float z"
         << "\nproperty float nx\nproperty float ny\nproperty float nz\n";
  for (const auto& name : color_properties) {
    output << "property uchar " << name << '\n';
  }
  if (write_quality) {
    output << "property float quality\n";
  }
  output << "end_header\n";
  for (const auto& point : points) {
    output << point.x << ' ' << point.y << ' ' << point.z << ' ' << point.nx
           << ' ' << point.ny << ' ' << point.nz;
    for (const auto& name : color_properties) {
      const auto value = name == "red" || name == "r"     ? point.r
                         : name == "green" || name == "g" ? point.g
                                                          : point.b;
      output << ' ' << static_cast<int>(value);
    }
    if (write_quality) {
      output << " 7";
    }
    output << '\n';
  }
  output.close();
  return output.good();
}

void CheckMeshProperties(const std::filesystem::path& path,
                         std::vector<std::string> auxiliary_properties) {
  const auto mesh = ReadPlyMesh(path).mesh;
  ASSERT_GE(mesh.vertices.size(), 3);
  ASSERT_FALSE(mesh.faces.empty());
  for (const auto& vertex : mesh.vertices) {
    EXPECT_TRUE(std::isfinite(vertex.x));
    EXPECT_TRUE(std::isfinite(vertex.y));
    EXPECT_TRUE(std::isfinite(vertex.z));
  }
  for (const auto& face : mesh.faces) {
    EXPECT_LT(face.vertex_idx1, mesh.vertices.size());
    EXPECT_LT(face.vertex_idx2, mesh.vertices.size());
    EXPECT_LT(face.vertex_idx3, mesh.vertices.size());
  }
  std::vector<PoissonRecon::PlyProperty> properties;
  PoissonRecon::PLY::ReadVertexHeader(path.string(), properties);
  std::vector<std::string> names;
  for (const auto& property : properties) {
    names.push_back(property.name);
  }
  auxiliary_properties.insert(auxiliary_properties.end(), {"x", "y", "z"});
  std::sort(names.begin(), names.end());
  std::sort(auxiliary_properties.begin(), auxiliary_properties.end());
  EXPECT_EQ(names, auxiliary_properties);

  std::vector<PoissonRecon::PlyProperty> attributes;
  std::vector<std::pair<float, float>> expected_values;
  for (const auto& property : properties) {
    if (property.name == "quality") {
      attributes.push_back(property);
      expected_values.emplace_back(7.0f, 0.01f);
    } else if (property.name == "red" || property.name == "r") {
      attributes.push_back(property);
      expected_values.emplace_back(17.0f, 1.0f);
    } else if (property.name == "green" || property.name == "g") {
      attributes.push_back(property);
      expected_values.emplace_back(109.0f, 1.0f);
    } else if (property.name == "blue" || property.name == "b") {
      attributes.push_back(property);
      expected_values.emplace_back(233.0f, 1.0f);
    }
  }
  if (!attributes.empty()) {
    using Factory = PoissonRecon::VertexFactory::DynamicFactory<float>;
    const Factory factory(attributes);
    std::vector<Factory::VertexType> values;
    std::vector<std::vector<int>> polygons;
    std::vector<std::string> comments;
    int file_type = 0;
    auto read_flags = std::make_unique<bool[]>(attributes.size());
    PoissonRecon::PLY::ReadPolygons(path.string(),
                                    factory,
                                    values,
                                    polygons,
                                    file_type,
                                    comments,
                                    read_flags.get());
    for (size_t index = 0; index < attributes.size(); ++index) {
      ASSERT_TRUE(read_flags[index]);
    }
    ASSERT_EQ(values.size(), mesh.vertices.size());
    for (const auto& value : values) {
      for (size_t index = 0; index < attributes.size(); ++index) {
        EXPECT_NEAR(value[index],
                    expected_values[index].first,
                    expected_values[index].second);
      }
    }
  }
}

#if GTEST_HAS_DEATH_TEST

// The vendor CLI stores omitted flags globally. Threadsafe death tests re-exec
// the binary, so each parameter instance invokes the solver exactly once.
// CreateTestDir also clears its directory on first use in each process: keeping
// one invocation per instance lets the parent read the actual child output.
template <typename Param>
class IsolatedPoissonTest : public testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    old_style_ = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
  }
  void TearDown() override { GTEST_FLAG_SET(death_test_style, old_style_); }

 private:
  std::string old_style_;
};

PoissonMeshingOptions ColorTestOptions(const bool color) {
  PoissonMeshingOptions options;
  options.depth = 3;
  options.point_weight = 1.0;
  options.trim = 0.0;
  options.num_threads = 1;
  options.color = color;
  return options;
}

using PoissonMeshingRgbDeathTest = IsolatedPoissonTest<std::tuple<int, bool>>;

TEST_P(PoissonMeshingRgbDeathTest,
       ColorOptionControlsStandardReorderedAndAliasRgb) {
  const auto test_dir = CreateTestDir();
  const auto input = test_dir / "points.ply";
  const auto output = test_dir / "mesh.ply";
  const auto [layout, color] = GetParam();
  const auto points = MakeDeterministicColoredCubePoints();
  const std::vector<std::string> colors =
      layout == 2 ? std::vector<std::string>{"r", "g", "b"}
                  : std::vector<std::string>{"red", "green", "blue"};
  if (layout == 0) {
    WriteBinaryPlyPoints(input, points, true, true);
  } else {
    const std::vector<std::string> order =
        layout == 1 ? std::vector<std::string>{"blue", "red", "green"} : colors;
    ASSERT_TRUE(WriteAuxiliaryAsciiPly(input, points, order, false));
  }
  const auto options = ColorTestOptions(color);
  ASSERT_EXIT(std::_Exit(PoissonMeshing(options, input, output) ? EXIT_SUCCESS
                                                                : EXIT_FAILURE),
              testing::ExitedWithCode(EXIT_SUCCESS),
              "");
  ASSERT_TRUE(ExistsFile(output));
  CheckMeshProperties(output, color ? colors : std::vector<std::string>{});
}

INSTANTIATE_TEST_SUITE_P(StandardReorderedAliases,
                         PoissonMeshingRgbDeathTest,
                         testing::Combine(testing::Values(0, 1, 2),
                                          testing::Bool()));

using PoissonMeshingColorlessDeathTest = IsolatedPoissonTest<bool>;

TEST_P(PoissonMeshingColorlessDeathTest,
       ColorlessInputRetainsGeometryOnlyOutput) {
  const auto test_dir = CreateTestDir();
  const auto input = test_dir / "points.ply";
  const auto output = test_dir / "mesh.ply";
  WriteBinaryPlyPoints(
      input, MakeDeterministicColoredCubePoints(), true, false);
  const auto options = ColorTestOptions(GetParam());
  ASSERT_EXIT(std::_Exit(PoissonMeshing(options, input, output) ? EXIT_SUCCESS
                                                                : EXIT_FAILURE),
              testing::ExitedWithCode(EXIT_SUCCESS),
              "");
  ASSERT_TRUE(ExistsFile(output));
  CheckMeshProperties(output, {});
}

INSTANTIATE_TEST_SUITE_P(ColorDisabledAndEnabled,
                         PoissonMeshingColorlessDeathTest,
                         testing::Bool());

using PoissonMeshingAuxiliaryDeathTest =
    IsolatedPoissonTest<std::tuple<int, bool>>;

TEST_P(PoissonMeshingAuxiliaryDeathTest,
       ColorOptionPreservesNonColorAuxiliaryProperties) {
  const auto test_dir = CreateTestDir();
  const auto input = test_dir / "points.ply";
  const auto output = test_dir / "mesh.ply";
  const auto [layout, color] = GetParam();
  const std::vector<std::vector<std::string>> layouts = {
      {},
      {"red", "green", "blue"},
      {"r", "g", "b"},
      {"red"},
      {"r", "green", "b"},
      {"red", "green", "blue", "r", "g", "b"}};
  ASSERT_TRUE(WriteAuxiliaryAsciiPly(
      input, MakeDeterministicColoredCubePoints(), layouts[layout], true));
  const auto options = ColorTestOptions(color);
  ASSERT_EXIT(std::_Exit(PoissonMeshing(options, input, output) ? EXIT_SUCCESS
                                                                : EXIT_FAILURE),
              testing::ExitedWithCode(EXIT_SUCCESS),
              "");
  ASSERT_TRUE(ExistsFile(output));
  std::vector<std::string> expected{"quality"};
  if (color) {
    expected.insert(
        expected.end(), layouts[layout].begin(), layouts[layout].end());
  }
  CheckMeshProperties(output, expected);
}

INSTANTIATE_TEST_SUITE_P(QualityWithRgbPartialMixedAndRedundantAliases,
                         PoissonMeshingAuxiliaryDeathTest,
                         testing::Combine(testing::Range(0, 6),
                                          testing::Bool()));

using PoissonMeshingGenericDeathTest = IsolatedPoissonTest<bool>;

TEST_P(PoissonMeshingGenericDeathTest,
       GenericEntryRetainsAutomaticAuxiliaryProperties) {
  const auto test_dir = CreateTestDir();
  const auto input = test_dir / "points.ply";
  const auto output = test_dir / "mesh.ply";
  const bool quality = GetParam();
  const std::vector<std::string> colors{"red", "green", "blue"};
  ASSERT_TRUE(WriteAuxiliaryAsciiPly(
      input, MakeDeterministicColoredCubePoints(), colors, quality));
  std::vector<std::string> args = {"poisson_recon",
                                   "--in",
                                   input.string(),
                                   "--out",
                                   output.string(),
                                   "--depth",
                                   "3",
                                   "--fullDepth",
                                   "3"};
  std::vector<char*> argv;
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  ASSERT_EXIT(
      {
        PoissonRecon::ThreadPool::SetNumThreads(1);
        std::_Exit(RunPoissonRecon(static_cast<int>(argv.size()), argv.data()));
      },
      testing::ExitedWithCode(EXIT_SUCCESS),
      "");
  ASSERT_TRUE(ExistsFile(output));
  std::vector<std::string> expected = colors;
  if (quality) {
    expected.push_back("quality");
  }
  CheckMeshProperties(output, expected);
}

INSTANTIATE_TEST_SUITE_P(RgbWithAndWithoutQuality,
                         PoissonMeshingGenericDeathTest,
                         testing::Bool());

#else

TEST(PoissonMeshing, ColorOptionRequiresProcessIsolation) {
  GTEST_SKIP() << "Color-option regression tests require death-test support";
}

#endif  // GTEST_HAS_DEATH_TEST

}  // namespace
}  // namespace mvs
}  // namespace colmap
