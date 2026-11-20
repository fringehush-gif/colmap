// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/testing.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

// Match the owner configuration before any vendor data headers.
// clang-format off
#include "PreProcessor.h"
// clang-format on

#include "Geometry.h"
#include "Ply.h"
#include "SurfaceTrimmer.h"
#include "VertexFactory.h"
#include <gtest/gtest.h>

namespace {

using Point = PoissonRecon::Point<float, 3>;
using Factory = PoissonRecon::VertexFactory::Factory<
    float,
    PoissonRecon::VertexFactory::PositionFactory<float, 3>,
    PoissonRecon::VertexFactory::ValueFactory<float>,
    PoissonRecon::VertexFactory::RGBColorFactory<float>>;
using Vertex = Factory::VertexType;

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<std::vector<int>> faces;
};

void TrimMesh(const Mesh& input, const char* area_ratio, Mesh* output) {
  const Factory factory;
  const auto directory = colmap::CreateTestDir();
  const auto input_path = directory / "input.ply";
  const auto output_path = directory / "output.ply";
  PoissonRecon::PLY::WritePolygons<Factory, int>(input_path.string(),
                                                 factory,
                                                 input.vertices,
                                                 input.faces,
                                                 PLY_BINARY_NATIVE,
                                                 {});
  // Every call supplies all varying options. No boolean option is changed:
  // this suite does not depend on the separate invocation-state repair.
  std::vector<std::string> args{"SurfaceTrimmer",
                                "--in",
                                input_path.string(),
                                "--out",
                                output_path.string(),
                                "--trim",
                                "5",
                                "--aRatio",
                                area_ratio};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) argv.push_back(arg.data());
  ASSERT_EQ(RunSurfaceTrimmer(static_cast<int>(argv.size()), argv.data()), 0);
  std::vector<std::string> comments;
  int file_type = 0;
  PoissonRecon::PLY::ReadPolygons<Factory, int>(output_path.string(),
                                                factory,
                                                output->vertices,
                                                output->faces,
                                                file_type,
                                                comments);
  EXPECT_TRUE(file_type == PLY_BINARY_LE || file_type == PLY_BINARY_BE);
  std::filesystem::remove(input_path);
  std::filesystem::remove(output_path);
}

void ExpectMesh(const Mesh& actual, const Mesh& expected) {
  ASSERT_EQ(actual.vertices.size(), expected.vertices.size());
  ASSERT_EQ(actual.faces.size(), expected.faces.size());
  std::vector<int> index_map(actual.vertices.size(), -1);
  std::vector<bool> used(expected.vertices.size(), false);
  for (size_t i = 0; i < actual.vertices.size(); ++i) {
    const auto& vertex = actual.vertices[i];
    for (size_t j = 0; j < expected.vertices.size(); ++j) {
      const auto& target = expected.vertices[j];
      if (vertex.template get<0>()[0] == target.template get<0>()[0] &&
          vertex.template get<0>()[1] == target.template get<0>()[1] &&
          vertex.template get<0>()[2] == target.template get<0>()[2]) {
        ASSERT_FALSE(used[j]);
        used[j] = true;
        index_map[i] = static_cast<int>(j);
        EXPECT_EQ(vertex.template get<1>(), target.template get<1>());
        for (size_t channel = 0; channel < 3; ++channel) {
          EXPECT_EQ(vertex.template get<2>()[channel],
                    target.template get<2>()[channel]);
        }
        break;
      }
    }
    ASSERT_GE(index_map[i], 0);
  }
  auto faces = actual.faces;
  for (auto& face : faces) {
    ASSERT_EQ(face.size(), 3);
    for (int& index : face) {
      ASSERT_GE(index, 0);
      ASSERT_LT(static_cast<size_t>(index), index_map.size());
      index = index_map[index];
    }
  }
  const auto Canonicalize = [](std::vector<std::vector<int>>* polygons) {
    for (auto& polygon : *polygons) {
      std::rotate(polygon.begin(),
                  std::min_element(polygon.begin(), polygon.end()),
                  polygon.end());
    }
    std::sort(polygons->begin(), polygons->end());
  };
  auto expected_faces = expected.faces;
  Canonicalize(&faces);
  Canonicalize(&expected_faces);
  // Only cyclic rotation is allowed within a face, preserving its winding.
  EXPECT_EQ(faces, expected_faces);
}

TEST(SurfaceTrimmerClipping, RetainsExactEndpointsAcrossWindingAndStart) {
  for (const std::array<float, 3>& values :
       {std::array<float, 3>{5, 6, 6}, std::array<float, 3>{5, 5, 6}}) {
    const std::vector<Vertex> vertices{
        Vertex(Point(0, 0, 0), values[0], Point(20, 40, 60)),
        Vertex(Point(1, 0, 0), values[1], Point(60, 80, 100)),
        Vertex(Point(0, 1, 0), values[2], Point(100, 120, 140))};
    for (const char* area_ratio : {"0", "0.001"}) {
      std::vector<int> face{0, 1, 2};
      do {
        SCOPED_TRACE(testing::Message() << values[1] << ":" << area_ratio << ":"
                                        << face[0] << face[1] << face[2]);
        const Mesh input{vertices, {face}};
        Mesh output;
        ASSERT_NO_FATAL_FAILURE(TrimMesh(input, area_ratio, &output));
        ExpectMesh(output, input);
      } while (std::next_permutation(face.begin(), face.end()));
    }
  }
}

TEST(SurfaceTrimmerClipping, PreservesStrictClassificationOfWholeFaces) {
  for (const float value : {5.0f, 6.0f}) {
    const Mesh input{{Vertex(Point(0, 0, 0), value, Point(20, 40, 60)),
                      Vertex(Point(1, 0, 0), value, Point(60, 80, 100)),
                      Vertex(Point(0, 1, 0), value, Point(100, 120, 140))},
                     {{0, 1, 2}}};
    Mesh output;
    ASSERT_NO_FATAL_FAILURE(TrimMesh(input, "0", &output));
    ExpectMesh(output, value > 5 ? input : Mesh{});
    Mesh repeated;
    ASSERT_NO_FATAL_FAILURE(TrimMesh(output, "0", &repeated));
    ExpectMesh(repeated, output);
  }
}

TEST(SurfaceTrimmerClipping, SharesInteriorIntersectionsAndInterpolatesColors) {
  const Mesh input{{Vertex(Point(0, 0, 0), 6, Point(20, 40, 60)),
                    Vertex(Point(1, 0, 0), 4, Point(60, 80, 100)),
                    Vertex(Point(1, 1, 0), 4, Point(100, 120, 140)),
                    Vertex(Point(0, 1, 0), 4, Point(140, 160, 180))},
                   {{0, 1, 2}, {0, 2, 3}}};
  const Mesh expected{{Vertex(Point(0, 0, 0), 6, Point(20, 40, 60)),
                       Vertex(Point(0.5f, 0, 0), 5, Point(40, 60, 80)),
                       Vertex(Point(0.5f, 0.5f, 0), 5, Point(60, 80, 100)),
                       Vertex(Point(0, 0.5f, 0), 5, Point(80, 100, 120))},
                      {{0, 1, 2}, {0, 2, 3}}};
  Mesh output;
  ASSERT_NO_FATAL_FAILURE(TrimMesh(input, "0", &output));
  ExpectMesh(output, expected);
}

TEST(SurfaceTrimmerClipping, KeepsTinyValidInteriorTriangle) {
  constexpr float edge = 1.0f / 1048576.0f;  // 2^-20.
  const Mesh input{{Vertex(Point(0, 0, 0), 6, Point(20, 40, 60)),
                    Vertex(Point(edge, 0, 0), 4, Point(60, 80, 100)),
                    Vertex(Point(0, edge, 0), 4, Point(100, 120, 140))},
                   {{0, 1, 2}}};
  const Mesh expected{{Vertex(Point(0, 0, 0), 6, Point(20, 40, 60)),
                       Vertex(Point(edge / 2, 0, 0), 5, Point(40, 60, 80)),
                       Vertex(Point(0, edge / 2, 0), 5, Point(60, 80, 100))},
                      {{0, 1, 2}}};
  Mesh output;
  ASSERT_NO_FATAL_FAILURE(TrimMesh(input, "0", &output));
  ASSERT_NO_FATAL_FAILURE(ExpectMesh(output, expected));
  const auto& face = output.faces[0];
  const auto& a = output.vertices[face[0]].template get<0>();
  const auto& b = output.vertices[face[1]].template get<0>();
  const auto& c = output.vertices[face[2]].template get<0>();
  const double cross = (double(b[0]) - a[0]) * (double(c[1]) - a[1]) -
                       (double(b[1]) - a[1]) * (double(c[0]) - a[0]);
  EXPECT_EQ(cross / 2, double(edge) * edge / 8);  // Area 2^-43.
}

TEST(SurfaceTrimmerClipping, RetainsBoundaryValuesWhenTrimmingAgain) {
  const std::vector<Vertex> vertices{
      Vertex(Point(0, 0, 0), 15, Point(20, 40, 60)),
      Vertex(Point(1, 0, 0), 0, Point(60, 80, 100)),
      Vertex(Point(0, 1, 0), 0, Point(100, 120, 140))};
  for (const char* area_ratio : {"0", "0.001"}) {
    std::vector<int> face{0, 1, 2};
    do {
      SCOPED_TRACE(testing::Message()
                   << area_ratio << ":" << face[0] << face[1] << face[2]);
      Mesh first;
      ASSERT_NO_FATAL_FAILURE(TrimMesh({vertices, {face}}, area_ratio, &first));
      ASSERT_EQ(first.vertices.size(), 3);
      ASSERT_EQ(first.faces.size(), 1);
      for (const auto& vertex : first.vertices) {
        // Every new crossing is on the requested trim level, even when its
        // rounded interpolation weights do not reproduce that scalar value.
        EXPECT_TRUE(vertex.template get<1>() == 15 ||
                    vertex.template get<1>() == 5);
      }
      Mesh previous = first;
      for (int pass = 0; pass < 2; ++pass) {
        SCOPED_TRACE(pass);
        Mesh next;
        ASSERT_NO_FATAL_FAILURE(TrimMesh(previous, area_ratio, &next));
        ASSERT_NO_FATAL_FAILURE(ExpectMesh(next, first));
        previous = next;
      }
    } while (std::next_permutation(face.begin(), face.end()));
  }
}

}  // namespace
