// SPDX-License-Identifier: BSD-3-Clause

// FEMTree requires the vendor configuration before its declarations.
// clang-format off
#include "thirdparty/PoissonRecon/PreProcessor.h"
#include "thirdparty/PoissonRecon/FEMTree.h"
// clang-format on

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using PoissonRecon::FEMTreeInitializer;
using PoissonRecon::ProjectiveData;
using PoissonRecon::ThreadPool;
using Tree = PoissonRecon::FEMTree<3, float>;
using Point3 = PoissonRecon::Point<float, 3>;
using Signatures = PoissonRecon::
    IsotropicUIntPack<3, PoissonRecon::FEMDegreeAndBType<2>::Signature>;
using Field = PoissonRecon::SparseNodeData<Point3, Signatures>;
using Address = std::array<int, 4>;
// Sorted addresses make the complete coefficient evidence deterministic.
using Coefficients = std::map<Address, std::array<float, 3>>;

constexpr float kNormalizationTolerance =
    4 * std::numeric_limits<float>::epsilon();
constexpr float kMetadataTolerance = 32 * std::numeric_limits<float>::epsilon();
// Bound normalization, local-coordinate conditioning and three B2 factors.
// Retain the vendor fast-math context instead of requiring exact division.
constexpr float kCoefficientTolerance =
    2048 * std::numeric_limits<float>::epsilon();

struct SerialThreadPool {
  const unsigned int threads = ThreadPool::NumThreads();
  const ThreadPool::ParallelType parallel = ThreadPool::ParallelizationType;
  SerialThreadPool() {
    ThreadPool::SetNumThreads(1);
    ThreadPool::ParallelizationType = ThreadPool::NONE;
  }
  ~SerialThreadPool() {
    ThreadPool::SetNumThreads(threads);
    ThreadPool::ParallelizationType = parallel;
  }
};

void ExpectFinite(float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t));
  static_assert(std::numeric_limits<float>::is_iec559 &&
                std::numeric_limits<float>::digits == 24);
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  // std::isfinite can be optimized away by the owner's -ffast-math flags.
  EXPECT_NE(bits & UINT32_C(0x7f800000), UINT32_C(0x7f800000));
}

// Centered cardinal B2, independent of the vendor's polynomial evaluators.
double QuadraticBasis(double x) {
  x = std::abs(x);
  if (x <= 0.5) return 0.75 - x * x;
  if (x < 1.5) return 0.5 * (1.5 - x) * (1.5 - x);
  return 0;
}

Coefficients ExpectedCoefficients(const Point3& position,
                                  const Point3& value,
                                  bool fractional) {
  Coefficients expected;
  for (int depth = 3; depth <= (fractional ? 4 : 3); ++depth) {
    const double width = std::ldexp(1.0, -depth);
    const double scale =
        (fractional ? 0.5 : 1.0) / (64 * width * width * width);
    const std::array<int, 3> cell = {
        static_cast<int>(std::floor(position[0] / width)),
        static_cast<int>(std::floor(position[1] / width)),
        static_cast<int>(std::floor(position[2] / width))};
    for (int z = -1; z <= 1; ++z) {
      for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
          const Address address{depth, cell[0] + x, cell[1] + y, cell[2] + z};
          double basis = 1;
          for (int axis = 0; axis < 3; ++axis) {
            basis *= QuadraticBasis(position[axis] / width -
                                    (address[axis + 1] + 0.5));
          }
          for (int component = 0; component < 3; ++component) {
            expected[address][component] =
                static_cast<float>(value[component] * scale * basis);
          }
        }
      }
    }
  }
  return expected;
}

Coefficients Snapshot(const Tree& tree, const Field& field) {
  Coefficients values;
  tree.tree().processNodes([&](const Tree::FEMTreeNode* node) {
    if (node->nodeData.nodeIndex < 0) return;
    const Point3* value = field(node);
    if (!value) return;
    int depth, offset[3];
    tree.depthAndOffset(node, depth, offset);
    for (int component = 0; component < 3; ++component) {
      ExpectFinite((*value)[component]);
    }
    values[{depth, offset[0], offset[1], offset[2]}] = {
        (*value)[0], (*value)[1], (*value)[2]};
  });
  return values;
}

TEST(PoissonWeightedSamples, SplatsFractionalDepthIntoBothLevels) {
  const SerialThreadPool serial;
  Tree tree(1024);
  FEMTreeInitializer<3, float>::Initialize(
      tree.spaceRoot(),
      4,
      [](int, int*) { return true; },
      tree.nodeAllocators.front(),
      tree.initializer());
  const Point3 position(15.f / 32, 17.f / 32, 19.f / 32);
  const Point3 value(1.f, 0.25f, -0.5f);
  std::vector<Tree::PointSample> samples;
  FEMTreeInitializer<3, float>::Initialize(
      tree.spaceRoot(),
      {ProjectiveData<Point3, float>(position, 1.f)},
      4,
      samples,
      tree.nodeAllocators,
      tree.initializer());
  ASSERT_EQ(samples.size(), 1);

  // Uniform B2 coefficients sum to one. Target density 1/2 gives raw depth
  // 3.5 and weight 1/64, safely above the coarser-density search threshold.
  Tree::DensityEstimator<2> density(3, 1, 0.5f);
  tree.tree().processNodes([&](const Tree::FEMTreeNode* node) {
    if (tree.depth(node) == 3) density[node] = 1.f;
  });
  struct Case {
    const char* name;
    float bias;
    int max_depth;
  };
  for (const Case test : {Case{"integer", -0.5f, 3},
                          Case{"fractional", 0.f, 4},
                          Case{"clamped", 0.f, 3}}) {
    SCOPED_TRACE(test.name);
    const bool fractional = test.max_depth == 4;
    const auto expected = ExpectedCoefficients(position, value, fractional);
    ASSERT_EQ(expected.size(), fractional ? 54 : 27);
    for (const bool use_functor : {false, true}) {
      SCOPED_TRACE(use_functor);
      int conversion_calls = 0, depth_calls = 0;
      std::function<bool(Point3, Point3&, float&)> convert =
          [&](Point3 input, Point3& output, float& bias) {
            ++conversion_calls;
            for (int component = 0; component < 3; ++component) {
              ExpectFinite(input[component]);
              EXPECT_NEAR(input[component],
                          value[component],
                          kNormalizationTolerance * std::abs(value[component]));
            }
            output = input;
            bias = test.bias;
            return true;
          };
      std::function<int(Point3)> maximum = [&](Point3 input) {
        ++depth_calls;
        for (int axis = 0; axis < 3; ++axis) {
          ExpectFinite(input[axis]);
          EXPECT_NEAR(input[axis],
                      position[axis],
                      kNormalizationTolerance * std::abs(position[axis]));
        }
        return test.max_depth;
      };
      ProjectiveData<PoissonRecon::Point<float, 2>, float> metadata;
      const auto field =
          use_functor
              ? tree.setInterpolatedDataField(Point3(),
                                              Signatures(),
                                              samples,
                                              std::vector<Point3>{value},
                                              &density,
                                              3,
                                              test.max_depth,
                                              maximum,
                                              0.f,
                                              metadata,
                                              convert)
              : tree.setInterpolatedDataField(Point3(),
                                              Signatures(),
                                              samples,
                                              std::vector<Point3>{value},
                                              &density,
                                              3,
                                              test.max_depth,
                                              0.f,
                                              metadata,
                                              convert);
      EXPECT_EQ(conversion_calls, 1);
      EXPECT_EQ(depth_calls, use_functor ? 1 : 0);
      ExpectFinite(metadata.data[0]);
      ExpectFinite(metadata.data[1]);
      ExpectFinite(metadata.weight);
      // Depth has an absolute bound; the density weight has a relative bound.
      EXPECT_NEAR(metadata.data[0], 3.5f + test.bias, kMetadataTolerance);
      EXPECT_NEAR(metadata.data[1], 1.f / 64, kMetadataTolerance / 64);
      EXPECT_EQ(metadata.weight, 1.f);
      const auto actual = Snapshot(tree, field);
      EXPECT_EQ(actual.size(), expected.size());
      for (const auto& item : expected) {
        const auto found = actual.find(item.first);
        EXPECT_NE(found, actual.end());
        if (found != actual.end()) {
          for (int component = 0; component < 3; ++component) {
            const float actual_value = found->second[component];
            const float expected_value = item.second[component];
            EXPECT_NEAR(actual_value,
                        expected_value,
                        kCoefficientTolerance * std::abs(expected_value));
          }
        }
      }
      std::ostringstream evidence;
      evidence << std::hexfloat << metadata.data[0] << ',' << metadata.data[1]
               << ',' << metadata.weight << ';';
      for (const auto& item : actual) {
        for (const int address : item.first) evidence << address << ',';
        for (const float coefficient : item.second)
          evidence << coefficient << ',';
        evidence << ';';
      }
      RecordProperty(
          std::string(test.name) + (use_functor ? "_functor" : "_fixed"),
          evidence.str());
    }
  }
}

}  // namespace
