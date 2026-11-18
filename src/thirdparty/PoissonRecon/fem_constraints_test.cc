// SPDX-License-Identifier: BSD-3-Clause

// FEMTree requires the vendor configuration before its declarations.
// clang-format off
#include "thirdparty/PoissonRecon/PreProcessor.h"
#include "thirdparty/PoissonRecon/FEMTree.h"
// clang-format on

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

namespace {

using PoissonRecon::FEMTreeInitializer;
using PoissonRecon::ThreadPool;
using Tree = PoissonRecon::FEMTree<3, float>;
using Point3 = PoissonRecon::Point<float, 3>;

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

double QuadraticBasis(double x) {
  x = std::abs(x);
  if (x <= 0.5) return 0.75 - x * x;
  if (x < 1.5) return 0.5 * (1.5 - x) * (1.5 - x);
  return 0;
}

// Integrate B1*B2 and B1'*B2 independently of the vendor integrator. Two
// Gauss points are exact on each polynomial interval (degree at most three).
std::array<double, 2> BasisIntegrals(double test_center,
                                     double test_width,
                                     double coefficient_center,
                                     double coefficient_width) {
  const double begin = std::max(test_center - test_width,
                                coefficient_center - 1.5 * coefficient_width);
  const double end = std::min(test_center + test_width,
                              coefficient_center + 1.5 * coefficient_width);
  if (begin >= end) return {0, 0};
  std::array<double, 7> knots{test_center - test_width,
                              test_center,
                              test_center + test_width,
                              coefficient_center - 1.5 * coefficient_width,
                              coefficient_center - 0.5 * coefficient_width,
                              coefficient_center + 0.5 * coefficient_width,
                              coefficient_center + 1.5 * coefficient_width};
  std::sort(knots.begin(), knots.end());
  std::array<double, 2> result{};
  for (size_t i = 1; i < knots.size(); ++i) {
    const double low = std::max(begin, knots[i - 1]);
    const double high = std::min(end, knots[i]);
    if (low >= high) continue;
    const double half = (high - low) / 2;
    const double midpoint = (high + low) / 2;
    for (const double x :
         {midpoint - half / std::sqrt(3.), midpoint + half / std::sqrt(3.)}) {
      const double weight =
          half * QuadraticBasis((x - coefficient_center) / coefficient_width);
      result[0] += weight * (1 - std::abs(x - test_center) / test_width);
      result[1] += weight * (x < test_center ? 1 : -1) / test_width;
    }
  }
  return result;
}

template <PoissonRecon::BoundaryType TestBoundary,
          PoissonRecon::BoundaryType CoefficientBoundary>
void ExpectImpulseConstraints(const std::array<int, 3>& source,
                              bool reflect_coefficient) {
  const SerialThreadPool serial;
  using TestSignatures = PoissonRecon::IsotropicUIntPack<
      3,
      PoissonRecon::FEMDegreeAndBType<1, TestBoundary>::Signature>;
  using CoefficientSignatures = PoissonRecon::IsotropicUIntPack<
      3,
      PoissonRecon::FEMDegreeAndBType<2, CoefficientBoundary>::Signature>;
  Tree tree(1024);
  FEMTreeInitializer<3, float>::Initialize(
      tree.spaceRoot(),
      5,
      [](int, int*) { return true; },
      tree.nodeAllocators.front(),
      tree.initializer());
  const Point3 normal(1.f, 0.25f, -0.5f);
  PoissonRecon::SparseNodeData<Point3, CoefficientSignatures> field;
  int assigned = 0;
  tree.tree().processNodes([&](const Tree::FEMTreeNode* node) {
    int depth, offset[3];
    tree.depthAndOffset(node, depth, offset);
    if (depth == 3 && std::equal(offset, offset + 3, source.begin())) {
      field[node] = normal;
      ++assigned;
    }
  });
  ASSERT_EQ(assigned, 1);
  Tree::HasNormalDataFunctor<CoefficientSignatures> present(field);
  tree.finalizeForMultigrid<2, 1>(
      0,
      [](int depth, const int*) { return depth <= 5; },
      present,
      std::make_tuple(),
      std::make_tuple(&field));
  auto rhs = tree.initDenseNodeData(TestSignatures());
  PoissonRecon::FEMIntegrator::Constraint<TestSignatures,
                                          PoissonRecon::IsotropicUIntPack<3, 1>,
                                          CoefficientSignatures,
                                          PoissonRecon::IsotropicUIntPack<3, 0>,
                                          3>
      integrator;
  for (int component = 0; component < 3; ++component) {
    unsigned int derivative[3]{};
    derivative[component] = 1;
    integrator.weights[component][PoissonRecon::TensorDerivatives<
        PoissonRecon::IsotropicUIntPack<3, 1>>::Index(derivative)][0] = 1;
  }
  tree.addFEMConstraints(integrator, field, rhs, 5);

  std::vector<std::array<int, 3>> coefficients{source};
  // At the x=0 wall, the Neumann quadratic basis has an equal reflected
  // coefficient at -h/2. Other supports are entirely inside the domain.
  if (reflect_coefficient) coefficients.push_back({-1, source[1], source[2]});
  std::array<int, 6> checked{}, nonzero{}, nonfinite{};
  std::array<double, 6> max_error{};
  tree.tree().processNodes([&](const Tree::FEMTreeNode* node) {
    int depth, offset[3];
    tree.depthAndOffset(node, depth, offset);
    if (node->nodeData.nodeIndex < 0 || depth < 2 || depth > 5) return;
    const int resolution = 1 << depth;
    for (const int off : offset)
      if (off <= 0 || off >= resolution) return;
    const double width = 1. / resolution;
    double expected = 0;
    for (const auto& coefficient : coefficients) {
      std::array<std::array<double, 2>, 3> integrals;
      for (int axis = 0; axis < 3; ++axis) {
        integrals[axis] = BasisIntegrals(
            offset[axis] * width, width, (coefficient[axis] + 0.5) / 8, 1. / 8);
      }
      for (int component = 0; component < 3; ++component) {
        expected += normal[component] * integrals[component][1] *
                    integrals[(component + 1) % 3][0] *
                    integrals[(component + 2) % 3][0];
      }
    }
    const float* actual = rhs(node);
    if (!actual) return;
    ++checked[depth];
    nonzero[depth] += std::abs(expected) > 1e-9;
    // Keep the finite check meaningful with the vendor's -ffast-math flags.
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    std::uint32_t bits;
    std::memcpy(&bits, actual, sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
      ++nonfinite[depth];
      return;
    }
    max_error[depth] = std::max(max_error[depth], std::abs(*actual - expected));
  });
  for (int depth = 2; depth <= 5; ++depth) {
    SCOPED_TRACE(depth);
    const int interior_size = (1 << depth) - 1;
    EXPECT_EQ(checked[depth], interior_size * interior_size * interior_size);
    EXPECT_GT(nonzero[depth], 0);
    EXPECT_EQ(nonfinite[depth], 0);
    EXPECT_LE(max_error[depth], 2e-7);
  }
}

TEST(PoissonFEMConstraints, ProlongsInCoefficientBasis) {
  ExpectImpulseConstraints<PoissonRecon::BOUNDARY_NEUMANN,
                           PoissonRecon::BOUNDARY_DIRICHLET>({3, 3, 3}, false);
}

TEST(PoissonFEMConstraints, UsesCoefficientBoundaryFlags) {
  ExpectImpulseConstraints<PoissonRecon::BOUNDARY_DIRICHLET,
                           PoissonRecon::BOUNDARY_NEUMANN>({0, 3, 3}, true);
}

}  // namespace
