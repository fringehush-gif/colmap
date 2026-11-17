// SPDX-License-Identifier: BSD-3-Clause
#include <tuple>

// The vendor configuration must precede FEMTree.h.
// clang-format off
#include "PreProcessor.h"
#include "FEMTree.h"
// clang-format on
#include <gtest/gtest.h>

namespace {

template <unsigned int Dim, unsigned int Degree>
void CheckPaddedSupport(const int full_depth) {
  PoissonRecon::FEMTree<Dim, float> tree(0);
  tree.template finalizeForMultigrid<Degree, Degree>(
      full_depth,
      [full_depth](const int depth, const int*) { return depth <= full_depth; },
      [](const auto*) { return true; },
      std::make_tuple());
  const PoissonRecon::IsotropicUIntPack<Dim, Degree> degrees;
  ASSERT_EQ(tree.maxDepth(), full_depth);
  ASSERT_EQ(tree.getFullDepth(degrees), full_depth);
  for (const unsigned int depth : {0u, 1u, 4u, 12u}) {
    const int resolution = static_cast<int>(1u << depth);
    typename PoissonRecon::FEMTree<Dim, float>::LocalOffset begin{};
    typename PoissonRecon::FEMTree<Dim, float>::LocalOffset end{};
    for (unsigned int corner = 0; corner < (1u << Dim); ++corner) {
      SCOPED_TRACE(::testing::Message()
                   << "degree=" << Degree << " full_depth=" << full_depth
                   << " depth=" << depth << " corner=" << corner);
      for (unsigned int dim = 0; dim < Dim; ++dim) {
        begin[dim] = (corner & (1u << dim)) ? resolution - 1 : 0;
        end[dim] = begin[dim] + 1;
      }
      // Corner support visits padded nodes with negative local offsets.
      EXPECT_EQ(tree.getFullDepth(degrees, depth, begin, end), full_depth);
    }
  }
}

class PoissonFEMTree : public ::testing::Test {
 protected:
  void SetUp() override {
    num_threads_ = PoissonRecon::ThreadPool::NumThreads();
    PoissonRecon::ThreadPool::SetNumThreads(1);
  }
  void TearDown() override {
    PoissonRecon::ThreadPool::SetNumThreads(num_threads_);
  }

 private:
  unsigned int num_threads_ = 0;
};

TEST_F(PoissonFEMTree, PaddedSquareSupport) {
  for (const int full_depth : {0, 2}) {
    CheckPaddedSupport<2, 1>(full_depth);
    CheckPaddedSupport<2, 2>(full_depth);
    CheckPaddedSupport<2, 3>(full_depth);
  }
}

TEST_F(PoissonFEMTree, PaddedCubeSupport) {
  for (const int full_depth : {0, 2}) {
    CheckPaddedSupport<3, 1>(full_depth);
    CheckPaddedSupport<3, 2>(full_depth);
    CheckPaddedSupport<3, 3>(full_depth);
  }
}

}  // namespace
