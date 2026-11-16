// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/inverted_index.h"

#include <bitset>

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

template <int kDescDim, int kEmbeddingDim>
void CheckProjectionAndWordThresholds() {
  SCOPED_TRACE(kDescDim);
  using Index = InvertedIndex<float, kDescDim, kEmbeddingDim>;
  Index index;
  index.Initialize(2);

  // The initial identity projection gives a closed-form oracle. Each word's
  // five training descriptors have the same value, so their thresholds are
  // exactly -1 and +1 in every projected dimension.
  typename Index::DescType training(10, kDescDim);
  training.topRows(5).setConstant(-1);
  training.bottomRows(5).setConstant(1);
  typename Index::WordIds word_ids(10, 1);
  word_ids.topRows(5).setZero();
  word_ids.bottomRows(5).setOnes();
  index.ComputeHammingEmbedding(training, word_ids);

  typename Index::DescType descriptors(3, kDescDim);
  for (int row = 0; row < descriptors.rows(); ++row) {
    for (int col = 0; col < descriptors.cols(); ++col) {
      descriptors(row, col) = (row + col) % 5 - 2;
    }
  }

  for (int row = 0; row < descriptors.rows(); ++row) {
    SCOPED_TRACE(row);
    const auto projected = index.ProjectDescriptor(descriptors.row(row));
    ASSERT_EQ(projected.size(), kEmbeddingDim);
    for (int col = 0; col < kEmbeddingDim; ++col) {
      EXPECT_EQ(projected[col], descriptors(row, col));
    }

    std::bitset<kEmbeddingDim> binary_descriptor;
    binary_descriptor.set();
    for (int word = 0; word < 2; ++word) {
      SCOPED_TRACE(word);
      std::bitset<kEmbeddingDim> expected;
      for (int col = 0; col < kEmbeddingDim; ++col) {
        expected[col] = descriptors(row, col) > (word == 0 ? -1 : 1);
      }
      // Reuse both the projection and the output storage across words. The
      // threshold comparison remains word-specific and strictly greater than,
      // including components exactly on either threshold.
      index.ConvertProjectedToBinaryDescriptor(
          word, projected, &binary_descriptor);
      EXPECT_EQ(binary_descriptor, expected);

      index.ConvertToBinaryDescriptor(
          word, descriptors.row(row), &binary_descriptor);
      EXPECT_EQ(binary_descriptor, expected);
    }
  }
}

TEST(InvertedIndex, ProjectionUsesWordSpecificThresholds) {
  ASSERT_NO_FATAL_FAILURE((CheckProjectionAndWordThresholds<32, 16>()));
  ASSERT_NO_FATAL_FAILURE((CheckProjectionAndWordThresholds<128, 64>()));
  ASSERT_NO_FATAL_FAILURE((CheckProjectionAndWordThresholds<256, 64>()));
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap
