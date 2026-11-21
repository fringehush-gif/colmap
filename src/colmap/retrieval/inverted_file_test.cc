// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/inverted_file.h"

#include "colmap/retrieval/inverted_index.h"

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

TEST(InvertedFile, ScoringRequiresEmbeddingAndSortedEntries) {
  const Eigen::VectorXf descriptor = Eigen::VectorXf::Zero(16);
  for (const bool has_embedding : {false, true}) {
    for (const bool sorted : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "has_embedding=" << has_embedding
                                        << ", sorted=" << sorted);
      InvertedFile<16> inverted_file;
      if (has_embedding) {
        inverted_file.ComputeHammingEmbedding(
            Eigen::Matrix<float, Eigen::Dynamic, 16>::Zero(2, 16));
      }
      inverted_file.AddEntry(2, 0, descriptor, FeatureGeometry());
      inverted_file.AddEntry(1, 0, descriptor, FeatureGeometry());
      if (sorted) {
        inverted_file.SortEntries();
      }
      inverted_file.ComputeIDFWeight(3);
      EXPECT_EQ(inverted_file.HasHammingEmbedding(), has_embedding);
      EXPECT_EQ(inverted_file.EntriesSorted(), sorted);
      EXPECT_EQ(inverted_file.IsUsable(), has_embedding && sorted);

      std::vector<ImageScore> scores(1);
      inverted_file.ScoreFeature(descriptor, &scores);
      if (has_embedding && sorted) {
        ASSERT_EQ(scores.size(), 2);
        EXPECT_EQ(scores[0].image_id, 1);
        EXPECT_EQ(scores[1].image_id, 2);
        EXPECT_GT(scores[0].score, 0);
        EXPECT_EQ(scores[0].score, scores[1].score);
      } else {
        EXPECT_TRUE(scores.empty());
      }
    }
  }
}

TEST(InvertedFile, UntrainedWordsDoNotContributeToQuery) {
  using Index = InvertedIndex<float, 32, 16>;
  for (const int num_samples_per_word : {4, 5}) {
    SCOPED_TRACE(num_samples_per_word);
    Index index;
    index.Initialize(2);
    // Straddle the five-sample minimum for learning each word's embedding.
    Index::DescType training =
        Index::DescType::Zero(2 * num_samples_per_word, 32);
    training.bottomRows(num_samples_per_word).setOnes();
    Index::WordIds word_ids(2 * num_samples_per_word, 1);
    word_ids.topRows(num_samples_per_word).setZero();
    word_ids.bottomRows(num_samples_per_word).setOnes();
    index.ComputeHammingEmbedding(training, word_ids);

    const Index::DescType query = Index::DescType::Zero(1, 32);
    index.AddEntry(1, 0, 0, query, FeatureGeometry());
    index.AddEntry(2, 1, 0, Index::DescType::Ones(1, 32), FeatureGeometry());
    index.Finalize();

    std::vector<ImageScore> scores(1);
    index.Query(query, Index::WordIds::Zero(1, 1), &scores);
    if (num_samples_per_word < 5) {
      EXPECT_TRUE(scores.empty());
    } else {
      ASSERT_EQ(scores.size(), 1);
      EXPECT_EQ(scores[0].image_id, 1);
      EXPECT_GT(scores[0].score, 0);
    }
  }
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap
