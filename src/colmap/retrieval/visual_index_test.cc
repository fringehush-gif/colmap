// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/visual_index.h"

#include "colmap/retrieval/inverted_index.h"

#include "colmap/feature/types.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <sstream>

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

// Helper to create random descriptors with a given type.
FeatureDescriptorsFloat CreateRandomDescriptors(int num_rows,
                                                int desc_dim,
                                                FeatureExtractorType type) {
  return FeatureDescriptorsFloat(
      type, FeatureDescriptorsFloatData::Random(num_rows, desc_dim));
}

TEST(InvertedFile, ReadinessRequiresEmbeddingAndSortedEntries) {
  for (const bool trained : {false, true}) {
    for (const bool sorted : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "trained=" << trained
                                      << " sorted=" << sorted);
      InvertedFile<16> file;
      EXPECT_FALSE(file.IsUsable());
      if (trained) {
        Eigen::Matrix<float, Eigen::Dynamic, 16> training(2, 16);
        training.row(0).setZero();
        training.row(1).setOnes();
        file.ComputeHammingEmbedding(training);
      }
      const Eigen::VectorXf descriptor = Eigen::VectorXf::Zero(16);
      file.AddEntry(2, 0, descriptor, FeatureGeometry());
      file.AddEntry(1, 0, descriptor, FeatureGeometry());
      if (sorted) {
        file.SortEntries();
      }
      file.ComputeIDFWeight(3);
      EXPECT_EQ(file.HasHammingEmbedding(), trained);
      EXPECT_EQ(file.EntriesSorted(), sorted);
      EXPECT_EQ(file.IsUsable(), trained && sorted);

      std::stringstream stream;
      file.Write(&stream);
      InvertedFile<16> restored;
      restored.Read(&stream);
      EXPECT_EQ(restored.HasHammingEmbedding(), trained);
      EXPECT_EQ(restored.EntriesSorted(), sorted);
      EXPECT_EQ(restored.IsUsable(), trained && sorted);
      std::vector<ImageScore> scores(1);
      restored.ScoreFeature(descriptor, &scores);
      if (trained && sorted) {
        ASSERT_EQ(scores.size(), 2);
        EXPECT_EQ(scores[0].image_id, 1);
        EXPECT_EQ(scores[1].image_id, 2);
        EXPECT_GT(scores[0].score, 0);
        EXPECT_FLOAT_EQ(scores[0].score, scores[1].score);
      } else {
        EXPECT_TRUE(scores.empty());
      }

      restored.AddEntry(3, 0, descriptor, FeatureGeometry());
      EXPECT_FALSE(restored.IsUsable());
      restored.SortEntries();
      EXPECT_EQ(restored.IsUsable(), trained);
      restored.ClearEntries();
      EXPECT_FALSE(restored.IsUsable());
      EXPECT_EQ(restored.HasHammingEmbedding(), trained);
      restored.Reset();
      EXPECT_FALSE(restored.HasHammingEmbedding());
      EXPECT_FALSE(restored.EntriesSorted());
    }
  }
}

TEST(InvertedIndex, SparseWordsDoNotSupplySpatialMatches) {
  using Index = InvertedIndex<float, 16, 16>;
  Index index;
  index.Initialize(2);
  index.GenerateHammingEmbeddingProjection();
  const Index::DescType training = Index::DescType::Zero(9, 16);
  Index::WordIds word_ids(9, 1);
  word_ids.topRows(5).setZero();
  word_ids.bottomRows(4).setOnes();
  index.ComputeHammingEmbedding(training, word_ids);
  const Index::DescType descriptor = Index::DescType::Zero(1, 16);
  index.AddEntry(1, 0, 0, descriptor, FeatureGeometry());
  index.AddEntry(2, 1, 0, descriptor, FeatureGeometry());
  const FlatHashSet<int> image_ids{1, 2};
  std::vector<const Index::EntryType*> matches(1, nullptr);
  index.FindMatches(0, image_ids, &matches);
  EXPECT_TRUE(matches.empty());
  index.Finalize();
  index.FindMatches(0, image_ids, &matches);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0]->image_id, 1);
  index.FindMatches(1, image_ids, &matches);
  EXPECT_TRUE(matches.empty());
  Index::WordIds query_words(1, 1);
  query_words(0, 0) = 1;
  std::vector<ImageScore> scores(1);
  index.Query(descriptor, query_words, &scores);
  EXPECT_TRUE(scores.empty());
  query_words(0, 0) = 0;
  index.Query(descriptor, query_words, &scores);
  ASSERT_EQ(scores.size(), 1);
  EXPECT_EQ(scores[0].image_id, 1);
  EXPECT_GT(scores[0].score, 0);

  std::stringstream stream;
  index.Write(&stream);
  Index restored;
  restored.Read(&stream);
  restored.FindMatches(0, image_ids, &matches);
  ASSERT_EQ(matches.size(), 1);
  restored.FindMatches(1, image_ids, &matches);
  EXPECT_TRUE(matches.empty());
  restored.AddEntry(3, 0, 0, descriptor, FeatureGeometry());
  restored.FindMatches(0, FlatHashSet<int>{1, 3}, &matches);
  EXPECT_TRUE(matches.empty());
  restored.Finalize();
  restored.FindMatches(0, FlatHashSet<int>{1, 3}, &matches);
  EXPECT_EQ(matches.size(), 2);
  restored.ClearEntries();
  restored.FindMatches(0, image_ids, &matches);
  EXPECT_TRUE(matches.empty());
}

class ParameterizedVisualIndexTests
    : public ::testing::TestWithParam<
          std::pair</*embedding_dim=*/int, /*embedding_dim=*/int>> {};

TEST_P(ParameterizedVisualIndexTests, Nominal) {
  const auto [desc_dim, embedding_dim] = GetParam();

  {
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
  }

  VisualIndex::BuildOptions build_options;
  // Keep test runtimes low.
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;

  {
    auto descriptors =
        CreateRandomDescriptors(200, desc_dim, FeatureExtractorType::SIFT);
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
    EXPECT_EQ(visual_index->NumImages(), 0);
    EXPECT_EQ(visual_index->DescDim(), desc_dim);
    EXPECT_EQ(visual_index->EmbeddingDim(), embedding_dim);
    build_options.num_visual_words = 5;
    visual_index->Build(build_options, descriptors);
    EXPECT_EQ(visual_index->NumVisualWords(), 5);
    EXPECT_EQ(visual_index->FeatureType(), FeatureExtractorType::SIFT);
  }

  {
    auto descriptors =
        CreateRandomDescriptors(4096, desc_dim, FeatureExtractorType::SIFT);
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
    EXPECT_EQ(visual_index->NumImages(), 0);
    build_options.num_visual_words = 512;
    visual_index->Build(build_options, descriptors);
    EXPECT_EQ(visual_index->NumVisualWords(), 512);
  }

  {
    auto descriptors =
        CreateRandomDescriptors(1000, desc_dim, FeatureExtractorType::SIFT);
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
    EXPECT_EQ(visual_index->NumImages(), 0);
    build_options.num_visual_words = 100;
    visual_index->Build(build_options, descriptors);
    EXPECT_EQ(visual_index->NumVisualWords(), 100);

    VisualIndex::IndexOptions index_options;
    FeatureKeypoints keypoints1(50);
    auto descriptors1 =
        CreateRandomDescriptors(50, desc_dim, FeatureExtractorType::SIFT);
    visual_index->Add(index_options, 1, keypoints1, descriptors1);
    EXPECT_EQ(visual_index->NumImages(), 1);
    FeatureKeypoints keypoints2(50);
    auto descriptors2 =
        CreateRandomDescriptors(50, desc_dim, FeatureExtractorType::SIFT);
    visual_index->Add(index_options, 2, keypoints2, descriptors2);
    EXPECT_EQ(visual_index->NumImages(), 2);
    visual_index->Prepare();

    VisualIndex::QueryOptions query_options;
    std::vector<ImageScore> image_scores;
    visual_index->Query(query_options, descriptors1, &image_scores);
    EXPECT_EQ(image_scores.size(), 2);
    EXPECT_EQ(image_scores[0].image_id, 1);
    EXPECT_EQ(image_scores[1].image_id, 2);
    EXPECT_GT(image_scores[0].score, image_scores[1].score);

    query_options.max_num_images = 1;
    visual_index->Query(query_options, descriptors1, &image_scores);
    EXPECT_EQ(image_scores.size(), 1);
    EXPECT_EQ(image_scores[0].image_id, 1);

    query_options.max_num_images = 3;
    visual_index->Query(query_options, descriptors1, &image_scores);
    EXPECT_EQ(image_scores.size(), 2);
    EXPECT_EQ(image_scores[0].image_id, 1);
    EXPECT_EQ(image_scores[1].image_id, 2);
    EXPECT_GT(image_scores[0].score, image_scores[1].score);

    query_options.max_num_images = 1;
    query_options.image_id_filter = [](const int image_id) {
      return image_id != 1;
    };
    visual_index->Query(query_options, descriptors1, &image_scores);
    ASSERT_EQ(image_scores.size(), 1);
    EXPECT_EQ(image_scores[0].image_id, 2);
  }
}

TEST_P(ParameterizedVisualIndexTests, ReadWrite) {
  const auto [desc_dim, embedding_dim] = GetParam();
  const auto test_dir = CreateTestDir();
  const auto vocab_tree_path = test_dir / "vocab_tree.bin";

  VisualIndex::BuildOptions build_options;
  // Keep test runtimes low.
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;

  auto descriptors =
      CreateRandomDescriptors(200, desc_dim, FeatureExtractorType::SIFT);
  auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
  EXPECT_EQ(visual_index->NumVisualWords(), 0);
  EXPECT_EQ(visual_index->DescDim(), desc_dim);
  EXPECT_EQ(visual_index->EmbeddingDim(), embedding_dim);
  build_options.num_visual_words = 5;
  visual_index->Build(build_options, descriptors);
  EXPECT_EQ(visual_index->NumVisualWords(), 5);
  EXPECT_EQ(visual_index->FeatureType(), FeatureExtractorType::SIFT);

  VisualIndex::IndexOptions index_options;
  FeatureKeypoints keypoints1(50);
  auto descriptors1 =
      CreateRandomDescriptors(50, desc_dim, FeatureExtractorType::SIFT);
  visual_index->Add(index_options, 1, keypoints1, descriptors1);
  FeatureKeypoints keypoints2(50);
  auto descriptors2 =
      CreateRandomDescriptors(50, desc_dim, FeatureExtractorType::SIFT);
  visual_index->Add(index_options, 2, keypoints2, descriptors2);

  EXPECT_EQ(visual_index->NumImages(), 2);

  visual_index->Write(vocab_tree_path);
  auto read_visual_index = VisualIndex::Read(vocab_tree_path);
  EXPECT_EQ(visual_index->NumVisualWords(),
            read_visual_index->NumVisualWords());
  EXPECT_EQ(visual_index->NumImages(), read_visual_index->NumImages());
  EXPECT_EQ(visual_index->DescDim(), read_visual_index->DescDim());
  EXPECT_EQ(visual_index->EmbeddingDim(), read_visual_index->EmbeddingDim());
  EXPECT_EQ(visual_index->FeatureType(), read_visual_index->FeatureType());
  EXPECT_TRUE(visual_index->IsImageIndexed(1));
  EXPECT_TRUE(visual_index->IsImageIndexed(2));
  EXPECT_FALSE(visual_index->IsImageIndexed(3));
}

TEST_P(ParameterizedVisualIndexTests, SpatialVerification) {
  const auto [desc_dim, embedding_dim] = GetParam();

  VisualIndex::BuildOptions build_options;
  // Keep test runtimes low.
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;
  build_options.num_visual_words = 200;
  // Use a single thread for deterministic results.
  build_options.num_threads = 1;

  const auto training_samples =
      CreateRandomDescriptors(1000, desc_dim, FeatureExtractorType::SIFT);
  // Every descriptor used below appears at least five times during training,
  // so its assigned word can learn the embedding required for verification.
  const FeatureDescriptorsFloat train_descriptors(
      FeatureExtractorType::SIFT, training_samples.data.replicate(5, 1));
  auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
  visual_index->Build(build_options, train_descriptors);
  EXPECT_EQ(visual_index->NumVisualWords(), build_options.num_visual_words);

  // Add multiple images with geometries and descriptors
  VisualIndex::IndexOptions index_options;
  constexpr int kNumImages = 5;
  constexpr int kNumFeatures = 50;

  FeatureKeypoints query_keypoints(kNumFeatures);
  for (size_t j = 0; j < query_keypoints.size(); ++j) {
    // Create a grid of keypoints (non-collinear)
    query_keypoints[j].x = static_cast<float>((j % 10) * kNumFeatures);
    query_keypoints[j].y = static_cast<float>((j / 10.f) * kNumFeatures);
  }

  // Use the same descriptors for all images.
  const FeatureDescriptorsFloat descriptors(
      FeatureExtractorType::SIFT, training_samples.data.topRows(kNumFeatures));

  for (int i = 0; i < kNumImages; ++i) {
    FeatureKeypoints keypoints(kNumFeatures);
    // For every other image, create keypoints with different but consistent
    // translational transformations to the query keypoints. The other outlier
    // images have inconsistent spatial transformations with random keypoints.
    if (i % 2 == 0) {
      // Create keypoints with different spatial transformations.
      const float x_offset = static_cast<float>(i * 20);
      const float y_offset = static_cast<float>(i * 30);
      for (size_t j = 0; j < keypoints.size(); ++j) {
        keypoints[j].x = query_keypoints[j].x + x_offset;
        keypoints[j].y = query_keypoints[j].y + y_offset;
      }
    } else {
      for (size_t j = 0; j < keypoints.size(); ++j) {
        const Eigen::Vector2f offset = RandomEigenVectorf<2>();
        keypoints[j].x = offset.x();
        keypoints[j].y = offset.y();
      }
    }

    visual_index->Add(index_options, /*image_id=*/i, keypoints, descriptors);
  }

  EXPECT_EQ(visual_index->NumImages(), kNumImages);
  visual_index->Prepare();

  VisualIndex::QueryOptions query_options;
  // Use the same word assignment for indexing and geometric verification.
  query_options.num_neighbors = index_options.num_neighbors;
  for (int feature_idx = 0; feature_idx < kNumFeatures; ++feature_idx) {
    const FeatureDescriptorsFloat descriptor(
        descriptors.type, descriptors.data.middleRows(feature_idx, 1));
    std::vector<ImageScore> scores;
    visual_index->Query(query_options, descriptor, &scores);
    ASSERT_EQ(scores.size(), kNumImages);
  }

  std::vector<ImageScore> image_scores_without_verification;
  visual_index->Query(query_options,
                      query_keypoints,
                      descriptors,
                      &image_scores_without_verification);
  EXPECT_EQ(image_scores_without_verification.size(), kNumImages);
  for (const auto& score : image_scores_without_verification) {
    EXPECT_LT(score.score, 1.0f);
  }

  query_options.num_images_after_verification = kNumImages;
  std::vector<ImageScore> image_scores_with_verification;
  visual_index->Query(query_options,
                      query_keypoints,
                      descriptors,
                      &image_scores_with_verification);
  EXPECT_EQ(image_scores_with_verification.size(), kNumImages);
  // Make sure inlier images have more keypoints as inliers.
  for (const auto& score : image_scores_with_verification) {
    if (score.image_id % 2 == 0) {
      EXPECT_GT(score.score, kNumFeatures / 2);
    } else {
      EXPECT_LT(score.score, kNumFeatures / 4);
    }
  }

  // Test with max_num_images constraint, should respect max_num_images.
  query_options.max_num_images = 2;
  query_options.num_images_after_verification = 4;
  std::vector<ImageScore> image_scores_limited;
  visual_index->Query(
      query_options, query_keypoints, descriptors, &image_scores_limited);
  EXPECT_LE(image_scores_limited.size(), query_options.max_num_images);
}

TEST_P(ParameterizedVisualIndexTests, TypeMismatch) {
  const auto [desc_dim, embedding_dim] = GetParam();

  VisualIndex::BuildOptions build_options;
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;
  build_options.num_visual_words = 10;

  // Build index with SIFT descriptors.
  auto sift_descriptors =
      CreateRandomDescriptors(100, desc_dim, FeatureExtractorType::SIFT);
  auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
  visual_index->Build(build_options, sift_descriptors);
  EXPECT_EQ(visual_index->FeatureType(), FeatureExtractorType::SIFT);

  // Adding descriptors with mismatched type should throw.
  VisualIndex::IndexOptions index_options;
  FeatureKeypoints keypoints(10);
  auto undefined_descriptors =
      CreateRandomDescriptors(10, desc_dim, FeatureExtractorType::UNDEFINED);
  EXPECT_THROW(
      visual_index->Add(index_options, 1, keypoints, undefined_descriptors),
      std::invalid_argument);

  // Adding with correct type should succeed.
  auto sift_descriptors_small =
      CreateRandomDescriptors(10, desc_dim, FeatureExtractorType::SIFT);
  EXPECT_NO_THROW(
      visual_index->Add(index_options, 1, keypoints, sift_descriptors_small));
  visual_index->Prepare();

  // Querying with mismatched type should throw.
  VisualIndex::QueryOptions query_options;
  std::vector<ImageScore> image_scores;
  EXPECT_THROW(
      visual_index->Query(query_options, undefined_descriptors, &image_scores),
      std::invalid_argument);

  // Querying with correct type should succeed.
  EXPECT_NO_THROW(visual_index->Query(
      query_options, sift_descriptors_small, &image_scores));
}

TEST(VisualIndex, Print) {
  VisualIndex::BuildOptions build_options;
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;
  build_options.num_visual_words = 10;

  auto descriptors =
      CreateRandomDescriptors(100, 128, FeatureExtractorType::SIFT);
  auto visual_index = VisualIndex::Create(128, 64);
  visual_index->Build(build_options, descriptors);

  VisualIndex::IndexOptions index_options;
  FeatureKeypoints keypoints(10);
  auto descriptors_small =
      CreateRandomDescriptors(10, 128, FeatureExtractorType::SIFT);
  visual_index->Add(index_options, 1, keypoints, descriptors_small);
  visual_index->Add(index_options, 2, keypoints, descriptors_small);

  std::ostringstream stream;
  stream << *visual_index;
  EXPECT_EQ(stream.str(),
            "VisualIndex(num_visual_words=10, num_images=2, desc_dim=128, "
            "embedding_dim=64, feature_type=SIFT)");
}

TEST(VisualIndex, CreateWithDescriptorDimension) {
  for (const auto& [desc_dim, embedding_dim] : {std::make_pair(32, 16),
                                                std::make_pair(128, 64),
                                                std::make_pair(256, 64)}) {
    const auto visual_index = VisualIndex::Create(desc_dim);
    EXPECT_EQ(visual_index->DescDim(), desc_dim);
    EXPECT_EQ(visual_index->EmbeddingDim(), embedding_dim);
  }
}

INSTANTIATE_TEST_SUITE_P(VisualIndexTests,
                         ParameterizedVisualIndexTests,
                         ::testing::Values(std::make_pair(128, 64),
                                           std::make_pair(32, 16),
                                           std::make_pair(256, 64)));

#ifdef COLMAP_DOWNLOAD_ENABLED

TEST(VisualIndex, Download) {
  const auto test_dir = CreateTestDir();
  const auto vocab_tree_path = test_dir / "server_vocab_tree.bin";
  OverwriteDownloadCacheDir(test_dir);

  auto descriptors =
      CreateRandomDescriptors(50, 32, FeatureExtractorType::SIFT);
  auto visual_index = VisualIndex::Create(32, 16);
  VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 5;
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  visual_index->Build(build_options, descriptors);
  visual_index->Write(vocab_tree_path);

  std::vector<char> vocab_tree_data;
  ReadBinaryBlob(vocab_tree_path, &vocab_tree_data);
  const std::string vocab_tree_uri =
      "file://" + std::filesystem::absolute(vocab_tree_path).string() +
      ";vocab_tree.bin;" +
      ComputeSHA256({vocab_tree_data.data(), vocab_tree_data.size()});
  auto downloaded_visual_index = VisualIndex::Read(vocab_tree_uri);
  EXPECT_EQ(downloaded_visual_index->NumVisualWords(),
            visual_index->NumVisualWords());
  EXPECT_EQ(downloaded_visual_index->FeatureType(),
            visual_index->FeatureType());
}

#endif

}  // namespace
}  // namespace retrieval
}  // namespace colmap
