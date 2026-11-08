// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/visual_index.h"

#include "colmap/feature/types.h"
#include "colmap/math/random_eigen.h"
#include "colmap/retrieval/inverted_index.h"
#include "colmap/retrieval/vote_and_verify.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <bitset>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
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

FeatureDescriptorsFloat CreateScalarDescriptors(
    const std::vector<float>& values) {
  FeatureDescriptorsFloatData descriptors =
      FeatureDescriptorsFloatData::Zero(values.size(), 128);
  for (size_t i = 0; i < values.size(); ++i) {
    descriptors(i, 0) = values[i];
  }
  return FeatureDescriptorsFloat(FeatureExtractorType::SIFT,
                                 std::move(descriptors));
}

void WriteExactVocabulary(const std::filesystem::path& path) {
  using Index = InvertedIndex<float, 128, 64>;
  // H/L/M come first; each remaining triple isolates one anchor feature.
  const std::vector<float> centers = {
      100, 200, 300, 1000, 1010, 1020, 2000, 2010, 2020, 3000, 3010, 3020};
  std::vector<float> samples;
  Index::WordIds word_ids(5 * centers.size(), 1);
  for (size_t word = 0; word < centers.size(); ++word) {
    for (int i = 0; i < 5; ++i) {
      word_ids(samples.size(), 0) = word;
      samples.push_back(centers[word]);
    }
  }
  const Index::DescType training = CreateScalarDescriptors(samples).data;
  const Index::DescType words = CreateScalarDescriptors(centers).data;
  faiss::IndexFlatL2 quantizer(128);
  // Borrow the quantizer, which outlives the IVF index. One flat list gives
  // exact word routing without depending on spectral-hash ordering.
  faiss::IndexIVFFlat vocabulary(&quantizer, 128, 1);
  vocabulary.train(training.rows(), training.data());
  vocabulary.add(words.rows(), words.data());
  const auto check_words = [&vocabulary](
                               const float value,
                               const std::vector<faiss::idx_t>& expected) {
    const Index::DescType descriptor = CreateScalarDescriptors({value}).data;
    std::vector<float> distances(expected.size());
    std::vector<faiss::idx_t> actual(expected.size());
    faiss::IVFSearchParameters options;
    options.nprobe = 64;
    vocabulary.search(1,
                      descriptor.data(),
                      expected.size(),
                      distances.data(),
                      actual.data(),
                      &options);
    ASSERT_EQ(actual, expected);
  };
  ASSERT_NO_FATAL_FAILURE(check_words(175, {1, 0, 2}));
  ASSERT_NO_FATAL_FAILURE(check_words(160, {1, 0}));
  ASSERT_NO_FATAL_FAILURE(check_words(240, {1, 2}));

  Index inverted;
  inverted.Initialize(centers.size());
  inverted.GenerateHammingEmbeddingProjection();
  inverted.ComputeHammingEmbedding(training, word_ids);
  const Index::DescType query = CreateScalarDescriptors({175}).data;
  const Index::DescType db_d = CreateScalarDescriptors({160}).data;
  const Index::DescType db_e = CreateScalarDescriptors({240}).data;
  std::bitset<64> query_bits, db_bits;
  for (const int word : {0, 1}) {
    inverted.ConvertToBinaryDescriptor(word, query, &query_bits);
    inverted.ConvertToBinaryDescriptor(word, db_d, &db_bits);
    ASSERT_EQ((query_bits ^ db_bits).count(), 0);
  }
  inverted.ConvertToBinaryDescriptor(1, db_e, &db_bits);
  ASSERT_GT((query_bits ^ db_bits).count(),
            HammingDistWeightFunctor<64>::kMaxHammingDistance);
  inverted.ConvertToBinaryDescriptor(2, query, &query_bits);
  inverted.ConvertToBinaryDescriptor(2, db_e, &db_bits);
  ASSERT_EQ((query_bits ^ db_bits).count(), 0);

  faiss::VectorIOWriter serialized;
  faiss::write_index(&vocabulary, &serialized);
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file);
  const int metadata[] = {
      2, 128, 64, static_cast<int>(FeatureExtractorType::SIFT)};
  file.write(reinterpret_cast<const char*>(metadata), sizeof(metadata));
  file.write(reinterpret_cast<const char*>(serialized.data.data()),
             serialized.data.size());
  inverted.Write(&file);
  file.close();
  ASSERT_TRUE(file);
}

TEST(VisualIndex, VerificationWithNoAssignedWords) {
  using Index = InvertedIndex<float, 128, 64>;
  const auto path = CreateTestDir() / "empty_ivf_list.bin";
  const Index::DescType centers = CreateScalarDescriptors({-100, 100}).data;
  const Index::DescType words = CreateScalarDescriptors({80, 120}).data;
  const Index::DescType query_descriptor = CreateScalarDescriptors({-100}).data;
  faiss::IndexFlatL2 quantizer(128);
  quantizer.add(centers.rows(), centers.data());
  faiss::IndexIVFFlat vocabulary(&quantizer, 128, 2);
  // Supply the trained coarse centroids directly; only the positive list
  // receives a visual word. Searching the negative list must return -1.
  vocabulary.is_trained = true;
  vocabulary.add(words.rows(), words.data());
  faiss::IVFSearchParameters search;
  search.nprobe = 1;
  float distance = 0;
  faiss::idx_t word = 0;
  vocabulary.search(1, query_descriptor.data(), 1, &distance, &word, &search);
  ASSERT_EQ(word, -1);

  Index inverted;
  inverted.Initialize(2);
  inverted.GenerateHammingEmbeddingProjection();
  Index::WordIds training_words(10, 1);
  training_words.topRows(5).setZero();
  training_words.bottomRows(5).setOnes();
  const Index::DescType training =
      CreateScalarDescriptors({80, 80, 80, 80, 80, 120, 120, 120, 120, 120})
          .data;
  inverted.ComputeHammingEmbedding(training, training_words);
  faiss::VectorIOWriter serialized;
  faiss::write_index(&vocabulary, &serialized);
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file);
  const int metadata[] = {
      2, 128, 64, static_cast<int>(FeatureExtractorType::SIFT)};
  file.write(reinterpret_cast<const char*>(metadata), sizeof(metadata));
  file.write(reinterpret_cast<const char*>(serialized.data.data()),
             serialized.data.size());
  inverted.Write(&file);
  file.close();
  ASSERT_TRUE(file);

  auto index = VisualIndex::Read(path);
  VisualIndex::IndexOptions indexing;
  indexing.num_threads = 1;
  indexing.num_checks = 1;
  index->Add(indexing,
             0,
             FeatureKeypoints{FeatureKeypoint(0, 0, 1, 0)},
             CreateScalarDescriptors({80}));
  index->Add(indexing,
             1,
             FeatureKeypoints{FeatureKeypoint(0, 0, 1, 0)},
             CreateScalarDescriptors({120}));
  index->Prepare();
  VisualIndex::QueryOptions options;
  options.num_threads = 1;
  options.num_checks = 1;
  options.num_neighbors = 1;
  options.num_images_after_verification = 1;
  // The populated list must be live after serialization and preparation.
  // Two distinct indexed words keep their IDF weights nonzero.
  ASSERT_EQ(index->NumImages(), 2);
  auto scoring_options = options;
  scoring_options.num_images_after_verification = 0;
  std::vector<ImageScore> positive_scores;
  index->Query(
      scoring_options, CreateScalarDescriptors({80}), &positive_scores);
  ASSERT_EQ(positive_scores.size(), 1);
  EXPECT_EQ(positive_scores[0].image_id, 0);
  EXPECT_GT(positive_scores[0].score, 0);

  std::vector<ImageScore> scores(1);
  index->Query(options,
               FeatureKeypoints{FeatureKeypoint(0, 0, 1, 0)},
               CreateScalarDescriptors({-100}),
               &scores);
  EXPECT_TRUE(scores.empty());
}

TEST(VisualIndex, MultiwordMatchesRetainMaximumScore) {
  const auto path = CreateTestDir() / "multiword.bin";
  ASSERT_NO_FATAL_FAILURE(WriteExactVocabulary(path));
  auto index = VisualIndex::Read(path);
  ASSERT_EQ(index->NumVisualWords(), 12);
  ASSERT_EQ(index->NumImages(), 0);
  FeatureKeypoints query_keypoints(5);
  const std::vector<Eigen::Vector2f> points = {
      {120, 100}, {1000, 180}, {180, 1100}, {1150, 950}, {650, 500}};
  std::vector<FeatureGeometryMatch> identity_matches;
  for (size_t i = 0; i < points.size(); ++i) {
    query_keypoints[i] =
        FeatureKeypoint(points[i].x(), points[i].y(), 1.0f, 0.0f);
    FeatureGeometry geometry;
    geometry.x = points[i].x();
    geometry.y = points[i].y();
    geometry.scale = 1;
    identity_matches.push_back({geometry, geometry});
  }
  const int expected = VoteAndVerify(VoteAndVerifyOptions(), identity_matches);
  ASSERT_EQ(expected, 5);
  FeatureKeypoints target_keypoints = query_keypoints;
  std::swap(target_keypoints[2], target_keypoints[3]);
  VisualIndex::IndexOptions indexing;
  indexing.num_threads = 1;
  indexing.num_neighbors = 2;
  // D at index2 matches words L/H; E at index3 matches L/M. The query
  // encounters L then H, so D must retain H rather than its first score L.
  index->Add(indexing,
             0,
             target_keypoints,
             CreateScalarDescriptors({1003, 2003, 160, 240, 3003}));
  indexing.num_neighbors = 1;
  // Eight images give document counts H=4, L=7, M=5 and hence
  // 0 < log(8/7)^2 < log(8/5)^2 < log(8/4)^2 < D's feature index2.
  for (int image = 1; image <= 7; ++image) {
    std::vector<float> values;
    if (image <= 3) {
      values.push_back(100);
    }
    if (image <= 4) {
      values.push_back(300);
    }
    if (image <= 6) {
      values.push_back(200);
    }
    if (image == 7) {
      values.push_back(1003);
    }
    index->Add(indexing,
               image,
               FeatureKeypoints(values.size(), FeatureKeypoint(0, 0, 1, 0)),
               CreateScalarDescriptors(values));
  }
  index->Prepare();
  ASSERT_EQ(index->NumImages(), 8);
  VisualIndex::QueryOptions query;
  query.num_threads = 1;
  query.num_neighbors = 3;
  query.image_id_filter = [](const int image) { return image == 0; };
  // The two ambiguous query features have equal degree; index3 is chosen
  // first. Correct weighting selects D, giving the identity geometry above.
  const auto descriptors =
      CreateScalarDescriptors({1003, 2003, 175, 175, 3003});
  std::vector<ImageScore> plain, verified;
  index->Query(query, query_keypoints, descriptors, &plain);
  query.num_images_after_verification = 1;
  index->Query(query, query_keypoints, descriptors, &verified);
  ASSERT_EQ(plain.size(), 1);
  ASSERT_EQ(verified.size(), 1);
  EXPECT_EQ(plain[0].image_id, 0);
  EXPECT_EQ(verified[0].image_id, 0);
  EXPECT_NEAR(verified[0].score - plain[0].score, expected, 1e-5);

  // FAISS pads the extra neighbor slots with invalid word IDs. They must not
  // change the weighted spatial matches or the final verified score.
  query.num_neighbors = 12;
  std::vector<ImageScore> all_words;
  index->Query(query, query_keypoints, descriptors, &all_words);
  query.num_neighbors = 16;
  std::vector<ImageScore> padded_words;
  index->Query(query, query_keypoints, descriptors, &padded_words);
  ASSERT_EQ(all_words.size(), 1);
  ASSERT_EQ(padded_words.size(), 1);
  EXPECT_EQ(all_words[0].image_id, padded_words[0].image_id);
  EXPECT_FLOAT_EQ(all_words[0].score, padded_words[0].score);

  // Empty descriptors and an empty candidate set remain valid with spatial
  // verification enabled.
  std::vector<ImageScore> empty(1);
  index->Query(query, FeatureKeypoints{}, CreateScalarDescriptors({}), &empty);
  EXPECT_TRUE(empty.empty());
  query.image_id_filter = [](const int) { return false; };
  index->Query(query, query_keypoints, descriptors, &empty);
  EXPECT_TRUE(empty.empty());

  // Identical database images must remain a score tie after verification.
  // Compare by image ID rather than depending on an order among equal scores.
  for (const int image_id : {8, 9}) {
    index->Add(indexing,
               image_id,
               target_keypoints,
               CreateScalarDescriptors({1003, 2003, 160, 240, 3003}));
  }
  index->Prepare();
  query.num_images_after_verification = 2;
  query.image_id_filter = [](const int image) {
    return image == 8 || image == 9;
  };
  query.num_neighbors = 12;
  std::vector<ImageScore> tied, padded_tied;
  index->Query(query, query_keypoints, descriptors, &tied);
  query.num_neighbors = 16;
  index->Query(query, query_keypoints, descriptors, &padded_tied);
  ASSERT_EQ(tied.size(), 2);
  ASSERT_EQ(padded_tied.size(), 2);
  EXPECT_NE(tied[0].image_id, tied[1].image_id);
  EXPECT_FLOAT_EQ(tied[0].score, tied[1].score);
  for (const auto& score : tied) {
    const auto match = std::find_if(
        padded_tied.begin(), padded_tied.end(), [&score](const auto& other) {
          return other.image_id == score.image_id;
        });
    ASSERT_NE(match, padded_tied.end());
    EXPECT_FLOAT_EQ(score.score, match->score);
  }
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

  auto train_descriptors =
      CreateRandomDescriptors(1000, desc_dim, FeatureExtractorType::SIFT);
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
  const auto descriptors = CreateRandomDescriptors(
      kNumFeatures, desc_dim, FeatureExtractorType::SIFT);

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
