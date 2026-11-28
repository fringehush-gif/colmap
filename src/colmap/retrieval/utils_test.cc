// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/utils.h"

#include "colmap/retrieval/inverted_file_entry.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

TEST(ScoredFeatureMatchGreater, BreaksTiesByFeatureIndex) {
  using Entry = InvertedFileEntry<16>;
  using Match = internal::ScoredFeatureMatch<Entry>;
  // The higher feature index has the lower address in this array.
  std::array<Entry, 2> features;
  features[0].feature_idx = 1;
  features[1].feature_idx = 0;

  const Match stronger{2.0f, {&features[1], &features[1]}};
  const Match higher_query{1.0f, {&features[0], &features[1]}};
  const Match higher_db{1.0f, {&features[1], &features[0]}};
  const Match lower{1.0f, {&features[1], &features[1]}};
  std::array matches{lower, higher_db, higher_query, stronger};
  std::sort(matches.begin(),
            matches.end(),
            internal::ScoredFeatureMatchGreater<Entry>);
  EXPECT_EQ(matches, (std::array{stronger, higher_query, higher_db, lower}));
}

TEST(HammingDistWeightFunctor, ZeroDistance) {
  HammingDistWeightFunctor<64> functor;
  EXPECT_EQ(functor(0), 1.0f);
}

TEST(HammingDistWeightFunctor, SmallDistance) {
  HammingDistWeightFunctor<64> functor;
  constexpr float kSigma = 16.0f;
  constexpr float kSigmaSquared = kSigma * kSigma;
  const size_t kDist = 5;
  const float expected =
      std::exp(-static_cast<float>(kDist * kDist) / kSigmaSquared);
  EXPECT_EQ(functor(kDist), expected);
}

TEST(HammingDistWeightFunctor, MaxDistance) {
  HammingDistWeightFunctor<64> functor;
  // At max hamming distance, should have non-zero weight
  EXPECT_GT(functor(functor.kMaxHammingDistance), 0.0f);
  EXPECT_LT(functor(functor.kMaxHammingDistance), 1.0f);
  // Beyond max hamming distance, weight should be 0
  EXPECT_EQ(functor(functor.kMaxHammingDistance + 1), 0.0f);
}

TEST(HammingDistWeightFunctor, Monotonicity) {
  HammingDistWeightFunctor<128, 20> functor;
  // Weight should decrease as distance increases (up to max distance)
  for (size_t i = 0; i <= functor.kMaxHammingDistance; ++i) {
    EXPECT_GT(functor(i), functor(i + 1)) << i;
  }
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap
