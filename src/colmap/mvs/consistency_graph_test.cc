// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"

#include "colmap/util/testing.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(ConsistencyGraph, Empty) {
  const std::vector<int> data;
  ConsistencyGraph consistency_graph(2, 2, data);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      int num_images;
      const int* image_idxs;
      consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
      EXPECT_EQ(num_images, 0);
      EXPECT_TRUE(image_idxs == nullptr);
    }
  }
  EXPECT_EQ(consistency_graph.GetNumBytes(), 16);
}

TEST(ConsistencyGraph, Partial) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33};
  ConsistencyGraph consistency_graph(2, 1, data);
  int num_images;
  const int* image_idxs;
  consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 3);
  EXPECT_EQ(image_idxs[0], 5);
  EXPECT_EQ(image_idxs[1], 7);
  EXPECT_EQ(image_idxs[2], 33);
  consistency_graph.GetImageIdxs(0, 1, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 0);
  EXPECT_TRUE(image_idxs == nullptr);
  EXPECT_EQ(consistency_graph.GetNumBytes(), 32);
}

TEST(ConsistencyGraph, Zero) {
  const std::vector<int> data = {0, 0, 0};
  ConsistencyGraph consistency_graph(2, 1, data);
  int num_images;
  const int* image_idxs;
  consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 0);
  EXPECT_TRUE(image_idxs == nullptr);
  consistency_graph.GetImageIdxs(0, 1, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 0);
  EXPECT_TRUE(image_idxs == nullptr);
  EXPECT_EQ(consistency_graph.GetNumBytes(), 20);
}

TEST(ConsistencyGraph, LvaluePayloadRemainsIndependent) {
  std::vector<int> data = {2, 1, 2, 5, 33};
  ConsistencyGraph graph(3, 2, data);
  int num_images;
  const int* image_idxs;
  graph.GetImageIdxs(1, 2, &num_images, &image_idxs);
  ASSERT_EQ(num_images, 2);
  EXPECT_NE(image_idxs, data.data() + 3);
  data[3] = 99;
  data.clear();
  data.shrink_to_fit();
  EXPECT_EQ(image_idxs[0], 5);
  EXPECT_EQ(image_idxs[1], 33);
}

TEST(ConsistencyGraph, ConsumesRvaluePayload) {
  std::vector<int> data = {2, 1, 2, 5, 33};
  const int* original_indices = data.data() + 3;
  ConsistencyGraph graph(3, 2, std::move(data));
  // A moved-from vector is valid but its size/capacity are unspecified.
  int num_images;
  const int* image_idxs;
  graph.GetImageIdxs(1, 2, &num_images, &image_idxs);
  ASSERT_EQ(num_images, 2);
  EXPECT_EQ(image_idxs, original_indices);
  EXPECT_EQ(image_idxs[0], 5);
  EXPECT_EQ(image_idxs[1], 33);
  EXPECT_EQ(graph.GetNumBytes(), 11 * sizeof(int));
}

TEST(ConsistencyGraph, ConsumesReturnedPayload) {
  // The actual PatchMatch callers construct directly from a returned vector.
  // This probes that expression's ownership, without emulating GPU readback.
  const int* original_indices = nullptr;
  const auto make_payload = [&original_indices]() {
    std::vector<int> data = {2, 1, 2, 5, 33};
    original_indices = data.data() + 3;
    return data;
  };
  ConsistencyGraph graph(3, 2, make_payload());
  int num_images;
  const int* image_idxs;
  graph.GetImageIdxs(1, 2, &num_images, &image_idxs);
  ASSERT_EQ(num_images, 2);
  EXPECT_EQ(image_idxs, original_indices);
  EXPECT_EQ(image_idxs[0], 5);
  EXPECT_EQ(image_idxs[1], 33);
}

TEST(ConsistencyGraph, EmptyRvaluePayload) {
  ConsistencyGraph graph(3, 2, std::vector<int>{});
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 3; ++col) {
      int num_images;
      const int* image_idxs;
      graph.GetImageIdxs(row, col, &num_images, &image_idxs);
      EXPECT_EQ(num_images, 0);
      EXPECT_EQ(image_idxs, nullptr);
    }
  }
  EXPECT_EQ(graph.GetNumBytes(), 6 * sizeof(int));
}

TEST(ConsistencyGraph, RvalueDuplicateAndZeroRecords) {
  // Preserve the current last-positive-record behavior. A zero record does
  // not erase an earlier positive record at the same coordinate.
  ConsistencyGraph graph(
      3, 2, std::vector<int>{2, 1, 1, 7, 2, 1, 2, 11, 13, 2, 1, 0, 1, 0, 0});
  int num_images;
  const int* image_idxs;
  graph.GetImageIdxs(1, 2, &num_images, &image_idxs);
  ASSERT_EQ(num_images, 2);
  EXPECT_EQ(image_idxs[0], 11);
  EXPECT_EQ(image_idxs[1], 13);
  graph.GetImageIdxs(0, 1, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 0);
  EXPECT_EQ(image_idxs, nullptr);
}

TEST(ConsistencyGraph, RejectsInvalidMetadataForBothOwnershipPaths) {
  // Only existing checked rejects belong here. Complete record-size and
  // reader-format validation are separate changes, not supplied by moving.
  const std::vector<std::vector<int>> invalid_payloads = {
      {0, 0}, {0, 0, -1}, {-1, 0, 0}, {3, 0, 0}, {0, -1, 0}, {0, 2, 0}};
  for (const auto& invalid : invalid_payloads) {
    SCOPED_TRACE(::testing::PrintToString(invalid));
    std::vector<int> lvalue(invalid.begin(), invalid.end());
    EXPECT_THROW(ConsistencyGraph(3, 2, lvalue), std::exception);
    EXPECT_EQ(lvalue, invalid);
    std::vector<int> rvalue = invalid;
    EXPECT_THROW(ConsistencyGraph(3, 2, std::move(rvalue)), std::exception);
    // The rvalue overload permits consumption before a validation error.
  }
}

TEST(ConsistencyGraph, RvalueNonsquareWriteReadRoundtrip) {
  ConsistencyGraph original(3, 2, std::vector<int>{2, 1, 2, 5, 33});
  const auto test_dir = CreateTestDir();
  const auto path = test_dir / "moved_consistency_graph.bin";
  original.Write(path);

  // Check dimensions without using the graph reader as their sole oracle.
  std::ifstream file(path, std::ios::binary);
  std::string header;
  for (int i = 0; i < 6; ++i) {
    header.push_back(static_cast<char>(file.get()));
  }
  EXPECT_EQ(header, "3&2&1&");
  file.close();

  ConsistencyGraph loaded;
  loaded.Read(path);
  EXPECT_EQ(loaded.GetNumBytes(), original.GetNumBytes());
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 3; ++col) {
      int num_images;
      const int* image_idxs;
      loaded.GetImageIdxs(row, col, &num_images, &image_idxs);
      if (row == 1 && col == 2) {
        ASSERT_EQ(num_images, 2);
        EXPECT_EQ(image_idxs[0], 5);
        EXPECT_EQ(image_idxs[1], 33);
      } else {
        EXPECT_EQ(num_images, 0);
        EXPECT_EQ(image_idxs, nullptr);
      }
    }
  }
}

TEST(ConsistencyGraph, Full) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33, 0, 1, 1, 100};
  ConsistencyGraph consistency_graph(1, 2, data);
  int num_images;
  const int* image_idxs;
  consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 3);
  EXPECT_EQ(image_idxs[0], 5);
  EXPECT_EQ(image_idxs[1], 7);
  EXPECT_EQ(image_idxs[2], 33);
  consistency_graph.GetImageIdxs(1, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 1);
  EXPECT_EQ(image_idxs[0], 100);
  EXPECT_EQ(consistency_graph.GetNumBytes(), 48);
}

TEST(ConsistencyGraph, DefaultConstructor) {
  ConsistencyGraph consistency_graph;
  EXPECT_EQ(consistency_graph.GetNumBytes(), 0);
}

TEST(ConsistencyGraph, WriteReadRoundtrip) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33, 0, 1, 1, 100};
  ConsistencyGraph original(1, 2, data);

  const auto test_dir = CreateTestDir();
  const auto path = test_dir / "consistency_graph.bin";
  original.Write(path);

  ConsistencyGraph loaded;
  loaded.Read(path);

  EXPECT_EQ(loaded.GetNumBytes(), original.GetNumBytes());

  // Verify data is preserved
  int num_images;
  const int* image_idxs;
  loaded.GetImageIdxs(0, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 3);
  EXPECT_EQ(image_idxs[0], 5);
  EXPECT_EQ(image_idxs[1], 7);
  EXPECT_EQ(image_idxs[2], 33);
  loaded.GetImageIdxs(1, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 1);
  EXPECT_EQ(image_idxs[0], 100);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
