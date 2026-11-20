// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"

#include "colmap/util/testing.h"

#include <exception>
#include <filesystem>

#ifdef __linux__
#include <csignal>

#include <sys/resource.h>
#include <unistd.h>
#endif

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

#ifdef __linux__
TEST(ConsistencyGraph, WritePreservesExistingHardLink) {
  const auto dir = CreateTestDir();
  const auto path = dir / "graph.bin";
  const auto link = dir / "alias.bin";
  ConsistencyGraph(2, 1, {}).Write(path);
  std::filesystem::create_hard_link(path, link);
  ConsistencyGraph(2, 1, {1, 0, 1, 7}).Write(path);
  EXPECT_TRUE(std::filesystem::equivalent(path, link));
  ConsistencyGraph read;
  read.Read(link);
  int count;
  const int* ids;
  read.GetImageIdxs(0, 1, &count, &ids);
  ASSERT_EQ(count, 1);
  EXPECT_EQ(ids[0], 7);
}

TEST(ConsistencyGraph, WriteReportsIncompleteHeaderAndPayload) {
  const auto path = CreateTestDir() / "consistency_graph.bin";
  for (const rlim_t limit : {0, 16}) {
    SCOPED_TRACE(limit);
    EXPECT_EXIT(
        {
          struct rlimit file_limit;
          file_limit.rlim_cur = limit;
          file_limit.rlim_max = limit;
          if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR ||
              setrlimit(RLIMIT_FSIZE, &file_limit) != 0)
            _exit(2);
          try {
            const auto graph =
                limit == 0 ? ConsistencyGraph(2, 1, {})
                           : ConsistencyGraph(2, 1, {0, 0, 3, 5, 7, 33});
            graph.Write(path);
          } catch (const std::exception&) {
            _exit(0);
          }
          _exit(1);
        },
        testing::ExitedWithCode(0),
        "");
  }
}
#endif

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
