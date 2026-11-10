// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/workspace.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/cuda.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#ifdef __linux__
#include <csignal>

#include <sys/resource.h>
#include <unistd.h>
#endif

namespace colmap {
namespace mvs {

class PatchMatchControllerTestAccess {
 public:
  using ConfigProblem = std::pair<std::string, std::vector<std::string>>;
  static std::vector<ConfigProblem> ReadConfig(PatchMatchController& c) {
    c.ReadWorkspace();
    c.ReadProblems();
    const auto& model = c.workspace_->GetModel();
    std::vector<ConfigProblem> result;
    for (const auto& problem : c.problems_) {
      ConfigProblem entry{model.GetImageName(problem.ref_image_idx), {}};
      for (const int source : problem.src_image_idxs) {
        entry.second.push_back(model.GetImageName(source));
      }
      result.push_back(std::move(entry));
    }
    return result;
  }
  static void Publish(
      const std::filesystem::path& depth,
      const std::filesystem::path& normal,
      const std::filesystem::path& graph,
      const std::function<void(const std::filesystem::path&)>& write_output) {
    PatchMatchController::PublishOutput(
        depth, normal, graph, false, write_output);
  }
  static std::vector<int> SourceIndices(PatchMatchController& c,
                                        const std::vector<std::string>& names) {
    c.ReadWorkspace();
    const auto& model = c.workspace_->GetModel();
    return {model.GetImageIdx(names[1]), model.GetImageIdx(names[2])};
  }
};

namespace {

class ControllerFixture {
 public:
  ControllerFixture() {
    SyntheticDatasetOptions dataset;
    dataset.num_rigs = 1;
    dataset.num_cameras_per_rig = 1;
    dataset.num_frames_per_rig = 5;
    dataset.camera_width = 96;
    dataset.camera_height = 64;
    SynthesizeDataset(dataset, &model_);
    for (int i = 1; i <= 5; ++i) names_.push_back(model_.Image(i).Name());
    root_ = CreateTestDir() / ("workspace_" + std::to_string(next_++));
    for (const auto* dir : {"images",
                            "sparse",
                            "stereo/depth_maps",
                            "stereo/normal_maps",
                            "stereo/consistency_graphs"}) {
      std::filesystem::create_directories(root_ / dir);
    }
    for (const auto& [id, camera] : model_.Cameras()) {
      model_.Camera(id).params = {80.0, 48.0, 32.0, 0.0};
    }
    for (const auto& [id, frame] : model_.Frames()) {
      Rigid3d pose;
      pose.translation() = Eigen::Vector3d(-0.1 * id, 0.0, 0.0);
      model_.Frame(id).SetRigFromWorld(pose);
    }
    model_.Write(root_ / "sparse");
    for (const auto& [id, image] : model_.Images()) {
      const double camera_x = 0.1 * image.FrameId();
      Bitmap bitmap(96, 64, false);
      for (int row = 0; row < 64; ++row) {
        for (int col = 0; col < 96; ++col) {
          const double x = camera_x + 4.0 * (col + 0.5 - 48.0) / 80.0;
          const double y = 4.0 * (row + 0.5 - 32.0) / 80.0;
          const double value = 128.0 + 42.0 * std::sin(8.0 * x + 3.0 * y) +
                               35.0 * std::cos(5.0 * x - 11.0 * y) +
                               24.0 * std::sin(17.0 * x + 13.0 * y);
          bitmap.SetPixel(col,
                          row,
                          BitmapColor<uint8_t>(static_cast<uint8_t>(
                              std::clamp(std::round(value), 0.0, 255.0))));
        }
      }
      bitmap.Write(root_ / "images" / image.Name());
    }
    WriteConfig(names_[0] + "\n" + names_[1] + "," + names_[2] + "\n");
  }
  void WriteConfig(const std::string& text) const {
    std::ofstream config(root_ / "stereo/patch-match.cfg");
    config << text;
    config.close();
  }
  PatchMatchOptions Options(bool graph = false) const {
    PatchMatchOptions o;
    o.gpu_index = "0";
    o.num_threads = 2;
    o.num_iterations = 5;
    o.num_samples = 15;
    o.geom_consistency = false;
    o.filter = graph;
    o.filter_min_num_consistent = 1;
    o.filter_min_triangulation_angle = 0.0;
    o.filter_min_ncc = -1.0;
    o.depth_min = 1;
    o.depth_max = 10;
    o.write_consistency_graph = graph;
    return o;
  }
  std::filesystem::path Output(const char* family) const {
    return root_ / "stereo" / family / (names_[0] + ".photometric.bin");
  }
  const std::filesystem::path& Root() const { return root_; }
  const std::vector<std::string>& Names() const { return names_; }

 private:
  inline static size_t next_ = 0;
  Reconstruction model_;
  std::filesystem::path root_;
  std::vector<std::string> names_;
};

TEST(PatchMatchController, RejectsDuplicateConfiguredReferences) {
  ControllerFixture f;
  const auto& n = f.Names();
  for (const bool different : {false, true}) {
    f.WriteConfig(n[0] + "\n" + n[1] + "\n" + n[0] + "\n" +
                  n[different ? 2 : 1] + "\n");
    PatchMatchController c(f.Options(), f.Root(), "COLMAP", "");
    try {
      PatchMatchControllerTestAccess::ReadConfig(c);
      ADD_FAILURE() << "Expected duplicate reference rejection";
    } catch (const std::exception& e) {
      EXPECT_NE(std::string(e.what()).find("Duplicate reference image"),
                std::string::npos);
      EXPECT_NE(std::string(e.what()).find(n[0]), std::string::npos);
    }
  }
}

TEST(PatchMatchController, PreservesUniqueReferenceAndSourceOrder) {
  ControllerFixture f;
  const auto& n = f.Names();
  f.WriteConfig(n[2] + "\n" + n[4] + "," + n[0] + "\n" + n[0] + "\n" + n[3] +
                "," + n[2] + "\n");
  PatchMatchController c(f.Options(), f.Root(), "COLMAP", "");
  const std::vector<PatchMatchControllerTestAccess::ConfigProblem> expected = {
      {n[2], {n[4], n[0]}}, {n[0], {n[3], n[2]}}};
  EXPECT_EQ(PatchMatchControllerTestAccess::ReadConfig(c), expected);
}

TEST(PatchMatchController, CleanupPreservesOriginalError) {
  const auto root = CreateTestDir();
  const auto normal = root / "normal.bin";
  std::filesystem::path stage;
  try {
    PatchMatchControllerTestAccess::Publish(
        root / "depth.bin", normal, root / "graph.bin", [&](const auto& path) {
          stage = path.parent_path();
          std::filesystem::create_directory(path);
          std::ofstream(path / "keep").put('x');
          throw std::runtime_error("original output failure");
        });
    FAIL() << "Expected the original output error";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "original output failure");
  }
  EXPECT_FALSE(ExistsFile(normal));
  EXPECT_TRUE(ExistsFile(stage / "normal.bin" / "keep"));
}

#if defined(__linux__) && defined(COLMAP_CUDA_ENABLED)
TEST(PatchMatchController, OrdinaryConsumerWriteFailureRetries) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device";
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        ControllerFixture f;
        NormalMap(96, 64).Write(f.Output("normal_maps"));
        struct rlimit previous;
        if (getrlimit(RLIMIT_FSIZE, &previous) != 0) _exit(2);
        auto limit = previous;
        limit.rlim_cur = 0;
        if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR ||
            setrlimit(RLIMIT_FSIZE, &limit) != 0)
          _exit(3);
        bool failed = false;
        try {
          PatchMatchController c(f.Options(), f.Root(), "COLMAP", "");
          c.Run();
        } catch (const std::exception& error) {
          failed =
              std::string(error.what()).find(f.Output("depth_maps").string()) !=
              std::string::npos;
        }
        if (setrlimit(RLIMIT_FSIZE, &previous) != 0) _exit(4);
        if (!failed || ExistsFile(f.Output("normal_maps"))) _exit(5);
        try {
          PatchMatchController c(f.Options(), f.Root(), "COLMAP", "");
          c.Run();
        } catch (...) {
          _exit(6);
        }
        if (!ExistsFile(f.Output("depth_maps")) ||
            !ExistsFile(f.Output("normal_maps")))
          _exit(7);
        DepthMap depth;
        NormalMap normal;
        depth.Read(f.Output("depth_maps"));
        normal.Read(f.Output("normal_maps"));
        if (depth.GetWidth() != 96 || depth.GetHeight() != 64 ||
            depth.GetDepth() != 1 || normal.GetWidth() != 96 ||
            normal.GetHeight() != 64 || normal.GetDepth() != 3)
          _exit(8);
        _exit(0);
      },
      testing::ExitedWithCode(0),
      "");
}

TEST(PatchMatchController, OrdinaryConsumerGraphOffThenOn) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device";
  ControllerFixture f;
  PatchMatchController index_reader(f.Options(), f.Root(), "COLMAP", "");
  const auto source_indices =
      PatchMatchControllerTestAccess::SourceIndices(index_reader, f.Names());
  NormalMap(96, 64).Write(f.Output("normal_maps"));
  ConsistencyGraph(96, 64, {}).Write(f.Output("consistency_graphs"));
  {
    PatchMatchController c(f.Options(false), f.Root(), "COLMAP", "");
    c.Run();
  }
  EXPECT_FALSE(ExistsFile(f.Output("consistency_graphs")));
  {
    PatchMatchController c(f.Options(true), f.Root(), "COLMAP", "");
    c.Run();
  }
  EXPECT_TRUE(ExistsFile(f.Output("consistency_graphs")));
  DepthMap depth;
  NormalMap output_normal;
  ConsistencyGraph graph;
  depth.Read(f.Output("depth_maps"));
  output_normal.Read(f.Output("normal_maps"));
  graph.Read(f.Output("consistency_graphs"));
  std::ifstream graph_file(f.Output("consistency_graphs"), std::ios::binary);
  size_t width = 0, height = 0, channels = 0;
  char a = 0, b = 0, c = 0;
  ASSERT_TRUE(graph_file >> width >> a >> height >> b >> channels >> c);
  ASSERT_EQ(width, 96);
  ASSERT_EQ(height, 64);
  ASSERT_EQ(channels, 1);
  ASSERT_EQ(a, '&');
  ASSERT_EQ(b, '&');
  ASSERT_EQ(c, '&');
  EXPECT_EQ(depth.GetWidth(), 96);
  EXPECT_EQ(depth.GetHeight(), 64);
  EXPECT_EQ(depth.GetDepth(), 1);
  EXPECT_EQ(output_normal.GetWidth(), 96);
  EXPECT_EQ(output_normal.GetHeight(), 64);
  EXPECT_EQ(output_normal.GetDepth(), 3);
  size_t nonempty = 0;
  for (int row = 0; row < 64; ++row) {
    for (int col = 0; col < 96; ++col) {
      int count = 0;
      const int* indices = nullptr;
      graph.GetImageIdxs(row, col, &count, &indices);
      if (count > 0) {
        ++nonempty;
        ASSERT_NE(indices, nullptr);
        for (int i = 0; i < count; ++i) {
          EXPECT_TRUE(indices[i] == source_indices[0] ||
                      indices[i] == source_indices[1]);
        }
      }
    }
  }
  EXPECT_GT(nonempty, 0);
  const auto normal = f.Output("normal_maps");
  const auto before = std::filesystem::last_write_time(normal);
  {
    PatchMatchController c(f.Options(true), f.Root(), "COLMAP", "");
    c.Run();
  }
  EXPECT_EQ(std::filesystem::last_write_time(normal), before);
}

TEST(PatchMatchController, OrdinaryConsumerRejectsDanglingGraph) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device";
  ControllerFixture f;
  DepthMap(96, 64, 1, 10).Write(f.Output("depth_maps"));
  const auto size = std::filesystem::file_size(f.Output("depth_maps"));
  const auto graph = f.Output("consistency_graphs");
  std::filesystem::create_symlink(graph.parent_path() / "missing.bin", graph);
  PatchMatchController c(f.Options(false), f.Root(), "COLMAP", "");
  try {
    c.Run();
    FAIL() << "Expected dangling graph rejection";
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what())
                  .find("Refusing to replace non-regular output"),
              std::string::npos);
    EXPECT_NE(std::string(error.what()).find(graph.string()),
              std::string::npos);
  }
  EXPECT_TRUE(std::filesystem::is_symlink(graph));
  EXPECT_EQ(std::filesystem::file_size(f.Output("depth_maps")), size);
  EXPECT_FALSE(ExistsFile(f.Output("normal_maps")));
}
#endif

}  // namespace
}  // namespace mvs
}  // namespace colmap
