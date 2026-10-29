// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/sensor/bitmap.h"
#ifndef __CUDACC__
#include "colmap/util/base_controller.h"
#include "colmap/util/threading.h"
#endif

#include <filesystem>
#include <memory>
#include <vector>

namespace colmap {
namespace mvs {

class ConsistencyGraph;
class PatchMatchCuda;
class Workspace;
struct SparseSourceTestFailure;

// This is a wrapper class around the actual PatchMatchCuda implementation. This
// class is necessary to hide Cuda code from any boost or Eigen code, since
// NVCC/MSVC cannot compile complex C++ code.
class PatchMatch {
 public:
  struct Problem {
    // Index of the reference image.
    int ref_image_idx = -1;

    // Indices of the source images.
    std::vector<int> src_image_idxs;

    // Input images for the photometric consistency term.
    std::vector<Image>* images = nullptr;

    // Optional source bitmap owners. Their actual payload is used by BOTH
    // dense and sparse source upload. The reference keeps its independent
    // bitmap. Callers must not mutate a retained payload through another alias
    // until all consumers and the source bank have finished. A lease must pin
    // the actual payload storage, not alias it from an unrelated lifetime
    // token.
    std::vector<std::shared_ptr<const Bitmap>>* source_bitmap_leases = nullptr;

    const Bitmap& InputBitmap(const int image_idx) const {
      const auto& image = images->at(image_idx);
      if (source_bitmap_leases != nullptr && image_idx != ref_image_idx &&
          image_idx >= 0 &&
          static_cast<size_t>(image_idx) < source_bitmap_leases->size()) {
        const auto& lease = source_bitmap_leases->at(image_idx);
        if (lease != nullptr && lease.use_count() > 0 && lease->IsGrey() &&
            lease->Width() == image.GetWidth() &&
            lease->Height() == image.GetHeight())
          return *lease;
      }
      return image.GetBitmap();
    }

    // Input depth maps for the geometric consistency term.
    std::vector<DepthMap>* depth_maps = nullptr;

    // Optional immutable source maps retained from Workspace leases. The
    // reference depth remains in depth_maps because initialization owns it.
    // As for source_bitmap_leases, the owner must retain the actual payload,
    // and no mutable alias may change it during upload or use.
    std::vector<std::shared_ptr<const DepthMap>>* source_depth_leases = nullptr;

    const DepthMap& InputDepthMap(const int image_idx) const {
      if (source_depth_leases != nullptr && image_idx != ref_image_idx &&
          image_idx >= 0 &&
          static_cast<size_t>(image_idx) < source_depth_leases->size()) {
        const auto& lease = source_depth_leases->at(image_idx);
        if (lease != nullptr && lease.use_count() > 0) return *lease;
      }
      return depth_maps->at(image_idx);
    }

    // Input normal maps for the geometric consistency term.
    std::vector<NormalMap>* normal_maps = nullptr;

    // Print the configuration to stdout.
    void Print() const;
  };

  PatchMatch(const PatchMatchOptions& options, const Problem& problem);
  ~PatchMatch();

  // Check the options and the problem for validity.
  void Check() const;

  // Run the patch match algorithm.
  void Run();

  // Get the computed values after running the algorithm.
  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  ConsistencyGraph GetConsistencyGraph() const;
  Mat<float> GetSelProbMap() const;

 private:
  const PatchMatchOptions options_;
  const Problem problem_;
  std::unique_ptr<PatchMatchCuda> patch_match_cuda_;
};

// This thread processes all problems in a workspace. A workspace has the
// following file structure, if the workspace format is "COLMAP":
//
//    images/*
//    sparse/{cameras.txt, images.txt, points3D.txt}
//    stereo/
//      depth_maps/*
//      normal_maps/*
//      consistency_graphs/*
//      patch-match.cfg
//
// The `patch-match.cfg` file specifies the images to be processed as:
//
//    image_name1.jpg
//    __all__
//    image_name2.jpg
//    __auto__, 20
//    image_name3.jpg
//    image_name1.jpg, image_name2.jpg
//
// Two consecutive lines specify the images used to compute one patch match
// problem. The first line specifies the reference image and the second line the
// source images. Image names are relative to the `images` directory. In this
// example, the first reference image uses all other images as source images,
// the second reference image uses the 20 most connected images as source
// images, and the third reference image uses the first and second as source
// images. Note that all specified images must be reconstructed in the COLMAP
// reconstruction provided in the `sparse` folder.

#ifndef __CUDACC__

class PatchMatchController : public BaseController {
 public:
  PatchMatchController(const PatchMatchOptions& options,
                       const std::filesystem::path& workspace_path,
                       const std::string& workspace_format,
                       const std::string& pmvs_option_name,
                       const std::filesystem::path& config_path = "");
  void Run();

 private:
  friend class PatchMatchControllerTestAccess;
  // Caller-owned only through Run; no CLI/environment/global injection.
  struct SparseTestHooks {
    const SparseSourceTestFailure* failure = nullptr;
    bool refuse_geometry_admission_once = false;
    size_t geometry_fallback_entries = 0;
    struct MemoryQuery {
      const char* phase;
      size_t free_bytes, total_bytes;
    };
    std::vector<MemoryQuery> memory_queries;
    std::vector<int> written_references;
  };
  SparseTestHooks* sparse_test_hooks_ = nullptr;

  struct PreparedProblem {
    bool valid = false;
    PatchMatchOptions options;
    PatchMatch::Problem problem;
    std::vector<Image> images;
    std::vector<std::shared_ptr<const Bitmap>> source_bitmap_leases;
    std::vector<DepthMap> depth_maps;
    std::vector<std::shared_ptr<const DepthMap>> source_depth_leases;
    std::vector<NormalMap> normal_maps;
    std::filesystem::path depth_path, normal_path, consistency_path;
    PreparedProblem() = default;
    PreparedProblem(const PreparedProblem&) = delete;
    PreparedProblem& operator=(const PreparedProblem&) = delete;
  };

  void ReadWorkspace();
  void ReadProblems();
  void ReadGpuIndices();
  bool PrepareProblem(const PatchMatchOptions& options,
                      size_t problem_idx,
                      PreparedProblem* prepared,
                      bool retain_sparse_sources = false);
  void ProcessProblem(const PatchMatchOptions& options, size_t problem_idx);
  void ProcessProblemPair(const PatchMatchOptions& options,
                          size_t first_problem_idx,
                          size_t second_problem_idx);
  void ProcessProblemBatch(const PatchMatchOptions& options,
                           std::vector<size_t> problem_indices);

  const PatchMatchOptions options_;
  const std::filesystem::path workspace_path_;
  const std::string workspace_format_;
  const std::string pmvs_option_name_;
  const std::filesystem::path config_path_;

  std::unique_ptr<ThreadPool> thread_pool_;
  std::unique_ptr<ThreadPool> io_thread_pool_;
  std::mutex workspace_mutex_;
  std::unique_ptr<Workspace> workspace_;
  std::vector<PatchMatch::Problem> problems_;
  std::vector<int> gpu_indices_;
  std::vector<std::pair<float, float>> depth_ranges_;
};

#endif

}  // namespace mvs
}  // namespace colmap
