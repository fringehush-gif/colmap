// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"

#include "colmap/math/math.h"
#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/patch_match_io.h"
#include "colmap/mvs/patch_match_source_array.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/cuda.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>

#define PrintOption(option) LOG(INFO) << #option ": " << option << std::endl

namespace colmap {
namespace mvs {

#ifndef COLMAP_HIP_ENABLED
namespace {
struct PairStreams {
  cudaStream_t first = nullptr, second = nullptr;
  PairStreams() {
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking));
    const cudaError_t second_status =
        cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking);
    if (second_status != cudaSuccess) {
      (void)cudaStreamDestroy(first);
      first = nullptr;
      CUDA_SAFE_CALL(second_status);
    }
  }
  ~PairStreams() {
    if (second) (void)cudaStreamDestroy(second);
    if (first) (void)cudaStreamDestroy(first);
  }
};

bool DefaultReferencePairSettings(const PatchMatchOptions& o) {
  const PatchMatchOptions d;
  return o.depth_min == d.depth_min && o.depth_max == d.depth_max &&
         o.sigma_spatial == d.sigma_spatial && o.sigma_color == d.sigma_color &&
         o.ncc_sigma == d.ncc_sigma &&
         o.min_triangulation_angle == d.min_triangulation_angle &&
         o.incident_angle_sigma == d.incident_angle_sigma &&
         o.geom_consistency_regularizer == d.geom_consistency_regularizer &&
         o.geom_consistency_max_cost == d.geom_consistency_max_cost &&
         o.filter_min_ncc == d.filter_min_ncc &&
         o.filter_min_triangulation_angle == d.filter_min_triangulation_angle &&
         o.filter_geom_consistency_max_cost ==
             d.filter_geom_consistency_max_cost &&
         o.cache_size == d.cache_size &&
         o.direct_source_upload == d.direct_source_upload &&
         o.reference_only_normal_materialization ==
             d.reference_only_normal_materialization &&
         o.window_radius == d.window_radius && o.window_step == d.window_step &&
         o.num_samples == d.num_samples &&
         o.num_iterations == d.num_iterations &&
         o.filter_min_num_consistent == d.filter_min_num_consistent &&
         o.allow_missing_files == d.allow_missing_files &&
         o.write_consistency_graph == d.write_consistency_graph &&
         !o.checkpoint_hmm && o.geom_consistency == o.filter;
}
}  // namespace
#endif

PatchMatch::PatchMatch(const PatchMatchOptions& options, const Problem& problem)
    : options_(options), problem_(problem) {}

PatchMatch::~PatchMatch() {}

void PatchMatch::Problem::Print() const {
  LOG_HEADING2("PatchMatch::Problem");
  LOG(INFO) << "ref_image_idx: " << ref_image_idx;
  THROW_CHECK(!src_image_idxs.empty());
  std::ostringstream src_image_idxs_stream;
  for (size_t i = 0; i < src_image_idxs.size() - 1; ++i) {
    src_image_idxs_stream << src_image_idxs[i] << " ";
  }
  src_image_idxs_stream << src_image_idxs.back();
  LOG(INFO) << "src_image_idxs: " << src_image_idxs_stream.str();
}

void PatchMatch::Check() const {
  THROW_CHECK(options_.Check());

  THROW_CHECK(!options_.gpu_index.empty());
  const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
  THROW_CHECK_EQ(gpu_indices.size(), 1);
  THROW_CHECK_GE(gpu_indices[0], -1);

  THROW_CHECK_NOTNULL(problem_.images);
  if (options_.geom_consistency) {
    THROW_CHECK_NOTNULL(problem_.depth_maps);
    THROW_CHECK_NOTNULL(problem_.normal_maps);
    THROW_CHECK_EQ(problem_.depth_maps->size(), problem_.images->size());
    THROW_CHECK_EQ(problem_.normal_maps->size(), problem_.images->size());
  }

  THROW_CHECK_GT(problem_.src_image_idxs.size(), 0);

  // Check that there are no duplicate images and that the reference image
  // is not defined as a source image.
  std::set<int> unique_image_idxs(problem_.src_image_idxs.begin(),
                                  problem_.src_image_idxs.end());
  unique_image_idxs.insert(problem_.ref_image_idx);
  THROW_CHECK_EQ(problem_.src_image_idxs.size() + 1, unique_image_idxs.size());

  // Check that input data is well-formed.
  for (const int image_idx : unique_image_idxs) {
    THROW_CHECK_GE(image_idx, 0) << image_idx;
    THROW_CHECK_LT(image_idx, problem_.images->size()) << image_idx;

    const Image& image = problem_.images->at(image_idx);
    THROW_CHECK_GT(image.GetBitmap().Width(), 0) << image_idx;
    THROW_CHECK_GT(image.GetBitmap().Height(), 0) << image_idx;
    THROW_CHECK(image.GetBitmap().IsGrey()) << image_idx;
    THROW_CHECK_EQ(image.GetWidth(), image.GetBitmap().Width()) << image_idx;
    THROW_CHECK_EQ(image.GetHeight(), image.GetBitmap().Height()) << image_idx;

    // Make sure, the calibration matrix only contains fx, fy, cx, cy.
    THROW_CHECK_LT(std::abs(image.GetK()[1] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[3] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[6] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[7] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_idx;

    if (options_.geom_consistency) {
      THROW_CHECK_LT(image_idx, problem_.depth_maps->size()) << image_idx;
      const DepthMap& depth_map = problem_.depth_maps->at(image_idx);
      THROW_CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_idx;
      THROW_CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_idx;
    }
  }

  if (options_.geom_consistency) {
    const Image& ref_image = problem_.images->at(problem_.ref_image_idx);
    const NormalMap& ref_normal_map =
        problem_.normal_maps->at(problem_.ref_image_idx);
    THROW_CHECK_EQ(ref_image.GetWidth(), ref_normal_map.GetWidth());
    THROW_CHECK_EQ(ref_image.GetHeight(), ref_normal_map.GetHeight());
  }
}

void PatchMatch::Run() {
  LOG_HEADING2("PatchMatch::Run");

  Check();

  patch_match_cuda_ = std::make_unique<PatchMatchCuda>(options_, problem_);
  patch_match_cuda_->Run();
}

DepthMap PatchMatch::GetDepthMap() const {
  return patch_match_cuda_->GetDepthMap();
}

NormalMap PatchMatch::GetNormalMap() const {
  return patch_match_cuda_->GetNormalMap();
}

Mat<float> PatchMatch::GetSelProbMap() const {
  return patch_match_cuda_->GetSelProbMap();
}

ConsistencyGraph PatchMatch::GetConsistencyGraph() const {
  const auto& ref_image = problem_.images->at(problem_.ref_image_idx);
  return ConsistencyGraph(ref_image.GetWidth(),
                          ref_image.GetHeight(),
                          patch_match_cuda_->GetConsistentImageIdxs());
}

PatchMatchController::PatchMatchController(
    const PatchMatchOptions& options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path)
    : options_(options),
      workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      pmvs_option_name_(pmvs_option_name),
      config_path_(config_path) {
  std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
}

void PatchMatchController::Run() {
  Timer run_timer;
  run_timer.Start();
  ReadWorkspace();
  ReadProblems();
  ReadGpuIndices();

  thread_pool_ = std::make_unique<ThreadPool>(gpu_indices_.size());
  io_thread_pool_ = std::make_unique<ThreadPool>(
      GetEffectiveNumThreads(options_.num_threads));

#ifndef COLMAP_HIP_ENABLED
  if (gpu_indices_.size() == 1 && DefaultReferencePairSettings(options_)) {
    SetBestCudaDevice(gpu_indices_.front());
    const auto& model = workspace_->GetModel();
    const auto shape = [&](size_t index) {
      const auto& problem = problems_.at(index);
      const auto& ref = model.images.at(problem.ref_image_idx);
      size_t source_width = 0, source_height = 0;
      for (const int source : problem.src_image_idxs) {
        const auto& image = model.images.at(source);
        source_width = std::max(source_width, image.GetWidth());
        source_height = std::max(source_height, image.GetHeight());
      }
      return std::make_tuple(ref.GetWidth(),
                             ref.GetHeight(),
                             problem.src_image_idxs.size(),
                             source_width,
                             source_height);
    };
    const auto run_phase = [&](const PatchMatchOptions& phase) {
      for (size_t index = 0; index < problems_.size();) {
        // Batch only adjacent matching default-path references. A rejected
        // group falls through to the unchanged pair/single scheduler below.
        size_t batch_end = index;
        const auto first_shape = shape(index);
        while (batch_end < problems_.size() && batch_end < index + 5 &&
               shape(batch_end) == first_shape)
          ++batch_end;
        std::vector<std::pair<size_t, size_t>> batch_shapes;
        for (size_t next = index; next < batch_end; ++next)
          batch_shapes.emplace_back(std::get<0>(shape(next)),
                                    std::get<1>(shape(next)));
        if (CanBatchPatchMatchViews(batch_shapes, phase.geom_consistency)) {
          std::vector<size_t> indices;
          for (size_t next = index; next < batch_end; ++next)
            indices.push_back(next);
          thread_pool_->AddTask(&PatchMatchController::ProcessProblemBatch,
                                this,
                                phase,
                                std::move(indices));
          index = batch_end;
          continue;
        }
        if (index + 1 < problems_.size()) {
          const auto first_shape = shape(index);
          const auto second_shape = shape(index + 1);
          if (first_shape == second_shape &&
              CanPairPatchMatchViews(std::get<0>(first_shape),
                                     std::get<1>(first_shape),
                                     std::get<0>(second_shape),
                                     std::get<1>(second_shape),
                                     phase.geom_consistency)) {
            thread_pool_->AddTask(&PatchMatchController::ProcessProblemPair,
                                  this,
                                  phase,
                                  index,
                                  index + 1);
            index += 2;
            continue;
          }
        }
        thread_pool_->AddTask(
            &PatchMatchController::ProcessProblem, this, phase, index);
        ++index;
      }
      thread_pool_->Wait();
    };
    if (options_.geom_consistency) {
      auto photo_options = options_;
      photo_options.geom_consistency = false;
      photo_options.filter = false;
      run_phase(photo_options);
    }
    run_phase(options_);
    run_timer.PrintMinutes();
    return;
  }
#endif

  // If geometric consistency is enabled, then photometric output must be
  // computed first for all images without filtering.
  if (options_.geom_consistency) {
    auto photometric_options = options_;
    photometric_options.geom_consistency = false;
    photometric_options.filter = false;

    for (size_t problem_idx = 0; problem_idx < problems_.size();
         ++problem_idx) {
      thread_pool_->AddTask(&PatchMatchController::ProcessProblem,
                            this,
                            photometric_options,
                            problem_idx);
    }

    thread_pool_->Wait();
  }

  for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx) {
    thread_pool_->AddTask(
        &PatchMatchController::ProcessProblem, this, options_, problem_idx);
  }

  thread_pool_->Wait();

  run_timer.PrintMinutes();
}

void PatchMatchController::ReadWorkspace() {
  LOG(INFO) << "Reading workspace...";

  Workspace::Options workspace_options;

  auto workspace_format_lower_case = workspace_format_;
  StringToLower(&workspace_format_lower_case);
  if (workspace_format_lower_case == "pmvs") {
    workspace_options.stereo_folder =
        StringPrintf("stereo-%s", pmvs_option_name_.c_str());
  }

  workspace_options.max_image_size = options_.max_image_size;
  workspace_options.image_as_rgb = false;
  workspace_options.cache_size = options_.cache_size;
  workspace_options.workspace_path = workspace_path_;
  workspace_options.workspace_format = workspace_format_;
  workspace_options.input_type = options_.geom_consistency ? "photometric" : "";

  workspace_ = std::make_unique<CachedWorkspace>(workspace_options);

  if (workspace_format_lower_case == "pmvs") {
    LOG(INFO) << StringPrintf("Importing PMVS workspace (option %s)...",
                              pmvs_option_name_.c_str());
    ImportPMVSWorkspace(*workspace_, pmvs_option_name_);
  }

  depth_ranges_ = workspace_->GetModel().ComputeDepthRanges();
}

void PatchMatchController::ReadProblems() {
  LOG(INFO) << "Reading configuration...";

  problems_.clear();

  const auto& model = workspace_->GetModel();

  const auto config_path = config_path_.empty()
                               ? workspace_path_ /
                                     workspace_->GetOptions().stereo_folder /
                                     "patch-match.cfg"
                               : config_path_;
  std::vector<std::string> config = ReadTextFileLines(config_path);

  std::vector<std::map<int, int>> shared_num_points;
  std::vector<std::map<int, float>> triangulation_angles;

  const float min_triangulation_angle_rad =
      DegToRad(options_.min_triangulation_angle);

  std::string ref_image_name;
  FlatHashSet<int> ref_image_idxs;

  struct ProblemConfig {
    std::string ref_image_name;
    std::vector<std::string> src_image_names;
  };
  std::vector<ProblemConfig> problem_configs;

  for (size_t i = 0; i < config.size(); ++i) {
    std::string& config_line = config[i];
    StringTrim(&config_line);

    if (config_line.empty() || config_line[0] == '#') {
      continue;
    }

    if (ref_image_name.empty()) {
      ref_image_name = config_line;
      continue;
    }

    ref_image_idxs.insert(model.GetImageIdx(ref_image_name));

    ProblemConfig problem_config;
    problem_config.ref_image_name = ref_image_name;
    problem_config.src_image_names = CSVToVector<std::string>(config_line);
    problem_configs.push_back(problem_config);

    ref_image_name.clear();
  }

  for (const auto& problem_config : problem_configs) {
    PatchMatch::Problem problem;

    problem.ref_image_idx = model.GetImageIdx(problem_config.ref_image_name);

    if (problem_config.src_image_names.size() == 1 &&
        problem_config.src_image_names[0] == "__all__") {
      // Use all images as source images.
      problem.src_image_idxs.clear();
      problem.src_image_idxs.reserve(model.images.size() - 1);
      for (size_t image_idx = 0; image_idx < model.images.size(); ++image_idx) {
        if (static_cast<int>(image_idx) != problem.ref_image_idx) {
          problem.src_image_idxs.push_back(image_idx);
        }
      }
    } else if (problem_config.src_image_names.size() == 2 &&
               problem_config.src_image_names[0] == "__auto__") {
      // Use maximum number of overlapping images as source images. Overlapping
      // will be sorted based on the number of shared points to the reference
      // image and the top ranked images are selected. Note that images are only
      // selected if some points have a sufficient triangulation angle.

      if (shared_num_points.empty()) {
        shared_num_points = model.ComputeSharedPoints();
      }
      if (triangulation_angles.empty()) {
        const float kTriangulationAnglePercentile = 75;
        triangulation_angles =
            model.ComputeTriangulationAngles(kTriangulationAnglePercentile);
      }

      const size_t max_num_src_images =
          std::stoll(problem_config.src_image_names[1]);

      const auto& overlapping_images =
          shared_num_points.at(problem.ref_image_idx);
      const auto& overlapping_triangulation_angles =
          triangulation_angles.at(problem.ref_image_idx);

      std::vector<std::pair<int, int>> src_images;
      src_images.reserve(overlapping_images.size());
      for (const auto& image : overlapping_images) {
        if (overlapping_triangulation_angles.at(image.first) >=
            min_triangulation_angle_rad) {
          src_images.emplace_back(image.first, image.second);
        }
      }

      const size_t eff_max_num_src_images =
          std::min(src_images.size(), max_num_src_images);

      std::partial_sort(src_images.begin(),
                        src_images.begin() + eff_max_num_src_images,
                        src_images.end(),
                        [](const std::pair<int, int>& image1,
                           const std::pair<int, int>& image2) {
                          return image1.second > image2.second;
                        });

      problem.src_image_idxs.reserve(eff_max_num_src_images);
      for (size_t i = 0; i < eff_max_num_src_images; ++i) {
        problem.src_image_idxs.push_back(src_images[i].first);
      }
    } else {
      problem.src_image_idxs.reserve(problem_config.src_image_names.size());
      for (const auto& src_image_name : problem_config.src_image_names) {
        problem.src_image_idxs.push_back(model.GetImageIdx(src_image_name));
      }
    }

    if (problem.src_image_idxs.empty()) {
      LOG(WARNING) << StringPrintf(
          "Ignoring reference image %s, because it has no "
          "source images.",
          problem_config.ref_image_name.c_str());
    } else {
      problems_.push_back(problem);
    }
  }

  LOG(INFO) << StringPrintf("Configuration has %d problems...",
                            problems_.size());
}

void PatchMatchController::ReadGpuIndices() {
  gpu_indices_ = CSVToVector<int>(options_.gpu_index);
  if (gpu_indices_.size() == 1 && gpu_indices_[0] == -1) {
    const int num_cuda_devices = GetNumCudaDevices();
    THROW_CHECK_GT(num_cuda_devices, 0);
    gpu_indices_.resize(num_cuda_devices);
    std::iota(gpu_indices_.begin(), gpu_indices_.end(), 0);
  }
}

bool PatchMatchController::PrepareProblem(const PatchMatchOptions& options,
                                          const size_t problem_idx,
                                          PreparedProblem* prepared) {
  if (CheckIfStopped()) {
    return false;
  }

  const auto& model = workspace_->GetModel();

  auto& shared_problem = problems_.at(problem_idx);
  auto& problem = prepared->problem;
  problem = shared_problem;
  const int gpu_index = gpu_indices_.at(thread_pool_->GetThreadIndex());
  THROW_CHECK_GE(gpu_index, -1);

  const std::string& stereo_folder = workspace_->GetOptions().stereo_folder;
  const std::string output_type =
      options.geom_consistency ? "geometric" : "photometric";
  const std::string image_name = model.GetImageName(problem.ref_image_idx);
  const std::string file_name =
      StringPrintf("%s.%s.bin", image_name.c_str(), output_type.c_str());
  const auto depth_map_path =
      workspace_path_ / stereo_folder / "depth_maps" / file_name;
  const auto normal_map_path =
      workspace_path_ / stereo_folder / "normal_maps" / file_name;
  const auto consistency_graph_path =
      workspace_path_ / stereo_folder / "consistency_graphs" / file_name;

  if (ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
      (!options.write_consistency_graph ||
       ExistsFile(consistency_graph_path))) {
    return false;
  }

  LOG_HEADING1(StringPrintf("Processing view %d / %d for %s",
                            problem_idx + 1,
                            problems_.size(),
                            image_name.c_str()));

  auto patch_match_options = options;

  if (patch_match_options.depth_min < 0 || patch_match_options.depth_max < 0) {
    patch_match_options.depth_min =
        depth_ranges_.at(problem.ref_image_idx).first;
    patch_match_options.depth_max =
        depth_ranges_.at(problem.ref_image_idx).second;
    THROW_CHECK(patch_match_options.depth_min > 0 &&
                patch_match_options.depth_max > 0)
        << " - You must manually set the minimum and maximum depth, since no "
           "sparse model is provided in the workspace.";
  }

  patch_match_options.gpu_index = std::to_string(gpu_index);

  if (patch_match_options.sigma_spatial <= 0.0f) {
    patch_match_options.sigma_spatial = patch_match_options.window_radius;
  }

  auto& images = prepared->images;
  auto& depth_maps = prepared->depth_maps;
  auto& normal_maps = prepared->normal_maps;
  images = model.images;
  if (options.geom_consistency) {
    depth_maps.resize(model.images.size());
    normal_maps.resize(model.images.size());
  }

  problem.images = &images;
  problem.depth_maps = &depth_maps;
  problem.normal_maps = &normal_maps;

  {
    // Collect all used images in current problem.
    FlatHashSet<int> used_image_idxs(problem.src_image_idxs.begin(),
                                     problem.src_image_idxs.end());
    used_image_idxs.insert(problem.ref_image_idx);

    patch_match_options.filter_min_num_consistent =
        std::min(static_cast<int>(used_image_idxs.size()) - 1,
                 patch_match_options.filter_min_num_consistent);

    // Only access workspace from one thread at a time and only spawn resample
    // threads from one master thread at a time.
    std::unique_lock<std::mutex> lock(workspace_mutex_);

    LOG(INFO) << "Reading inputs...";

    // Filter by file existence first (fast, no heavy I/O).
    std::vector<int> src_image_idxs;
    std::vector<int> valid_image_idxs;
    for (const auto image_idx : used_image_idxs) {
      if (!workspace_->HasBitmap(image_idx) ||
          (options.geom_consistency && !workspace_->HasDepthMap(image_idx)) ||
          (options.geom_consistency && !workspace_->HasNormalMap(image_idx))) {
        if (options.allow_missing_files) {
          LOG(WARNING) << StringPrintf(
              "Skipping source image %d: %s for missing "
              "image or depth/normal map",
              image_idx,
              model.GetImageName(image_idx).c_str());
          continue;
        } else {
          LOG(ERROR) << StringPrintf(
              "Missing image or map dependency for image %d: %s",
              image_idx,
              model.GetImageName(image_idx).c_str());
        }
      }

      valid_image_idxs.push_back(image_idx);
      if (image_idx != problem.ref_image_idx) {
        src_image_idxs.push_back(image_idx);
      }
    }

    // Read images in parallel using the shared I/O thread pool.
    RunPatchMatchInputReads(*io_thread_pool_, [&](auto& futures) {
      futures.reserve(valid_image_idxs.size());
      for (const auto image_idx : valid_image_idxs) {
        futures.push_back(io_thread_pool_->AddTask([&, image_idx]() {
          images.at(image_idx).SetBitmap(
              *workspace_->GetBitmapLease(image_idx));
          if (options.geom_consistency) {
            depth_maps.at(image_idx) = *workspace_->GetDepthMapLease(image_idx);
            if (!options.reference_only_normal_materialization ||
                image_idx == problem.ref_image_idx ||
                !workspace_->ValidateSourceNormalWithoutMaterialization(
                    image_idx)) {
              normal_maps.at(image_idx) =
                  *workspace_->GetNormalMapLease(image_idx);
            }
          }
        }));
      }
    });

    shared_problem.src_image_idxs = src_image_idxs;
    problem.src_image_idxs = src_image_idxs;
  }

  problem.Print();
  patch_match_options.Print();

  prepared->options = patch_match_options;
  prepared->depth_path = depth_map_path;
  prepared->normal_path = normal_map_path;
  prepared->consistency_path = consistency_graph_path;
  prepared->valid = true;
  return true;
}

void PatchMatchController::ProcessProblem(const PatchMatchOptions& options,
                                          const size_t problem_idx) {
  PreparedProblem prepared;
  if (!PrepareProblem(options, problem_idx, &prepared)) return;
  PatchMatch patch_match(prepared.options, prepared.problem);
  patch_match.Run();

  LOG(INFO) << std::endl
            << StringPrintf(
                   "Writing %s output for %s",
                   options.geom_consistency ? "geometric" : "photometric",
                   workspace_->GetModel()
                       .GetImageName(prepared.problem.ref_image_idx)
                       .c_str());

  patch_match.GetDepthMap().Write(prepared.depth_path);
  patch_match.GetNormalMap().Write(prepared.normal_path);
  if (options.write_consistency_graph) {
    patch_match.GetConsistencyGraph().Write(prepared.consistency_path);
  }
}

void PatchMatchController::ProcessProblemPair(const PatchMatchOptions& options,
                                              const size_t first_problem_idx,
                                              const size_t second_problem_idx) {
#ifdef COLMAP_HIP_ENABLED
  ProcessProblem(options, first_problem_idx);
  ProcessProblem(options, second_problem_idx);
#else
  const auto write_cuda = [](const PreparedProblem& input,
                             const PatchMatchCuda& cuda) {
    cuda.GetDepthMap().Write(input.depth_path);
    cuda.GetNormalMap().Write(input.normal_path);
    if (input.options.write_consistency_graph) {
      const auto& ref = input.problem.images->at(input.problem.ref_image_idx);
      ConsistencyGraph(
          ref.GetWidth(), ref.GetHeight(), cuda.GetConsistentImageIdxs())
          .Write(input.consistency_path);
    }
  };
  const auto run_native = [](const PreparedProblem& input) {
    PatchMatch solver(input.options, input.problem);
    solver.Run();
    solver.GetDepthMap().Write(input.depth_path);
    solver.GetNormalMap().Write(input.normal_path);
    if (input.options.write_consistency_graph) {
      solver.GetConsistencyGraph().Write(input.consistency_path);
    }
  };

  PreparedProblem first, second;
  PrepareProblem(options, first_problem_idx, &first);
  PrepareProblem(options, second_problem_idx, &second);
  if (!first.valid || !second.valid) {
    if (first.valid) run_native(first);
    if (second.valid) run_native(second);
    return;
  }

  PatchMatch(first.options, first.problem).Check();
  PatchMatch(second.options, second.problem).Check();
  // CUDA's current device is per host thread. The controller's metadata
  // query selected a device on the caller, not on this pool worker.
  SetBestCudaDevice(gpu_indices_.at(thread_pool_->GetThreadIndex()));
  const auto& first_ref = first.problem.images->at(first.problem.ref_image_idx);
  const auto& second_ref =
      second.problem.images->at(second.problem.ref_image_idx);
  if (first.problem.src_image_idxs.size() !=
          second.problem.src_image_idxs.size() ||
      !CanPairPatchMatchViews(first_ref.GetWidth(),
                              first_ref.GetHeight(),
                              second_ref.GetWidth(),
                              second_ref.GetHeight(),
                              options.geom_consistency)) {
    run_native(first);
    run_native(second);
    return;
  }

  auto first_cuda =
      std::make_unique<PatchMatchCuda>(first.options, first.problem);
  CUDA_SYNC_AND_CHECK();
  size_t free_after_first = 0, total_after_first = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_after_first, &total_after_first));
  constexpr size_t kExternalMargin = size_t(256) << 20;
  const size_t owned_peak = first_cuda->PairPeakBytes();
  const bool enough_for_second_and_rotation =
      owned_peak <= std::numeric_limits<size_t>::max() - kExternalMargin &&
      free_after_first >= owned_peak + kExternalMargin;
  const auto finish_first_native = [&]() {
    first_cuda->Run();
    write_cuda(first, *first_cuda);
    first_cuda.reset();
    run_native(second);
  };
  if (!enough_for_second_and_rotation) {
    finish_first_native();
    return;
  }
  auto second_cuda =
      std::make_unique<PatchMatchCuda>(second.options, second.problem);
  CUDA_SYNC_AND_CHECK();
  size_t free_after_second = 0, total_after_second = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_after_second, &total_after_second));
  // A banked rotation has allocated both orientations and masks in each
  // constructor. The native path can still allocate rotation outputs.
  const size_t remaining_needed = options.rotation_buffer_reuse
                                      ? kExternalMargin
                                      : owned_peak + kExternalMargin;
  if (free_after_second < remaining_needed) {
    second_cuda.reset();
    finish_first_native();
    return;
  }

  LOG(INFO) << "Running paired PatchMatch references " << first_problem_idx
            << " and " << second_problem_idx;
  PairStreams streams;
  RunPatchMatchPair(*first_cuda, *second_cuda, streams.first, streams.second);
  write_cuda(first, *first_cuda);
  write_cuda(second, *second_cuda);
#endif
}

void PatchMatchController::ProcessProblemBatch(
    const PatchMatchOptions& options, std::vector<size_t> problem_indices) {
#ifdef COLMAP_HIP_ENABLED
  for (const size_t index : problem_indices) ProcessProblem(options, index);
#else
  THROW_CHECK_GE(problem_indices.size(), 3);
  THROW_CHECK_LE(problem_indices.size(), 5);
  // PreparedProblem::problem points into its own vectors: heap owners keep
  // those addresses stable while CUDA owners and readback remain alive.
  std::vector<std::unique_ptr<PreparedProblem>> input;
  input.reserve(problem_indices.size());
  for (const size_t index : problem_indices) {
    auto prepared = std::make_unique<PreparedProblem>();
    if (PrepareProblem(options, index, prepared.get())) {
      PatchMatch(prepared->options, prepared->problem).Check();
      input.push_back(std::move(prepared));
    }
  }
  if (input.empty()) return;
  SetBestCudaDevice(gpu_indices_.at(thread_pool_->GetThreadIndex()));
  const auto shape = [](const PreparedProblem& p) {
    const auto& ref = p.problem.images->at(p.problem.ref_image_idx);
    size_t source_width = 0, source_height = 0;
    for (const int source : p.problem.src_image_idxs) {
      const auto& image = p.problem.images->at(source);
      source_width = std::max(source_width, image.GetWidth());
      source_height = std::max(source_height, image.GetHeight());
    }
    return std::make_tuple(ref.GetWidth(),
                           ref.GetHeight(),
                           p.problem.src_image_idxs.size(),
                           source_width,
                           source_height);
  };
  const auto write_cuda = [](const PreparedProblem& p,
                             const PatchMatchCuda& cuda) {
    cuda.GetDepthMap().Write(p.depth_path);
    cuda.GetNormalMap().Write(p.normal_path);
    if (p.options.write_consistency_graph) {
      const auto& ref = p.problem.images->at(p.problem.ref_image_idx);
      ConsistencyGraph(
          ref.GetWidth(), ref.GetHeight(), cuda.GetConsistentImageIdxs())
          .Write(p.consistency_path);
    }
  };
  const auto run_native = [](const PreparedProblem& p) {
    PatchMatch solver(p.options, p.problem);
    solver.Run();
    solver.GetDepthMap().Write(p.depth_path);
    solver.GetNormalMap().Write(p.normal_path);
    if (p.options.write_consistency_graph)
      solver.GetConsistencyGraph().Write(p.consistency_path);
  };
  std::vector<std::unique_ptr<PatchMatchCuda>> owners(input.size());
  constexpr size_t kExternalMargin = size_t(256) << 20;
  const auto pair_or_single_fallback = [&]() {
    for (size_t i = 0; i < input.size();) {
      if (i + 1 < input.size()) {
        const auto& a =
            input[i]->problem.images->at(input[i]->problem.ref_image_idx);
        const auto& b = input[i + 1]->problem.images->at(
            input[i + 1]->problem.ref_image_idx);
        if (shape(*input[i]) == shape(*input[i + 1]) &&
            CanPairPatchMatchViews(a.GetWidth(),
                                   a.GetHeight(),
                                   b.GetWidth(),
                                   b.GetHeight(),
                                   options.geom_consistency)) {
          if (!owners[i])
            owners[i] = std::make_unique<PatchMatchCuda>(input[i]->options,
                                                         input[i]->problem);
          CUDA_SYNC_AND_CHECK();
          const size_t owned_peak = owners[i]->PairPeakBytes();
          size_t free_bytes = 0, total_bytes = 0;
          CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
          if (!owners[i + 1] &&
              owned_peak <=
                  std::numeric_limits<size_t>::max() - kExternalMargin &&
              free_bytes >= owned_peak + kExternalMargin) {
            owners[i + 1] = std::make_unique<PatchMatchCuda>(
                input[i + 1]->options, input[i + 1]->problem);
            CUDA_SYNC_AND_CHECK();
          }
          if (owners[i + 1]) {
            CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
            const bool enough_for_rotation =
                options.rotation_buffer_reuse
                    ? free_bytes >= kExternalMargin
                    : owned_peak <= std::numeric_limits<size_t>::max() -
                                        kExternalMargin &&
                          free_bytes >= owned_peak + kExternalMargin;
            if (enough_for_rotation) {
              PairStreams streams;
              RunPatchMatchPair(
                  *owners[i], *owners[i + 1], streams.first, streams.second);
              write_cuda(*input[i], *owners[i]);
              write_cuda(*input[i + 1], *owners[i + 1]);
              owners[i].reset();
              owners[i + 1].reset();
              i += 2;
              continue;
            }
            // Match the incumbent pair's low-memory fallback: the first
            // constructed owner is retained, the second is released before
            // running the first view singly.
            owners[i + 1].reset();
          }
        }
      }
      if (owners[i]) {
        owners[i]->Run();
        write_cuda(*input[i], *owners[i]);
        owners[i].reset();
      } else {
        run_native(*input[i]);
      }
      ++i;
    }
  };

  if (input.size() < 3) {
    pair_or_single_fallback();
    return;
  }
  std::vector<std::pair<size_t, size_t>> shapes;
  const auto first_shape = shape(*input.front());
  for (const auto& p : input) {
    const auto this_shape = shape(*p);
    if (this_shape != first_shape) {
      pair_or_single_fallback();
      return;
    }
    shapes.emplace_back(std::get<0>(this_shape), std::get<1>(this_shape));
  }
  if (!CanBatchPatchMatchViews(shapes, options.geom_consistency)) {
    pair_or_single_fallback();
    return;
  }
  owners[0] =
      std::make_unique<PatchMatchCuda>(input[0]->options, input[0]->problem);
  CUDA_SYNC_AND_CHECK();
  const size_t first_peak = owners[0]->PairPeakBytes();
  std::vector<const PatchMatch::Problem*> source_problems;
  for (const auto& prepared : input)
    source_problems.push_back(&prepared->problem);
  std::shared_ptr<const PatchMatchSparseSources> shared;
  const size_t union_budget = PatchMatchSparseSources::Budget(
      source_problems, options.geom_consistency);
  size_t free_bytes = 0, total_bytes = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
  // The first independent owner stays intact during union construction.
  // Reserve the full remaining independent peaks plus the incumbent extra
  // transient peak. Rejecting sharing retains the exact old batch path.
  const size_t maximum = std::numeric_limits<size_t>::max();
  if (union_budget && union_budget <= maximum - kExternalMargin &&
      first_peak <= (maximum - kExternalMargin - union_budget) / input.size() &&
      free_bytes >=
          union_budget + input.size() * first_peak + kExternalMargin) {
    shared = PatchMatchSparseSources::Create(source_problems,
                                             options.geom_consistency);
    if (shared) owners[0]->UseSparseSources(shared);
  }
  const auto admit_remaining = [&]() {
    bool memory_admitted = true;
    for (size_t i = 1; i < input.size(); ++i) {
      size_t free_bytes = 0, total_bytes = 0;
      CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
      const size_t remaining = input.size() - i + 1;
      if (first_peak > (std::numeric_limits<size_t>::max() - kExternalMargin) /
                           remaining ||
          free_bytes < remaining * first_peak + kExternalMargin) {
        memory_admitted = false;
        break;
      }
      owners[i] = std::make_unique<PatchMatchCuda>(
          input[i]->options, input[i]->problem, shared);
      CUDA_SYNC_AND_CHECK();
      if (owners[i]->PairPeakBytes() > first_peak) {
        owners[i].reset();
        memory_admitted = false;
        break;
      }
    }
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (first_peak > std::numeric_limits<size_t>::max() - kExternalMargin ||
        free_bytes < first_peak + kExternalMargin)
      memory_admitted = false;
    return memory_admitted;
  };
  bool memory_admitted = admit_remaining();
  if (!memory_admitted && shared) {
    // Do not lower concurrency because the optional bank was rejected. Drop
    // every shared owner/bank before reconstructing the incumbent first view.
    for (auto& owner : owners) owner.reset();
    shared.reset();
    owners[0] =
        std::make_unique<PatchMatchCuda>(input[0]->options, input[0]->problem);
    CUDA_SYNC_AND_CHECK();
    memory_admitted =
        owners[0]->PairPeakBytes() <= first_peak && admit_remaining();
  }
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (!memory_admitted) {
    if (first_peak > std::numeric_limits<size_t>::max() - kExternalMargin ||
        free_bytes < first_peak + kExternalMargin)
      for (size_t i = 1; i < owners.size(); ++i) owners[i].reset();
    pair_or_single_fallback();
    return;
  }
  std::vector<PatchMatchCuda*> views;
  for (auto& owner : owners) views.push_back(owner.get());
  struct BatchStreams {
    std::vector<cudaStream_t> values;
    explicit BatchStreams(size_t count) : values(count, nullptr) {
      try {
        for (auto& stream : values)
          CUDA_SAFE_CALL(
              cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      } catch (...) {
        for (auto stream : values)
          if (stream) (void)cudaStreamDestroy(stream);
        throw;
      }
    }
    ~BatchStreams() {
      for (auto stream : values) {
        if (stream) {
          (void)cudaStreamSynchronize(stream);
          (void)cudaStreamDestroy(stream);
        }
      }
    }
  } streams(input.size());
  LOG(INFO) << "Running " << input.size() << " reference PatchMatch batch"
            << (shared ? " (sparse source aliases)" : "");
  RunPatchMatchBatch(views, streams.values);
  for (size_t i = 0; i < input.size(); ++i) write_cuda(*input[i], *owners[i]);
  for (auto& owner : owners) owner.reset();
#endif
}

}  // namespace mvs
}  // namespace colmap
