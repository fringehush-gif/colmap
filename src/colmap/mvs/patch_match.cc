// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"

#include "colmap/math/math.h"
#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/patch_match_io.h"
#include "colmap/mvs/patch_match_source_array.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/cuda.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace colmap {
namespace mvs {

namespace {
template <typename Cleanup>
void CheckedCudaCleanup(Cleanup&& cleanup) {
  CudaCleanupScope scope;
  try {
    cleanup();
  } catch (...) {
    scope.Check();
    throw;
  }
  scope.Check();
}

std::unique_ptr<PatchMatchCuda> CreateCudaOwner(
    const PatchMatchOptions& options,
    const PatchMatch::Problem& problem,
    std::shared_ptr<const PatchMatchSparseSources> shared = {},
    const PatchMatchCuda* donor = nullptr,
    const SparseSourceTestFailure* failure = nullptr) {
  CudaCleanupScope scope;
  std::unique_ptr<PatchMatchCuda> owner;
  try {
    owner = std::make_unique<PatchMatchCuda>(
        options, problem, std::move(shared), donor, failure);
  } catch (...) {
    scope.Check();
    throw;
  }
  scope.Check();
  return owner;
}

#ifndef COLMAP_HIP_ENABLED
class OptionalCudaAllocationRefused : public CudaAllocationRefused {
 public:
  explicit OptionalCudaAllocationRefused(const CudaAllocationRefused& error)
      : CudaAllocationRefused(error.Operation().c_str(), error.Code()) {}
};
#endif
}  // namespace

#ifndef COLMAP_HIP_ENABLED
namespace {
thread_local bool work_stream_construction_active = false;

class WorkStreamConstructionScope {
 public:
  WorkStreamConstructionScope()
      : previous_(std::exchange(work_stream_construction_active, true)) {}
  ~WorkStreamConstructionScope() {
    work_stream_construction_active = previous_;
  }
  WorkStreamConstructionScope(const WorkStreamConstructionScope&) = delete;
  WorkStreamConstructionScope& operator=(const WorkStreamConstructionScope&) =
      delete;

 private:
  const bool previous_;
};

class WorkStreams {
 public:
  WorkStreams(size_t count, bool synchronize_on_close)
      : values(count, nullptr), synchronize_on_close_(synchronize_on_close) {
    WorkStreamConstructionScope construction;
    CudaCleanupScope cleanup;
    try {
      for (auto& stream : values) {
        CUDA_SAFE_ALLOCATE(
            "PatchMatch work stream",
            stream,
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        THROW_CHECK(stream != nullptr);
      }
      cleanup.Check();
    } catch (...) {
      // No sweep has started. Attempt every partial release before allowing
      // a clean resource refusal to reach a smaller admission path.
      Close(false);
      cleanup.Check();
      throw;
    }
  }
  // Normal destruction is enclosed in CheckedCudaCleanup by each caller.
  // Unwinding closes best-effort without turning a sweep failure into retry.
  ~WorkStreams() { Close(synchronize_on_close_); }
  WorkStreams(const WorkStreams&) = delete;
  WorkStreams& operator=(const WorkStreams&) = delete;

  std::vector<cudaStream_t> values;

 private:
  void Close(bool synchronize) noexcept {
    for (size_t offset = 0; offset < values.size(); ++offset) {
      // Preserve the batch's forward synchronized teardown and the pair's
      // reverse construction order. A failed release is never retried here.
      const size_t index =
          synchronize_on_close_ ? offset : values.size() - 1 - offset;
      const auto stream = std::exchange(values[index], nullptr);
      if (!stream) continue;
      if (synchronize) {
        CudaCleanupScope::ObserveCleanup(
            "cudaStreamSynchronize PatchMatch work stream",
            cudaStreamSynchronize(stream));
      }
      CudaCleanupScope::ObserveCleanup(
          "cudaStreamDestroy PatchMatch work stream",
          cudaStreamDestroy(stream));
    }
  }

  const bool synchronize_on_close_;
};

bool DefaultReferencePairSettings(const PatchMatchOptions& o) {
  // sparse_source_sharing is intentionally independent of batch/pair admission.
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
         o.geom_consistency == o.filter;
}
}  // namespace

namespace internal {
// Observation only: fault selection belongs to the native test wrappers.
// No installed header or runtime option exposes this construction phase.
bool IsPatchMatchWorkStreamConstructionActive() noexcept {
  return work_stream_construction_active;
}
}  // namespace internal
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
    const Bitmap& bitmap = problem_.InputBitmap(image_idx);
    THROW_CHECK_GT(bitmap.Width(), 0) << image_idx;
    THROW_CHECK_GT(bitmap.Height(), 0) << image_idx;
    THROW_CHECK(bitmap.IsGrey()) << image_idx;
    THROW_CHECK_EQ(image.GetWidth(), bitmap.Width()) << image_idx;
    THROW_CHECK_EQ(image.GetHeight(), bitmap.Height()) << image_idx;

    // Make sure, the calibration matrix only contains fx, fy, cx, cy.
    THROW_CHECK_LT(std::abs(image.GetK()[1] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[3] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[6] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[7] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_idx;

    if (options_.geom_consistency) {
      THROW_CHECK_LT(image_idx, problem_.depth_maps->size()) << image_idx;
      const DepthMap& depth_map = problem_.InputDepthMap(image_idx);
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

  patch_match_cuda_ = CreateCudaOwner(options_, problem_);
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

  RunPatchMatchTasks(*thread_pool_, [&](auto&) {
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
          if (CanBatchPatchMatchViews(
                  batch_shapes, phase.geom_consistency, phase.checkpoint_hmm)) {
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
                                       phase.geom_consistency,
                                       phase.checkpoint_hmm)) {
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
      };
      if (options_.geom_consistency) {
        auto photo_options = options_;
        photo_options.geom_consistency = false;
        photo_options.filter = false;
        run_phase(photo_options);
        thread_pool_->Wait();
      }
      run_phase(options_);
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
  });
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
                                          PreparedProblem* prepared,
                                          const bool retain_sparse_sources) {
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
  if (retain_sparse_sources && options.sparse_source_sharing) {
    prepared->source_bitmap_leases.resize(model.images.size());
    problem.source_bitmap_leases = &prepared->source_bitmap_leases;
  }
  if (options.geom_consistency) {
    depth_maps.resize(model.images.size());
    normal_maps.resize(model.images.size());
    prepared->source_depth_leases.resize(model.images.size());
    problem.source_depth_leases = &prepared->source_depth_leases;
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
    }

    // Read images in parallel using the shared I/O thread pool.
    RunPatchMatchTasks(*io_thread_pool_, [&](auto& futures) {
      futures.reserve(valid_image_idxs.size());
      for (const auto image_idx : valid_image_idxs) {
        futures.push_back(io_thread_pool_->AddTask([&, image_idx]() {
          const auto bitmap_lease = workspace_->GetBitmapLease(image_idx);
          if (retain_sparse_sources && options.sparse_source_sharing &&
              image_idx != problem.ref_image_idx &&
              bitmap_lease.use_count() > 0 && bitmap_lease->IsGrey() &&
              static_cast<size_t>(bitmap_lease->Width()) ==
                  images.at(image_idx).GetWidth() &&
              static_cast<size_t>(bitmap_lease->Height()) ==
                  images.at(image_idx).GetHeight()) {
            prepared->source_bitmap_leases.at(image_idx) = bitmap_lease;
          } else {
            images.at(image_idx).SetBitmap(*bitmap_lease);
          }
          if (options.geom_consistency) {
            const auto depth_lease = workspace_->GetDepthMapLease(image_idx);
            // Only source maps may alias the immutable workspace owner. The
            // reference retains its independent mutable initialization copy.
            if (image_idx != problem.ref_image_idx &&
                depth_lease.use_count() > 0 &&
                depth_lease->GetWidth() == images.at(image_idx).GetWidth() &&
                depth_lease->GetHeight() == images.at(image_idx).GetHeight()) {
              prepared->source_depth_leases.at(image_idx) = depth_lease;
            } else {
              depth_maps.at(image_idx) = *depth_lease;
            }
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

    // Preserve configured source order while excluding missing and repeated
    // images. Keep the reference separate from the source list.
    used_image_idxs.clear();
    used_image_idxs.insert(valid_image_idxs.begin(), valid_image_idxs.end());
    for (const int image_idx : problem.src_image_idxs) {
      if (image_idx != problem.ref_image_idx &&
          used_image_idxs.erase(image_idx) > 0) {
        src_image_idxs.push_back(image_idx);
      }
    }
    shared_problem.src_image_idxs = src_image_idxs;
    problem.src_image_idxs = std::move(src_image_idxs);
  }

  problem.Print();
  patch_match_options.Print();

  prepared->options = std::move(patch_match_options);
  prepared->depth_path = depth_map_path;
  prepared->normal_path = normal_map_path;
  prepared->consistency_path = consistency_graph_path;
  prepared->valid = true;
  return true;
}

void PatchMatchController::PublishOutput(
    const PreparedProblem& prepared,
    const std::function<void(const std::filesystem::path&)>& write_output) {
  const auto inspect = [](const std::filesystem::path& path,
                          const bool required) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec == std::errc::no_such_file_or_directory) return false;
    if (ec) throw std::filesystem::filesystem_error("stat output", path, ec);
    if (status.type() == std::filesystem::file_type::not_found) return false;
    if (!std::filesystem::is_regular_file(status)) {
      // A dangling graph symlink may later resolve to a different reference's
      // graph and satisfy the filename-based resume predicate.
      THROW_CHECK(!required && !std::filesystem::is_symlink(status) &&
                  !ExistsFile(path))
          << "Refusing to replace non-regular output: " << path;
      return false;
    }
    return true;
  };
  // Refuse unsuitable destinations before invalidating a complete result.
  const bool old_normal = inspect(prepared.normal_path, true);
  const bool old_depth = inspect(prepared.depth_path, true);
  const bool old_graph = inspect(prepared.consistency_path,
                                 prepared.options.write_consistency_graph);
  // Own a fresh directory next to the completion name, so only the final
  // normal-map rename must be atomic. This is not a concurrent-writer protocol
  // or a durability guarantee across power loss.
  static std::atomic<uint64_t> sequence{0};
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path stage;
  for (int attempt = 0; attempt < 128; ++attempt) {
    stage = prepared.normal_path.parent_path() /
            (".patchmatch-stage-" + std::to_string(tick) + "-" +
             std::to_string(sequence.fetch_add(1)));
    if (std::filesystem::create_directory(stage)) break;
    stage.clear();
  }
  THROW_CHECK(!stage.empty()) << "Cannot create PatchMatch staging directory";
  std::filesystem::path stage_normal;
  try {
    stage_normal = stage / "normal.bin";
    // PrepareProblem requires the normal map in every mode. Invalidate it
    // before changing any other final output; failed writes must remain
    // eligible for retry even when old results already exist. Also invalidate
    // an old graph when graph output is disabled, so a later enabled run cannot
    // accept a graph from a different generation.
    if (old_normal) std::filesystem::remove(prepared.normal_path);
    if (old_depth) std::filesystem::remove(prepared.depth_path);
    if (old_graph) std::filesystem::remove(prepared.consistency_path);
    write_output(stage_normal);
    std::filesystem::rename(stage_normal, prepared.normal_path);
  } catch (...) {
    // A failed cleanup must not mask the writer's error or restore completion.
    std::error_code ignored;
    std::filesystem::remove(stage_normal, ignored);
    std::filesystem::remove(stage, ignored);
    throw;
  }
  std::error_code ignored;
  std::filesystem::remove(stage, ignored);
}

void PatchMatchController::WriteOutput(const PreparedProblem& prepared,
                                       const PatchMatch& patch_match) {
  PublishOutput(prepared, [&](const auto& normal_path) {
    // Keep readbacks sequential, with only one temporary map alive at a time.
    patch_match.GetDepthMap().Write(prepared.depth_path);
    if (prepared.options.write_consistency_graph) {
      patch_match.GetConsistencyGraph().Write(prepared.consistency_path);
    }
    patch_match.GetNormalMap().Write(normal_path);
  });
}

#ifndef COLMAP_HIP_ENABLED
void PatchMatchController::WriteOutput(const PreparedProblem& prepared,
                                       const PatchMatchCuda& patch_match) {
  PublishOutput(prepared, [&](const auto& normal_path) {
    patch_match.GetDepthMap().Write(prepared.depth_path);
    if (prepared.options.write_consistency_graph) {
      const auto& ref =
          prepared.problem.images->at(prepared.problem.ref_image_idx);
      ConsistencyGraph(
          ref.GetWidth(), ref.GetHeight(), patch_match.GetConsistentImageIdxs())
          .Write(prepared.consistency_path);
    }
    patch_match.GetNormalMap().Write(normal_path);
  });
}
#endif

void PatchMatchController::ProcessProblem(const PatchMatchOptions& options,
                                          const size_t problem_idx) {
  PreparedProblem prepared;
  if (!PrepareProblem(options, problem_idx, &prepared)) return;
  CheckedCudaCleanup([&] {
    PatchMatch patch_match(prepared.options, prepared.problem);
    patch_match.Run();

    LOG(INFO) << '\n'
              << StringPrintf(
                     "Writing %s output for %s",
                     options.geom_consistency ? "geometric" : "photometric",
                     workspace_->GetModel()
                         .GetImageName(prepared.problem.ref_image_idx)
                         .c_str());

    WriteOutput(prepared, patch_match);
    if (sparse_test_hooks_)
      sparse_test_hooks_->written_references.push_back(
          prepared.problem.ref_image_idx);
  });
}

void PatchMatchController::ProcessProblemPair(const PatchMatchOptions& options,
                                              const size_t first_problem_idx,
                                              const size_t second_problem_idx) {
#ifdef COLMAP_HIP_ENABLED
  ProcessProblem(options, first_problem_idx);
  ProcessProblem(options, second_problem_idx);
#else
  const auto write_cuda = [this](const PreparedProblem& input,
                                 const auto& solver) {
    WriteOutput(input, solver);
    if (sparse_test_hooks_)
      sparse_test_hooks_->written_references.push_back(
          input.problem.ref_image_idx);
  };
  const auto run_native = [&](const PreparedProblem& input) {
    CheckedCudaCleanup([&] {
      PatchMatch solver(input.options, input.problem);
      solver.Run();
      write_cuda(input, solver);
    });
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
                              options.geom_consistency,
                              options.checkpoint_hmm)) {
    run_native(first);
    run_native(second);
    return;
  }

  auto first_cuda = CreateCudaOwner(first.options, first.problem);
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
    CheckedCudaCleanup([&] { first_cuda.reset(); });
    run_native(second);
  };
  if (!enough_for_second_and_rotation) {
    finish_first_native();
    return;
  }
  std::unique_ptr<PatchMatchCuda> second_cuda;
  try {
    second_cuda = CreateCudaOwner(second.options, second.problem);
  } catch (const CudaAllocationRefused& error) {
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
    LOG(WARNING) << "Optional second CUDA owner refused; running singly: "
                 << error.what();
    finish_first_native();
    return;
  }
  CUDA_SYNC_AND_CHECK();
  size_t free_after_second = 0, total_after_second = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_after_second, &total_after_second));
  // A banked rotation has allocated both orientations and masks in each
  // constructor. The native path can still allocate rotation outputs.
  const size_t remaining_needed = options.rotation_buffer_reuse
                                      ? kExternalMargin
                                      : owned_peak + kExternalMargin;
  if (free_after_second < remaining_needed) {
    CheckedCudaCleanup([&] { second_cuda.reset(); });
    finish_first_native();
    return;
  }

  std::optional<WorkStreams> streams;
  try {
    streams.emplace(2, /*synchronize_on_close=*/false);
  } catch (const CudaAllocationRefused& error) {
    CheckedCudaCleanup([&] { second_cuda.reset(); });
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
    LOG(WARNING) << "Optional paired work streams refused; running singly: "
                 << error.what();
    finish_first_native();
    return;
  }
  LOG(INFO) << "Running paired PatchMatch references " << first_problem_idx
            << " and " << second_problem_idx;
  RunPatchMatchPair(
      *first_cuda, *second_cuda, streams->values[0], streams->values[1]);
  write_cuda(first, *first_cuda);
  write_cuda(second, *second_cuda);
  CheckedCudaCleanup([&] {
    streams.reset();
    second_cuda.reset();
    first_cuda.reset();
  });
#endif
}

void PatchMatchController::ProcessProblemBatch(
    const PatchMatchOptions& options,
    const std::vector<size_t>& problem_indices) {
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
    if (PrepareProblem(options, index, prepared.get(), true)) {
      PatchMatch(prepared->options, prepared->problem).Check();
      input.push_back(std::move(prepared));
    }
  }
  if (input.empty()) return;
  SetBestCudaDevice(gpu_indices_.at(thread_pool_->GetThreadIndex()));
  const auto* sparse_failure =
      sparse_test_hooks_ ? sparse_test_hooks_->failure : nullptr;
  const auto query_memory =
      [&](const char* phase, size_t* free, size_t* total) {
        CUDA_SAFE_CALL(cudaMemGetInfo(free, total));
        if (sparse_test_hooks_)
          sparse_test_hooks_->memory_queries.push_back({phase, *free, *total});
      };
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
  const auto write_cuda = [&](const PreparedProblem& p,
                              const PatchMatchCuda& cuda) {
    WriteOutput(p, cuda);
    if (sparse_test_hooks_)
      sparse_test_hooks_->written_references.push_back(p.problem.ref_image_idx);
  };
  const auto run_native = [&](const PreparedProblem& p) {
    CheckedCudaCleanup([&] {
      PatchMatch solver(p.options, p.problem);
      solver.Run();
      WriteOutput(p, solver);
      if (sparse_test_hooks_)
        sparse_test_hooks_->written_references.push_back(
            p.problem.ref_image_idx);
    });
  };
  std::vector<std::unique_ptr<PatchMatchCuda>> owners(input.size());
  constexpr size_t kExternalMargin = size_t(256) << 20;
  const auto pair_or_single_fallback = [&]() {
    for (size_t i = 0; i < input.size();) {
      // Include unpairable and final singleton selections. This observation
      // does not reserve capacity or guarantee ordinary owner/rotation malloc.
      size_t selection_free_bytes = 0, selection_total_bytes = 0;
      query_memory("pair-or-single-selection",
                   &selection_free_bytes,
                   &selection_total_bytes);
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
                                   options.geom_consistency,
                                   options.checkpoint_hmm)) {
          if (!owners[i])
            owners[i] = CreateCudaOwner(input[i]->options, input[i]->problem);
          CUDA_SYNC_AND_CHECK();
          const size_t owned_peak = owners[i]->PairPeakBytes();
          size_t free_bytes = 0, total_bytes = 0;
          query_memory("admission", &free_bytes, &total_bytes);
          if (!owners[i + 1] &&
              owned_peak <=
                  std::numeric_limits<size_t>::max() - kExternalMargin &&
              free_bytes >= owned_peak + kExternalMargin) {
            try {
              owners[i + 1] =
                  CreateCudaOwner(input[i + 1]->options, input[i + 1]->problem);
            } catch (const CudaAllocationRefused& error) {
              query_memory(
                  "after-optional-pair-refusal", &free_bytes, &total_bytes);
              LOG(WARNING)
                  << "Optional pair CUDA owner refused; running singly: "
                  << error.what();
            }
            if (owners[i + 1]) CUDA_SYNC_AND_CHECK();
          }
          if (owners[i + 1]) {
            query_memory("admission", &free_bytes, &total_bytes);
            const bool enough_for_rotation =
                options.rotation_buffer_reuse
                    ? free_bytes >= kExternalMargin
                    : owned_peak <= std::numeric_limits<size_t>::max() -
                                        kExternalMargin &&
                          free_bytes >= owned_peak + kExternalMargin;
            if (enough_for_rotation) {
              std::optional<WorkStreams> streams;
              try {
                streams.emplace(2, /*synchronize_on_close=*/false);
              } catch (const CudaAllocationRefused& error) {
                CheckedCudaCleanup([&] { owners[i + 1].reset(); });
                query_memory(
                    "after-pair-stream-refusal", &free_bytes, &total_bytes);
                LOG(WARNING)
                    << "Optional pair work streams refused; running singly: "
                    << error.what();
              }
              if (streams) {
                RunPatchMatchPair(*owners[i],
                                  *owners[i + 1],
                                  streams->values[0],
                                  streams->values[1]);
                write_cuda(*input[i], *owners[i]);
                write_cuda(*input[i + 1], *owners[i + 1]);
                CheckedCudaCleanup([&] {
                  owners[i].reset();
                  owners[i + 1].reset();
                  streams.reset();
                });
                i += 2;
                continue;
              }
            }
            // Match the incumbent pair's low-memory fallback: the first
            // constructed owner is retained, the second is released before
            // running the first view singly.
            if (owners[i + 1])
              CheckedCudaCleanup([&] { owners[i + 1].reset(); });
          }
        }
      }
      if (owners[i]) {
        owners[i]->Run();
        write_cuda(*input[i], *owners[i]);
        CheckedCudaCleanup([&] { owners[i].reset(); });
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
  if (!CanBatchPatchMatchViews(
          shapes, options.geom_consistency, options.checkpoint_hmm)) {
    pair_or_single_fallback();
    return;
  }
  owners[0] = CreateCudaOwner(input[0]->options, input[0]->problem);
  CUDA_SYNC_AND_CHECK();
  const size_t first_peak = owners[0]->PairPeakBytes();
  bool shared_geometry = owners[0]->CanShareGeometryRotation();
  std::vector<const PatchMatch::Problem*> source_problems;
  source_problems.reserve(input.size());
  for (const auto& prepared : input)
    source_problems.push_back(&prepared->problem);
  std::shared_ptr<const PatchMatchSparseSources> shared;
  size_t free_bytes = 0, total_bytes = 0;
  try {
    const size_t union_budget =
        options.sparse_source_sharing
            ? PatchMatchSparseSources::Budget(
                  source_problems, options.geom_consistency, sparse_failure)
            : 0;
    query_memory("admission", &free_bytes, &total_bytes);
    // The first independent owner stays intact during union construction.
    // Reserve the full remaining independent peaks plus the incumbent extra
    // transient peak. Rejecting sharing retains the exact old batch path.
    const size_t maximum = std::numeric_limits<size_t>::max();
    if (union_budget && union_budget <= maximum - kExternalMargin &&
        first_peak <=
            (maximum - kExternalMargin - union_budget) / input.size() &&
        free_bytes >=
            union_budget + input.size() * first_peak + kExternalMargin) {
      CheckedCudaCleanup([&] {
        shared = PatchMatchSparseSources::Create(
            source_problems, options.geom_consistency, sparse_failure);
        if (shared) owners[0]->UseSparseSources(shared, sparse_failure);
      });
    }
  } catch (const SparseSourceResourceRefused& error) {
    // No consuming kernel has run and the first dense owner is unchanged.
    // Close must succeed before any fresh memory query or smaller admission.
    CheckedCudaCleanup([&] {
      if (shared) shared->Close();
      shared.reset();
    });
    query_memory("after-initial-sparse-refusal", &free_bytes, &total_bytes);
    LOG(WARNING)
        << "Sparse source setup refused; readmitting pairs/singletons: "
        << error.what();
    pair_or_single_fallback();
    return;
  }
  const auto admit_remaining = [&]() {
    bool memory_admitted = true;
    for (size_t i = 1; i < input.size(); ++i) {
      size_t free_bytes = 0, total_bytes = 0;
      query_memory("admission", &free_bytes, &total_bytes);
      const size_t remaining = input.size() - i + 1;
      if (first_peak > (std::numeric_limits<size_t>::max() - kExternalMargin) /
                           remaining ||
          free_bytes < remaining * first_peak + kExternalMargin) {
        memory_admitted = false;
        break;
      }
      try {
        owners[i] = CreateCudaOwner(input[i]->options,
                                    input[i]->problem,
                                    shared,
                                    shared_geometry ? owners[0].get() : nullptr,
                                    sparse_failure);
      } catch (const CudaAllocationRefused& error) {
        throw OptionalCudaAllocationRefused(error);
      }
      CUDA_SYNC_AND_CHECK();
      if (owners[i]->PairPeakBytes() > first_peak ||
          (shared_geometry && !owners[i]->CanShareGeometryRotation())) {
        CheckedCudaCleanup([&] { owners[i].reset(); });
        memory_admitted = false;
        break;
      }
    }
    size_t free_bytes = 0, total_bytes = 0;
    query_memory("admission", &free_bytes, &total_bytes);
    if (first_peak > std::numeric_limits<size_t>::max() - kExternalMargin ||
        free_bytes < first_peak + kExternalMargin)
      memory_admitted = false;
    if (memory_admitted && options.geom_consistency && shared &&
        shared_geometry && sparse_test_hooks_ &&
        sparse_test_hooks_->refuse_geometry_admission_once) {
      sparse_test_hooks_->refuse_geometry_admission_once = false;
      memory_admitted = false;
    }
    return memory_admitted;
  };
  const auto cleanup_and_readmit = [&](const char* phase, const char* reason) {
    CheckedCudaCleanup([&] {
      for (auto& owner : owners) owner.reset();
      if (shared) shared->Close();
      shared.reset();
    });
    query_memory(phase, &free_bytes, &total_bytes);
    LOG(WARNING) << "PatchMatch setup refused; retrying fewer references: "
                 << reason;
    pair_or_single_fallback();
  };
  bool memory_admitted = false;
  try {
    memory_admitted = admit_remaining();
    if (!memory_admitted && shared_geometry) {
      if (sparse_test_hooks_) ++sparse_test_hooks_->geometry_fallback_entries;
      // Reject geometry sharing independently: keep any qualified source bank
      // while restoring the parent's private alternate geometry owners.
      CheckedCudaCleanup([&] {
        for (auto& owner : owners) owner.reset();
      });
      shared_geometry = false;
      owners[0] = CreateCudaOwner(input[0]->options,
                                  input[0]->problem,
                                  shared,
                                  nullptr,
                                  sparse_failure);
      CUDA_SYNC_AND_CHECK();
      memory_admitted =
          owners[0]->PairPeakBytes() <= first_peak && admit_remaining();
    }
    if (!memory_admitted && shared) {
      // Do not lower concurrency because the optional source bank was rejected.
      // Restore the incumbent independent-source closure only after its parent
      // source-sharing admission has also failed.
      CheckedCudaCleanup([&] {
        for (auto& owner : owners) owner.reset();
        shared->Close();
        shared.reset();
      });
      owners[0] = CreateCudaOwner(input[0]->options, input[0]->problem);
      CUDA_SYNC_AND_CHECK();
      memory_admitted =
          owners[0]->PairPeakBytes() <= first_peak && admit_remaining();
    }
  } catch (const SparseSourceResourceRefused& error) {
    // The first owner can already use sparse views. Retain none of the
    // attached owners, then close the bank before smaller readmission.
    cleanup_and_readmit("after-later-sparse-refusal", error.what());
    return;
  } catch (const OptionalCudaAllocationRefused& error) {
    cleanup_and_readmit("after-optional-cuda-refusal", error.what());
    return;
  }
  query_memory("admission", &free_bytes, &total_bytes);
  if (!memory_admitted) {
    if (first_peak > std::numeric_limits<size_t>::max() - kExternalMargin ||
        free_bytes < first_peak + kExternalMargin)
      CheckedCudaCleanup([&] {
        for (size_t i = 1; i < owners.size(); ++i) owners[i].reset();
      });
    pair_or_single_fallback();
    return;
  }
  std::vector<PatchMatchCuda*> views;
  views.reserve(owners.size());
  for (auto& owner : owners) views.push_back(owner.get());
  std::optional<WorkStreams> streams;
  try {
    streams.emplace(input.size(), /*synchronize_on_close=*/true);
  } catch (const CudaAllocationRefused& error) {
    if (sparse_test_hooks_)
      sparse_test_hooks_->sparse_bank_at_work_stream_refusal =
          shared != nullptr;
    cleanup_and_readmit("after-batch-stream-refusal", error.what());
    return;
  }
  LOG(INFO) << "Running " << input.size() << " reference PatchMatch batch"
            << (shared ? " (sparse source aliases)" : "");
  RunPatchMatchBatch(views, streams->values);
  for (size_t i = 0; i < input.size(); ++i) write_cuda(*input[i], *owners[i]);
  CheckedCudaCleanup([&] {
    for (auto& owner : owners) owner.reset();
    streams.reset();
  });  // Batch streams are destroyed before the shared bank.
  CheckedCudaCleanup([&] {
    if (shared) shared->Close();
    shared.reset();
  });
#endif
}

}  // namespace mvs
}  // namespace colmap
