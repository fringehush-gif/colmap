// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/option_manager.h"

#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/feature/sift.h"

#if defined(COLMAP_MVS_ENABLED)
#include "colmap/mvs/fusion.h"
#endif
#include "colmap/mvs/patch_match_options.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(OptionManager, Reset) {
  OptionManager options;
  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  options.AddDatabaseOptions();
  options.AddImageOptions();
  EXPECT_EQ(*options.database_path, "/test/path");
  EXPECT_EQ(*options.image_path, "/test/images");

  options.Reset();

  EXPECT_EQ(*options.database_path, "");
  EXPECT_EQ(*options.image_path, "");
}

TEST(OptionManager, ResetOptions) {
  OptionManager options;
  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  const int original_num_threads = options.feature_extraction->num_threads;
  options.feature_extraction->num_threads = original_num_threads + 42;

  options.ResetOptions(/*reset_paths=*/true);
  EXPECT_EQ(*options.database_path, "");
  EXPECT_EQ(*options.image_path, "");
  EXPECT_EQ(options.feature_extraction->num_threads, original_num_threads);

  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  options.feature_extraction->num_threads = original_num_threads + 42;
  options.ResetOptions(/*reset_paths=*/false);
  EXPECT_EQ(*options.database_path, "/test/path");
  EXPECT_EQ(*options.image_path, "/test/images");
  EXPECT_EQ(options.feature_extraction->num_threads, original_num_threads);
}

TEST(OptionManager, AddOptionsIdempotent) {
  OptionManager options;

  // Adding options multiple times should not cause issues
  options.AddLogOptions();
  options.AddLogOptions();

  options.AddRandomOptions();
  options.AddRandomOptions();

  options.AddFeatureExtractionOptions();
  options.AddFeatureExtractionOptions();

  options.AddFeatureMatchingOptions();
  options.AddFeatureMatchingOptions();

  options.AddMapperOptions();
  options.AddMapperOptions();

  // If idempotency is not maintained, the above would cause errors
  SUCCEED();
}

TEST(OptionManager, AddAllOptions) {
  OptionManager options;
  options.AddAllOptions();

  // Verify that at least some key options are initialized
  EXPECT_NE(options.image_reader, nullptr);
  EXPECT_NE(options.feature_extraction, nullptr);
  EXPECT_NE(options.feature_matching, nullptr);
  EXPECT_NE(options.bundle_adjustment, nullptr);
  EXPECT_NE(options.mapper, nullptr);
#if defined(COLMAP_MVS_ENABLED)
  EXPECT_NE(options.patch_match_stereo, nullptr);
#endif
}

#if defined(COLMAP_MVS_ENABLED)
TEST(OptionManager, PatchMatchSourceUploadRoundTrip) {
  const auto config_path = CreateTestDir() / "source-upload.ini";
  OptionManager writer;
  writer.AddPatchMatchStereoOptions();
  EXPECT_FALSE(writer.patch_match_stereo->direct_source_upload);
  writer.patch_match_stereo->direct_source_upload = true;
  writer.Write(config_path);
  OptionManager reader;
  reader.AddPatchMatchStereoOptions();
  ASSERT_TRUE(reader.Read(config_path));
  EXPECT_TRUE(reader.patch_match_stereo->direct_source_upload);
  reader.ResetOptions(/*reset_paths=*/false);
  EXPECT_FALSE(reader.patch_match_stereo->direct_source_upload);
}

TEST(OptionManager, PatchMatchRotationBufferReuseRoundTrip) {
  const auto config_path = CreateTestDir() / "rotation-buffer-reuse.ini";
  OptionManager writer;
  writer.AddPatchMatchStereoOptions();
  EXPECT_FALSE(writer.patch_match_stereo->rotation_buffer_reuse);
  EXPECT_TRUE(writer.patch_match_stereo->Check());
  writer.patch_match_stereo->rotation_buffer_reuse = true;
#ifdef COLMAP_HIP_ENABLED
  EXPECT_FALSE(writer.patch_match_stereo->Check());
#else
  EXPECT_TRUE(writer.patch_match_stereo->Check());
  writer.Write(config_path);
  OptionManager reader;
  reader.AddPatchMatchStereoOptions();
  ASSERT_TRUE(reader.Read(config_path));
  EXPECT_TRUE(reader.patch_match_stereo->rotation_buffer_reuse);
  reader.ResetOptions(/*reset_paths=*/false);
  EXPECT_FALSE(reader.patch_match_stereo->rotation_buffer_reuse);
#endif
}

TEST(OptionManager, StereoFusionFullRamNormalStorage) {
  using mvs::StereoFusionOptions;
  StereoFusionOptions defaults;
  EXPECT_EQ(defaults.full_ram_normal_storage, "fp32");
  EXPECT_TRUE(defaults.Check());
  for (const auto* unknown : {"",
                              "FP32",
                              "oct",
                              "fp16",
                              "oct2x16-mag",
                              "snorm16x3",
                              "oct2x16-mag16"}) {
    auto invalid = defaults;
    invalid.full_ram_normal_storage = unknown;
    EXPECT_FALSE(invalid.Check());
  }
  for (const auto* name : {"oct2x16"}) {
    SCOPED_TRACE(name);
    auto fusion = defaults;
    fusion.full_ram_normal_storage = name;
    fusion.num_threads = 1;
    fusion.full_ram_load_threads = 4;
    ASSERT_TRUE(fusion.Check());
    OptionManager registered(false);
    registered.AddStereoFusionOptions();
    std::vector<std::string> args{
        "test", "--StereoFusion.full_ram_normal_storage", name};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    ASSERT_TRUE(registered.Parse(static_cast<int>(argv.size()), argv.data()));
    EXPECT_EQ(registered.stereo_fusion->full_ram_normal_storage, name);
    for (const auto conflict :
         {&StereoFusionOptions::use_cache,
          &StereoFusionOptions::bounded_cache,
          &StereoFusionOptions::bounded_cache_normal_snorm16,
          &StereoFusionOptions::bounded_cache_normal_snorm16_guard,
          &StereoFusionOptions::bounded_cache_exact_depth,
          &StereoFusionOptions::bounded_cache_components,
          &StereoFusionOptions::bounded_cache_diagnostics,
          &StereoFusionOptions::bounded_cache_neighborhood,
          &StereoFusionOptions::bounded_cache_compact_rgb}) {
      auto invalid = fusion;
      invalid.*conflict = true;
      EXPECT_FALSE(invalid.Check());
    }
    for (const int conflict : {0, 1, 2, 3}) {
      auto invalid = fusion;
      if (conflict == 0) invalid.bounded_cache_normal_tile_size = 16;
      if (conflict == 1) invalid.bounded_cache_normal_mapped_bytes = 1000;
      if (conflict == 2) invalid.bounded_cache_normal_mapped_files = 1;
      if (conflict == 3) invalid.bounded_cache_prefill_threads = 4;
      EXPECT_FALSE(invalid.Check());
    }
  }
}

TEST(OptionManager, StereoFusionDiagnosticOptionRoundTrip) {
  const auto config_path = CreateTestDir() / "fusion-diagnostic.ini";
  OptionManager writer;
  writer.AddStereoFusionOptions();
  EXPECT_FALSE(writer.stereo_fusion->bounded_cache_diagnostics);
  writer.stereo_fusion->bounded_cache = true;
  writer.stereo_fusion->bounded_cache_diagnostics = true;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddStereoFusionOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_TRUE(reader.stereo_fusion->bounded_cache);
  EXPECT_TRUE(reader.stereo_fusion->bounded_cache_diagnostics);
  reader.ResetOptions(/*reset_paths=*/false);
  EXPECT_FALSE(reader.stereo_fusion->bounded_cache_diagnostics);
}
#endif

TEST(OptionManager, WriteAndRead) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir / "images");

  // Create and configure an OptionManager
  OptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();
  options_write.AddFeatureExtractionOptions();
  options_write.AddMapperOptions();
  options_write.AddGlobalMapperOptions();

  *options_write.database_path = test_dir / "database.db";
  *options_write.image_path = test_dir / "images";
  options_write.feature_extraction->max_image_size = 2048;
  options_write.feature_extraction->sift->max_num_features = 4096;
  options_write.mapper->min_num_matches = 20;

  // Write to file
  options_write.Write(config_path);
  EXPECT_TRUE(ExistsFile(config_path));

  // Read from file
  OptionManager options_read;
  options_read.AddDatabaseOptions();
  options_read.AddImageOptions();
  options_read.AddFeatureExtractionOptions();
  options_read.AddMapperOptions();
  options_read.AddGlobalMapperOptions();

  EXPECT_TRUE(options_read.Read(config_path));

  // Verify that values were read correctly
  EXPECT_EQ(*options_read.database_path, *options_write.database_path);
  EXPECT_EQ(*options_read.image_path, *options_write.image_path);
  EXPECT_EQ(options_read.feature_extraction->max_image_size,
            options_write.feature_extraction->max_image_size);
  EXPECT_EQ(options_read.feature_extraction->sift->max_num_features,
            options_write.feature_extraction->sift->max_num_features);
  EXPECT_EQ(options_read.mapper->min_num_matches,
            options_write.mapper->min_num_matches);
}

TEST(OptionManager, ReRead) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir / "images");

  // Create and write initial config
  OptionManager options_write;
  options_write.AddAllOptions();
  *options_write.database_path = test_dir / "database.db";
  *options_write.image_path = test_dir / "images";
  options_write.feature_extraction->max_image_size = 2048;
  options_write.Write(config_path);

  // Read with ReRead
  OptionManager options_read;
  EXPECT_TRUE(options_read.ReRead(config_path));

  // Verify values
  EXPECT_EQ(*options_read.database_path, *options_write.database_path);
  EXPECT_EQ(*options_read.image_path, *options_write.image_path);
  EXPECT_EQ(options_read.feature_extraction->max_image_size, 2048);
}

TEST(OptionManager, ReadNonExistentFile) {
  OptionManager options;
  options.AddAllOptions();

  EXPECT_FALSE(options.Read("/path/that/does/not/exist.ini"));
}

TEST(OptionManager, Check) {
  const auto test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();

  // Should fail with non-existent paths
  *options.database_path = test_dir / "database.db";
  *options.image_path = "/path/that/does/not/exist";
  EXPECT_FALSE(options.Check());

  // Should succeed with valid paths
  CreateDirIfNotExists(test_dir / "images");
  *options.image_path = test_dir / "images";
  EXPECT_TRUE(options.Check());
}

TEST(OptionManager, CheckDatabaseParentDir) {
  const auto test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();

  // Should succeed when database parent dir exists
  *options.database_path = test_dir / "database.db";
  EXPECT_TRUE(options.Check());

  // Should fail when database path is a directory
  CreateDirIfNotExists(test_dir / "bad_database");
  *options.database_path = test_dir / "bad_database";
  EXPECT_FALSE(options.Check());
}

TEST(OptionManager, ParseWithOptions) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddFeatureExtractionOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  // Create argv with additional options
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
      "--FeatureExtraction.max_image_size",
      "1024",
      "--SiftExtraction.max_num_features",
      "2048",
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Verify parsed values
  EXPECT_EQ(*options.database_path, database_path);
  EXPECT_EQ(*options.image_path, image_path);
  EXPECT_EQ(options.feature_extraction->max_image_size, 1024);
  EXPECT_EQ(options.feature_extraction->sift->max_num_features, 2048);
}

TEST(OptionManager, ParseWithProjectPath) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";
  CreateDirIfNotExists(test_dir / "images");

  // Create and write a config file
  OptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();
  options_write.AddFeatureExtractionOptions();

  *options_write.database_path = test_dir / "database.db";
  *options_write.image_path = test_dir / "images";
  options_write.feature_extraction->max_image_size = 3000;
  options_write.Write(config_path);

  // Parse using project_path
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddFeatureExtractionOptions();

  const std::vector<std::string> args = {
      "colmap",
      "--project_path",
      config_path.string(),
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Verify values were loaded from config file
  EXPECT_EQ(*options.database_path, *options_write.database_path);
  EXPECT_EQ(*options.image_path, *options_write.image_path);
  EXPECT_EQ(options.feature_extraction->max_image_size, 3000);
}

TEST(OptionManager, ParseEmptyArguments) {
  OptionManager options;

  const std::vector<std::string> args = {"colmap"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  // Should succeed with no required options
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));
}

TEST(OptionManager, ParseUnknownArgumentsFails) {
  const auto test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();

  const auto database_path = test_dir / "database.db";

  // Create argv with an unknown option
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--unknown_option",
      "value",
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  // Should return false when encountering unknown option
  EXPECT_FALSE(options.Parse(argv.size(), argv.data()));
}

TEST(OptionManager, WriteAfterResetOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager options;
  options.AddAllOptions();
  *options.database_path = test_dir / "database.db";
  CreateDirIfNotExists(test_dir / "images");
  *options.image_path = test_dir / "images";

  // ResetOptions reassigns option structs, which reallocates sub-objects
  // (e.g., feature_matching->sift, bundle_adjustment->ceres). This must not
  // invalidate the raw pointers registered by AddAllOptions, otherwise
  // Write() will dereference dangling pointers.
  options.ResetOptions(/*reset_paths=*/false);

  EXPECT_NO_THROW(options.Write(config_path));
  EXPECT_TRUE(ExistsFile(config_path));
}

}  // namespace
}  // namespace colmap
