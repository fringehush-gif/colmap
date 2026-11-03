// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/undistorters.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Reconstruction CreateSyntheticReconstructionWithBitmaps(
    const std::filesystem::path& image_path,
    int num_images = 2,
    int image_width = 100,
    int image_height = 100,
    const std::string& image_extension = ".png",
    const std::vector<double>& camera_params = {}) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = num_images;
  synthetic_dataset_options.camera_width = image_width;
  synthetic_dataset_options.camera_height = image_height;
  if (!camera_params.empty()) {
    synthetic_dataset_options.camera_params = camera_params;
  }
  synthetic_dataset_options.image_extension = image_extension;

  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Create dummy images.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    Bitmap bitmap(image_width, image_height, true);
    bitmap.Fill(BitmapColor<uint8_t>(128, 128, 128));
    EXPECT_TRUE(bitmap.Write(image_path / image.Name()));
  }

  return reconstruction;
}

template <typename Func>
void ExpectOutputOpenFailure(Func&& run, const std::string& filename) {
  try {
    run();
    FAIL() << "Expected output open failure";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr("Could not open"));
    EXPECT_THAT(e.what(), testing::HasSubstr(filename));
  }
}

template <typename Func>
void ExpectCopyFailure(Func&& run, const std::filesystem::path& output) {
  try {
    run();
    FAIL() << "Expected copy failure";
  } catch (const std::filesystem::filesystem_error& e) {
    EXPECT_TRUE(std::filesystem::is_regular_file(e.path1()));
    EXPECT_EQ(e.path2().parent_path(), output);
    EXPECT_TRUE(std::filesystem::is_directory(e.path2()));
  }
}

TEST(COLMAPUndistorter, PropagatesCopyFailure) {
  const auto root = CreateTestDir();
  const auto input = root / "input";
  const auto output = root / "output";
  CreateDirIfNotExists(input);
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(
          input, 2, 16, 16, ".png", {16, 8, 8, 0.05});
  for (const auto& [id, image] : reconstruction.Images()) {
    Bitmap bitmap;
    ASSERT_TRUE(bitmap.Read(input / image.Name()));
  }
  for (const auto& [id, image] : reconstruction.Images()) {
    Camera& camera = reconstruction.Camera(image.CameraId());
    camera.model_id = SimplePinholeCameraModel::model_id;
    camera.params = {16, 8, 8};
    CreateDirIfNotExists(output / "images" / image.Name(), /*recursive=*/true);
  }
  COLMAPUndistorter::Options options;
  options.num_threads = 1;
  COLMAPUndistorter owner(
      options, UndistortCameraOptions(), reconstruction, input, output);
  ExpectCopyFailure([&]() { owner.Run(); }, output / "images");
}

TEST(PMVSUndistorter, PropagatesOutputOpenFailure) {
  const auto root = CreateTestDir();
  const auto input = root / "input";
  const auto output = root / "output";
  CreateDirIfNotExists(input);
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(
          input, 2, 16, 16, ".png", {16, 8, 8, 0.05});
  for (const auto& [id, image] : reconstruction.Images()) {
    Bitmap bitmap;
    ASSERT_TRUE(bitmap.Read(input / image.Name()));
  }
  CreateDirIfNotExists(output / "pmvs" / "txt" / "00000000.txt",
                       /*recursive=*/true);
  PMVSUndistorter::Options options;
  options.num_threads = 1;
  PMVSUndistorter owner(
      options, UndistortCameraOptions(), reconstruction, input, output);
  ExpectOutputOpenFailure([&]() { owner.Run(); }, "00000000.txt");
}

TEST(CMPMVSUndistorter, PropagatesOutputOpenFailure) {
  const auto root = CreateTestDir();
  const auto input = root / "input";
  const auto output = root / "output";
  CreateDirIfNotExists(input);
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(
          input, 2, 16, 16, ".png", {16, 8, 8, 0.05});
  for (const auto& [id, image] : reconstruction.Images()) {
    Bitmap bitmap;
    ASSERT_TRUE(bitmap.Read(input / image.Name()));
  }
  CreateDirIfNotExists(output / "00001_P.txt", /*recursive=*/true);
  CMPMVSUndistorter::Options options;
  options.num_threads = 1;
  CMPMVSUndistorter owner(
      options, UndistortCameraOptions(), reconstruction, input, output);
  ExpectOutputOpenFailure([&]() { owner.Run(); }, "00001_P.txt");
}

TEST(StandaloneImageUndistorter, PropagatesCopyFailure) {
  const auto root = CreateTestDir();
  const auto input = root / "input";
  const auto output = root / "output";
  CreateDirIfNotExists(input);
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(
          input, 2, 16, 16, ".png", {16, 8, 8, 0.05});
  for (const auto& [id, image] : reconstruction.Images()) {
    Bitmap bitmap;
    ASSERT_TRUE(bitmap.Read(input / image.Name()));
  }
  StandaloneImageUndistorter::Options options;
  options.num_threads = 1;
  for (const auto& [id, image] : reconstruction.Images()) {
    Camera& camera = reconstruction.Camera(image.CameraId());
    camera.model_id = SimplePinholeCameraModel::model_id;
    camera.params = {16, 8, 8};
    options.image_names_and_cameras.emplace_back(image.Name(),
                                                 *image.CameraPtr());
    CreateDirIfNotExists(output / image.Name(), /*recursive=*/true);
  }
  StandaloneImageUndistorter owner(
      options, UndistortCameraOptions(), input, output);
  ExpectCopyFailure([&]() { owner.Run(); }, output);
}

TEST(StereoImageRectifier, PropagatesOutputOpenFailure) {
  const auto root = CreateTestDir();
  const auto input = root / "input";
  const auto output = root / "output";
  CreateDirIfNotExists(input);
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(
          input, 2, 16, 16, ".png", {16, 8, 8, 0.05});
  for (const auto& [id, image] : reconstruction.Images()) {
    Bitmap bitmap;
    ASSERT_TRUE(bitmap.Read(input / image.Name()));
  }
  const auto ids = reconstruction.RegImageIds();
  ASSERT_GE(ids.size(), 2);
  const auto pair_name =
      StringReplace(reconstruction.Image(ids[0]).Name(), "/", "-") + "-" +
      StringReplace(reconstruction.Image(ids[1]).Name(), "/", "-");
  CreateDirIfNotExists(output / pair_name / "Q.txt", /*recursive=*/true);
  StereoImageRectifier::Options options;
  options.num_threads = 1;
  options.stereo_pairs.emplace_back(ids[0], ids[1]);
  StereoImageRectifier owner(
      options, UndistortCameraOptions(), reconstruction, input, output);
  ExpectOutputOpenFailure([&]() { owner.Run(); }, "Q.txt");
}

TEST(COLMAPUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run COLMAP undistorter.
  COLMAPUndistorter undistorter(COLMAPUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify output directories were created.
  EXPECT_TRUE(ExistsDir(output_path / "images"));
  EXPECT_TRUE(ExistsDir(output_path / "sparse"));
  EXPECT_TRUE(ExistsDir(output_path / "stereo"));

  // Verify undistorted images were written.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_TRUE(ExistsFile(output_path / "images" / image.Name()));
  }

  // Expect dense reconstruction files to be written.
  EXPECT_TRUE(ExistsFile(output_path / "stereo/patch-match.cfg"));
  EXPECT_TRUE(ExistsFile(output_path / "stereo/fusion.cfg"));
}

TEST(COLMAPUndistorter, StopsPendingWork) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path, /*num_images=*/10);
  COLMAPUndistorter undistorter(COLMAPUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  bool stop_checked = false;
  undistorter.SetCheckIfStoppedFunc([&stop_checked]() {
    stop_checked = true;
    return true;
  });
  undistorter.Run();

  EXPECT_TRUE(stop_checked);
  EXPECT_FALSE(ExistsFile(output_path / "stereo/patch-match.cfg"));
  EXPECT_FALSE(ExistsFile(output_path / "stereo/fusion.cfg"));
}

TEST(COLMAPUndistorter, SpecificImages) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path,
                                               /*num_images=*/2,
                                               /*image_width=*/100,
                                               /*image_height=*/100,
                                               /*image_extension=*/".jpg");

  const Image& image = reconstruction.Image(reconstruction.RegImageIds()[0]);

  // Run COLMAP undistorter.
  COLMAPUndistorter::Options options;
  options.image_ids = {image.ImageId()};
  COLMAPUndistorter undistorter(options,
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify that only the specified image was written.
  EXPECT_THAT(
      GetRecursiveFileList(output_path / "images"),
      testing::UnorderedElementsAre(output_path / "images" / image.Name()));
}

TEST(COLMAPUndistorter, JpegQuality) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path,
                                               /*num_images=*/1,
                                               /*image_width=*/100,
                                               /*image_height=*/100,
                                               /*image_extension=*/".jpg");

  // Run COLMAP undistorter.
  COLMAPUndistorter::Options options;
  options.jpeg_quality = 50;
  COLMAPUndistorter undistorter(options,
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify undistorted images were written.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_TRUE(ExistsFile(output_path / "images" / image.Name()));
  }
}

TEST(PMVSUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "pmvs_output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run PMVS undistorter.
  PMVSUndistorter undistorter(PMVSUndistorter::Options(),
                              UndistortCameraOptions(),
                              reconstruction,
                              image_path,
                              output_path);
  undistorter.Run();

  // Verify PMVS output structure was created (under pmvs/ subdirectory).
  EXPECT_TRUE(ExistsDir(output_path / "pmvs"));
  EXPECT_TRUE(ExistsDir(output_path / "pmvs" / "models"));
  EXPECT_TRUE(ExistsDir(output_path / "pmvs" / "txt"));
  EXPECT_TRUE(ExistsDir(output_path / "pmvs" / "visualize"));

  // Verify undistorted images were written with numbered names.
  // PMVS writes images as 00000000.jpg, 00000001.jpg, etc.
  const size_t num_images = reconstruction.NumRegImages();
  for (size_t i = 0; i < num_images; ++i) {
    const std::string image_name = StringPrintf("%08zu.jpg", i);
    EXPECT_TRUE(ExistsFile(output_path / "pmvs" / "visualize" / image_name));
  }
}

TEST(CMPMVSUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "cmpmvs_output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run CMP-MVS undistorter.
  CMPMVSUndistorter undistorter(CMPMVSUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify CMP-MVS output structure was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify undistorted images were written with sequential numbering.
  // CMP-MVS writes images as 00001.jpg, 00002.jpg, etc.
  const size_t num_images = reconstruction.NumRegImages();
  for (size_t i = 1; i <= num_images; ++i) {
    const std::string image_name = StringPrintf("%05zu.jpg", i);
    EXPECT_TRUE(ExistsFile(output_path / image_name));
  }
}

TEST(StandaloneImageUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "pure_output";
  CreateDirIfNotExists(image_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  StandaloneImageUndistorter::Options options;
  for (const auto& [_, image] : reconstruction.Images()) {
    options.image_names_and_cameras.emplace_back(image.Name(),
                                                 *image.CameraPtr());
  }

  // Run standalone image undistorter.
  StandaloneImageUndistorter undistorter(
      options, UndistortCameraOptions(), image_path, output_path);
  undistorter.Run();

  // Verify output directory was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify undistorted images were written.
  for (const auto& [image_name, camera] : options.image_names_and_cameras) {
    EXPECT_TRUE(ExistsFile(output_path / image_name));
  }
}

TEST(StereoImageRectifier, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "stereo_output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Create stereo pair from first two images.
  StereoImageRectifier::Options options;
  const std::vector<image_t> image_ids = reconstruction.RegImageIds();
  ASSERT_GE(image_ids.size(), 2);
  options.stereo_pairs.emplace_back(image_ids[0], image_ids[1]);

  // Run stereo image rectifier.
  StereoImageRectifier rectifier(options,
                                 UndistortCameraOptions(),
                                 reconstruction,
                                 image_path,
                                 output_path);
  rectifier.Run();

  // Verify output directory was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify rectified images were written.
  // StereoImageRectifier creates a subdirectory for each stereo pair.
  const auto& image1 = reconstruction.Image(options.stereo_pairs[0].first);
  const auto& image2 = reconstruction.Image(options.stereo_pairs[0].second);
  const std::string stereo_pair_name =
      StringPrintf("%s-%s", image1.Name().c_str(), image2.Name().c_str());
  EXPECT_TRUE(ExistsDir(output_path / stereo_pair_name));
  EXPECT_TRUE(ExistsFile(output_path / stereo_pair_name / image1.Name()));
  EXPECT_TRUE(ExistsFile(output_path / stereo_pair_name / image2.Name()));
}

}  // namespace
}  // namespace colmap
