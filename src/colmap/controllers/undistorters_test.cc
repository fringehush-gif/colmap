// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/undistorters.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Reconstruction CreateSyntheticReconstructionWithBitmaps(
    const std::filesystem::path& image_path,
    int num_images = 2,
    int image_width = 100,
    int image_height = 100,
    const std::string& image_extension = ".png") {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = num_images;
  synthetic_dataset_options.camera_width = image_width;
  synthetic_dataset_options.camera_height = image_height;
  synthetic_dataset_options.image_extension = image_extension;

  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Create dummy images.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    Bitmap bitmap(image_width, image_height, true);
    bitmap.Fill(BitmapColor<uint8_t>(128, 128, 128));
    bitmap.Write(image_path / image.Name());
  }

  return reconstruction;
}

TEST(ImageUndistorters, ApplyRegionOfInterestToPinhole) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  CreateDirIfNotExists(image_path);
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path, /*num_images=*/1);
  for (const auto& [camera_id, _] : reconstruction.Cameras()) {
    reconstruction.Camera(camera_id) = Camera::CreateFromModelId(
        camera_id, CameraModelId::kPinhole, 100, 100, 100);
  }
  const Image& image = reconstruction.Images().begin()->second;
  Bitmap input(100, 100, true);
  for (int y = 0; y < 100; ++y) {
    for (int x = 0; x < 100; ++x) {
      input.SetPixel(x,
                     y,
                     BitmapColor<uint8_t>(
                         10 * (x / 5), 10 * (y / 5), 10 * ((x + y) / 10)));
    }
  }
  ASSERT_TRUE(input.Write(image_path / image.Name()));
  {
    // These bytes survive copying, but not decoding and writing the image.
    std::ofstream file(image_path / image.Name(),
                       std::ios::binary | std::ios::app);
    file << "copy-through-marker";
  }
  std::vector<char> input_bytes;
  ReadBinaryBlob(image_path / image.Name(), &input_bytes);

  for (const bool crop : {false, true}) {
    for (const auto copy_type : {FileCopyType::COPY,
                                 FileCopyType::HARD_LINK,
                                 FileCopyType::SOFT_LINK}) {
      // The unchanged control needs no symlink privileges on Windows.
      if (!crop && copy_type != FileCopyType::COPY) {
        continue;
      }
      SCOPED_TRACE(crop);
      SCOPED_TRACE(static_cast<int>(copy_type));
      const auto output_path =
          temp_dir / StringPrintf("output_%d_%d", crop, int(copy_type));
      const auto standalone_path =
          temp_dir / StringPrintf("standalone_%d_%d", crop, int(copy_type));
      CreateDirIfNotExists(output_path);
      UndistortCameraOptions camera_options;
      if (crop) {
        camera_options.roi_min_x = 0.1;
        camera_options.roi_min_y = 0.2;
        camera_options.roi_max_x = 0.9;
        camera_options.roi_max_y = 0.8;
      }
      COLMAPUndistorter::Options options;
      options.num_threads = 1;
      options.copy_type = copy_type;
      COLMAPUndistorter undistorter(
          options, camera_options, reconstruction, image_path, output_path);
      undistorter.Run();
      StandaloneImageUndistorter::Options standalone_options;
      standalone_options.num_threads = 1;
      standalone_options.copy_type = copy_type;
      standalone_options.image_names_and_cameras.emplace_back(
          image.Name(), *image.CameraPtr());
      StandaloneImageUndistorter standalone(
          standalone_options, camera_options, image_path, standalone_path);
      standalone.Run();

      for (const auto& path : {output_path / "images", standalone_path}) {
        if (crop) {
          Bitmap bitmap;
          ASSERT_TRUE(bitmap.Read(path / image.Name()));
          EXPECT_EQ(bitmap.Width(), 80);
          EXPECT_EQ(bitmap.Height(), 60);
          const auto pixel = bitmap.GetPixel(22, 22);
          ASSERT_TRUE(pixel.has_value());
          // Output (22, 22) comes from input (32, 42), not (22, 22).
          EXPECT_EQ(*pixel, BitmapColor<uint8_t>(60, 80, 70));
          EXPECT_FALSE(std::filesystem::equivalent(image_path / image.Name(),
                                                   path / image.Name()));
        } else {
          std::vector<char> output_bytes;
          ReadBinaryBlob(path / image.Name(), &output_bytes);
          EXPECT_EQ(output_bytes, input_bytes);
        }
      }
      Reconstruction output_reconstruction;
      output_reconstruction.Read(output_path / "sparse");
      const Camera& camera = output_reconstruction.Camera(image.CameraId());
      EXPECT_EQ(camera.width, crop ? 80 : 100);
      EXPECT_EQ(camera.height, crop ? 60 : 100);
      EXPECT_EQ(camera.PrincipalPointX(), crop ? 40 : 50);
      EXPECT_EQ(camera.PrincipalPointY(), crop ? 30 : 50);
      std::vector<char> remaining_input_bytes;
      ReadBinaryBlob(image_path / image.Name(), &remaining_input_bytes);
      EXPECT_EQ(remaining_input_bytes, input_bytes);
    }
  }
}

TEST(ImageUndistorters, ThinRegionOfInterestWithMaxImageSize) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  CreateDirIfNotExists(image_path);
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path, /*num_images=*/1);
  for (const auto& [camera_id, _] : reconstruction.Cameras()) {
    reconstruction.Camera(camera_id) = Camera::CreateFromModelId(
        camera_id, CameraModelId::kPinhole, 100, 100, 100);
  }
  Image& image = reconstruction.Image(reconstruction.Images().begin()->first);
  ASSERT_GT(image.NumPoints2D(), 0);
  image.Point2D(0).xy = Eigen::Vector2d(50.5, 50.5);

  for (const bool horizontal : {false, true}) {
    SCOPED_TRACE(horizontal);
    UndistortCameraOptions camera_options;
    camera_options.max_image_size = 40;
    if (horizontal) {
      camera_options.roi_min_y = 0.501;
      camera_options.roi_max_y = 0.502;
    } else {
      camera_options.roi_min_x = 0.501;
      camera_options.roi_max_x = 0.502;
    }
    const auto output_path = temp_dir / StringPrintf("output_%d", horizontal);
    const auto standalone_path =
        temp_dir / StringPrintf("standalone_%d", horizontal);
    CreateDirIfNotExists(output_path);
    COLMAPUndistorter::Options options;
    options.num_threads = 1;
    COLMAPUndistorter undistorter(
        options, camera_options, reconstruction, image_path, output_path);
    ASSERT_NO_THROW(undistorter.Run());

    StandaloneImageUndistorter::Options standalone_options;
    standalone_options.num_threads = 1;
    standalone_options.image_names_and_cameras.emplace_back(image.Name(),
                                                            *image.CameraPtr());
    StandaloneImageUndistorter standalone(
        standalone_options, camera_options, image_path, standalone_path);
    ASSERT_NO_THROW(standalone.Run());

    for (const auto& path : {output_path / "images", standalone_path}) {
      Bitmap bitmap;
      ASSERT_TRUE(bitmap.Read(path / image.Name()));
      EXPECT_EQ(bitmap.Width(), horizontal ? 40 : 1);
      EXPECT_EQ(bitmap.Height(), horizontal ? 1 : 40);
      for (const uint8_t pixel : bitmap.RowMajorData()) {
        EXPECT_GT(pixel, 0);
      }
    }

    Reconstruction output_reconstruction;
    output_reconstruction.Read(output_path / "sparse");
    const Camera& camera = output_reconstruction.Camera(image.CameraId());
    EXPECT_EQ(camera.width, horizontal ? 40 : 1);
    EXPECT_EQ(camera.height, horizontal ? 1 : 40);
    EXPECT_EQ(camera.FocalLengthX(), horizontal ? 40 : 100);
    EXPECT_EQ(camera.FocalLengthY(), horizontal ? 100 : 40);
    EXPECT_EQ(camera.PrincipalPointX(), horizontal ? 20 : 0);
    EXPECT_EQ(camera.PrincipalPointY(), horizontal ? 0 : 20);
    const Image& output_image = output_reconstruction.Image(image.ImageId());
    ASSERT_EQ(output_image.NumPoints2D(), image.NumPoints2D());
    EXPECT_NEAR(output_image.Point2D(0).xy.x(), horizontal ? 20.2 : 0.5, 1e-12);
    EXPECT_NEAR(output_image.Point2D(0).xy.y(), horizontal ? 0.5 : 20.2, 1e-12);
  }
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
