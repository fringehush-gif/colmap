// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/undistortion.h"

#include "colmap/geometry/pose.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Bitmap CreateCheckerboardBitmap(const int image_size, const int cell_size) {
  Bitmap image(image_size, image_size, false);
  for (int y = 0; y < image.Height(); ++y) {
    for (int x = 0; x < image.Width(); ++x) {
      image.SetPixel(x,
                     y,
                     BitmapColor<uint8_t>(static_cast<uint8_t>(
                         (x / cell_size + y / cell_size) % 2 == 0 ? 0 : 255)));
    }
  }
  return image;
}

uint64_t TotalAbsoluteDifference(const Bitmap& bitmap1, const Bitmap& bitmap2) {
  const std::vector<uint8_t>& data1 = bitmap1.RowMajorData();
  const std::vector<uint8_t>& data2 = bitmap2.RowMajorData();
  THROW_CHECK_EQ(data1.size(), data2.size());
  uint64_t total_absolute_difference = 0;
  for (size_t i = 0; i < data1.size(); ++i) {
    total_absolute_difference +=
        std::abs(static_cast<int>(data1[i]) - static_cast<int>(data2[i]));
  }
  return total_absolute_difference;
}

double MeanAbsoluteDifference(const Bitmap& bitmap1, const Bitmap& bitmap2) {
  THROW_CHECK_GT(bitmap1.RowMajorData().size(), 0);
  return static_cast<double>(TotalAbsoluteDifference(bitmap1, bitmap2)) /
         bitmap1.RowMajorData().size();
}

int MaxAbsoluteDifference(const Bitmap& bitmap1, const Bitmap& bitmap2) {
  const std::vector<uint8_t>& data1 = bitmap1.RowMajorData();
  const std::vector<uint8_t>& data2 = bitmap2.RowMajorData();
  THROW_CHECK_EQ(data1.size(), data2.size());
  int max_absolute_difference = 0;
  for (size_t i = 0; i < data1.size(); ++i) {
    max_absolute_difference = std::max(
        max_absolute_difference,
        std::abs(static_cast<int>(data1[i]) - static_cast<int>(data2[i])));
  }
  return max_absolute_difference;
}

TEST(UndistortCamera, Nominal) {
  UndistortCameraOptions options;
  Camera distorted_camera;
  Camera undistorted_camera;

  distorted_camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 1);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 1);
  EXPECT_EQ(undistorted_camera.width, 1);
  EXPECT_EQ(undistorted_camera.height, 1);

  distorted_camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimpleRadial, 1, 1, 1);
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 1);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 1);
  EXPECT_EQ(undistorted_camera.width, 1);
  EXPECT_EQ(undistorted_camera.height, 1);

  distorted_camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimpleRadial, 100, 100, 100);
  distorted_camera.params[3] = 0.5;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.width, 84);
  EXPECT_EQ(undistorted_camera.height, 84);

  options.blank_pixels = 1;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.width, 90);
  EXPECT_EQ(undistorted_camera.height, 90);

  options.max_scale = 0.75;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.width, 75);
  EXPECT_EQ(undistorted_camera.height, 75);

  options.max_scale = 1.0;
  options.roi_min_x = 0.1;
  options.roi_min_y = 0.2;
  options.roi_max_x = 0.9;
  options.roi_max_y = 0.8;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  Camera cropped_camera = distorted_camera;
  cropped_camera.width = 80;
  cropped_camera.height = 60;
  cropped_camera.SetPrincipalPointX(40);
  cropped_camera.SetPrincipalPointY(30);
  options.roi_min_x = options.roi_min_y = 0.0;
  options.roi_max_x = options.roi_max_y = 1.0;
  const Camera expected_camera = UndistortCamera(options, cropped_camera);
  EXPECT_EQ(undistorted_camera.width, expected_camera.width);
  EXPECT_EQ(undistorted_camera.height, expected_camera.height);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(),
            expected_camera.PrincipalPointX());
  EXPECT_EQ(undistorted_camera.PrincipalPointY(),
            expected_camera.PrincipalPointY());
}

TEST(UndistortCamera, PinholeRegionOfInterest) {
  UndistortCameraOptions options;
  options.roi_min_x = 0.1;
  options.roi_min_y = 0.2;
  options.roi_max_x = 0.9;
  options.roi_max_y = 0.8;
  for (const CameraModelId model : {CameraModelId::kSimplePinhole,
                                    CameraModelId::kPinhole,
                                    CameraModelId::kSimpleRadial,
                                    CameraModelId::kRadial,
                                    CameraModelId::kOpenCV,
                                    CameraModelId::kFullOpenCV,
                                    CameraModelId::kFOV,
                                    CameraModelId::kSimpleDivision,
                                    CameraModelId::kDivision,
                                    CameraModelId::kSkewedPinhole,
                                    CameraModelId::kEUCM}) {
    SCOPED_TRACE(static_cast<int>(model));
    Camera camera =
        Camera::CreateFromModelId(1, model, 100, 100, 100, CameraSource::EXIF);
    if (model == CameraModelId::kFOV) {
      camera.params[4] = 0;
    } else if (model == CameraModelId::kEUCM) {
      // A neutral alpha is independent of the valid positive beta value.
      camera.params[5] = 2;
    }
    const Camera cropped = UndistortCamera(options, camera);
    EXPECT_EQ(cropped.source, camera.source);
    EXPECT_TRUE(cropped.HasPriorFocalLength());
    EXPECT_EQ(cropped.width, 80);
    EXPECT_EQ(cropped.height, 60);
    EXPECT_EQ(cropped.PrincipalPointX(), 40);
    EXPECT_EQ(cropped.PrincipalPointY(), 30);
    EXPECT_EQ(cropped.FocalLengthX(), 100);
    EXPECT_EQ(cropped.FocalLengthY(), 100);
  }
}

TEST(UndistortCamera, NeutralPinholeScaleOptions) {
  for (const CameraModelId model : {CameraModelId::kPinhole,
                                    CameraModelId::kSimpleRadial,
                                    CameraModelId::kEUCM}) {
    SCOPED_TRACE(static_cast<int>(model));
    const Camera camera = Camera::CreateFromModelId(1, model, 100, 100, 80);
    for (const int max_image_size : {-1, 50}) {
      SCOPED_TRACE(max_image_size);
      UndistortCameraOptions options;
      options.min_scale = 2;
      options.max_scale = 3;
      options.max_image_size = max_image_size;
      const Camera output = UndistortCamera(options, camera);
      const double scale = max_image_size < 0 ? 1.0 : 0.5;
      EXPECT_EQ(output.width, max_image_size < 0 ? 100 : 50);
      EXPECT_EQ(output.height, max_image_size < 0 ? 80 : 40);
      EXPECT_EQ(output.FocalLengthX(), 100 * scale);
      EXPECT_EQ(output.FocalLengthY(), 100 * scale);
      EXPECT_EQ(output.PrincipalPointX(), 50 * scale);
      EXPECT_EQ(output.PrincipalPointY(), 40 * scale);
    }
  }
}

TEST(UndistortCamera, RegionOfInterestRoundingAndSizeLimit) {
  const Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 100, 100, 100);
  UndistortCameraOptions options;
  options.roi_min_x = 0.1;
  options.roi_min_y = 0.2;
  options.roi_max_x = 0.9;
  options.roi_max_y = 0.8;
  options.max_image_size = 40;
  const Camera limited = UndistortCamera(options, camera);
  EXPECT_EQ(limited.width, 40);
  EXPECT_EQ(limited.height, 30);
  EXPECT_EQ(limited.FocalLengthX(), 50);
  EXPECT_EQ(limited.FocalLengthY(), 50);
  EXPECT_EQ(limited.PrincipalPointX(), 20);
  EXPECT_EQ(limited.PrincipalPointY(), 15);

  // Crops need not include the principal point.
  options.max_image_size = -1;
  options.roi_min_x = options.roi_min_y = 0.8;
  options.roi_max_x = options.roi_max_y = 0.9;
  const Camera off_center = UndistortCamera(options, camera);
  EXPECT_EQ(off_center.width, 10);
  EXPECT_EQ(off_center.height, 10);
  EXPECT_EQ(off_center.PrincipalPointX(), -30);
  EXPECT_EQ(off_center.PrincipalPointY(), -30);

  // A positive normalized crop that rounds to one pixel boundary stays
  // nonempty.
  options.roi_min_x = options.roi_min_y = 0.501;
  options.roi_max_x = options.roi_max_y = 0.502;
  const Camera tiny = UndistortCamera(options, camera);
  EXPECT_EQ(tiny.width, 1);
  EXPECT_EQ(tiny.height, 1);
  EXPECT_EQ(tiny.PrincipalPointX(), 0);
  EXPECT_EQ(tiny.PrincipalPointY(), 0);
}

TEST(UndistortCamera, RegionOfInterestMatchesVirtualCrop) {
  for (const CameraModelId model : {CameraModelId::kSimpleRadial,
                                    CameraModelId::kOpenCVFisheye,
                                    CameraModelId::kSimpleFisheye}) {
    Camera camera = Camera::CreateFromModelId(1, model, 400, 400, 300);
    if (!camera.ExtraParamsIdxs().empty()) {
      camera.params[camera.ExtraParamsIdxs()[0]] = 0.05;
    }
    camera.SetPrincipalPointX(210);
    camera.SetPrincipalPointY(130);
    Camera cropped = camera;
    cropped.width = 280;
    cropped.height = 210;
    cropped.SetPrincipalPointX(170);
    cropped.SetPrincipalPointY(70);
    for (const double blank_pixels : {0.0, 1.0}) {
      UndistortCameraOptions options;
      options.blank_pixels = blank_pixels;
      options.roi_min_x = 0.1;
      options.roi_min_y = 0.2;
      options.roi_max_x = 0.8;
      options.roi_max_y = 0.9;
      const Camera actual = UndistortCamera(options, camera);
      options.roi_min_x = options.roi_min_y = 0.0;
      options.roi_max_x = options.roi_max_y = 1.0;
      const Camera expected = UndistortCamera(options, cropped);
      EXPECT_EQ(actual.width, expected.width);
      EXPECT_EQ(actual.height, expected.height);
      EXPECT_NEAR(actual.PrincipalPointX(), expected.PrincipalPointX(), 1e-10);
      EXPECT_NEAR(actual.PrincipalPointY(), expected.PrincipalPointY(), 1e-10);
      EXPECT_EQ(actual.FocalLengthX(), expected.FocalLengthX());
      EXPECT_EQ(actual.FocalLengthY(), expected.FocalLengthY());
    }
  }
}

TEST(UndistortCamera, MaxImageSizeKeepsPositiveDimensions) {
  struct TestCase {
    size_t width;
    size_t height;
    int max_image_size;
    size_t expected_width;
    size_t expected_height;
  };
  const TestCase test_cases[] = {{1, 100, 40, 1, 40},
                                 {100, 1, 40, 40, 1},
                                 {1, 100, 1, 1, 1},
                                 {100, 1, 1, 1, 1},
                                 {1, 1, 1, 1, 1},
                                 {51, 101, 40, 20, 40},
                                 {101, 51, 40, 40, 20}};
  for (const auto& test : test_cases) {
    SCOPED_TRACE(test.width);
    SCOPED_TRACE(test.height);
    SCOPED_TRACE(test.max_image_size);
    UndistortCameraOptions options;
    options.max_image_size = test.max_image_size;
    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kPinhole, 100, test.width, test.height);
    const Camera resized = UndistortCamera(options, camera);
    EXPECT_EQ(resized.width, test.expected_width);
    EXPECT_EQ(resized.height, test.expected_height);
    EXPECT_NEAR(resized.FocalLengthX(),
                100.0 * test.expected_width / test.width,
                1e-12);
    EXPECT_NEAR(resized.FocalLengthY(),
                100.0 * test.expected_height / test.height,
                1e-12);
    EXPECT_EQ(resized.PrincipalPointX(), test.expected_width / 2.0);
    EXPECT_EQ(resized.PrincipalPointY(), test.expected_height / 2.0);

    // Nearest-neighbor interpolation also supports a one-pixel source axis.
    options.warp_options.interpolation =
        WarpImageOptions::Interpolation::kNearestNeighbor;
    Bitmap input(test.width, test.height, false);
    input.Fill(BitmapColor<uint8_t>(255));
    for (const CameraModelId model :
         {CameraModelId::kPinhole, CameraModelId::kEquirectangular}) {
      SCOPED_TRACE(static_cast<int>(model));
      Bitmap output;
      Camera output_camera;
      ASSERT_NO_THROW(UndistortImage(
          options,
          input,
          Camera::CreateFromModelId(1, model, 100, test.width, test.height),
          &output,
          &output_camera));
      EXPECT_EQ(output.Width(), test.expected_width);
      EXPECT_EQ(output.Height(), test.expected_height);
      EXPECT_EQ(output_camera.width, test.expected_width);
      EXPECT_EQ(output_camera.height, test.expected_height);
      EXPECT_EQ(output_camera.model_id, model);
      EXPECT_EQ(output.GetPixel(0, 0), BitmapColor<uint8_t>(255));
    }
  }
}

TEST(UndistortImage, ThinRegionOfInterestWithMaxImageSize) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kPinhole, 100, 100, 100, CameraSource::EXIF);
  Bitmap input(100, 100, false);
  input.Fill(BitmapColor<uint8_t>(255));
  for (const bool horizontal : {false, true}) {
    SCOPED_TRACE(horizontal);
    UndistortCameraOptions options;
    options.max_image_size = 40;
    if (horizontal) {
      options.roi_min_y = 0.501;
      options.roi_max_y = 0.502;
    } else {
      options.roi_min_x = 0.501;
      options.roi_max_x = 0.502;
    }
    Bitmap output;
    Camera output_camera;
    ASSERT_NO_THROW(
        UndistortImage(options, input, camera, &output, &output_camera));
    EXPECT_EQ(output.Width(), horizontal ? 40 : 1);
    EXPECT_EQ(output.Height(), horizontal ? 1 : 40);
    EXPECT_EQ(output_camera.width, horizontal ? 40 : 1);
    EXPECT_EQ(output_camera.height, horizontal ? 1 : 40);
    EXPECT_EQ(output_camera.FocalLengthX(), horizontal ? 40 : 100);
    EXPECT_EQ(output_camera.FocalLengthY(), horizontal ? 100 : 40);
    EXPECT_EQ(output_camera.PrincipalPointX(), horizontal ? 20 : 0);
    EXPECT_EQ(output_camera.PrincipalPointY(), horizontal ? 0 : 20);
    EXPECT_EQ(output_camera.source, camera.source);
    EXPECT_TRUE(output_camera.HasPriorFocalLength());
    // Keep the default bilinear/intermediate-warp path for a normal input
    // image. Every output pixel must retain a valid source contribution.
    for (const uint8_t pixel : output.RowMajorData()) {
      EXPECT_GT(pixel, 0);
    }
  }
}

TEST(UndistortImage, PinholeCropPreservesPixels) {
  struct TestCase {
    CameraModelId model;
    int image_size;
    int crop_offset;
    int crop_size;
    int max_difference;
  };
  const TestCase test_cases[] = {
      {CameraModelId::kPinhole, 128, 48, 32, 0},
      {CameraModelId::kSimplePinhole, 128, 48, 32, 0},
      {CameraModelId::kSimpleRadial, 128, 48, 32, 0},
      {CameraModelId::kRadial, 128, 48, 32, 0},
      {CameraModelId::kOpenCV, 128, 48, 32, 0},
      {CameraModelId::kFullOpenCV, 128, 48, 32, 0},
      {CameraModelId::kFOV, 128, 48, 32, 0},
      {CameraModelId::kSimpleDivision, 128, 48, 32, 0},
      {CameraModelId::kDivision, 128, 48, 32, 0},
      {CameraModelId::kSkewedPinhole, 128, 48, 32, 0},
      {CameraModelId::kEUCM, 128, 48, 32, 0},
      {CameraModelId::kPinhole, 100, 45, 10, 1},
      {CameraModelId::kSimplePinhole, 100, 45, 10, 1}};
  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(test_case.image_size);
    SCOPED_TRACE(static_cast<int>(test_case.model));
    Camera camera = Camera::CreateFromModelId(1,
                                              test_case.model,
                                              test_case.image_size,
                                              test_case.image_size,
                                              test_case.image_size);
    if (test_case.model == CameraModelId::kFOV) {
      camera.params[4] = 0;
    }
    const Bitmap input = CreateCheckerboardBitmap(test_case.image_size, 1);
    Bitmap expected = input.Clone();
    expected.Crop(test_case.crop_offset,
                  test_case.crop_offset,
                  test_case.crop_size,
                  test_case.crop_size);

    for (const auto interpolation :
         {WarpImageOptions::Interpolation::kNearestNeighbor,
          WarpImageOptions::Interpolation::kBilinear}) {
      SCOPED_TRACE(static_cast<int>(interpolation));
      const int max_difference =
          interpolation == WarpImageOptions::Interpolation::kNearestNeighbor
              ? 0
              : test_case.max_difference;
      UndistortCameraOptions options;
      options.warp_options.interpolation = interpolation;
      options.roi_min_x = options.roi_min_y =
          static_cast<double>(test_case.crop_offset) / test_case.image_size;
      options.roi_max_x = options.roi_max_y =
          static_cast<double>(test_case.crop_offset + test_case.crop_size) /
          test_case.image_size;
      Bitmap output;
      Camera output_camera;
      UndistortImage(options, input, camera, &output, &output_camera);
      ASSERT_EQ(output.Width(), test_case.crop_size);
      ASSERT_EQ(output.Height(), test_case.crop_size);
      EXPECT_EQ(output_camera.FocalLengthX(), camera.FocalLengthX());
      EXPECT_EQ(output_camera.FocalLengthY(), camera.FocalLengthY());
      EXPECT_EQ(output_camera.PrincipalPointX(),
                camera.PrincipalPointX() - test_case.crop_offset);
      EXPECT_EQ(output_camera.PrincipalPointY(),
                camera.PrincipalPointY() - test_case.crop_offset);

      options.warp_options.direct_warp_min_scale = 0;
      Bitmap direct_output;
      Camera direct_camera;
      UndistortImage(options, input, camera, &direct_output, &direct_camera);
      ASSERT_EQ(direct_camera, output_camera);
      const int direct_difference =
          MaxAbsoluteDifference(direct_output, expected);
      ASSERT_LE(direct_difference, max_difference);
      EXPECT_LE(MaxAbsoluteDifference(output, expected), max_difference)
          << "direct maximum difference: " << direct_difference
          << "; default mean difference: "
          << MeanAbsoluteDifference(output, expected);

      options.warp_options.direct_warp_min_scale = 1;
      UndistortImage(options, input, camera, &output, &output_camera);
      EXPECT_EQ(output_camera, direct_camera);
      EXPECT_LE(MaxAbsoluteDifference(output, expected), max_difference)
          << "direct_warp_min_scale = 1";
    }
  }
}

TEST(UndistortImage, PinholeCropRetainsFilteredWarpOptions) {
  const Bitmap input = CreateCheckerboardBitmap(128, 2);
  for (const CameraModelId model : {CameraModelId::kPinhole,
                                    CameraModelId::kSimplePinhole,
                                    CameraModelId::kSimpleRadial,
                                    CameraModelId::kEUCM}) {
    SCOPED_TRACE(static_cast<int>(model));
    const Camera camera = Camera::CreateFromModelId(1, model, 128, 128, 128);
    for (const bool downscale : {false, true}) {
      SCOPED_TRACE(downscale);
      UndistortCameraOptions options;
      options.roi_min_x = options.roi_min_y = 0.375;
      options.roi_max_x = options.roi_max_y = 0.625;
      if (downscale) {
        options.max_image_size = 16;
      } else {
        options.warp_options.direct_warp_min_scale = 2;
      }
      Bitmap output;
      Camera output_camera;
      UndistortImage(options, input, camera, &output, &output_camera);
      EXPECT_EQ(output_camera.width, downscale ? 16 : 32);
      EXPECT_EQ(output_camera.height, downscale ? 16 : 32);
      EXPECT_EQ(output_camera.FocalLengthX(), downscale ? 64 : 128);
      EXPECT_EQ(output_camera.FocalLengthY(), downscale ? 64 : 128);
      Bitmap expected;
      WarpImageBetweenCameras(
          options.warp_options, camera, output_camera, input, &expected);
      EXPECT_EQ(output.RowMajorData(), expected.RowMajorData());

      options.warp_options.direct_warp_min_scale = 0;
      Bitmap direct;
      WarpImageBetweenCameras(
          options.warp_options, camera, output_camera, input, &direct);
      EXPECT_NE(expected.RowMajorData(), direct.RowMajorData());
    }
  }
}

TEST(UndistortImage, NonNeutralCropRetainsFilteredWarp) {
  const Bitmap input = CreateCheckerboardBitmap(128, 2);
  struct TestCase {
    CameraModelId model;
    double first_extra_param;
  };
  const TestCase test_cases[] = {{CameraModelId::kSimpleRadial, 1e-9},
                                 {CameraModelId::kSimpleRadial, 0.05},
                                 {CameraModelId::kFOV, 0.01},
                                 {CameraModelId::kEUCM, 0.1},
                                 {CameraModelId::kOpenCVFisheye, 0.0}};
  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(static_cast<int>(test_case.model));
    SCOPED_TRACE(test_case.first_extra_param);
    Camera camera =
        Camera::CreateFromModelId(1, test_case.model, 128, 128, 128);
    camera.params[camera.ExtraParamsIdxs()[0]] = test_case.first_extra_param;
    UndistortCameraOptions options;
    options.roi_min_x = options.roi_min_y = 0.375;
    options.roi_max_x = options.roi_max_y = 0.625;
    Bitmap output;
    Camera output_camera;
    UndistortImage(options, input, camera, &output, &output_camera);
    Bitmap expected;
    WarpImageBetweenCameras(
        options.warp_options, camera, output_camera, input, &expected);
    EXPECT_EQ(output.RowMajorData(), expected.RowMajorData());

    options.warp_options.direct_warp_min_scale = 0;
    Bitmap direct;
    WarpImageBetweenCameras(
        options.warp_options, camera, output_camera, input, &direct);
    EXPECT_NE(expected.RowMajorData(), direct.RowMajorData());
  }
}

TEST(UndistortImage, PinholeCropRejectsInvalidWarpOptions) {
  const Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 128, 128, 128);
  const Bitmap input(128, 128, false);
  UndistortCameraOptions options;
  options.roi_min_x = options.roi_min_y = 0.375;
  options.roi_max_x = options.roi_max_y = 0.625;
  Bitmap output;
  Camera output_camera;
  for (const double threshold :
       {-0.1, std::numeric_limits<double>::quiet_NaN()}) {
    options.warp_options.direct_warp_min_scale = threshold;
    EXPECT_THROW(
        UndistortImage(options, input, camera, &output, &output_camera),
        std::invalid_argument);
  }
}

TEST(UndistortImage, RejectsZeroMaxImageSize) {
  UndistortCameraOptions options;
  options.max_image_size = 0;
  Bitmap input(10, 10, false);
  Bitmap output;
  Camera output_camera;
  for (const CameraModelId model :
       {CameraModelId::kPinhole, CameraModelId::kEquirectangular}) {
    SCOPED_TRACE(static_cast<int>(model));
    const Camera camera = Camera::CreateFromModelId(1, model, 100, 10, 10);
    EXPECT_THROW(
        UndistortImage(options, input, camera, &output, &output_camera),
        std::invalid_argument);
  }
}

TEST(UndistortCamera, MaxCamPointNorm) {
  // Fisheye camera with off-center principal point: pixels far from (cx, cy)
  // have theta close to pi/2, so CamFromImg returns very large values
  // (|cam_point| = tan(theta)) that blow up the output dimensions.
  Camera distorted_camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimpleFisheye, 130, 200, 100);
  distorted_camera.SetPrincipalPointX(10);
  distorted_camera.SetPrincipalPointY(50);

  UndistortCameraOptions options;
  // Exercise the path driven by extreme samples.
  options.blank_pixels = 1.0;

  // Default (max_cam_point_norm = -1): the extreme cam_points push the scale
  // factor against max_scale, so the output is the maximum allowed size.
  const Camera undistorted_camera_unbounded =
      UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera_unbounded.width,
            distorted_camera.width * options.max_scale);
  EXPECT_EQ(undistorted_camera_unbounded.height,
            distorted_camera.height * options.max_scale);

  // With a finite threshold, extreme border samples are skipped, yielding
  // a smaller output that is no longer hitting the max_scale clamp.
  options.max_cam_point_norm = 2.0;
  const Camera undistorted_camera_bounded =
      UndistortCamera(options, distorted_camera);
  EXPECT_LT(undistorted_camera_bounded.width,
            undistorted_camera_unbounded.width);
  EXPECT_LT(undistorted_camera_bounded.height,
            undistorted_camera_unbounded.height);

  // max_cam_point_norm = 0 is invalid (would skip every sample).
  options.max_cam_point_norm = 0;
  EXPECT_ANY_THROW(UndistortCamera(options, distorted_camera));
}

TEST(UndistortCamera, BlankPixels) {
  UndistortCameraOptions options;
  options.blank_pixels = 1;

  Camera distorted_camera;
  distorted_camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimpleRadial, 100, 100, 100);
  distorted_camera.params[3] = 0.5;

  Bitmap distorted_image(100, 100, false);
  distorted_image.Fill(BitmapColor<uint8_t>(255));

  Bitmap undistorted_image;
  Camera undistorted_camera;
  UndistortImage(options,
                 distorted_image,
                 distorted_camera,
                 &undistorted_image,
                 &undistorted_camera);

  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 90.0 / 2.0);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 90.0 / 2.0);
  EXPECT_EQ(undistorted_camera.width, 90);
  EXPECT_EQ(undistorted_camera.height, 90);

  // Make sure that there is no blank pixel.
  size_t num_blank_pixels = 0;
  for (int y = 0; y < undistorted_image.Height(); ++y) {
    for (int x = 0; x < undistorted_image.Width(); ++x) {
      const auto color = undistorted_image.GetPixel(x, y);
      ASSERT_TRUE(color.has_value());
      if (*color == BitmapColor<uint8_t>(0)) {
        num_blank_pixels += 1;
      }
    }
  }

  EXPECT_GT(num_blank_pixels, 0);
}

TEST(UndistortCamera, NoBlankPixels) {
  UndistortCameraOptions options;
  options.blank_pixels = 0;

  Camera distorted_camera;
  distorted_camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimpleRadial, 100, 100, 100);
  distorted_camera.params[3] = 0.5;

  Bitmap distorted_image(100, 100, false);
  distorted_image.Fill(BitmapColor<uint8_t>(255));

  Bitmap undistorted_image;
  Camera undistorted_camera;
  UndistortImage(options,
                 distorted_image,
                 distorted_camera,
                 &undistorted_image,
                 &undistorted_camera);

  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.width, 84);
  EXPECT_EQ(undistorted_camera.height, 84);

  // Make sure that there is no blank pixel.
  for (int y = 0; y < undistorted_image.Height(); ++y) {
    for (int x = 0; x < undistorted_image.Width(); ++x) {
      const auto color = undistorted_image.GetPixel(x, y);
      ASSERT_TRUE(color.has_value());
      ASSERT_NE(color->r, 0);
      ASSERT_NE(color->g, 0);
      ASSERT_NE(color->b, 0);
    }
  }
}

TEST(UndistortImage, WarpOptions) {
  Camera distorted_camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimpleRadial, 100, 100, 100);
  distorted_camera.params[3] = 0.5;

  Bitmap distorted_image(100, 100, true);
  for (int y = 0; y < distorted_image.Height(); ++y) {
    for (int x = 0; x < distorted_image.Width(); ++x) {
      distorted_image.SetPixel(
          x,
          y,
          BitmapColor<uint8_t>(static_cast<uint8_t>(x),
                               static_cast<uint8_t>(y),
                               static_cast<uint8_t>((x + y) / 2)));
    }
  }

  UndistortCameraOptions options;
  Bitmap direct_image;
  Camera direct_camera;
  UndistortImage(options,
                 distorted_image,
                 distorted_camera,
                 &direct_image,
                 &direct_camera);
  const double direct_scale =
      std::min(static_cast<double>(direct_camera.width) /
                   static_cast<double>(distorted_camera.width),
               static_cast<double>(direct_camera.height) /
                   static_cast<double>(distorted_camera.height));
  ASSERT_GE(direct_scale, options.warp_options.direct_warp_min_scale);

  options.warp_options.direct_warp_min_scale = 1.0;
  Bitmap resized_image;
  Camera resized_camera;
  UndistortImage(options,
                 distorted_image,
                 distorted_camera,
                 &resized_image,
                 &resized_camera);

  EXPECT_EQ(direct_camera, resized_camera);
  EXPECT_EQ(direct_image.Width(), resized_image.Width());
  EXPECT_EQ(direct_image.Height(), resized_image.Height());
  EXPECT_NE(direct_image.RowMajorData(), resized_image.RowMajorData());

  // The extra resize changes rounding, but not by a perceptible amount for
  // smooth image content.
  EXPECT_LT(MeanAbsoluteDifference(direct_image, resized_image), 0.5);
  EXPECT_LE(MaxAbsoluteDifference(direct_image, resized_image), 1);
}

TEST(UndistortReconstruction, Nominal) {
  const size_t kNumImages = 10;
  const size_t kNumPoints2D = 10;

  Reconstruction reconstruction;

  Camera camera = Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 1, 1, 1);
  camera.params[4] = 1.0;
  reconstruction.AddCamera(camera);
  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  reconstruction.AddRig(rig);

  for (image_t image_id = 1; image_id <= kNumImages; ++image_id) {
    Frame frame;
    frame.SetRigId(1);
    frame.SetFrameId(image_id);
    frame.SetRigFromWorld(Rigid3d());
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetFrameId(frame.FrameId());
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Ones()));
    frame.AddDataId(image.DataId());
    reconstruction.AddFrame(frame);
    reconstruction.AddImage(image);
    reconstruction.RegisterFrame(frame.FrameId());
  }

  UndistortCameraOptions options;
  UndistortReconstruction(options, &reconstruction);
  for (const auto& camera : reconstruction.Cameras()) {
    EXPECT_EQ(camera.second.ModelName(), "PINHOLE");
  }

  for (const auto& image : reconstruction.Images()) {
    for (const auto& point2D : image.second.Points2D()) {
      EXPECT_NE(point2D.xy, Eigen::Vector2d::Ones());
    }
  }
}

TEST(UndistortReconstruction, RescalesAlreadyUndistortedCameras) {
  constexpr size_t kNumImages = 3;

  Reconstruction reconstruction;

  // Already-undistorted (PINHOLE) camera at a high resolution. focal=100,
  // width=height=100 => principal point at (50, 50).
  Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kPinhole, 100, 100, 100, CameraSource::USER);
  ASSERT_TRUE(camera.IsUndistorted());
  reconstruction.AddCamera(camera);
  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  reconstruction.AddRig(rig);

  const Eigen::Vector2d point_xy(60, 40);
  for (image_t image_id = 1; image_id <= kNumImages; ++image_id) {
    Frame frame;
    frame.SetRigId(1);
    frame.SetFrameId(image_id);
    frame.SetRigFromWorld(Rigid3d());
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetFrameId(frame.FrameId());
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(std::vector<Eigen::Vector2d>(1, point_xy));
    frame.AddDataId(image.DataId());
    reconstruction.AddFrame(frame);
    reconstruction.AddImage(image);
    reconstruction.RegisterFrame(frame.FrameId());
  }

  Reconstruction unchanged = reconstruction;
  UndistortReconstruction(UndistortCameraOptions(), &unchanged);
  EXPECT_EQ(unchanged.Camera(1).width, 100);
  EXPECT_EQ(unchanged.Camera(1).height, 100);
  ASSERT_EQ(unchanged.NumImages(), kNumImages);
  for (const auto& [image_id, image] : unchanged.Images()) {
    ASSERT_EQ(image.NumPoints2D(), 1);
    EXPECT_THAT(image.Point2D(0).xy, EigenMatrixNear(point_xy, 1e-10));
  }

  Reconstruction cropped = reconstruction;
  // The crop shifts every observation, including those outside its bounds.
  // It does not remove observations or filter tracks.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    cropped.Image(image_id).SetPoints2D(
        std::vector<Eigen::Vector2d>{point_xy, Eigen::Vector2d(0, 5)});
  }
  UndistortCameraOptions roi_options;
  roi_options.roi_min_x = 0.1;
  roi_options.roi_min_y = 0.2;
  roi_options.roi_max_x = 0.9;
  roi_options.roi_max_y = 0.8;
  UndistortReconstruction(roi_options, &cropped);
  EXPECT_EQ(cropped.Camera(1).width, 80);
  EXPECT_EQ(cropped.Camera(1).height, 60);
  EXPECT_EQ(cropped.Camera(1).PrincipalPointX(), 40);
  EXPECT_EQ(cropped.Camera(1).PrincipalPointY(), 30);
  EXPECT_EQ(cropped.Camera(1).camera_id, 1);
  EXPECT_EQ(cropped.Camera(1).source, camera.source);
  EXPECT_TRUE(cropped.Camera(1).HasPriorFocalLength());
  Reconstruction copy = cropped;
  EXPECT_EQ(copy.Camera(1).camera_id, 1);
  ASSERT_EQ(cropped.NumImages(), kNumImages);
  for (const auto& [image_id, image] : cropped.Images()) {
    EXPECT_EQ(copy.Image(image_id).CameraId(), 1);
    EXPECT_EQ(copy.Image(image_id).CameraPtr(), &copy.Camera(1));
    EXPECT_NE(copy.Image(image_id).CameraPtr(), image.CameraPtr());
    ASSERT_EQ(image.NumPoints2D(), 2);
    EXPECT_THAT(image.Point2D(0).xy,
                EigenMatrixNear(Eigen::Vector2d(50, 20), 1e-10));
    EXPECT_THAT(image.Point2D(1).xy,
                EigenMatrixNear(Eigen::Vector2d(-10, -15), 1e-10));
  }

  UndistortCameraOptions options;
  options.max_image_size = 50;
  UndistortReconstruction(options, &reconstruction);

  // The camera resolution and intrinsics are rescaled to match max_image_size.
  const Camera& undistorted_camera = reconstruction.Camera(1);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.width, 50);
  EXPECT_EQ(undistorted_camera.height, 50);
  EXPECT_EQ(undistorted_camera.source, camera.source);
  EXPECT_TRUE(undistorted_camera.HasPriorFocalLength());
  EXPECT_NEAR(undistorted_camera.FocalLengthX(), 50, 1e-6);
  EXPECT_NEAR(undistorted_camera.FocalLengthY(), 50, 1e-6);
  EXPECT_NEAR(undistorted_camera.PrincipalPointX(), 25, 1e-6);
  EXPECT_NEAR(undistorted_camera.PrincipalPointY(), 25, 1e-6);

  // The observations are rescaled consistently with the camera: (60, 40) maps
  // to normalized (0.1, -0.1) and back through the halved intrinsics to
  // (0.1 * 50 + 25, -0.1 * 50 + 25) = (30, 20).
  for (const auto& image : reconstruction.Images()) {
    ASSERT_EQ(image.second.NumPoints2D(), 1);
    EXPECT_THAT(image.second.Point2D(0).xy,
                EigenMatrixNear(Eigen::Vector2d(30, 20), 1e-6));
  }
}

TEST(UndistortReconstruction, RescalesSphericalCameras) {
  constexpr size_t kNumImages = 3;

  Reconstruction reconstruction;

  // Spherical (EQUIRECTANGULAR) camera at a high resolution. It has no pinhole
  // plane to undistort to, but can still be resized to a smaller image of the
  // same model.
  Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);
  ASSERT_TRUE(camera.IsSpherical());
  reconstruction.AddCamera(camera);
  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  reconstruction.AddRig(rig);

  // A front-hemisphere pixel (azimuth 36 deg) and a back-hemisphere pixel
  // (azimuth -144 deg, behind the camera). The back-hemisphere observation has
  // no forward normalized (CamFromImg) representation, so it must be handled by
  // plain linear scaling rather than a bearing round-trip.
  const std::vector<Eigen::Vector2d> points_xy = {Eigen::Vector2d(600, 200),
                                                  Eigen::Vector2d(100, 400)};
  for (image_t image_id = 1; image_id <= kNumImages; ++image_id) {
    Frame frame;
    frame.SetRigId(1);
    frame.SetFrameId(image_id);
    frame.SetRigFromWorld(Rigid3d());
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetFrameId(frame.FrameId());
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(points_xy);
    frame.AddDataId(image.DataId());
    reconstruction.AddFrame(frame);
    reconstruction.AddImage(image);
    reconstruction.RegisterFrame(frame.FrameId());
  }

  Reconstruction unchanged = reconstruction;
  UndistortCameraOptions roi_options;
  roi_options.roi_min_x = roi_options.roi_min_y = 0.2;
  roi_options.roi_max_x = roi_options.roi_max_y = 0.8;
  UndistortReconstruction(roi_options, &unchanged);
  EXPECT_EQ(unchanged.Camera(1), camera);
  ASSERT_EQ(unchanged.NumImages(), kNumImages);
  for (const auto& [image_id, image] : unchanged.Images()) {
    ASSERT_EQ(image.NumPoints2D(), points_xy.size());
    for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
      EXPECT_THAT(image.Point2D(idx).xy,
                  EigenMatrixNear(points_xy[idx], 1e-10));
    }
  }

  UndistortCameraOptions options;
  options.max_image_size = 250;
  UndistortReconstruction(options, &reconstruction);

  // The camera keeps its model but is resized so its larger dimension (width)
  // matches max_image_size: scale = 250 / 1000 = 0.25.
  const Camera& undistorted_camera = reconstruction.Camera(1);
  EXPECT_EQ(undistorted_camera.ModelName(), "EQUIRECTANGULAR");
  EXPECT_EQ(undistorted_camera.width, 250);
  EXPECT_EQ(undistorted_camera.height, 125);

  // The observations are rescaled consistently with the camera: the fractional
  // position within the image is preserved (a plain 0.25 scaling here), so both
  // the front- and back-hemisphere points map through without loss.
  for (const auto& image : reconstruction.Images()) {
    ASSERT_EQ(image.second.NumPoints2D(), 2);
    EXPECT_THAT(image.second.Point2D(0).xy,
                EigenMatrixNear(Eigen::Vector2d(150, 50), 1e-6));
    EXPECT_THAT(image.second.Point2D(1).xy,
                EigenMatrixNear(Eigen::Vector2d(25, 100), 1e-6));
  }
}

TEST(RectifyStereoCameras, Nominal) {
  Camera camera1;
  camera1 = Camera::CreateFromModelId(1, CameraModelId::kPinhole, 1, 1, 1);

  Camera camera2;
  camera2 = Camera::CreateFromModelId(1, CameraModelId::kPinhole, 1, 1, 1);

  const Rigid3d cam2_from_cam1(
      Eigen::Quaterniond(EulerAnglesToRotationMatrix(0.1, 0.2, 0.3)),
      Eigen::Vector3d(0.1, 0.2, 0.3));

  Camera rectified_camera1;
  Camera rectified_camera2;
  Eigen::Matrix3d H1;
  Eigen::Matrix3d H2;
  Eigen::Matrix4d Q;
  RectifyStereoCameras(camera1, camera2, cam2_from_cam1, &H1, &H2, &Q);

  Eigen::Matrix3d H1_ref;
  H1_ref << -0.202759, -0.815848, -0.897034, 0.416329, 0.733069, -0.199657,
      0.910839, -0.175408, 0.942638;
  EXPECT_THAT(H1, EigenMatrixNear<Eigen::Matrix3d>(H1_ref.transpose(), 1e-5));

  Eigen::Matrix3d H2_ref;
  H2_ref << -0.082173, -1.01288, -0.698868, 0.301854, 0.472844, -0.465336,
      0.963533, 0.292411, 1.12528;
  EXPECT_THAT(H2, EigenMatrixNear<Eigen::Matrix3d>(H2_ref.transpose(), 1e-5));

  Eigen::Matrix4d Q_ref;
  Q_ref << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -2.67261, -0.5, -0.5, 1, 0;
  EXPECT_THAT(Q, EigenMatrixNear(Q_ref, 1e-5));
}

TEST(RectifyStereoCameras, DisparityTo3DRoundTrip) {
  for (const size_t height : {480, 640}) {
    const Camera camera1 =
        Camera::CreateFromModelId(1, CameraModelId::kPinhole, 800, 640, height);
    const Camera camera2 =
        Camera::CreateFromModelId(2, CameraModelId::kPinhole, 750, 640, height);
    const Eigen::Matrix3d K1 = camera1.CalibrationMatrix();
    const Eigen::Matrix3d K2 = camera2.CalibrationMatrix();
    // Mirror the rectified calibration matrix construction of
    // RectifyStereoCameras (minimum focal length, first camera's cx, averaged
    // cy), so that the test also locks in that convention.
    Eigen::Matrix3d K = K1;
    K(0, 0) = K(1, 1) = 750;

    for (const double baseline : {-0.2, 0.2}) {
      for (const double angle : {0.0, 0.1}) {
        // Use a non-zero Y translation component to cover general,
        // non-axis-aligned translations in all cases.
        const Rigid3d cam2_from_cam1(
            Eigen::Quaterniond(EulerAnglesToRotationMatrix(angle, 0.0, 0.0)),
            Eigen::Vector3d(baseline, 0.05, 0.0));
        Eigen::Matrix3d H1;
        Eigen::Matrix3d H2;
        Eigen::Matrix4d Q;
        RectifyStereoCameras(camera1, camera2, cam2_from_cam1, &H1, &H2, &Q);
        const Eigen::Matrix3d rectified_from_cam1 = K.inverse() * H1 * K1;

        for (const Eigen::Vector3d& point : {Eigen::Vector3d(0.2, -0.1, 4.0),
                                             Eigen::Vector3d(-0.5, 0.3, 8.0)}) {
          const Eigen::Vector2d xy1 = (H1 * K1 * point).hnormalized();
          const Eigen::Vector2d xy2 =
              (H2 * K2 * (cam2_from_cam1 * point)).hnormalized();
          ASSERT_NEAR(xy1.y(), xy2.y(), 1e-10);
          const Eigen::Vector4d xy_disparity(
              xy1.x(), xy1.y(), xy1.x() - xy2.x(), 1.0);
          const Eigen::Vector3d recovered_point =
              (Q.transpose() * xy_disparity).hnormalized();
          EXPECT_THAT(
              recovered_point,
              EigenMatrixNear((rectified_from_cam1 * point).eval(), 1e-10));
        }
      }
    }
  }
}

TEST(RectifyAndUndistortStereoImages, Nominal) {
  UndistortCameraOptions options;

  // Create two distorted cameras with radial distortion.
  Camera distorted_camera1 =
      Camera::CreateFromModelId(1, CameraModelId::kSimpleRadial, 100, 100, 100);
  distorted_camera1.params[3] = 0.1;  // Add some radial distortion

  Camera distorted_camera2 =
      Camera::CreateFromModelId(2, CameraModelId::kSimpleRadial, 100, 100, 100);
  distorted_camera2.params[3] = 0.1;  // Add some radial distortion

  // Create dummy distorted images.
  Bitmap distorted_image1(100, 100, true);
  distorted_image1.Fill(BitmapColor<uint8_t>(255, 0, 0));  // Red image

  Bitmap distorted_image2(100, 100, true);
  distorted_image2.Fill(BitmapColor<uint8_t>(0, 255, 0));  // Green image

  // Create relative pose between cameras (typical stereo baseline).
  const Rigid3d cam2_from_cam1(
      Eigen::Quaterniond(EulerAnglesToRotationMatrix(0.0, 0.05, 0.0)),
      Eigen::Vector3d(0.1, 0.0, 0.0));  // 0.1m baseline

  Bitmap undistorted_image1;
  Bitmap undistorted_image2;
  Camera undistorted_camera;
  Eigen::Matrix4d Q;

  // Rectify and undistort stereo images.
  RectifyAndUndistortStereoImages(options,
                                  distorted_image1,
                                  distorted_image2,
                                  distorted_camera1,
                                  distorted_camera2,
                                  cam2_from_cam1,
                                  &undistorted_image1,
                                  &undistorted_image2,
                                  &undistorted_camera,
                                  &Q);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.width, undistorted_image1.Width());
  EXPECT_EQ(undistorted_camera.height, undistorted_image1.Height());
  EXPECT_EQ(undistorted_image1.Width(), undistorted_image2.Width());
  EXPECT_EQ(undistorted_image1.Height(), undistorted_image2.Height());
}

}  // namespace
}  // namespace colmap
