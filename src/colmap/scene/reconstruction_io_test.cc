// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/reconstruction_io.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <streambuf>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct ReaderWriter {
  virtual ~ReaderWriter() = default;

  virtual void ReadRigs(Reconstruction& reconstruction) = 0;
  virtual void ReadCameras(Reconstruction& reconstruction) = 0;
  virtual void ReadFrames(Reconstruction& reconstruction) = 0;
  virtual void ReadImages(Reconstruction& reconstruction) = 0;
  virtual void ReadPoints3D(Reconstruction& reconstruction) = 0;

  virtual void WriteRigs(const Reconstruction& reconstruction) = 0;
  virtual void WriteCameras(const Reconstruction& reconstruction) = 0;
  virtual void WriteFrames(const Reconstruction& reconstruction) = 0;
  virtual void WriteImages(const Reconstruction& reconstruction) = 0;
  virtual void WritePoints3D(const Reconstruction& reconstruction) = 0;

  virtual std::string RigsStr() const = 0;
  virtual std::string CamerasStr() const = 0;
  virtual std::string FramesStr() const = 0;
  virtual std::string ImagesStr() const = 0;
  virtual std::string Points3DStr() const = 0;
};

struct ReaderWriterTextStringStream : public ReaderWriter {
  void ReadRigs(Reconstruction& reconstruction) override {
    ReadRigsText(reconstruction, rigs_stream_);
  }
  void ReadCameras(Reconstruction& reconstruction) override {
    ReadCamerasText(reconstruction, cameras_stream_);
  }
  void ReadFrames(Reconstruction& reconstruction) override {
    ReadFramesText(reconstruction, frames_stream_);
  }
  void ReadImages(Reconstruction& reconstruction) override {
    ReadImagesText(reconstruction, images_stream_);
  }
  void ReadPoints3D(Reconstruction& reconstruction) override {
    ReadPoints3DText(reconstruction, points3D_stream_);
  }

  void WriteRigs(const Reconstruction& reconstruction) override {
    WriteRigsText(reconstruction, rigs_stream_);
  }
  void WriteCameras(const Reconstruction& reconstruction) override {
    WriteCamerasText(reconstruction, cameras_stream_);
  }
  void WriteFrames(const Reconstruction& reconstruction) override {
    WriteFramesText(reconstruction, frames_stream_);
  }
  void WriteImages(const Reconstruction& reconstruction) override {
    WriteImagesText(reconstruction, images_stream_);
  }
  void WritePoints3D(const Reconstruction& reconstruction) override {
    WritePoints3DText(reconstruction, points3D_stream_);
  }

  virtual std::string RigsStr() const override { return rigs_stream_.str(); }
  virtual std::string CamerasStr() const override {
    return cameras_stream_.str();
  }
  virtual std::string FramesStr() const override {
    return frames_stream_.str();
  }
  virtual std::string ImagesStr() const override {
    return images_stream_.str();
  }
  virtual std::string Points3DStr() const override {
    return points3D_stream_.str();
  }

 private:
  std::stringstream rigs_stream_;
  std::stringstream cameras_stream_;
  std::stringstream frames_stream_;
  std::stringstream images_stream_;
  std::stringstream points3D_stream_;
};

struct ReaderWriterBinaryStringStream : public ReaderWriter {
  void ReadRigs(Reconstruction& reconstruction) override {
    ReadRigsBinary(reconstruction, rigs_stream_);
  }
  void ReadCameras(Reconstruction& reconstruction) override {
    ReadCamerasBinary(reconstruction, cameras_stream_);
  }
  void ReadImages(Reconstruction& reconstruction) override {
    ReadImagesBinary(reconstruction, images_stream_);
  }
  void ReadFrames(Reconstruction& reconstruction) override {
    ReadFramesBinary(reconstruction, frames_stream_);
  }
  void ReadPoints3D(Reconstruction& reconstruction) override {
    ReadPoints3DBinary(reconstruction, points3D_stream_);
  }

  void WriteRigs(const Reconstruction& reconstruction) override {
    WriteRigsBinary(reconstruction, rigs_stream_);
  }
  void WriteCameras(const Reconstruction& reconstruction) override {
    WriteCamerasBinary(reconstruction, cameras_stream_);
  }
  void WriteFrames(const Reconstruction& reconstruction) override {
    WriteFramesBinary(reconstruction, frames_stream_);
  }
  void WriteImages(const Reconstruction& reconstruction) override {
    WriteImagesBinary(reconstruction, images_stream_);
  }
  void WritePoints3D(const Reconstruction& reconstruction) override {
    WritePoints3DBinary(reconstruction, points3D_stream_);
  }

  virtual std::string RigsStr() const override { return rigs_stream_.str(); }
  virtual std::string CamerasStr() const override {
    return cameras_stream_.str();
  }
  virtual std::string FramesStr() const override {
    return frames_stream_.str();
  }
  virtual std::string ImagesStr() const override {
    return images_stream_.str();
  }
  virtual std::string Points3DStr() const override {
    return points3D_stream_.str();
  }

 private:
  std::stringstream rigs_stream_;
  std::stringstream cameras_stream_;
  std::stringstream frames_stream_;
  std::stringstream images_stream_;
  std::stringstream points3D_stream_;
};

std::string ReadFileAsString(const std::filesystem::path& path) {
  std::fstream file(path);
  THROW_CHECK(file.good());
  std::stringstream buf;
  buf << file.rdbuf();
  return buf.str();
}

struct ReaderWriterFileStream : public ReaderWriter {
  explicit ReaderWriterFileStream(const std::string& ext)
      : test_dir_(CreateTestDir()),
        rigs_path_(test_dir_ / ("rigs." + ext)),
        cameras_path_(test_dir_ / ("cameras." + ext)),
        frames_path_(test_dir_ / ("frames." + ext)),
        images_path_(test_dir_ / ("images." + ext)),
        points3D_path_(test_dir_ / ("points3D." + ext)) {}

  virtual std::string RigsStr() const override {
    return ReadFileAsString(rigs_path_);
  }
  virtual std::string CamerasStr() const override {
    return ReadFileAsString(cameras_path_);
  }
  virtual std::string FramesStr() const override {
    return ReadFileAsString(frames_path_);
  }
  virtual std::string ImagesStr() const override {
    return ReadFileAsString(images_path_);
  }
  virtual std::string Points3DStr() const override {
    return ReadFileAsString(points3D_path_);
  }

 protected:
  const std::filesystem::path test_dir_;
  const std::filesystem::path rigs_path_;
  const std::filesystem::path cameras_path_;
  const std::filesystem::path frames_path_;
  const std::filesystem::path images_path_;
  const std::filesystem::path points3D_path_;
};

struct ReaderWriterTextFileStream : public ReaderWriterFileStream {
  ReaderWriterTextFileStream() : ReaderWriterFileStream("txt") {}

  void ReadRigs(Reconstruction& reconstruction) override {
    ReadRigsText(reconstruction, rigs_path_);
  }
  void ReadCameras(Reconstruction& reconstruction) override {
    ReadCamerasText(reconstruction, cameras_path_);
  }
  void ReadFrames(Reconstruction& reconstruction) override {
    ReadFramesText(reconstruction, frames_path_);
  }
  void ReadImages(Reconstruction& reconstruction) override {
    ReadImagesText(reconstruction, images_path_);
  }
  void ReadPoints3D(Reconstruction& reconstruction) override {
    ReadPoints3DText(reconstruction, points3D_path_);
  }

  void WriteRigs(const Reconstruction& reconstruction) override {
    WriteRigsText(reconstruction, rigs_path_);
  }
  void WriteCameras(const Reconstruction& reconstruction) override {
    WriteCamerasText(reconstruction, cameras_path_);
  }
  void WriteFrames(const Reconstruction& reconstruction) override {
    WriteFramesText(reconstruction, frames_path_);
  }
  void WriteImages(const Reconstruction& reconstruction) override {
    WriteImagesText(reconstruction, images_path_);
  }
  void WritePoints3D(const Reconstruction& reconstruction) override {
    WritePoints3DText(reconstruction, points3D_path_);
  }
};

struct ReaderWriterBinaryFileStream : public ReaderWriterFileStream {
  ReaderWriterBinaryFileStream() : ReaderWriterFileStream("bin") {}

  void ReadRigs(Reconstruction& reconstruction) override {
    ReadRigsBinary(reconstruction, rigs_path_);
  }
  void ReadCameras(Reconstruction& reconstruction) override {
    ReadCamerasBinary(reconstruction, cameras_path_);
  }
  void ReadFrames(Reconstruction& reconstruction) override {
    ReadFramesBinary(reconstruction, frames_path_);
  }
  void ReadImages(Reconstruction& reconstruction) override {
    ReadImagesBinary(reconstruction, images_path_);
  }
  void ReadPoints3D(Reconstruction& reconstruction) override {
    ReadPoints3DBinary(reconstruction, points3D_path_);
  }

  void WriteRigs(const Reconstruction& reconstruction) override {
    WriteRigsBinary(reconstruction, rigs_path_);
  }
  void WriteCameras(const Reconstruction& reconstruction) override {
    WriteCamerasBinary(reconstruction, cameras_path_);
  }
  void WriteFrames(const Reconstruction& reconstruction) override {
    WriteFramesBinary(reconstruction, frames_path_);
  }
  void WriteImages(const Reconstruction& reconstruction) override {
    WriteImagesBinary(reconstruction, images_path_);
  }
  void WritePoints3D(const Reconstruction& reconstruction) override {
    WritePoints3DBinary(reconstruction, points3D_path_);
  }
};

class ParameterizedReaderWriterTests
    : public ::testing::TestWithParam<
          std::function<std::unique_ptr<ReaderWriter>()>> {};

TEST_P(ParameterizedReaderWriterTests, Roundtrip) {
  std::unique_ptr<ReaderWriter> reader_writer = GetParam()();

  Reconstruction orig;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 4;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 321;
  SynthesizeDataset(synthetic_dataset_options, &orig);

  reader_writer->WriteRigs(orig);
  EXPECT_FALSE(reader_writer->RigsStr().empty());

  reader_writer->WriteCameras(orig);
  EXPECT_FALSE(reader_writer->CamerasStr().empty());

  reader_writer->WriteFrames(orig);
  EXPECT_FALSE(reader_writer->FramesStr().empty());

  reader_writer->WriteImages(orig);
  EXPECT_FALSE(reader_writer->ImagesStr().empty());

  reader_writer->WritePoints3D(orig);
  EXPECT_FALSE(reader_writer->Points3DStr().empty());

  Reconstruction test;
  reader_writer->ReadCameras(test);
  EXPECT_EQ(orig.Cameras(), test.Cameras());
  reader_writer->ReadRigs(test);
  EXPECT_EQ(orig.Rigs(), test.Rigs());
  reader_writer->ReadFrames(test);
  EXPECT_EQ(orig.Frames(), test.Frames());
  reader_writer->ReadImages(test);
  EXPECT_EQ(orig.Images(), test.Images());
  reader_writer->ReadPoints3D(test);
  EXPECT_EQ(orig.Points3D(), test.Points3D());
}

TEST_P(ParameterizedReaderWriterTests, LegacyWithoutRigsAndFrames) {
  std::unique_ptr<ReaderWriter> reader_writer = GetParam()();

  Reconstruction orig;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 3;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 321;
  SynthesizeDataset(synthetic_dataset_options, &orig);

  reader_writer->WriteCameras(orig);
  EXPECT_FALSE(reader_writer->CamerasStr().empty());

  reader_writer->WriteImages(orig);
  EXPECT_FALSE(reader_writer->ImagesStr().empty());

  reader_writer->WritePoints3D(orig);
  EXPECT_FALSE(reader_writer->Points3DStr().empty());

  Reconstruction test;
  reader_writer->ReadCameras(test);
  EXPECT_EQ(orig.Cameras(), test.Cameras());
  reader_writer->ReadImages(test);
  EXPECT_EQ(orig.Rigs(), test.Rigs());
  EXPECT_EQ(orig.Frames(), test.Frames());
  EXPECT_EQ(orig.Images(), test.Images());
  reader_writer->ReadPoints3D(test);
  EXPECT_EQ(orig.Points3D(), test.Points3D());
}

TEST_P(ParameterizedReaderWriterTests, RoundtripImageNamesWithWhitespace) {
  std::unique_ptr<ReaderWriter> reader_writer = GetParam()();

  Reconstruction orig;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 10;
  SynthesizeDataset(synthetic_dataset_options, &orig);

  const std::vector<std::string> names = {
      "image (1).jpg", "sub dir/image  2.jpg", "image3.jpg"};
  const std::vector<image_t> image_ids = orig.RegImageIds();
  ASSERT_EQ(image_ids.size(), names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    orig.Image(image_ids[i]).SetName(names[i]);
  }

  reader_writer->WriteRigs(orig);
  reader_writer->WriteCameras(orig);
  reader_writer->WriteFrames(orig);
  reader_writer->WriteImages(orig);

  Reconstruction test;
  reader_writer->ReadCameras(test);
  reader_writer->ReadRigs(test);
  reader_writer->ReadFrames(test);
  reader_writer->ReadImages(test);
  EXPECT_EQ(orig.Images(), test.Images());
}

INSTANTIATE_TEST_SUITE_P(
    ReaderWriterTests,
    ParameterizedReaderWriterTests,
    ::testing::Values(
        []() { return std::make_unique<ReaderWriterTextStringStream>(); },
        []() { return std::make_unique<ReaderWriterBinaryStringStream>(); },
        []() { return std::make_unique<ReaderWriterTextFileStream>(); },
        []() { return std::make_unique<ReaderWriterBinaryFileStream>(); }));

TEST(BinaryFileIO, ReportsWriteFailure) {
#if defined(__linux__)
  // Verify the device is available and rejects even a buffered header write.
  std::ofstream full("/dev/full", std::ios::binary);
  if (!full.is_open()) {
    GTEST_SKIP() << "/dev/full is unavailable";
  }
  full.put('\0');
  full.flush();
  ASSERT_TRUE(full.fail());
  full.close();

  Reconstruction reconstruction;
  for (const bool populated : {false, true}) {
    SCOPED_TRACE(populated);
    if (populated) {
      SyntheticDatasetOptions options;
      options.num_rigs = 1;
      options.num_cameras_per_rig = 2;
      options.num_frames_per_rig = 2;
      options.num_points3D = 10;
      SynthesizeDataset(options, &reconstruction);
    }
    EXPECT_THROW(WriteRigsBinary(reconstruction, "/dev/full"),
                 std::invalid_argument);
    EXPECT_THROW(WriteCamerasBinary(reconstruction, "/dev/full"),
                 std::invalid_argument);
    EXPECT_THROW(WriteFramesBinary(reconstruction, "/dev/full"),
                 std::invalid_argument);
    EXPECT_THROW(WriteImagesBinary(reconstruction, "/dev/full"),
                 std::invalid_argument);
    EXPECT_THROW(WritePoints3DBinary(reconstruction, "/dev/full"),
                 std::invalid_argument);
  }
#else
  GTEST_SKIP() << "/dev/full failure injection requires Linux";
#endif
}

TEST(ExportNVM, Nominal) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 10;
  synthetic_dataset_options.camera_model_id = SimpleRadialCameraModel::model_id;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  const auto nvm_path = test_dir / "export.nvm";

  EXPECT_TRUE(ExportNVM(reconstruction, nvm_path));
  EXPECT_TRUE(std::filesystem::exists(nvm_path));
}

TEST(ExportNVM, UnsupportedCameraModel) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 5;
  synthetic_dataset_options.camera_model_id = FullOpenCVCameraModel::model_id;
  synthetic_dataset_options.camera_params = {
      1280, 1280, 512, 384, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  const auto nvm_path = test_dir / "export.nvm";

  EXPECT_FALSE(ExportNVM(reconstruction, nvm_path));
}

TEST(ExportCam, Nominal) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 10;
  synthetic_dataset_options.camera_model_id = SimpleRadialCameraModel::model_id;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();

  EXPECT_TRUE(ExportCam(reconstruction, test_dir));

  int num_cam_files = 0;
  for (const auto& entry : std::filesystem::directory_iterator(test_dir)) {
    if (entry.path().extension() == ".cam") {
      num_cam_files++;
    }
  }
  EXPECT_EQ(num_cam_files, reconstruction.NumRegImages());
}

TEST(ExportCam, UnsupportedCameraModel) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 5;
  synthetic_dataset_options.camera_model_id = FullOpenCVCameraModel::model_id;
  synthetic_dataset_options.camera_params = {
      1280, 1280, 512, 384, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();

  EXPECT_FALSE(ExportCam(reconstruction, test_dir));
}

TEST(ExportRecon3D, Nominal) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 15;
  synthetic_dataset_options.camera_model_id = SimpleRadialCameraModel::model_id;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();

  EXPECT_TRUE(ExportRecon3D(reconstruction, test_dir));

  const std::filesystem::path recon_dir = test_dir / "Recon";
  EXPECT_TRUE(std::filesystem::exists(recon_dir / "synth_0.out"));
  EXPECT_TRUE(std::filesystem::exists(recon_dir / "urd-images.txt"));
  EXPECT_TRUE(std::filesystem::exists(recon_dir / "imagemap_0.txt"));
}

TEST(ExportRecon3D, UnsupportedCameraModel) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 5;
  synthetic_dataset_options.camera_model_id = FullOpenCVCameraModel::model_id;
  synthetic_dataset_options.camera_params = {
      1280, 1280, 512, 384, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();

  EXPECT_FALSE(ExportRecon3D(reconstruction, test_dir));
}

TEST(ExportBundler, Nominal) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.camera_model_id = SimpleRadialCameraModel::model_id;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  const auto bundler_path = test_dir / "bundle.out";
  const auto list_path = test_dir / "list.txt";

  EXPECT_TRUE(ExportBundler(reconstruction, bundler_path, list_path));
  EXPECT_TRUE(std::filesystem::exists(bundler_path));
  EXPECT_TRUE(std::filesystem::exists(list_path));
}

TEST(ExportBundler, UnsupportedCameraModel) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 5;
  synthetic_dataset_options.camera_model_id = FullOpenCVCameraModel::model_id;
  synthetic_dataset_options.camera_params = {
      1280, 1280, 512, 384, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  const auto bundler_path = test_dir / "bundle.out";
  const auto list_path = test_dir / "list.txt";

  EXPECT_FALSE(ExportBundler(reconstruction, bundler_path, list_path, false));
}

TEST(ExportPLY, Nominal) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  const auto ply_path = test_dir / "points.ply";

  ExportPLY(reconstruction, ply_path);
  EXPECT_TRUE(std::filesystem::exists(ply_path));
  EXPECT_GT(std::filesystem::file_size(ply_path), 0);
}

TEST(ExportVRML, Nominal) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 25;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  const auto images_path = test_dir / "images.wrl";
  const auto points3D_path = test_dir / "points3D.wrl";
  const double image_scale = 1.0;
  const Eigen::Vector3d image_rgb(1.0, 0.0, 0.0);

  ExportVRML(
      reconstruction, images_path, points3D_path, image_scale, image_rgb);
  EXPECT_TRUE(std::filesystem::exists(images_path));
  EXPECT_TRUE(std::filesystem::exists(points3D_path));
}

// Custom numpunct facet that uses comma as decimal separator, for testing
// locale independence without requiring a specific system locale.
struct CommaDecimalFacet : std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
};

TEST(TextIO, LocaleIndependentRoundtrip) {
  // Set global locale to use comma as decimal separator.
  const std::locale original_locale = std::locale::global(
      std::locale(std::locale::classic(), new CommaDecimalFacet));

  // Verify that the locale is used correctly.
  std::stringstream ss;
  ss << 1.23;
  EXPECT_EQ(ss.str(), "1,23");

  Reconstruction orig;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &orig);

  // Write under comma-decimal locale.
  ReaderWriterTextStringStream rw;
  rw.WriteRigs(orig);
  rw.WriteCameras(orig);
  rw.WriteFrames(orig);
  rw.WriteImages(orig);
  rw.WritePoints3D(orig);

  // Verify written float data uses dot, not comma, as decimal separator.
  // Check a data line (skip comment lines starting with '#').
  {
    std::istringstream iss(rw.CamerasStr());
    std::string line;
    while (std::getline(iss, line)) {
      if (!line.empty() && line[0] != '#') {
        EXPECT_THAT(line, testing::HasSubstr("."));
        EXPECT_THAT(line, testing::Not(testing::HasSubstr(",")));
      }
    }
  }

  // Read back under comma-decimal locale.
  Reconstruction test;
  rw.ReadCameras(test);
  EXPECT_EQ(orig.Cameras(), test.Cameras());
  rw.ReadRigs(test);
  EXPECT_EQ(orig.Rigs(), test.Rigs());
  rw.ReadFrames(test);
  EXPECT_EQ(orig.Frames(), test.Frames());
  rw.ReadImages(test);
  EXPECT_EQ(orig.Images(), test.Images());
  rw.ReadPoints3D(test);
  EXPECT_EQ(orig.Points3D(), test.Points3D());

  // Restore original locale.
  std::locale::global(original_locale);
}

}  // namespace

namespace {

// Encode the documented fields independently of production endian/stream
// helpers.
void AppendWireUnsigned(uint64_t value,
                        const size_t bytes,
                        std::string* output) {
  for (size_t i = 0; i < bytes; ++i) {
    output->push_back(static_cast<char>((value >> (8 * i)) & 0xff));
  }
}

void AppendWireDouble(const double value, std::string* output) {
  uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  AppendWireUnsigned(bits, 8, output);
}

std::string ExpectedImageWire(const Reconstruction& reconstruction) {
  std::string output;
  AppendWireUnsigned(reconstruction.NumRegImages(), 8, &output);
  for (const image_t id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(id);
    AppendWireUnsigned(id, 4, &output);
    const Rigid3d pose = image.CamFromWorld();
    for (const double value : {pose.rotation().w(),
                               pose.rotation().x(),
                               pose.rotation().y(),
                               pose.rotation().z(),
                               pose.translation().x(),
                               pose.translation().y(),
                               pose.translation().z()}) {
      AppendWireDouble(value, &output);
    }
    AppendWireUnsigned(image.CameraId(), 4, &output);
    output += image.Name();
    output.push_back('\0');
    AppendWireUnsigned(image.NumPoints2D(), 8, &output);
    for (const Point2D& point : image.Points2D()) {
      AppendWireDouble(point.xy.x(), &output);
      AppendWireDouble(point.xy.y(), &output);
      AppendWireUnsigned(point.point3D_id, 8, &output);
    }
  }
  return output;
}

std::string ExpectedPointWire(const Reconstruction& reconstruction) {
  std::vector<point3D_t> ids;
  for (const auto& [id, point] : reconstruction.Points3D()) ids.push_back(id);
  std::sort(ids.begin(), ids.end());
  std::string output;
  AppendWireUnsigned(ids.size(), 8, &output);
  for (const point3D_t id : ids) {
    const Point3D& point = reconstruction.Point3D(id);
    AppendWireUnsigned(id, 8, &output);
    for (int axis = 0; axis < 3; ++axis) {
      AppendWireDouble(point.xyz(axis), &output);
    }
    for (int axis = 0; axis < 3; ++axis) {
      AppendWireUnsigned(point.color(axis), 1, &output);
    }
    AppendWireDouble(point.error, &output);
    AppendWireUnsigned(point.track.Length(), 8, &output);
    for (const auto& element : point.track.Elements()) {
      AppendWireUnsigned(element.image_id, 4, &output);
      AppendWireUnsigned(element.point2D_idx, 4, &output);
    }
  }
  return output;
}

Reconstruction CreateBinaryWireFixture(const int num_points) {
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 2;
  options.num_points3D = num_points;
  options.num_points2D_without_point3D = 3;
  Reconstruction reconstruction;
  SynthesizeDataset(options, &reconstruction);
  const auto& ids = reconstruction.RegImageIds();
  reconstruction.Image(ids[0]).SetName("sub dir/camera \xc3\xa5.png");
  reconstruction.Image(ids[1]).SetName(std::string(32773, 'x'));
  auto& image = reconstruction.Image(ids[0]);
  image.Point2D(0).xy.x() = -0.0;
  image.Point2D(0).xy.y() = std::numeric_limits<double>::infinity();
  const uint64_t nan_bits = UINT64_C(0x7ff8000000001234);
  std::memcpy(&image.Point2D(1).xy.x(), &nan_bits, sizeof(nan_bits));
  for (const auto& [id, unused] : reconstruction.Points3D()) {
    auto& point = reconstruction.Point3D(id);
    point.color = Eigen::Vector3ub(0, 128, 255);
    point.error = (id % 2) ? -0.0 : 0.125;
  }
  return reconstruction;
}

class LimitedWriteBuffer : public std::streambuf {
 public:
  explicit LimitedWriteBuffer(const size_t capacity) : capacity_(capacity) {}
  const std::string& Bytes() const { return bytes_; }

 protected:
  std::streamsize xsputn(const char* data, std::streamsize size) override {
    const size_t count = std::min<size_t>(size, capacity_ - bytes_.size());
    bytes_.append(data, count);
    return count;
  }

 private:
  const size_t capacity_;
  std::string bytes_;
};

}  // namespace

TEST(ReconstructionBinary, ExactDocumentedWireAndBufferBoundaries) {
  Reconstruction empty;
  std::ostringstream empty_images, empty_points;
  WriteImagesBinary(empty, empty_images);
  WritePoints3DBinary(empty, empty_points);
  EXPECT_EQ(empty_images.str(), std::string(8, '\0'));
  EXPECT_EQ(empty_points.str(), std::string(8, '\0'));
  for (const int num_points : {3, 1371}) {
    SCOPED_TRACE(num_points);
    const Reconstruction reconstruction = CreateBinaryWireFixture(num_points);
    std::ostringstream images, points;
    WriteImagesBinary(reconstruction, images);
    WritePoints3DBinary(reconstruction, points);
    EXPECT_EQ(images.str(), ExpectedImageWire(reconstruction));
    EXPECT_EQ(points.str(), ExpectedPointWire(reconstruction));
  }
}

TEST(ReconstructionBinary, ShortWritePreservesStreamFailureAndPrefix) {
  const Reconstruction reconstruction = CreateBinaryWireFixture(1371);
  for (const bool write_images : {false, true}) {
    const std::string expected = write_images
                                     ? ExpectedImageWire(reconstruction)
                                     : ExpectedPointWire(reconstruction);
    const auto write = [&](std::ostream& stream) {
      if (write_images)
        WriteImagesBinary(reconstruction, stream);
      else
        WritePoints3DBinary(reconstruction, stream);
    };
    for (const size_t capacity : {size_t{0},
                                  size_t{1},
                                  size_t{7},
                                  size_t{8},
                                  size_t{32767},
                                  size_t{32768},
                                  size_t{32769},
                                  expected.size() - 1,
                                  expected.size()}) {
      SCOPED_TRACE(write_images);
      SCOPED_TRACE(capacity);
      for (const bool exceptions : {false, true}) {
        SCOPED_TRACE(exceptions);
        LimitedWriteBuffer buffer(capacity);
        std::ostream stream(&buffer);
        if (exceptions) stream.exceptions(std::ios::badbit | std::ios::failbit);
        if (exceptions && capacity < expected.size()) {
          EXPECT_THROW(write(stream), std::ios_base::failure);
        } else {
          EXPECT_NO_THROW(write(stream));
        }
        EXPECT_EQ(stream.good(), capacity == expected.size());
        EXPECT_EQ(buffer.Bytes(), expected.substr(0, capacity));
      }
    }
  }
}

namespace {

class WireSinkFailure : public std::runtime_error {
 public:
  WireSinkFailure() : std::runtime_error("original wire sink failure") {}
};

class ThrowingWriteBuffer : public LimitedWriteBuffer {
 public:
  using LimitedWriteBuffer::LimitedWriteBuffer;

 protected:
  std::streamsize xsputn(const char* data,
                         const std::streamsize size) override {
    const std::streamsize accepted = LimitedWriteBuffer::xsputn(data, size);
    if (accepted != size) {
      throw WireSinkFailure();
    }
    return accepted;
  }
};

bool LaterWriteThrowsOnBadStream() {
  std::ostringstream stream;
  stream.setstate(std::ios::badbit);
  stream.exceptions(std::ios::failbit);
  try {
    stream.write("", 0);
  } catch (const std::ios_base::failure&) {
    return true;
  }
  return false;
}

}  // namespace

TEST(ReconstructionBinary, InitiallyFailedStreamPublishesNothing) {
  const Reconstruction reconstruction;
  for (const bool write_images : {false, true}) {
    SCOPED_TRACE(write_images);
    for (const auto state :
         {std::ios::badbit, std::ios::failbit, std::ios::eofbit}) {
      SCOPED_TRACE(state);
      LimitedWriteBuffer buffer(64);
      std::ostream stream(&buffer);
      stream.setstate(state);
      if (write_images) {
        EXPECT_THROW(WriteImagesBinary(reconstruction, stream),
                     std::invalid_argument);
      } else {
        EXPECT_THROW(WritePoints3DBinary(reconstruction, stream),
                     std::invalid_argument);
      }
      EXPECT_EQ(stream.rdstate(), state);
      EXPECT_TRUE(buffer.Bytes().empty());
    }
  }
}

TEST(ReconstructionBinary, ThrowingStreamPreservesOriginalErrorAndPrefix) {
  const Reconstruction reconstruction = CreateBinaryWireFixture(1371);
  for (const bool write_images : {false, true}) {
    SCOPED_TRACE(write_images);
    const std::string expected = write_images
                                     ? ExpectedImageWire(reconstruction)
                                     : ExpectedPointWire(reconstruction);
    const auto write = [&](std::ostream& stream) {
      if (write_images) {
        WriteImagesBinary(reconstruction, stream);
      } else {
        WritePoints3DBinary(reconstruction, stream);
      }
    };
    for (const size_t capacity :
         {size_t{0}, size_t{8}, size_t{32768}, expected.size() - 1}) {
      SCOPED_TRACE(capacity);
      ASSERT_LT(capacity, expected.size());
      for (const bool exceptions : {false, true}) {
        SCOPED_TRACE(exceptions);
        ThrowingWriteBuffer buffer(capacity);
        std::ostream stream(&buffer);
        if (exceptions) {
          stream.exceptions(std::ios::badbit | std::ios::failbit);
          EXPECT_THROW(write(stream), WireSinkFailure);
        } else {
          EXPECT_NO_THROW(write(stream));
        }
        EXPECT_TRUE(stream.bad());
        EXPECT_EQ(buffer.Bytes(), expected.substr(0, capacity));
      }
    }
  }
}

TEST(ReconstructionBinary, ExceptionMasksAndLogicalFinalField) {
  const bool expected_later_throw = LaterWriteThrowsOnBadStream();
  Reconstruction reconstruction = CreateBinaryWireFixture(3);
  for (const image_t id : reconstruction.RegImageIds()) {
    reconstruction.Image(id).SetName("small");
  }
  for (const bool write_images : {false, true}) {
    SCOPED_TRACE(write_images);
    const std::string expected = write_images
                                     ? ExpectedImageWire(reconstruction)
                                     : ExpectedPointWire(reconstruction);
    ASSERT_LT(expected.size(), 32768);
    // The last image field is point3D_id; the last point field is point2D_idx.
    const size_t last_field_start = expected.size() - (write_images ? 8 : 4);
    const auto write = [&](std::ostream& stream) {
      if (write_images) {
        WriteImagesBinary(reconstruction, stream);
      } else {
        WritePoints3DBinary(reconstruction, stream);
      }
    };
    for (const auto mask : {std::ios::goodbit,
                            std::ios::failbit,
                            std::ios::badbit,
                            std::ios::failbit | std::ios::badbit}) {
      SCOPED_TRACE(mask);
      for (const size_t capacity : {size_t{0},
                                    size_t{8},
                                    last_field_start - 1,
                                    last_field_start,
                                    expected.size() - 1,
                                    expected.size()}) {
        SCOPED_TRACE(capacity);
        const auto check = [&](auto& buffer, const bool throwing_sink) {
          SCOPED_TRACE(throwing_sink);
          std::ostream stream(&buffer);
          stream.exceptions(mask);
          const bool fails = capacity < expected.size();
          if (fails && (mask & std::ios::badbit) != 0) {
            if (throwing_sink) {
              EXPECT_THROW(write(stream), WireSinkFailure);
            } else {
              EXPECT_THROW(write(stream), std::ios_base::failure);
            }
          } else if (fails && (mask & std::ios::failbit) != 0 &&
                     capacity < last_field_start) {
            // A later sentry may add failbit when it observes badbit.
            bool caught = false;
            try {
              write(stream);
            } catch (const std::ios_base::failure&) {
              caught = true;
            }
            EXPECT_EQ(caught, (stream.rdstate() & std::ios::failbit) != 0);
            EXPECT_EQ(caught, expected_later_throw);
          } else {
            EXPECT_NO_THROW(write(stream));
          }
          EXPECT_EQ(stream.good(), !fails);
          EXPECT_EQ(stream.bad(), fails);
          EXPECT_EQ(buffer.Bytes(), expected.substr(0, capacity));
        };
        LimitedWriteBuffer short_buffer(capacity);
        check(short_buffer, false);
        ThrowingWriteBuffer throwing_buffer(capacity);
        check(throwing_buffer, true);
      }
    }
  }
}

namespace {

class RejectLogicalWriteBuffer : public std::streambuf {
 public:
  RejectLogicalWriteBuffer(const size_t rejected_call, const bool throwing)
      : rejected_call_(rejected_call), throwing_(throwing) {}
  const std::string& Bytes() const { return bytes_; }

 protected:
  std::streamsize xsputn(const char* data,
                         const std::streamsize size) override {
    if (num_calls_++ == rejected_call_) {
      if (throwing_) {
        throw WireSinkFailure();
      }
      return 0;
    }
    bytes_.append(data, size);
    return size;
  }

 private:
  const size_t rejected_call_;
  const bool throwing_;
  size_t num_calls_ = 0;
  std::string bytes_;
};

}  // namespace

TEST(ReconstructionBinary, FailbitOnlyRetainsNameTerminatorField) {
  const bool expected_later_throw = LaterWriteThrowsOnBadStream();
  for (const std::string& name : {std::string{},
                                  std::string("short \xc3\xa5"),
                                  std::string("raw\0name", 8)}) {
    SCOPED_TRACE(name);
    Reconstruction reconstruction = CreateBinaryWireFixture(3);
    const auto& ids = reconstruction.RegImageIds();
    reconstruction.Image(ids[0]).SetName(name);
    reconstruction.Image(ids[1]).SetName("second");
    const std::string expected = ExpectedImageWire(reconstruction);
    ASSERT_LT(expected.size(), 32768);
    // Count, image ID, quaternion, translation, and camera ID precede name.
    constexpr size_t kNameOffset = 8 + 4 + 7 * 8 + 4;
    // Zero-based logical writes 10 and 11 are name+NUL and point count.
    for (const size_t rejected_call : {size_t{10}, size_t{11}}) {
      SCOPED_TRACE(rejected_call);
      const size_t accepted =
          kNameOffset + (rejected_call == 10 ? 0 : name.size() + 1);
      for (const bool throwing_sink : {false, true}) {
        SCOPED_TRACE(throwing_sink);
        RejectLogicalWriteBuffer buffer(rejected_call, throwing_sink);
        std::ostream stream(&buffer);
        stream.exceptions(std::ios::failbit);
        // With badbit exceptions disabled, a later sentry may add failbit.
        bool caught = false;
        try {
          WriteImagesBinary(reconstruction, stream);
        } catch (const std::ios_base::failure&) {
          caught = true;
        }
        EXPECT_EQ(caught, (stream.rdstate() & std::ios::failbit) != 0);
        EXPECT_EQ(caught, expected_later_throw);
        EXPECT_TRUE(stream.bad());
        EXPECT_EQ(buffer.Bytes(), expected.substr(0, accepted));
      }
    }
  }
}

}  // namespace colmap
