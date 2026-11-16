// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/mvs.h"

#include "colmap/mvs/model.h"
#include "colmap/mvs/texture_mapping.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

std::string ReadFileAsString(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

class MeshTexturerTest : public testing::Test {
 protected:
  void SetUp() override {
    test_path_ = CreateTestDir();
    CreateDirIfNotExists(test_path_ / "sparse");
    CreateDirIfNotExists(test_path_ / "images");

    Reconstruction reconstruction;
    reconstruction.AddCameraWithTrivialRig(Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 64, 64, 64));
    Image image;
    image.SetImageId(1);
    image.SetCameraId(1);
    image.SetName("image.png");
    reconstruction.AddImageWithTrivialFrame(
        image,
        Rigid3d(Eigen::Quaterniond(0, 1, 0, 0), Eigen::Vector3d(-0.5, 0.5, 5)));
    reconstruction.WriteBinary(test_path_ / "sparse");

    Bitmap bitmap(64, 64, /*as_rgb=*/true);
    for (int y = 0; y < 64; ++y) {
      for (int x = 0; x < 64; ++x) {
        bitmap.SetPixel(x, y, BitmapColor<uint8_t>(4 * x, 4 * y, 2 * (x + y)));
      }
    }
    ASSERT_TRUE(bitmap.Write(test_path_ / "images" / "image.png"));

    mesh_.vertices = {PlyMeshVertex(0, 0, 0),
                      PlyMeshVertex(1, 0, 0),
                      PlyMeshVertex(1, 1, 0),
                      PlyMeshVertex(0, 1, 0)};
    mesh_.faces = {PlyMeshFace(0, 1, 2), PlyMeshFace(0, 2, 3)};
    WriteBinaryPlyMesh(test_path_ / "input.ply", {mesh_, {}, {}});
  }

  int Run(const std::filesystem::path& output_path,
          const std::string& output_type = "BIN") const {
    std::vector<std::string> args = {
        "mesh_texturer",
        "--workspace_path",
        test_path_.string(),
        "--input_path",
        (test_path_ / "input.ply").string(),
        "--output_path",
        output_path.string(),
        "--output_type",
        output_type,
        "--MeshTextureMapping.apply_color_correction",
        "0",
        "--MeshTextureMapping.view_selection_smoothing_iterations",
        "0",
        "--MeshTextureMapping.inpaint_radius",
        "0",
        "--MeshTextureMapping.num_threads",
        "1",
    };
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) argv.push_back(arg.data());
    return RunMeshTexturer(argv.size(), argv.data());
  }

  std::filesystem::path test_path_;
  PlyMesh mesh_;
};

TEST_F(MeshTexturerTest, WritesCompleteBinaryAndTextOutputs) {
  mvs::Model model;
  model.ReadFromCOLMAP(test_path_);
  ASSERT_EQ(model.images.size(), 1);
  Bitmap bitmap;
  ASSERT_TRUE(bitmap.Read(model.images.front().GetPath(), /*as_rgb=*/true));
  model.images.front().SetBitmap(std::move(bitmap));

  mvs::MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  options.num_threads = 1;
  const auto expected = mvs::MeshTextureMapping(mesh_, model.images, options);
  ASSERT_EQ(expected.face_view_ids, (std::vector<int>{0, 0}));
  ASSERT_FALSE(expected.texture_atlas.IsEmpty());
  const PlyTexturedMesh expected_mesh{mesh_, expected.face_uvs, "texture.png"};

  for (const std::string output_type : {"BIN", "TXT"}) {
    SCOPED_TRACE(output_type);
    const auto output_path = test_path_ / output_type;
    ASSERT_EQ(Run(output_path, output_type), EXIT_SUCCESS);
    const auto expected_path = test_path_ / (output_type + "-expected.ply");
    if (output_type == "BIN") {
      WriteBinaryPlyMesh(expected_path, expected_mesh);
    } else {
      WriteTextPlyMesh(expected_path, expected_mesh);
    }
    EXPECT_EQ(ReadFileAsString(output_path / "mesh.ply"),
              ReadFileAsString(expected_path));
    Bitmap atlas;
    ASSERT_TRUE(atlas.Read(output_path / "texture.png", /*as_rgb=*/true));
    EXPECT_EQ(atlas.Width(), expected.atlas_width);
    EXPECT_EQ(atlas.Height(), expected.atlas_height);
    EXPECT_EQ(atlas.RowMajorData(), expected.texture_atlas.RowMajorData());
  }
}

TEST_F(MeshTexturerTest, ReportsAtlasWriteFailureBeforeWritingMesh) {
  const auto output_path = test_path_ / "output";
  CreateDirIfNotExists(output_path);
  CreateDirIfNotExists(output_path / "texture.png");
  EXPECT_THAT([&] { Run(output_path); },
              testing::ThrowsMessage<std::invalid_argument>(
                  testing::HasSubstr("Failed to write texture atlas:")));
  EXPECT_FALSE(std::filesystem::exists(output_path / "mesh.ply"));
}

TEST_F(MeshTexturerTest, ReportsEmptyAtlasBeforeCreatingOutput) {
  for (const bool empty_mesh : {false, true}) {
    SCOPED_TRACE(empty_mesh);
    PlyMesh mesh = mesh_;
    if (empty_mesh) {
      mesh = PlyMesh();
    } else {
      for (auto& vertex : mesh.vertices) vertex.z = 6;
    }
    WriteBinaryPlyMesh(test_path_ / "input.ply", {mesh, {}, {}});
    const auto output_path = test_path_ / (empty_mesh ? "empty" : "invisible");
    EXPECT_THAT([&] { Run(output_path); },
                testing::ThrowsMessage<std::invalid_argument>(
                    testing::HasSubstr("No texture atlas was generated")));
    EXPECT_FALSE(std::filesystem::exists(output_path));
  }
}

}  // namespace
}  // namespace colmap
