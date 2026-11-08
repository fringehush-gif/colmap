// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/texture_mapping.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

// Create a 2-triangle quad mesh (4 vertices, 2 faces).
// Quad in XY plane at Z=0, from (0,0,0) to (1,1,0).
PlyMesh MakeTriangleMesh() {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 1.0f, 0.0f),
      PlyMeshVertex(0.0f, 1.0f, 0.0f),
  };
  mesh.faces = {
      PlyMeshFace(0, 1, 2),
      PlyMeshFace(0, 2, 3),
  };
  return mesh;
}

// Create a camera looking down -Z at the XY plane.
// Camera at (0.5, 0.5, 5), looking toward origin along -Z in world.
// In COLMAP convention: P_cam = R * P_world + T, camera looks along +Z_cam.
// To look along -Z_world, use 180° rotation around X: R maps -Z_world to
// +Z_cam.
Image MakeTestImage(int width, int height, const BitmapColor<uint8_t>& color) {
  // K: focal length = width, principal point at center.
  float K[9] = {0};
  K[0] = static_cast<float>(width);          // fx
  K[4] = static_cast<float>(height);         // fy
  K[2] = static_cast<float>(width) / 2.0f;   // cx
  K[5] = static_cast<float>(height) / 2.0f;  // cy
  K[8] = 1.0f;

  // R: 180° rotation around X-axis. Row-major: [1,0,0, 0,-1,0, 0,0,-1].
  // This makes the camera look along -Z in world, with +Y up.
  float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};

  // C = (0.5, 0.5, 5), T = -R*C.
  // T = -[1,0,0;0,-1,0;0,0,-1] * [0.5,0.5,5] = [-0.5, 0.5, 5]
  float T[3] = {-0.5f, 0.5f, 5.0f};

  Image image("test_image.png",
              static_cast<size_t>(width),
              static_cast<size_t>(height),
              K,
              R,
              T);

  Bitmap bitmap(width, height, /*as_rgb=*/true);
  bitmap.Fill(color);
  image.SetBitmap(std::move(bitmap));

  return image;
}

// Create a camera where the mesh at Z=0 is behind the camera.
// Camera at (0.5, 0.5, -5), looking along -Z in world (away from mesh).
Image MakeBehindCameraImage(int width, int height) {
  float K[9] = {0};
  K[0] = static_cast<float>(width);
  K[4] = static_cast<float>(height);
  K[2] = static_cast<float>(width) / 2.0f;
  K[5] = static_cast<float>(height) / 2.0f;
  K[8] = 1.0f;

  // R: 180° around X, looks along -Z in world.
  float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};

  // C = (0.5, 0.5, -5), T = -R*C = [-0.5, 0.5, -5]
  float T[3] = {-0.5f, 0.5f, -5.0f};

  Image image("behind.png",
              static_cast<size_t>(width),
              static_cast<size_t>(height),
              K,
              R,
              T);

  Bitmap bitmap(width, height, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(0));
  image.SetBitmap(std::move(bitmap));

  return image;
}

// Create a camera that views the mesh at a very grazing angle.
// Camera nearly in-plane with XY mesh, at (0.5, 0.5, 0.01), looking along -Z.
Image MakeGrazingAngleImage(int width, int height) {
  float K[9] = {0};
  K[0] = static_cast<float>(width);
  K[4] = static_cast<float>(height);
  K[2] = static_cast<float>(width) / 2.0f;
  K[5] = static_cast<float>(height) / 2.0f;
  K[8] = 1.0f;

  // R: 180° around X.
  float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};

  // C = (0.5, 0.5, 0.01), T = -R*C = [-0.5, 0.5, 0.01]
  float T[3] = {-0.5f, 0.5f, 0.01f};

  Image image("grazing.png",
              static_cast<size_t>(width),
              static_cast<size_t>(height),
              K,
              R,
              T);

  Bitmap bitmap(width, height, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(128));
  image.SetBitmap(std::move(bitmap));

  return image;
}

// Create a 4-face pyramid mesh.
PlyMesh MakePyramidMesh() {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 1.0f, 0.0f),
      PlyMeshVertex(0.0f, 1.0f, 0.0f),
      PlyMeshVertex(0.5f, 0.5f, 1.0f),
  };
  mesh.faces = {
      PlyMeshFace(0, 1, 4),
      PlyMeshFace(1, 2, 4),
      PlyMeshFace(2, 3, 4),
      PlyMeshFace(3, 0, 4),
  };
  return mesh;
}

// Create a 12-face cube mesh (2 triangles per face).
PlyMesh MakeCubeMesh() {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0, 0, 0),
      PlyMeshVertex(1, 0, 0),
      PlyMeshVertex(1, 1, 0),
      PlyMeshVertex(0, 1, 0),
      PlyMeshVertex(0, 0, 1),
      PlyMeshVertex(1, 0, 1),
      PlyMeshVertex(1, 1, 1),
      PlyMeshVertex(0, 1, 1),
  };
  mesh.faces = {
      // Front (Z=0): 0,2,1 and 0,3,2
      PlyMeshFace(0, 2, 1),
      PlyMeshFace(0, 3, 2),
      // Back (Z=1): 4,5,6 and 4,6,7
      PlyMeshFace(4, 5, 6),
      PlyMeshFace(4, 6, 7),
      // Left (X=0): 0,7,3 and 0,4,7
      PlyMeshFace(0, 7, 3),
      PlyMeshFace(0, 4, 7),
      // Right (X=1): 1,6,5 and 1,2,6
      PlyMeshFace(1, 6, 5),
      PlyMeshFace(1, 2, 6),
      // Bottom (Y=0): 0,5,4 and 0,1,5
      PlyMeshFace(0, 5, 4),
      PlyMeshFace(0, 1, 5),
      // Top (Y=1): 3,6,2 and 3,7,6
      PlyMeshFace(3, 6, 2),
      PlyMeshFace(3, 7, 6),
  };
  return mesh;
}

TEST(MeshTextureMapping, EndToEnd) {
  const PlyMesh mesh = MakeTriangleMesh();
  const BitmapColor<uint8_t> red(200, 50, 50);
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, red));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // Check basic properties.
  EXPECT_GT(result.atlas_width, 0);
  EXPECT_GT(result.atlas_height, 0);
  EXPECT_EQ(result.face_uvs.size(), mesh.faces.size() * 6);
  EXPECT_EQ(result.face_view_ids.size(), mesh.faces.size());

  // Both faces should be assigned to view 0.
  EXPECT_EQ(result.face_view_ids[0], 0);
  EXPECT_EQ(result.face_view_ids[1], 0);

  // All UVs should be in [0, 1].
  for (size_t i = 0; i < result.face_uvs.size(); ++i) {
    EXPECT_GE(result.face_uvs[i], 0.0f);
    EXPECT_LE(result.face_uvs[i], 1.0f);
  }

  // Atlas should contain approximately the red color.
  // Check a pixel that should be baked.
  bool found_colored_pixel = false;
  for (int y = 0; y < result.atlas_height && !found_colored_pixel; ++y) {
    for (int x = 0; x < result.atlas_width && !found_colored_pixel; ++x) {
      const auto color = result.texture_atlas.GetPixel(x, y);
      if (color && color->r > 100 && color->g < 150 && color->b < 150) {
        found_colored_pixel = true;
      }
    }
  }
  EXPECT_TRUE(found_colored_pixel);
}

TEST(MeshTextureMapping, EmptyMesh) {
  PlyMesh mesh;
  std::vector<Image> images;
  images.push_back(MakeTestImage(64, 64, BitmapColor<uint8_t>(128)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;

  const auto result = MeshTextureMapping(mesh, images, options);

  EXPECT_EQ(result.atlas_width, 0);
  EXPECT_EQ(result.atlas_height, 0);
  EXPECT_TRUE(result.face_uvs.empty());
  EXPECT_TRUE(result.face_view_ids.empty());
}

TEST(MeshTextureMapping, NoVisibleFaces) {
  const PlyMesh mesh = MakeTriangleMesh();
  std::vector<Image> images;
  images.push_back(MakeBehindCameraImage(256, 256));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // All faces should be unassigned.
  for (size_t i = 0; i < result.face_view_ids.size(); ++i) {
    EXPECT_EQ(result.face_view_ids[i], -1);
  }
}

TEST(MeshTextureMapping, SingleFaceSingleView) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, BitmapColor<uint8_t>(100, 200, 50)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);
  EXPECT_EQ(result.face_view_ids[0], 0);
}

TEST(MeshTextureMapping, SingleFaceTwoViews) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  // Image 0: closer camera -> larger projected area.
  auto img_close = MakeTestImage(256, 256, BitmapColor<uint8_t>(255, 0, 0));

  // Image 1: farther camera -> smaller projected area.
  float K[9] = {0};
  K[0] = 256.0f;
  K[4] = 256.0f;
  K[2] = 128.0f;
  K[5] = 128.0f;
  K[8] = 1.0f;
  // R: 180° around X, same orientation as MakeTestImage.
  float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};
  // C = (0.5, 0.5, 20), T = -R*C = [-0.5, 0.5, 20]
  float T[3] = {-0.5f, 0.5f, 20.0f};
  Image img_far("far.png", 256, 256, K, R, T);
  Bitmap bitmap(256, 256, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 255, 0));
  img_far.SetBitmap(std::move(bitmap));

  std::vector<Image> images = {img_close, img_far};

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // Should pick the closer camera (view 0) as it has larger projected area.
  EXPECT_EQ(result.face_view_ids[0], 0);
}

TEST(MeshTextureMapping, FaceBehindCamera) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  std::vector<Image> images;
  images.push_back(MakeBehindCameraImage(256, 256));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;

  const auto result = MeshTextureMapping(mesh, images, options);
  EXPECT_EQ(result.face_view_ids[0], -1);
}

TEST(MeshTextureMapping, GrazingAngleRejected) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  std::vector<Image> images;
  images.push_back(MakeGrazingAngleImage(256, 256));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  // Set high threshold to reject grazing angles.
  options.min_cos_normal_angle = 0.9;

  const auto result = MeshTextureMapping(mesh, images, options);
  EXPECT_EQ(result.face_view_ids[0], -1);
}

TEST(MeshTextureMapping, NeighborSmoothing) {
  // Create a strip mesh centered around (0.5, 0.5) so all faces are visible.
  // Smaller extent so all vertices project within image bounds.
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.0f, 1.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
      PlyMeshVertex(1.0f, 1.0f, 0.0f),
  };
  mesh.faces = {
      PlyMeshFace(0, 1, 4),
      PlyMeshFace(0, 4, 3),
      PlyMeshFace(1, 2, 5),
      PlyMeshFace(1, 5, 4),
  };

  // Both views see all faces.
  auto img0 = MakeTestImage(512, 512, BitmapColor<uint8_t>(255, 0, 0));
  auto img1 = MakeTestImage(512, 512, BitmapColor<uint8_t>(0, 255, 0));

  std::vector<Image> images = {img0, img1};

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 3;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // After smoothing, all faces should be assigned since they're all visible.
  int assigned = 0;
  for (size_t i = 0; i < result.face_view_ids.size(); ++i) {
    if (result.face_view_ids[i] >= 0) ++assigned;
  }
  EXPECT_EQ(assigned, static_cast<int>(mesh.faces.size()));

  // All assigned faces should have the same view (smoothing promotes
  // uniformity).
  int first_view = result.face_view_ids[0];
  for (size_t i = 1; i < result.face_view_ids.size(); ++i) {
    EXPECT_EQ(result.face_view_ids[i], first_view);
  }
}

TEST(MeshTextureMapping, ViewSelectionPreservesScoresAndSmoothing) {
  const auto no_images =
      MeshTextureMapping(MakeTriangleMesh(), {}, MeshTextureMappingOptions());
  EXPECT_EQ(no_images.face_view_ids, (std::vector<int>{-1, -1}));
  EXPECT_EQ(no_images.face_uvs, std::vector<float>(12, 0.0f));
  EXPECT_EQ(no_images.atlas_width, 0);
  EXPECT_EQ(no_images.atlas_height, 0);

  const Image regular =
      MakeTestImage(64, 64, BitmapColor<uint8_t>(180, 70, 40));
  const auto MakeView = [&](const float distance) {
    const float T[3] = {-0.5f, 0.5f, distance};
    Image image("other.png", 64, 64, regular.GetK(), regular.GetR(), T);
    Bitmap bitmap(64, 64, /*as_rgb=*/true);
    bitmap.Fill(BitmapColor<uint8_t>(30, 150, 200));
    image.SetBitmap(std::move(bitmap));
    return image;
  };
  const Image far = MakeView(1e9f);
  const Image close = MakeView(1.5f);

  const auto CheckSelection = [](const PlyMesh& mesh,
                                 const std::vector<Image>& images,
                                 const int smoothing,
                                 const std::vector<int>& expected) {
    MeshTextureMappingOptions options;
    options.apply_color_correction = false;
    options.inpaint_radius = 0;
    options.view_selection_smoothing_iterations = smoothing;
    MeshTextureMappingResult reference;
    for (const int num_threads : {1, 3}) {
      SCOPED_TRACE(num_threads);
      options.num_threads = num_threads;
      auto result = MeshTextureMapping(mesh, images, options);
      ASSERT_EQ(result.face_view_ids, expected);
      ASSERT_EQ(result.face_uvs.size(), 6 * mesh.faces.size());
      for (const float uv : result.face_uvs) EXPECT_TRUE(std::isfinite(uv));
      if (num_threads == 1) {
        reference = std::move(result);
      } else {
        EXPECT_EQ(result.face_uvs, reference.face_uvs);
        EXPECT_EQ(result.atlas_width, reference.atlas_width);
        EXPECT_EQ(result.atlas_height, reference.atlas_height);
        EXPECT_EQ(result.texture_atlas.RowMajorData(),
                  reference.texture_atlas.RowMajorData());
      }
    }
  };

  PlyMesh mesh = MakeTriangleMesh();
  mesh.faces.emplace_back(0, 0, 0);  // No normal, hence no eligible view.
  for (const int smoothing : {0, 3}) {
    SCOPED_TRACE(smoothing);
    // Equal positive areas choose the first view; positive areas beat zero.
    CheckSelection(mesh, {regular, regular}, smoothing, {0, 0, -1});
    CheckSelection(mesh, {far, regular}, smoothing, {1, 1, -1});
  }

  // The thin face has a valid normal but collapses under float projection.
  // Zero area wins the initial argmax over -1, but must not enable smoothing.
  mesh.vertices = {PlyMeshVertex(0, 0, 0),
                   PlyMeshVertex(1, 0, 0),
                   PlyMeshVertex(0, 1e-8f, 0),
                   PlyMeshVertex(0, -1, 0)};
  mesh.faces = {PlyMeshFace(0, 1, 2), PlyMeshFace(1, 0, 3)};
  CheckSelection(mesh, {far, regular}, 0, {0, 1});
  CheckSelection(mesh, {far, regular}, 3, {0, 1});

  // Only the middle face fits in the close camera. Its two neighbors vote
  // for the wider view, which has a positive score for all three faces.
  mesh.vertices = {PlyMeshVertex(0, 0, 0),
                   PlyMeshVertex(1, 0, 0),
                   PlyMeshVertex(0, 1, 0),
                   PlyMeshVertex(1.5f, 1.5f, 0),
                   PlyMeshVertex(-1, 0, 0)};
  mesh.faces = {
      PlyMeshFace(0, 1, 2), PlyMeshFace(1, 3, 2), PlyMeshFace(0, 2, 4)};
  CheckSelection(mesh, {close, regular}, 0, {0, 1, 1});
  CheckSelection(mesh, {close, regular}, 3, {1, 1, 1});
}

TEST(MeshTextureMapping, CubeAdjacency) {
  // Verify the cube mesh produces faces with proper adjacency
  // by running the full pipeline.
  const PlyMesh mesh = MakeCubeMesh();
  std::vector<Image> images;

  // Camera looking at front face.
  images.push_back(MakeTestImage(512, 512, BitmapColor<uint8_t>(200, 100, 50)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  options.min_visible_vertices = 1;

  const auto result = MeshTextureMapping(mesh, images, options);

  // At least the front-facing triangles should be assigned.
  EXPECT_EQ(result.face_view_ids.size(), 12u);
  int assigned = 0;
  for (int v : result.face_view_ids) {
    if (v >= 0) ++assigned;
  }
  EXPECT_GT(assigned, 0);
}

TEST(MeshTextureMapping, UVRange) {
  const PlyMesh mesh = MakeTriangleMesh();
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, BitmapColor<uint8_t>(128)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  for (size_t i = 0; i < result.face_uvs.size(); ++i) {
    EXPECT_GE(result.face_uvs[i], 0.0f) << "UV index " << i;
    EXPECT_LE(result.face_uvs[i], 1.0f) << "UV index " << i;
  }
}

TEST(MeshTextureMapping, BakeSolidColor) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  const BitmapColor<uint8_t> src_color(180, 90, 45);
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, src_color));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // Find baked pixels and verify they match the source color.
  int baked_count = 0;
  for (int y = 0; y < result.atlas_height; ++y) {
    for (int x = 0; x < result.atlas_width; ++x) {
      const auto color = result.texture_atlas.GetPixel(x, y);
      ASSERT_TRUE(color.has_value());
      if (color->r != 0 || color->g != 0 || color->b != 0) {
        // Due to bilinear interpolation, allow some tolerance.
        EXPECT_NEAR(color->r, src_color.r, 5);
        EXPECT_NEAR(color->g, src_color.g, 5);
        EXPECT_NEAR(color->b, src_color.b, 5);
        ++baked_count;
      }
    }
  }
  EXPECT_GT(baked_count, 0);
}

TEST(MeshTextureMapping, InpaintFillsNearbyPixels) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, BitmapColor<uint8_t>(200, 100, 50)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 3;

  const auto result_inpaint = MeshTextureMapping(mesh, images, options);

  // Without inpainting for comparison.
  options.inpaint_radius = 0;
  const auto result_no_inpaint = MeshTextureMapping(mesh, images, options);

  // Count non-black pixels. Inpainted result should have more.
  int count_inpaint = 0, count_no_inpaint = 0;
  for (int y = 0; y < result_inpaint.atlas_height; ++y) {
    for (int x = 0; x < result_inpaint.atlas_width; ++x) {
      const auto color = result_inpaint.texture_atlas.GetPixel(x, y);
      if (color && (color->r != 0 || color->g != 0 || color->b != 0)) {
        ++count_inpaint;
      }
    }
  }
  for (int y = 0; y < result_no_inpaint.atlas_height; ++y) {
    for (int x = 0; x < result_no_inpaint.atlas_width; ++x) {
      const auto color = result_no_inpaint.texture_atlas.GetPixel(x, y);
      if (color && (color->r != 0 || color->g != 0 || color->b != 0)) {
        ++count_no_inpaint;
      }
    }
  }
  EXPECT_GT(count_inpaint, count_no_inpaint);
}

TEST(MeshTextureMapping, InpaintDoesNotOverwriteBaked) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  std::vector<Image> images;
  images.push_back(MakeTestImage(64, 64, BitmapColor<uint8_t>(0)));
  Bitmap pattern(64, 64, /*as_rgb=*/true);
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) {
      pattern.SetPixel(
          x,
          y,
          x < 32 ? BitmapColor<uint8_t>(0)
                 : BitmapColor<uint8_t>(static_cast<uint8_t>(1 + 3 * x),
                                        static_cast<uint8_t>(1 + 3 * y),
                                        200));
    }
  }
  images.back().SetBitmap(std::move(pattern));

  // Run without inpainting.
  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  const auto result_base = MeshTextureMapping(mesh, images, options);
  // A companion white render exposes coverage independently of the actual RGB.
  // Black source pixels are valid baked seeds, not holes in the atlas.
  const auto coverage = MeshTextureMapping(
      mesh, {MakeTestImage(64, 64, BitmapColor<uint8_t>(255))}, options);
  ASSERT_EQ(coverage.atlas_width, result_base.atlas_width);
  ASSERT_EQ(coverage.atlas_height, result_base.atlas_height);
  ASSERT_EQ(coverage.face_view_ids, result_base.face_view_ids);
  ASSERT_EQ(coverage.face_uvs, result_base.face_uvs);

  ASSERT_GT(result_base.atlas_width, 0);
  ASSERT_GT(result_base.atlas_height, 0);

  // The nearest covered seed in Manhattan distance wins, including black.
  // Equal distances retain the first seed in scan order; no queue is replayed.
  std::vector<uint8_t> baked_mask(result_base.atlas_width *
                                  result_base.atlas_height);
  for (int y = 0; y < result_base.atlas_height; ++y) {
    for (int x = 0; x < result_base.atlas_width; ++x) {
      const auto pixel = coverage.texture_atlas.GetPixel(x, y);
      ASSERT_TRUE(pixel.has_value());
      baked_mask[y * result_base.atlas_width + x] =
          *pixel != BitmapColor<uint8_t>(0);
    }
  }
  for (const int radius : {1, 3, 5, 1000}) {
    SCOPED_TRACE(radius);
    options.inpaint_radius = radius;
    const auto result_inpaint = MeshTextureMapping(mesh, images, options);
    ASSERT_EQ(result_inpaint.atlas_width, result_base.atlas_width);
    ASSERT_EQ(result_inpaint.atlas_height, result_base.atlas_height);
    ASSERT_EQ(result_inpaint.face_view_ids, result_base.face_view_ids);
    ASSERT_EQ(result_inpaint.face_uvs, result_base.face_uvs);
    size_t baked_count = 0;
    size_t black_baked_count = 0;
    size_t black_winners_against_color = 0;
    size_t filled_count = 0;
    size_t distinct_color_ties = 0;
    for (int y = 0; y < result_base.atlas_height; ++y) {
      for (int x = 0; x < result_base.atlas_width; ++x) {
        const auto base_color = result_base.texture_atlas.GetPixel(x, y);
        const auto inpaint_color = result_inpaint.texture_atlas.GetPixel(x, y);
        ASSERT_TRUE(base_color.has_value());
        ASSERT_TRUE(inpaint_color.has_value());
        auto expected = *base_color;
        if (baked_mask[y * result_base.atlas_width + x]) {
          ++baked_count;
          black_baked_count += expected == BitmapColor<uint8_t>(0);
        } else {
          int best_distance = options.inpaint_radius + 1;
          size_t pixel_ties = 0;
          bool colored_seed_in_range = false;
          for (int sy = std::max(0, y - options.inpaint_radius);
               sy <= std::min(result_base.atlas_height - 1,
                              y + options.inpaint_radius);
               ++sy) {
            for (int sx = std::max(0, x - options.inpaint_radius);
                 sx <= std::min(result_base.atlas_width - 1,
                                x + options.inpaint_radius);
                 ++sx) {
              const int distance = std::abs(sx - x) + std::abs(sy - y);
              if (distance > options.inpaint_radius) continue;
              const auto seed = result_base.texture_atlas.GetPixel(sx, sy);
              ASSERT_TRUE(seed.has_value());
              if (!baked_mask[sy * result_base.atlas_width + sx]) continue;
              colored_seed_in_range |= *seed != BitmapColor<uint8_t>(0);
              if (distance < best_distance) {
                best_distance = distance;
                expected = *seed;
                pixel_ties = 0;
              } else if (distance == best_distance && *seed != expected) {
                ++pixel_ties;
              }
            }
          }
          distinct_color_ties += pixel_ties;
          black_winners_against_color +=
              best_distance <= options.inpaint_radius &&
              expected == BitmapColor<uint8_t>(0) && colored_seed_in_range;
          filled_count += expected != BitmapColor<uint8_t>(0);
        }
        EXPECT_EQ(*inpaint_color, expected) << "atlas pixel " << x << ", " << y;
      }
    }
    EXPECT_GT(baked_count, 0);
    EXPECT_GT(black_baked_count, 0);
    // Ignoring black seeds must cause a visible error, not simply leave holes
    // black.
    if (radius == 5) {
      EXPECT_GT(black_winners_against_color, 0);
    }
    EXPECT_GT(filled_count, 0);
    if (radius == 5) {
      EXPECT_GT(distinct_color_ties, 0);
    }
  }
}

TEST(MeshTextureMapping, GlobalColorCorrectionNoSeams) {
  // Single view, single region -> no seams -> no correction applied.
  PlyMesh mesh = MakeTriangleMesh();
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, BitmapColor<uint8_t>(150)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = true;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result_with_cc = MeshTextureMapping(mesh, images, options);

  options.apply_color_correction = false;
  const auto result_without_cc = MeshTextureMapping(mesh, images, options);

  // Results should be identical since there are no seams.
  EXPECT_EQ(result_with_cc.atlas_width, result_without_cc.atlas_width);
  EXPECT_EQ(result_with_cc.atlas_height, result_without_cc.atlas_height);

  for (int y = 0; y < result_with_cc.atlas_height; ++y) {
    for (int x = 0; x < result_with_cc.atlas_width; ++x) {
      const auto c1 = result_with_cc.texture_atlas.GetPixel(x, y);
      const auto c2 = result_without_cc.texture_atlas.GetPixel(x, y);
      ASSERT_TRUE(c1.has_value());
      ASSERT_TRUE(c2.has_value());
      EXPECT_EQ(c1->r, c2->r);
      EXPECT_EQ(c1->g, c2->g);
      EXPECT_EQ(c1->b, c2->b);
    }
  }
}

TEST(MeshTextureMapping, GlobalColorCorrectionMatchesAnalyticSeam) {
  PlyMesh mesh;
  mesh.vertices = {PlyMeshVertex(0, 0, 0),
                   PlyMeshVertex(1, 0, 0),
                   PlyMeshVertex(0, 1, 0),
                   PlyMeshVertex(0, 0, 1)};
  mesh.faces = {PlyMeshFace(0, 1, 2), PlyMeshFace(1, 0, 3)};
  const float K[9] = {64, 0, 32, 0, 64, 32, 0, 0, 1};
  const float R_front[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};
  const float R_side[9] = {1, 0, 0, 0, 0, 1, 0, -1, 0};

  for (const int scenario : {0, 1, 2}) {
    SCOPED_TRACE(scenario);
    const bool skip_first_endpoint = scenario == 1;
    const bool equal_colors = scenario == 2;
    const std::array<BitmapColor<uint8_t>, 2> colors = {
        BitmapColor<uint8_t>(30, 100, 200),
        equal_colors ? BitmapColor<uint8_t>(30, 100, 200)
                     : BitmapColor<uint8_t>(210, 60, 80)};
    // In the boundary case, vertex 0 projects to x=0.25 in the front
    // image. Its bilinear sample at x=-0.25 is unavailable for all channels.
    const float T_front[3] = {
        skip_first_endpoint ? -2.48046875f : -0.5f, 0.5f, 5};
    const float T_side[3] = {-0.5f, -0.5f, 5};
    std::vector<Image> images;
    images.emplace_back("front.png", 64, 64, K, R_front, T_front);
    images.emplace_back("side.png", 64, 64, K, R_side, T_side);
    for (size_t i = 0; i < images.size(); ++i) {
      Bitmap bitmap(64, 64, /*as_rgb=*/true);
      bitmap.Fill(colors[i]);
      images[i].SetBitmap(std::move(bitmap));
    }

    MeshTextureMappingOptions options;
    options.min_cos_normal_angle = 0.5;
    options.view_selection_smoothing_iterations = 0;
    options.inpaint_radius = 0;
    options.num_threads = 1;
    options.apply_color_correction = false;
    const auto base = MeshTextureMapping(mesh, images, options);
    ASSERT_EQ(base.face_view_ids, std::vector<int>({0, 1}));
    const auto& base_pixels = base.texture_atlas.RowMajorData();

    for (const double beta : {0.1, 2.0}) {
      SCOPED_TRACE(beta);
      options.apply_color_correction = true;
      options.color_correction_regularization = beta;
      const auto result = MeshTextureMapping(mesh, images, options);
      ASSERT_EQ(result.face_view_ids, base.face_view_ids);
      ASSERT_EQ(result.face_uvs, base.face_uvs);
      ASSERT_EQ(result.atlas_width, base.atlas_width);
      ASSERT_EQ(result.atlas_height, base.atlas_height);
      std::vector<uint8_t> expected = base_pixels;
      size_t checked_pixels = 0;
      for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
        std::array<Eigen::Vector2d, 3> p;
        for (size_t vi = 0; vi < p.size(); ++vi) {
          p[vi] = Eigen::Vector2d(
              base.face_uvs[6 * fi + 2 * vi] * base.atlas_width,
              (1.0 - base.face_uvs[6 * fi + 2 * vi + 1]) * base.atlas_height);
        }
        const Eigen::Vector2d ab = p[1] - p[0];
        const Eigen::Vector2d ac = p[2] - p[0];
        const double area = ab.x() * ac.y() - ab.y() * ac.x();
        ASSERT_NE(area, 0);
        size_t face_pixels = 0;
        for (int y = 0; y < base.atlas_height; ++y) {
          for (int x = 0; x < base.atlas_width; ++x) {
            const Eigen::Vector2d q = Eigen::Vector2d(x + 0.5, y + 0.5) - p[0];
            const double w1 = (q.x() * ac.y() - q.y() * ac.x()) / area;
            const double w2 = (ab.x() * q.y() - ab.y() * q.x()) / area;
            const double w0 = 1 - w1 - w2;
            if (std::min({w0, w1, w2}) < -1e-4) continue;
            const auto color = base.texture_atlas.GetPixel(x, y);
            ASSERT_TRUE(color.has_value());
            if (*color == BitmapColor<uint8_t>(0)) continue;
            ASSERT_EQ(*color, colors[fi]);
            ++face_pixels;
            // Each available shared vertex has the independent 2x2 system
            // [[1+beta,-1],[-1,1+beta]], hence offsets +/-difference/(beta+2).
            // The unshared vertex and any unsampled endpoint have zero offset.
            const double weight =
                skip_first_endpoint ? (fi == 0 ? w1 : w0) : w0 + w1;
            const std::array<double, 3> difference = {
                double(colors[1].r) - colors[0].r,
                double(colors[1].g) - colors[0].g,
                double(colors[1].b) - colors[0].b};
            const size_t pixel =
                3 * (static_cast<size_t>(y) * base.atlas_width + x);
            for (size_t ch = 0; ch < difference.size(); ++ch) {
              const double offset =
                  (fi == 0 ? 1 : -1) * weight * difference[ch] / (beta + 2);
              expected[pixel + ch] = static_cast<uint8_t>(
                  std::clamp(expected[pixel + ch] + offset, 0.0, 255.0));
            }
          }
        }
        EXPECT_GT(face_pixels, 0);
        checked_pixels += face_pixels;
      }
      EXPECT_GT(checked_pixels, 0);
      const auto& actual = result.texture_atlas.RowMajorData();
      ASSERT_EQ(actual.size(), expected.size());
      size_t changed_bytes = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        // The oracle reconstructs atlas vertices from normalized UVs and uses
        // double barycentrics; allow one final byte of raster quantization.
        EXPECT_NEAR(actual[i], expected[i], 1);
        changed_bytes += actual[i] != base_pixels[i];
        if (base_pixels[i] == 0) {
          EXPECT_EQ(actual[i], 0);
        }
      }
      if (equal_colors) {
        EXPECT_EQ(changed_bytes, 0);
      } else {
        EXPECT_GT(changed_bytes, 0);
      }
    }
  }
}

TEST(MeshTextureMapping, PyramidMultiFace) {
  const PlyMesh mesh = MakePyramidMesh();
  std::vector<Image> images;
  images.push_back(
      MakeTestImage(512, 512, BitmapColor<uint8_t>(100, 150, 200)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  options.min_visible_vertices = 1;

  const auto result = MeshTextureMapping(mesh, images, options);
  EXPECT_EQ(result.face_view_ids.size(), 4u);
  EXPECT_EQ(result.face_uvs.size(), 24u);

  // At least some faces should be assigned.
  int assigned = 0;
  for (int v : result.face_view_ids) {
    if (v >= 0) ++assigned;
  }
  EXPECT_GT(assigned, 0);
}

TEST(MeshTextureMapping, SamplesAtPixelCenters) {
  // With the MakeTestImage camera (fx = fy = 256, cx = cy = 128,
  // R = 180-degree rotation about X, C = (0.5, 0.5, 5)), these vertices
  // project exactly onto source texel centers: V0 -> (100.5, 100.5),
  // V1 -> (100.5, 110.5), V2 -> (110.5, 100.5). All values are exactly
  // representable in floating point, so the projections are exact.
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(-19.0f / 512, 531.0f / 512, 0.0f),
      PlyMeshVertex(-19.0f / 512, 431.0f / 512, 0.0f),
      PlyMeshVertex(81.0f / 512, 531.0f / 512, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  // Source image: white texel at (100, 100), black elsewhere.
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, BitmapColor<uint8_t>(0)));
  Bitmap bitmap(256, 256, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(0));
  bitmap.SetPixel(100, 100, BitmapColor<uint8_t>(255));
  images.back().SetBitmap(std::move(bitmap));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);
  ASSERT_EQ(result.face_view_ids.size(), 1u);
  ASSERT_EQ(result.face_view_ids[0], 0);
  ASSERT_EQ(result.face_uvs.size(), 6u);

  // The atlas texel under vertex 0 must reproduce the white source texel.
  // Without the 0.5 sampling correction it samples gray instead.
  const int px = static_cast<int>(
      std::round(result.face_uvs[0] * result.atlas_width - 0.5));
  const int py = static_cast<int>(
      std::round((1.0f - result.face_uvs[1]) * result.atlas_height - 0.5));
  const auto color = result.texture_atlas.GetPixel(px, py);
  ASSERT_TRUE(color.has_value());
  EXPECT_EQ(color->r, 255);
  EXPECT_EQ(color->g, 255);
  EXPECT_EQ(color->b, 255);
}

TEST(MeshTextureMapping, RotatedCameraCenterSelection) {
  const Eigen::Matrix3f rotation =
      (Eigen::AngleAxisf(0.23f, Eigen::Vector3f::UnitZ()) *
       Eigen::AngleAxisf(0.12f, Eigen::Vector3f::UnitY()) *
       Eigen::AngleAxisf(3.03f, Eigen::Vector3f::UnitX()))
          .toRotationMatrix();
  const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> row_rotation = rotation;
  const float K[9] = {256, 0, 128, 0, 256, 128, 0, 0, 1};
  std::vector<Image> images;
  for (const float z : {5.0f, 12.0f}) {
    const Eigen::Vector3f center(0.5f, 0.5f, z);
    const Eigen::Vector3f translation = -rotation * center;
    Image image(
        "rotated.png", 256, 256, K, row_rotation.data(), translation.data());
    Bitmap bitmap(256, 256, /*as_rgb=*/true);
    bitmap.Fill(BitmapColor<uint8_t>(100, 150, 200));
    image.SetBitmap(std::move(bitmap));
    images.push_back(std::move(image));
  }
  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.inpaint_radius = 0;
  options.view_selection_smoothing_iterations = 0;
  options.num_threads = 1;
  const auto result = MeshTextureMapping(MakeTriangleMesh(), images, options);
  EXPECT_EQ(result.face_view_ids, (std::vector<int>{0, 0}));
  ASSERT_EQ(result.face_uvs.size(), 12);
  for (float uv : result.face_uvs) EXPECT_TRUE(std::isfinite(uv));
  const auto far_only =
      MeshTextureMapping(MakeTriangleMesh(), {images[1]}, options);
  EXPECT_EQ(far_only.face_view_ids, (std::vector<int>{0, 0}));
  std::reverse(images.begin(), images.end());
  const auto reversed = MeshTextureMapping(MakeTriangleMesh(), images, options);
  EXPECT_EQ(reversed.face_view_ids, (std::vector<int>{1, 1}));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
