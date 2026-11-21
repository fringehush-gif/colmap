// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/texture_mapping.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gmock/gmock.h>
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
Image MakeTestImage(int width,
                    int height,
                    const BitmapColor<uint8_t>& color,
                    const float translation = 0.0f,
                    const float focal_scale = 1.0f) {
  // K: focal length = width, principal point at center.
  float K[9] = {0};
  K[0] = focal_scale * static_cast<float>(width);   // fx
  K[4] = focal_scale * static_cast<float>(height);  // fy
  K[2] = static_cast<float>(width) / 2.0f;          // cx
  K[5] = static_cast<float>(height) / 2.0f;         // cy
  K[8] = 1.0f;

  // R: 180° rotation around X-axis. Row-major: [1,0,0, 0,-1,0, 0,0,-1].
  // This makes the camera look along -Z in world, with +Y up.
  float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};

  // C = (0.5, 0.5, 5), T = -R*C.
  // T = -[1,0,0;0,-1,0;0,0,-1] * [0.5,0.5,5] = [-0.5, 0.5, 5]
  // Apply the optional world translation to all three coordinates.
  float T[3] = {-0.5f - translation, 0.5f + translation, 5.0f + translation};

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

PlyMesh MakeAtlasTestMesh(const int width, const int height) {
  const float dx = 5.0f * (width - 1) / (64 * 8192);
  const float dy = 5.0f * (height - 1) / (64 * 8192);
  PlyMesh mesh;
  mesh.vertices = {PlyMeshVertex(0.5f, 0.5f, 0),
                   PlyMeshVertex(0.5f + dx, 0.5f, 0),
                   PlyMeshVertex(0.5f, 0.5f + dy, 0)};
  mesh.faces = {PlyMeshFace(0, 1, 2)};
  return mesh;
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

TEST(MeshTextureMapping, OccludedViewSelection) {
  PlyMesh mesh;
  mesh.vertices = {PlyMeshVertex(0, 0, 0),
                   PlyMeshVertex(1, 0, 0),
                   PlyMeshVertex(0, 1, 0),
                   PlyMeshVertex(-0.5f, -0.5f, 2),
                   PlyMeshVertex(0.5f, 1.5f, 2),
                   PlyMeshVertex(1.5f, -0.5f, 2)};
  // The blocker faces away from both cameras but still occludes the target.
  mesh.faces = {PlyMeshFace(0, 1, 2), PlyMeshFace(3, 4, 5)};
  const Image front = MakeTestImage(64, 64, BitmapColor<uint8_t>(200, 50, 50));
  const float K[9] = {16, 0, 32, 0, 16, 32, 0, 0, 1};
  const float T[3] = {-5, 0.5f, 5};
  Image lateral("lateral.png", 64, 64, K, front.GetR(), T);
  Bitmap bitmap(64, 64, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(50, 200, 50));
  lateral.SetBitmap(std::move(bitmap));
  const std::vector<Image> images = {front, lateral};

  // Front-camera rays intersect the blocker at z=2; lateral-camera rays
  // pass beyond its right edge. Without occlusion testing, the larger
  // projected area of the front view wins instead.
#if defined(COLMAP_CGAL_ENABLED)
  constexpr int kExpectedView = 1;
#else
  constexpr int kExpectedView = 0;
#endif
  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.inpaint_radius = 0;
  MeshTextureMappingResult reference;
  for (const int smoothing : {0, 3}) {
    SCOPED_TRACE(smoothing);
    options.view_selection_smoothing_iterations = smoothing;
    for (const int num_threads : {1, 3}) {
      SCOPED_TRACE(num_threads);
      options.num_threads = num_threads;
      auto result = MeshTextureMapping(mesh, images, options);
      ASSERT_EQ(result.face_view_ids, (std::vector<int>{kExpectedView, -1}));
      ASSERT_GT(result.atlas_width, 0);
      ASSERT_GT(result.atlas_height, 0);
      if (smoothing == 0 && num_threads == 1) {
        reference = std::move(result);
      } else {
        EXPECT_EQ(result.face_uvs, reference.face_uvs);
        EXPECT_EQ(result.atlas_width, reference.atlas_width);
        EXPECT_EQ(result.atlas_height, reference.atlas_height);
        EXPECT_EQ(result.texture_atlas.RowMajorData(),
                  reference.texture_atlas.RowMajorData());
      }
    }
  }
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
  const float K[9] = {64, 0, 32, 0, 64, 32, 0, 0, 1};
  const float R_front[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};
  const float R_side[9] = {1, 0, 0, 0, 0, 1, 0, -1, 0};
  const float R_right[9] = {0, 1, 0, 0, 0, -1, -1, 0, 0};

  for (const int scenario : {0, 1, 2, 3, 4, 5, 6}) {
    PlyMesh mesh;
    mesh.vertices = {PlyMeshVertex(0, 0, 0),
                     PlyMeshVertex(1, 0, 0),
                     PlyMeshVertex(0, 1, 0),
                     PlyMeshVertex(0, 0, 1)};

    SCOPED_TRACE(scenario);
    const bool skip_first_endpoint = scenario == 1 || scenario == 5;
    const bool skip_all_endpoints = scenario == 6;
    const bool equal_colors = scenario == 2;
    const bool three_regions = scenario == 3;
    const bool split_front = scenario == 4 || scenario == 5;
    mesh.faces = {PlyMeshFace(0, 1, 2), PlyMeshFace(1, 0, 3)};
    if (split_front) {
      mesh.vertices.emplace_back(0.5f, 0.5f, 0.0f);
      mesh.faces = {
          PlyMeshFace(0, 1, 4), PlyMeshFace(0, 4, 2), PlyMeshFace(1, 0, 3)};
    }
    std::vector<BitmapColor<uint8_t>> colors = {
        BitmapColor<uint8_t>(30, 100, 200),
        equal_colors ? BitmapColor<uint8_t>(30, 100, 200)
                     : BitmapColor<uint8_t>(210, 60, 80)};
    // In the boundary case, vertex 0 projects to x=0.25 in the front
    // image. Its bilinear sample at x=-0.25 is unavailable for all channels.
    // At y=63.75, both seam vertices are visible but their bilinear samples
    // need a row outside the image. The remaining front vertex stays inside.
    const float T_front[3] = {skip_first_endpoint ? -2.48046875f : -0.5f,
                              skip_all_endpoints ? 2.48046875f : 0.5f,
                              5};
    const float T_side[3] = {-0.5f, -0.5f, 5};
    std::vector<Image> images;
    images.emplace_back("front.png", 64, 64, K, R_front, T_front);
    images.emplace_back("side.png", 64, 64, K, R_side, T_side);
    if (three_regions) {
      mesh.faces.emplace_back(0, 2, 3);
      colors.emplace_back(90, 220, 40);
      images.emplace_back("right.png", 64, 64, K, R_right, T_front);
    }
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
    ASSERT_EQ(base.face_view_ids,
              three_regions ? std::vector<int>({0, 1, 2})
              : split_front ? std::vector<int>({0, 0, 1})
                            : std::vector<int>({0, 1}));
    const auto& base_pixels = base.texture_atlas.RowMajorData();

    for (const double beta : {0.001, 0.1, 2.0, 100.0}) {
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
      size_t shared_pixels = 0;
      std::vector<bool> covered(base_pixels.size() / 3, false);
      for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
        const int view_id = base.face_view_ids[fi];
        const auto& face = mesh.faces[fi];
        const std::array<size_t, 3> vertices = {
            face.vertex_idx1, face.vertex_idx2, face.vertex_idx3};
        std::array<Eigen::Vector3d, 3> vertex_offsets;
        for (size_t vi = 0; vi < vertices.size(); ++vi) {
          const Eigen::Vector3d color(
              colors[view_id].r, colors[view_id].g, colors[view_id].b);
          Eigen::Vector3d difference = Eigen::Vector3d::Zero();
          int num_neighbors = 0;
          std::vector<bool> neighboring_views(images.size(), false);
          if (!skip_all_endpoints &&
              !(skip_first_endpoint && vertices[vi] == 0)) {
            for (size_t other = 0; other < mesh.faces.size(); ++other) {
              const auto& other_face = mesh.faces[other];
              const int other_view = base.face_view_ids[other];
              if (other_view == view_id || neighboring_views[other_view] ||
                  (vertices[vi] != other_face.vertex_idx1 &&
                   vertices[vi] != other_face.vertex_idx2 &&
                   vertices[vi] != other_face.vertex_idx3)) {
                continue;
              }
              difference += Eigen::Vector3d(colors[other_view].r,
                                            colors[other_view].g,
                                            colors[other_view].b) -
                            color;
              neighboring_views[other_view] = true;
              ++num_neighbors;
            }
          }
          // At each vertex these two/three regions form a complete graph.
          // Its zero-sum color difference has eigenvalue num_regions + beta.
          vertex_offsets[vi] = difference / (num_neighbors + 1 + beta);
        }
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
            ASSERT_EQ(*color, colors[view_id]);
            ++face_pixels;
            const Eigen::Vector3d offset = w0 * vertex_offsets[0] +
                                           w1 * vertex_offsets[1] +
                                           w2 * vertex_offsets[2];
            const size_t pixel =
                3 * (static_cast<size_t>(y) * base.atlas_width + x);
            // A face writes a corrected source sample, never a correction
            // added to another face's already-corrected shared-edge sample.
            shared_pixels += covered[pixel / 3];
            covered[pixel / 3] = true;
            for (size_t ch = 0; ch < 3; ++ch) {
              expected[pixel + ch] = static_cast<uint8_t>(
                  std::clamp(base_pixels[pixel + ch] + offset[ch], 0.0, 255.0));
            }
          }
        }
        EXPECT_GT(face_pixels, 0);
        checked_pixels += face_pixels;
      }
      EXPECT_GT(checked_pixels, 0);
      if (split_front && !skip_first_endpoint) {
        EXPECT_GT(shared_pixels, 0);
      }
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
      if (equal_colors || skip_all_endpoints) {
        EXPECT_EQ(changed_bytes, 0);
      } else {
        EXPECT_GT(changed_bytes, 0);
      }
    }

    // With accepted seam samples, a tiny beta is lost when added to unit
    // coefficients and the singular-system fallback leaves every channel
    // unchanged. With no samples, subnormal beta can instead overflow the
    // inverse diagonal despite a zero RHS. Positive infinity is accepted by
    // Check() and must also yield the original bake.
    for (const double beta : {1e-30,
                              std::numeric_limits<double>::denorm_min(),
                              std::numeric_limits<double>::infinity()}) {
      SCOPED_TRACE(beta);
      options.color_correction_regularization = beta;
      const auto unchanged = MeshTextureMapping(mesh, images, options);
      EXPECT_EQ(unchanged.face_view_ids, base.face_view_ids);
      EXPECT_EQ(unchanged.face_uvs, base.face_uvs);
      EXPECT_EQ(unchanged.texture_atlas.RowMajorData(), base_pixels);
    }
  }
}

TEST(MeshTextureMapping, GlobalColorCorrectionInteriorAndRepeatedSeams) {
  // Two tessellated sheets meet along one edge. Internal seam vertices have
  // two observations of the same region pair; the endpoints have only one.
  constexpr size_t kSide = 4;
  PlyMesh mesh;
  for (size_t region = 0; region < 2; ++region) {
    const auto vertex = [region](const size_t x, const size_t y) {
      if (region == 0 || y == 0) return y * (kSide + 1) + x;
      return (kSide + 1) * (kSide + 1) + (y - 1) * (kSide + 1) + x;
    };
    for (size_t y = region; y <= kSide; ++y) {
      for (size_t x = 0; x <= kSide; ++x) {
        mesh.vertices.emplace_back(
            static_cast<float>(x) / kSide,
            region == 0 ? static_cast<float>(y) / kSide : 0,
            region == 1 ? static_cast<float>(y) / kSide : 0);
      }
    }
    for (size_t y = 0; y < kSide; ++y) {
      for (size_t x = 0; x < kSide; ++x) {
        const size_t a = vertex(x, y);
        const size_t b = vertex(x + 1, y);
        const size_t c = vertex(x, y + 1);
        const size_t d = vertex(x + 1, y + 1);
        mesh.faces.emplace_back(region == 0 ? a : b, region == 0 ? b : a, c);
        mesh.faces.emplace_back(region == 0 ? b : d, region == 0 ? d : b, c);
      }
    }
  }
  // A disconnected region uses the front image but has no seam variables.
  const size_t isolated = mesh.vertices.size();
  mesh.vertices.emplace_back(-1, 1, 0);
  mesh.vertices.emplace_back(-0.5f, 1, 0);
  mesh.vertices.emplace_back(-1, 1.5f, 0);
  mesh.faces.emplace_back(isolated, isolated + 1, isolated + 2);

  const std::array<BitmapColor<uint8_t>, 2> colors = {
      BitmapColor<uint8_t>(30, 100, 200), BitmapColor<uint8_t>(210, 60, 80)};
  std::vector<Image> images;
  images.push_back(MakeTestImage(128, 128, colors[0]));
  const float K[9] = {128, 0, 64, 0, 128, 64, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 0, 1, 0, -1, 0};
  const float T[3] = {-0.5f, -0.5f, 5};
  images.emplace_back("side.png", 128, 128, K, R, T);
  Bitmap side_image(128, 128, /*as_rgb=*/true);
  side_image.Fill(colors[1]);
  images.back().SetBitmap(std::move(side_image));

  MeshTextureMappingOptions options;
  options.min_cos_normal_angle = 0.5;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  options.num_threads = 1;
  options.apply_color_correction = false;
  const auto base = MeshTextureMapping(mesh, images, options);
  std::vector<int> expected_views(mesh.faces.size(), 0);
  std::fill(
      expected_views.begin() + 2 * kSide * kSide, expected_views.end() - 1, 1);
  ASSERT_EQ(base.face_view_ids, expected_views);
  options.apply_color_correction = true;
  options.color_correction_regularization = 2;
  const auto result = MeshTextureMapping(mesh, images, options);
  ASSERT_EQ(result.face_view_ids, base.face_view_ids);
  ASSERT_EQ(result.face_uvs, base.face_uvs);
  ASSERT_EQ(result.atlas_width, base.atlas_width);
  ASSERT_EQ(result.atlas_height, base.atlas_height);

  size_t interior_faces = 0;
  size_t changed_faces = 0;
  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    SCOPED_TRACE(fi);
    const int view = expected_views[fi];
    const auto& face = mesh.faces[fi];
    const std::array<size_t, 3> vertices = {
        face.vertex_idx1, face.vertex_idx2, face.vertex_idx3};
    std::array<Eigen::Vector2d, 3> p;
    std::array<Eigen::Vector3d, 3> offsets;
    for (size_t vi = 0; vi < vertices.size(); ++vi) {
      p[vi] = Eigen::Vector2d(
          base.face_uvs[6 * fi + 2 * vi] * base.atlas_width,
          (1.0 - base.face_uvs[6 * fi + 2 * vi + 1]) * base.atlas_height);
      offsets[vi].setZero();
      if (vertices[vi] <= kSide) {
        const int observations =
            vertices[vi] == 0 || vertices[vi] == kSide ? 1 : 2;
        const auto& own = colors[view];
        const auto& other = colors[1 - view];
        // For weight m, the two-region system gives m*(other-own)/(2*m+beta).
        offsets[vi] =
            observations *
            Eigen::Vector3d(other.r - own.r, other.g - own.g, other.b - own.b) /
            (2 * observations + options.color_correction_regularization);
      }
    }
    const Eigen::Vector2d center = (p[0] + p[1] + p[2]) / 3;
    const int x = static_cast<int>(std::floor(center.x()));
    const int y = static_cast<int>(std::floor(center.y()));
    const Eigen::Vector2d ab = p[1] - p[0];
    const Eigen::Vector2d ac = p[2] - p[0];
    const Eigen::Vector2d q = Eigen::Vector2d(x + 0.5, y + 0.5) - p[0];
    const double area = ab.x() * ac.y() - ab.y() * ac.x();
    ASSERT_NE(area, 0);
    const double w1 = (q.x() * ac.y() - q.y() * ac.x()) / area;
    const double w2 = (ab.x() * q.y() - ab.y() * q.x()) / area;
    const double w0 = 1 - w1 - w2;
    ASSERT_GT(std::min({w0, w1, w2}), 0);
    const auto base_color = base.texture_atlas.GetPixel(x, y);
    const auto actual = result.texture_atlas.GetPixel(x, y);
    ASSERT_TRUE(base_color.has_value());
    ASSERT_TRUE(actual.has_value());
    ASSERT_EQ(*base_color, colors[view]);
    const Eigen::Vector3d offset =
        w0 * offsets[0] + w1 * offsets[1] + w2 * offsets[2];
    const BitmapColor<uint8_t> expected(
        static_cast<uint8_t>(colors[view].r + offset.x()),
        static_cast<uint8_t>(colors[view].g + offset.y()),
        static_cast<uint8_t>(colors[view].b + offset.z()));
    EXPECT_NEAR(actual->r, expected.r, 1);
    EXPECT_NEAR(actual->g, expected.g, 1);
    EXPECT_NEAR(actual->b, expected.b, 1);
    if (offset.isZero()) {
      EXPECT_EQ(*actual, *base_color);
      ++interior_faces;
    } else {
      EXPECT_NE(*actual, *base_color);
      ++changed_faces;
    }
  }
  EXPECT_EQ(interior_faces, 4 * kSide * (kSide - 1) + 1);
  EXPECT_EQ(changed_faces, 4 * kSide);
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

TEST(MeshTextureMapping, OcclusionWithSkippedTriangles) {
  PlyMesh mesh;
  mesh.vertices = {PlyMeshVertex(0, 0, 0),
                   PlyMeshVertex(1, 0, 0),
                   PlyMeshVertex(0, 1, 0),
                   PlyMeshVertex(-0.5f, -0.5f, 2),
                   PlyMeshVertex(0.5f, 1.5f, 2),
                   PlyMeshVertex(1.5f, -0.5f, 2)};
  // Skipping the first triangle must preserve the remaining face indices.
  // The reverse-wound blocker is not textured but still occludes the target.
  mesh.faces = {
      PlyMeshFace(0, 0, 0), PlyMeshFace(0, 1, 2), PlyMeshFace(3, 4, 5)};
  const Image front = MakeTestImage(64, 64, BitmapColor<uint8_t>(200, 50, 50));
  const float K[9] = {16, 0, 32, 0, 16, 32, 0, 0, 1};
  const float T[3] = {-5, 0.5f, 5};
  Image lateral("lateral.png", 64, 64, K, front.GetR(), T);
  Bitmap bitmap(64, 64, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(50, 200, 50));
  lateral.SetBitmap(std::move(bitmap));
  const std::vector<Image> images = {front, lateral};

  // Front-camera rays hit the blocker. Lateral-camera rays pass to its right.
#if defined(COLMAP_CGAL_ENABLED)
  constexpr int kExpectedView = 1;
#else
  constexpr int kExpectedView = 0;
#endif
  const auto expected_color = images[kExpectedView].GetBitmap().GetPixel(0, 0);
  ASSERT_TRUE(expected_color.has_value());
  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.inpaint_radius = 0;
  MeshTextureMappingResult reference;
  for (const int num_threads : {1, 3}) {
    SCOPED_TRACE(num_threads);
    options.num_threads = num_threads;
    auto result = MeshTextureMapping(mesh, images, options);
    ASSERT_EQ(result.face_view_ids, (std::vector<int>{-1, kExpectedView, -1}));
    size_t num_colored = 0;
    for (int y = 0; y < result.atlas_height; ++y) {
      for (int x = 0; x < result.atlas_width; ++x) {
        const auto color = result.texture_atlas.GetPixel(x, y);
        ASSERT_TRUE(color.has_value());
        if (color->r != 0 || color->g != 0 || color->b != 0) {
          EXPECT_EQ(color->r, expected_color->r);
          EXPECT_EQ(color->g, expected_color->g);
          EXPECT_EQ(color->b, expected_color->b);
          ++num_colored;
        }
      }
    }
    EXPECT_GT(num_colored, 0);
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
}

TEST(MeshTextureMapping, SharedEdgeIncidence) {
  const std::vector<Image> images = {
      MakeTestImage(64, 64, BitmapColor<uint8_t>(128))};
  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  options.num_threads = 1;

  for (const bool duplicate_face : {false, true}) {
    for (const size_t num_faces : {2, 3, 4, 258}) {
      SCOPED_TRACE(duplicate_face);
      SCOPED_TRACE(num_faces);
      PlyMesh mesh = MakeTriangleMesh();
      if (duplicate_face) mesh.faces[1] = mesh.faces[0];
      mesh.faces.resize(num_faces, mesh.faces[0]);
      const auto result = MeshTextureMapping(mesh, images, options);
      ASSERT_EQ(result.face_view_ids, std::vector<int>(num_faces, 0));
      ASSERT_EQ(result.face_uvs.size(), num_faces * 6);
      // Shared vertices have the same UV only when the faces share a region.
      const bool same_uv = result.face_uvs[0] == result.face_uvs[6] &&
                           result.face_uvs[1] == result.face_uvs[7];
      EXPECT_EQ(same_uv, num_faces == 2);
    }
  }

  // Count edge occurrences, including repeated edges within a degenerate face.
  PlyMesh mesh = MakeTriangleMesh();
  mesh.faces.emplace_back(0, 2, 2);
  const auto result = MeshTextureMapping(mesh, images, options);
  ASSERT_EQ(result.face_view_ids, (std::vector<int>{0, 0, -1}));
  ASSERT_EQ(result.face_uvs.size(), 18);
  EXPECT_TRUE(result.face_uvs[0] != result.face_uvs[6] ||
              result.face_uvs[1] != result.face_uvs[7]);
}

TEST(MeshTextureMapping, SmoothingUsesPreviousIteration) {
  // Both views see both sides of this ridge. Each face initially prefers
  // its nearer view, then takes its only neighbor's label every iteration.
  PlyMesh mesh;
  mesh.vertices = {PlyMeshVertex(-1, -1, 0),
                   PlyMeshVertex(0, -1, 1),
                   PlyMeshVertex(0, 1, 1),
                   PlyMeshVertex(1, -1, 0),
                   PlyMeshVertex(0, 0, -1)};
  mesh.faces = {
      PlyMeshFace(0, 1, 2), PlyMeshFace(1, 3, 2), PlyMeshFace(4, 4, 4)};

  const float K[9] = {64, 0, 128, 0, 64, 128, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};
  std::vector<Image> images;
  for (const float camera_x : {-3.0f, 3.0f}) {
    const float T[3] = {-camera_x, 0, 5};
    images.emplace_back("ridge.png", 256, 256, K, R, T);
    Bitmap bitmap(256, 256, /*as_rgb=*/true);
    bitmap.Fill(camera_x < 0 ? BitmapColor<uint8_t>(180, 70, 40)
                             : BitmapColor<uint8_t>(30, 150, 200));
    images.back().SetBitmap(std::move(bitmap));
  }

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.inpaint_radius = 0;
  options.num_threads = 1;
  std::array<MeshTextureMappingResult, 2> references;
  for (const int iterations : {0, 1, 2, 3, 5}) {
    SCOPED_TRACE(iterations);
    options.view_selection_smoothing_iterations = iterations;
    auto result = MeshTextureMapping(mesh, images, options);
    const int parity = iterations % 2;
    ASSERT_EQ(result.face_view_ids, (std::vector<int>{parity, 1 - parity, -1}));
    ASSERT_GT(result.atlas_width, 0);
    ASSERT_GT(result.atlas_height, 0);
    ASSERT_EQ(result.face_uvs.size(), 18);
    for (const float uv : result.face_uvs) EXPECT_TRUE(std::isfinite(uv));
    EXPECT_EQ(
        std::vector<float>(result.face_uvs.begin() + 12, result.face_uvs.end()),
        std::vector<float>(6, 0.0f));
    if (iterations < 2) {
      references[parity] = std::move(result);
    } else {
      const auto& reference = references[parity];
      EXPECT_EQ(result.face_uvs, reference.face_uvs);
      EXPECT_EQ(result.atlas_width, reference.atlas_width);
      EXPECT_EQ(result.atlas_height, reference.atlas_height);
      EXPECT_EQ(result.texture_atlas.RowMajorData(),
                reference.texture_atlas.RowMajorData());
    }
  }
}

TEST(MeshTextureMapping, RejectsInvalidTextureScale) {
  const PlyMesh mesh = MakeTriangleMesh();
  const std::vector<Image> images = {
      MakeTestImage(64, 64, BitmapColor<uint8_t>(128))};
  MeshTextureMappingOptions options;
  const double max_scale = std::numeric_limits<float>::max();
  const double min_scale = 1.0 / max_scale;
  for (const double scale :
       {std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::max(),
        1e-100,
        std::nextafter(min_scale, 0.0),
        std::nextafter(max_scale, std::numeric_limits<double>::infinity())}) {
    SCOPED_TRACE(scale);
    options.texture_scale_factor = scale;
    EXPECT_FALSE(options.Check());
    EXPECT_THAT([&] { MeshTextureMapping(mesh, images, options); },
                testing::ThrowsMessage<std::invalid_argument>(
                    testing::HasSubstr("options.Check()")));
  }
  for (const double scale : {min_scale, 1.0, max_scale}) {
    SCOPED_TRACE(scale);
    options.texture_scale_factor = scale;
    EXPECT_TRUE(options.Check());
    EXPECT_TRUE(std::isfinite(static_cast<float>(scale)));
    // BakeTexture divides in double precision before narrowing the reciprocal.
    EXPECT_TRUE(std::isfinite(static_cast<float>(1.0 / scale)));
  }
}

TEST(MeshTextureMapping, RejectsScaledProjectionOverflow) {
  const PlyMesh mesh = MakeTriangleMesh();
  const std::vector<Image> images = {
      MakeTestImage(64, 64, BitmapColor<uint8_t>(128))};
  MeshTextureMappingOptions options;
  options.num_threads = 1;
  for (const double scale : {1e10, double{std::numeric_limits<float>::max()}}) {
    SCOPED_TRACE(scale);
    options.texture_scale_factor = scale;
    ASSERT_TRUE(options.Check());
    EXPECT_THAT(
        [&] { MeshTextureMapping(mesh, images, options); },
        testing::ThrowsMessage<std::invalid_argument>(testing::HasSubstr(
            "Projected texture coordinates exceed integer range")));
  }
}

TEST(MeshTextureMapping, RejectsOffImageProjectionOverflow) {
  const std::vector<Image> images = {
      MakeTestImage(64, 64, BitmapColor<uint8_t>(128), 0.0f, 1e9f)};
  MeshTextureMappingOptions options;
  options.num_threads = 1;
  options.min_visible_vertices = 1;
  for (const float sign : {-1.0f, 1.0f}) {
    SCOPED_TRACE(sign);
    PlyMesh mesh = MakeAtlasTestMesh(65537, 2);
    mesh.vertices[1].x = 0.5f + sign * 0.625f;
    if (sign < 0) {
      mesh.faces = {PlyMeshFace(0, 2, 1)};
    }
    EXPECT_THAT(
        [&] { MeshTextureMapping(mesh, images, options); },
        testing::ThrowsMessage<std::invalid_argument>(testing::HasSubstr(
            "Projected texture coordinates exceed integer range")));
  }
}

TEST(MeshTextureMapping, RejectsProjectionExtentOverflow) {
  PlyMesh mesh = MakeAtlasTestMesh(65537, 2);
  mesh.vertices[2].x = -0.25f;
  const std::vector<Image> images = {
      MakeTestImage(64, 64, BitmapColor<uint8_t>(128), 0.0f, float{1 << 27})};
  MeshTextureMappingOptions options;
  options.num_threads = 1;
  options.min_visible_vertices = 1;
  EXPECT_THAT([&] { MeshTextureMapping(mesh, images, options); },
              testing::ThrowsMessage<std::invalid_argument>(testing::HasSubstr(
                  "Projected texture dimensions exceed integer range")));
}

TEST(MeshTextureMapping, RejectsPaddingOverflow) {
  const PlyMesh mesh = MakeTriangleMesh();
  const std::vector<Image> images = {
      MakeTestImage(64, 64, BitmapColor<uint8_t>(128))};
  MeshTextureMappingOptions options;
  options.num_threads = 1;
  options.atlas_patch_padding = std::numeric_limits<int>::max();
  EXPECT_THAT([&] { MeshTextureMapping(mesh, images, options); },
              testing::ThrowsMessage<std::invalid_argument>(
                  testing::HasSubstr("Atlas patch dimensions exceed maximum")));
}

TEST(MeshTextureMapping, AtlasDimensionLimit) {
  const std::vector<Image> images = {
      MakeTestImage(64, 64, BitmapColor<uint8_t>(128), 0.0f, 8192)};
  MeshTextureMappingOptions options;
  options.num_threads = 1;
  options.min_visible_vertices = 1;
  options.atlas_patch_padding = 0;
  options.apply_color_correction = false;
  options.inpaint_radius = 0;
  const auto result =
      MeshTextureMapping(MakeAtlasTestMesh(65536, 2), images, options);
  EXPECT_EQ(result.face_view_ids, std::vector<int>{0});
  EXPECT_EQ(result.atlas_width, 65536);
  EXPECT_EQ(result.atlas_height, 2);
  const auto color = result.texture_atlas.GetPixel(0, 0);
  ASSERT_TRUE(color.has_value());
  EXPECT_EQ(color->r, 128);
  EXPECT_EQ(color->g, 128);
  EXPECT_EQ(color->b, 128);

  // This would otherwise pack into a 131072x2 atlas on the first attempt.
  EXPECT_THAT(
      [&] { MeshTextureMapping(MakeAtlasTestMesh(65537, 2), images, options); },
      testing::ThrowsMessage<std::invalid_argument>(
          testing::HasSubstr("Atlas patch dimensions exceed maximum")));
}

TEST(MeshTextureMapping, RejectsAtlasGrowthPastLimit) {
  PlyMesh mesh = MakeAtlasTestMesh(65536, 2);
  const PlyMesh tall_mesh = MakeAtlasTestMesh(2, 65536);
  mesh.vertices.insert(mesh.vertices.end(),
                       tall_mesh.vertices.begin(),
                       tall_mesh.vertices.end());
  mesh.faces.emplace_back(3, 4, 5);
  const std::vector<Image> images = {
      MakeTestImage(64, 64, BitmapColor<uint8_t>(128), 0.0f, 8192)};
  MeshTextureMappingOptions options;
  options.num_threads = 1;
  options.min_visible_vertices = 1;
  options.atlas_patch_padding = 0;
  // The two thin patches cannot share a 65536-wide shelf. Reject before
  // allocating the oversized atlas required by the next packing attempt.
  EXPECT_THAT([&] { MeshTextureMapping(mesh, images, options); },
              testing::ThrowsMessage<std::invalid_argument>(
                  testing::HasSubstr("Atlas dimensions exceeded maximum")));
}

TEST(MeshTextureMapping, RejectsAtlasStorageOverflow) {
  const PlyMesh mesh = MakeTriangleMesh();
  const std::vector<Image> images = {
      MakeTestImage(64, 64, BitmapColor<uint8_t>(128))};
  MeshTextureMappingOptions options;
  options.num_threads = 1;
  // Exercise both total patch area and the larger power-of-two atlas area.
  for (const int padding : {20000, 13000}) {
    SCOPED_TRACE(padding);
    options.atlas_patch_padding = padding;
    EXPECT_THAT(
        [&] { MeshTextureMapping(mesh, images, options); },
        testing::ThrowsMessage<std::invalid_argument>(
            testing::HasSubstr("Atlas area exceeds RGB bitmap storage limit")));
  }
}

#if defined(COLMAP_CGAL_ENABLED)

TEST(MeshTextureMapping, OcclusionSearchesPastRejectedHits) {
  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  for (const float translation : {0.0f, 32768.0f}) {
    std::vector<Image> images;
    images.push_back(MakeTestImage(64,
                                   64,
                                   BitmapColor<uint8_t>(96, 160, 224),
                                   translation,
                                   /*focal_scale=*/0.5f));
    // Zero denotes no blocker. Positive/negative Z puts it in front/behind.
    for (const float blocker_z : {0.0f, 2.5f, -2.5f}) {
      for (const bool duplicate : {false, true}) {
        for (const bool reversed : {false, true}) {
          if (blocker_z == 0 && reversed) continue;
          PlyMesh mesh;
          mesh.vertices = {PlyMeshVertex(0, 0, 0),
                           PlyMeshVertex(1, 0, 0),
                           PlyMeshVertex(0, 1, 0)};
          std::vector<PlyMeshFace> core = {PlyMeshFace(0, 1, 2)};
          std::vector<int> labels = {blocker_z > 0 ? -1 : 0};
          if (duplicate) {
            core.emplace_back(0, 1, 2);
            labels.push_back(labels.front());
          }
          if (blocker_z != 0) {
            mesh.vertices.insert(mesh.vertices.end(),
                                 {PlyMeshVertex(0, 0, blocker_z),
                                  PlyMeshVertex(1.25f, 0, blocker_z),
                                  PlyMeshVertex(0, 1.25f, blocker_z)});
            core.emplace_back(3, reversed ? 5 : 4, reversed ? 4 : 5);
            labels.push_back(blocker_z > 0 && !reversed ? 0 : -1);
          }
          const size_t control = mesh.vertices.size();
          mesh.vertices.insert(mesh.vertices.end(),
                               {PlyMeshVertex(2, 0, 0),
                                PlyMeshVertex(3, 0, 0),
                                PlyMeshVertex(2, 1, 0)});
          for (auto& vertex : mesh.vertices) {
            vertex.x += translation;
            vertex.y += translation;
            vertex.z += translation;
          }
          std::vector<size_t> order;
          for (size_t i = 0; i < core.size(); ++i) order.push_back(i);
          do {
            mesh.faces.clear();
            std::vector<int> expected;
            for (const size_t index : order) {
              mesh.faces.push_back(core[index]);
              expected.push_back(labels[index]);
            }
            mesh.faces.emplace_back(control, control + 1, control + 2);
            expected.push_back(0);
            for (const int workers : {1, 3}) {
              SCOPED_TRACE(::testing::Message()
                           << translation << ", " << blocker_z << ", "
                           << duplicate << ", " << reversed << ", " << workers);
              options.num_threads = workers;
              const auto result = MeshTextureMapping(mesh, images, options);
              EXPECT_EQ(result.face_view_ids, expected);
              EXPECT_EQ(result.face_uvs.size(), mesh.faces.size() * 6);
              EXPECT_GT(result.atlas_width, 0);
              EXPECT_GT(result.atlas_height, 0);
              const auto& pixels = result.texture_atlas.RowMajorData();
              EXPECT_TRUE(std::any_of(
                  pixels.begin(), pixels.end(), [](const uint8_t value) {
                    return value != 0;
                  }));
            }
          } while (std::next_permutation(order.begin(), order.end()));
        }
      }
    }
  }
}

TEST(MeshTextureMapping, ValidatesOcclusionIntersections) {
  struct Fixture {
    std::array<float, 3> center;
    std::array<PlyMeshVertex, 3> target;
    std::array<PlyMeshVertex, 3> blocker;
    int target_view = 0;
  };
  // The first two blockers are exactly clear of all three target rays before
  // their endpoints, despite a float intersection outside the triangle.
  const Fixture fixtures[] = {
      {{-7.28137064f, -7.57667017f, 0.204446003f},
       {PlyMeshVertex(-20.5156918f, -10.2649059f, 0.0925593078f),
        PlyMeshVertex(-20.5188713f, -10.2629061f, 0.085884355f),
        PlyMeshVertex(-20.5368862f, -10.256238f, 0.0856725201f)},
       {PlyMeshVertex(-20.5156918f, -10.2649059f, 0.0925593078f),
        PlyMeshVertex(-20.5152416f, -10.2641706f, 0.0859291628f),
        PlyMeshVertex(-20.5368862f, -10.256238f, 0.0856725201f)}},
      {{-15.8817997f, -3.15083027f, 0.0592618994f},
       {PlyMeshVertex(-19.1156158f, -9.98168564f, 0.602901161f),
        PlyMeshVertex(-19.2763901f, -10.1321306f, 0.624929011f),
        PlyMeshVertex(-19.1166496f, -9.98350906f, 0.607432544f)},
       {PlyMeshVertex(-19.2763901f, -10.1321306f, 0.624929011f),
        PlyMeshVertex(-19.2370853f, -10.0486422f, 0.598454177f),
        PlyMeshVertex(-19.2776642f, -10.1302834f, 0.595369875f)}},
      // The ray to (1/74, -5/82, 0) hits this blocker before the cutoff in
      // float and double working coordinates. Its float hit distance rounds
      // to the cutoff and must not reject the exact hit.
      {{0, 0, 0},
       {PlyMeshVertex(1.0f / 74, -5.0f / 82, 0),
        PlyMeshVertex(1.0f / 74, -5.0f / 82 - 0.02f, 0),
        PlyMeshVertex(1.0f / 74, -5.0f / 82, 0.02f)},
       {PlyMeshVertex(0x1.ba1a10p-7f, -0x1.f2b66p-5f + 0.001f, -0.001f),
        PlyMeshVertex(0x1.ba1a10p-7f, -0x1.f2b66p-5f - 0.001f, -0.001f),
        PlyMeshVertex(0x1.ba1a10p-7f, -0x1.f2b66p-5f, 0.001f)},
       -1},
  };
  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  for (const auto& fixture : fixtures) {
    const auto& center = fixture.center;
    const float K[9] = {32, 0, 192, 0, 32, 32, 0, 0, 1};
    const float R[9] = {1, 0, 0, 0, 0, 1, 0, -1, 0};
    const float T[3] = {-center[0], -center[2], center[1]};
    Image image("occlusion.png", 256, 64, K, R, T);
    Bitmap bitmap(256, 64, /*as_rgb=*/true);
    bitmap.Fill(BitmapColor<uint8_t>(96, 160, 224));
    image.SetBitmap(std::move(bitmap));
    const std::vector<Image> images = {image};
    PlyMesh mesh;
    mesh.vertices.assign(fixture.target.begin(), fixture.target.end());
    mesh.vertices.insert(
        mesh.vertices.end(), fixture.blocker.begin(), fixture.blocker.end());
    mesh.vertices.insert(
        mesh.vertices.end(),
        {PlyMeshVertex(center[0] + 2, center[1] - 5, center[2]),
         PlyMeshVertex(center[0] + 2, center[1] - 5, center[2] + 1),
         PlyMeshVertex(center[0] + 3, center[1] - 5, center[2])});
    for (size_t cycle = 0; cycle < 3; ++cycle) {
      for (const bool reversed : {false, true}) {
        const std::array<PlyMeshFace, 3> faces = {
            PlyMeshFace(cycle, (cycle + 1) % 3, (cycle + 2) % 3),
            PlyMeshFace(3, reversed ? 5 : 4, reversed ? 4 : 5),
            PlyMeshFace(6, 7, 8)};
        std::array<size_t, 3> order = {0, 1, 2};
        do {
          mesh.faces.clear();
          for (const size_t index : order) mesh.faces.push_back(faces[index]);
          for (const int workers : {1, 3}) {
            SCOPED_TRACE(::testing::Message()
                         << center[0] << ", " << cycle << ", " << reversed
                         << ", " << workers);
            options.num_threads = workers;
            const auto result = MeshTextureMapping(mesh, images, options);
            ASSERT_EQ(result.face_view_ids.size(), 3u);
            for (size_t i = 0; i < order.size(); ++i) {
              if (order[i] == 0 || order[i] == 2) {
                EXPECT_EQ(result.face_view_ids[i],
                          order[i] == 0 ? fixture.target_view : 0);
              }
            }
            EXPECT_EQ(result.face_uvs.size(), 18u);
            EXPECT_GT(result.atlas_width, 0);
            EXPECT_GT(result.atlas_height, 0);
            const auto& pixels = result.texture_atlas.RowMajorData();
            EXPECT_TRUE(std::any_of(
                pixels.begin(), pixels.end(), [](const uint8_t value) {
                  return value != 0;
                }));
          }
        } while (std::next_permutation(order.begin(), order.end()));
      }
    }
  }
}

#endif  // COLMAP_CGAL_ENABLED

TEST(MeshTextureMapping, PreservesDoublePositionGeometry) {
  for (const double offset : {0.0, 16777216.0}) {
    for (const bool blocked : {false, true}) {
      SCOPED_TRACE(::testing::Message() << offset << ", " << blocked);
      PlyMesh mesh;
      mesh.vertices = {{offset + 0.25, 0.25, 1.0},
                       {offset + 0.75, 0.25, 1.0},
                       {offset + 0.25, 0.75, 1.0}};
      mesh.faces = {{0, 2, 1}};
      if (blocked) {
        // All three target rays meet the strict interior of this nearer face.
        mesh.vertices.insert(mesh.vertices.end(),
                             {{offset + 0.0625, 0.0625, 0.5},
                              {offset + 0.5, 0.0625, 0.5},
                              {offset + 0.0625, 0.5, 0.5}});
        mesh.faces.emplace_back(3, 5, 4);
      }
      // The public float camera coefficients are exact, including at 2^24.
      // The target image corners remain (16,16), (16,48), and (48,16).
      const float K[9] = {64, 0, 0, 0, 64, 0, 0, 0, 1};
      const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      const float T[3] = {-static_cast<float>(offset), 0, 0};
      Image image("position-precision.png", 128, 128, K, R, T);
      const BitmapColor<uint8_t> color(61, 137, 223);
      Bitmap bitmap(128, 128, true);
      bitmap.Fill(color);
      image.SetBitmap(std::move(bitmap));

      MeshTextureMappingOptions options;
      options.num_threads = 1;
      options.view_selection_smoothing_iterations = 0;
      options.apply_color_correction = false;
      options.inpaint_radius = 0;
      const auto result = MeshTextureMapping(mesh, {image}, options);
      std::vector<int> expected(mesh.faces.size(), 0);
#if defined(COLMAP_CGAL_ENABLED)
      if (blocked) expected[0] = -1;
#endif
      ASSERT_EQ(result.face_view_ids, expected);
      ASSERT_EQ(result.face_uvs.size(), mesh.faces.size() * 6);
      for (size_t face = 0; face < mesh.faces.size(); ++face) {
        const float* uv = result.face_uvs.data() + face * 6;
        for (size_t i = 0; i < 6; ++i) {
          EXPECT_TRUE(std::isfinite(uv[i]));
          EXPECT_GE(uv[i], 0);
          EXPECT_LE(uv[i], 1);
        }
        const double area = (double(uv[2]) - uv[0]) * (double(uv[5]) - uv[1]) -
                            (double(uv[4]) - uv[0]) * (double(uv[3]) - uv[1]);
        if (expected[face] >= 0) EXPECT_NE(area, 0);
      }
      EXPECT_GT(result.atlas_width, 0);
      EXPECT_GT(result.atlas_height, 0);
      size_t colored_pixels = 0;
      for (int y = 0; y < result.atlas_height; ++y) {
        for (int x = 0; x < result.atlas_width; ++x) {
          const auto pixel = result.texture_atlas.GetPixel(x, y);
          ASSERT_TRUE(pixel.has_value());
          if (pixel->r == color.r && pixel->g == color.g &&
              pixel->b == color.b) {
            ++colored_pixels;
          }
        }
      }
      EXPECT_GT(colored_pixels, 0);
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
