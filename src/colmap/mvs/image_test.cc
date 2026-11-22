// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/image.h"

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image_size.h"
#include "colmap/mvs/normal_map.h"

#include <cmath>
#include <limits>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(Image, DefaultConstructor) {
  Image image;
  EXPECT_EQ(image.GetWidth(), 0);
  EXPECT_EQ(image.GetHeight(), 0);
  EXPECT_TRUE(image.GetPath().empty());
}

TEST(Image, ParameterizedConstructor) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);

  EXPECT_EQ(image.GetWidth(), 100);
  EXPECT_EQ(image.GetHeight(), 100);
  EXPECT_EQ(image.GetPath(), "test.jpg");

  const float* image_K = image.GetK();
  const float* image_R = image.GetR();
  const float* image_T = image.GetT();

  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(image_K[i], K[i]);
    EXPECT_EQ(image_R[i], R[i]);
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(image_T[i], T[i]);
  }
}

TEST(Image, Rescale) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  image.Rescale(0.5);

  EXPECT_EQ(image.GetWidth(), 50);
  EXPECT_EQ(image.GetHeight(), 50);

  const float* image_K = image.GetK();
  EXPECT_FLOAT_EQ(image_K[0], 50.0f);  // fx scaled
  EXPECT_FLOAT_EQ(image_K[2], 25.0f);  // cx scaled
  EXPECT_FLOAT_EQ(image_K[4], 50.0f);  // fy scaled
  EXPECT_FLOAT_EQ(image_K[5], 25.0f);  // cy scaled
}

TEST(Image, RescaleNonUniform) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  image.Rescale(0.5, 0.25);

  EXPECT_EQ(image.GetWidth(), 50);
  EXPECT_EQ(image.GetHeight(), 25);

  const float* image_K = image.GetK();
  EXPECT_FLOAT_EQ(image_K[0], 50.0f);  // fx scaled by factor_x
  EXPECT_FLOAT_EQ(image_K[2], 25.0f);  // cx scaled by factor_x
  EXPECT_FLOAT_EQ(image_K[4], 25.0f);  // fy scaled by factor_y
  EXPECT_FLOAT_EQ(image_K[5], 12.5f);  // cy scaled by factor_y
}

TEST(Image, Downsize) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  image.Downsize(50, 50);

  EXPECT_EQ(image.GetWidth(), 50);
  EXPECT_EQ(image.GetHeight(), 50);

  const float* image_K = image.GetK();
  EXPECT_FLOAT_EQ(image_K[0], 50.0f);  // fx scaled
  EXPECT_FLOAT_EQ(image_K[2], 25.0f);  // cx scaled
  EXPECT_FLOAT_EQ(image_K[4], 50.0f);  // fy scaled
  EXPECT_FLOAT_EQ(image_K[5], 25.0f);  // cy scaled
}

TEST(Image, DownsizeThinWorkspaceData) {
  const struct {
    size_t width, height, max_size;
    size_t image_width, image_height, map_width, map_height;
  } cases[] = {{1, 40, 16, 1, 16, 1, 16},
               {40, 1, 16, 16, 1, 16, 1},
               {2, 7, 1, 1, 1, 1, 1},
               {7, 2, 1, 1, 1, 1, 1},
               {37, 21, 16, 16, 9, 16, 9},
               {3, 40, 16, 1, 16, 1, 13},
               {40, 3, 16, 16, 1, 13, 1},
               {40, 40, 16, 16, 16, 16, 16},
               {1, 1, 1, 1, 1, 1, 1}};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message()
                 << c.width << "x" << c.height << " -> " << c.max_size);
    const float K[9] = {80, 0, c.width * 0.5f, 0, 60, c.height * 0.5f, 0, 0, 1};
    const float scale_x = c.image_width / static_cast<float>(c.width);
    const float scale_y = c.image_height / static_cast<float>(c.height);
    const float fx = K[0] * scale_x, cx = K[2] * scale_x;
    const float fy = K[4] * scale_y, cy = K[5] * scale_y;
    const float P[12] = {fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0};
    const float inv_P[12] = {
        1 / fx, 0, -cx / fx, 0, 0, 1 / fy, -cy / fy, 0, 0, 0, 1, 0};
    for (const bool with_bitmap : {false, true}) {
      Image image("test.jpg", c.width, c.height, K, R, T);
      if (with_bitmap) {
        image.SetBitmap(Bitmap(c.width, c.height, false));
      }
      ASSERT_NO_THROW(image.Downsize(c.max_size, c.max_size));
      ASSERT_EQ(image.GetWidth(), c.image_width);
      ASSERT_EQ(image.GetHeight(), c.image_height);
      if (with_bitmap) {
        EXPECT_EQ(image.GetBitmap().Width(), c.image_width);
        EXPECT_EQ(image.GetBitmap().Height(), c.image_height);
      }
      EXPECT_FLOAT_EQ(image.GetK()[0], fx);
      EXPECT_FLOAT_EQ(image.GetK()[2], cx);
      EXPECT_FLOAT_EQ(image.GetK()[4], fy);
      EXPECT_FLOAT_EQ(image.GetK()[5], cy);
      for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(image.GetP()[i], P[i]);
        EXPECT_FLOAT_EQ(image.GetInvP()[i], inv_P[i]);
      }
    }

    // Workspace downsizes each map to the image bounds, preserving its aspect
    // ratio. Rounding can therefore leave a map smaller than the image.
    DepthMap depth(c.width, c.height, 1, 3);
    depth.Fill(2);
    NormalMap normal(c.width, c.height);
    normal.Fill(0);
    for (size_t y = 0; y < c.height; ++y) {
      for (size_t x = 0; x < c.width; ++x) {
        normal.Set(y, x, 2, -1);
      }
    }
    ASSERT_NO_THROW(depth.Downsize(c.image_width, c.image_height));
    ASSERT_NO_THROW(normal.Downsize(c.image_width, c.image_height));
    ASSERT_EQ(depth.GetWidth(), c.map_width);
    ASSERT_EQ(depth.GetHeight(), c.map_height);
    ASSERT_EQ(normal.GetWidth(), c.map_width);
    ASSERT_EQ(normal.GetHeight(), c.map_height);
    for (size_t y = 0; y < c.map_height; ++y) {
      for (size_t x = 0; x < c.map_width; ++x) {
        EXPECT_NEAR(depth.Get(y, x), 2, 1e-5);
        EXPECT_FLOAT_EQ(normal.Get(y, x, 0), 0);
        EXPECT_FLOAT_EQ(normal.Get(y, x, 1), 0);
        EXPECT_FLOAT_EQ(normal.Get(y, x, 2), -1);
      }
    }
  }
}

TEST(Image, DownsizeRespectsLargeIntegerBounds) {
  const struct {
    size_t width, height, max_width, max_height, expected_width,
        expected_height;
  } cases[] = {
      {16777220, 1, 16777219, 1, 16777219, 1},
      {1, 16777220, 1, 16777219, 1, 16777219},
      {16777219, 16777220, 16777221, 16777219, 16777219, 16777219},
      {16777220, 16777219, 16777219, 16777221, 16777219, 16777219},
  };
  const float K[9] = {1, 0, 1, 0, 1, 1, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << c.width << "x" << c.height << " -> "
                                      << c.max_width << "x" << c.max_height);
    Image image("dimensions-only", c.width, c.height, K, R, T);
    ASSERT_TRUE(image.GetBitmap().IsEmpty());
    image.Downsize(c.max_width, c.max_height);
    EXPECT_EQ(image.GetWidth(), c.expected_width);
    EXPECT_EQ(image.GetHeight(), c.expected_height);
    EXPECT_LE(image.GetWidth(), c.width);
    EXPECT_LE(image.GetHeight(), c.height);
    EXPECT_LE(image.GetWidth(), c.max_width);
    EXPECT_LE(image.GetHeight(), c.max_height);
    EXPECT_TRUE(image.GetBitmap().IsEmpty());

    // Each changed axis loses one pixel out of 16777220. Its ratio rounds
    // to the float immediately below one; the unchanged axis stays at one.
    const float sx =
        c.expected_width == c.width ? 1.0f : std::nextafter(1.0f, 0.0f);
    const float sy =
        c.expected_height == c.height ? 1.0f : std::nextafter(1.0f, 0.0f);
    EXPECT_EQ(image.GetK()[0], sx);
    EXPECT_EQ(image.GetK()[2], sx);
    EXPECT_EQ(image.GetK()[4], sy);
    EXPECT_EQ(image.GetK()[5], sy);
    const float expected_P[12] = {sx, 0, sx, 0, 0, sy, sy, 0, 0, 0, 1, 0};
    const float expected_inv_P[12] = {
        1 / sx, 0, -1, 0, 0, 1 / sy, -1, 0, 0, 0, 1, 0};
    for (int i = 0; i < 12; ++i) {
      EXPECT_EQ(image.GetP()[i], expected_P[i]);
      EXPECT_FLOAT_EQ(image.GetInvP()[i], expected_inv_P[i]);
    }
  }
}

TEST(Image, DownsizeDimensionsBoundBeforeIntegerConversion) {
  // Keep the actual helper's conversion live in the instrumented test.
  volatile size_t largest_dimension = std::numeric_limits<size_t>::max();
  const size_t kMax = largest_dimension;
  EXPECT_EQ(internal::ComputeDownsizeDimensions(kMax, 1, kMax - 1, 1),
            std::make_pair(kMax - 1, size_t{1}));
  EXPECT_EQ(internal::ComputeDownsizeDimensions(1, kMax, 1, kMax - 1),
            std::make_pair(size_t{1}, kMax - 1));
  // A zero bound keeps the legacy one-pixel floor.
  EXPECT_EQ(internal::ComputeDownsizeDimensions(7, 5, 0, 0),
            std::make_pair(size_t{1}, size_t{1}));
}

TEST(Image, DownsizePreservesEmptyWorkspaceMaps) {
  for (const auto [width, height] :
       {std::pair<size_t, size_t>{0, 0}, {0, 7}, {7, 0}}) {
    DepthMap depth(width, height, 1, 3);
    NormalMap normal(width, height);
    depth.Downsize(1, 1);
    normal.Downsize(1, 1);
    depth.Rescale(0.5f);
    normal.Rescale(0.5f);
    EXPECT_EQ(depth.GetWidth(), width);
    EXPECT_EQ(depth.GetHeight(), height);
    EXPECT_EQ(normal.GetWidth(), width);
    EXPECT_EQ(normal.GetHeight(), height);
    EXPECT_TRUE(depth.GetData().empty());
    EXPECT_TRUE(normal.GetData().empty());
  }
}

TEST(Image, DownsizeNoChange) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  image.Downsize(200, 200);

  EXPECT_EQ(image.GetWidth(), 100);
  EXPECT_EQ(image.GetHeight(), 100);
}

TEST(Image, GetViewingDirection) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  const float* viewing_dir = image.GetViewingDirection();

  EXPECT_EQ(viewing_dir[0], R[6]);
  EXPECT_EQ(viewing_dir[1], R[7]);
  EXPECT_EQ(viewing_dir[2], R[8]);
}

TEST(ComputeProjectionCenter, Identity) {
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {1, 2, 3};
  float C[3];
  ComputeProjectionCenter(R, T, C);

  EXPECT_FLOAT_EQ(C[0], -1.0f);
  EXPECT_FLOAT_EQ(C[1], -2.0f);
  EXPECT_FLOAT_EQ(C[2], -3.0f);
}

TEST(ComposeProjectionMatrix, Identity) {
  const float K[9] = {2, 0, 0, 0, 2, 0, 0, 0, 1};
  const float R[9] = {0, 1, 0, 1, 0, 0, 0, 0, 1};
  const float T[3] = {1, 2, 3};
  float P[12];
  ComposeProjectionMatrix(K, R, T, P);

  const float expected[12] = {0, 2, 0, 2, 2, 0, 0, 4, 0, 0, 1, 3};
  for (int i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ(P[i], expected[i]) << i;
  }
}

TEST(RotatePose, Identity) {
  const float RR[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float T[3] = {1, 2, 3};

  RotatePose(RR, R, T);

  const float expected_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float expected_T[3] = {1, 2, 3};

  for (int i = 0; i < 9; ++i) {
    EXPECT_FLOAT_EQ(R[i], expected_R[i]);
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(T[i], expected_T[i]);
  }
}

TEST(RotatePose, Rotation90DegreesZ) {
  // 90 degree rotation around Z axis
  const float RR[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float T[3] = {1, 0, 0};

  RotatePose(RR, R, T);

  const float expected_R[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  const float expected_T[3] = {0, 1, 0};

  for (int i = 0; i < 9; ++i) {
    EXPECT_FLOAT_EQ(R[i], expected_R[i]) << i;
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(T[i], expected_T[i]) << i;
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
