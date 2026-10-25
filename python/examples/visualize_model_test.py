# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("open3d") is None:
    raise unittest.SkipTest("Open3D is required for camera visualization")
if importlib.util.find_spec("pycolmap") is None:
    raise unittest.SkipTest(
        "pycolmap is required for the visualization example"
    )

from visualize_model import draw_camera


def check_camera_plane_bounds(width, height, cx, cy, scale, transformed):
    fx, fy = 500 * width / 640, 630 * height / 480
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    angle = 0.43 if transformed else 0.0
    R = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    t = np.array([1.0, -2.0, 0.5]) if transformed else np.zeros(3)

    _, plane, frustum = draw_camera(K, R, t, width, height, scale)
    vertices = (np.asarray(plane.vertices) - t) @ R
    corners = (np.asarray(frustum.points)[1:] - t) @ R
    expected_min = scale * np.array([-cx / fx, -cy / fy])
    expected_max = scale * np.array([(width - cx) / fx, (height - cy) / fy])

    np.testing.assert_allclose(
        vertices[:, :2].min(axis=0), expected_min, atol=1e-12
    )
    np.testing.assert_allclose(
        vertices[:, :2].max(axis=0), expected_max, atol=1e-12
    )
    np.testing.assert_allclose(
        corners[:, :2].min(axis=0), expected_min, atol=1e-12
    )
    np.testing.assert_allclose(
        corners[:, :2].max(axis=0), expected_max, atol=1e-12
    )


class CameraPlaneTest(unittest.TestCase):
    def test_bounds(self):
        for width, height in [
            (640, 480),
            (1536, 1024),
            (3072, 2048),
            (6144, 4096),
        ]:
            for base_cx, base_cy in [
                (320, 240),
                (100, 240),
                (320, 90),
                (-100, 240),
                (800, 240),
                (320, 800),
            ]:
                cx, cy = base_cx * width / 640, base_cy * height / 480
                for scale in [0.25, 1.0, 3.0]:
                    for transformed in [False, True]:
                        with self.subTest(
                            width=width,
                            height=height,
                            cx=cx,
                            cy=cy,
                            scale=scale,
                            transformed=transformed,
                        ):
                            check_camera_plane_bounds(
                                width, height, cx, cy, scale, transformed
                            )


if __name__ == "__main__":
    unittest.main()
