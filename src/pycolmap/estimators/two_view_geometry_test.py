# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import pycolmap


def test_two_view_geometry_options_default_init() -> None:
    options = pycolmap.TwoViewGeometryOptions()
    assert options is not None
    assert options.use_optimistic_essential_scoring is False


def test_optimistic_essential_scoring_option_roundtrip():
    options = pycolmap.TwoViewGeometryOptions()
    options.use_optimistic_essential_scoring = True
    copied = pycolmap.TwoViewGeometryOptions(options.todict())
    assert copied.use_optimistic_essential_scoring is True
    copied.use_optimistic_essential_scoring = False
    assert copied.use_optimistic_essential_scoring is False
    assert (
        pycolmap.TwoViewGeometryOptions(
            copied.todict()
        ).use_optimistic_essential_scoring
        is False
    )


@pytest.mark.parametrize(
    "model,params",
    [
        (
            "SIMPLE_RADIAL",
            [2999.3562894415713, 2016, 1512, 0.02908154716756155],
        ),
        ("OPENCV", [2900, 2950, 2016, 1512, 0.04, -0.01, 0.001, -0.002]),
    ],
)
def test_optimistic_essential_scoring_serial_geometry(model, params):
    camera = pycolmap.Camera(
        model=model, width=4032, height=3024, params=params
    )
    rng = np.random.default_rng(3122)
    xyz = rng.uniform([-1.2, -0.8, 3.0], [1.2, 0.8, 7.0], size=(80, 3))
    points1 = camera.img_from_cam(xyz)
    points2 = camera.img_from_cam(xyz + [0.4, 0.02, 0.0])
    indices = np.arange(len(xyz), dtype=np.uint32)
    matches = np.column_stack((indices, indices))
    options = pycolmap.TwoViewGeometryOptions()
    options.ransac.num_threads = 1
    options.ransac.random_seed = 3122
    options.ransac.min_num_trials = 100
    options.ransac.max_num_trials = 1000
    baseline = pycolmap.estimate_calibrated_two_view_geometry(
        camera, points1, camera, points2, matches=matches, options=options
    )
    options.use_optimistic_essential_scoring = True
    candidate = pycolmap.estimate_calibrated_two_view_geometry(
        camera, points1, camera, points2, matches=matches, options=options
    )
    assert len(baseline.inlier_matches) == len(xyz)
    assert candidate.config == baseline.config
    for name in ("E", "F", "H", "inlier_matches"):
        expected = getattr(baseline, name)
        actual = getattr(candidate, name)
        if expected is None:
            assert actual is None
        else:
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()


# Separate synthetic full-state fixture from the retained precise-calibration
# E/F/H/inlier regression above; both historical roles remain explicit.
@pytest.mark.parametrize(
    "model,params",
    [
        ("SIMPLE_RADIAL", [1200, 800, 600, 0.03]),
        ("OPENCV", [1150, 1180, 800, 600, 0.04, -0.01, 0.001, -0.002]),
    ],
)
def test_optimistic_essential_scoring_synthetic_full_state(model, params):
    camera = pycolmap.Camera(
        model=model, width=1600, height=1200, params=params
    )
    rng = np.random.default_rng(3122)
    xyz = rng.uniform([-1.2, -0.8, 3.0], [1.2, 0.8, 7.0], size=(80, 3))
    points1 = camera.img_from_cam(xyz)
    points2 = camera.img_from_cam(xyz + [0.4, 0.02, 0.0])
    indices = np.arange(len(xyz), dtype=np.uint32)
    matches = np.column_stack((indices, indices))
    options = pycolmap.TwoViewGeometryOptions()
    options.ransac.num_threads = 1
    options.ransac.random_seed = 3122
    options.ransac.min_num_trials = 100
    options.ransac.max_num_trials = 1000
    options.compute_relative_pose = True
    options.use_optimistic_essential_scoring = False
    baseline = pycolmap.estimate_calibrated_two_view_geometry(
        camera, points1, camera, points2, matches=matches, options=options
    )
    options.use_optimistic_essential_scoring = True
    candidate = pycolmap.estimate_calibrated_two_view_geometry(
        camera, points1, camera, points2, matches=matches, options=options
    )
    assert len(baseline.inlier_matches) == len(xyz)
    assert candidate.config == baseline.config
    for name in ("E", "F", "H", "inlier_matches"):
        expected = getattr(baseline, name)
        actual = getattr(candidate, name)
        if expected is None:
            assert actual is None
        else:
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()
    assert (
        np.float64(candidate.tri_angle).tobytes()
        == np.float64(baseline.tri_angle).tobytes()
    )
    expected_pose = baseline.cam2_from_cam1
    actual_pose = candidate.cam2_from_cam1
    assert expected_pose is not None
    assert actual_pose is not None
    assert actual_pose.params.tobytes() == expected_pose.params.tobytes()
    for name in ("camera1", "camera2"):
        expected = getattr(baseline, name)
        actual = getattr(candidate, name)
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            for field in (
                "camera_id",
                "model",
                "width",
                "height",
                "source",
                "has_prior_focal_length",
            ):
                assert getattr(actual, field) == getattr(expected, field)
            assert actual.params.dtype == expected.params.dtype
            assert actual.params.shape == expected.params.shape
            assert actual.params.tobytes() == expected.params.tobytes()


def test_two_view_geometry_options_min_num_inliers_readwrite() -> None:
    options = pycolmap.TwoViewGeometryOptions()
    assert isinstance(options.min_num_inliers, int)
    options.min_num_inliers = 30
    assert options.min_num_inliers == 30


@pytest.mark.parametrize(
    "name",
    [
        "estimate_calibrated_two_view_geometry",
        "estimate_two_view_geometry",
        "estimate_two_view_geometry_pose",
        "compute_squared_sampson_error",
    ],
)
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))


def test_compute_squared_sampson_error_with_identity() -> None:
    points1 = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    points2 = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    matrix = np.eye(3)
    residuals = pycolmap.compute_squared_sampson_error(points1, points2, matrix)
    assert isinstance(residuals, list)
    assert len(residuals) == 2
