# SPDX-License-Identifier: BSD-3-Clause

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pycolmap


def test_reconstruction_default_init() -> None:
    reconstruction = pycolmap.Reconstruction()
    assert reconstruction is not None


def test_reconstruction_copy_init(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    assert (
        reconstruction_copy.num_cameras()
        == synthetic_reconstruction.num_cameras()
    )


def test_reconstruction_num_rigs(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    assert isinstance(synthetic_reconstruction.num_rigs(), int)
    assert synthetic_reconstruction.num_rigs() > 0


def test_reconstruction_num_cameras(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    assert isinstance(synthetic_reconstruction.num_cameras(), int)
    assert synthetic_reconstruction.num_cameras() > 0


def test_reconstruction_num_frames(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    assert isinstance(synthetic_reconstruction.num_frames(), int)
    assert synthetic_reconstruction.num_frames() > 0


def test_reconstruction_num_images(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    assert isinstance(synthetic_reconstruction.num_images(), int)
    assert synthetic_reconstruction.num_images() > 0


def test_reconstruction_num_points3d(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    assert isinstance(synthetic_reconstruction.num_points3D(), int)
    assert synthetic_reconstruction.num_points3D() > 0


def test_reconstruction_num_reg_frames(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    assert isinstance(synthetic_reconstruction.num_reg_frames(), int)


def test_reconstruction_num_reg_images(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    assert isinstance(synthetic_reconstruction.num_reg_images(), int)


def test_reconstruction_cameras_property(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    cameras = synthetic_reconstruction.cameras
    assert len(cameras) > 0


def test_reconstruction_frames_property(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    frames = synthetic_reconstruction.frames
    assert frames is not None


def test_reconstruction_images_property(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    images = synthetic_reconstruction.images
    assert len(images) > 0


def test_reconstruction_points3d_property(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    points = synthetic_reconstruction.points3D
    assert len(points) > 0


def test_reconstruction_camera_access(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    camera_ids = list(synthetic_reconstruction.cameras.keys())
    camera = synthetic_reconstruction.camera(camera_ids[0])
    assert camera is not None


def test_reconstruction_frame_access(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    frame_ids = list(synthetic_reconstruction.frames.keys())
    frame = synthetic_reconstruction.frame(frame_ids[0])
    assert frame is not None


def test_reconstruction_image_access(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    image_ids = list(synthetic_reconstruction.images.keys())
    image = synthetic_reconstruction.image(image_ids[0])
    assert image is not None


def test_reconstruction_point3d_access(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    point_ids = list(synthetic_reconstruction.points3D.keys())
    point = synthetic_reconstruction.point3D(point_ids[0])
    assert point is not None


def test_reconstruction_exists_camera(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    camera_ids = list(synthetic_reconstruction.cameras.keys())
    assert synthetic_reconstruction.exists_camera(camera_ids[0])
    assert not synthetic_reconstruction.exists_camera(99999)


def test_reconstruction_exists_frame(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    frame_ids = list(synthetic_reconstruction.frames.keys())
    assert synthetic_reconstruction.exists_frame(frame_ids[0])


def test_reconstruction_exists_image(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    image_ids = list(synthetic_reconstruction.images.keys())
    assert synthetic_reconstruction.exists_image(image_ids[0])
    assert not synthetic_reconstruction.exists_image(99999)


def test_reconstruction_exists_point3d(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    point_ids = list(synthetic_reconstruction.points3D.keys())
    assert synthetic_reconstruction.exists_point3D(point_ids[0])
    assert not synthetic_reconstruction.exists_point3D(99999)


def test_reconstruction_reg_image_ids(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reg_ids = synthetic_reconstruction.reg_image_ids()
    assert isinstance(reg_ids, list)


def test_reconstruction_reg_frame_ids(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reg_ids = synthetic_reconstruction.reg_frame_ids()
    assert isinstance(reg_ids, list)


def test_reconstruction_point3d_ids(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    point_ids = synthetic_reconstruction.point3D_ids()
    assert len(point_ids) > 0


def test_reconstruction_is_valid(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    result = synthetic_reconstruction.is_valid()
    assert isinstance(result, bool)


def test_reconstruction_compute_num_observations(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    count = synthetic_reconstruction.compute_num_observations()
    assert isinstance(count, int)


def test_reconstruction_compute_mean_track_length(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    length = synthetic_reconstruction.compute_mean_track_length()
    assert isinstance(length, float)


def test_reconstruction_compute_mean_reprojection_error(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    error = reconstruction_copy.compute_mean_reprojection_error()
    assert isinstance(error, float)


def test_reconstruction_compute_mean_observations_per_reg_image(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    mean_obs = (
        synthetic_reconstruction.compute_mean_observations_per_reg_image()
    )
    assert isinstance(mean_obs, float)


def test_reconstruction_summary(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    summary = synthetic_reconstruction.summary()
    assert isinstance(summary, str)
    assert "Reconstruction" in summary


def test_reconstruction_compute_bounding_box(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    bbox = synthetic_reconstruction.compute_bounding_box()
    assert bbox is not None


def test_reconstruction_compute_centroid(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    centroid = synthetic_reconstruction.compute_centroid()
    assert centroid.shape == (3,)


def test_reconstruction_normalize(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    reconstruction_copy.normalize()
    assert reconstruction_copy.num_points3D() > 0


def test_reconstruction_transform(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    transform = pycolmap.Sim3d()
    reconstruction_copy.transform(transform)
    assert reconstruction_copy.num_points3D() > 0


def test_reconstruction_write_read_binary_roundtrip(
    synthetic_reconstruction: pycolmap.Reconstruction, tmp_path: Path
) -> None:
    output_dir = str(tmp_path / "binary")

    os.makedirs(output_dir)
    synthetic_reconstruction.write_binary(output_dir)
    loaded = pycolmap.Reconstruction()
    loaded.read_binary(output_dir)
    assert loaded.num_cameras() == synthetic_reconstruction.num_cameras()
    assert loaded.num_images() == synthetic_reconstruction.num_images()


def test_reconstruction_write_read_text_roundtrip(
    synthetic_reconstruction: pycolmap.Reconstruction, tmp_path: Path
) -> None:
    output_dir = str(tmp_path / "text")

    os.makedirs(output_dir)
    synthetic_reconstruction.write_text(output_dir)
    loaded = pycolmap.Reconstruction()
    loaded.read_text(output_dir)
    assert loaded.num_cameras() == synthetic_reconstruction.num_cameras()
    assert loaded.num_images() == synthetic_reconstruction.num_images()


def test_reconstruction_add_camera() -> None:
    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, 500.0, 1024, 768
    )
    reconstruction.add_camera(camera)
    assert reconstruction.exists_camera(1)


def test_reconstruction_find_image_with_name(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    image_ids = list(synthetic_reconstruction.images.keys())
    first_image = synthetic_reconstruction.image(image_ids[0])
    found = synthetic_reconstruction.find_image_with_name(first_image.name)
    assert found is not None


def test_reconstruction_find_common_reg_image_ids(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    common_ids = synthetic_reconstruction.find_common_reg_image_ids(
        reconstruction_copy
    )
    assert isinstance(common_ids, list)


def test_reconstruction_export_import_ply_roundtrip(
    synthetic_reconstruction: pycolmap.Reconstruction, tmp_path: Path
) -> None:
    ply_path = str(tmp_path / "points.ply")
    synthetic_reconstruction.export_PLY(ply_path)
    new_reconstruction = pycolmap.Reconstruction()
    new_reconstruction.import_PLY(ply_path)
    assert new_reconstruction.num_points3D() > 0


def test_reconstruction_crop(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    bbox = reconstruction_copy.compute_bounding_box()
    reconstruction_copy.crop(bbox)
    assert reconstruction_copy is not None


def test_reconstruction_delete_all_points2d_and_points3d(
    synthetic_reconstruction: pycolmap.Reconstruction,
) -> None:
    reconstruction_copy = pycolmap.Reconstruction(synthetic_reconstruction)
    reconstruction_copy.delete_all_points2D_and_points3D()
    assert reconstruction_copy.num_points3D() == 0


def test_extract_colors_with_python_bitmap_loader_completes(tmp_path: Path) -> None:
    # Run in a child so a worker waiting for the GIL cannot hang pytest.
    script = textwrap.dedent(
        """\
        import importlib
        import json
        import sys
        from pathlib import Path

        sys.path[:] = json.loads(sys.argv[2])
        import numpy as np
        import pycolmap

        core = importlib.import_module("pycolmap._core")
        assert Path(core.__file__).resolve() == Path(sys.argv[3]).resolve()
        pycolmap.set_random_seed(0)
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs = 1
        options.num_cameras_per_rig = 1
        options.num_frames_per_rig = 3
        options.num_points3D = 50
        reconstruction = pycolmap.synthesize_dataset(options)
        image_ids = reconstruction.reg_image_ids()
        assert image_ids and reconstruction.num_points3D() > 0

        root = Path(sys.argv[1])
        paths = [root / reconstruction.image(i).name for i in image_ids]
        width, height = options.camera_width, options.camera_height
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[:] = (23, 71, 157)
        expected_ids = set()
        for image_id in image_ids:
            for point in reconstruction.image(image_id).points2D:
                x, y = point.xy
                if (
                    point.has_point3D()
                    and 1 <= x < width - 1
                    and 1 <= y < height - 1
                ):
                    expected_ids.add(point.point3D_id)
        assert expected_ids
        calls = []

        def loader(path):
            calls.append(Path(path))
            return pixels

        with pycolmap.BitmapReadScope(
            paths, loader, True, True, [(width, height)] * len(paths)
        ) as scope:
            reconstruction.extract_colors_for_all_images(root, num_threads=1)
            assert len(calls) == len(paths) and set(calls) == set(paths)
            assert scope.stats()["bitmapReads"] == len(paths)
            for point_id in expected_ids:
                np.testing.assert_array_equal(
                    reconstruction.point3D(point_id).color, (23, 71, 157)
                )
        assert scope.closed

        def failing_loader(path):
            raise RuntimeError("expected bitmap loader failure")

        with pycolmap.BitmapReadScope(
            paths, failing_loader, True, True, [(width, height)] * len(paths)
        ) as failed_scope:
            try:
                reconstruction.extract_colors_for_all_images(root, num_threads=1)
            except RuntimeError as error:
                assert "expected bitmap loader failure" in str(error)
            else:
                raise AssertionError("loader failure did not propagate")
        assert failed_scope.closed

        calls.clear()
        with pycolmap.BitmapReadScope(
            paths, loader, True, True, [(width, height)] * len(paths)
        ) as recovery_scope:
            reconstruction.extract_colors_for_all_images(root, num_threads=1)
            assert len(calls) == len(paths) and set(calls) == set(paths)
            assert recovery_scope.stats()["bitmapReads"] == len(paths)
            for point_id in expected_ids:
                np.testing.assert_array_equal(
                    reconstruction.point3D(point_id).color, (23, 71, 157)
                )
        assert recovery_scope.closed
        """
    )
    core_path = importlib.import_module("pycolmap._core").__file__
    assert core_path is not None
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path.resolve()),
            json.dumps(sys.path),
            str(Path(core_path).resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
