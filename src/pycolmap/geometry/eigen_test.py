# SPDX-License-Identifier: BSD-3-Clause

import gc
import json
import subprocess
import sys
import tracemalloc
import weakref
from pathlib import Path

import numpy as np
import pytest

import pycolmap


def _wrong_size_allocation_peaks() -> dict[str, int]:
    module_name = pycolmap.Rotation3d.__module__
    assert pycolmap.Rigid3d.__module__ == module_name
    module = sys.modules[module_name]
    module_path = Path(module.__file__).resolve(strict=True)
    search_paths = [str(module_path.parent)]
    search_paths.extend(
        str(Path(entry or ".").resolve(strict=True))
        for entry in sys.path
        if Path(entry or ".").is_dir()
    )
    search_paths = list(dict.fromkeys(search_paths))
    script = r"""
import importlib
import json
from pathlib import Path
import sys
import tracemalloc

sys.path[:0] = json.loads(sys.argv[1])
module_name = sys.argv[2]
expected_module_path = Path(sys.argv[3])
module = importlib.import_module(module_name)
assert Path(module.__file__).resolve(strict=True) == expected_module_path

import numpy as np

array = np.broadcast_to(np.array(1.0), (2_000_000,))
rigid = module.Rigid3d()

tracemalloc.start()
try:
    module.Rotation3d.from_buffer(array)
except ValueError:
    pass
else:
    raise AssertionError("wrong-sized from_buffer input was accepted")
from_buffer_peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()

tracemalloc.start()
try:
    rigid.rotation = array
except TypeError:
    pass
else:
    raise AssertionError("wrong-sized caster input was accepted")
caster_peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
print(json.dumps({"from_buffer": from_buffer_peak, "caster": caster_peak}))
"""
    parent_tracing = tracemalloc.is_tracing()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            json.dumps(search_paths),
            module_name,
            str(module_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert tracemalloc.is_tracing() is parent_tracing
    return json.loads(result.stdout.splitlines()[-1])


def test_rotation3d_default_init() -> None:
    rotation = pycolmap.Rotation3d()
    assert rotation is not None


def test_rotation3d_init_from_xyzw() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    assert rotation is not None


def test_rotation3d_init_from_matrix() -> None:
    rotation = pycolmap.Rotation3d(matrix=np.eye(3))
    assert rotation is not None


def test_rotation3d_init_from_axis_angle() -> None:
    rotation = pycolmap.Rotation3d(axis_angle=np.zeros(3))
    assert rotation is not None


def test_rotation3d_from_buffer() -> None:
    array = np.array([0.0, 0.0, 0.0, 1.0])
    rotation = pycolmap.Rotation3d.from_buffer(array)
    assert rotation is not None


@pytest.mark.parametrize(
    "array",
    [
        np.array([0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 1.0])[::2],
        np.array([1.0, 0.0, 0.0, 0.0])[::-1],
    ],
)
def test_rotation3d_from_strided_buffer(array: np.ndarray) -> None:
    rotation = pycolmap.Rotation3d.from_buffer(array)
    np.testing.assert_array_equal(rotation.quat, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(rotation.matrix(), np.eye(3))


def test_rotation3d_from_strided_buffer_owns_logical_copy() -> None:
    array = np.array([0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 1.0])[::2]
    array_ref = weakref.ref(array)
    rotation = pycolmap.Rotation3d.from_buffer(array)

    array[:] = [1.0, 0.0, 0.0, 0.0]
    np.testing.assert_array_equal(rotation.quat, [0.0, 0.0, 0.0, 1.0])

    del array
    gc.collect()
    assert array_ref() is None
    np.testing.assert_array_equal(rotation.matrix(), np.eye(3))


def test_rotation3d_from_buffer_preserves_contiguous_alias() -> None:
    array = np.array([0.0, 0.0, 0.0, 1.0])
    rotation = pycolmap.Rotation3d.from_buffer(array)
    array[2] = 1.0
    assert rotation.quat[2] == 1.0
    rotation.quat[2] = 0.0
    assert array[2] == 0.0


def test_rotation3d_from_buffer_forcecasts_float32() -> None:
    array = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    rotation = pycolmap.Rotation3d.from_buffer(array)
    assert rotation.quat.dtype == np.float64
    np.testing.assert_array_equal(rotation.quat, array)


@pytest.mark.parametrize(
    "array",
    [
        np.arange(6.0)[::2],
        np.broadcast_to(np.array(1.0), (5,)),
    ],
)
def test_rotation3d_from_buffer_rejects_wrong_size_view(
    array: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        pycolmap.Rotation3d.from_buffer(array)


def test_rotation3d_from_buffer_checks_size_before_contiguous_copy() -> None:
    assert _wrong_size_allocation_peaks()["from_buffer"] < 2 * 1024 * 1024


def test_rotation3d_caster_checks_size_before_contiguous_copy() -> None:
    assert _wrong_size_allocation_peaks()["caster"] < 2 * 1024 * 1024


def test_rotation3d_readonly_buffer_can_be_consumed() -> None:
    array = np.array([0.0, 0.0, 0.0, 1.0])
    array.flags.writeable = False
    rotation = pycolmap.Rotation3d.from_buffer(array)
    np.testing.assert_array_equal(rotation.matrix(), np.eye(3))
    with pytest.raises(ValueError, match="writeable"):
        rotation.normalize()


def test_rotation3d_readonly_buffer_passes_through_transform_casters() -> None:
    array = np.array([0.0, 0.0, 0.0, 1.0])
    array.flags.writeable = False
    rotation = pycolmap.Rotation3d.from_buffer(array)

    rigid = pycolmap.Rigid3d(rotation=rotation, translation=np.zeros(3))
    rigid.rotation = rotation
    np.testing.assert_array_equal(rigid.rotation.quat, array)

    similarity = pycolmap.Sim3d(
        scale=1.0, rotation=rotation, translation=np.zeros(3)
    )
    similarity.rotation = rotation
    np.testing.assert_array_equal(similarity.rotation.quat, array)


def test_rotation3d_quat_readwrite() -> None:
    rotation = pycolmap.Rotation3d()
    quat = rotation.quat
    assert quat.shape == (4,)
    rotation.quat = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(rotation.quat, [0.0, 0.0, 0.0, 1.0])


def test_rotation3d_matrix() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    matrix = rotation.matrix()
    assert matrix.shape == (3, 3)
    np.testing.assert_array_almost_equal(matrix, np.eye(3))


def test_rotation3d_norm() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    assert isinstance(rotation.norm(), float)


def test_rotation3d_angle() -> None:
    rotation = pycolmap.Rotation3d(axis_angle=np.zeros(3))
    angle = rotation.angle()
    assert isinstance(angle, float)
    assert angle >= 0.0


def test_rotation3d_angle_to() -> None:
    rotation1 = pycolmap.Rotation3d()
    rotation2 = pycolmap.Rotation3d()
    angle = rotation1.angle_to(rotation2)
    assert isinstance(angle, float)


def test_rotation3d_inverse() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    inverse = rotation.inverse()
    assert inverse is not None


def test_rotation3d_normalize() -> None:
    rotation = pycolmap.Rotation3d(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    rotation.normalize()
    assert abs(rotation.norm() - 1.0) < 1e-10


def test_rotation3d_multiply_rotation3d() -> None:
    rotation1 = pycolmap.Rotation3d()
    rotation2 = pycolmap.Rotation3d()
    result = rotation1 * rotation2
    assert result is not None


def test_rotation3d_multiply_vector3d() -> None:
    rotation = pycolmap.Rotation3d()
    vector = np.array([1.0, 2.0, 3.0])
    result = rotation * vector
    assert result.shape == (3,)


def test_rotation3d_multiply_nx3_matrix() -> None:
    rotation = pycolmap.Rotation3d()
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = rotation * points
    assert result.shape == (2, 3)


def test_alignedbox3d_default_init() -> None:
    bbox = pycolmap.AlignedBox3d()
    assert bbox is not None


def test_alignedbox3d_init_min_max() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 1.0, 1.0]),
    )
    assert bbox is not None


def test_alignedbox3d_min_max_readwrite() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 1.0, 1.0]),
    )
    np.testing.assert_array_almost_equal(bbox.min, [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(bbox.max, [1.0, 1.0, 1.0])
    bbox.min = np.array([-1.0, -1.0, -1.0])
    bbox.max = np.array([2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(bbox.min, [-1.0, -1.0, -1.0])
    np.testing.assert_array_almost_equal(bbox.max, [2.0, 2.0, 2.0])


def test_alignedbox3d_diagonal() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 2.0, 3.0]),
    )
    diagonal = bbox.diagonal()
    np.testing.assert_array_almost_equal(diagonal, [1.0, 2.0, 3.0])


def test_alignedbox3d_contains_point_inside() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 1.0, 1.0]),
    )
    assert bbox.contains_point(np.array([0.5, 0.5, 0.5]))


def test_alignedbox3d_contains_point_outside() -> None:
    bbox = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([1.0, 1.0, 1.0]),
    )
    assert not bbox.contains_point(np.array([2.0, 2.0, 2.0]))


def test_alignedbox3d_contains_bbox() -> None:
    outer = pycolmap.AlignedBox3d(
        min=np.array([0.0, 0.0, 0.0]),
        max=np.array([10.0, 10.0, 10.0]),
    )
    inner = pycolmap.AlignedBox3d(
        min=np.array([1.0, 1.0, 1.0]),
        max=np.array([2.0, 2.0, 2.0]),
    )
    assert outer.contains_bbox(inner)
    assert not inner.contains_bbox(outer)
