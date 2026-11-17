# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest

import pycolmap


def test_bitmap_rescale_filter_enum() -> None:
    assert {
        k: int(v) for k, v in pycolmap.BitmapRescaleFilter.__members__.items()
    } == {
        "BILINEAR": 0,
        "BOX": 1,
    }


def test_bitmap_default_init() -> None:
    bitmap = pycolmap.Bitmap()
    assert bitmap is not None
    assert bitmap.is_empty


def test_bitmap_init_width_height_rgb() -> None:
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    assert bitmap.width == 64
    assert bitmap.height == 48
    assert bitmap.is_rgb


def test_bitmap_init_width_height_grey() -> None:
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=False)
    assert bitmap.width == 64
    assert bitmap.height == 48
    assert bitmap.is_grey


def test_bitmap_init_with_linear_colorspace() -> None:
    bitmap = pycolmap.Bitmap(
        width=64, height=48, as_rgb=True, linear_colorspace=True
    )
    assert bitmap.width == 64
    assert bitmap.height == 48


def test_bitmap_readonly_props() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    assert isinstance(bitmap.width, int)
    assert isinstance(bitmap.height, int)
    assert isinstance(bitmap.channels, int)
    assert isinstance(bitmap.is_rgb, bool)
    assert isinstance(bitmap.is_grey, bool)
    assert isinstance(bitmap.is_empty, bool)
    assert isinstance(bitmap.bits_per_pixel, int)
    assert isinstance(bitmap.pitch, int)
    assert bitmap.channels == 3
    assert bitmap.bits_per_pixel == 24
    assert not bitmap.is_empty


def test_bitmap_readonly_props_grey() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=False)
    assert bitmap.channels == 1
    assert bitmap.bits_per_pixel == 8
    assert bitmap.is_grey
    assert not bitmap.is_rgb


def test_bitmap_from_array_grey() -> None:
    array = np.zeros((48, 64), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array)
    assert bitmap.width == 64
    assert bitmap.height == 48
    assert bitmap.is_grey


def test_bitmap_from_array_rgb() -> None:
    array = np.zeros((48, 64, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array)
    assert bitmap.width == 64
    assert bitmap.height == 48
    assert bitmap.is_rgb


@pytest.mark.parametrize("as_rgb", [False, True])
def test_bitmap_from_array_noncontiguous(as_rgb: bool) -> None:
    shape = (6, 8, 3) if as_rgb else (6, 8)
    source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    view = source[::2, ::-1]
    assert not view.flags.c_contiguous
    bitmap = pycolmap.Bitmap.from_array(view)
    np.testing.assert_array_equal(bitmap.to_array(), view)


def test_bitmap_from_array_broadcast() -> None:
    view = np.broadcast_to(np.array([[71]], dtype=np.uint8), (3, 5))
    assert not view.flags.c_contiguous
    bitmap = pycolmap.Bitmap.from_array(view)
    np.testing.assert_array_equal(bitmap.to_array(), view)


@pytest.mark.parametrize("as_rgb", [False, True])
def test_bitmap_from_array_memoryview_noncontiguous(as_rgb: bool) -> None:
    shape = (6, 8, 3) if as_rgb else (6, 8)
    source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    view = source[::2, ::-1]
    buffer = memoryview(view)
    assert not buffer.c_contiguous
    bitmap = pycolmap.Bitmap.from_array(buffer)
    np.testing.assert_array_equal(bitmap.to_array(), view)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, object])
def test_bitmap_from_array_memoryview_rejects_unsafe_dtype(dtype) -> None:
    buffer = memoryview(np.zeros((3, 5), dtype=dtype))
    with pytest.raises(TypeError):
        pycolmap.Bitmap.from_array(buffer)


def test_bitmap_from_array_sequence() -> None:
    source = [[0, 17, 255], [71, 131, 3]]
    bitmap = pycolmap.Bitmap.from_array(source)
    np.testing.assert_array_equal(
        bitmap.to_array(), np.array(source, dtype=np.uint8)
    )


@pytest.mark.parametrize("dtype", [np.int16, np.float32, object])
def test_bitmap_from_array_rejects_unsafe_dtype(dtype) -> None:
    source = np.zeros((3, 5), dtype=dtype)
    with pytest.raises(TypeError):
        pycolmap.Bitmap.from_array(source)


def test_bitmap_to_array_roundtrip_grey() -> None:
    array_in = np.random.randint(0, 256, (48, 64), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array_in)
    array_out = bitmap.to_array()
    np.testing.assert_array_equal(array_in, array_out)


def test_bitmap_to_array_roundtrip_rgb() -> None:
    array_in = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array_in)
    array_out = bitmap.to_array()
    np.testing.assert_array_equal(array_in, array_out)


def test_bitmap_from_array_linear_colorspace() -> None:
    array = np.zeros((48, 64, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array, linear_colorspace=True)
    assert bitmap.is_rgb


def test_bitmap_clone() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    cloned = bitmap.clone()
    assert cloned.width == bitmap.width
    assert cloned.height == bitmap.height
    assert cloned.is_rgb == bitmap.is_rgb


def test_bitmap_clone_as_grey() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    grey = bitmap.clone_as_grey()
    assert grey.is_grey
    assert grey.width == bitmap.width
    assert grey.height == bitmap.height


def test_bitmap_clone_as_rgb() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=False)
    rgb = bitmap.clone_as_rgb()
    assert rgb.is_rgb
    assert rgb.width == bitmap.width
    assert rgb.height == bitmap.height


def test_bitmap_rescale_default_filter() -> None:
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    bitmap.rescale(32, 24)
    assert bitmap.width == 32
    assert bitmap.height == 24


def test_bitmap_rescale_explicit_filter() -> None:
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    bitmap.rescale(32, 24, filter=pycolmap.BitmapRescaleFilter.BOX)
    assert bitmap.width == 32
    assert bitmap.height == 24


def test_bitmap_rot90_once() -> None:
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    bitmap.rot90(1)
    assert bitmap.width == 48
    assert bitmap.height == 64


def test_bitmap_rot90_twice() -> None:
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    bitmap.rot90(2)
    assert bitmap.width == 64
    assert bitmap.height == 48


def test_bitmap_write_read(tmp_path: Path) -> None:
    array = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array)
    filepath = str(tmp_path / "test.png")
    bitmap.write(filepath)
    loaded = pycolmap.Bitmap.read(filepath, as_rgb=True)
    assert loaded is not None
    assert loaded.width == 64
    assert loaded.height == 48
    np.testing.assert_array_equal(loaded.to_array(), array)


def test_bitmap_set_jpeg_quality() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    bitmap.set_jpeg_quality(85)


def test_bitmap_exif_orientation() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_orientation()
    # Synthetic bitmap has no EXIF data, so expect None.
    assert result is None


def test_bitmap_exif_camera_model() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_camera_model()
    assert result is None


def test_bitmap_exif_focal_length() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_focal_length()
    assert result is None


def test_bitmap_exif_latitude() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_latitude()
    assert result is None


def test_bitmap_exif_longitude() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_longitude()
    assert result is None


def test_bitmap_exif_altitude() -> None:
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_altitude()
    assert result is None


def test_bitmap_thumbnail() -> None:
    bitmap = pycolmap.Bitmap(width=200, height=100, as_rgb=True)
    scale = bitmap.thumbnail(max_image_size=50)
    # The longest side is scaled to fit max_image_size: 50 / max(200, 100).
    assert scale == 0.25
    assert bitmap.width == 50
    assert bitmap.height == 25


def test_bitmap_thumbnail_noop() -> None:
    bitmap = pycolmap.Bitmap(width=40, height=30, as_rgb=True)
    scale = bitmap.thumbnail(max_image_size=100)
    assert scale == 1.0
    assert bitmap.width == 40
    assert bitmap.height == 30


@pytest.mark.parametrize("as_buffer", [False, True])
@pytest.mark.parametrize(
    ("shape", "error"),
    [
        ((1, (1 << 32) + 1), "dimensions must be positive and fit in int"),
        (((1 << 32) + 1, 1), "dimensions must be positive and fit in int"),
        ((1, 1, (1 << 32) + 1), "must have 1 or 3 channels"),
        ((46341, 46341), "exceeds bitmap storage capacity"),
        ((1, (1 << 31) // 3 + 1, 3), "exceeds bitmap storage capacity"),
    ],
)
def test_bitmap_from_array_rejects_oversized_buffer_before_copy(
    shape, error, as_buffer
):
    if np.dtype(np.intc).itemsize != 4 or np.dtype(np.intp).itemsize <= 4:
        pytest.skip("requires 32-bit C++ int and wider NumPy dimensions")
    # Every virtual pixel refers to the same allocated byte. Unlike fabricated
    # contiguous strides, this is a valid buffer even at these large shapes.
    backing = np.array(71, dtype=np.uint8)
    view = np.broadcast_to(backing, shape)
    assert all(stride == 0 for stride in view.strides)
    assert np.shares_memory(backing, view)
    source = memoryview(view) if as_buffer else view
    with pytest.raises(RuntimeError, match=error):
        pycolmap.Bitmap.from_array(source)
    assert backing.item() == 71


@pytest.mark.parametrize("width,height", [(0, 0), (0, 4), (4, 0)])
@pytest.mark.parametrize("as_rgb", [False, True])
def test_bitmap_to_array_empty_dimensions(width, height, as_rgb):
    bitmap = pycolmap.Bitmap(width=width, height=height, as_rgb=as_rgb)
    array = bitmap.to_array()
    expected_shape = (height, width, 3) if as_rgb else (height, width)
    assert array.shape == expected_shape
    assert array.dtype == np.uint8
    assert array.size == 0
    assert array.flags.c_contiguous
