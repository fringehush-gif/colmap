# SPDX-License-Identifier: BSD-3-Clause

import threading
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


def test_gray_bitmap_read_scope_native_copy_and_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resident-only.png"
    gray = np.array([[0, 71, 255], [19, 128, 234]], dtype=np.uint8)
    calls = []

    def loader(requested: Path) -> np.ndarray:
        calls.append(requested)
        return gray

    with pycolmap.BitmapReadScope.gray_only(
        [path], loader, True, True, [(3, 2)]
    ) as scope:
        assert scope.identity_token > 0
        assert scope.image_size(path) == (3, 2)
        assert calls == []  # metadata does not materialize pixels
        bitmap = pycolmap.Bitmap.read(path, as_rgb=False)
        assert bitmap is not None and bitmap.is_grey
        np.testing.assert_array_equal(bitmap.to_array(), gray)
        assert calls == [path]
        assert scope.stats()["bitmapReads"] == 1
        assert scope.stats()["ownedRgbCopyBytes"] == 0
        assert scope.stats()["grayOutputBytes"] == gray.size
    assert scope.closed
    gray[0, 0] = 99
    assert bitmap.to_array()[0, 0] == 0  # native Bitmap owns the copied bytes


def test_gray_bitmap_read_scope_refuses_rgb_before_loader(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resident-only.png"
    calls = []
    with pycolmap.BitmapReadScope.gray_only(
        [path], lambda requested: calls.append(requested), True, True, [(2, 1)]
    ) as scope:
        with pytest.raises(ValueError, match="refuses RGB reads"):
            pycolmap.Bitmap.read(path, as_rgb=True)
        assert calls == []
        assert scope.stats()["bitmapReads"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.zeros((2, 3), dtype=np.uint16),
        np.zeros((2, 6), dtype=np.uint8)[:, ::2],
    ],
)
def test_gray_bitmap_read_scope_rejects_nonpacked_2d_uint8(
    tmp_path: Path, payload: np.ndarray
) -> None:
    path = tmp_path / "resident-only.png"
    with pycolmap.BitmapReadScope.gray_only(
        [path], lambda requested: payload, True, True, [(3, 2)]
    ):
        with pytest.raises(ValueError, match="packed 2D uint8"):
            pycolmap.Bitmap.read(path, as_rgb=False)


def test_gray_bitmap_read_scope_rejects_wrong_grid_and_linearized_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resident-only.png"
    payload = np.zeros((2, 3), dtype=np.uint8)
    with pycolmap.BitmapReadScope.gray_only(
        [path], lambda requested: payload, True, True, [(4, 2)]
    ):
        with pytest.raises(ValueError, match="declared resident dimensions"):
            pycolmap.Bitmap.read(path, as_rgb=False)
        with pytest.raises(ValueError, match="linear-light"):
            pycolmap.Bitmap.read(path, as_rgb=False, linearize_colorspace=True)


@pytest.mark.parametrize(
    "normalized_srgb,external", [(False, True), (True, False)]
)
def test_gray_bitmap_read_scope_requires_admission(
    tmp_path: Path, normalized_srgb: bool, external: bool
) -> None:
    with pytest.raises(ValueError, match="normalized sRGB and external camera"):
        pycolmap.BitmapReadScope.gray_only(
            [tmp_path / "resident-only.png"],
            lambda requested: None,
            normalized_srgb,
            external,
            [(3, 2)],
        )


def test_rgb_bitmap_read_scope_keeps_existing_gray_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resident-only.png"
    rgb = np.array([[[0, 71, 255], [19, 128, 234]]], dtype=np.uint8)
    expected = pycolmap.Bitmap.from_array(rgb).clone_as_grey().to_array()
    with pycolmap.BitmapReadScope(
        [path], lambda requested: rgb, True, True, [(2, 1)]
    ) as scope:
        color = pycolmap.Bitmap.read(path, as_rgb=True)
        gray = pycolmap.Bitmap.read(path, as_rgb=False)
        assert color is not None and gray is not None
        np.testing.assert_array_equal(color.to_array(), rgb)
        np.testing.assert_array_equal(gray.to_array(), expected)
        assert scope.stats()["bitmapReads"] == 2
        assert scope.stats()["ownedRgbCopyBytes"] == rgb.size
        assert scope.stats()["grayOutputBytes"] == 2


def test_gray_bitmap_read_scope_requires_declared_roster_grid(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires image_sizes_wh"):
        pycolmap.BitmapReadScope.gray_only(
            [tmp_path / "resident-only.png"],
            lambda requested: None,
            True,
            True,
            [],
        )


def test_gray_bitmap_read_scope_rejects_overlapping_exact_owner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resident-only.png"
    gray = np.zeros((2, 3), dtype=np.uint8)
    with pycolmap.BitmapReadScope.gray_only(
        [path], lambda requested: gray, True, True, [(3, 2)]
    ):
        with pytest.raises(ValueError, match="already has an owner"):
            pycolmap.BitmapReadScope.gray_only(
                [path], lambda requested: gray, True, True, [(3, 2)]
            )


def test_gray_bitmap_read_scope_close_preserves_admitted_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resident-only.png"
    gray = np.array([[1, 2, 3]], dtype=np.uint8)
    entered = threading.Event()
    release = threading.Event()
    outputs = []
    errors = []

    def loader(requested: Path) -> np.ndarray:
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("loader join timed out")
        return gray

    scope = pycolmap.BitmapReadScope.gray_only(
        [path], loader, True, True, [(3, 1)]
    )

    def read() -> None:
        try:
            outputs.append(pycolmap.Bitmap.read(path, as_rgb=False))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=read)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        scope.close()
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []
    assert len(outputs) == 1 and outputs[0] is not None
    np.testing.assert_array_equal(outputs[0].to_array(), gray)
