# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pycolmap

pytestmark = pytest.mark.skipif(
    not hasattr(pycolmap, "PatchMatchOptions"),
    reason="PatchMatchOptions not available (requires CUDA or HIP)",
)


def test_patch_match_options_init() -> None:
    options = pycolmap.PatchMatchOptions()
    assert options is not None


def test_stereo_fusion_options_init() -> None:
    options = pycolmap.StereoFusionOptions()
    assert options is not None


def test_stereo_fusion_fp32_normal_tile_policy() -> None:
    options = pycolmap.StereoFusionOptions()
    assert options.bounded_cache_normal_tile_size == 0
    assert options.bounded_cache_normal_mapped_bytes == 0
    assert options.bounded_cache_normal_mapped_files == 0
    options.bounded_cache = True
    options.bounded_cache_components = True
    options.bounded_cache_normal_tile_size = 4
    options.bounded_cache_normal_mapped_bytes = 1 << 20
    options.bounded_cache_normal_mapped_files = 2
    assert options.check()
    options.bounded_cache_neighborhood = True
    assert options.check()


def test_stereo_fusion_signed_normal_guard_policy() -> None:
    options = pycolmap.StereoFusionOptions()
    assert not options.bounded_cache_normal_snorm16
    assert not options.bounded_cache_normal_snorm16_guard
    options.bounded_cache_normal_snorm16_guard = True
    assert not options.check()
    options.bounded_cache_normal_snorm16 = True
    options.bounded_cache = True
    options.bounded_cache_components = True
    options.bounded_cache_normal_tile_size = 4
    options.bounded_cache_normal_mapped_bytes = 1 << 20
    options.bounded_cache_normal_mapped_files = 2
    assert options.check() == options.supports_normal_precision_guard()
    options.bounded_cache_normal_snorm16_guard = False
    assert options.check()

@pytest.mark.parametrize(
    "name",
    [
        "patch_match_stereo",
        "stereo_fusion",
    ],
)
def test_public_api_callable(name: str) -> None:
    assert callable(getattr(pycolmap, name))


def test_stereo_fusion_full_ram_load_threads_api() -> None:
    options = pycolmap.StereoFusionOptions()
    assert options.full_ram_load_threads == -1
    assert options.todict()["full_ram_load_threads"] == -1
    assert options.check()
    for workers in (1, 4):
        options.full_ram_load_threads = workers
        assert options.check()
        copied = pycolmap.StereoFusionOptions(options.todict())
        assert copied.full_ram_load_threads == workers
        assert copied.check()
    for invalid in (-2, 0):
        options.full_ram_load_threads = invalid
        assert not options.check()
    options.full_ram_load_threads = 2
    options.use_cache = True
    assert not options.check()
    options.use_cache = False
    options.bounded_cache = True
    assert not options.check()
    options.full_ram_load_threads = -1
    assert options.check()


def test_stereo_fusion_bounded_prefill_threads_api() -> None:
    options = pycolmap.StereoFusionOptions()
    assert options.bounded_cache_prefill_threads == -1
    assert options.todict()["bounded_cache_prefill_threads"] == -1
    assert options.check()
    options.bounded_cache_prefill_threads = 2
    assert not options.check()
    options.bounded_cache = True
    assert not options.check()
    options.bounded_cache_neighborhood = True
    for workers in (1, 4):
        options.bounded_cache_prefill_threads = workers
        assert options.check()
        copied = pycolmap.StereoFusionOptions(options.todict())
        assert copied.bounded_cache_prefill_threads == workers
        assert copied.check()
    for invalid in (-2, 0):
        options.bounded_cache_prefill_threads = invalid
        assert not options.check()
    options.bounded_cache_prefill_threads = -1
    assert options.check()
