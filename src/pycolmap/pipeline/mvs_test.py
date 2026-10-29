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



def test_patch_match_ncc_cost_unorm16_option() -> None:
    options = pycolmap.PatchMatchOptions()
    assert options.ncc_cost_unorm16 is False
    assert options.todict()["ncc_cost_unorm16"] is False
    options.ncc_cost_unorm16 = True
    options.checkpoint_hmm = False
    assert not options.check()
    copied = pycolmap.PatchMatchOptions(options.todict())
    assert copied.ncc_cost_unorm16 is True
    assert copied.checkpoint_hmm is False
    assert not copied.check()
    options.checkpoint_hmm = True
    options.window_radius = 4
    assert not options.check()
    options.window_radius = 5
    options.window_step = 2
    assert not options.check()
    options.window_step = 1
    assert options.check() == pycolmap.has_cuda
