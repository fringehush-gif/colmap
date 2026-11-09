# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import pycolmap


def test_normalization_enum() -> None:
    assert pycolmap.Normalization.L1_ROOT is not None
    assert pycolmap.Normalization.L2 is not None


def test_sift_extraction_options_default_init() -> None:
    options = pycolmap.SiftExtractionOptions()
    assert options is not None


def test_sift_extraction_options_num_octaves_readwrite() -> None:
    options = pycolmap.SiftExtractionOptions()
    original = options.num_octaves
    assert isinstance(original, int)
    options.num_octaves = 5
    assert options.num_octaves == 5


def test_sift_extraction_options_max_num_features_readwrite() -> None:
    options = pycolmap.SiftExtractionOptions()
    assert isinstance(options.max_num_features, int)
    options.max_num_features = 4096
    assert options.max_num_features == 4096


def test_sift_extraction_options_peak_threshold_readwrite() -> None:
    options = pycolmap.SiftExtractionOptions()
    assert isinstance(options.peak_threshold, float)
    options.peak_threshold = 0.01
    assert abs(options.peak_threshold - 0.01) < 1e-6


def test_sift_extraction_options_check() -> None:
    options = pycolmap.SiftExtractionOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_extraction_options_default_init() -> None:
    options = pycolmap.FeatureExtractionOptions()
    assert options is not None


def test_feature_extraction_options_type_readwrite() -> None:
    options = pycolmap.FeatureExtractionOptions()
    options.type = pycolmap.FeatureExtractorType.SIFT
    assert options.type == pycolmap.FeatureExtractorType.SIFT


def test_feature_extraction_options_max_image_size_readwrite() -> None:
    options = pycolmap.FeatureExtractionOptions()
    options.max_image_size = 2048
    assert options.max_image_size == 2048


def test_feature_extraction_options_num_threads_readwrite() -> None:
    options = pycolmap.FeatureExtractionOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_feature_extraction_options_use_gpu_readwrite() -> None:
    options = pycolmap.FeatureExtractionOptions()
    options.use_gpu = False
    assert options.use_gpu is False


def test_feature_extraction_options_sift_property() -> None:
    options = pycolmap.FeatureExtractionOptions()
    sift = options.sift
    assert isinstance(sift, pycolmap.SiftExtractionOptions)


def test_feature_extraction_options_requires_rgb() -> None:
    options = pycolmap.FeatureExtractionOptions()
    result = options.requires_rgb()
    assert isinstance(result, bool)


def test_feature_extraction_options_requires_opengl() -> None:
    options = pycolmap.FeatureExtractionOptions()
    result = options.requires_opengl()
    assert isinstance(result, bool)


def test_feature_extraction_options_eff_max_image_size() -> None:
    options = pycolmap.FeatureExtractionOptions()
    result = options.eff_max_image_size()
    assert isinstance(result, int)
    assert result > 0


def test_feature_extraction_options_check() -> None:
    options = pycolmap.FeatureExtractionOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_extractor_create() -> None:
    extractor = pycolmap.FeatureExtractor.create(device=pycolmap.Device.cpu)
    assert extractor is not None


def test_feature_extractor_array_paths_match_bitmap() -> None:
    # Binary pixels normalize exactly to float32 zero/one. This isolates the
    # shared Bitmap conversion from float-to-byte quantization differences.
    y, x = np.indices((128, 128))
    pattern = (37 * x + 17 * y + (x * y) % 251) % 13
    image = (pattern < 6).astype(np.uint8) * np.uint8(255)
    normalized = image.astype(np.float32) / np.float32(255)
    options = pycolmap.FeatureExtractionOptions()
    options.num_threads = 1
    options.use_gpu = False
    extractor = pycolmap.FeatureExtractor.create(
        options=options, device=pycolmap.Device.cpu
    )
    reference_keypoints, reference_descriptors = extractor.extract(
        pycolmap.Bitmap.from_array(image)
    )
    assert len(reference_keypoints) > 0
    reference = pycolmap.keypoints_to_matrix(reference_keypoints)
    for keypoints, descriptors in (
        extractor.extract_from_uint8_array(image),
        extractor.extract_from_float32_array(normalized),
    ):
        np.testing.assert_array_equal(
            pycolmap.keypoints_to_matrix(keypoints), reference
        )
        assert descriptors.type == reference_descriptors.type
        np.testing.assert_array_equal(descriptors.data, reference_descriptors.data)
