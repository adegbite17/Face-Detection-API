import pytest
import numpy as np
from PIL import Image
import base64
from image_processing import (
    compute_phash_from_b64,
    align_face,
    smooth_contour,
    generate_svg,
    process_request
)
from utils1 import pil_to_b64


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    img = Image.new('RGB', (100, 100), 'white')
    return pil_to_b64(img)


@pytest.fixture
def sample_landmarks():
    """Create sample facial landmarks"""
    landmarks = [[0.0, 0.0]] * 68  # 68 landmark points
    # Set eye landmarks (indices 36 and 45)
    landmarks[36] = [30.0, 40.0]  # Left eye
    landmarks[45] = [70.0, 40.0]  # Right eye
    return landmarks


@pytest.fixture
def sample_segmentation_map():
    """Create a sample segmentation map"""
    seg_map = np.zeros((100, 100), dtype=np.uint8)
    # Create regions with different values
    seg_map[20:40, 20:40] = 1
    seg_map[50:70, 20:40] = 2
    seg_map[20:40, 50:70] = 3
    seg_map[50:70, 50:70] = 4
    img = Image.fromarray(seg_map)
    return pil_to_b64(img)


def test_compute_phash():
    """Test perceptual hash computation"""
    img = Image.new('RGB', (100, 100), 'white')
    b64_img = pil_to_b64(img)
    phash = compute_phash_from_b64(b64_img)
    assert isinstance(phash, str)
    assert len(phash) > 0


def test_align_face():
    """Test face alignment"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = [[0.0, 0.0]] * 68
    landmarks[36] = [30.0, 40.0]  # Left eye
    landmarks[45] = [70.0, 40.0]  # Right eye

    aligned = align_face(img, landmarks)
    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == img.shape


def test_smooth_contour():
    """Test contour smoothing"""
    # Create a simple square contour
    contour = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)
    smoothed = smooth_contour(contour)
    assert isinstance(smoothed, np.ndarray)
    assert smoothed.shape[2] == 2  # x,y coordinates


def test_generate_svg():
    """Test SVG generation"""
    img_shape = (100, 100)
    contours = {
        "region_1": np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)
    }
    svg_b64 = generate_svg(img_shape, contours)
    assert isinstance(svg_b64, str)
    # Test if it's valid base64
    svg_decoded = base64.b64decode(svg_b64).decode()
    assert svg_decoded.startswith('<svg')
    assert svg_decoded.endswith('</svg>')


def test_process_request(sample_image, sample_landmarks, sample_segmentation_map):
    """Test the main processing function"""
    result, phash = process_request(sample_image, sample_landmarks, sample_segmentation_map)

    assert isinstance(result, dict)
    assert "svg" in result
    assert "mask_contours" in result
    assert isinstance(phash, str)


def test_invalid_image():
    """Test handling of invalid image input"""
    with pytest.raises(Exception):
        compute_phash_from_b64("invalid_base64")


def test_invalid_landmarks():
    """Test handling of invalid landmarks"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    invalid_landmarks = [[0.0, 0.0]] * 2  # Too few landmarks

    with pytest.raises(Exception):
        align_face(img, invalid_landmarks)


@pytest.mark.parametrize("smoothing_factor", [0.05, 0.1, 0.2])
def test_smooth_contour_different_factors(smoothing_factor):
    """Test contour smoothing with different factors"""
    contour = np.array([[[i, i]] for i in range(10)], dtype=np.int32)
    smoothed = smooth_contour(contour, smoothing_factor)
    assert isinstance(smoothed, np.ndarray)
    assert smoothed.shape[0] == contour.shape[0]


def test_process_request_with_different_sizes():
    """Test processing with different image sizes"""
    sizes = [(100, 100), (200, 200), (150, 100)]

    for size in sizes:
        img = Image.new('RGB', size, 'white')
        seg_map = Image.new('L', size, 0)
        landmarks = [[0.0, 0.0]] * 68
        landmarks[36] = [size[0] * 0.3, size[1] * 0.4]  # Left eye
        landmarks[45] = [size[0] * 0.7, size[1] * 0.4]  # Right eye

        result, phash = process_request(
            pil_to_b64(img),
            landmarks,
            pil_to_b64(seg_map)
        )
        assert isinstance(result, dict)
        assert isinstance(phash, str)


def test_process_request_segmentation_regions():
    """Test processing with different segmentation regions"""
    img = Image.new('RGB', (100, 100), 'white')
    seg_map = np.zeros((100, 100), dtype=np.uint8)

    # Create distinct regions
    for i in range(1, 5):
        seg_map[25 * i:25 * (i + 1), 25:75] = i

    landmarks = [[0.0, 0.0]] * 68
    landmarks[36] = [30.0, 40.0]
    landmarks[45] = [70.0, 40.0]

    result, _ = process_request(
        pil_to_b64(img),
        landmarks,
        pil_to_b64(Image.fromarray(seg_map))
    )

    assert len(result["mask_contours"]) > 0
