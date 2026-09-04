import cv2
import numpy as np
import imagehash
from typing import Tuple, List, Dict
import base64
import logging
from ..utils import b64_to_pil, b64_to_seg_array, to_numpy

logger = logging.getLogger(__name__)


def create_face_mask(image_shape: tuple, face_data: dict) -> np.ndarray:
    """Create elliptical mask from face detection box"""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    box = face_data['box']
    x, y, w, h = box['x'], box['y'], box['width'], box['height']

    # creates elliptical mask for natural face shape
    center = (x + w // 2, y + h // 2)

    # makes ellipse slightly larger than box
    axes = (int(w * 0.55), int(h * 0.65))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    return mask


def process_mask_for_contours(mask: np.ndarray) -> np.ndarray:
    """Process mask to create complete face boundary"""
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    # morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # fill holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, -1)

    # smooth edges
    filled_mask = cv2.GaussianBlur(filled_mask, (5, 5), 0)
    _, filled_mask = cv2.threshold(filled_mask, 127, 255, cv2.THRESH_BINARY)

    return filled_mask


def extract_mask_contours(mask: np.ndarray, mask_id: int = 255) -> Dict[str, List]:
    """
    Extract contours with hierarchy information.
    Returns dict with 'outer' and 'inner' contour lists.
    """
    binary_mask = (mask == mask_id).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {"outer": [], "inner": []}

    outer_contours = []
    inner_contours = []

    if hierarchy is not None:
        hierarchy = hierarchy[0]

        for i, contour in enumerate(contours):
            if len(contour) >= 3:
                points = contour.squeeze().tolist()
                if isinstance(points[0], int):
                    points = [points]

                # hierarchy[i][3] == -1 means outer contour
                if hierarchy[i][3] == -1:
                    outer_contours.append(points)
                else:
                    inner_contours.append(points)

    return {"outer": outer_contours, "inner": inner_contours}


def compute_phash_from_b64(b64_img: str) -> str:
    """Compute perceptual hash from base64 image"""
    try:
        img = b64_to_pil(b64_img)
        phash = str(imagehash.average_hash(img))
        return phash
    except Exception as e:
        raise RuntimeError(f"Failed to compute perceptual hash: {str(e)}")


def align_face(img: np.ndarray, landmarks: List[List[float]]) -> np.ndarray:
    """Align face using eye landmarks"""
    try:
        # Ensure image is in correct format
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # convert landmarks to proper format
        left_eye = np.array([float(landmarks[36][0]), float(landmarks[36][1])])
        right_eye = np.array([float(landmarks[45][0]), float(landmarks[45][1])])

        # calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # rotate image
        center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        return rotated
    except Exception as e:
        raise RuntimeError(f"Face alignment failed: {str(e)}")


def smooth_contour(contour: np.ndarray, smoothing_factor: float = 0.02) -> np.ndarray:
    """Apply Gaussian smoothing to contour points"""
    try:
        x = contour[:, 0, 0].astype(float)
        y = contour[:, 0, 1].astype(float)

        # use smaller window for less aggressive smoothing
        window = max(3, int(len(x) * smoothing_factor) | 1)

        x_smooth = cv2.GaussianBlur(x.reshape(-1, 1), (window, 1), 0).flatten()
        y_smooth = cv2.GaussianBlur(y.reshape(-1, 1), (window, 1), 0).flatten()

        smoothed = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
        return smoothed.reshape((-1, 1, 2))
    except Exception as e:
        raise RuntimeError(f"Contour smoothing failed: {str(e)}")



def generate_svg(img_shape: Tuple[int, int], contours: Dict[str, np.ndarray]) -> str:
    """Generate SVG with smooth, dashed contours"""
    try:
        height, width = img_shape[:2]
        height = int(height)
        width = int(width)

        svg_parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeead']

        for i, (region_name, contour) in enumerate(contours.items()):
            color = colors[i % len(colors)]

            # build path correctly with proper spacing
            path_commands = []
            for j, point in enumerate(contour):
                x = int(point[0][0])
                y = int(point[0][1])
                if j == 0:
                    path_commands.append(f"M{x} {y}")
                else:
                    path_commands.append(f"L{x} {y}")

            # join all commands with space and close path
            path_data = ' '.join(path_commands) + ' Z'

            svg_parts.append(f'''
                <path d="{path_data}" 
                    fill="{color}" 
                    fill-opacity="0.2"
                    stroke="{color}"
                    stroke-width="2"
                    stroke-dasharray="5,5"/>
            ''')

        svg_parts.append('</svg>')
        svg_string = ''.join(svg_parts)

        return base64.b64encode(svg_string.encode()).decode()
    except Exception as e:
        raise RuntimeError(f"SVG generation failed: {str(e)}")


def process_request(image_b64: str, landmarks: List[List[float]],
                    segmentation_map_b64: str, db_cache=None) -> Tuple[Dict, str]:
    """Main processing function"""
    try:
        # convert base64 to images
        img = b64_to_pil(image_b64)
        img_np = to_numpy(img)

        # segmentation map: use the raw-label loader, NOT b64_to_pil.
        # b64_to_pil forces RGBA, which can blend/interpolate pixel values
        # at region edges and corrupt the exact `seg_np == region_id`
        # matches below — small regions (eyes, nose, mouth, ears) are the
        # first to lose enough pixels to vanish.
        seg_np = b64_to_seg_array(segmentation_map_b64)

        if seg_np.ndim != 2:
            # should not happen with a correctly-produced label map; log
            # loudly instead of silently proceeding with wrong comparisons.
            logger.warning(
                f"Segmentation map is not single-channel (shape={seg_np.shape}); "
                "collapsing to first channel. Check how this map was produced."
            )
            seg_np = seg_np[..., 0]

        if seg_np.dtype != np.uint8:
            seg_np = seg_np.astype(np.uint8)

        # align face
        aligned = align_face(img_np, landmarks)

        # get unique region IDs dynamically
        unique_regions = np.unique(seg_np)
        unique_regions = unique_regions[unique_regions > 0]  # Exclude background (0)
        logger.info(f"Segmentation map contains {len(unique_regions)} label(s): {sorted(unique_regions.tolist())}")

        # process segmentation map regions
        regions = {}
        dropped_regions = []
        for region_id in unique_regions:
            # Create binary mask
            mask = (seg_np == region_id).astype(np.uint8) * 255
            pixel_count = int(np.count_nonzero(mask))

            # clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # find ALL contours (not just largest)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if contours:
                # use CHAIN_APPROX_NONE to keep all points
                largest = max(contours, key=cv2.contourArea)
                # reduce smoothing factor for more detail
                smoothed = smooth_contour(largest, smoothing_factor=0.02)
                regions[f"region_{int(region_id)}"] = smoothed
            else:
                dropped_regions.append((int(region_id), pixel_count))

        if dropped_regions:
            # this is exactly the failure mode that produced a partial SVG
            # (e.g. an eyebrow present but the matching eye missing): a
            # label existed in the map but had too few/no matching pixels
            # to trace a contour. Surface it instead of returning silently.
            logger.warning(
                f"{len(dropped_regions)} region(s) present in the segmentation "
                f"map produced no contour and were dropped: {dropped_regions} "
                "(label_id, pixel_count). Check the segmentation map source "
                "for lossy resizing/compression before it reached this pipeline."
            )

        # generate SVG
        svg_b64 = generate_svg(aligned.shape, regions)

        # convert contours preserve all points
        mask_contours = {
            region_name: [[int(p[0][0]), int(p[0][1])] for p in cont]
            for region_name, cont in regions.items()
        }

        result = {
            "svg": svg_b64,
            "mask_contours": mask_contours
        }

        # compute hash for caching
        phash = compute_phash_from_b64(image_b64)

        return result, phash

    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")
