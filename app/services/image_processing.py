# app/services/image_processing.py
import cv2
import numpy as np
from PIL import Image
import imagehash
from typing import Tuple, List, Dict
import base64
from ..utils import b64_to_pil, pil_to_b64, to_numpy


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

        # Convert landmarks to proper format
        left_eye = np.array([float(landmarks[36][0]), float(landmarks[36][1])])
        right_eye = np.array([float(landmarks[45][0]), float(landmarks[45][1])])

        # Calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate image
        center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        return rotated
    except Exception as e:
        raise RuntimeError(f"Face alignment failed: {str(e)}")


def smooth_contour(contour: np.ndarray, smoothing_factor: float = 0.1) -> np.ndarray:
    """Apply Gaussian smoothing to contour points"""
    try:
        # Convert to periodic signal
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]

        # Apply Gaussian smoothing
        window = int(len(x) * smoothing_factor) | 1  # Ensure odd
        if window < 3:
            window = 3

        x_smooth = cv2.GaussianBlur(x, (window, 1), 0)
        y_smooth = cv2.GaussianBlur(y, (window, 1), 0)

        # Reconstruct contour
        smoothed = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
        return smoothed.reshape((-1, 1, 2))
    except Exception as e:
        raise RuntimeError(f"Contour smoothing failed: {str(e)}")


def generate_svg(img_shape: Tuple[int, int], contours: Dict[str, np.ndarray]) -> str:
    """Generate SVG with smooth, dashed contours"""
    try:
        height, width = img_shape[:2]

        # Convert dimensions to integers
        height = int(height)
        width = int(width)

        # SVG header
        svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'

        # Random colors for regions
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeead']

        # Add each contour as a path
        for i, (region_name, contour) in enumerate(contours.items()):
            color = colors[i % len(colors)]
            # Convert coordinates to integers
            points = ' '.join([
                f"M{int(p[0][0])},{int(p[0][1])}" if j == 0
                else f"L{int(p[0][0])},{int(p[0][1])}"
                for j, p in enumerate(contour)
            ])

            svg += f'''
                <path d="{points} Z" 
                    fill="{color}" 
                    fill-opacity="0.2"
                    stroke="{color}"
                    stroke-width="2"
                    stroke-dasharray="5,5"/>
            '''

        svg += '</svg>'
        return base64.b64encode(svg.encode()).decode()
    except Exception as e:
        raise RuntimeError(f"SVG generation failed: {str(e)}")


def process_request(image_b64: str, landmarks: List[List[float]],
                    segmentation_map_b64: str, db_cache=None) -> Tuple[Dict, str]:
    """Main processing function"""
    try:
        # Convert base64 to images
        img = b64_to_pil(image_b64)
        seg_map = b64_to_pil(segmentation_map_b64)

        # Convert to numpy for processing
        img_np = to_numpy(img)
        seg_np = to_numpy(seg_map)

        # Ensure correct data type
        if seg_np.dtype != np.uint8:
            seg_np = seg_np.astype(np.uint8)

        # Align face
        aligned = align_face(img_np, landmarks)

        # Process segmentation map regions
        regions = {}
        for region_id in range(1, 5):  # Assuming 4 regions
            # Create binary mask
            mask = (seg_np == region_id)
            # Convert to uint8 and scale to 0-255 range
            mask = (mask * 255).astype(np.uint8)

            # Ensure mask is 2D (single channel)
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]  # Take first channel if multi-channel

            # Optional: Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get largest contour
                largest = max(contours, key=cv2.contourArea)
                # Smooth contour
                smoothed = smooth_contour(largest)
                regions[f"region_{region_id}"] = smoothed

        # Generate SVG
        svg_b64 = generate_svg(aligned.shape, regions)

        # Convert contours to list format for response
        # Ensure all values are regular Python types (not numpy types)
        mask_contours = [
            [(int(p[0][0]), int(p[0][1])) for p in cont]
            for cont in regions.values()
        ]

        result = {
            "svg": svg_b64,
            "mask_contours": mask_contours
        }

        # Compute hash for caching
        phash = compute_phash_from_b64(image_b64)

        return result, phash

    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")
