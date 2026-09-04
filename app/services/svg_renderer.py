import math
from typing import List, Dict, Tuple, Sequence, Union
import numpy as np

Point = Tuple[int, int]
Contour = Union[np.ndarray, Sequence[Point]]


def _normalize_contour(contour: Contour) -> List[Point]:
    """
    Normalize an incoming contour to a list of (x, y) integer tuples.
    Accepts OpenCV contour formats or plain sequences.
    """
    if isinstance(contour, np.ndarray):
        if contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2:
            # OpenCV format (N, 1, 2)
            pts = contour.reshape(-1, 2)
        elif contour.ndim == 2 and contour.shape[1] == 2:
            pts = contour
        else:
            raise ValueError(f"Unsupported contour ndarray shape: {contour.shape}")
        return [(int(x), int(y)) for x, y in pts]
    # generic sequence
    return [(int(p[0]), int(p[1])) for p in contour]


def _contour_to_path_d(points: List[Point]) -> str:
    """
    Convert a list of points to an SVG path 'd' attribute.
    """
    if not points:
        return ""
    # move to first, then line to each subsequent
    parts = [f"M{points[0][0]},{points[0][1]}"]
    for x, y in points[1:]:
        parts.append(f"L{x},{y}")
    parts.append("Z")
    return " ".join(parts)


def generate_svg_from_contours(
    contours: List[Contour],
    width: int,
    height: int,
    fill: str = "#ffffff",
    stroke: str = "#000000",
    stroke_width: int = 1
) -> str:
    """
    Generate an SVG string from a list of contours.
    Uses evenodd fill rule to support holes.
    """
    norm_contours: List[List[Point]] = [_normalize_contour(c) for c in contours]

    paths = []
    for pts in norm_contours:
        if len(pts) < 3:
            continue
        d = _contour_to_path_d(pts)
        paths.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" fill-rule="evenodd"/>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(paths)
        + "</svg>"
    )
    return svg


def contours_to_mask_contours(contours: List[Contour]) -> Dict[str, List[List[Point]]]:
    """
    Convert contours to API mask_contours structure:
    { "1": [ [ (x,y), ... ], [ (x,y), ... ], ... ] }
    All contours grouped under a single mask id "1".
    """
    norm_contours: List[List[Point]] = []
    for c in contours:
        pts = _normalize_contour(c)
        if len(pts) >= 3:
            norm_contours.append(pts)
    return {"1": norm_contours}


__all__ = [
    "generate_svg_from_contours",
    "contours_to_mask_contours",
]
