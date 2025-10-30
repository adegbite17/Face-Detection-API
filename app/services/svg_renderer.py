from xml.sax.saxutils import escape
import random


def polygon_to_path(points):
    if not points:
        return ''
    d = 'M ' + ' L '.join(f'{x} {y}' for x, y in points) + ' Z'
    return d


def random_color_for_index(i):
    """Generate a deterministic random RGB color for each index."""
    random.seed(i)
    r = random.randint(50, 200)
    g = random.randint(50, 200)
    b = random.randint(50, 200)
    return f'rgb({r},{g},{b})'


def contours_to_svg(contours, size):
    """Convert a list of polygon contours into an SVG string."""
    width, height = size
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
    ]

    for i, contour in enumerate(contours):
        path = polygon_to_path(contour)
        color = random_color_for_index(i)
        svg_parts.append(
            f'<path d="{escape(path)}" '
            f'fill="{color}" fill-opacity="0.18" '
            f'stroke="{color}" stroke-dasharray="4 4" stroke-width="2" />'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)
