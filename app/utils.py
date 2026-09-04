import base64
import io
from PIL import Image
import numpy as np


def b64_to_pil(b64str: str) -> Image.Image:
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert('RGBA')


def b64_to_seg_array(b64str: str) -> np.ndarray:
    """
    Decode a base64 segmentation/label map WITHOUT forcing it through RGBA.

    Segmentation maps store a class/region id per pixel (0=background,
    1=face, 2=left eyebrow, ...). Converting them to RGBA can quietly
    blend or reinterpolate pixel values at region boundaries, which
    corrupts the exact-match comparisons downstream (`seg_np == region_id`)
    and can make small/thin regions (eyes, nose, mouth, ears) lose enough
    pixels to disappear entirely from the output.

    This loader keeps the image in its native single-channel mode
    ('L'/'P'/'I') and only falls back to a channel-collapse if the source
    file genuinely isn't single-channel (which should not happen for a
    correctly-produced label map).
    """
    data = base64.b64decode(b64str)
    img = Image.open(io.BytesIO(data))

    if img.mode == 'P':
        # Palette image: resolve to raw index values, not RGB colors
        img = img.convert('L')
    elif img.mode not in ('L', 'I', 'I;16'):
        # Not single-channel as expected — collapse safely rather than
        # silently blending through RGBA. Log-worthy at the call site.
        img = img.convert('L')

    return np.array(img)


def pil_to_b64(img: Image.Image, fmt='PNG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def to_numpy(img: Image.Image):
    return np.array(img)

