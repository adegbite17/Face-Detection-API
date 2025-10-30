import base64
import io
from PIL import Image
import numpy as np


def b64_to_pil(b64str: str) -> Image.Image:
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert('RGBA')


def pil_to_b64(img: Image.Image, fmt='PNG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def to_numpy(img: Image.Image):
    return np.array(img)