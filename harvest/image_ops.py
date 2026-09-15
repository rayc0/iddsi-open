from __future__ import annotations

import hashlib
import io
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImage(ValueError):
    pass


def decode_image(data: bytes, *, max_pixels: int = 50_000_000) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width * opened.height > max_pixels:
                raise InvalidImage(f"image exceeds {max_pixels} pixels")
            if opened.width < 64 or opened.height < 64:
                raise InvalidImage("image is smaller than 64x64")
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, InvalidImage):
            raise
        raise InvalidImage(f"cannot decode image: {exc}") from exc
    return image


@lru_cache(maxsize=4)
def _dct_matrix(size: int) -> np.ndarray:
    points = np.arange(size, dtype=np.float64)
    frequencies = points[:, None]
    matrix = np.cos((np.pi / size) * (points + 0.5) * frequencies)
    matrix[0, :] *= 1.0 / np.sqrt(size)
    matrix[1:, :] *= np.sqrt(2.0 / size)
    return matrix


def perceptual_hash(image: Image.Image, *, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Return a conventional 64-bit DCT pHash as sixteen lowercase hex digits."""
    size = hash_size * highfreq_factor
    grayscale = ImageOps.grayscale(image).resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float64)
    transform = _dct_matrix(size) @ pixels @ _dct_matrix(size).T
    low = transform[:hash_size, :hash_size]
    median = float(np.median(low[1:, :]))
    bits = (low > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


class PHashIndex:
    def __init__(self, threshold: int = 8) -> None:
        if not 0 <= threshold <= 64:
            raise ValueError("pHash threshold must be between 0 and 64")
        self.threshold = threshold
        self._items: list[tuple[str, str]] = []

    def add(self, event_id: str, value: str) -> None:
        self._items.append((event_id, value))

    def nearest_duplicate(self, value: str) -> tuple[str, int] | None:
        best: tuple[str, int] | None = None
        for event_id, existing in self._items:
            distance = hamming_distance(value, existing)
            if distance <= self.threshold and (best is None or distance < best[1]):
                best = (event_id, distance)
        return best


def normalized_jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, optimize=True, progressive=True)
    return output.getvalue()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
