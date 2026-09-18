"""Deciding what an upload actually is, and decoding it safely.

Type comes from magic bytes, never the filename. An extension is a
claim by the client; the first bytes are evidence.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from .errors import CorruptMediaError, MediaTooLargeError, UnsupportedMediaError

MediaKind = Literal["image", "video"]

IMAGE_FORMATS = frozenset({"png", "jpeg", "webp", "bmp", "tiff", "gif"})
VIDEO_FORMATS = frozenset({"mp4", "webm", "avi", "mov"})

# Pillow's own bomb guard is a DecompressionBombWarning by default. We do our
# own explicit check and raise this ceiling out of the way of it.
Image.MAX_IMAGE_PIXELS = 64_000_000


@dataclass(frozen=True)
class MediaInfo:
    kind: MediaKind
    format: str


def _match_prefix(data: bytes) -> MediaInfo | None:
    prefixes: tuple[tuple[bytes, MediaInfo], ...] = (
        (b"\x89PNG\r\n\x1a\n", MediaInfo("image", "png")),
        (b"\xff\xd8\xff", MediaInfo("image", "jpeg")),
        (b"BM", MediaInfo("image", "bmp")),
        (b"II*\x00", MediaInfo("image", "tiff")),
        (b"MM\x00*", MediaInfo("image", "tiff")),
        # An animated GIF is handled as an image: load_image() converts, which
        # takes the first frame. Callers wanting motion should upload a video.
        (b"GIF87a", MediaInfo("image", "gif")),
        (b"GIF89a", MediaInfo("image", "gif")),
        (b"\x1a\x45\xdf\xa3", MediaInfo("video", "webm")),
    )
    for prefix, info in prefixes:
        if data.startswith(prefix):
            return info
    return None


def sniff(data: bytes) -> MediaInfo:
    """Identify an upload from its leading bytes."""
    if len(data) < 12:
        raise UnsupportedMediaError("Upload is empty or too short to identify.")

    direct = _match_prefix(data)
    if direct is not None:
        return direct

    # RIFF containers carry their real type at offset 8.
    if data.startswith(b"RIFF"):
        tag = data[8:12]
        if tag == b"WEBP":
            return MediaInfo("image", "webp")
        if tag == b"AVI ":
            return MediaInfo("video", "avi")

    # ISO base media (mp4/mov) puts "ftyp" at offset 4.
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        return MediaInfo("video", "mov" if brand in (b"qt  ",) else "mp4")

    raise UnsupportedMediaError(
        f"Unrecognised file type. Supported: {', '.join(sorted(IMAGE_FORMATS | VIDEO_FORMATS))}."
    )


def load_image(data: bytes, *, max_pixels: int) -> np.ndarray:
    """Decode an image upload to RGB uint8, guarding against bombs."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > max_pixels:
                raise MediaTooLargeError(
                    f"Image is {width}x{height} ({width * height:,} pixels); "
                    f"the limit is {max_pixels:,} pixels."
                )
            return np.asarray(probe.convert("RGB"), dtype=np.uint8)
    except MediaTooLargeError:
        raise
    except Exception as exc:
        raise CorruptMediaError("Image could not be decoded; it may be truncated.") from exc


def encode_image(array: np.ndarray, fmt: str) -> bytes:
    """Encode an RGB, RGBA, or 2-D grayscale array."""
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    elif array.shape[2] == 4:
        image = Image.fromarray(array, mode="RGBA")
    else:
        image = Image.fromarray(array, mode="RGB")

    buffer = io.BytesIO()
    options = {"optimize": True} if fmt.lower() == "png" else {"quality": 95}
    image.save(buffer, format=fmt.upper(), **options)
    return buffer.getvalue()
