"""Turning an image plus its alpha into the output the user asked for."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

OUTPUT_MODES: tuple[str, ...] = ("transparent", "color", "blur", "image", "matte")

DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)
DEFAULT_BLUR_RADIUS = 24


def _as_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def cover_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to fill (h, w), preserving aspect, centre-cropping the overflow.

    Cover rather than stretch: a background photo should not be squashed
    to the subject's aspect ratio.
    """
    target_h, target_w = size
    src_h, src_w = image.shape[:2]
    scale = max(target_h / src_h, target_w / src_w)
    scaled = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))

    resized = np.asarray(Image.fromarray(image).resize(scaled, Image.LANCZOS))
    top = (resized.shape[0] - target_h) // 2
    left = (resized.shape[1] - target_w) // 2
    return resized[top : top + target_h, left : left + target_w]


def _blend(rgb: np.ndarray, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    return _as_uint8(rgb.astype(np.float32) * a + background.astype(np.float32) * (1.0 - a))


def composite(
    rgb: np.ndarray,
    alpha: np.ndarray,
    mode: str,
    *,
    color: tuple[int, int, int] = DEFAULT_COLOR,
    background: np.ndarray | None = None,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
) -> np.ndarray:
    """Apply an output mode. Returns RGBA for transparent, 2-D for matte, else RGB."""
    if mode not in OUTPUT_MODES:
        raise ValueError(f"Unknown output mode {mode!r}. Valid: {list(OUTPUT_MODES)}")

    if mode == "matte":
        return _as_uint8(alpha * 255.0)

    if mode == "transparent":
        # Straight (unpremultiplied) alpha: the colour channels are the
        # original pixels, untouched. Compositing them onto white here is
        # what produced the white halo in the previous implementation.
        out = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
        out[..., :3] = rgb
        out[..., 3] = _as_uint8(alpha * 255.0)
        return out

    if mode == "color":
        plate = np.empty_like(rgb)
        plate[..., :] = np.asarray(color, dtype=np.uint8)
        return _blend(rgb, alpha, plate)

    if mode == "blur":
        blurred = np.asarray(
            Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=blur_radius))
        )
        return _blend(rgb, alpha, blurred)

    # mode == "image"
    if background is None:
        raise ValueError("mode 'image' requires a background array")
    return _blend(rgb, alpha, cover_resize(background, rgb.shape[:2]))
