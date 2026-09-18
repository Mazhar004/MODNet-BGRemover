import io

import numpy as np
import pytest
from PIL import Image

from modnet_bg import media
from modnet_bg.errors import CorruptMediaError, MediaTooLargeError, UnsupportedMediaError


def _encode(fmt: str, size=(20, 10), mode="RGB") -> bytes:
    # Fill has to match the mode: "L" takes an int, "RGBA" a 4-tuple.
    fill = {"L": 255, "RGBA": (255, 0, 0, 255)}.get(mode, (255, 0, 0))
    buffer = io.BytesIO()
    Image.new(mode, size, fill).save(buffer, format=fmt)
    return buffer.getvalue()


def test_sniffs_image_formats_by_magic_bytes():
    assert media.sniff(_encode("PNG")).kind == "image"
    assert media.sniff(_encode("PNG")).format == "png"
    assert media.sniff(_encode("JPEG")).format == "jpeg"
    assert media.sniff(_encode("WEBP")).format == "webp"


def test_sniffs_video_formats():
    mp4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32
    assert media.sniff(mp4) == media.MediaInfo(kind="video", format="mp4")

    webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 32
    assert media.sniff(webm).kind == "video"


def test_extension_lies_are_irrelevant():
    """A .png that is really a JPEG is still handled as a JPEG."""
    assert media.sniff(_encode("JPEG")).format == "jpeg"


def test_unknown_bytes_rejected():
    with pytest.raises(UnsupportedMediaError):
        media.sniff(b"MZ\x90\x00this is a windows executable")


def test_empty_upload_rejected():
    with pytest.raises(UnsupportedMediaError):
        media.sniff(b"")


def test_load_image_returns_rgb_uint8():
    array = media.load_image(_encode("PNG", size=(20, 10)), max_pixels=10_000)
    assert array.shape == (10, 20, 3)
    assert array.dtype == np.uint8
    np.testing.assert_array_equal(array[0, 0], [255, 0, 0])


def test_load_image_flattens_rgba_and_grayscale():
    assert media.load_image(_encode("PNG", mode="RGBA"), max_pixels=10_000).shape[2] == 3
    assert media.load_image(_encode("PNG", mode="L"), max_pixels=10_000).shape[2] == 3


def test_pixel_bomb_rejected():
    with pytest.raises(MediaTooLargeError, match="pixels"):
        media.load_image(_encode("PNG", size=(200, 200)), max_pixels=1_000)


def test_truncated_image_rejected():
    payload = _encode("PNG")[:40]
    with pytest.raises(CorruptMediaError):
        media.load_image(payload, max_pixels=10_000)


def test_encode_roundtrip_preserves_alpha():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 128
    decoded = Image.open(io.BytesIO(media.encode_image(rgba, "png")))
    assert decoded.mode == "RGBA"
    assert decoded.getpixel((0, 0))[3] == 128


def test_gif_is_accepted_as_an_image():
    assert media.sniff(_encode("GIF", mode="P")) == media.MediaInfo(kind="image", format="gif")


def test_animated_gif_decodes_to_its_first_frame():
    """Motion is dropped deliberately; uploading a video is the way to keep it."""
    buffer = io.BytesIO()
    frames = [Image.new("P", (8, 8), 0), Image.new("P", (8, 8), 1)]
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])

    array = media.load_image(buffer.getvalue(), max_pixels=10_000)
    assert array.shape == (8, 8, 3)


def test_real_project_media_is_typed_by_content_not_extension():
    """output/male.png is really a JPEG. The extension must not be believed."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    mislabelled = root / "output" / "male.png"
    if not mislabelled.exists():
        pytest.skip("demo asset not present")
    assert media.sniff(mislabelled.read_bytes()[:64]).format == "jpeg"
