import io

import numpy as np
import pytest
from PIL import Image

from modnet_bg.web.realtime import RealtimeOptions, process_frame


def _jpeg(size=(320, 240)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_returns_a_grayscale_png_mask(fake_pipeline):
    payload = process_frame(fake_pipeline, _jpeg(), RealtimeOptions())

    with Image.open(io.BytesIO(payload)) as mask:
        assert mask.format == "PNG"
        assert mask.mode == "L"


def test_mask_matches_the_downscaled_frame_size(fake_pipeline):
    payload = process_frame(fake_pipeline, _jpeg((1280, 720)), RealtimeOptions(max_edge=512))

    with Image.open(io.BytesIO(payload)) as mask:
        assert max(mask.size) <= 512
        assert mask.size[0] / mask.size[1] == pytest.approx(1280 / 720, rel=0.02)


def test_small_frames_are_not_upscaled(fake_pipeline):
    payload = process_frame(fake_pipeline, _jpeg((160, 120)), RealtimeOptions(max_edge=512))

    with Image.open(io.BytesIO(payload)) as mask:
        assert mask.size == (160, 120)


def test_mask_has_real_variation(fake_pipeline):
    """The fake model produces a central box; a flat mask means a wiring bug."""
    payload = process_frame(fake_pipeline, _jpeg(), RealtimeOptions())

    with Image.open(io.BytesIO(payload)) as mask:
        values = np.asarray(mask)
    assert values.min() == 0 and values.max() == 255


def test_oversized_frame_is_refused(fake_pipeline):
    from modnet_bg.errors import BGRemoverError

    with pytest.raises(BGRemoverError, match="too large"):
        process_frame(fake_pipeline, b"\x00" * (5 * 1024 * 1024), RealtimeOptions())


def test_garbage_frame_raises_a_domain_error(fake_pipeline):
    from modnet_bg.errors import BGRemoverError

    with pytest.raises(BGRemoverError):
        process_frame(fake_pipeline, b"not a jpeg at all", RealtimeOptions())


def test_socket_route_is_registered(fake_pipeline):
    from modnet_bg.web import create_app

    app = create_app(settings=fake_pipeline.settings, pipeline=fake_pipeline)
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/ws/realtime" in rules
