import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class FakeMatteModel(nn.Module):
    """Stand-in for MODNet with the same call signature and output shape.

    Returns a deterministic matte: 1.0 in the central half of the frame,
    0.0 outside, with a one-cell ramp so tests can exercise partial alpha.
    The whole suite runs on this, which is why no checkpoint is needed.
    """

    def forward(self, img, inference):
        n, _, h, w = img.shape
        matte = torch.zeros((n, 1, h, w), dtype=torch.float32, device=img.device)
        matte[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
        matte[:, :, h // 4, w // 4 : 3 * w // 4] = 0.5  # deliberate soft edge
        return None, None, matte


@pytest.fixture
def fake_engine():
    from modnet_bg.engine import MattingEngine

    return MattingEngine(model=FakeMatteModel(), device="cpu")


@pytest.fixture
def fake_engine_on_real_device():
    """Fake model, but on whatever device this machine actually has.

    On CI that is cpu; on an Apple machine it is mps. Device-specific
    operator gaps only show up through this fixture.
    """
    from modnet_bg.device import resolve_device
    from modnet_bg.engine import MattingEngine

    model = FakeMatteModel()
    device = resolve_device("auto")
    model.to(device)
    return MattingEngine(model=model, device=device)


@pytest.fixture
def sample_rgb():
    """A 128x96 RGB image, solid red, so colour bleed is obvious."""
    image = np.zeros((128, 96, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)
    return image


@pytest.fixture
def tiny_video(tmp_path):
    """A real 10-frame 64x48 MP4, written with the same encoder the app uses."""
    import imageio_ffmpeg

    dest = tmp_path / "tiny.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(dest),
        size=(64, 48),
        fps=10.0,
        pix_fmt_in="rgb24",
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    writer.send(None)
    for i in range(10):
        frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
        writer.send(frame.tobytes())
    writer.close()
    return dest


@pytest.fixture
def animated_webp(tmp_path):
    """Animated WebP: ffmpeg's bundled build cannot demux it, Pillow can."""
    from PIL import Image

    dest = tmp_path / "anim.webp"
    frames = [Image.new("RGB", (32, 24), (i * 60 % 256, 0, 0)) for i in range(5)]
    frames[0].save(
        dest, format="WEBP", save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    return dest


@pytest.fixture
def engine_factory():
    """An engine_factory that never touches a checkpoint.

    Exposed as a fixture rather than imported directly: `tests/` is not a
    package (no __init__.py, pytest prepend import mode), so
    `from tests.conftest import ...` would fail.
    """
    from modnet_bg.engine import MattingEngine

    return lambda path, device: MattingEngine(FakeMatteModel(), device)


@pytest.fixture
def fake_pipeline(tmp_path, engine_factory, monkeypatch):
    """A Pipeline whose engine never touches a checkpoint."""
    from modnet_bg import pipeline as pipeline_module
    from modnet_bg.config import load_settings
    from modnet_bg.pipeline import Pipeline

    # resolve_weights would demand a real checkpoint; the fake engine ignores it.
    monkeypatch.setattr(
        pipeline_module.weights, "resolve_weights", lambda key, **kw: tmp_path / f"{key}.ckpt"
    )
    settings = load_settings(
        {"MODNET_DATA_DIR": str(tmp_path / "data"), "MODNET_DEVICE": "cpu"}
    )
    return Pipeline(settings, engine_factory=engine_factory)
