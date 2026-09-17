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
