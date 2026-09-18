"""Stateless matting.

An engine holds a model and a device and nothing else. Every per-image
value is a local or a return value, which is what makes it safe to share
one engine across the worker pool. The previous implementation kept
image dimensions and alpha on the instance; two concurrent uploads
overwrote each other.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models.modnet import MODNet

logger = logging.getLogger(__name__)

REF_SIZE = 512
STRIDE = 32

_MEAN = 0.5
_STD = 0.5

_engine_cache: dict[tuple[str, str], MattingEngine] = {}
_cache_lock = threading.Lock()


def target_size(h: int, w: int) -> tuple[int, int]:
    """Inference dimensions: short side near REF_SIZE, both stride-aligned.

    The clamp to STRIDE is load-bearing. A 600x16 strip lands in the
    pass-through branch, and 16 - (16 % 32) is 0, which crashes the
    interpolate. The old code had no clamp.
    """
    if max(h, w) < REF_SIZE or min(h, w) > REF_SIZE:
        if w >= h:
            rh, rw = REF_SIZE, int(w / h * REF_SIZE)
        else:
            rh, rw = int(h / w * REF_SIZE), REF_SIZE
    else:
        rh, rw = h, w

    rh = max(STRIDE, rh - rh % STRIDE)
    rw = max(STRIDE, rw - rw % STRIDE)
    return rh, rw


class MattingEngine:
    def __init__(self, model, device: str) -> None:
        self._model = model
        self._device = device
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def from_checkpoint(cls, weights_path: Path, device: str) -> MattingEngine:
        model = MODNet(backbone_pretrained=False)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Upstream checkpoints were saved from a DataParallel wrapper, so every
        # key carries a "module." prefix. We do not use DataParallel.
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        model.to(device)
        return cls(model=model, device=device)

    def matte(self, rgb: np.ndarray) -> np.ndarray:
        """Alpha for one RGB uint8 image. float32 [H, W] in 0..1."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) array, got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"expected uint8 input, got {rgb.dtype}")

        h, w = rgb.shape[:2]
        th, tw = target_size(h, w)

        # Convert to float32 in numpy first. Arrays that came from Pillow are
        # read-only, and torch.from_numpy warns about undefined behaviour on
        # those; the dtype change produces a fresh writable buffer. We needed
        # the float conversion anyway, so this costs nothing extra.
        array = np.ascontiguousarray(rgb, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor / 255.0 - _MEAN) / _STD
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=(th, tw), mode="area")

        alpha = self._forward(tensor)

        # Upscale on the CPU, always. mode="area" lowers to adaptive_avg_pool2d,
        # which on MPS requires the input size to be divisible by the output
        # size (pytorch#96056); arbitrary source dimensions almost never are.
        # Moving the small matte first is also cheaper than moving the
        # full-resolution one back.
        alpha = alpha.to("cpu")
        alpha = F.interpolate(alpha, size=(h, w), mode="area")
        return alpha[0, 0].clamp(0.0, 1.0).float().numpy()

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            try:
                # inference=True skips the semantic and detail heads. The matte
                # is computed from lr8x/hr2x and is unaffected.
                _, _, alpha = self._model(tensor.to(self._device), True)
                return alpha
            except (NotImplementedError, RuntimeError) as exc:
                if self._device != "mps":
                    raise
                logger.warning("MPS forward failed (%s); falling back to CPU", exc)
                with self._lock:
                    self._model.to("cpu")
                    self._device = "cpu"
                _, _, alpha = self._model(tensor.to("cpu"), True)
                return alpha


def get_engine(weights_path: Path, device: str) -> MattingEngine:
    """One engine per (checkpoint, device). Built once, shared thereafter."""
    key = (str(weights_path), device)
    engine = _engine_cache.get(key)
    if engine is not None:
        return engine

    with _cache_lock:
        engine = _engine_cache.get(key)
        if engine is None:
            logger.info("Loading %s onto %s", Path(weights_path).name, device)
            engine = MattingEngine.from_checkpoint(Path(weights_path), device)
            _engine_cache[key] = engine
        return engine
