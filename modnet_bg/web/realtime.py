"""Webcam matting over a WebSocket.

Throughput is bounded by the model, not this code: expect roughly 2-5
fps on a CPU and 15-30 fps on CUDA. The client measures and displays the
real rate rather than implying it should be smooth.

Only the alpha mask travels back -- one channel instead of four. The
client already holds the source frame on a canvas, so compositing there
also means changing the background costs no round trip.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
from flask import Flask
from flask_sock import Sock
from PIL import Image

from ..errors import BGRemoverError
from ..media import encode_image, load_image
from ..pipeline import Pipeline

logger = logging.getLogger(__name__)

sock = Sock()

DEFAULT_MAX_EDGE = 512
_MAX_FRAME_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class RealtimeOptions:
    device: str = "auto"
    max_edge: int = DEFAULT_MAX_EDGE


def _downscale(rgb: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return rgb
    scale = max_edge / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR))


def process_frame(pipeline: Pipeline, data: bytes, options: RealtimeOptions) -> bytes:
    """One webcam frame in, one PNG-encoded grayscale alpha mask out."""
    if len(data) > _MAX_FRAME_BYTES:
        raise BGRemoverError("Frame too large.")

    rgb = _downscale(load_image(data, max_pixels=16_000_000), options.max_edge)
    engine = pipeline.resolve_engine(options.device, key="webcam")
    alpha = engine.matte(rgb)
    return encode_image(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), "png")


def register_realtime(app: Flask) -> None:
    sock.init_app(app)

    @sock.route("/ws/realtime", app)
    def realtime(ws):  # pragma: no cover - exercised manually with a real camera
        options = RealtimeOptions()
        state = app.extensions["modnet"]

        while True:
            message = ws.receive()
            if message is None:
                return

            if isinstance(message, str):
                try:
                    payload = json.loads(message)
                    options = RealtimeOptions(
                        device=payload.get("device", "auto"),
                        max_edge=int(payload.get("max_edge", DEFAULT_MAX_EDGE)),
                    )
                    ws.send(json.dumps({"ready": True}))
                except (ValueError, TypeError):
                    ws.send(json.dumps({"error": "Malformed options message."}))
                continue

            try:
                ws.send(process_frame(state["pipeline"], message, options))
            except BGRemoverError as exc:
                ws.send(json.dumps({"error": str(exc)}))
            except Exception:
                logger.exception("Realtime frame failed")
                ws.send(json.dumps({"error": "Frame could not be processed."}))
