"""HTTP endpoints."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template

from .. import __version__
from ..compositing import OUTPUT_MODES
from ..device import available_devices, resolve_device
from ..errors import BGRemoverError
from ..media import IMAGE_FORMATS, VIDEO_FORMATS

pages = Blueprint("pages", __name__)
api = Blueprint("api", __name__)


class BadRequest(BGRemoverError):
    """Malformed request parameters. -> HTTP 400"""


def _state():
    return current_app.extensions["modnet"]


@pages.route("/")
def index():
    return render_template("index.html")


@pages.route("/health")
def health():
    """Liveness only. Must never touch the model."""
    return jsonify(status="ok", version=__version__)


@api.route("/capabilities")
def capabilities():
    settings = _state()["settings"]
    return jsonify(
        devices=available_devices(),
        default_device=resolve_device(settings.device),
        configured_device=settings.device,
        modes=list(OUTPUT_MODES),
        formats={"image": sorted(IMAGE_FORMATS), "video": sorted(VIDEO_FORMATS)},
        limits={
            "max_image_mb": settings.max_image_mb,
            "max_video_mb": settings.max_video_mb,
            "max_video_seconds": settings.max_video_seconds,
        },
    )
