"""HTTP endpoints."""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from .. import __version__
from ..compositing import OUTPUT_MODES
from ..device import available_devices, resolve_device
from ..errors import BGRemoverError, JobNotFoundError, MediaTooLargeError
from ..media import IMAGE_FORMATS, VIDEO_FORMATS, load_image, sniff
from ..pipeline import ProcessOptions

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


_HEX_COLOR = re.compile(r"^#(?P<r>[0-9a-fA-F]{2})(?P<g>[0-9a-fA-F]{2})(?P<b>[0-9a-fA-F]{2})$")


def _parse_color(raw: str) -> tuple[int, int, int]:
    match = _HEX_COLOR.match(raw.strip())
    if not match:
        raise BadRequest(f"color must be #rrggbb, got {raw!r}")
    return tuple(int(match.group(c), 16) for c in ("r", "g", "b"))  # type: ignore[return-value]


def _display_stem(filename: str) -> str:
    """A safe stem for the download name. Never used as a path."""
    cleaned = secure_filename(filename or "output")
    return Path(cleaned).stem or "output"


def _read_options() -> ProcessOptions:
    form = request.form
    mode = form.get("mode", "transparent")
    if mode not in OUTPUT_MODES:
        raise BadRequest(f"Unknown mode {mode!r}. Valid: {list(OUTPUT_MODES)}")

    try:
        blur = int(form.get("blur_radius", 24))
    except ValueError as exc:
        raise BadRequest("blur_radius must be an integer") from exc
    if not 1 <= blur <= 100:
        raise BadRequest("blur_radius must be between 1 and 100")

    background = None
    if mode == "image":
        upload = request.files.get("background")
        if upload is None:
            raise BadRequest("mode 'image' requires a background file")
        settings = _state()["settings"]
        background = load_image(upload.read(), max_pixels=settings.max_image_pixels)

    return ProcessOptions(
        mode=mode,
        color=_parse_color(form.get("color", "#ffffff")),
        background=background,
        blur_radius=blur,
        device=form.get("device", "auto"),
    )


@api.route("/jobs", methods=["POST"])
def create_job():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise BadRequest("No file was uploaded under the field name 'file'.")

    options = _read_options()
    data = upload.read()
    info = sniff(data)

    state = _state()
    settings, pipeline, registry = state["settings"], state["pipeline"], state["registry"]
    stem = _display_stem(upload.filename)

    if info.kind == "image":
        if len(data) > settings.max_image_bytes:
            raise MediaTooLargeError(f"Image exceeds the {settings.max_image_mb}MB limit.")
        runner = pipeline.image_job(data, options, stem)
    else:
        if len(data) > settings.max_video_bytes:
            raise MediaTooLargeError(f"Video exceeds the {settings.max_video_mb}MB limit.")
        scratch_dir = Path(settings.data_dir) / "uploads"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(dir=scratch_dir)) / f"source.{info.format}"
        scratch.write_bytes(data)
        runner = pipeline.video_job(scratch, options, stem)

    registry.reap()
    job = registry.submit(runner, kind=info.kind, filename=upload.filename)
    return jsonify(job.to_dict()), 202


@api.route("/jobs/<job_id>")
def job_status(job_id: str):
    return jsonify(_state()["registry"].get(job_id).to_dict())


@api.route("/jobs/<job_id>", methods=["DELETE"])
def job_cancel(job_id: str):
    return jsonify(_state()["registry"].cancel(job_id).to_dict())


@api.route("/jobs/<job_id>/result")
def job_result(job_id: str):
    job = _state()["registry"].get(job_id)
    if job.status != "done" or job.result_path is None:
        raise JobNotFoundError("Result is not ready.")
    return send_file(job.result_path, as_attachment=True, download_name=job.result_name)


@api.route("/jobs/zip", methods=["POST"])
def jobs_zip():
    ids = (request.get_json(silent=True) or {}).get("ids") or []
    if not ids:
        raise BadRequest("Provide a non-empty 'ids' list.")

    registry = _state()["registry"]
    buffer = io.BytesIO()
    written = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, job_id in enumerate(ids):
            job = registry.get(job_id)
            if job.status == "done" and job.result_path:
                archive.write(job.result_path, arcname=f"{index + 1:02d}-{job.result_name}")
                written += 1

    if not written:
        raise BadRequest("None of those jobs have finished.")

    buffer.seek(0)
    return send_file(
        buffer, mimetype="application/zip", as_attachment=True, download_name="backgrounds.zip"
    )
