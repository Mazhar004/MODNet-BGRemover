"""Flask application factory.

Importing and calling create_app() must stay free of model loading:
gunicorn imports this at boot, and the test suite runs without weights.
The first request that needs a model pays for it.
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify

from ..config import Settings, load_settings
from ..errors import (
    BGRemoverError,
    CorruptMediaError,
    JobNotFoundError,
    MediaTooLargeError,
    UnsupportedMediaError,
)
from ..jobs import JobRegistry
from ..pipeline import Pipeline

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> Flask:
    settings = settings or load_settings()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=settings.secret_key,
        MAX_CONTENT_LENGTH=settings.max_upload_bytes,
    )

    app.extensions["modnet"] = {
        "settings": settings,
        "pipeline": pipeline or Pipeline(settings),
        "registry": JobRegistry(
            max_workers=settings.max_workers, ttl_seconds=settings.job_ttl_seconds
        ),
    }

    from .routes import api, pages

    app.register_blueprint(pages)
    app.register_blueprint(api, url_prefix="/api")

    _register_error_handlers(app)
    return app


def _register_error_handlers(app: Flask) -> None:
    # Imported here rather than at module scope: routes imports from this
    # module, so a top-level import would be circular.
    from .routes import BadRequest

    status_for: tuple[tuple[type[BGRemoverError], int], ...] = (
        (BadRequest, 400),
        (UnsupportedMediaError, 415),
        (MediaTooLargeError, 413),
        (CorruptMediaError, 422),
        (JobNotFoundError, 404),
    )

    @app.errorhandler(BGRemoverError)
    def _domain_error(exc: BGRemoverError):
        status = next((code for cls, code in status_for if isinstance(exc, cls)), 400)
        return jsonify(error=str(exc)), status

    @app.errorhandler(404)
    def _not_found(_exc):
        return jsonify(error="Not found."), 404

    @app.errorhandler(413)
    def _too_large(_exc):
        limit = app.extensions["modnet"]["settings"].max_upload_bytes // (1024 * 1024)
        return jsonify(error=f"Upload exceeds the {limit}MB limit."), 413

    @app.errorhandler(500)
    def _server_error(exc):
        logger.exception("Unhandled error", exc_info=exc)
        return jsonify(error="Something went wrong on our side."), 500
