"""Runtime settings, read once from the environment."""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .weights import WEIGHTS

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_MAX_WORKERS_CEILING = 16


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", key, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    device: str = "auto"
    weights_dir: Path | None = None
    weights_urls: dict[str, str] = field(default_factory=dict)
    port: int = 8000
    secret_key: str = ""
    secret_key_generated: bool = False
    max_image_mb: int = 25
    max_video_mb: int = 250
    max_video_seconds: int = 120
    max_workers: int = 2
    job_ttl_seconds: int = 3600
    max_image_pixels: int = 50_000_000
    data_dir: Path = Path("/tmp/modnet-bg")  # noqa: S108

    @property
    def max_image_bytes(self) -> int:
        return self.max_image_mb * _MB

    @property
    def max_video_bytes(self) -> int:
        return self.max_video_mb * _MB

    @property
    def max_upload_bytes(self) -> int:
        """Flask's MAX_CONTENT_LENGTH. Per-kind caps are enforced after sniffing."""
        return max(self.max_image_bytes, self.max_video_bytes)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env

    explicit_secret = env.get("SECRET_KEY")
    if not explicit_secret:
        logger.warning(
            "SECRET_KEY is unset; generating a random one. "
            "Sessions will not survive a restart. Set it explicitly in production."
        )

    weights_dir = env.get("MODNET_WEIGHTS_DIR")
    urls = {
        key: env[f"MODNET_WEIGHTS_URL_{key.upper()}"]
        for key in WEIGHTS
        if env.get(f"MODNET_WEIGHTS_URL_{key.upper()}")
    }

    workers = _int(env, "MODNET_MAX_WORKERS", 2)
    workers = max(1, min(workers, _MAX_WORKERS_CEILING))

    return Settings(
        device=env.get("MODNET_DEVICE", "auto"),
        weights_dir=Path(weights_dir) if weights_dir else None,
        weights_urls=urls,
        port=_int(env, "PORT", 8000),
        secret_key=explicit_secret or secrets.token_urlsafe(32),
        secret_key_generated=not explicit_secret,
        max_image_mb=_int(env, "MODNET_MAX_IMAGE_MB", 25),
        max_video_mb=_int(env, "MODNET_MAX_VIDEO_MB", 250),
        max_video_seconds=_int(env, "MODNET_MAX_VIDEO_SECONDS", 120),
        max_workers=workers,
        job_ttl_seconds=_int(env, "MODNET_JOB_TTL_SECONDS", 3600),
        max_image_pixels=_int(env, "MODNET_MAX_IMAGE_PIXELS", 50_000_000),
        data_dir=Path(env.get("MODNET_DATA_DIR", "/tmp/modnet-bg")),  # noqa: S108
    )
