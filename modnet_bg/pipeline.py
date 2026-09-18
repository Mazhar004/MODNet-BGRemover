"""Wiring: decode, matte, composite, encode.

Everything above this layer deals in jobs and bytes; everything below
deals in arrays. This is the only module that knows about all of them.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import media, video, weights
from .compositing import DEFAULT_BLUR_RADIUS, DEFAULT_COLOR, composite
from .config import Settings
from .device import resolve_device
from .engine import MattingEngine, get_engine
from .errors import MediaTooLargeError
from .jobs import Job

logger = logging.getLogger(__name__)

EngineFactory = Callable[[Path, str], MattingEngine]


@dataclass(frozen=True)
class ProcessOptions:
    mode: str = "transparent"
    color: tuple[int, int, int] = DEFAULT_COLOR
    background: np.ndarray | None = None
    blur_radius: int = DEFAULT_BLUR_RADIUS
    device: str = "auto"


class Pipeline:
    def __init__(self, settings: Settings, engine_factory: EngineFactory | None = None) -> None:
        self._settings = settings
        self._engine_factory = engine_factory or get_engine
        self._results = Path(settings.data_dir) / "results"
        self._results.mkdir(parents=True, exist_ok=True)

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def results_dir(self) -> Path:
        return self._results

    def resolve_engine(self, device_pref: str, key: str = "photographic") -> MattingEngine:
        device = resolve_device(device_pref if device_pref != "auto" else self._settings.device)
        search: list[Path] = []
        if self._settings.weights_dir:
            search.append(self._settings.weights_dir)
        search.append(Path.cwd() / "weights")

        path = weights.resolve_weights(
            key,
            search_dirs=search,
            download_url=self._settings.weights_urls.get(key),
            cache_dir=Path(self._settings.data_dir) / "weights",
        )
        return self._engine_factory(path, device)

    def _output(self, suffix: str) -> Path:
        # Server-generated name. No user-supplied string reaches a path.
        return self._results / f"{secrets.token_urlsafe(16)}{suffix}"

    def _composite(self, frame: np.ndarray, alpha: np.ndarray, options: ProcessOptions):
        return composite(
            frame,
            alpha,
            options.mode,
            color=options.color,
            background=options.background,
            blur_radius=options.blur_radius,
        )

    def image_job(
        self, data: bytes, options: ProcessOptions, stem: str
    ) -> Callable[[Job], tuple[Path, str]]:
        def run(job: Job) -> tuple[Path, str]:
            # Five real steps rather than one. A single step meant the bar read
            # 0% for the whole job and then jumped to done, which looks hung on
            # a large photo where each step takes seconds.
            job.set_total(5)
            job.raise_if_cancelled()

            job.set_stage("Loading model")
            engine = self.resolve_engine(options.device)
            job.device_used = engine.device
            job.advance()

            job.set_stage("Reading image")
            rgb = media.load_image(data, max_pixels=self._settings.max_image_pixels)
            job.advance()
            job.raise_if_cancelled()

            job.set_stage(f"Removing background on {engine.device}")
            alpha = engine.matte(rgb)
            job.advance()

            job.set_stage("Applying background")
            out = self._composite(rgb, alpha, options)
            job.advance()

            job.set_stage("Encoding PNG")
            path = self._output(".png")
            path.write_bytes(media.encode_image(out, "png"))
            job.advance()

            job.set_stage("Done")
            return path, f"{stem}.png"

        return run

    def video_job(
        self, source: Path, options: ProcessOptions, stem: str
    ) -> Callable[[Job], tuple[Path, str]]:
        def run(job: Job) -> tuple[Path, str]:
            info = video.probe(source)
            limit = self._settings.max_video_seconds
            if info.duration > limit:
                raise MediaTooLargeError(
                    f"Video is {info.duration:.0f}s, longer than the {limit}s limit."
                )

            job.set_total(info.frame_count or None)
            job.set_stage("Loading model")
            job.raise_if_cancelled()

            engine = self.resolve_engine(options.device)
            job.device_used = engine.device

            suffix, wants_alpha = video.container_for(options.mode)
            path = self._output(suffix)

            job.set_stage(f"Processing {info.frame_count} frames on {engine.device}")

            def processed():
                for frame in video.iter_frames(source):
                    job.raise_if_cancelled()
                    alpha = engine.matte(frame)
                    out = self._composite(frame, alpha, options)
                    if out.ndim == 2:  # matte mode: grayscale to 3-channel
                        out = np.repeat(out[..., None], 3, axis=2)
                    job.advance()
                    yield out

            video.write_video(
                processed(),
                path,
                fps=info.fps,
                size=(info.width, info.height),
                alpha=wants_alpha,
            )

            if not wants_alpha:
                job.set_stage("Adding audio")
                muxed = path.with_name(f"{path.stem}-audio{suffix}")
                if video.mux_audio(path, source, muxed):
                    path.unlink(missing_ok=True)
                    path = muxed

            job.set_stage("Done")
            return path, f"{stem}{suffix}"

        return run
