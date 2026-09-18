"""Video decoding and encoding.

Two readers, chosen per file:

* **ffmpeg** (``imageio_ffmpeg``) for everything it can demux -- MP4, WebM,
  AVI, MOV, animated GIF and APNG. We call ``read_frames`` directly rather
  than going through ``imageio.v3``, whose FFMPEG plugin refuses ``.gif`` on
  the extension alone even though ffmpeg decodes it happily.
* **Pillow** for animated containers ffmpeg cannot open. The bundled ffmpeg
  build has no libwebp demuxer, so an animated WebP would otherwise lose
  every frame but the first.

No OpenCV anywhere. imageio-ffmpeg ships the static ffmpeg binary that the
audio mux also shells out to, so there is exactly one media dependency.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageSequence

from .errors import CorruptMediaError

logger = logging.getLogger(__name__)

# Alpha needs VP9 in WebM; H.264 has no alpha channel. The mode decides
# the container, so this is not independently selectable by the user.
_ALPHA_MODES = frozenset({"transparent"})

# Used when a container records no frame timing at all.
_FALLBACK_FPS = 10.0


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


def container_for(mode: str) -> tuple[str, bool]:
    """(extension, needs_alpha) for an output mode."""
    if mode in _ALPHA_MODES:
        return ".webm", True
    return ".mp4", False


# --------------------------------------------------------------------------
# Reader selection


def _ffmpeg_meta(path: Path) -> dict | None:
    """ffmpeg's view of a file, or None if it cannot open it."""
    try:
        reader = imageio_ffmpeg.read_frames(str(path))
        try:
            return reader.__next__()
        finally:
            reader.close()
    except Exception as exc:
        logger.debug("ffmpeg cannot read %s: %s", path.name, exc)
        return None


def _pillow_animation(path: Path) -> Image.Image | None:
    """An open multi-frame Pillow image, or None. Caller closes it."""
    try:
        image = Image.open(path)
    except Exception:
        return None
    if int(getattr(image, "n_frames", 1)) <= 1:
        image.close()
        return None
    return image


def _pillow_fps(image: Image.Image) -> float:
    """Average frame rate from per-frame durations, which GIF/WebP vary."""
    durations = []
    for frame in ImageSequence.Iterator(image):
        durations.append(float(frame.info.get("duration") or 0.0))
    usable = [d for d in durations if d > 0]
    if not usable:
        return _FALLBACK_FPS
    return 1000.0 / (sum(usable) / len(usable))


def probe(path: Path) -> VideoInfo:
    path = Path(path)

    meta = _ffmpeg_meta(path)
    if meta is not None:
        width, height = meta["size"]
        fps = float(meta.get("fps") or 0.0)
        duration = float(meta.get("duration") or 0.0)
        if width and height and fps > 0:
            frame_count = int(round(duration * fps)) if duration else 0
            return VideoInfo(width, height, fps, frame_count, duration)

    animation = _pillow_animation(path)
    if animation is not None:
        try:
            width, height = animation.size
            frames = int(getattr(animation, "n_frames", 1))
            fps = _pillow_fps(animation)
            return VideoInfo(width, height, fps, frames, frames / fps if fps else 0.0)
        finally:
            animation.close()

    raise CorruptMediaError("Video could not be read; it may be corrupt or unsupported.")


def iter_frames(path: Path) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames one at a time; never materialise the whole video."""
    path = Path(path)

    if _ffmpeg_meta(path) is not None:
        yield from _iter_ffmpeg(path)
        return

    animation = _pillow_animation(path)
    if animation is None:
        raise CorruptMediaError("Video decoding failed; the file may be corrupt.")
    try:
        for frame in ImageSequence.Iterator(animation):
            yield np.asarray(frame.convert("RGB"), dtype=np.uint8)
    finally:
        animation.close()


def _iter_ffmpeg(path: Path) -> Iterator[np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = reader.__next__()
        width, height = meta["size"]
        for raw in reader:
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    except CorruptMediaError:
        raise
    except Exception as exc:
        raise CorruptMediaError("Video decoding failed partway through.") from exc
    finally:
        reader.close()


# --------------------------------------------------------------------------
# Writing


def write_video(
    frames: Iterable[np.ndarray],
    dest: Path,
    *,
    fps: float,
    size: tuple[int, int],
    alpha: bool,
) -> None:
    """Encode frames. size is (width, height)."""
    if alpha:
        kwargs = dict(
            pix_fmt_in="rgba",
            codec="libvpx-vp9",
            output_params=["-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-b:v", "2M"],
        )
    else:
        kwargs = dict(
            pix_fmt_in="rgb24",
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20"],
        )

    writer = imageio_ffmpeg.write_frames(str(dest), size=size, fps=float(fps), **kwargs)
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame).tobytes())
    finally:
        writer.close()


def mux_audio(video_path: Path, source: Path, dest: Path) -> bool:
    """Copy the source's audio onto video_path, writing dest.

    Returns False and leaves dest untouched when the source has no audio
    track, which is the common case for screen recordings and GIFs. Never
    raises for a missing track.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(dest),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, timeout=600, check=False)  # noqa: S603
    if result.returncode != 0:
        logger.info("No audio track muxed for %s", Path(source).name)
        Path(dest).unlink(missing_ok=True)
        return False
    return True
