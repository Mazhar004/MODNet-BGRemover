"""Locating and verifying MODNet checkpoints.

Local-first by design. Upstream distributes these weights via Google
Drive, which serves an HTML interstitial rather than the file, and the
obvious mirrors do not resolve. So we search directories the operator
controls and download only when handed an explicit URL. Nothing is ever
used without matching its pinned SHA-256.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .errors import WeightsError

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024
_MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class WeightSpec:
    key: str
    filename: str
    sha256: str


WEIGHTS: dict[str, WeightSpec] = {
    "photographic": WeightSpec(
        key="photographic",
        filename="modnet_photographic_portrait_matting.ckpt",
        sha256="7c22235f0925deba15d4d63e53afcb654c47055bbcd98f56e393ab2584007ed8",
    ),
    "webcam": WeightSpec(
        key="webcam",
        filename="modnet_webcam_portrait_matting.ckpt",
        sha256="913b82b66558db39b6286c150f809017d7528c872b156eb14333c9c6cb52108b",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, spec: WeightSpec) -> None:
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise WeightsError(
            f"Checkpoint {path} failed its checksum. "
            f"Expected {spec.sha256}, got {actual}. Refusing to load it."
        )


def _http_get(url: str, dest: Path) -> None:
    """Stream a URL to dest. Separated out so tests can replace it."""
    # The URL comes from MODNET_WEIGHTS_URL_*, which an operator sets. Restrict
    # it to http(s): urllib would otherwise happily honour file:// and copy an
    # arbitrary local path into the weights cache.
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise WeightsError(f"Refusing to fetch weights over {scheme or 'an unknown'} scheme.")

    request = urllib.request.Request(url, headers={"User-Agent": "modnet-bgremover"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > _MAX_DOWNLOAD_BYTES:
            raise WeightsError(f"Refusing to download {declared} bytes from {url}")
        written = 0
        with dest.open("wb") as handle:
            while chunk := response.read(_CHUNK):
                written += len(chunk)
                if written > _MAX_DOWNLOAD_BYTES:
                    raise WeightsError(f"Download from {url} exceeded the size cap")
                handle.write(chunk)


def _download(spec: WeightSpec, url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / spec.filename
    partial = cache_dir / f"{spec.filename}.part"

    try:
        logger.info("Downloading %s", spec.filename)
        _http_get(url, partial)
        _verify(partial, spec)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    partial.replace(final)
    return final


def resolve_weights(
    key: str,
    *,
    search_dirs: list[Path],
    download_url: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Return a verified checkpoint path, downloading only if given a URL."""
    spec = WEIGHTS.get(key)
    if spec is None:
        raise WeightsError(f"Unknown checkpoint {key!r}. Known: {sorted(WEIGHTS)}")

    for directory in search_dirs:
        candidate = Path(directory) / spec.filename
        if candidate.is_file():
            _verify(candidate, spec)
            return candidate

    if download_url:
        target = cache_dir or Path(search_dirs[0])
        return _download(spec, download_url, Path(target))

    searched = ", ".join(str(Path(d)) for d in search_dirs) or "(no directories configured)"
    raise WeightsError(
        f"Checkpoint {spec.filename} not found. Searched: {searched}. "
        f"Place the file in one of those directories, point MODNET_WEIGHTS_DIR at it, "
        f"or set MODNET_WEIGHTS_URL_{key.upper()} to a direct download URL."
    )


def default_search_dirs() -> list[Path]:
    """Directories searched when the caller does not specify."""
    dirs: list[Path] = []
    configured = os.environ.get("MODNET_WEIGHTS_DIR")
    if configured:
        dirs.append(Path(configured))
    dirs.append(Path.cwd() / "weights")
    return dirs
