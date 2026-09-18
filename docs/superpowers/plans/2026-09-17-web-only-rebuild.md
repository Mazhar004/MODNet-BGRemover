# MODNet-BGRemover Web-Only Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CLI and demo Flask app with a single tested, containerized web application that removes backgrounds from photos, videos, and a live webcam, with the GPU usable and switchable off.

**Architecture:** A new `modnet_bg/` package holds a stateless matting engine, output compositing, media validation, video I/O, and an in-process job queue with progress reporting. A Flask application factory exposes a JSON API plus one no-build vanilla-JS page. The vendored MODNet network moves in unchanged; every other existing module is deleted.

**Tech Stack:** Python 3.12+, PyTorch 2.14, Flask 3.1, flask-sock (WebSocket), Pillow 12, NumPy 2, imageio + imageio-ffmpeg (video), gunicorn, pytest, ruff, Docker.

**Spec:** `docs/superpowers/specs/2026-09-17-web-bgremover-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python floor:** `>=3.12`. Local development is 3.13.9.
- **Pinned dependency versions** (verified against PyPI on 2026-09-17; pin with `==`):
  - `torch==2.14.0`, `torchvision==0.29.0`
  - `numpy==2.5.3`, `pillow==12.3.0`
  - `imageio-ffmpeg==0.6.0` (plain `imageio` was dropped during Task 8: its v3
    FFMPEG plugin refuses `.gif` on the extension alone, so `video.py` calls
    `imageio_ffmpeg.read_frames` directly and nothing else needed it)
  - `flask==3.1.3`, `werkzeug==3.1.8`, `flask-sock==0.7.0`, `gunicorn==26.2.0`
  - Dev: `pytest==9.1.1`, `pytest-cov==7.1.0`, `ruff==0.16.8`, `pip-audit==2.10.1`
- **No OpenCV.** The rebuild drops `opencv-python` entirely. Rationale: OpenCV has moved to 5.0.0.93 with breaking API changes (`cv2.VideoWriter_fourcc` relocated), it is a ~60MB wheel, and it is a recurring CVE source. Pillow covers resize/blur/encode and imageio-ffmpeg covers video I/O. The vendored `models/` code imports only `torch`, so nothing in the network depends on cv2. **This is a deliberate deviation from spec section 5.6, which named `cv2.VideoCapture`.**
- **No `flask-cors`.** The app is same-origin. Do not reintroduce it.
- **Every `torch.load` call must pass `weights_only=True`.** No exceptions.
- **The full test suite must pass with no model checkpoint present and no GPU.** Any test that needs real weights must `skip` when they are absent, and must never be required for CI to go green.
- **`create_app()` must import and construct without loading a model.** Gunicorn and the test suite both depend on this.
- **Colour channel order is RGB everywhere** inside the package. The old code mixed BGR and RGB; do not carry that in.
- **Alpha is `float32` in `0.0..1.0`** at every internal boundary. Convert to `uint8` only when encoding an output file.
- Commit after every task. Conventional-commit prefixes (`feat:`, `test:`, `chore:`, `docs:`).

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Packaging, pinned deps, pytest + ruff config |
| `modnet_bg/__init__.py` | Version constant only; no imports with side effects |
| `modnet_bg/models/` | Vendored MODNet network, moved verbatim from `src/models/` |
| `modnet_bg/errors.py` | Typed exception hierarchy shared by every module |
| `modnet_bg/config.py` | `Settings` dataclass loaded from environment |
| `modnet_bg/device.py` | Device availability and preference resolution |
| `modnet_bg/weights.py` | Checkpoint location, SHA-256 verification, optional download |
| `modnet_bg/engine.py` | Stateless matting: RGB array in, alpha array out |
| `modnet_bg/compositing.py` | The five output modes |
| `modnet_bg/media.py` | Upload decode, magic-byte typing, size and pixel guards |
| `modnet_bg/video.py` | Frame iteration, encoding, fps and audio preservation |
| `modnet_bg/jobs.py` | Job registry, worker pool, cancellation, TTL reaping |
| `modnet_bg/pipeline.py` | Wires engine + compositing + video into one callable per job |
| `modnet_bg/web/__init__.py` | `create_app()` factory |
| `modnet_bg/web/routes.py` | HTTP endpoints |
| `modnet_bg/web/realtime.py` | Webcam WebSocket handler |
| `modnet_bg/web/templates/index.html` | The single page |
| `modnet_bg/web/static/css/app.css` | Styles |
| `modnet_bg/web/static/js/app.js` | Upload, options, polling, results |
| `modnet_bg/web/static/js/compare.js` | Before/after slider |
| `modnet_bg/web/static/js/webcam.js` | getUserMedia + WebSocket loop |
| `tests/conftest.py` | `FakeMatteModel`, sample media fixtures, app fixture |
| `tests/test_*.py` | One module per package module |
| `Dockerfile` / `Dockerfile.cuda` / `docker-compose.yml` | Containers |
| `.github/workflows/ci.yml` | Lint, test, audit, image build |

**Deleted:** `api.py`, `bg_remove.py`, `inference.py`, `web_solution/`, `src/`, `pretrained/`, `requirements.txt`, `web_requirements.txt`.

---

## Phase 1 — Engine foundation

### Task 1: Scaffolding, vendored model move, CLI removal

**Files:**
- Create: `pyproject.toml`, `modnet_bg/__init__.py`, `modnet_bg/errors.py`, `tests/conftest.py`, `tests/test_packaging.py`
- Move: `src/models/` → `modnet_bg/models/` (git mv, contents unchanged except one security fix)
- Modify: `modnet_bg/models/backbones/wrapper.py:59`
- Delete: `api.py`, `bg_remove.py`, `inference.py`, `web_solution/`, `pretrained/`, `requirements.txt`, `web_requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: the `modnet_bg` package root; `modnet_bg.errors.BGRemoverError` and its subclasses, used by every later task.

- [ ] **Step 1: Write the failing test**

Create `tests/test_packaging.py`:

```python
import importlib


def test_package_imports_without_torch_side_effects():
    mod = importlib.import_module("modnet_bg")
    assert isinstance(mod.__version__, str)


def test_model_class_is_importable():
    from modnet_bg.models.modnet import MODNet

    assert MODNet is not None


def test_error_hierarchy():
    from modnet_bg.errors import (
        BGRemoverError,
        UnsupportedMediaError,
        MediaTooLargeError,
        WeightsError,
        JobNotFoundError,
    )

    for cls in (UnsupportedMediaError, MediaTooLargeError, WeightsError, JobNotFoundError):
        assert issubclass(cls, BGRemoverError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_packaging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg'`

- [ ] **Step 3: Create the package and move the vendored model**

```bash
git mv src/models modnet_bg_models_tmp
mkdir -p modnet_bg
git mv modnet_bg_models_tmp modnet_bg/models
git rm -r --cached src 2>/dev/null || true
rm -rf src
```

Create `modnet_bg/__init__.py`:

```python
"""MODNet background remover — web application package."""

__version__ = "2.0.0"
```

Create `modnet_bg/errors.py`:

```python
"""Exception hierarchy shared across the package.

Every error surfaced to a user is one of these. The web layer maps each
to an HTTP status; nothing else is allowed to reach a client.
"""


class BGRemoverError(Exception):
    """Base class for every error this package raises deliberately."""


class UnsupportedMediaError(BGRemoverError):
    """Upload is not an image or video format we accept. -> HTTP 415"""


class MediaTooLargeError(BGRemoverError):
    """Upload exceeds a configured byte, pixel, or duration cap. -> HTTP 413"""


class CorruptMediaError(BGRemoverError):
    """Upload has an accepted type but cannot be decoded. -> HTTP 422"""


class WeightsError(BGRemoverError):
    """Checkpoint is missing, unreachable, or fails its digest check."""


class JobNotFoundError(BGRemoverError):
    """No job with that id, or it has passed its TTL. -> HTTP 404"""


class JobFailedError(BGRemoverError):
    """Processing raised. The message is safe to show a user."""
```

Fix the unguarded load in `modnet_bg/models/backbones/wrapper.py` (line 59). Replace:

```python
        ckpt = torch.load(ckpt_path)
```

with:

```python
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
```

Create `tests/conftest.py` with a placeholder that later tasks extend:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 4: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "modnet-bgremover"
version = "2.0.0"
description = "Web application for removing image and video backgrounds with MODNet"
requires-python = ">=3.12"
dependencies = [
    "torch==2.14.0",
    "torchvision==0.29.0",
    "numpy==2.5.3",
    "pillow==12.3.0",
    "imageio==2.37.4",
    "imageio-ffmpeg==0.6.0",
    "flask==3.1.3",
    "werkzeug==3.1.8",
    "flask-sock==0.7.0",
    "gunicorn==26.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest==9.1.1",
    "pytest-cov==7.1.0",
    "ruff==0.16.8",
    "pip-audit==2.10.1",
]

[tool.setuptools.packages.find]
include = ["modnet_bg*"]

[tool.setuptools.package-data]
"modnet_bg.web" = ["templates/*.html", "static/css/*.css", "static/js/*.js"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = ["needs_weights: requires a real MODNet checkpoint on disk"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "S"]
ignore = ["S101"]

[tool.ruff.lint.per-file-ignores]
"modnet_bg/models/**" = ["ALL"]
```

Note the `per-file-ignores`: the vendored network is upstream code and is not ours to restyle.

- [ ] **Step 5: Delete the old entry points**

```bash
git rm -r api.py bg_remove.py inference.py web_solution pretrained requirements.txt web_requirements.txt
```

- [ ] **Step 6: Install and run the tests**

Run:
```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_packaging.py -v
```
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: scaffold modnet_bg package, vendor model, remove CLI

Moves src/models to modnet_bg/models unchanged apart from adding
weights_only=True to the backbone checkpoint load. Deletes the CLI
entry points and the demo Flask app; the web UI becomes the only
interface."
```

---

### Task 2: Device detection and policy

**Files:**
- Create: `modnet_bg/device.py`, `tests/test_device.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `available_devices() -> list[str]` — ordered subset of `["cuda", "mps", "cpu"]`, always contains `"cpu"`.
  - `resolve_device(pref: str | None) -> str` — never raises; clamps to an available device.
  - `DEVICE_PREFERENCES: tuple[str, ...]` — `("auto", "cuda", "mps", "cpu")`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_device.py`:

```python
import pytest

from modnet_bg import device as dev


@pytest.fixture
def fake_availability(monkeypatch):
    """Control what torch reports as available."""

    def apply(*, cuda: bool, mps: bool):
        monkeypatch.setattr(dev, "_cuda_available", lambda: cuda)
        monkeypatch.setattr(dev, "_mps_available", lambda: mps)

    return apply


def test_cpu_is_always_available(fake_availability):
    fake_availability(cuda=False, mps=False)
    assert dev.available_devices() == ["cpu"]


def test_ordering_puts_cuda_first(fake_availability):
    fake_availability(cuda=True, mps=True)
    assert dev.available_devices() == ["cuda", "mps", "cpu"]


def test_auto_picks_best(fake_availability):
    fake_availability(cuda=True, mps=False)
    assert dev.resolve_device("auto") == "cuda"
    fake_availability(cuda=False, mps=True)
    assert dev.resolve_device("auto") == "mps"
    fake_availability(cuda=False, mps=False)
    assert dev.resolve_device("auto") == "cpu"


def test_explicit_preference_is_honoured_when_available(fake_availability):
    fake_availability(cuda=True, mps=True)
    assert dev.resolve_device("cpu") == "cpu"
    assert dev.resolve_device("mps") == "mps"


def test_unavailable_preference_clamps_instead_of_raising(fake_availability):
    fake_availability(cuda=False, mps=False)
    assert dev.resolve_device("cuda") == "cpu"


def test_unknown_and_empty_preferences_fall_back_to_auto(fake_availability):
    fake_availability(cuda=True, mps=False)
    assert dev.resolve_device("tpu") == "cuda"
    assert dev.resolve_device(None) == "cuda"
    assert dev.resolve_device("") == "cuda"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.device'`

- [ ] **Step 3: Write the implementation**

Create `modnet_bg/device.py`:

```python
"""Which compute device to run on, and how a user preference maps onto it.

Preference resolution never fails. An unavailable or unknown request
clamps down to the best device that does exist, and the caller reports
what was actually used rather than erroring.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

DEVICE_PREFERENCES: tuple[str, ...] = ("auto", "cuda", "mps", "cpu")

# Best first. available_devices() filters this; resolve_device() scans it.
_PRIORITY: tuple[str, ...] = ("cuda", "mps", "cpu")


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def available_devices() -> list[str]:
    """Devices this process can actually use, best first. Always includes cpu."""
    present = {"cpu": True, "cuda": _cuda_available(), "mps": _mps_available()}
    return [name for name in _PRIORITY if present[name]]


def resolve_device(pref: str | None) -> str:
    """Map a preference onto a real device. Never raises."""
    devices = available_devices()
    normalised = (pref or "auto").strip().lower()

    if normalised in devices:
        return normalised

    if normalised not in DEVICE_PREFERENCES:
        logger.warning("Unknown device preference %r; using auto", pref)
    elif normalised != "auto":
        logger.info("Device %r unavailable; falling back to %s", normalised, devices[0])

    return devices[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_device.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add modnet_bg/device.py tests/test_device.py
git commit -m "feat: device detection and preference resolution"
```

---

### Task 3: Weights location and verification

**Files:**
- Create: `modnet_bg/weights.py`, `tests/test_weights.py`

**Interfaces:**
- Consumes: `modnet_bg.errors.WeightsError`.
- Produces:
  - `WeightSpec` frozen dataclass with fields `key`, `filename`, `sha256`.
  - `WEIGHTS: dict[str, WeightSpec]` keyed `"photographic"` and `"webcam"`.
  - `sha256_file(path: Path) -> str`
  - `resolve_weights(key, *, search_dirs, download_url=None, cache_dir=None) -> Path`

**Design note — read before implementing.** Spec section 5.2 assumed a first-run
download. No reliable public direct URL exists: the upstream weights live on
Google Drive (which serves an HTML interstitial rather than the file), and the
candidate HuggingFace and GitHub-release mirrors return 401/404. So the contract
is **local-first**: search configured directories, and download *only* if an
explicit URL is supplied. A missing checkpoint raises `WeightsError` carrying
instructions. Docker bind-mounts `./weights` rather than downloading.

- [ ] **Step 1: Write the failing test**

Create `tests/test_weights.py`:

```python
import hashlib

import pytest

from modnet_bg import weights as w
from modnet_bg.errors import WeightsError


def _write(path, payload: bytes):
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_known_specs_are_registered():
    assert set(w.WEIGHTS) == {"photographic", "webcam"}
    spec = w.WEIGHTS["photographic"]
    assert spec.filename == "modnet_photographic_portrait_matting.ckpt"
    assert len(spec.sha256) == 64


def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / "blob.bin"
    expected = _write(target, b"hello modnet")
    assert w.sha256_file(target) == expected


def test_resolves_first_matching_dir(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    spec = w.WEIGHTS["photographic"]
    digest = _write(second / spec.filename, b"payload")
    monkeypatch.setitem(
        w.WEIGHTS, "photographic", spec.__class__(spec.key, spec.filename, digest)
    )

    found = w.resolve_weights("photographic", search_dirs=[first, second])
    assert found == second / spec.filename


def test_digest_mismatch_is_fatal(tmp_path):
    spec = w.WEIGHTS["photographic"]
    (tmp_path / spec.filename).write_bytes(b"not the real checkpoint")

    with pytest.raises(WeightsError, match="checksum"):
        w.resolve_weights("photographic", search_dirs=[tmp_path])


def test_missing_weights_explains_how_to_fix(tmp_path):
    with pytest.raises(WeightsError) as excinfo:
        w.resolve_weights("photographic", search_dirs=[tmp_path])

    message = str(excinfo.value)
    assert "modnet_photographic_portrait_matting.ckpt" in message
    assert "MODNET_WEIGHTS_DIR" in message


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(WeightsError, match="Unknown"):
        w.resolve_weights("nope", search_dirs=[tmp_path])


def test_download_writes_atomically_and_verifies(tmp_path, monkeypatch):
    spec = w.WEIGHTS["webcam"]
    payload = b"downloaded bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(w.WEIGHTS, "webcam", spec.__class__(spec.key, spec.filename, digest))
    monkeypatch.setattr(w, "_http_get", lambda url, dest: dest.write_bytes(payload))

    cache = tmp_path / "cache"
    found = w.resolve_weights(
        "webcam", search_dirs=[tmp_path / "empty"], download_url="https://example/x", cache_dir=cache
    )

    assert found == cache / spec.filename
    assert found.read_bytes() == payload
    assert not list(cache.glob("*.part")), "temporary download file was left behind"


def test_bad_download_is_not_left_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "_http_get", lambda url, dest: dest.write_bytes(b"corrupt"))
    cache = tmp_path / "cache"

    with pytest.raises(WeightsError, match="checksum"):
        w.resolve_weights(
            "webcam", search_dirs=[tmp_path / "empty"], download_url="https://e/x", cache_dir=cache
        )

    assert not (cache / w.WEIGHTS["webcam"].filename).exists()
    assert not list(cache.glob("*.part"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_weights.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.weights'`

- [ ] **Step 3: Write the implementation**

Create `modnet_bg/weights.py`:

```python
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
    request = urllib.request.Request(url, headers={"User-Agent": "modnet-bgremover"})
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_weights.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add modnet_bg/weights.py tests/test_weights.py
git commit -m "feat: local-first weight resolution with SHA-256 verification

No reliable public mirror exists for the MODNet checkpoints, so
downloading requires an explicit URL. Nothing loads without matching
its pinned digest."
```

---

### Task 4: Stateless matting engine

**Files:**
- Create: `modnet_bg/engine.py`, `tests/test_engine.py`
- Modify: `tests/conftest.py` (add `FakeMatteModel` and `fake_engine`)

**Interfaces:**
- Consumes: `modnet_bg.device.resolve_device`, `modnet_bg.weights`.
- Produces:
  - `MattingEngine(model, device: str)` with `.matte(rgb: np.ndarray) -> np.ndarray` and `.device: str`.
  - `MattingEngine.from_checkpoint(path: Path, device: str) -> MattingEngine`
  - `get_engine(weights_path: Path, device: str) -> MattingEngine` — process-wide cache.
  - `target_size(h: int, w: int) -> tuple[int, int]` — exported for direct testing.

**Why `inference=True`.** The vendored `MODNet.forward(img, inference)` returns
`(pred_semantic, pred_detail, pred_matte)`. Reading `LRBranch.forward` and
`HRBranch.forward`: the `inference` flag gates *only* the two auxiliary heads
(`pred_semantic`, `pred_detail`). `pred_matte` is computed from `lr8x` and `hr2x`
identically either way. So `inference=True` yields a byte-identical matte while
skipping two convolution heads. The old code passed `False` and threw both
results away.

- [ ] **Step 1: Add the fake model to `tests/conftest.py`**

Append to `tests/conftest.py`:

```python
import numpy as np
import pytest
import torch
import torch.nn as nn


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
def sample_rgb():
    """A 128x96 RGB image, solid red, so colour bleed is obvious."""
    image = np.zeros((128, 96, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)
    return image
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_engine.py`:

```python
import numpy as np
import pytest

from modnet_bg.engine import MattingEngine, target_size


@pytest.mark.parametrize(
    "h,w,expected",
    [
        (512, 512, (512, 512)),
        (1000, 2000, (512, 1024)),   # landscape: short side to 512
        (2000, 1000, (1024, 512)),   # portrait: short side to 512
        (600, 16, (608, 32)),        # degenerate: width would floor to 0
        (16, 600, (32, 608)),        # the same, transposed
        (10, 10, (32, 32)),          # smaller than one stride
    ],
)
def test_target_size_is_stride_aligned_and_never_zero(h, w, expected):
    th, tw = target_size(h, w)
    assert (th, tw) == expected
    assert th % 32 == 0 and tw % 32 == 0
    assert th >= 32 and tw >= 32


def test_matte_shape_and_range(fake_engine, sample_rgb):
    alpha = fake_engine.matte(sample_rgb)

    assert alpha.shape == sample_rgb.shape[:2]
    assert alpha.dtype == np.float32
    assert 0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0


def test_extreme_aspect_ratio_does_not_crash(fake_engine):
    strip = np.full((600, 16, 3), 200, dtype=np.uint8)
    alpha = fake_engine.matte(strip)
    assert alpha.shape == (600, 16)


def test_engine_holds_no_per_image_state(fake_engine):
    first = np.full((64, 64, 3), 10, dtype=np.uint8)
    second = np.full((200, 120, 3), 20, dtype=np.uint8)

    fake_engine.matte(first)
    alpha = fake_engine.matte(second)

    assert alpha.shape == (200, 120)
    leaked = [a for a in vars(fake_engine) if a not in {"_model", "_device", "_lock"}]
    assert leaked == [], f"engine stored per-request state: {leaked}"


def test_concurrent_calls_do_not_cross_contaminate(fake_engine):
    """Regression test for the statefulness that forced this rewrite."""
    import concurrent.futures as cf

    shapes = [(64, 48), (128, 96), (200, 150), (32, 240)] * 6
    images = [np.full((h, w, 3), i % 256, dtype=np.uint8) for i, (h, w) in enumerate(shapes)]

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fake_engine.matte, images))

    for (h, w), alpha in zip(shapes, results, strict=True):
        assert alpha.shape == (h, w)


def test_rejects_malformed_input(fake_engine):
    with pytest.raises(ValueError, match="H, W, 3"):
        fake_engine.matte(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        fake_engine.matte(np.zeros((10, 10, 3), dtype=np.float32))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.engine'`

- [ ] **Step 4: Write the implementation**

Create `modnet_bg/engine.py`:

```python
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

_engine_cache: dict[tuple[str, str], "MattingEngine"] = {}
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
    def from_checkpoint(cls, weights_path: Path, device: str) -> "MattingEngine":
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

        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
        tensor = (tensor / 255.0 - _MEAN) / _STD
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=(th, tw), mode="area")

        alpha = self._forward(tensor)
        alpha = F.interpolate(alpha, size=(h, w), mode="area")
        return alpha[0, 0].clamp_(0.0, 1.0).float().cpu().numpy()

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_engine.py -v`
Expected: 11 passed (6 parametrised + 5).

- [ ] **Step 6: Commit**

```bash
git add modnet_bg/engine.py tests/test_engine.py tests/conftest.py
git commit -m "feat: stateless matting engine

Holds only a model and a device, so one instance is safe to share
across the worker pool. Clamps stride-aligned dimensions to 32 so
degenerate aspect ratios cannot produce a zero-sized tensor, runs
under inference_mode, and passes weights_only=True."
```

---

### Task 5: Output compositing

**Files:**
- Create: `modnet_bg/compositing.py`, `tests/test_compositing.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `OUTPUT_MODES: tuple[str, ...]` — `("transparent", "color", "blur", "image", "matte")`
  - `composite(rgb, alpha, mode, *, color=(255,255,255), background=None, blur_radius=24) -> np.ndarray`
  - `cover_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray` — `size` is `(h, w)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compositing.py`:

```python
import numpy as np
import pytest

from modnet_bg.compositing import OUTPUT_MODES, composite, cover_resize


@pytest.fixture
def red():
    return np.full((40, 60, 3), (255, 0, 0), dtype=np.uint8)


@pytest.fixture
def half_alpha():
    return np.full((40, 60), 0.5, dtype=np.float32)


def test_transparent_keeps_straight_rgb(red, half_alpha):
    """Regression: the old save() composited onto white AND stored alpha,
    fringing every semi-transparent pixel. RGB must be untouched."""
    out = composite(red, half_alpha, "transparent")

    assert out.shape == (40, 60, 4)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[..., :3], red)
    assert set(np.unique(out[..., 3])) <= {127, 128}


def test_transparent_fully_opaque_and_transparent_edges(red):
    alpha = np.zeros((40, 60), dtype=np.float32)
    alpha[:, 30:] = 1.0
    out = composite(red, alpha, "transparent")

    assert out[0, 0, 3] == 0
    assert out[0, 59, 3] == 255
    np.testing.assert_array_equal(out[0, 0, :3], [255, 0, 0])


def test_color_mode_blends_towards_the_chosen_colour(red, half_alpha):
    out = composite(red, half_alpha, "color", color=(0, 0, 255))

    assert out.shape == (40, 60, 3)
    np.testing.assert_allclose(out[0, 0], [127, 0, 128], atol=1)


def test_color_mode_fully_transparent_is_the_colour(red):
    alpha = np.zeros((40, 60), dtype=np.float32)
    out = composite(red, alpha, "color", color=(10, 20, 30))
    np.testing.assert_array_equal(out[0, 0], [10, 20, 30])


def test_matte_mode_is_single_channel_grayscale(red, half_alpha):
    out = composite(red, half_alpha, "matte")
    assert out.shape == (40, 60)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {127, 128}


def test_blur_mode_keeps_subject_and_softens_background(red):
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, ::2] = (255, 255, 255)  # high-frequency stripes
    alpha = np.zeros((40, 60), dtype=np.float32)
    alpha[10:30, 20:40] = 1.0

    out = composite(image, alpha, "blur", blur_radius=8)

    np.testing.assert_array_equal(out[20, 21], image[20, 21])   # subject preserved
    assert out[0, 0].std() < image[0, :].std()                  # background softened


def test_image_mode_uses_the_supplied_background(red):
    alpha = np.zeros((40, 60), dtype=np.float32)
    background = np.full((40, 60, 3), (0, 255, 0), dtype=np.uint8)

    out = composite(red, alpha, "image", background=background)
    np.testing.assert_array_equal(out[0, 0], [0, 255, 0])


def test_image_mode_requires_a_background(red, half_alpha):
    with pytest.raises(ValueError, match="background"):
        composite(red, half_alpha, "image")


def test_cover_resize_preserves_aspect_and_fills(red):
    wide = np.zeros((10, 100, 3), dtype=np.uint8)
    wide[:, :50] = (255, 0, 0)
    wide[:, 50:] = (0, 0, 255)

    out = cover_resize(wide, (40, 40))

    assert out.shape == (40, 40, 3)
    assert not (out == 0).all(axis=2).any(), "cover must not letterbox"


def test_unknown_mode_rejected(red, half_alpha):
    with pytest.raises(ValueError, match="Unknown output mode"):
        composite(red, half_alpha, "hologram")


def test_all_modes_are_reachable(red, half_alpha):
    background = np.zeros((40, 60, 3), dtype=np.uint8)
    for mode in OUTPUT_MODES:
        out = composite(red, half_alpha, mode, background=background)
        assert out.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compositing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.compositing'`

- [ ] **Step 3: Write the implementation**

Create `modnet_bg/compositing.py`:

```python
"""Turning an image plus its alpha into the output the user asked for."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

OUTPUT_MODES: tuple[str, ...] = ("transparent", "color", "blur", "image", "matte")

DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)
DEFAULT_BLUR_RADIUS = 24


def _as_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def cover_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to fill (h, w), preserving aspect, centre-cropping the overflow.

    Cover rather than stretch: a background photo should not be squashed
    to the subject's aspect ratio.
    """
    target_h, target_w = size
    src_h, src_w = image.shape[:2]
    scale = max(target_h / src_h, target_w / src_w)
    scaled = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))

    resized = np.asarray(Image.fromarray(image).resize(scaled, Image.LANCZOS))
    top = (resized.shape[0] - target_h) // 2
    left = (resized.shape[1] - target_w) // 2
    return resized[top : top + target_h, left : left + target_w]


def _blend(rgb: np.ndarray, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    return _as_uint8(rgb.astype(np.float32) * a + background.astype(np.float32) * (1.0 - a))


def composite(
    rgb: np.ndarray,
    alpha: np.ndarray,
    mode: str,
    *,
    color: tuple[int, int, int] = DEFAULT_COLOR,
    background: np.ndarray | None = None,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
) -> np.ndarray:
    """Apply an output mode. Returns RGBA for transparent, 2-D for matte, else RGB."""
    if mode not in OUTPUT_MODES:
        raise ValueError(f"Unknown output mode {mode!r}. Valid: {list(OUTPUT_MODES)}")

    if mode == "matte":
        return _as_uint8(alpha * 255.0)

    if mode == "transparent":
        # Straight (unpremultiplied) alpha: the colour channels are the
        # original pixels, untouched. Compositing them onto white here is
        # what produced the white halo in the previous implementation.
        out = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
        out[..., :3] = rgb
        out[..., 3] = _as_uint8(alpha * 255.0)
        return out

    if mode == "color":
        plate = np.empty_like(rgb)
        plate[..., :] = np.asarray(color, dtype=np.uint8)
        return _blend(rgb, alpha, plate)

    if mode == "blur":
        blurred = np.asarray(
            Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=blur_radius))
        )
        return _blend(rgb, alpha, blurred)

    # mode == "image"
    if background is None:
        raise ValueError("mode 'image' requires a background array")
    return _blend(rgb, alpha, cover_resize(background, rgb.shape[:2]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_compositing.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add modnet_bg/compositing.py tests/test_compositing.py
git commit -m "feat: output compositing with straight alpha

Transparent output keeps unpremultiplied RGB, which removes the white
edge halo the old save path produced. Background images are cover-fitted
rather than stretched."
```

---

### Task 6: Upload validation and decoding

**Files:**
- Create: `modnet_bg/media.py`, `tests/test_media.py`

**Interfaces:**
- Consumes: `modnet_bg.errors.{UnsupportedMediaError, MediaTooLargeError, CorruptMediaError}`.
- Produces:
  - `MediaKind` = `Literal["image", "video"]`
  - `MediaInfo` frozen dataclass: `kind: MediaKind`, `format: str`
  - `sniff(data: bytes) -> MediaInfo`
  - `load_image(data: bytes, *, max_pixels: int) -> np.ndarray` — RGB uint8
  - `encode_image(array: np.ndarray, fmt: str) -> bytes`
  - `IMAGE_FORMATS`, `VIDEO_FORMATS` — sets of format strings

- [ ] **Step 1: Write the failing test**

Create `tests/test_media.py`:

```python
import io

import numpy as np
import pytest
from PIL import Image

from modnet_bg import media
from modnet_bg.errors import CorruptMediaError, MediaTooLargeError, UnsupportedMediaError


def _encode(fmt: str, size=(20, 10), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, (255, 0, 0)).save(buffer, format=fmt)
    return buffer.getvalue()


def test_sniffs_image_formats_by_magic_bytes():
    assert media.sniff(_encode("PNG")).kind == "image"
    assert media.sniff(_encode("PNG")).format == "png"
    assert media.sniff(_encode("JPEG")).format == "jpeg"
    assert media.sniff(_encode("WEBP")).format == "webp"


def test_sniffs_video_formats():
    mp4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32
    assert media.sniff(mp4) == media.MediaInfo(kind="video", format="mp4")

    webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 32
    assert media.sniff(webm).kind == "video"


def test_extension_lies_are_irrelevant():
    """A .png that is really a JPEG is still handled as a JPEG."""
    assert media.sniff(_encode("JPEG")).format == "jpeg"


def test_unknown_bytes_rejected():
    with pytest.raises(UnsupportedMediaError):
        media.sniff(b"MZ\x90\x00this is a windows executable")


def test_empty_upload_rejected():
    with pytest.raises(UnsupportedMediaError):
        media.sniff(b"")


def test_load_image_returns_rgb_uint8():
    array = media.load_image(_encode("PNG", size=(20, 10)), max_pixels=10_000)
    assert array.shape == (10, 20, 3)
    assert array.dtype == np.uint8
    np.testing.assert_array_equal(array[0, 0], [255, 0, 0])


def test_load_image_flattens_rgba_and_grayscale():
    assert media.load_image(_encode("PNG", mode="RGBA"), max_pixels=10_000).shape[2] == 3
    assert media.load_image(_encode("PNG", mode="L"), max_pixels=10_000).shape[2] == 3


def test_pixel_bomb_rejected():
    with pytest.raises(MediaTooLargeError, match="pixels"):
        media.load_image(_encode("PNG", size=(200, 200)), max_pixels=1_000)


def test_truncated_image_rejected():
    payload = _encode("PNG")[:40]
    with pytest.raises(CorruptMediaError):
        media.load_image(payload, max_pixels=10_000)


def test_encode_roundtrip_preserves_alpha():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 128
    decoded = Image.open(io.BytesIO(media.encode_image(rgba, "png")))
    assert decoded.mode == "RGBA"
    assert decoded.getpixel((0, 0))[3] == 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_media.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.media'`

- [ ] **Step 3: Write the implementation**

Create `modnet_bg/media.py`:

```python
"""Deciding what an upload actually is, and decoding it safely.

Type comes from magic bytes, never the filename. An extension is a
claim by the client; the first bytes are evidence.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from .errors import CorruptMediaError, MediaTooLargeError, UnsupportedMediaError

MediaKind = Literal["image", "video"]

IMAGE_FORMATS = frozenset({"png", "jpeg", "webp", "bmp", "tiff"})
VIDEO_FORMATS = frozenset({"mp4", "webm", "avi", "mov"})

# Pillow's own bomb guard is a DecompressionBombWarning by default. We do our
# own explicit check and turn Pillow's into a hard error as a second line.
Image.MAX_IMAGE_PIXELS = 64_000_000


@dataclass(frozen=True)
class MediaInfo:
    kind: MediaKind
    format: str


def _match_prefix(data: bytes) -> MediaInfo | None:
    prefixes: tuple[tuple[bytes, MediaInfo], ...] = (
        (b"\x89PNG\r\n\x1a\n", MediaInfo("image", "png")),
        (b"\xff\xd8\xff", MediaInfo("image", "jpeg")),
        (b"BM", MediaInfo("image", "bmp")),
        (b"II*\x00", MediaInfo("image", "tiff")),
        (b"MM\x00*", MediaInfo("image", "tiff")),
        (b"\x1a\x45\xdf\xa3", MediaInfo("video", "webm")),
    )
    for prefix, info in prefixes:
        if data.startswith(prefix):
            return info
    return None


def sniff(data: bytes) -> MediaInfo:
    """Identify an upload from its leading bytes."""
    if len(data) < 12:
        raise UnsupportedMediaError("Upload is empty or too short to identify.")

    direct = _match_prefix(data)
    if direct is not None:
        return direct

    # RIFF containers carry their real type at offset 8.
    if data.startswith(b"RIFF"):
        tag = data[8:12]
        if tag == b"WEBP":
            return MediaInfo("image", "webp")
        if tag == b"AVI ":
            return MediaInfo("video", "avi")

    # ISO base media (mp4/mov) puts "ftyp" at offset 4.
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        return MediaInfo("video", "mov" if brand in (b"qt  ",) else "mp4")

    raise UnsupportedMediaError(
        "Unrecognised file type. Supported: "
        f"{', '.join(sorted(IMAGE_FORMATS | VIDEO_FORMATS))}."
    )


def load_image(data: bytes, *, max_pixels: int) -> np.ndarray:
    """Decode an image upload to RGB uint8, guarding against bombs."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > max_pixels:
                raise MediaTooLargeError(
                    f"Image is {width}x{height} ({width * height:,} pixels); "
                    f"the limit is {max_pixels:,} pixels."
                )
            return np.asarray(probe.convert("RGB"), dtype=np.uint8)
    except MediaTooLargeError:
        raise
    except Exception as exc:
        raise CorruptMediaError("Image could not be decoded; it may be truncated.") from exc


def encode_image(array: np.ndarray, fmt: str) -> bytes:
    """Encode an RGB, RGBA, or 2-D grayscale array."""
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    elif array.shape[2] == 4:
        image = Image.fromarray(array, mode="RGBA")
    else:
        image = Image.fromarray(array, mode="RGB")

    buffer = io.BytesIO()
    options = {"optimize": True} if fmt.lower() == "png" else {"quality": 95}
    image.save(buffer, format=fmt.upper(), **options)
    return buffer.getvalue()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_media.py -v`
Expected: 10 passed.

- [ ] **Step 5: Run the whole Phase 1 suite and lint**

Run:
```bash
python -m pytest -v
python -m ruff check modnet_bg tests
```
Expected: all green, no lint findings.

- [ ] **Step 6: Commit**

```bash
git add modnet_bg/media.py tests/test_media.py
git commit -m "feat: magic-byte upload validation and safe decoding

Type detection ignores the filename. Adds an explicit pixel-count cap
so a decompression bomb is rejected before allocation."
```

---

## Phase 2 — Jobs and HTTP API

### Task 7: Settings from environment

**Files:**
- Create: `modnet_bg/config.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` frozen dataclass and `load_settings(env: Mapping[str, str] | None = None) -> Settings`.
  Fields: `device`, `weights_dir`, `weights_urls`, `port`, `secret_key`, `secret_key_generated`,
  `max_image_mb`, `max_video_mb`, `max_video_seconds`, `max_workers`, `job_ttl_seconds`,
  `max_image_pixels`, `data_dir`. Properties: `max_image_bytes`, `max_video_bytes`, `max_upload_bytes`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from pathlib import Path

from modnet_bg.config import load_settings


def test_defaults_are_usable_with_an_empty_environment():
    settings = load_settings({})

    assert settings.device == "auto"
    assert settings.port == 8000
    assert settings.max_image_mb == 25
    assert settings.max_video_mb == 250
    assert settings.max_video_seconds == 120
    assert settings.max_workers == 2
    assert settings.job_ttl_seconds == 3600
    assert settings.weights_dir is None


def test_environment_overrides_every_field():
    settings = load_settings(
        {
            "MODNET_DEVICE": "cpu",
            "PORT": "9000",
            "MODNET_WEIGHTS_DIR": "/opt/weights",
            "MODNET_MAX_IMAGE_MB": "5",
            "MODNET_MAX_VIDEO_MB": "50",
            "MODNET_MAX_VIDEO_SECONDS": "30",
            "MODNET_MAX_WORKERS": "4",
            "MODNET_JOB_TTL_SECONDS": "60",
            "SECRET_KEY": "explicit",
        }
    )

    assert settings.device == "cpu"
    assert settings.port == 9000
    assert settings.weights_dir == Path("/opt/weights")
    assert settings.max_workers == 4
    assert settings.secret_key == "explicit"
    assert settings.secret_key_generated is False


def test_generated_secret_is_flagged_and_random():
    first = load_settings({})
    second = load_settings({})

    assert first.secret_key_generated is True
    assert first.secret_key != second.secret_key
    assert len(first.secret_key) >= 32


def test_upload_cap_is_the_larger_of_the_two():
    settings = load_settings({"MODNET_MAX_IMAGE_MB": "5", "MODNET_MAX_VIDEO_MB": "50"})
    assert settings.max_image_bytes == 5 * 1024 * 1024
    assert settings.max_upload_bytes == 50 * 1024 * 1024


def test_weight_urls_are_collected_per_key():
    settings = load_settings({"MODNET_WEIGHTS_URL_PHOTOGRAPHIC": "https://example/p.ckpt"})
    assert settings.weights_urls == {"photographic": "https://example/p.ckpt"}


def test_invalid_numbers_fall_back_to_the_default():
    settings = load_settings({"MODNET_MAX_WORKERS": "not-a-number"})
    assert settings.max_workers == 2


def test_worker_count_is_clamped_to_something_sane():
    assert load_settings({"MODNET_MAX_WORKERS": "0"}).max_workers == 1
    assert load_settings({"MODNET_MAX_WORKERS": "999"}).max_workers == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.config'`

- [ ] **Step 3: Write the implementation**

Create `modnet_bg/config.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add modnet_bg/config.py tests/test_config.py
git commit -m "feat: environment-driven settings"
```

---

### Task 8: Video I/O

**Files:**
- Create: `modnet_bg/video.py`, `tests/test_video.py`
- Modify: `tests/conftest.py` (add `tiny_video` fixture)

**Interfaces:**
- Consumes: `modnet_bg.errors.{CorruptMediaError, MediaTooLargeError}`.
- Produces:
  - `VideoInfo` frozen dataclass: `width`, `height`, `fps`, `frame_count`, `duration`.
  - `probe(path: Path) -> VideoInfo`
  - `iter_frames(path: Path) -> Iterator[np.ndarray]` — RGB uint8
  - `write_video(frames, dest: Path, *, fps: float, size: tuple[int, int], alpha: bool) -> None` — `size` is `(w, h)`
  - `mux_audio(video: Path, source: Path, dest: Path) -> bool` — returns False when the source has no audio
  - `container_for(mode: str) -> tuple[str, bool]` — `(extension, needs_alpha)`

- [ ] **Step 1: Add a video fixture to `tests/conftest.py`**

Append:

```python
@pytest.fixture
def tiny_video(tmp_path):
    """A real 10-frame 64x48 MP4, written with the same encoder the app uses."""
    import imageio_ffmpeg

    dest = tmp_path / "tiny.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(dest), size=(64, 48), fps=10.0, pix_fmt_in="rgb24", codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    writer.send(None)
    for i in range(10):
        frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
        writer.send(frame.tobytes())
    writer.close()
    return dest
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_video.py`:

```python
import numpy as np
import pytest

from modnet_bg import video


def test_probe_reports_real_dimensions_and_fps(tiny_video):
    info = video.probe(tiny_video)

    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(10.0, abs=0.1)
    assert info.duration == pytest.approx(1.0, abs=0.2)


def test_iter_frames_yields_rgb_uint8(tiny_video):
    frames = list(video.iter_frames(tiny_video))

    assert len(frames) == 10
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8


def test_fps_is_preserved_not_hardcoded(tmp_path, tiny_video):
    """Regression: the old writer was pinned to 20fps regardless of source."""
    dest = tmp_path / "out.mp4"
    frames = list(video.iter_frames(tiny_video))
    source_fps = video.probe(tiny_video).fps

    video.write_video(frames, dest, fps=source_fps, size=(64, 48), alpha=False)

    assert video.probe(dest).fps == pytest.approx(source_fps, abs=0.1)


def test_alpha_output_is_webm_and_round_trips(tmp_path):
    dest = tmp_path / "out.webm"
    frames = []
    for _ in range(5):
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[8:24, 8:24, 3] = 255
        frames.append(rgba)

    video.write_video(frames, dest, fps=10.0, size=(32, 32), alpha=True)

    assert dest.exists() and dest.stat().st_size > 0
    assert video.probe(dest).frame_count >= 4


def test_container_for_each_mode():
    assert video.container_for("transparent") == (".webm", True)
    assert video.container_for("color") == (".mp4", False)
    assert video.container_for("blur") == (".mp4", False)
    assert video.container_for("image") == (".mp4", False)
    assert video.container_for("matte") == (".mp4", False)


def test_mux_audio_reports_false_when_source_is_silent(tmp_path, tiny_video):
    dest = tmp_path / "muxed.mp4"
    assert video.mux_audio(tiny_video, tiny_video, dest) is False


def test_probe_rejects_a_non_video(tmp_path):
    from modnet_bg.errors import CorruptMediaError

    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64)

    with pytest.raises(CorruptMediaError):
        video.probe(bogus)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_video.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.video'`

- [ ] **Step 4: Write the implementation**

Create `modnet_bg/video.py`:

```python
"""Video decoding and encoding, via ffmpeg.

No OpenCV. imageio-ffmpeg ships a static ffmpeg binary, which is also
what we shell out to for audio muxing, so there is exactly one media
dependency rather than two.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np

from .errors import CorruptMediaError

logger = logging.getLogger(__name__)

# Alpha needs VP9 in WebM; H.264 has no alpha channel. The mode decides
# the container, so this is not independently selectable by the user.
_ALPHA_MODES = frozenset({"transparent"})


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


def probe(path: Path) -> VideoInfo:
    try:
        meta = iio.immeta(str(path), plugin="FFMPEG")
        width, height = meta["size"]
        fps = float(meta["fps"])
        duration = float(meta.get("duration") or 0.0)
    except Exception as exc:
        raise CorruptMediaError("Video could not be read; it may be corrupt.") from exc

    if not width or not height or fps <= 0:
        raise CorruptMediaError("Video reports no usable dimensions or frame rate.")

    frame_count = int(meta.get("nframes") or 0)
    if frame_count <= 0:
        frame_count = int(round(duration * fps))

    return VideoInfo(width, height, fps, frame_count, duration)


def iter_frames(path: Path) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames one at a time; never materialise the whole video."""
    try:
        yield from iio.imiter(str(path), plugin="FFMPEG")
    except Exception as exc:
        raise CorruptMediaError("Video decoding failed partway through.") from exc


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
    track, which is the common case for screen recordings and GIF-derived
    clips. Never raises for a missing track.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(dest),
    ]
    result = subprocess.run(command, capture_output=True, timeout=600, check=False)  # noqa: S603
    if result.returncode != 0:
        logger.info("No audio track muxed for %s", source.name)
        dest.unlink(missing_ok=True)
        return False
    return True
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_video.py -v`
Expected: 7 passed. If VP9 encoding is unavailable in the bundled ffmpeg,
`test_alpha_output_is_webm_and_round_trips` will fail — that is a real
finding, not a flaky test. Report it rather than skipping it.

- [ ] **Step 6: Commit**

```bash
git add modnet_bg/video.py tests/test_video.py tests/conftest.py
git commit -m "feat: ffmpeg video I/O with source fps preserved

Replaces the OpenCV writer that hardcoded 20fps and emitted a
double-width side-by-side frame. Alpha output goes to VP9/WebM;
everything else to H.264/MP4 with the original audio muxed back."
```

---

### Task 9: Job registry and worker pool

**Files:**
- Create: `modnet_bg/jobs.py`, `tests/test_jobs.py`

**Interfaces:**
- Consumes: `modnet_bg.errors.JobNotFoundError`.
- Produces:
  - `JobCancelled` exception.
  - `Job` with `id`, `status`, `kind`, `filename`, `progress`, `frames_done`, `frames_total`,
    `device_used`, `result_path`, `result_name`, `error`, `created_at`, `finished_at`;
    methods `set_total(n)`, `advance()`, `cancel()`, `raise_if_cancelled()`, `to_dict()`.
    Cancellation flag is the public `cancel_event`; callers use `job.cancel()`.
  - `JobRegistry(max_workers, ttl_seconds)` with `submit(fn, *, kind, filename) -> Job`,
    `get(job_id) -> Job`, `cancel(job_id) -> Job`, `reap() -> int`, `shutdown()`.
  - The submitted callable has signature `fn(job: Job) -> tuple[Path, str]` returning
    `(result_path, download_name)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_jobs.py`:

```python
import threading
import time

import pytest

from modnet_bg.errors import JobNotFoundError
from modnet_bg.jobs import JobCancelled, JobRegistry


@pytest.fixture
def registry(tmp_path):
    reg = JobRegistry(max_workers=2, ttl_seconds=3600)
    yield reg
    reg.shutdown()


def _wait(job, *, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.status in {"done", "error", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job stuck in {job.status}")


def test_successful_job_reports_done_and_a_result(registry, tmp_path):
    output = tmp_path / "out.png"
    output.write_bytes(b"x")

    job = registry.submit(lambda j: (output, "out.png"), kind="image", filename="in.png")
    _wait(job)

    assert job.status == "done"
    assert job.progress == 1.0
    assert job.result_path == output
    assert job.result_name == "out.png"
    assert job.error is None


def test_failing_job_records_a_safe_message(registry):
    def boom(job):
        raise RuntimeError("internal detail /srv/secret")

    job = registry.submit(boom, kind="image", filename="in.png")
    _wait(job)

    assert job.status == "error"
    assert job.error
    assert "/srv/secret" not in job.error


def test_progress_advances_towards_one(registry):
    def work(job):
        job.set_total(4)
        for _ in range(4):
            job.advance()
        return None, ""

    job = registry.submit(work, kind="video", filename="v.mp4")
    _wait(job)

    assert job.frames_done == 4
    assert job.frames_total == 4
    assert job.progress == 1.0


def test_cancel_stops_a_running_job(registry):
    started = threading.Event()

    def slow(job):
        job.set_total(1000)
        started.set()
        for _ in range(1000):
            job.raise_if_cancelled()
            time.sleep(0.005)
        return None, ""

    job = registry.submit(slow, kind="video", filename="v.mp4")
    assert started.wait(5), "job never started"
    registry.cancel(job.id)
    _wait(job)

    assert job.status == "cancelled"
    assert job.frames_done < 1000


def test_cancel_before_start_is_honoured(registry):
    blocker = threading.Event()
    for _ in range(2):
        registry.submit(lambda j: (blocker.wait(5), (None, ""))[1], kind="image", filename="b")

    queued = registry.submit(lambda j: (None, ""), kind="image", filename="q.png")
    registry.cancel(queued.id)
    blocker.set()
    _wait(queued)

    assert queued.status == "cancelled"


def test_unknown_job_raises(registry):
    with pytest.raises(JobNotFoundError):
        registry.get("not-a-job")
    with pytest.raises(JobNotFoundError):
        registry.cancel("not-a-job")


def test_reap_removes_expired_jobs_and_their_files(tmp_path):
    registry = JobRegistry(max_workers=1, ttl_seconds=0)
    try:
        artefact = tmp_path / "old.png"
        artefact.write_bytes(b"x")
        job = registry.submit(lambda j: (artefact, "old.png"), kind="image", filename="o.png")
        _wait(job)

        assert registry.reap() == 1
        assert not artefact.exists()
        with pytest.raises(JobNotFoundError):
            registry.get(job.id)
    finally:
        registry.shutdown()


def test_to_dict_is_json_safe_and_hides_paths(registry, tmp_path):
    output = tmp_path / "out.png"
    output.write_bytes(b"x")
    job = registry.submit(lambda j: (output, "out.png"), kind="image", filename="in.png")
    _wait(job)

    payload = job.to_dict()
    assert payload["status"] == "done"
    assert payload["id"] == job.id
    assert "result_path" not in payload
    assert str(tmp_path) not in repr(payload)


def test_job_ids_are_unguessable(registry):
    ids = {
        registry.submit(lambda j: (None, ""), kind="image", filename="x").id for _ in range(20)
    }
    assert len(ids) == 20
    assert all(len(i) >= 32 for i in ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.jobs'`

- [ ] **Step 3: Write the implementation**

Create `modnet_bg/jobs.py`:

```python
"""In-process job queue with progress and cancellation.

Deliberately not Celery. A single gunicorn process with a bounded thread
pool covers this workload, and it keeps the deployment to one container.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .errors import BGRemoverError, JobNotFoundError

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]

_GENERIC_ERROR = "Processing failed. Please try a different file."


class JobCancelled(Exception):
    """Raised inside a worker when the job has been cancelled."""


@dataclass
class Job:
    id: str
    kind: str
    filename: str
    status: JobStatus = "queued"
    frames_done: int = 0
    frames_total: int | None = None
    device_used: str | None = None
    result_path: Path | None = None
    result_name: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def progress(self) -> float:
        if self.status == "done":
            return 1.0
        if not self.frames_total:
            return 0.0
        return min(1.0, self.frames_done / self.frames_total)

    def set_total(self, total: int) -> None:
        with self._lock:
            self.frames_total = total

    def advance(self, by: int = 1) -> None:
        with self._lock:
            self.frames_done += by

    def raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled

    def to_dict(self) -> dict:
        """JSON-safe view. Never exposes a filesystem path."""
        return {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "status": self.status,
            "progress": round(self.progress, 4),
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "device_used": self.device_used,
            "result_name": self.result_name,
            "error": self.error,
        }


class JobRegistry:
    def __init__(self, *, max_workers: int, ttl_seconds: int) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="modnet")
        self._ttl = ttl_seconds

    def submit(
        self, fn: Callable[[Job], tuple[Path | None, str]], *, kind: str, filename: str
    ) -> Job:
        job = Job(id=secrets.token_urlsafe(24), kind=kind, filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], tuple[Path | None, str]]) -> None:
        if job._cancel.is_set():
            job.status = "cancelled"
            job.finished_at = time.time()
            return

        job.status = "running"
        try:
            result_path, result_name = fn(job)
            job.result_path = result_path
            job.result_name = result_name
            job.status = "done"
        except JobCancelled:
            job.status = "cancelled"
        except BGRemoverError as exc:
            # Our own errors carry messages written for users.
            job.status = "error"
            job.error = str(exc)
            logger.info("Job %s rejected: %s", job.id, exc)
        except Exception:
            # Anything else may contain paths or internals. Log it, do not ship it.
            job.status = "error"
            job.error = _GENERIC_ERROR
            logger.exception("Job %s failed", job.id)
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError("No such job, or it has expired.")
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
        return job

    def reap(self) -> int:
        """Delete finished jobs past their TTL, and their artefacts."""
        cutoff = time.time() - self._ttl
        removed = 0
        with self._lock:
            expired = [
                job
                for job in self._jobs.values()
                if job.finished_at is not None and job.finished_at <= cutoff
            ]
            for job in expired:
                if job.result_path is not None:
                    Path(job.result_path).unlink(missing_ok=True)
                del self._jobs[job.id]
                removed += 1
        return removed

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add modnet_bg/jobs.py tests/test_jobs.py
git commit -m "feat: in-process job registry with progress and cancellation

Unexpected exceptions are logged and replaced with a generic message,
so internal paths never reach a client."
```

---

### Task 10: Processing pipeline

**Files:**
- Create: `modnet_bg/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `engine.get_engine`, `compositing.composite`, `media`, `video`, `weights`, `config.Settings`, `jobs.Job`.
- Produces:
  - `ProcessOptions` frozen dataclass: `mode`, `color`, `background`, `blur_radius`, `device`.
  - `Pipeline(settings, engine_factory=None)` with
    `resolve_engine(device_pref) -> MattingEngine`,
    `image_job(data: bytes, options, stem: str) -> Callable[[Job], tuple[Path, str]]`,
    `video_job(source: Path, options, stem: str) -> Callable[[Job], tuple[Path, str]]`.
  - `engine_factory` signature: `(weights_path: Path, device: str) -> MattingEngine`.

- [ ] **Step 1: Add a pipeline fixture to `tests/conftest.py`**

Append:

```python
@pytest.fixture
def engine_factory():
    """An engine_factory that never touches a checkpoint.

    Exposed as a fixture rather than imported directly: `tests/` is not a
    package (no __init__.py, pytest prepend import mode), so
    `from tests.conftest import ...` would fail.
    """
    from modnet_bg.engine import MattingEngine

    return lambda path, device: MattingEngine(FakeMatteModel(), device)


@pytest.fixture
def fake_pipeline(tmp_path, engine_factory):
    """A Pipeline whose engine never touches a checkpoint."""
    from modnet_bg.config import load_settings
    from modnet_bg.pipeline import Pipeline

    settings = load_settings({"MODNET_DATA_DIR": str(tmp_path / "data"), "MODNET_DEVICE": "cpu"})
    return Pipeline(settings, engine_factory=engine_factory)
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import io

import numpy as np
import pytest
from PIL import Image

from modnet_bg.jobs import Job, JobCancelled
from modnet_bg.pipeline import Pipeline, ProcessOptions


def _png(size=(64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _job(kind="image") -> Job:
    return Job(id="test", kind=kind, filename="in.png")


def test_image_job_writes_a_transparent_png(fake_pipeline):
    run = fake_pipeline.image_job(_png(), ProcessOptions(mode="transparent"), stem="portrait")
    path, name = run(_job())

    assert path.exists()
    assert name == "portrait.png"
    with Image.open(path) as out:
        assert out.mode == "RGBA"
        assert out.size == (64, 48)


def test_image_job_colour_mode_writes_opaque_png(fake_pipeline):
    run = fake_pipeline.image_job(
        _png(), ProcessOptions(mode="color", color=(0, 0, 255)), stem="portrait"
    )
    path, _ = run(_job())

    with Image.open(path) as out:
        assert out.mode == "RGB"
        assert out.getpixel((0, 0)) == (0, 0, 255)  # corner is background


def test_image_job_records_the_device_used(fake_pipeline):
    job = _job()
    fake_pipeline.image_job(_png(), ProcessOptions(), stem="x")(job)
    assert job.device_used == "cpu"


def test_video_job_sets_total_and_advances_per_frame(fake_pipeline, tiny_video):
    job = _job(kind="video")
    run = fake_pipeline.video_job(tiny_video, ProcessOptions(mode="color"), stem="clip")
    path, name = run(job)

    assert path.exists()
    assert name == "clip.mp4"
    assert job.frames_total == 10
    assert job.frames_done == 10


def test_video_job_honours_cancellation(fake_pipeline, tiny_video):
    job = _job(kind="video")
    job.cancel()
    run = fake_pipeline.video_job(tiny_video, ProcessOptions(mode="color"), stem="clip")

    with pytest.raises(JobCancelled):
        run(job)


def test_video_duration_cap_is_enforced(tmp_path, tiny_video, engine_factory):
    from modnet_bg.config import load_settings
    from modnet_bg.errors import MediaTooLargeError

    settings = load_settings(
        {"MODNET_DATA_DIR": str(tmp_path), "MODNET_MAX_VIDEO_SECONDS": "0"}
    )
    pipeline = Pipeline(settings, engine_factory=engine_factory)

    with pytest.raises(MediaTooLargeError, match="longer than"):
        pipeline.video_job(tiny_video, ProcessOptions(), stem="clip")(_job("video"))


def test_transparent_video_produces_webm(fake_pipeline, tiny_video):
    run = fake_pipeline.video_job(tiny_video, ProcessOptions(mode="transparent"), stem="clip")
    path, name = run(_job("video"))

    assert path.suffix == ".webm"
    assert name == "clip.webm"


def test_matte_mode_produces_a_grayscale_image(fake_pipeline):
    run = fake_pipeline.image_job(_png(), ProcessOptions(mode="matte"), stem="m")
    path, _ = run(_job())

    with Image.open(path) as out:
        assert out.mode == "L"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.pipeline'`

- [ ] **Step 4: Write the implementation**

Create `modnet_bg/pipeline.py`:

```python
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
    def results_dir(self) -> Path:
        return self._results

    def resolve_engine(self, device_pref: str, key: str = "photographic") -> MattingEngine:
        device = resolve_device(device_pref if device_pref != "auto" else self._settings.device)
        search = []
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

    def _output(self, stem: str, suffix: str) -> Path:
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
            job.set_total(1)
            job.raise_if_cancelled()

            engine = self.resolve_engine(options.device)
            job.device_used = engine.device

            rgb = media.load_image(data, max_pixels=self._settings.max_image_pixels)
            alpha = engine.matte(rgb)
            out = self._composite(rgb, alpha, options)

            path = self._output(stem, ".png")
            path.write_bytes(media.encode_image(out, "png"))
            job.advance()
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
            job.raise_if_cancelled()

            engine = self.resolve_engine(options.device)
            job.device_used = engine.device

            suffix, wants_alpha = video.container_for(options.mode)
            path = self._output(stem, suffix)

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
                processed(), path, fps=info.fps, size=(info.width, info.height), alpha=wants_alpha
            )

            if not wants_alpha:
                muxed = path.with_name(f"{path.stem}-audio{suffix}")
                if video.mux_audio(path, source, muxed):
                    path.unlink(missing_ok=True)
                    path = muxed

            return path, f"{stem}{suffix}"

        return run
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add modnet_bg/pipeline.py tests/test_pipeline.py tests/conftest.py
git commit -m "feat: processing pipeline for images and video

Output filenames are server-generated tokens; no user-supplied string
reaches the filesystem."
```

---

### Task 11: Application factory, health, capabilities

**Files:**
- Create: `modnet_bg/web/__init__.py`, `modnet_bg/web/routes.py`, `tests/test_web_basics.py`
- Create: `modnet_bg/web/templates/index.html` (placeholder body; Task 13 builds the real page)

**Interfaces:**
- Consumes: `config.load_settings`, `pipeline.Pipeline`, `jobs.JobRegistry`, `device.available_devices`.
- Produces:
  - `create_app(settings=None, pipeline=None) -> Flask`
  - `app.extensions["modnet"]` holding `{"settings", "pipeline", "registry"}`.
  - Blueprint `api` registered at `/api`.

**Non-negotiable:** `create_app()` must not construct an engine or read a
checkpoint. Gunicorn imports the module at boot and the test suite runs with no
weights on disk.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_basics.py`:

```python
import pytest

from modnet_bg.web import create_app


@pytest.fixture
def client(fake_pipeline):
    app = create_app(settings=fake_pipeline._settings, pipeline=fake_pipeline)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_create_app_does_not_load_a_model(monkeypatch, tmp_path):
    import modnet_bg.engine as engine_module

    def explode(*args, **kwargs):
        raise AssertionError("create_app loaded a model at import time")

    monkeypatch.setattr(engine_module.MattingEngine, "from_checkpoint", explode)
    from modnet_bg.config import load_settings

    app = create_app(settings=load_settings({"MODNET_DATA_DIR": str(tmp_path)}))
    assert app is not None


def test_health_is_cheap_and_never_loads_a_model(client, monkeypatch):
    import modnet_bg.engine as engine_module

    monkeypatch.setattr(
        engine_module.MattingEngine,
        "from_checkpoint",
        lambda *a, **k: pytest.fail("/health loaded a model"),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_capabilities_lists_real_devices_and_limits(client):
    payload = client.get("/api/capabilities").get_json()

    assert "cpu" in payload["devices"]
    assert payload["default_device"]
    assert set(payload["modes"]) == {"transparent", "color", "blur", "image", "matte"}
    assert payload["limits"]["max_image_mb"] == 25
    assert payload["limits"]["max_video_seconds"] == 120
    assert "png" in payload["formats"]["image"]
    assert "mp4" in payload["formats"]["video"]


def test_no_cors_header_is_emitted(client):
    response = client.get("/api/capabilities")
    assert "Access-Control-Allow-Origin" not in response.headers


def test_unknown_route_returns_json_not_html(client):
    response = client.get("/api/nope")

    assert response.status_code == 404
    assert response.is_json
    assert "error" in response.get_json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_basics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.web'`

- [ ] **Step 3: Write the factory**

Create `modnet_bg/web/__init__.py`:

```python
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

_STATUS_FOR = {
    UnsupportedMediaError: 415,
    MediaTooLargeError: 413,
    CorruptMediaError: 422,
    JobNotFoundError: 404,
}


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> Flask:
    settings = settings or load_settings()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=settings.secret_key,
        MAX_CONTENT_LENGTH=settings.max_upload_bytes,
        JSON_SORT_KEYS=False,
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
    @app.errorhandler(BGRemoverError)
    def _domain_error(exc: BGRemoverError):
        status = next(
            (code for cls, code in _STATUS_FOR.items() if isinstance(exc, cls)), 400
        )
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
```

- [ ] **Step 4: Write the first routes**

Create `modnet_bg/web/routes.py`:

```python
"""HTTP endpoints. Task 12 adds the job routes to the same blueprints."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template

from .. import __version__
from ..compositing import OUTPUT_MODES
from ..device import available_devices, resolve_device
from ..media import IMAGE_FORMATS, VIDEO_FORMATS

pages = Blueprint("pages", __name__)
api = Blueprint("api", __name__)


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
```

Create a minimal `modnet_bg/web/templates/index.html` so the route renders. Task
13 replaces it wholesale:

```html
<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"><title>Background Remover</title></head>
  <body><main id="app">Loading…</main></body>
</html>
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_web_basics.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add modnet_bg/web tests/test_web_basics.py
git commit -m "feat: app factory with model-free import, health and capabilities

Errors return JSON with no internals. flask-cors is deliberately absent."
```

---

### Task 12: Job endpoints

**Files:**
- Modify: `modnet_bg/web/routes.py`
- Create: `tests/test_web_jobs.py`

**Interfaces:**
- Consumes: `jobs.JobRegistry`, `pipeline.Pipeline`, `pipeline.ProcessOptions`, `media.sniff`.
- Produces: `POST /api/jobs`, `GET /api/jobs/<id>`, `GET /api/jobs/<id>/result`,
  `DELETE /api/jobs/<id>`, `POST /api/jobs/zip`.

**Form contract for `POST /api/jobs`** (multipart):

| Field | Required | Values |
|---|---|---|
| `file` | yes | the image or video |
| `mode` | no | one of `OUTPUT_MODES`, default `transparent` |
| `color` | no | `#rrggbb`, default `#ffffff` |
| `blur_radius` | no | integer 1–100, default 24 |
| `device` | no | `auto`/`cuda`/`mps`/`cpu`, default `auto` |
| `background` | only when `mode=image` | a background image file |

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_jobs.py`:

```python
import io
import time

import pytest
from PIL import Image

from modnet_bg.web import create_app


@pytest.fixture
def client(fake_pipeline):
    app = create_app(settings=fake_pipeline._settings, pipeline=fake_pipeline)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _png_bytes(size=(64, 48), color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _upload(client, data=None, **fields):
    payload = {"file": (io.BytesIO(data or _png_bytes()), "portrait.png")}
    payload.update(fields)
    return client.post("/api/jobs", data=payload, content_type="multipart/form-data")


def _poll(client, job_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").get_json()
        if body["status"] in {"done", "error", "cancelled"}:
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_upload_returns_a_job_id(client):
    response = _upload(client)

    assert response.status_code == 202
    assert response.get_json()["id"]


def test_job_runs_to_completion_and_result_downloads(client):
    job_id = _upload(client).get_json()["id"]
    body = _poll(client, job_id)

    assert body["status"] == "done"
    assert body["progress"] == 1.0

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.headers["Content-Type"].startswith("image/png")
    assert "portrait.png" in result.headers["Content-Disposition"]


def test_result_is_404_before_completion_and_for_unknown_ids(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    assert client.get("/api/jobs/does-not-exist/result").status_code == 404


def test_upload_without_a_file_is_rejected(client):
    response = client.post("/api/jobs", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_media_upload_is_415(client):
    response = _upload(client, data=b"MZ\x90\x00 definitely not an image")
    assert response.status_code == 415


def test_invalid_mode_is_rejected(client):
    response = _upload(client, mode="hologram")
    assert response.status_code == 400
    assert "mode" in response.get_json()["error"].lower()


def test_image_mode_without_a_background_is_rejected(client):
    response = _upload(client, mode="image")
    assert response.status_code == 400


def test_image_mode_with_a_background_succeeds(client):
    response = client.post(
        "/api/jobs",
        data={
            "file": (io.BytesIO(_png_bytes()), "portrait.png"),
            "background": (io.BytesIO(_png_bytes(color=(0, 255, 0))), "bg.png"),
            "mode": "image",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    assert _poll(client, response.get_json()["id"])["status"] == "done"


def test_colour_is_parsed_from_hex(client):
    job_id = _upload(client, mode="color", color="#00ff00").get_json()["id"]
    assert _poll(client, job_id)["status"] == "done"


def test_malformed_colour_is_rejected(client):
    assert _upload(client, mode="color", color="lime").status_code == 400


def test_cancel_returns_the_job(client):
    job_id = _upload(client).get_json()["id"]
    response = client.delete(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert response.get_json()["status"] in {"cancelled", "done", "running"}


def test_oversized_upload_is_413(fake_pipeline):
    from modnet_bg.config import load_settings

    settings = load_settings(
        {"MODNET_DATA_DIR": str(fake_pipeline.results_dir.parent), "MODNET_MAX_VIDEO_MB": "1",
         "MODNET_MAX_IMAGE_MB": "1"}
    )
    app = create_app(settings=settings, pipeline=fake_pipeline)
    app.config.update(TESTING=True)

    with app.test_client() as client:
        response = client.post(
            "/api/jobs",
            data={"file": (io.BytesIO(b"\x00" * (2 * 1024 * 1024)), "big.png")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 413
    assert response.is_json


def test_zip_bundles_completed_results(client):
    import zipfile

    ids = []
    for _ in range(2):
        job_id = _upload(client).get_json()["id"]
        _poll(client, job_id)
        ids.append(job_id)

    response = client.post("/api/jobs/zip", json={"ids": ids})

    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.data))
    assert len(archive.namelist()) == 2


def test_zip_rejects_an_empty_selection(client):
    assert client.post("/api/jobs/zip", json={"ids": []}).status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_jobs.py -v`
Expected: FAIL — `POST /api/jobs` returns 404, no such route.

- [ ] **Step 3: Extend `modnet_bg/web/routes.py`**

Add these imports at the top of the file. `OUTPUT_MODES` is already imported by
Task 11 — do not add it twice:

```python
import io
import re
import tempfile
import zipfile
from pathlib import Path

from flask import request, send_file
from werkzeug.utils import secure_filename

from ..errors import BGRemoverError, JobNotFoundError, MediaTooLargeError
from ..media import load_image, sniff
from ..pipeline import ProcessOptions
```

Append to the same file:

```python
class BadRequest(BGRemoverError):
    """Malformed request parameters. -> HTTP 400"""


_HEX_COLOR = re.compile(r"^#(?P<r>[0-9a-fA-F]{2})(?P<g>[0-9a-fA-F]{2})(?P<b>[0-9a-fA-F]{2})$")


def _parse_color(raw: str) -> tuple[int, int, int]:
    match = _HEX_COLOR.match(raw.strip())
    if not match:
        raise BadRequest(f"color must be #rrggbb, got {raw!r}")
    return tuple(int(match.group(c), 16) for c in ("r", "g", "b"))  # type: ignore[return-value]


def _stem(filename: str) -> str:
    """A safe display stem for the download name. Never used as a path."""
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
    stem = _stem(upload.filename)

    if info.kind == "image":
        if len(data) > settings.max_image_bytes:
            raise MediaTooLargeError(f"Image exceeds the {settings.max_image_mb}MB limit.")
        runner = pipeline.image_job(data, options, stem)
    else:
        if len(data) > settings.max_video_bytes:
            raise MediaTooLargeError(f"Video exceeds the {settings.max_video_mb}MB limit.")
        scratch = Path(tempfile.mkdtemp(dir=settings.data_dir)) / f"source.{info.format}"
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
```

Register `BadRequest` in `modnet_bg/web/__init__.py`'s `_STATUS_FOR` map:

```python
from .routes import BadRequest  # inside _register_error_handlers, after blueprints import

_STATUS_FOR = {
    BadRequest: 400,
    UnsupportedMediaError: 415,
    MediaTooLargeError: 413,
    CorruptMediaError: 422,
    JobNotFoundError: 404,
}
```

Because `BadRequest` subclasses `BGRemoverError`, the existing handler covers it;
it only needs the status mapping. Move the `_STATUS_FOR` construction inside
`_register_error_handlers` to avoid a circular import at module load.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_jobs.py -v`
Expected: 14 passed.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v && python -m ruff check modnet_bg tests`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add modnet_bg/web/routes.py modnet_bg/web/__init__.py tests/test_web_jobs.py
git commit -m "feat: job endpoints with upload validation and zip download"
```

---

## Phase 3 — Frontend

Frontend tasks have no JS test runner (deliberate: no build step, no npm). They
are verified two ways, and **both are required**:

1. **Automated contract tests** in `tests/test_frontend_assets.py` assert the
   template and scripts expose the hooks the other layer depends on. These catch
   a renamed id or a dropped element.
2. **Manual browser verification** at the end of each task, driven through the
   built-in browser against a locally running server. Each task lists the exact
   checks. Do not mark a task complete on the automated tests alone.

### Task 13: Page shell, upload, options, progress

**Files:**
- Replace: `modnet_bg/web/templates/index.html`
- Create: `modnet_bg/web/static/css/app.css`, `modnet_bg/web/static/js/app.js`
- Create: `tests/test_frontend_assets.py`

**Interfaces:**
- Consumes: `GET /api/capabilities`, `POST /api/jobs`, `GET /api/jobs/<id>`, `DELETE /api/jobs/<id>`.
- Produces: DOM ids `#drop-zone`, `#file-input`, `#mode`, `#color`, `#blur-radius`,
  `#background-input`, `#device`, `#queue`, `#status-region`; and
  `window.BGR = { state, addFiles, submit, poll }` for the later tasks to extend.

**Assets are local.** The old page loaded Bootstrap, jQuery, Popper, and axios
from three CDNs. Do not reintroduce any CDN reference: it is a supply-chain
exposure and it makes the app fail offline.

- [ ] **Step 1: Write the failing test**

Create `tests/test_frontend_assets.py`:

```python
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "modnet_bg" / "web"
INDEX = WEB / "templates" / "index.html"
APP_JS = WEB / "static" / "js" / "app.js"
APP_CSS = WEB / "static" / "css" / "app.css"


@pytest.fixture(scope="module")
def index_html() -> str:
    return INDEX.read_text()


def test_required_assets_exist():
    for path in (INDEX, APP_JS, APP_CSS):
        assert path.is_file(), f"missing {path}"


@pytest.mark.parametrize(
    "element_id",
    ["drop-zone", "file-input", "mode", "color", "blur-radius",
     "background-input", "device", "queue", "status-region"],
)
def test_template_exposes_every_hook(index_html, element_id):
    assert f'id="{element_id}"' in index_html


def test_no_cdn_references(index_html):
    """Every asset must be served from this origin."""
    for marker in ("http://", "https://", "//cdn", "unpkg", "jsdelivr", "googleapis"):
        assert marker not in index_html, f"external reference {marker!r} in template"


def test_scripts_are_loaded_via_url_for(index_html):
    assert "url_for('static'" in index_html or 'url_for("static"' in index_html


def test_status_region_is_announced_to_screen_readers(index_html):
    region = index_html.split('id="status-region"')[1][:200]
    assert "aria-live" in region


def test_file_input_has_a_label(index_html):
    assert 'for="file-input"' in index_html


def test_app_js_exports_the_public_surface():
    source = APP_JS.read_text()
    for symbol in ("window.BGR", "addFiles", "submit", "poll"):
        assert symbol in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frontend_assets.py -v`
Expected: FAIL on `test_required_assets_exist` (`app.js` and `app.css` missing).

- [ ] **Step 3: Write `modnet_bg/web/templates/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Background Remover</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/app.css') }}">
</head>
<body>
<header class="masthead">
  <h1>Background Remover</h1>
  <p class="subtitle">Drop in a photo or a video. Everything runs on this machine.</p>
</header>

<main class="layout">
  <section class="panel" aria-labelledby="upload-heading">
    <h2 id="upload-heading">1 &middot; Choose files</h2>

    <div id="drop-zone" class="drop-zone" tabindex="0"
         role="button" aria-describedby="drop-hint">
      <p class="drop-zone__title">Drag files here</p>
      <p id="drop-hint" class="drop-zone__hint">or use the button below. Images and video.</p>
    </div>

    <label class="field" for="file-input">Files</label>
    <input id="file-input" type="file" multiple accept="image/*,video/*">
  </section>

  <section class="panel" aria-labelledby="options-heading">
    <h2 id="options-heading">2 &middot; Choose a background</h2>

    <label class="field" for="mode">Output</label>
    <select id="mode">
      <option value="transparent">Transparent</option>
      <option value="color">Solid colour</option>
      <option value="blur">Blurred original</option>
      <option value="image">Your own image</option>
      <option value="matte">Alpha matte only</option>
    </select>

    <div class="option-row" data-when-mode="color">
      <label class="field" for="color">Colour</label>
      <input id="color" type="color" value="#ffffff">
    </div>

    <div class="option-row" data-when-mode="blur">
      <label class="field" for="blur-radius">Blur strength</label>
      <input id="blur-radius" type="range" min="1" max="100" value="24">
    </div>

    <div class="option-row" data-when-mode="image">
      <label class="field" for="background-input">Background image</label>
      <input id="background-input" type="file" accept="image/*">
    </div>

    <label class="field" for="device">Processing device</label>
    <select id="device"><option value="auto">Auto</option></select>
    <p id="device-note" class="hint"></p>
  </section>

  <section class="panel panel--wide" aria-labelledby="queue-heading">
    <h2 id="queue-heading">3 &middot; Results</h2>
    <div class="queue-actions">
      <button id="start" type="button" class="button button--primary">Remove backgrounds</button>
      <button id="download-all" type="button" class="button" hidden>Download all as zip</button>
    </div>
    <p id="status-region" class="status" role="status" aria-live="polite"></p>
    <ul id="queue" class="queue"></ul>
  </section>
</main>

<script src="{{ url_for('static', filename='js/app.js') }}" defer></script>
<script src="{{ url_for('static', filename='js/compare.js') }}" defer></script>
</body>
</html>
```

- [ ] **Step 4: Write `modnet_bg/web/static/css/app.css`**

```css
:root {
  --bg: #0f1115;
  --surface: #171a21;
  --surface-raised: #1f242d;
  --border: #2b313c;
  --text: #e8eaed;
  --text-dim: #9aa3b2;
  --accent: #5b8cff;
  --danger: #ff6b6b;
  --ok: #4ade80;
  --radius: 10px;
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: #f6f7f9; --surface: #ffffff; --surface-raised: #f0f2f5;
    --border: #d7dbe2; --text: #1a1d23; --text-dim: #5c6472;
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}

.masthead { padding: 32px 16px 8px; max-width: 1100px; margin: 0 auto; }
.masthead h1 { margin: 0; font-size: 1.75rem; letter-spacing: -0.02em; }
.subtitle { color: var(--text-dim); margin: 6px 0 0; }

.layout {
  max-width: 1100px; margin: 0 auto; padding: 16px;
  display: grid; gap: 16px; grid-template-columns: repeat(2, 1fr);
}
.panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 18px;
}
.panel--wide { grid-column: 1 / -1; }
.panel h2 { margin: 0 0 14px; font-size: 0.85rem; text-transform: uppercase;
            letter-spacing: 0.08em; color: var(--text-dim); }

.drop-zone {
  border: 2px dashed var(--border); border-radius: var(--radius);
  padding: 28px 16px; text-align: center; cursor: pointer;
  transition: border-color 120ms ease, background 120ms ease;
}
.drop-zone:hover, .drop-zone:focus-visible { border-color: var(--accent); }
.drop-zone.is-dragging { border-color: var(--accent); background: var(--surface-raised); }
.drop-zone__title { margin: 0; font-weight: 600; }
.drop-zone__hint { margin: 4px 0 0; color: var(--text-dim); font-size: 0.875rem; }

.field { display: block; margin: 14px 0 6px; font-size: 0.8rem;
         font-weight: 600; color: var(--text-dim); }
input, select {
  width: 100%; padding: 9px 10px; color: var(--text);
  background: var(--surface-raised); border: 1px solid var(--border);
  border-radius: 8px; font: inherit;
}
input[type="color"] { padding: 4px; height: 40px; }
.option-row[hidden] { display: none; }
.hint { color: var(--text-dim); font-size: 0.8rem; margin: 6px 0 0; }

.button {
  padding: 10px 16px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface-raised); color: var(--text);
  font: inherit; font-weight: 600; cursor: pointer;
}
.button--primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.button:disabled { opacity: 0.5; cursor: not-allowed; }
.queue-actions { display: flex; gap: 10px; flex-wrap: wrap; }

.status { min-height: 1.4em; margin: 12px 0; color: var(--text-dim); }
.status[data-tone="error"] { color: var(--danger); }
.status[data-tone="ok"] { color: var(--ok); }

.queue { list-style: none; margin: 0; padding: 0; display: grid; gap: 12px; }
.queue-item {
  background: var(--surface-raised); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px;
  display: grid; gap: 10px; grid-template-columns: 1fr auto; align-items: center;
}
.queue-item__name { font-weight: 600; word-break: break-all; }
.queue-item__meta { color: var(--text-dim); font-size: 0.8rem; }
.queue-item__preview { grid-column: 1 / -1; }

progress { width: 100%; height: 6px; grid-column: 1 / -1; }

/* Checkerboard so transparency is actually visible. */
.checker {
  background-image:
    linear-gradient(45deg, #8d8d8d 25%, transparent 25%),
    linear-gradient(-45deg, #8d8d8d 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #8d8d8d 75%),
    linear-gradient(-45deg, transparent 75%, #8d8d8d 75%);
  background-size: 16px 16px;
  background-position: 0 0, 0 8px, 8px -8px, -8px 0;
  background-color: #b6b6b6;
}

:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

@media (max-width: 720px) {
  .layout { grid-template-columns: 1fr; }
}
```

- [ ] **Step 5: Write `modnet_bg/web/static/js/app.js`**

```javascript
/* Background Remover — upload, options, and job polling.
   No framework, no build step: this file runs as-is. */
(() => {
  "use strict";

  const POLL_INTERVAL_MS = 400;

  const state = { files: [], jobs: new Map(), capabilities: null };

  const $ = (id) => document.getElementById(id);
  const els = {};

  function setStatus(message, tone = "") {
    els.status.textContent = message;
    els.status.dataset.tone = tone;
  }

  async function loadCapabilities() {
    const response = await fetch("/api/capabilities");
    if (!response.ok) throw new Error("capabilities unavailable");
    state.capabilities = await response.json();

    const labels = { auto: "Auto", cuda: "NVIDIA GPU (CUDA)", mps: "Apple GPU (Metal)", cpu: "CPU" };
    els.device.innerHTML = "";
    for (const name of ["auto", ...state.capabilities.devices]) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = labels[name] || name;
      els.device.append(option);
    }

    const active = state.capabilities.default_device;
    const gpu = state.capabilities.devices.some((d) => d !== "cpu");
    els.deviceNote.textContent = gpu
      ? `A GPU is available. Auto currently selects ${active}. Choose CPU to turn it off.`
      : "No GPU detected on this machine; everything runs on the CPU.";
  }

  function syncModeOptions() {
    const mode = els.mode.value;
    for (const row of document.querySelectorAll("[data-when-mode]")) {
      row.hidden = row.dataset.whenMode !== mode;
    }
  }

  function addFiles(fileList) {
    for (const file of fileList) state.files.push(file);
    renderQueue();
    setStatus(`${state.files.length} file(s) ready.`);
  }

  function renderQueue() {
    els.queue.innerHTML = "";
    state.files.forEach((file, index) => {
      const job = state.jobs.get(index);
      const item = document.createElement("li");
      item.className = "queue-item";
      item.dataset.index = String(index);

      const name = document.createElement("div");
      name.className = "queue-item__name";
      name.textContent = file.name;

      const meta = document.createElement("div");
      meta.className = "queue-item__meta";
      meta.textContent = describe(job);

      item.append(name, meta);

      if (job && job.status === "running") {
        const bar = document.createElement("progress");
        bar.max = 1;
        bar.value = job.progress || 0;
        item.append(bar);
      }

      if (job && job.status === "done") {
        const link = document.createElement("a");
        link.className = "button";
        link.href = `/api/jobs/${job.id}/result`;
        link.textContent = "Download";
        link.download = job.result_name || "";
        item.append(link);
        item.append(buildPreview(file, job));
      }

      els.queue.append(item);
    });

    const anyDone = [...state.jobs.values()].some((j) => j.status === "done");
    els.downloadAll.hidden = !anyDone;
  }

  function describe(job) {
    if (!job) return "Waiting to start";
    if (job.status === "error") return job.error || "Failed";
    if (job.status === "cancelled") return "Cancelled";
    if (job.status === "done") return `Done on ${job.device_used || "cpu"}`;
    if (job.frames_total) return `Frame ${job.frames_done} of ${job.frames_total}`;
    return "Processing";
  }

  function buildPreview(file, job) {
    const wrapper = document.createElement("div");
    wrapper.className = "queue-item__preview";
    // compare.js upgrades this into a draggable before/after slider.
    wrapper.dataset.compare = "";
    wrapper.dataset.before = URL.createObjectURL(file);
    wrapper.dataset.after = `/api/jobs/${job.id}/result`;
    wrapper.dataset.kind = job.kind;
    document.dispatchEvent(new CustomEvent("bgr:preview", { detail: { wrapper } }));
    return wrapper;
  }

  function buildForm(file) {
    const form = new FormData();
    form.append("file", file);
    form.append("mode", els.mode.value);
    form.append("color", els.color.value);
    form.append("blur_radius", els.blur.value);
    form.append("device", els.device.value);
    if (els.mode.value === "image" && els.background.files[0]) {
      form.append("background", els.background.files[0]);
    }
    return form;
  }

  async function submit() {
    if (state.files.length === 0) {
      setStatus("Choose at least one file first.", "error");
      return;
    }
    if (els.mode.value === "image" && !els.background.files[0]) {
      setStatus("Pick a background image, or choose a different output mode.", "error");
      return;
    }

    els.start.disabled = true;
    setStatus("Uploading…");

    for (const [index, file] of state.files.entries()) {
      try {
        const response = await fetch("/api/jobs", { method: "POST", body: buildForm(file) });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || `Upload failed (${response.status})`);
        state.jobs.set(index, body);
        renderQueue();
        await poll(index, body.id);
      } catch (error) {
        state.jobs.set(index, { status: "error", error: error.message });
        renderQueue();
      }
    }

    els.start.disabled = false;
    const failed = [...state.jobs.values()].filter((j) => j.status === "error").length;
    setStatus(failed ? `${failed} file(s) failed.` : "All done.", failed ? "error" : "ok");
  }

  async function poll(index, jobId) {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      const response = await fetch(`/api/jobs/${jobId}`);
      if (!response.ok) throw new Error("Lost track of that job.");
      const body = await response.json();
      state.jobs.set(index, body);
      renderQueue();
      if (["done", "error", "cancelled"].includes(body.status)) return body;
    }
  }

  async function downloadAll() {
    const ids = [...state.jobs.values()].filter((j) => j.status === "done").map((j) => j.id);
    const response = await fetch("/api/jobs/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    if (!response.ok) {
      setStatus("Could not build the zip.", "error");
      return;
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "backgrounds.zip";
    link.click();
    URL.revokeObjectURL(url);
  }

  function wireDropZone() {
    const zone = els.dropZone;
    zone.addEventListener("click", () => els.fileInput.click());
    zone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        els.fileInput.click();
      }
    });
    for (const type of ["dragenter", "dragover"]) {
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        zone.classList.add("is-dragging");
      });
    }
    for (const type of ["dragleave", "drop"]) {
      zone.addEventListener(type, () => zone.classList.remove("is-dragging"));
    }
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      addFiles(event.dataTransfer.files);
    });
  }

  function init() {
    Object.assign(els, {
      dropZone: $("drop-zone"), fileInput: $("file-input"), mode: $("mode"),
      color: $("color"), blur: $("blur-radius"), background: $("background-input"),
      device: $("device"), deviceNote: $("device-note"), queue: $("queue"),
      status: $("status-region"), start: $("start"), downloadAll: $("download-all"),
    });

    wireDropZone();
    els.fileInput.addEventListener("change", (event) => addFiles(event.target.files));
    els.mode.addEventListener("change", syncModeOptions);
    els.start.addEventListener("click", submit);
    els.downloadAll.addEventListener("click", downloadAll);

    syncModeOptions();
    loadCapabilities().catch(() => setStatus("Could not reach the server.", "error"));
  }

  window.BGR = { state, addFiles, submit, poll, renderQueue };
  document.addEventListener("DOMContentLoaded", init);
})();
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_frontend_assets.py -v`
Expected: 15 passed (9 parametrised + 6). `compare.js` is referenced by the
template but does not exist yet; that is fine — the browser logs a 404 and Task
14 creates it.

- [ ] **Step 7: Verify in a real browser**

Start the server:

```bash
MODNET_WEIGHTS_DIR=$PWD/weights python -m flask --app "modnet_bg.web:create_app()" run --port 8000
```

Then, using the built-in browser at `http://localhost:8000`, confirm each of:
- The page renders with three panels and no console errors other than the
  `compare.js` 404.
- The device dropdown is populated from `/api/capabilities`, and on this Mac
  offers Auto, Apple GPU (Metal), and CPU.
- Changing **Output** shows and hides exactly the matching option row.
- Dragging a file onto the drop zone highlights it and adds the name to the queue.
- Tabbing reaches the drop zone and Enter opens the file picker.
- Uploading `assets/sample_image/male.jpeg` runs to "Done" and the Download link
  returns a transparent PNG.

- [ ] **Step 8: Commit**

```bash
git add modnet_bg/web/templates modnet_bg/web/static tests/test_frontend_assets.py
git commit -m "feat: upload, options, and progress UI

All assets served from this origin; the previous page pulled Bootstrap,
jQuery, Popper, and axios from three CDNs."
```

---

### Task 14: Before/after comparison slider

**Files:**
- Create: `modnet_bg/web/static/js/compare.js`
- Modify: `tests/test_frontend_assets.py`

**Interfaces:**
- Consumes: the `bgr:preview` CustomEvent from `app.js`, whose `detail.wrapper`
  carries `dataset.before`, `dataset.after`, and `dataset.kind`.
- Produces: an upgraded DOM node with class `compare`, and `window.BGRCompare.mount(wrapper)`.

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_frontend_assets.py`:

```python
COMPARE_JS = WEB / "static" / "js" / "compare.js"


def test_compare_js_exists_and_is_wired():
    assert COMPARE_JS.is_file()
    source = COMPARE_JS.read_text()
    assert "bgr:preview" in source
    assert "window.BGRCompare" in source


def test_compare_is_keyboard_accessible():
    source = COMPARE_JS.read_text()
    assert "keydown" in source, "slider must be operable without a mouse"
    assert "ArrowLeft" in source and "ArrowRight" in source


def test_transparency_is_shown_against_a_checkerboard():
    assert "checker" in COMPARE_JS.read_text()
    assert ".checker" in APP_CSS.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frontend_assets.py -v`
Expected: FAIL on `test_compare_js_exists_and_is_wired`.

- [ ] **Step 3: Write `modnet_bg/web/static/js/compare.js`**

```javascript
/* Draggable before/after divider over a result.
   Listens for app.js's bgr:preview event and upgrades the node in place. */
(() => {
  "use strict";

  function mediaElement(source, kind) {
    if (kind === "video") {
      const video = document.createElement("video");
      video.src = source;
      video.controls = false;
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      video.play().catch(() => {});
      return video;
    }
    const image = document.createElement("img");
    image.src = source;
    image.alt = "";
    return image;
  }

  function mount(wrapper) {
    const { before, after, kind } = wrapper.dataset;
    if (!before || !after) return;

    wrapper.classList.add("compare", "checker");
    wrapper.innerHTML = "";

    const afterLayer = document.createElement("div");
    afterLayer.className = "compare__layer compare__layer--after";
    afterLayer.append(mediaElement(after, kind));

    const beforeLayer = document.createElement("div");
    beforeLayer.className = "compare__layer compare__layer--before";
    beforeLayer.append(mediaElement(before, kind));

    const handle = document.createElement("div");
    handle.className = "compare__handle";
    handle.setAttribute("role", "slider");
    handle.setAttribute("tabindex", "0");
    handle.setAttribute("aria-label", "Reveal the original image");
    handle.setAttribute("aria-valuemin", "0");
    handle.setAttribute("aria-valuemax", "100");

    let position = 50;

    function apply(next) {
      position = Math.min(100, Math.max(0, next));
      beforeLayer.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
      handle.style.left = `${position}%`;
      handle.setAttribute("aria-valuenow", String(Math.round(position)));
    }

    function fromPointer(event) {
      const box = wrapper.getBoundingClientRect();
      apply(((event.clientX - box.left) / box.width) * 100);
    }

    let dragging = false;
    wrapper.addEventListener("pointerdown", (event) => {
      dragging = true;
      wrapper.setPointerCapture(event.pointerId);
      fromPointer(event);
    });
    wrapper.addEventListener("pointermove", (event) => dragging && fromPointer(event));
    wrapper.addEventListener("pointerup", () => { dragging = false; });
    wrapper.addEventListener("pointercancel", () => { dragging = false; });

    handle.addEventListener("keydown", (event) => {
      const step = event.shiftKey ? 10 : 2;
      if (event.key === "ArrowLeft") { apply(position - step); event.preventDefault(); }
      if (event.key === "ArrowRight") { apply(position + step); event.preventDefault(); }
      if (event.key === "Home") { apply(0); event.preventDefault(); }
      if (event.key === "End") { apply(100); event.preventDefault(); }
    });

    wrapper.append(afterLayer, beforeLayer, handle);
    apply(50);
  }

  document.addEventListener("bgr:preview", (event) => mount(event.detail.wrapper));
  window.BGRCompare = { mount };
})();
```

- [ ] **Step 4: Add the slider styles**

Append to `modnet_bg/web/static/css/app.css`:

```css
.compare {
  position: relative; overflow: hidden; border-radius: var(--radius);
  touch-action: none; user-select: none; max-height: 420px;
}
.compare__layer { position: absolute; inset: 0; }
.compare__layer img, .compare__layer video {
  width: 100%; height: 100%; object-fit: contain; display: block;
}
.compare__layer--after { position: relative; }
.compare__handle {
  position: absolute; top: 0; bottom: 0; width: 3px; margin-left: -1.5px;
  background: #fff; box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35); cursor: ew-resize;
}
.compare__handle::after {
  content: ""; position: absolute; top: 50%; left: 50%;
  width: 34px; height: 34px; transform: translate(-50%, -50%);
  border-radius: 50%; background: #fff; box-shadow: 0 1px 6px rgba(0, 0, 0, 0.4);
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_frontend_assets.py -v`
Expected: 18 passed.

- [ ] **Step 6: Verify in a real browser**

With the server running, process `assets/sample_image/female.jpeg` and confirm:
- The result shows a checkerboard behind the transparent region.
- Dragging the handle wipes between original and cut-out, and following the
  pointer outside the element does not break the drag.
- Tab reaches the handle; Arrow keys move it in 2% steps, Shift+Arrow in 10%,
  Home and End jump to the ends.
- Process `assets/sample_video/sample.mp4` with **Solid colour** and confirm the
  result plays, is the same size as the source, and is not double-width.

- [ ] **Step 7: Commit**

```bash
git add modnet_bg/web/static tests/test_frontend_assets.py
git commit -m "feat: keyboard-accessible before/after comparison slider"
```

---

## Phase 4 — Webcam, containers, CI

### Task 15: Realtime WebSocket backend

**Files:**
- Create: `modnet_bg/web/realtime.py`, `tests/test_realtime.py`
- Modify: `modnet_bg/web/__init__.py` (register the socket)

**Interfaces:**
- Consumes: `pipeline.Pipeline`, `media.load_image`, `compositing.composite`, `media.encode_image`.
- Produces:
  - `process_frame(pipeline, data: bytes, options: RealtimeOptions) -> bytes` — pure, testable, returns PNG-encoded grayscale alpha.
  - `RealtimeOptions` frozen dataclass: `device`, `max_edge`.
  - `register_realtime(app) -> None` mounting `WS /ws/realtime`.

**Protocol.** The client opens the socket, sends one JSON text frame of options,
then alternates: send a binary JPEG, receive a binary PNG alpha mask. Exactly one
frame is in flight — the client does not send the next until the previous reply
lands, so a slow server drops frames instead of queueing them. Errors arrive as a
JSON text frame with an `error` key.

**Why return only the mask.** One channel instead of four, and the client already
holds the source frame on a canvas. Compositing client-side also means changing
the background colour does not require a round trip.

- [ ] **Step 1: Write the failing test**

Create `tests/test_realtime.py`:

```python
import io

import numpy as np
import pytest
from PIL import Image

from modnet_bg.web.realtime import RealtimeOptions, process_frame


def _jpeg(size=(320, 240)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_returns_a_grayscale_png_mask(fake_pipeline):
    payload = process_frame(fake_pipeline, _jpeg(), RealtimeOptions())

    with Image.open(io.BytesIO(payload)) as mask:
        assert mask.format == "PNG"
        assert mask.mode == "L"


def test_mask_matches_the_downscaled_frame_size(fake_pipeline):
    payload = process_frame(fake_pipeline, _jpeg((1280, 720)), RealtimeOptions(max_edge=512))

    with Image.open(io.BytesIO(payload)) as mask:
        assert max(mask.size) <= 512
        assert mask.size[0] / mask.size[1] == pytest.approx(1280 / 720, rel=0.02)


def test_small_frames_are_not_upscaled(fake_pipeline):
    payload = process_frame(fake_pipeline, _jpeg((160, 120)), RealtimeOptions(max_edge=512))

    with Image.open(io.BytesIO(payload)) as mask:
        assert mask.size == (160, 120)


def test_mask_has_real_variation(fake_pipeline):
    """The fake model produces a central box; a flat mask means a wiring bug."""
    payload = process_frame(fake_pipeline, _jpeg(), RealtimeOptions())

    with Image.open(io.BytesIO(payload)) as mask:
        values = np.asarray(mask)
    assert values.min() == 0 and values.max() == 255


def test_garbage_frame_raises_a_domain_error(fake_pipeline):
    from modnet_bg.errors import BGRemoverError

    with pytest.raises(BGRemoverError):
        process_frame(fake_pipeline, b"not a jpeg at all", RealtimeOptions())


def test_socket_route_is_registered(fake_pipeline):
    from modnet_bg.web import create_app

    app = create_app(settings=fake_pipeline._settings, pipeline=fake_pipeline)
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/ws/realtime" in rules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_realtime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modnet_bg.web.realtime'`

- [ ] **Step 3: Write `modnet_bg/web/realtime.py`**

```python
"""Webcam matting over a WebSocket.

Throughput is bounded by the model, not this code: expect roughly 2-5
fps on a CPU and 15-30 fps on CUDA. The client measures and displays the
real rate rather than implying it should be smooth.
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
    def realtime(ws):  # pragma: no cover - exercised manually in Step 6
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
```

- [ ] **Step 4: Register it in `modnet_bg/web/__init__.py`**

Inside `create_app`, immediately after the blueprints are registered:

```python
    from .realtime import register_realtime

    register_realtime(app)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_realtime.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add modnet_bg/web/realtime.py modnet_bg/web/__init__.py tests/test_realtime.py
git commit -m "feat: webcam matting over a WebSocket

Returns a single-channel mask and composites client-side, so changing
the background needs no round trip."
```

---

### Task 16: Webcam UI

**Files:**
- Create: `modnet_bg/web/static/js/webcam.js`
- Modify: `modnet_bg/web/templates/index.html`, `modnet_bg/web/static/css/app.css`, `tests/test_frontend_assets.py`

**Interfaces:**
- Consumes: `WS /ws/realtime`, `GET /api/capabilities`.
- Produces: DOM ids `#webcam-panel`, `#webcam-start`, `#webcam-stop`, `#webcam-canvas`, `#webcam-fps`.

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_frontend_assets.py`:

```python
WEBCAM_JS = WEB / "static" / "js" / "webcam.js"


@pytest.mark.parametrize(
    "element_id", ["webcam-panel", "webcam-start", "webcam-stop", "webcam-canvas", "webcam-fps"]
)
def test_webcam_hooks_present(index_html, element_id):
    assert f'id="{element_id}"' in index_html


def test_webcam_js_keeps_one_frame_in_flight():
    source = WEBCAM_JS.read_text()
    assert "getUserMedia" in source
    assert "inFlight" in source, "backpressure flag missing; frames will queue up"


def test_webcam_reports_measured_fps():
    assert "webcam-fps" in WEBCAM_JS.read_text()


def test_webcam_stops_all_tracks_on_stop():
    """Leaving the camera light on after Stop is a privacy bug."""
    source = WEBCAM_JS.read_text()
    assert "getTracks" in source and "stop()" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frontend_assets.py -v`
Expected: FAIL on the webcam hook parametrisation.

- [ ] **Step 3: Add the panel to `index.html`**

Insert before the closing `</main>`:

```html
  <section class="panel panel--wide" id="webcam-panel" aria-labelledby="webcam-heading">
    <h2 id="webcam-heading">Live camera</h2>
    <p class="hint">
      Runs frame by frame on this machine. Expect a few frames per second on a
      CPU, and smooth video only with a GPU.
    </p>
    <div class="queue-actions">
      <button id="webcam-start" type="button" class="button">Start camera</button>
      <button id="webcam-stop" type="button" class="button" disabled>Stop camera</button>
      <span id="webcam-fps" class="hint" role="status" aria-live="polite"></span>
    </div>
    <canvas id="webcam-canvas" class="checker webcam-canvas" width="640" height="480"></canvas>
  </section>
```

And add the script alongside the others:

```html
<script src="{{ url_for('static', filename='js/webcam.js') }}" defer></script>
```

- [ ] **Step 4: Write `modnet_bg/web/static/js/webcam.js`**

```javascript
/* Live camera matting. One frame in flight at a time: the next frame is
   captured only after the previous mask arrives, so a slow server drops
   frames rather than building a backlog. */
(() => {
  "use strict";

  const SEND_WIDTH = 512;

  let socket = null;
  let stream = null;
  let video = null;
  let inFlight = false;
  let running = false;
  let lastFrameAt = 0;
  let smoothedFps = 0;

  const $ = (id) => document.getElementById(id);

  function composite(canvas, frame, maskImage) {
    const ctx = canvas.getContext("2d");
    canvas.width = frame.width;
    canvas.height = frame.height;

    const scratch = document.createElement("canvas");
    scratch.width = frame.width;
    scratch.height = frame.height;
    const scratchCtx = scratch.getContext("2d");
    scratchCtx.drawImage(frame, 0, 0);

    const mask = document.createElement("canvas");
    mask.width = frame.width;
    mask.height = frame.height;
    mask.getContext("2d").drawImage(maskImage, 0, 0, frame.width, frame.height);

    const pixels = scratchCtx.getImageData(0, 0, frame.width, frame.height);
    const maskPixels = mask.getContext("2d").getImageData(0, 0, frame.width, frame.height);
    for (let i = 0; i < pixels.data.length; i += 4) {
      pixels.data[i + 3] = maskPixels.data[i]; // red channel of a grayscale PNG
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(pixels, 0, 0);
  }

  function captureFrame() {
    const scale = SEND_WIDTH / video.videoWidth;
    const canvas = document.createElement("canvas");
    canvas.width = SEND_WIDTH;
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  function updateFps() {
    const now = performance.now();
    if (lastFrameAt) {
      const instant = 1000 / (now - lastFrameAt);
      smoothedFps = smoothedFps ? smoothedFps * 0.8 + instant * 0.2 : instant;
      $("webcam-fps").textContent = `${smoothedFps.toFixed(1)} frames per second`;
    }
    lastFrameAt = now;
  }

  function pump() {
    if (!running || inFlight || socket.readyState !== WebSocket.OPEN) return;
    const frame = captureFrame();
    inFlight = true;
    frame.toBlob((blob) => blob && socket.send(blob), "image/jpeg", 0.8);
  }

  async function start() {
    const canvas = $("webcam-canvas");
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280 } });
    } catch {
      $("webcam-fps").textContent = "Camera access was refused.";
      return;
    }

    video = document.createElement("video");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws/realtime`);
    socket.binaryType = "blob";

    socket.addEventListener("open", () => {
      const device = $("device") ? $("device").value : "auto";
      socket.send(JSON.stringify({ device, max_edge: SEND_WIDTH }));
      running = true;
      pump();
    });

    socket.addEventListener("message", async (event) => {
      if (typeof event.data === "string") {
        $("webcam-fps").textContent = JSON.parse(event.data).error || "";
        inFlight = false;
        pump();
        return;
      }
      const mask = await createImageBitmap(event.data);
      composite(canvas, captureFrame(), mask);
      updateFps();
      inFlight = false;
      pump();
    });

    socket.addEventListener("close", stop);

    $("webcam-start").disabled = true;
    $("webcam-stop").disabled = false;
  }

  function stop() {
    running = false;
    inFlight = false;
    lastFrameAt = 0;
    smoothedFps = 0;

    if (stream) {
      // Release the camera. Anything less leaves the indicator light on.
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    if (socket && socket.readyState === WebSocket.OPEN) socket.close();
    socket = null;

    $("webcam-start").disabled = false;
    $("webcam-stop").disabled = true;
    $("webcam-fps").textContent = "";
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      $("webcam-panel").hidden = true;
      return;
    }
    $("webcam-start").addEventListener("click", start);
    $("webcam-stop").addEventListener("click", stop);
    window.addEventListener("beforeunload", stop);
  });
})();
```

- [ ] **Step 5: Add the canvas style**

Append to `app.css`:

```css
.webcam-canvas {
  width: 100%; max-height: 480px; object-fit: contain;
  border-radius: var(--radius); border: 1px solid var(--border); margin-top: 12px;
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_frontend_assets.py -v`
Expected: 26 passed.

- [ ] **Step 7: Verify in a real browser**

The webcam needs a real camera, so this check is manual and cannot be skipped:
- Click **Start camera**, grant permission, and confirm the canvas shows the
  subject cut out against the checkerboard.
- Confirm the fps readout appears and is plausible (low single digits on CPU).
- Click **Stop camera** and confirm the camera indicator light goes out.
- Reload mid-stream and confirm the light goes out (the `beforeunload` handler).
- Deny camera permission and confirm the message reads "Camera access was refused."

- [ ] **Step 8: Commit**

```bash
git add modnet_bg/web tests/test_frontend_assets.py
git commit -m "feat: live camera background removal

Single-frame backpressure, a measured fps readout rather than an implied
one, and every track stopped on exit."
```

---

### Task 17: CPU container

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `tests/test_docker_assets.py`

**Interfaces:**
- Consumes: `pyproject.toml`, `modnet_bg/`.
- Produces: image `modnet-bgremover:cpu`, service `web` on port 8000.

- [ ] **Step 1: Write the failing test**

Create `tests/test_docker_assets.py`:

```python
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text()


def test_dockerfile_exists(dockerfile):
    assert "FROM python:" in dockerfile


def test_runs_as_a_non_root_user(dockerfile):
    # The *last* USER directive is what the container runs as.
    last_user = [line for line in dockerfile.splitlines() if line.startswith("USER ")][-1]
    assert last_user.strip() == "USER appuser"


def test_uses_the_cpu_torch_index(dockerfile):
    assert "download.pytorch.org/whl/cpu" in dockerfile


def test_serves_with_gunicorn_not_the_dev_server(dockerfile):
    assert "gunicorn" in dockerfile
    assert "flask run" not in dockerfile


def test_healthcheck_targets_the_cheap_endpoint(dockerfile):
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile


def test_no_checkpoints_are_baked_in(dockerfile):
    assert ".ckpt" not in dockerfile


def test_dockerignore_excludes_weights_and_git():
    ignored = (ROOT / ".dockerignore").read_text()
    for entry in ("weights", ".git", "output", "assets"):
        assert entry in ignored


def test_compose_mounts_weights_readonly():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "./weights" in compose
    assert ":ro" in compose
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_docker_assets.py -v`
Expected: FAIL with `FileNotFoundError: Dockerfile`

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
# CPU image. The CUDA variant lives in Dockerfile.cuda.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODNET_DATA_DIR=/data \
    MODNET_WEIGHTS_DIR=/weights

# curl is for HEALTHCHECK; ffmpeg comes from the imageio-ffmpeg wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The CPU torch wheel is roughly a quarter the size of the CUDA one.
COPY pyproject.toml ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchvision==0.29.0

COPY modnet_bg ./modnet_bg
RUN pip install .

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data /weights \
 && chown -R appuser:appuser /data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# One worker, threaded: the job pool lives in-process, so a second worker
# would not see the first worker's jobs.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", \
     "--timeout", "0", "modnet_bg.web:create_app()"]
```

- [ ] **Step 4: Write `.dockerignore`**

```
.git
.github
weights
output
assets
docs
tests
**/__pycache__
*.ckpt
.DS_Store
.venv
.claude-flow
.swarm
ruvector.db
```

- [ ] **Step 5: Write `docker-compose.yml`**

```yaml
services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    image: modnet-bgremover:cpu
    ports:
      - "8000:8000"
    environment:
      MODNET_DEVICE: auto
      MODNET_WEIGHTS_DIR: /weights
      SECRET_KEY: ${SECRET_KEY:-change-me-in-production}
    volumes:
      # Checkpoints are not baked into the image. Put them here.
      - ./weights:/weights:ro
      - results:/data
    restart: unless-stopped

  web-cuda:
    profiles: ["cuda"]
    build:
      context: .
      dockerfile: Dockerfile.cuda
    image: modnet-bgremover:cuda
    ports:
      - "8000:8000"
    environment:
      MODNET_DEVICE: auto
      MODNET_WEIGHTS_DIR: /weights
      SECRET_KEY: ${SECRET_KEY:-change-me-in-production}
    volumes:
      - ./weights:/weights:ro
      - results:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  results:
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_docker_assets.py -v`
Expected: 8 passed. `test_compose_mounts_weights_readonly` covers the compose
file; `Dockerfile.cuda` is created in Task 18, so `docker compose --profile cuda
build` will not work until then.

- [ ] **Step 7: Build and smoke-test the image for real**

```bash
docker build -t modnet-bgremover:cpu .
docker run --rm -d --name bgr-smoke -p 8000:8000 \
  -v "$PWD/weights:/weights:ro" modnet-bgremover:cpu
sleep 15
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/api/capabilities
curl -fsS -F "file=@assets/sample_image/male.jpeg" -F "mode=transparent" \
  http://localhost:8000/api/jobs
docker stop bgr-smoke
```

Expected: `/health` returns `{"status":"ok",...}`, capabilities lists `["cpu"]`,
and the upload returns a job id with status `queued` or `running`.

- [ ] **Step 8: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml tests/test_docker_assets.py
git commit -m "feat: CPU container with gunicorn and a non-root user

Checkpoints are bind-mounted, not baked in, so the image stays lean and
the weights stay out of the build cache."
```

---

### Task 18: CUDA container

**Files:**
- Create: `Dockerfile.cuda`
- Modify: `tests/test_docker_assets.py`

**Interfaces:**
- Consumes: the same `pyproject.toml` and package.
- Produces: image `modnet-bgremover:cuda`, compose profile `cuda`.

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_docker_assets.py`:

```python
@pytest.fixture(scope="module")
def cuda_dockerfile() -> str:
    return (ROOT / "Dockerfile.cuda").read_text()


def test_cuda_image_uses_a_cuda_base(cuda_dockerfile):
    assert "nvidia/cuda" in cuda_dockerfile


def test_cuda_image_does_not_use_the_cpu_wheel_index(cuda_dockerfile):
    assert "whl/cpu" not in cuda_dockerfile


def test_cuda_image_is_also_non_root(cuda_dockerfile):
    assert "USER appuser" in cuda_dockerfile


def test_cuda_profile_is_gated_in_compose():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert 'profiles: ["cuda"]' in compose
    assert "capabilities: [gpu]" in compose
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_docker_assets.py -v`
Expected: FAIL with `FileNotFoundError: Dockerfile.cuda`

- [ ] **Step 3: Write `Dockerfile.cuda`**

```dockerfile
# CUDA image, roughly 6GB. Built only via: docker compose --profile cuda build
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODNET_DATA_DIR=/data \
    MODNET_WEIGHTS_DIR=/weights

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3-pip curl \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.12 /usr/local/bin/python

WORKDIR /app

# Default index: the CUDA-enabled wheels.
COPY pyproject.toml ./
RUN python -m pip install torch==2.14.0 torchvision==0.29.0

COPY modnet_bg ./modnet_bg
RUN python -m pip install .

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data /weights \
 && chown -R appuser:appuser /data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", \
     "--timeout", "0", "modnet_bg.web:create_app()"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_docker_assets.py -v`
Expected: 12 passed.

- [ ] **Step 5: Verify the build is reachable**

```bash
docker compose config --profile cuda >/dev/null && echo "compose config valid"
```

Expected: `compose config valid`. **Do not build this image on a machine without
an NVIDIA GPU** — it is a ~6GB pull that cannot be smoke-tested here. CI builds
it; a GPU host runs it.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile.cuda tests/test_docker_assets.py
git commit -m "feat: optional CUDA image behind a compose profile"
```

---

### Task 19: Continuous integration and dependency auditing

**Files:**
- Create: `.github/workflows/ci.yml`, `tests/test_ci_config.py`

**Interfaces:**
- Consumes: `pyproject.toml`, `Dockerfile`, `Dockerfile.cuda`.
- Produces: a CI pipeline with four jobs — `lint`, `test`, `audit`, `images`.

**`pip-audit` is the point of this task.** The security digest that started this
work was stale pins going unnoticed. A scheduled audit is what stops it
recurring, so the weekly `schedule` trigger is not optional.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ci_config.py`:

```python
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow() -> str:
    return WORKFLOW.read_text()


def test_workflow_exists(workflow):
    assert "jobs:" in workflow


@pytest.mark.parametrize("job", ["lint", "test", "audit", "images"])
def test_every_job_is_defined(workflow, job):
    assert f"  {job}:" in workflow


def test_runs_on_push_pr_and_a_schedule(workflow):
    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "schedule:" in workflow, "a scheduled audit is what catches new CVEs"


def test_audit_job_runs_pip_audit(workflow):
    assert "pip-audit" in workflow


def test_tests_run_without_weights(workflow):
    """CI has no checkpoints; the suite must not need them."""
    assert "MODNET_WEIGHTS_DIR" not in workflow


def test_both_images_are_built(workflow):
    assert "Dockerfile.cuda" in workflow
    assert "/health" in workflow, "the CPU image must be smoke-tested, not just built"


def test_python_version_matches_the_package_floor(workflow):
    pyproject = (WORKFLOW.parent.parent.parent / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.12"' in pyproject
    assert "3.12" in workflow
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ci_config.py -v`
Expected: FAIL with `FileNotFoundError: .github/workflows/ci.yml`

- [ ] **Step 3: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    # Weekly. This is what catches a CVE published against a pin we already have.
    - cron: "17 6 * * 1"

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff==0.16.8
      - run: ruff check modnet_bg tests
      - run: ruff format --check modnet_bg tests

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      # CPU wheels only: the runner has no GPU and the CUDA wheel is ~2.5GB.
      - run: pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0
      - run: pip install -e ".[dev]"
      # No checkpoints on the runner. The suite must be green regardless.
      - run: pytest --cov=modnet_bg --cov-report=term-missing

  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0
      - run: pip install -e . pip-audit==2.10.1
      # Audits the resolved environment, so transitive dependencies are covered
      # too -- not just the pins written in pyproject.toml.
      - run: pip-audit --strict --desc

  images:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build the CPU image
        run: docker build -t modnet-bgremover:cpu .
      - name: Smoke-test the CPU image
        run: |
          docker run --rm -d --name bgr -p 8000:8000 modnet-bgremover:cpu
          for attempt in $(seq 1 30); do
            if curl -fsS http://localhost:8000/health; then break; fi
            sleep 2
          done
          curl -fsS http://localhost:8000/health | grep -q '"status":"ok"'
          curl -fsS http://localhost:8000/api/capabilities | grep -q 'cpu'
          docker stop bgr
      - name: Build the CUDA image
        # Built but never run: the runner has no GPU. This catches a broken
        # Dockerfile, which is the failure mode we can actually check here.
        run: docker build -f Dockerfile.cuda -t modnet-bgremover:cuda .
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ci_config.py -v`
Expected: 10 passed (4 parametrised + 6).

- [ ] **Step 5: Run `ruff format` over the package once**

The `lint` job checks formatting, so normalise the tree before the first CI run:

```bash
python -m ruff format modnet_bg tests
python -m ruff check --fix modnet_bg tests
python -m pytest
```
Expected: formatting changes applied, suite still green.

- [ ] **Step 6: Run pip-audit locally**

```bash
python -m pip_audit --strict --desc
```
Expected: no known vulnerabilities. **If any pin reports a CVE, bump it now** and
update both the `pyproject.toml` pin and the Global Constraints table at the top
of this plan. That is the whole point of this task; do not defer it.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/ci.yml tests/test_ci_config.py
git commit -m "ci: lint, test, weekly dependency audit, and image builds

The scheduled pip-audit run is what stops pins going stale again."
```

---

### Task 20: README, demo assets, and end-to-end verification

**Files:**
- Rewrite: `README.md`
- Decide: `output/male.png`, `output/sample.gif`, `output/sample.mp4`, `output/web_view.png`
- Create: `tests/test_readme.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the repository's public documentation.

**Open decision, carried from the design review.** `output/` holds four tracked
demo files (~7.7MB total), all produced by the deleted CLI in the old
side-by-side format that this rebuild no longer emits. `web_view.png` shows the
old Bootstrap page. Three options — **ask the user before doing this step**:

1. **Regenerate** — run the new UI, screenshot it, replace all four. Accurate
   docs, repository size unchanged. *This is the recommended default.*
2. **Delete and reference nothing** — smallest repository, plainest README.
3. **Keep as-is** — fastest, but the README would show output the software can
   no longer produce.

Do not silently pick one. If the user is unavailable, take option 1 and say so.

- [ ] **Step 1: Write the failing test**

Create `tests/test_readme.py`:

```python
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


def test_no_references_to_the_deleted_cli(readme):
    for gone in ("inference.py", "bg_remove.py", "--webcam", "--ckpt_image", "web_solution"):
        assert gone not in readme, f"README still documents the removed {gone}"


def test_documents_how_to_run_it(readme):
    assert "docker compose up" in readme
    assert "http://localhost:8000" in readme


def test_documents_the_weights_requirement(readme):
    assert "MODNET_WEIGHTS_DIR" in readme
    assert "weights/" in readme


def test_documents_the_gpu_toggle(readme):
    assert "MODNET_DEVICE" in readme
    for device in ("cuda", "mps", "cpu"):
        assert device in readme


def test_every_referenced_local_image_exists(readme):
    import re

    for match in re.finditer(r"!\[[^\]]*\]\(([^)h][^)]*)\)", readme):
        target = README.parent / match.group(1).split()[0]
        assert target.exists(), f"README references a missing image: {match.group(1)}"


def test_documents_the_webcam_performance_expectation(readme):
    """Users should not discover 3fps by surprise."""
    assert "fps" in readme.lower() or "frames per second" in readme.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_readme.py -v`
Expected: FAIL — the current README documents `inference.py` and the CLI flags.

- [ ] **Step 3: Resolve the `output/` decision**

Ask the user, then act. For option 1:

```bash
# with the server running and a sample processed in the browser
git rm output/sample.gif output/sample.mp4 output/web_view.png output/male.png
# add the new screenshot(s) under docs/images/
mkdir -p docs/images
```

For option 2:

```bash
git rm -r output
```

- [ ] **Step 4: Rewrite `README.md`**

````markdown
# MODNet Background Remover

Remove the background from photos and videos in your browser. Everything runs
locally — nothing is uploaded anywhere.

- Images and video, plus live camera capture
- Transparent, solid colour, blurred, custom image, or raw alpha matte output
- Before/after comparison slider
- Batch upload with a zip download
- Runs on an NVIDIA GPU, an Apple GPU, or the CPU, switchable from the UI

## Requirements

The MODNet checkpoints are not distributed with this repository. Download
`modnet_photographic_portrait_matting.ckpt` and
`modnet_webcam_portrait_matting.ckpt` and put them in `weights/`:

```
weights/
  modnet_photographic_portrait_matting.ckpt
  modnet_webcam_portrait_matting.ckpt
```

Both files are verified against a pinned SHA-256 before they are loaded. A file
that does not match is rejected rather than used.

## Run it with Docker

```bash
docker compose up --build
```

Open <http://localhost:8000>.

On a machine with an NVIDIA GPU:

```bash
docker compose --profile cuda up --build web-cuda
```

The CUDA image is roughly 6GB; the default CPU image is roughly 1.5GB.

## Run it without Docker

```bash
pip install -e ".[dev]"
python -m flask --app "modnet_bg.web:create_app()" run --port 8000
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODNET_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. `auto` prefers cuda, then mps, then cpu |
| `MODNET_WEIGHTS_DIR` | `./weights` | where the checkpoints live |
| `PORT` | `8000` | listen port |
| `SECRET_KEY` | generated | set this in production |
| `MODNET_MAX_IMAGE_MB` | `25` | image upload cap |
| `MODNET_MAX_VIDEO_MB` | `250` | video upload cap |
| `MODNET_MAX_VIDEO_SECONDS` | `120` | video duration cap |
| `MODNET_MAX_WORKERS` | `2` | concurrent jobs |
| `MODNET_JOB_TTL_SECONDS` | `3600` | how long results are kept |

## About the GPU toggle

The device selector in the UI lists only what this machine actually has. Picking
**CPU** forces processing onto the CPU even when a GPU is present, which is
useful for comparison or when the GPU is busy. An unavailable choice falls back
rather than failing, and each result reports which device processed it.

## About the live camera

Camera matting runs one frame at a time through the model. Expect roughly **2–5
frames per second on a CPU** and 15–30 on CUDA — the readout under the canvas
shows the real measured rate. It is genuinely slow without a GPU.

## Development

```bash
pip install -e ".[dev]"
pytest                          # runs without any checkpoint present
ruff check modnet_bg tests
pip-audit --strict
```

The test suite substitutes a fake model, so it needs neither the checkpoints nor
a GPU.

## Licence and credit

MODNet is by Zhanghan Ke et al. The network in `modnet_bg/models/` is upstream
code, vendored unchanged.
````

Adjust the screenshot references to match the Step 3 decision.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_readme.py -v`
Expected: 6 passed.

- [ ] **Step 6: Full-suite verification**

```bash
python -m pytest -v
python -m ruff check modnet_bg tests
python -m ruff format --check modnet_bg tests
python -m pip_audit --strict
```
Expected: every test passing, no lint findings, no vulnerabilities.

- [ ] **Step 7: Verify the suite really is checkpoint-free**

This is the claim the whole test strategy rests on, so prove it rather than
assuming it:

```bash
mv weights weights.hidden
python -m pytest
mv weights.hidden weights
```
Expected: identical pass count. If anything fails or skips, a test has grown a
hidden dependency on real weights — fix the test, do not mark it skipped.

- [ ] **Step 8: End-to-end verification in a browser**

With `docker compose up --build` running, confirm all of:
- An image upload produces a transparent PNG with no white fringe on hair edges.
- **Solid colour**, **blur**, **custom image**, and **matte** each produce the
  expected output.
- A video upload shows per-frame progress, and the result plays at the source
  speed, at the source size, with its audio intact.
- Cancelling a running video job stops it promptly.
- Batch-uploading three files processes all three and the zip contains three.
- The device selector switches between CPU and the GPU, and the per-result
  device readout changes accordingly.
- The live camera cuts out correctly and stops the camera light on **Stop**.
- Uploading a `.txt` file renamed to `.png` is rejected with a clear message.

- [ ] **Step 9: Commit**

```bash
git add README.md tests/test_readme.py docs/images output
git commit -m "docs: rewrite README for the web-only application"
```

- [ ] **Step 10: Open the pull request**

```bash
git push -u origin feat/web-only-rebuild
gh pr create --title "Web-only rebuild: tested, containerized, GPU-switchable" \
  --body "Implements docs/superpowers/specs/2026-09-17-web-bgremover-design.md"
```

---

## Requirements Traceability

Each spec requirement mapped to the task that implements it. Anything not in
this table is out of scope.

| Spec section | Requirement | Task |
|---|---|---|
| 2 | Delete the CLI | 1 |
| 2 | Image background removal | 4, 5, 10 |
| 2 | Video background removal | 8, 10 |
| 2 | Five output modes | 5, 10 |
| 2 | Comparison slider | 14 |
| 2 | Batch upload and zip | 12, 13 |
| 2 | Live webcam | 15, 16 |
| 2 | GPU auto-detect and toggle | 2, 11, 13 |
| 2 | Dependency upgrade and hardening | 1, 19 |
| 2 | Checkpoint-free test suite | 4, 20 |
| 2 | Docker CPU and CUDA, CI | 17, 18, 19 |
| 5.1 | `resolve_device` never raises | 2 |
| 5.1 | MPS falls back to CPU | 2, 4 |
| 5.2 | SHA-256 pinned checkpoints | 3 |
| 5.2 | Atomic download, no partial files | 3 |
| 5.3 | Stateless engine | 4 |
| 5.3 | `weights_only=True` | 1, 4 |
| 5.3 | `DataParallel` dropped, prefix stripped | 4 |
| 5.3 | Stride clamp for extreme ratios | 4 |
| 5.3 | `inference_mode()` | 4 |
| 5.4 | Straight alpha, no white halo | 5 |
| 5.4 | Cover-fit backgrounds | 5 |
| 5.5 | Magic-byte typing, bomb guard | 6 |
| 5.6 | Source fps preserved | 8 |
| 5.6 | Container follows mode | 8, 10 |
| 5.6 | Audio muxed onto MP4 | 8 |
| 5.7 | Job registry, cancel, TTL | 9 |
| 5.8 | Model-free `create_app()` | 11 |
| 5.8 | Every endpoint | 11, 12 |
| 5.8 | UUID-addressed results | 10, 12 |
| 5.8 | JSON-only errors, 413/422 | 11, 12 |
| 5.9 | Local assets, no CDN | 13 |
| 5.9 | Accessibility | 13, 14 |
| 5.10 | Single frame in flight | 15, 16 |
| 5.10 | Measured fps displayed | 16 |
| 6 | Every environment variable | 7 |
| 7 | No `flask-cors` | 11 |
| 7 | Non-root container, gunicorn | 17, 18 |
| 7 | `pip-audit` in CI | 19 |
| 8 | Concurrency regression test | 4 |
| 9 | Weights not baked into images | 17, 18 |

## Known deviations from the spec

Both were discovered while checking facts for this plan, and both are
improvements rather than reductions in scope. They are called out here so a
reviewer can reject them.

1. **OpenCV is dropped entirely** (spec 5.6 named `cv2.VideoCapture`). OpenCV
   has moved to 5.0.0.93 with breaking API changes, it is a ~60MB wheel, and it
   is a recurring CVE source. Pillow and imageio-ffmpeg cover everything the app
   needs, and the vendored model code imports only `torch`.
2. **Weights are local-first, not download-on-first-run** (spec 5.2). No
   reliable public direct URL exists: upstream distributes via Google Drive,
   which serves an HTML interstitial, and the candidate HuggingFace and
   GitHub-release mirrors return 401 and 404. Downloading now requires an
   explicit `MODNET_WEIGHTS_URL_<KEY>`; otherwise the user places the file and
   gets a clear error if they have not.
