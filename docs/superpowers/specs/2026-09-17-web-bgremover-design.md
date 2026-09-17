# MODNet-BGRemover: Web-Only Rebuild — Design

**Date:** 2026-09-17
**Status:** Approved for implementation planning

## 1. Problem

The repository is at its original upstream state. It carries three problems that
compound each other:

1. **Stale, vulnerable dependencies.** Every pin is four to five years old and
   generates GitHub security alerts.
2. **A codebase split between a CLI and a fragile web demo.** `inference.py` and
   `bg_remove.py` serve a command-line workflow that is no longer wanted, and
   `api.py` plus `web_solution/` form a demo-grade Flask app with a wildcard CORS
   policy, a hardcoded secret, and a route that interpolates user-supplied
   filenames into paths.
3. **No tests, no container, no GPU control.**

The goal is a single, tested, containerized web application that lets a person
remove the background from a photo or a video interactively, with the GPU used
when present and switchable off.

## 2. Scope

**In scope**

- Delete the CLI entirely. The web UI is the only interface.
- Image and video background removal.
- Output modes: transparent, solid color, blurred original, custom background
  image, and raw alpha matte.
- Before/after comparison slider.
- Batch upload with per-file status and a zip download.
- Live webcam capture in the browser.
- GPU auto-detection (CUDA / Apple MPS / CPU) with an explicit on/off control.
- Dependency upgrade to current versions and code-level security hardening.
- A pytest suite that runs without model checkpoints and without a GPU.
- Docker images (CPU default, CUDA optional) and CI.

**Out of scope**

- Training or fine-tuning MODNet. Inference only.
- User accounts, persistence beyond job TTL, or multi-tenancy.
- Changes to the MODNet network architecture itself.

## 3. Approach

A new `modnet_bg/` package replaces the existing entry points. `src/models/`
moves in verbatim — it is upstream MODNet and is not the problem. Everything
else is rewritten.

The alternative considered was refactoring `bg_remove.py` in place. It was
rejected because `BGRemove` stores per-request data on `self` and mixes matting,
compositing, file I/O, video writing, and OpenCV display windows in a single
class. That statefulness is incompatible with a worker pool and has to be
dismantled regardless, so refactoring in place would produce the same rewrite
with a worse file layout.

## 4. Structure

```
modnet_bg/
  __init__.py
  models/            # moved from src/models, unchanged
    modnet.py
    backbones/
  device.py          # GPU detection and policy
  weights.py         # download, SHA-256 verify, cache
  engine.py          # stateless matting
  compositing.py     # output modes
  media.py           # decode, validate, guard
  video.py           # frame I/O, fps and audio preservation
  jobs.py            # job registry and worker pool
  config.py          # env-driven settings
  web/
    __init__.py      # create_app() factory
    routes.py        # HTTP API
    realtime.py      # webcam WebSocket
    templates/index.html
    static/css/app.css
    static/js/app.js
tests/
Dockerfile
Dockerfile.cuda
docker-compose.yml
pyproject.toml
```

**Removed:** `api.py`, `bg_remove.py`, `inference.py`, `web_solution/`, `src/`,
`pretrained/`, `requirements.txt`, `web_requirements.txt`.

## 5. Components

### 5.1 `device.py`

```python
def available_devices() -> list[str]        # subset of ["cuda", "mps", "cpu"]
def resolve_device(pref: str) -> str        # pref in {auto, cuda, mps, cpu}
```

`resolve_device` never raises on an unavailable preference; it clamps to the
best available device and the caller reports what was actually used. Precedence
for the default: explicit per-request value, then the `MODNET_DEVICE`
environment variable, then `auto`. Under `auto` the order is CUDA, MPS, CPU.

MPS is known to lack coverage for some operators. Inference on MPS is wrapped so
that an unsupported-operator error falls back to CPU once, logs a warning, and
records the fallback on the job.

### 5.2 `weights.py`

Two checkpoints, pinned by SHA-256:

| File | SHA-256 |
|---|---|
| `modnet_photographic_portrait_matting.ckpt` | `7c22235f0925deba15d4d63e53afcb654c47055bbcd98f56e393ab2584007ed8` |
| `modnet_webcam_portrait_matting.ckpt` | `913b82b66558db39b6286c150f809017d7528c872b156eb14333c9c6cb52108b` |

Resolution order: a path given by `MODNET_WEIGHTS_DIR`, then a local
`weights/` directory, then download into the cache directory. `weights/` is
gitignored and is a developer convenience only; nothing in the repository or the
images ships a checkpoint. Every path
verifies the digest before the file is used; a mismatch is a hard error, never a
warning. Downloads validate content type and size before writing, and write
atomically through a temporary file so an interrupted download cannot leave a
file that later passes a partial read.

### 5.3 `engine.py`

```python
class MattingEngine:
    def __init__(self, weights_path: Path, device: str) -> None
    def matte(self, rgb: np.ndarray) -> np.ndarray   # float32 [H, W] in 0..1
```

Stateless with respect to requests: no per-image data is stored on the instance.
All intermediate values are locals; the alpha matte is the return value.

Fixes carried in from the current implementation:

- `torch.load(..., weights_only=True)` — closes the arbitrary-code-execution
  path through a malicious checkpoint.
- `nn.DataParallel` dropped. The `module.` key prefix is stripped at load time.
- Resize dimensions clamped to a minimum of 32 after the stride rounding, so
  extreme aspect ratios such as 16x600 cannot floor a dimension to zero.
- Forward passes run under `torch.inference_mode()`.

One engine instance per (weights, device) pair, created lazily behind a lock and
cached. The model is in `eval()` mode and is not mutated after construction, so
concurrent forward passes from the worker pool share it safely.

### 5.4 `compositing.py`

```python
def composite(rgb, alpha, mode, *, color=None, background=None, blur_radius=None)
    -> RGBA or RGB array
```

Modes: `transparent`, `color`, `blur`, `image`, `matte`.

`transparent` keeps **straight (unpremultiplied)** RGB alongside the alpha
channel. The current implementation composites onto white and also writes the
alpha, which fringes every semi-transparent edge pixel white. A regression test
covers this directly.

`image` resizes the supplied background with cover semantics (aspect preserved,
center-cropped) rather than stretching it.

### 5.5 `media.py`

Decode and validation for uploads. Type is determined by magic bytes, not by the
filename extension. Guards: a maximum pixel count to stop decompression bombs,
`Image.MAX_IMAGE_PIXELS` set accordingly, and separate byte caps for images and
videos. Failures raise typed errors that the web layer maps to 415 or 422.

### 5.6 `video.py`

Reading via `cv2.VideoCapture`. Writing via `imageio-ffmpeg`.

- **Source fps is read and preserved.** The current writer is hardcoded to
  `20.0`, which changes playback speed for any other source.
- Output dimensions match the source. The existing side-by-side double-width
  output is a demo artifact and is dropped.
- Container follows the output mode, and is not independently selectable:
  `transparent` requires an alpha channel and so produces VP9 in WebM; every
  other mode produces H.264 in MP4. The UI defaults video to a non-transparent
  mode, so MP4 is what a user gets unless they ask for alpha.
- The original audio track is muxed onto MP4 output in a single ffmpeg call.
  WebM alpha output is video-only.
- Frame callback per frame so the job can report progress and observe
  cancellation.
- Configurable caps on duration and total frames.

### 5.7 `jobs.py`

```python
@dataclass
class Job:
    id: str
    status: Literal["queued", "running", "done", "error", "cancelled"]
    progress: float
    frames_done: int
    frames_total: int | None
    device_used: str | None
    result_path: Path | None
    error: str | None
```

A `JobRegistry` holds jobs in a dict behind a lock, backed by a bounded
`ThreadPoolExecutor`. Each job carries a `threading.Event` for cancellation,
checked between frames. A reaper thread deletes jobs and their files past a TTL.

Batch upload creates one job per file. There is no batch entity; the client
tracks the set of ids and requests a zip of the completed ones.

### 5.8 `web/`

An application factory, `create_app()`, that imports cleanly without loading a
model — required for gunicorn and for the test suite. The model loads lazily on
first use behind a double-checked lock.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | single page |
| GET | `/health` | liveness; never loads the model |
| GET | `/api/capabilities` | devices, limits, supported formats |
| POST | `/api/jobs` | upload plus options, returns `{job_id}` |
| GET | `/api/jobs/<id>` | status |
| GET | `/api/jobs/<id>/result` | download |
| POST | `/api/jobs/zip` | zip of several completed job results |
| DELETE | `/api/jobs/<id>` | cancel |
| WS | `/ws/realtime` | webcam frames |

Results are addressed by a server-generated UUID. No user-supplied string ever
reaches a filesystem path, which removes the injection shape of the current
`GET /process/<input_filename>`.

Errors return JSON only and never leak internal detail. Handlers exist for 413
and 422 so an oversized or unreadable upload produces a usable message rather
than an HTML stack page.

### 5.9 Frontend

One page, vanilla JavaScript and modern CSS, no build step. Assets are served
locally — the current page pulls Bootstrap, jQuery, Popper, and axios from three
CDNs, which is both a supply-chain exposure and a hard dependency on outbound
network access from the browser.

Panels: upload (drag-and-drop, multi-file), options (output mode, background
color or image, device toggle), results (comparison slider over a checkerboard
backdrop, per-file progress, download), and webcam.

Accessibility is a requirement, not a polish item: keyboard-operable controls,
`aria-live` on progress and error regions, visible focus, and alt text.

### 5.10 `realtime.py`

WebSocket via `flask-sock`. The browser captures with `getUserMedia`, downscales
to 512px on a canvas, and sends a JPEG. The server returns the alpha mask; the
client composites against the selected background locally, which keeps the
return payload to one channel.

Exactly one frame is in flight at a time — the client sends the next only after
the previous reply. A slow server therefore drops frames rather than
accumulating a backlog.

Expected throughput is roughly 2–5 fps on CPU and 15–30 fps on CUDA. The UI
displays the measured rate so the behavior is legible rather than looking broken.

## 6. Configuration

All via environment, read in `config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `MODNET_DEVICE` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `MODNET_WEIGHTS_DIR` | unset | override weights location |
| `PORT` | `8000` | listen port |
| `SECRET_KEY` | generated | Flask secret |
| `MODNET_MAX_IMAGE_MB` | `25` | image upload cap |
| `MODNET_MAX_VIDEO_MB` | `250` | video upload cap |
| `MODNET_MAX_VIDEO_SECONDS` | `120` | video duration cap |
| `MODNET_MAX_WORKERS` | `2` | job pool size |
| `MODNET_JOB_TTL_SECONDS` | `3600` | result retention |

A generated `SECRET_KEY` is logged as a warning, because it invalidates sessions
across restarts and should be set explicitly in production.

## 7. Security

The upgrade is not optional and is not gated on further discussion:

- All dependencies resolved to current versions and pinned at implementation
  time, rather than assumed now.
- `weights_only=True` on every `torch.load`.
- SHA-256 verification of all checkpoints.
- `flask-cors` removed. The app is same-origin; the wildcard policy was
  gratuitous.
- `SECRET_KEY` from the environment.
- Magic-byte type validation, pixel-count bomb guard, separate size caps.
- UUID-addressed results; no user input in paths.
- Container runs as a non-root user; gunicorn; debug mode off.
- `pip-audit` in CI so the alert digest stays quiet.

## 8. Testing

`pytest`. A `FakeMatteModel` fixture substitutes for MODNet so the **entire suite
runs with no checkpoint and no GPU** — this is what makes it viable in CI.

- **Device:** resolution for each preference against each availability
  combination; MPS fallback path.
- **Engine:** output shape and range; extreme aspect ratios including 16x600;
  `weights_only` is passed; `module.` prefix stripping.
- **Compositing:** each mode; a direct regression test asserting that
  transparent output leaves edge RGB unfringed; cover-semantics resizing.
- **Weights:** digest mismatch is fatal; partial download does not yield a
  usable file; atomic replace.
- **Media:** magic-byte detection; extension/content mismatch rejected; pixel
  bomb rejected; size caps.
- **Video:** synthetic 10-frame clip; fps preserved; progress callback fires per
  frame; cancellation stops mid-clip.
- **Jobs:** full lifecycle; cancel while running; TTL reaping; pool bound
  respected.
- **Web:** every endpoint via the Flask test client, success and failure paths,
  413 and 422 handlers, poll-to-completion, `create_app()` imports without
  loading a model, `/health` never loads a model.
- **Realtime:** WebSocket round-trip.
- **Concurrency:** N threads through the engine simultaneously, asserting no
  cross-contamination of dimensions or results. This is the regression test for
  the statefulness that forced the rewrite.

CI additionally builds both Docker images and runs a `/health` smoke test
against the CPU image.

## 9. Docker

`Dockerfile` — a `python:3.x-slim` base, CPU-only torch wheel, roughly 1.5GB,
gunicorn, non-root user, `HEALTHCHECK` against `/health`. The exact minor version
is chosen in phase 4 against torch wheel availability; `3.12` is the expected
answer and `3.13` (the local development version) is preferred if wheels for the
full dependency set exist for it.

`Dockerfile.cuda` — CUDA base with the matching torch build, behind a
`docker-compose` profile so the default `up` does not pull six gigabytes.

Weights are **not** baked into either image. They download on first run into a
named volume and are checksum-verified, keeping the images lean and the
checkpoints out of the build cache.

## 10. Implementation phases

1. Package skeleton, `device.py`, `weights.py`, `engine.py`, `compositing.py`,
   `media.py`, and their tests. Old CLI entry points deleted.
2. `jobs.py`, `video.py`, `config.py`, `web/` HTTP API, and their tests.
3. Frontend: upload, options, progress, comparison slider, batch, downloads.
4. Webcam realtime, Docker (both images), compose, CI, README rewrite.

Each phase ends with a green suite.

## 11. Risks

- **Webcam throughput on CPU is genuinely poor** (2–5 fps). Mitigated by
  displaying the measured rate and by single-frame-in-flight backpressure, but
  the underlying cost is the model's.
- **MPS operator coverage** varies by torch release. Mitigated by the
  tested CPU fallback path.
- **VP9 alpha encoding** is slower than H.264 and not playable everywhere.
  Mitigated by defaulting the video output mode to a non-transparent one, so
  alpha WebM is produced only on explicit request.
- **numpy 2.x and torch 2.x** may surface incompatibilities in the vendored
  MODNet code. Phase 1 exists partly to find that early.
