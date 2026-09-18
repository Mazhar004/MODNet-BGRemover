# MODNet-BGRemover

Web-only background remover for photos and video. No CLI.

## Commits

- One short line, imperative mood. No body unless the change genuinely needs one.
- **No trailers.** No `Co-Authored-By`, no generated-with attribution.
- Never commit to `main` directly; branch first.

## Current work

Rebuild complete on `feat/web-only-rebuild`. 192 tests, no known vulnerabilities.

- Spec: `docs/superpowers/specs/2026-09-17-web-bgremover-design.md`
- Plan: `docs/superpowers/plans/2026-09-17-web-only-rebuild.md`

## Invariants

These are decisions, not preferences. Changing one needs a reason.

- **No OpenCV.** Pillow for image ops, imageio-ffmpeg for video. OpenCV 5.x broke
  its API, is a ~60MB wheel, and is a recurring CVE source.
- **No `flask-cors`.** The app is same-origin.
- **Every `torch.load` passes `weights_only=True`.**
- **The matting engine stays stateless.** Per-request data on `self` is what
  corrupted concurrent uploads in the old `BGRemove` class.
- **Transparent output keeps straight (unpremultiplied) alpha.** Compositing onto
  white and also writing alpha is what caused the white edge halo.
- **`create_app()` must not load a model.** Gunicorn and the tests depend on it.
- **The test suite must pass with no checkpoint and no GPU.** Tests substitute a
  `FakeMatteModel`. A test that needs real weights must skip, never fail CI.
- **Checkpoints are never committed or baked into an image.** `weights/` is
  gitignored and bind-mounted.
- **`modnet_bg/models/` is vendored upstream code.** Excluded from both the
  linter and the formatter so upstream diffs stay readable. Do not restyle it.
- **Media type comes from magic bytes, and image-vs-video from frame count.**
  A one-frame GIF is a picture; a sixty-frame GIF is a video.

## Commands

```bash
pip install -e ".[dev]"
pytest
ruff check modnet_bg tests && ruff format --check modnet_bg tests
docker compose up --build        # http://localhost:8000
```

Dependency audit needs the frozen-env form. Auditing the environment directly
makes `--strict` fail on our own package, which is not on PyPI; the `sed` strips
PEP 440 local segments, because the Linux CPU wheel is `torch==2.14.0+cpu` and
that exact string exists only on download.pytorch.org:

```bash
pip freeze --exclude-editable | sed -E 's/\+[A-Za-z0-9][A-Za-z0-9.]*$//' > /tmp/reqs.txt
pip-audit --strict --desc -r /tmp/reqs.txt
```
