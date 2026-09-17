# MODNet-BGRemover

Web-only background remover for photos and video. No CLI.

## Commits

- One short line, imperative mood. No body unless the change genuinely needs one.
- **No trailers.** No `Co-Authored-By`, no generated-with attribution.
- Never commit to `main` directly; branch first.

## Current work

Rebuild in progress on `feat/web-only-rebuild`:

- Spec: `docs/superpowers/specs/2026-09-17-web-bgremover-design.md`
- Plan: `docs/superpowers/plans/2026-09-17-web-only-rebuild.md` (20 TDD tasks)

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

## Commands

```bash
pip install -e ".[dev]"
pytest
ruff check modnet_bg tests
docker compose up --build        # http://localhost:8000
```
