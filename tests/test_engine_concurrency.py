"""Concurrency safety of the matting engine on the real compute device.

These run in a subprocess on purpose. The failure mode being guarded against
is a SIGABRT/SIGSEGV inside PyTorch, which would take the pytest process down
with it and report nothing useful. Checking a child's exit code is the only
way to assert "did not crash".
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

SCRIPT = """
import concurrent.futures as cf
from pathlib import Path

import numpy as np

from modnet_bg.engine import MattingEngine
from modnet_bg.weights import resolve_weights

device = {device!r}
ckpt = resolve_weights("photographic", search_dirs=[Path({weights!r})])
engine = MattingEngine.from_checkpoint(ckpt, device)

images = [np.full((480, 640, 3), i * 7 % 256, dtype=np.uint8) for i in range(12)]
with cf.ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(engine.matte, images))

assert len(results) == 12
assert all(r.shape == (480, 640) for r in results)
print("OK")
"""


def _run(device: str) -> subprocess.CompletedProcess:
    script = SCRIPT.format(device=device, weights=str(WEIGHTS))
    # The command is this file's own constant plus a repo path; nothing here
    # comes from a request or the environment.
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=ROOT,
        check=False,
    )


@pytest.mark.needs_weights
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_concurrent_inference_does_not_crash_the_process(device):
    """Regression: MPS is not thread-safe.

    Six threads through one shared engine used to abort with exit 134 and a
    Metal assertion ("Unable to reach MTLCompilerService"). Under gunicorn that
    killed the worker, and macOS Obj-C fork safety then prevented every respawn
    from booting -- so a single concurrent upload burst bricked the server until
    it was restarted by hand.
    """
    if not WEIGHTS.is_dir() or not list(WEIGHTS.glob("*.ckpt")):
        pytest.skip("real checkpoints not present")

    import torch

    if device == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            pytest.skip("no MPS device on this machine")

    result = _run(device)

    assert result.returncode == 0, (
        f"{device} concurrency crashed with exit {result.returncode}"
        f"{' (SIGABRT)' if result.returncode == 134 else ''}"
        f"{' (SIGSEGV)' if result.returncode == 139 else ''}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )
    assert "OK" in result.stdout


def test_mps_work_is_serialised():
    """The guard must stay. Without it the crash above returns."""
    source = (ROOT / "modnet_bg" / "engine.py").read_text()
    assert "_mps_lock" in source
    assert 'if self._device == "mps":' in source, "MPS forward passes are no longer serialised"
