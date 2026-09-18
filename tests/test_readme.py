import re
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
    for match in re.finditer(r"!\[[^\]]*\]\(([^)h][^)]*)\)", readme):
        target = README.parent / match.group(1).split()[0]
        assert target.exists(), f"README references a missing image: {match.group(1)}"


def test_documents_the_webcam_performance_expectation(readme):
    """Users should not discover 3fps by surprise."""
    assert "fps" in readme.lower() or "frames per second" in readme.lower()


def test_stale_demo_assets_are_gone():
    """output/ held four files produced by the deleted CLI, in a side-by-side
    format this build no longer emits. male.png was in fact a JPEG."""
    stale = README.parent / "output"
    assert not stale.exists(), "stale CLI-era demo assets still present"
