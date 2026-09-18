"""The native entry point must not fork, because Metal cannot survive it."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "modnet_bg" / "__main__.py"


def test_entrypoint_exists_and_is_runnable():
    assert MAIN.is_file()
    assert "def main()" in MAIN.read_text()


def test_entrypoint_uses_a_threaded_non_forking_server():
    """Regression: gunicorn forks, and a forked worker cannot reach
    MTLCompilerService after it idles out. Measured: with a 240s idle before
    the first MPS request, gunicorn aborted and this path completed."""
    source = MAIN.read_text()
    assert "threaded=True" in source
    assert "gunicorn" not in source.replace("# ", "").split('"""')[2]


def test_entrypoint_explains_why_not_gunicorn():
    assert "MTLCompilerService" in MAIN.read_text()


def test_docker_still_uses_gunicorn():
    """Linux has no Metal, so forking is fine and gunicorn stays correct there."""
    assert "gunicorn" in (ROOT / "Dockerfile").read_text()
    assert "gunicorn" in (ROOT / "Dockerfile.cuda").read_text()


def test_readme_documents_the_native_command():
    assert "python -m modnet_bg" in (ROOT / "README.md").read_text()
