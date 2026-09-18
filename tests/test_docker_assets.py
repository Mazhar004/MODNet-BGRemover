from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text()


@pytest.fixture(scope="module")
def cuda_dockerfile() -> str:
    return (ROOT / "Dockerfile.cuda").read_text()


@pytest.fixture(scope="module")
def compose() -> str:
    return (ROOT / "docker-compose.yml").read_text()


def test_dockerfile_exists(dockerfile):
    assert "FROM python:" in dockerfile


def test_runs_as_a_non_root_user(dockerfile):
    # The *last* USER directive is what the container actually runs as.
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


def test_single_worker_because_the_job_registry_is_in_process(dockerfile):
    """Two workers would each hold their own registry, so a poll could hit a
    worker that has never heard of the job."""
    assert "--workers" in dockerfile
    tokens = dockerfile.split()
    index = tokens.index('"--workers",')
    assert tokens[index + 1].strip('",') == "1"


def test_dockerignore_excludes_weights_and_git():
    ignored = (ROOT / ".dockerignore").read_text()
    for entry in ("weights", ".git", "output", "assets"):
        assert entry in ignored


def test_compose_mounts_weights_readonly(compose):
    assert "./weights" in compose
    assert ":ro" in compose


def test_cuda_image_uses_a_cuda_base(cuda_dockerfile):
    assert "nvidia/cuda" in cuda_dockerfile


def test_cuda_image_does_not_use_the_cpu_wheel_index(cuda_dockerfile):
    assert "whl/cpu" not in cuda_dockerfile


def test_cuda_image_is_also_non_root(cuda_dockerfile):
    assert "USER appuser" in cuda_dockerfile


def test_cuda_profile_is_gated_in_compose(compose):
    assert 'profiles: ["cuda"]' in compose
    assert "capabilities: [gpu]" in compose
