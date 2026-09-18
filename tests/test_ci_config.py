from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


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
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.12"' in pyproject
    assert "3.12" in workflow


def test_workflow_is_valid_yaml(workflow):
    import tomllib  # noqa: F401  (stdlib availability check for the runner)

    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    parsed = yaml.safe_load(workflow)
    assert set(parsed["jobs"]) == {"lint", "test", "audit", "images"}
