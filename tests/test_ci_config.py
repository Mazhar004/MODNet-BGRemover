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


def test_audit_strips_local_version_segments(workflow):
    """On Linux the CPU wheel installs as torch==2.14.0+cpu, and that exact
    string exists only on download.pytorch.org. pip-audit resolves against
    PyPI, so the frozen file must drop the +local segment or the job fails
    with 'No matching distribution found for torch==2.14.0+cpu'."""
    audit = workflow.split("  audit:")[1].split("\n  images:")[0]
    # Match the command, not the word: the explanatory comment above it also
    # says "sed", which made an earlier version of this assertion vacuous.
    command_lines = [
        line for line in audit.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    body = "\n".join(command_lines)
    assert "sed -E" in body, "frozen requirements are not stripped of +local segments"
    assert "[A-Za-z0-9.]*$" in body, "the local-segment pattern is missing"
    assert "requirements-audit.txt" in body


def test_audit_keeps_strict(workflow):
    """Without --strict a dependency that cannot be resolved is a warning, and
    the job would go green having audited nothing."""
    assert "pip-audit --strict" in workflow


def test_ci_upgrades_setuptools_before_auditing(workflow):
    """The runner preinstalls setuptools 78.1.0, which has two advisories. The
    audit job would otherwise fail on the runner's own tooling."""
    audit = workflow.split("  audit:")[1].split("\n  images:")[0]
    assert "setuptools>=83" in audit
