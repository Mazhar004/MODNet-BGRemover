import threading
import time

import pytest

from modnet_bg.errors import JobNotFoundError
from modnet_bg.jobs import JobRegistry


@pytest.fixture
def registry():
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


def test_domain_errors_keep_their_user_facing_message(registry):
    from modnet_bg.errors import MediaTooLargeError

    def too_big(job):
        raise MediaTooLargeError("Video is 300s, longer than the 120s limit.")

    job = registry.submit(too_big, kind="video", filename="v.mp4")
    _wait(job)

    assert job.status == "error"
    assert "longer than the 120s limit" in job.error


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

    def blocked(job):
        blocker.wait(5)
        return None, ""

    for _ in range(2):
        registry.submit(blocked, kind="image", filename="b")

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


def test_reap_leaves_running_jobs_alone(tmp_path):
    registry = JobRegistry(max_workers=1, ttl_seconds=0)
    release = threading.Event()
    try:

        def slow(job):
            release.wait(5)
            return None, ""

        job = registry.submit(slow, kind="video", filename="v.mp4")
        time.sleep(0.05)
        assert registry.reap() == 0, "reaped a job that had not finished"
        release.set()
        _wait(job)
    finally:
        registry.shutdown()


def test_to_dict_is_json_safe_and_hides_paths(registry, tmp_path):
    import json

    output = tmp_path / "out.png"
    output.write_bytes(b"x")
    job = registry.submit(lambda j: (output, "out.png"), kind="image", filename="in.png")
    _wait(job)

    payload = job.to_dict()
    assert payload["status"] == "done"
    assert payload["id"] == job.id
    assert "result_path" not in payload
    assert str(tmp_path) not in json.dumps(payload)
    json.dumps(payload)  # must be serialisable


def test_job_ids_are_unguessable(registry):
    ids = {registry.submit(lambda j: (None, ""), kind="image", filename="x").id for _ in range(20)}
    assert len(ids) == 20
    assert all(len(i) >= 32 for i in ids)
