import io
import time

import pytest
from PIL import Image

from modnet_bg.web import create_app


@pytest.fixture
def client(fake_pipeline):
    app = create_app(settings=fake_pipeline.settings, pipeline=fake_pipeline)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _png_bytes(size=(64, 48), color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _upload(client, data=None, filename="portrait.png", **fields):
    payload = {"file": (io.BytesIO(data or _png_bytes()), filename)}
    payload.update(fields)
    return client.post("/api/jobs", data=payload, content_type="multipart/form-data")


def _poll(client, job_id, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").get_json()
        if body["status"] in {"done", "error", "cancelled"}:
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_upload_returns_a_job_id(client):
    response = _upload(client)

    assert response.status_code == 202
    assert response.get_json()["id"]


def test_job_runs_to_completion_and_result_downloads(client):
    job_id = _upload(client).get_json()["id"]
    body = _poll(client, job_id)

    assert body["status"] == "done", body
    assert body["progress"] == 1.0

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.headers["Content-Type"].startswith("image/png")
    assert "portrait.png" in result.headers["Content-Disposition"]


def test_result_is_404_before_completion_and_for_unknown_ids(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    assert client.get("/api/jobs/does-not-exist/result").status_code == 404


def test_upload_without_a_file_is_rejected(client):
    response = client.post("/api/jobs", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_media_upload_is_415(client):
    response = _upload(client, data=b"MZ\x90\x00 definitely not an image at all")
    assert response.status_code == 415


def test_invalid_mode_is_rejected(client):
    response = _upload(client, mode="hologram")
    assert response.status_code == 400
    assert "mode" in response.get_json()["error"].lower()


def test_image_mode_without_a_background_is_rejected(client):
    response = _upload(client, mode="image")
    assert response.status_code == 400


def test_image_mode_with_a_background_succeeds(client):
    response = client.post(
        "/api/jobs",
        data={
            "file": (io.BytesIO(_png_bytes()), "portrait.png"),
            "background": (io.BytesIO(_png_bytes(color=(0, 255, 0))), "bg.png"),
            "mode": "image",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    assert _poll(client, response.get_json()["id"])["status"] == "done"


def test_colour_is_parsed_from_hex(client):
    job_id = _upload(client, mode="color", color="#00ff00").get_json()["id"]
    assert _poll(client, job_id)["status"] == "done"


def test_malformed_colour_is_rejected(client):
    assert _upload(client, mode="color", color="lime").status_code == 400


def test_blur_radius_is_bounded(client):
    assert _upload(client, mode="blur", blur_radius="0").status_code == 400
    assert _upload(client, mode="blur", blur_radius="500").status_code == 400
    assert _upload(client, mode="blur", blur_radius="abc").status_code == 400


def test_cancel_returns_the_job(client):
    job_id = _upload(client).get_json()["id"]
    response = client.delete(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert response.get_json()["status"] in {"cancelled", "done", "running", "queued"}


def test_traversal_filename_never_reaches_the_filesystem(client, fake_pipeline):
    """The old app interpolated the upload name straight into a path."""
    job_id = _upload(client, filename="../../../../etc/passwd.png").get_json()["id"]
    assert _poll(client, job_id)["status"] == "done"

    written = list(fake_pipeline.results_dir.iterdir())
    assert written, "no result written"
    for path in written:
        assert ".." not in path.name
        assert path.parent == fake_pipeline.results_dir


def test_error_responses_never_leak_internals(client):
    response = _upload(client, data=b"MZ\x90\x00 definitely not an image at all")
    body = response.get_json()
    assert "Traceback" not in str(body)
    assert "/Users/" not in str(body)


def test_oversized_upload_is_413(fake_pipeline):
    from modnet_bg.config import load_settings

    settings = load_settings(
        {
            "MODNET_DATA_DIR": str(fake_pipeline.results_dir.parent),
            "MODNET_MAX_VIDEO_MB": "1",
            "MODNET_MAX_IMAGE_MB": "1",
        }
    )
    app = create_app(settings=settings, pipeline=fake_pipeline)
    app.config.update(TESTING=True)

    with app.test_client() as client:
        response = client.post(
            "/api/jobs",
            data={"file": (io.BytesIO(b"\x00" * (2 * 1024 * 1024)), "big.png")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 413
    assert response.is_json


def test_zip_bundles_completed_results(client):
    import zipfile

    ids = []
    for _ in range(2):
        job_id = _upload(client).get_json()["id"]
        _poll(client, job_id)
        ids.append(job_id)

    response = client.post("/api/jobs/zip", json={"ids": ids})

    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.data))
    assert len(archive.namelist()) == 2


def test_zip_rejects_an_empty_selection(client):
    assert client.post("/api/jobs/zip", json={"ids": []}).status_code == 400


def test_device_preference_is_honoured(client):
    job_id = _upload(client, device="cpu").get_json()["id"]
    body = _poll(client, job_id)
    assert body["device_used"] == "cpu"
