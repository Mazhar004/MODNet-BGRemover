import pytest

from modnet_bg.web import create_app


@pytest.fixture
def client(fake_pipeline):
    app = create_app(settings=fake_pipeline.settings, pipeline=fake_pipeline)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_create_app_does_not_load_a_model(monkeypatch, tmp_path):
    import modnet_bg.engine as engine_module

    def explode(*args, **kwargs):
        raise AssertionError("create_app loaded a model at import time")

    monkeypatch.setattr(engine_module.MattingEngine, "from_checkpoint", explode)
    from modnet_bg.config import load_settings

    app = create_app(settings=load_settings({"MODNET_DATA_DIR": str(tmp_path)}))
    assert app is not None


def test_health_is_cheap_and_never_loads_a_model(client, monkeypatch):
    import modnet_bg.engine as engine_module

    monkeypatch.setattr(
        engine_module.MattingEngine,
        "from_checkpoint",
        lambda *a, **k: pytest.fail("/health loaded a model"),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_capabilities_lists_real_devices_and_limits(client):
    payload = client.get("/api/capabilities").get_json()

    assert "cpu" in payload["devices"]
    assert payload["default_device"]
    assert set(payload["modes"]) == {"transparent", "color", "blur", "image", "matte"}
    assert payload["limits"]["max_image_mb"] == 25
    assert payload["limits"]["max_video_seconds"] == 120
    assert "png" in payload["formats"]["image"]
    assert "mp4" in payload["formats"]["video"]


def test_no_cors_header_is_emitted(client):
    response = client.get("/api/capabilities")
    assert "Access-Control-Allow-Origin" not in response.headers


def test_unknown_route_returns_json_not_html(client):
    response = client.get("/api/nope")

    assert response.status_code == 404
    assert response.is_json
    assert "error" in response.get_json()


def test_secret_key_is_not_a_hardcoded_literal(client, fake_pipeline):
    """The old app shipped SECRET_KEY = 'PrinceAPI' in source."""
    from modnet_bg.web import create_app as factory

    app = factory(settings=fake_pipeline.settings, pipeline=fake_pipeline)
    assert app.config["SECRET_KEY"] == fake_pipeline.settings.secret_key
    assert len(app.config["SECRET_KEY"]) >= 32
