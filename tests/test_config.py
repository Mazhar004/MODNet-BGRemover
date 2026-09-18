from pathlib import Path

from modnet_bg.config import load_settings


def test_defaults_are_usable_with_an_empty_environment():
    settings = load_settings({})

    assert settings.device == "auto"
    assert settings.port == 8000
    assert settings.max_image_mb == 25
    assert settings.max_video_mb == 250
    assert settings.max_video_seconds == 120
    assert settings.max_workers == 2
    assert settings.job_ttl_seconds == 3600
    assert settings.weights_dir is None


def test_environment_overrides_every_field():
    settings = load_settings(
        {
            "MODNET_DEVICE": "cpu",
            "PORT": "9000",
            "MODNET_WEIGHTS_DIR": "/opt/weights",
            "MODNET_MAX_IMAGE_MB": "5",
            "MODNET_MAX_VIDEO_MB": "50",
            "MODNET_MAX_VIDEO_SECONDS": "30",
            "MODNET_MAX_WORKERS": "4",
            "MODNET_JOB_TTL_SECONDS": "60",
            "SECRET_KEY": "explicit",
        }
    )

    assert settings.device == "cpu"
    assert settings.port == 9000
    assert settings.weights_dir == Path("/opt/weights")
    assert settings.max_workers == 4
    assert settings.secret_key == "explicit"
    assert settings.secret_key_generated is False


def test_generated_secret_is_flagged_and_random():
    first = load_settings({})
    second = load_settings({})

    assert first.secret_key_generated is True
    assert first.secret_key != second.secret_key
    assert len(first.secret_key) >= 32


def test_upload_cap_is_the_larger_of_the_two():
    settings = load_settings({"MODNET_MAX_IMAGE_MB": "5", "MODNET_MAX_VIDEO_MB": "50"})
    assert settings.max_image_bytes == 5 * 1024 * 1024
    assert settings.max_upload_bytes == 50 * 1024 * 1024


def test_weight_urls_are_collected_per_key():
    settings = load_settings({"MODNET_WEIGHTS_URL_PHOTOGRAPHIC": "https://example/p.ckpt"})
    assert settings.weights_urls == {"photographic": "https://example/p.ckpt"}


def test_invalid_numbers_fall_back_to_the_default():
    settings = load_settings({"MODNET_MAX_WORKERS": "not-a-number"})
    assert settings.max_workers == 2


def test_worker_count_is_clamped_to_something_sane():
    assert load_settings({"MODNET_MAX_WORKERS": "0"}).max_workers == 1
    assert load_settings({"MODNET_MAX_WORKERS": "999"}).max_workers == 16
