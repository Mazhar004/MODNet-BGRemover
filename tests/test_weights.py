import hashlib

import pytest

from modnet_bg import weights as w
from modnet_bg.errors import WeightsError


def _write(path, payload: bytes):
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_known_specs_are_registered():
    assert set(w.WEIGHTS) == {"photographic", "webcam"}
    spec = w.WEIGHTS["photographic"]
    assert spec.filename == "modnet_photographic_portrait_matting.ckpt"
    assert len(spec.sha256) == 64


def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / "blob.bin"
    expected = _write(target, b"hello modnet")
    assert w.sha256_file(target) == expected


def test_resolves_first_matching_dir(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    spec = w.WEIGHTS["photographic"]
    digest = _write(second / spec.filename, b"payload")
    monkeypatch.setitem(
        w.WEIGHTS, "photographic", spec.__class__(spec.key, spec.filename, digest)
    )

    found = w.resolve_weights("photographic", search_dirs=[first, second])
    assert found == second / spec.filename


def test_digest_mismatch_is_fatal(tmp_path):
    spec = w.WEIGHTS["photographic"]
    (tmp_path / spec.filename).write_bytes(b"not the real checkpoint")

    with pytest.raises(WeightsError, match="checksum"):
        w.resolve_weights("photographic", search_dirs=[tmp_path])


def test_missing_weights_explains_how_to_fix(tmp_path):
    with pytest.raises(WeightsError) as excinfo:
        w.resolve_weights("photographic", search_dirs=[tmp_path])

    message = str(excinfo.value)
    assert "modnet_photographic_portrait_matting.ckpt" in message
    assert "MODNET_WEIGHTS_DIR" in message


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(WeightsError, match="Unknown"):
        w.resolve_weights("nope", search_dirs=[tmp_path])


def test_download_writes_atomically_and_verifies(tmp_path, monkeypatch):
    spec = w.WEIGHTS["webcam"]
    payload = b"downloaded bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(w.WEIGHTS, "webcam", spec.__class__(spec.key, spec.filename, digest))
    monkeypatch.setattr(w, "_http_get", lambda url, dest: dest.write_bytes(payload))

    cache = tmp_path / "cache"
    found = w.resolve_weights(
        "webcam",
        search_dirs=[tmp_path / "empty"],
        download_url="https://example/x",
        cache_dir=cache,
    )

    assert found == cache / spec.filename
    assert found.read_bytes() == payload
    assert not list(cache.glob("*.part")), "temporary download file was left behind"


def test_bad_download_is_not_left_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "_http_get", lambda url, dest: dest.write_bytes(b"corrupt"))
    cache = tmp_path / "cache"

    with pytest.raises(WeightsError, match="checksum"):
        w.resolve_weights(
            "webcam",
            search_dirs=[tmp_path / "empty"],
            download_url="https://e/x",
            cache_dir=cache,
        )

    assert not (cache / w.WEIGHTS["webcam"].filename).exists()
    assert not list(cache.glob("*.part"))
