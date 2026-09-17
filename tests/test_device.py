import pytest

from modnet_bg import device as dev


@pytest.fixture
def fake_availability(monkeypatch):
    """Control what torch reports as available."""

    def apply(*, cuda: bool, mps: bool):
        monkeypatch.setattr(dev, "_cuda_available", lambda: cuda)
        monkeypatch.setattr(dev, "_mps_available", lambda: mps)

    return apply


def test_cpu_is_always_available(fake_availability):
    fake_availability(cuda=False, mps=False)
    assert dev.available_devices() == ["cpu"]


def test_ordering_puts_cuda_first(fake_availability):
    fake_availability(cuda=True, mps=True)
    assert dev.available_devices() == ["cuda", "mps", "cpu"]


def test_auto_picks_best(fake_availability):
    fake_availability(cuda=True, mps=False)
    assert dev.resolve_device("auto") == "cuda"
    fake_availability(cuda=False, mps=True)
    assert dev.resolve_device("auto") == "mps"
    fake_availability(cuda=False, mps=False)
    assert dev.resolve_device("auto") == "cpu"


def test_explicit_preference_is_honoured_when_available(fake_availability):
    fake_availability(cuda=True, mps=True)
    assert dev.resolve_device("cpu") == "cpu"
    assert dev.resolve_device("mps") == "mps"


def test_unavailable_preference_clamps_instead_of_raising(fake_availability):
    fake_availability(cuda=False, mps=False)
    assert dev.resolve_device("cuda") == "cpu"


def test_unknown_and_empty_preferences_fall_back_to_auto(fake_availability):
    fake_availability(cuda=True, mps=False)
    assert dev.resolve_device("tpu") == "cuda"
    assert dev.resolve_device(None) == "cuda"
    assert dev.resolve_device("") == "cuda"
