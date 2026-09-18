import importlib


def test_package_imports_without_torch_side_effects():
    mod = importlib.import_module("modnet_bg")
    assert isinstance(mod.__version__, str)


def test_model_class_is_importable():
    from modnet_bg.models.modnet import MODNet

    assert MODNet is not None


def test_error_hierarchy():
    from modnet_bg.errors import (
        BGRemoverError,
        JobNotFoundError,
        MediaTooLargeError,
        UnsupportedMediaError,
        WeightsError,
    )

    for cls in (UnsupportedMediaError, MediaTooLargeError, WeightsError, JobNotFoundError):
        assert issubclass(cls, BGRemoverError)
