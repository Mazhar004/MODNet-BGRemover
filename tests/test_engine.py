import numpy as np
import pytest

from modnet_bg.engine import target_size


@pytest.mark.parametrize(
    "h,w,expected",
    [
        (512, 512, (512, 512)),
        (1000, 2000, (512, 1024)),  # landscape: short side to 512
        (2000, 1000, (1024, 512)),  # portrait: short side to 512
        # Degenerate strips take the pass-through branch, where the stride
        # rounding alone would give a zero: 16 - (16 % 32) == 0. The clamp
        # is what turns that into 32. 600 - (600 % 32) == 576.
        (600, 16, (576, 32)),
        (16, 600, (32, 576)),
        # Anything smaller than REF_SIZE is upscaled to it, so this does not
        # reach the clamp at all.
        (10, 10, (512, 512)),
    ],
)
def test_target_size_is_stride_aligned_and_never_zero(h, w, expected):
    th, tw = target_size(h, w)
    assert (th, tw) == expected
    assert th % 32 == 0 and tw % 32 == 0
    assert th >= 32 and tw >= 32


def test_matte_shape_and_range(fake_engine, sample_rgb):
    alpha = fake_engine.matte(sample_rgb)

    assert alpha.shape == sample_rgb.shape[:2]
    assert alpha.dtype == np.float32
    assert 0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0


def test_extreme_aspect_ratio_does_not_crash(fake_engine):
    strip = np.full((600, 16, 3), 200, dtype=np.uint8)
    alpha = fake_engine.matte(strip)
    assert alpha.shape == (600, 16)


def test_engine_holds_no_per_image_state(fake_engine):
    first = np.full((64, 64, 3), 10, dtype=np.uint8)
    second = np.full((200, 120, 3), 20, dtype=np.uint8)

    fake_engine.matte(first)
    alpha = fake_engine.matte(second)

    assert alpha.shape == (200, 120)
    leaked = [a for a in vars(fake_engine) if a not in {"_model", "_device", "_lock"}]
    assert leaked == [], f"engine stored per-request state: {leaked}"


def test_concurrent_calls_do_not_cross_contaminate(fake_engine):
    """Regression test for the statefulness that forced this rewrite."""
    import concurrent.futures as cf

    shapes = [(64, 48), (128, 96), (200, 150), (32, 240)] * 6
    images = [np.full((h, w, 3), i % 256, dtype=np.uint8) for i, (h, w) in enumerate(shapes)]

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fake_engine.matte, images))

    for (h, w), alpha in zip(shapes, results, strict=True):
        assert alpha.shape == (h, w)


@pytest.mark.parametrize("h,w", [(5622, 4516), (1080, 1920), (333, 777), (97, 43)])
def test_matte_on_the_real_device_with_non_divisible_sizes(fake_engine_on_real_device, h, w):
    """Regression: the upscale back to source resolution must not run on MPS.

    mode='area' lowers to adaptive_avg_pool2d, which on MPS requires the
    output size to be divisible by the input size (pytorch#96056). Source
    dimensions are arbitrary, so that almost never holds.
    """
    image = np.full((h, w, 3), 128, dtype=np.uint8)
    alpha = fake_engine_on_real_device.matte(image)
    assert alpha.shape == (h, w)
    assert alpha.dtype == np.float32


def test_rejects_malformed_input(fake_engine):
    with pytest.raises(ValueError, match="H, W, 3"):
        fake_engine.matte(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        fake_engine.matte(np.zeros((10, 10, 3), dtype=np.float32))
