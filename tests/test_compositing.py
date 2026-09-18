import numpy as np
import pytest

from modnet_bg.compositing import OUTPUT_MODES, composite, cover_resize


@pytest.fixture
def red():
    return np.full((40, 60, 3), (255, 0, 0), dtype=np.uint8)


@pytest.fixture
def half_alpha():
    return np.full((40, 60), 0.5, dtype=np.float32)


def test_transparent_keeps_straight_rgb(red, half_alpha):
    """Regression: the old save() composited onto white AND stored alpha,
    fringing every semi-transparent pixel. RGB must be untouched."""
    out = composite(red, half_alpha, "transparent")

    assert out.shape == (40, 60, 4)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[..., :3], red)
    assert set(np.unique(out[..., 3])) <= {127, 128}


def test_transparent_fully_opaque_and_transparent_edges(red):
    alpha = np.zeros((40, 60), dtype=np.float32)
    alpha[:, 30:] = 1.0
    out = composite(red, alpha, "transparent")

    assert out[0, 0, 3] == 0
    assert out[0, 59, 3] == 255
    np.testing.assert_array_equal(out[0, 0, :3], [255, 0, 0])


def test_color_mode_blends_towards_the_chosen_colour(red, half_alpha):
    out = composite(red, half_alpha, "color", color=(0, 0, 255))

    assert out.shape == (40, 60, 3)
    np.testing.assert_allclose(out[0, 0], [127, 0, 128], atol=1)


def test_color_mode_fully_transparent_is_the_colour(red):
    alpha = np.zeros((40, 60), dtype=np.float32)
    out = composite(red, alpha, "color", color=(10, 20, 30))
    np.testing.assert_array_equal(out[0, 0], [10, 20, 30])


def test_matte_mode_is_single_channel_grayscale(red, half_alpha):
    out = composite(red, half_alpha, "matte")
    assert out.shape == (40, 60)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {127, 128}


def test_blur_mode_keeps_subject_and_softens_background(red):
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, ::2] = (255, 255, 255)  # high-frequency stripes
    alpha = np.zeros((40, 60), dtype=np.float32)
    alpha[10:30, 20:40] = 1.0

    out = composite(image, alpha, "blur", blur_radius=8)

    np.testing.assert_array_equal(out[20, 21], image[20, 21])  # subject preserved
    assert out[0, 0].std() < image[0, :].std()  # background softened


def test_image_mode_uses_the_supplied_background(red):
    alpha = np.zeros((40, 60), dtype=np.float32)
    background = np.full((40, 60, 3), (0, 255, 0), dtype=np.uint8)

    out = composite(red, alpha, "image", background=background)
    np.testing.assert_array_equal(out[0, 0], [0, 255, 0])


def test_image_mode_requires_a_background(red, half_alpha):
    with pytest.raises(ValueError, match="background"):
        composite(red, half_alpha, "image")


def test_cover_resize_preserves_aspect_and_fills(red):
    wide = np.zeros((10, 100, 3), dtype=np.uint8)
    wide[:, :50] = (255, 0, 0)
    wide[:, 50:] = (0, 0, 255)

    out = cover_resize(wide, (40, 40))

    assert out.shape == (40, 40, 3)
    assert not (out == 0).all(axis=2).any(), "cover must not letterbox"


def test_unknown_mode_rejected(red, half_alpha):
    with pytest.raises(ValueError, match="Unknown output mode"):
        composite(red, half_alpha, "hologram")


def test_all_modes_are_reachable(red, half_alpha):
    background = np.zeros((40, 60, 3), dtype=np.uint8)
    for mode in OUTPUT_MODES:
        out = composite(red, half_alpha, mode, background=background)
        assert out.dtype == np.uint8
