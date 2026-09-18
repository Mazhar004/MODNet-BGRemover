import numpy as np
import pytest

from modnet_bg import video
from modnet_bg.errors import CorruptMediaError


def test_probe_reports_real_dimensions_and_fps(tiny_video):
    info = video.probe(tiny_video)

    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(10.0, abs=0.1)
    assert info.duration == pytest.approx(1.0, abs=0.2)


def test_iter_frames_yields_rgb_uint8(tiny_video):
    frames = list(video.iter_frames(tiny_video))

    assert len(frames) == 10
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8


def test_fps_is_preserved_not_hardcoded(tmp_path, tiny_video):
    """Regression: the old writer was pinned to 20fps regardless of source."""
    dest = tmp_path / "out.mp4"
    frames = list(video.iter_frames(tiny_video))
    source_fps = video.probe(tiny_video).fps

    video.write_video(frames, dest, fps=source_fps, size=(64, 48), alpha=False)

    assert video.probe(dest).fps == pytest.approx(source_fps, abs=0.1)


def test_alpha_output_is_webm_and_round_trips(tmp_path):
    dest = tmp_path / "out.webm"
    frames = []
    for _ in range(5):
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[8:24, 8:24, 3] = 255
        frames.append(rgba)

    video.write_video(frames, dest, fps=10.0, size=(32, 32), alpha=True)

    assert dest.exists() and dest.stat().st_size > 0
    assert video.probe(dest).frame_count >= 4


def test_container_for_each_mode():
    assert video.container_for("transparent") == (".webm", True)
    assert video.container_for("color") == (".mp4", False)
    assert video.container_for("blur") == (".mp4", False)
    assert video.container_for("image") == (".mp4", False)
    assert video.container_for("matte") == (".mp4", False)


def test_mux_audio_reports_false_when_source_is_silent(tmp_path, tiny_video):
    dest = tmp_path / "muxed.mp4"
    assert video.mux_audio(tiny_video, tiny_video, dest) is False


def test_probe_rejects_a_non_video(tmp_path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64)

    with pytest.raises(CorruptMediaError):
        video.probe(bogus)


def test_animated_gif_reads_through_ffmpeg(tmp_path):
    """ffmpeg handles GIF; imageio's v3 plugin refuses it on extension alone."""
    from PIL import Image

    gif = tmp_path / "anim.gif"
    frames = [Image.new("RGB", (640, 360), (i * 4 % 256, 0, 0)) for i in range(61)]
    frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)

    info = video.probe(gif)
    assert info.frame_count == 61
    assert (info.width, info.height) == (640, 360)

    read = list(video.iter_frames(gif))
    assert len(read) == 61
    assert read[0].shape == (360, 640, 3)


def test_animated_webp_falls_back_to_pillow(animated_webp):
    """The bundled ffmpeg has no libwebp demuxer. Frames must not be lost."""
    info = video.probe(animated_webp)
    assert info.frame_count == 5
    assert (info.width, info.height) == (32, 24)
    assert info.fps == pytest.approx(10.0, abs=0.5)

    frames = list(video.iter_frames(animated_webp))
    assert len(frames) == 5
    assert frames[0].shape == (24, 32, 3)
    assert frames[0].dtype == np.uint8
    # Frames must actually differ; returning frame 0 five times would pass a
    # naive count check.
    assert not np.array_equal(frames[0], frames[-1])
