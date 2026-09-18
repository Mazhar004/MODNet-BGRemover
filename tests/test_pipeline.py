import io

import numpy as np
import pytest
from PIL import Image

from modnet_bg.jobs import Job, JobCancelled
from modnet_bg.pipeline import Pipeline, ProcessOptions


def _png(size=(64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _job(kind="image") -> Job:
    return Job(id="test", kind=kind, filename="in.png")


def test_image_job_writes_a_transparent_png(fake_pipeline):
    run = fake_pipeline.image_job(_png(), ProcessOptions(mode="transparent"), stem="portrait")
    path, name = run(_job())

    assert path.exists()
    assert name == "portrait.png"
    with Image.open(path) as out:
        assert out.mode == "RGBA"
        assert out.size == (64, 48)


def test_image_job_colour_mode_writes_opaque_png(fake_pipeline):
    run = fake_pipeline.image_job(
        _png(), ProcessOptions(mode="color", color=(0, 0, 255)), stem="portrait"
    )
    path, _ = run(_job())

    with Image.open(path) as out:
        assert out.mode == "RGB"
        assert out.getpixel((0, 0)) == (0, 0, 255)  # corner is background


def test_image_job_records_the_device_used(fake_pipeline):
    job = _job()
    fake_pipeline.image_job(_png(), ProcessOptions(), stem="x")(job)
    assert job.device_used == "cpu"


def test_output_filename_is_server_generated(fake_pipeline):
    """No user-supplied string may reach the filesystem."""
    run = fake_pipeline.image_job(_png(), ProcessOptions(), stem="../../etc/passwd")
    path, name = run(_job())

    assert path.parent == fake_pipeline.results_dir
    assert "etc" not in path.name and ".." not in path.name
    assert name == "../../etc/passwd.png"  # display name only, never a path


def test_video_job_sets_total_and_advances_per_frame(fake_pipeline, tiny_video):
    job = _job(kind="video")
    run = fake_pipeline.video_job(tiny_video, ProcessOptions(mode="color"), stem="clip")
    path, name = run(job)

    assert path.exists()
    assert name == "clip.mp4"
    assert job.frames_total == 10
    assert job.frames_done == 10


def test_video_job_honours_cancellation(fake_pipeline, tiny_video):
    job = _job(kind="video")
    job.cancel()
    run = fake_pipeline.video_job(tiny_video, ProcessOptions(mode="color"), stem="clip")

    with pytest.raises(JobCancelled):
        run(job)


def test_video_duration_cap_is_enforced(tmp_path, tiny_video, engine_factory, monkeypatch):
    from modnet_bg import pipeline as pipeline_module
    from modnet_bg.config import load_settings
    from modnet_bg.errors import MediaTooLargeError

    monkeypatch.setattr(
        pipeline_module.weights, "resolve_weights", lambda key, **kw: tmp_path / "x.ckpt"
    )
    settings = load_settings({"MODNET_DATA_DIR": str(tmp_path), "MODNET_MAX_VIDEO_SECONDS": "0"})
    pipeline = Pipeline(settings, engine_factory=engine_factory)

    with pytest.raises(MediaTooLargeError, match="longer than"):
        pipeline.video_job(tiny_video, ProcessOptions(), stem="clip")(_job("video"))


def test_transparent_video_produces_webm(fake_pipeline, tiny_video):
    run = fake_pipeline.video_job(tiny_video, ProcessOptions(mode="transparent"), stem="clip")
    path, name = run(_job("video"))

    assert path.suffix == ".webm"
    assert name == "clip.webm"


def test_matte_mode_produces_a_grayscale_image(fake_pipeline):
    run = fake_pipeline.image_job(_png(), ProcessOptions(mode="matte"), stem="m")
    path, _ = run(_job())

    with Image.open(path) as out:
        assert out.mode == "L"


def test_animated_gif_is_processed_as_video(fake_pipeline, tmp_path):
    """Every frame must survive, not just the first."""
    source = tmp_path / "anim.gif"
    frames = [Image.new("RGB", (32, 24), (i * 60 % 256, 0, 0)) for i in range(5)]
    frames[0].save(
        source, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0
    )

    job = _job(kind="video")
    path, name = fake_pipeline.video_job(source, ProcessOptions(mode="color"), stem="anim")(job)

    assert path.exists()
    assert job.frames_done == 5
    assert name == "anim.mp4"


def test_video_result_keeps_source_dimensions(fake_pipeline, tiny_video):
    from modnet_bg import video

    path, _ = fake_pipeline.video_job(tiny_video, ProcessOptions(mode="color"), stem="clip")(
        _job("video")
    )
    info = video.probe(path)

    # Regression: the old writer emitted a double-width side-by-side frame.
    assert (info.width, info.height) == (64, 48)


def test_image_mode_composites_the_supplied_background(fake_pipeline):
    background = np.full((48, 64, 3), (0, 255, 0), dtype=np.uint8)
    run = fake_pipeline.image_job(
        _png(), ProcessOptions(mode="image", background=background), stem="bg"
    )
    path, _ = run(_job())

    with Image.open(path) as out:
        assert out.getpixel((0, 0)) == (0, 255, 0)
