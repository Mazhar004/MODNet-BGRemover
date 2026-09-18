from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "modnet_bg" / "web"
INDEX = WEB / "templates" / "index.html"
APP_JS = WEB / "static" / "js" / "app.js"
APP_CSS = WEB / "static" / "css" / "app.css"


@pytest.fixture(scope="module")
def index_html() -> str:
    return INDEX.read_text()


def test_required_assets_exist():
    for path in (INDEX, APP_JS, APP_CSS):
        assert path.is_file(), f"missing {path}"


@pytest.mark.parametrize(
    "element_id",
    [
        "drop-zone",
        "file-input",
        "mode",
        "color",
        "blur-radius",
        "background-input",
        "device",
        "queue",
        "status-region",
    ],
)
def test_template_exposes_every_hook(index_html, element_id):
    assert f'id="{element_id}"' in index_html


def test_no_cdn_references(index_html):
    """Every asset must be served from this origin.

    The old page pulled Bootstrap, jQuery, Popper and axios from three
    CDNs: a supply-chain exposure and a hard dependency on the browser
    having outbound network access.
    """
    for marker in ("http://", "https://", "//cdn", "unpkg", "jsdelivr", "googleapis"):
        assert marker not in index_html, f"external reference {marker!r} in template"


def test_scripts_are_loaded_via_url_for(index_html):
    assert "url_for('static'" in index_html or 'url_for("static"' in index_html


def test_status_region_is_announced_to_screen_readers(index_html):
    region = index_html.split('id="status-region"')[1][:200]
    assert "aria-live" in region


def test_file_input_has_a_label(index_html):
    assert 'for="file-input"' in index_html


def test_app_js_exports_the_public_surface():
    source = APP_JS.read_text()
    for symbol in ("window.BGR", "addFiles", "submit", "poll"):
        assert symbol in source


def test_app_js_has_no_infinite_busy_loop():
    """The old detection.js spun a 100000-iteration console.log loop."""
    source = APP_JS.read_text()
    assert "100000" not in source


COMPARE_JS = WEB / "static" / "js" / "compare.js"
WEBCAM_JS = WEB / "static" / "js" / "webcam.js"


def test_compare_js_exists_and_is_wired():
    assert COMPARE_JS.is_file()
    source = COMPARE_JS.read_text()
    assert "bgr:preview" in source
    assert "window.BGRCompare" in source


def test_compare_is_keyboard_accessible():
    source = COMPARE_JS.read_text()
    assert "keydown" in source, "slider must be operable without a mouse"
    assert "ArrowLeft" in source and "ArrowRight" in source


def test_transparency_is_shown_against_a_checkerboard():
    assert "checker" in COMPARE_JS.read_text()
    assert ".checker" in APP_CSS.read_text()


@pytest.mark.parametrize(
    "element_id", ["webcam-panel", "webcam-start", "webcam-stop", "webcam-canvas", "webcam-fps"]
)
def test_webcam_hooks_present(index_html, element_id):
    assert f'id="{element_id}"' in index_html


def test_webcam_js_keeps_one_frame_in_flight():
    source = WEBCAM_JS.read_text()
    assert "getUserMedia" in source
    assert "inFlight" in source, "backpressure flag missing; frames will queue up"


def test_webcam_reports_measured_fps():
    assert "webcam-fps" in WEBCAM_JS.read_text()


def test_webcam_stops_all_tracks_on_stop():
    """Leaving the camera light on after Stop is a privacy bug."""
    source = WEBCAM_JS.read_text()
    assert "getTracks" in source and "stop()" in source


def test_thumbnail_preview_is_built_from_the_local_file():
    """The preview must appear as soon as a file is chosen, without a round
    trip -- so it comes from an object URL, not from the server."""
    source = APP_JS.read_text()
    assert "thumbnailFor" in source
    assert "createObjectURL" in source
    assert ".thumb" in APP_CSS.read_text()


def test_object_urls_are_revoked():
    """Otherwise every added-and-removed file leaks its decoded bitmap."""
    assert "revokeObjectURL" in APP_JS.read_text()


def test_progress_shows_a_numeric_readout_and_a_stage():
    """Regression: a bar alone sat at 0% and read as stuck."""
    source = APP_JS.read_text()
    assert "bar__pct" in source, "no percentage readout"
    assert "job.stage" in source, "server stage label is not surfaced"
    assert 'setAttribute("role", "progressbar")' in source


def test_no_hardcoded_white_in_the_comparison_slider():
    """Regression: the divider was #fff, which is the brightest thing on a dark
    page and reads as an unwanted white border."""
    import re

    css = APP_CSS.read_text()
    compare_block = css[css.index(".compare {") : css.index(".webcam-canvas {")]
    # Strip comments: the explanation above the rule names #fff on purpose.
    declarations = re.sub(r"/\*.*?\*/", "", compare_block, flags=re.S).lower()
    for literal in ("#fff", "#ffffff", "white"):
        assert literal not in declarations, f"{literal} still hardcoded in the slider"


def test_checkerboard_is_theme_aware():
    """A light checker against a near-black UI reads as a white block rather
    than as transparency."""
    css = APP_CSS.read_text()
    checker = css[css.index(".checker {") : css.index(".checker {") + 600]
    assert "var(--checker-a)" in checker and "var(--checker-b)" in checker
    assert "#8d8d8d" not in checker and "#b6b6b6" not in checker
    # Both themes must define the tokens.
    assert css.count("--checker-a") >= 3


def test_borders_are_translucent_not_solid_hairlines():
    css = APP_CSS.read_text()
    assert "--border: rgba(" in css, "borders should tint with what is behind them"


def test_slider_handle_meets_touch_target_size():
    css = APP_CSS.read_text()
    handle = css[css.index(".compare__handle {") :][:400]
    assert "width: 44px" in handle, "drag handle must be at least 44px wide"


def test_reduced_motion_is_respected():
    assert "prefers-reduced-motion" in APP_CSS.read_text()


def test_comparison_sides_are_labelled():
    source = COMPARE_JS.read_text()
    assert "compare__tag" in source
    assert "Original" in source and "Removed" in source
