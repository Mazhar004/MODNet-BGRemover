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
