"""Real Streamlit app smoke tests; use the existing synthetic demo data."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
VIEWS = ["Daftar prioritas", "Kasus perusahaan", "Benchmark", "Keadilan & jalur acak", "Cek kepesertaan saya"]


def test_sidebar_navigation_opens_case_view():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "view" in [radio.key for radio in app.sidebar.radio], "The workspace needs a persistent sidebar navigation."
    assert app.sidebar.radio(key="view").options == VIEWS
    app.sidebar.radio(key="view").set_value("Kasus perusahaan").run()
    assert not app.exception
    assert "case_employer" in [box.key for box in app.selectbox]
    assert app.sidebar.radio(key="view").value == "Kasus perusahaan"


def test_workspace_copy_has_no_emojis():
    import re

    emoji = re.compile(r"[\U0001f300-\U0001faff\u2600-\u27bf\ufe0f]")
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    visible = []
    for kind in ("title", "header", "subheader", "caption", "markdown", "info", "warning", "error", "success", "button"):
        for element in app.get(kind):
            visible.append(str(getattr(element, "value", "")))
            visible.append(str(getattr(element, "label", "")))
    assert not emoji.search("\n".join(visible)), "Use typography and a coherent icon system, not emojis."


def test_priority_preview_opens_the_ranked_employer():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    expected = app.dataframe[0].value.index[0]
    assert "open_case" in [button.key for button in app.button], "The queue needs a direct path into its first case."
    app.button(key="open_case").click().run()
    assert not app.exception
    assert app.sidebar.radio(key="view").value == "Kasus perusahaan"
    assert app.selectbox(key="case_employer").value == expected


def test_interface_text_avoids_typographic_tells():
    """Em dashes, middle dots and ellipsis glyphs read as generated copy, not product writing."""
    sources = {
        "app": APP.read_text(encoding="utf-8"),
        "css": (APP.parent / "assets" / "workspace.css").read_text(encoding="utf-8"),
    }
    for name, source in sources.items():
        for glyph in ("\u2014", "\u2013", "\u00b7", "\u2026"):
            assert glyph not in source, f"{name} still uses {glyph!r}"


def test_workspace_opens_on_the_task_not_on_decoration():
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not app.exception
    markup = "\n".join(element.value for element in app.markdown)
    for decoration in ("workspace-topbar", "sidebar-brand svg", "<svg", "class=\"pill"):
        assert decoration not in markup, f"Decorative {decoration} outlives its purpose."
    app.sidebar.radio(key="view").set_value("Kasus perusahaan").run()
    assert not any(" | " in element.value for element in app.caption), "Write the case identity as a sentence."


@pytest.mark.parametrize("branch", ["B", "C", "A"])
@pytest.mark.parametrize("view", VIEWS)
def test_each_view_renders_on_every_synthetic_branch(branch, view):
    # Cold branches load their parquet slices on first use, so allow for real I/O.
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.sidebar.selectbox(key="branch").set_value(branch).run()
    app.sidebar.radio(key="view").set_value(view).run()
    assert not app.exception
    assert app.title[0].value == view
    assert any("Data sintetis" in element.value for element in app.markdown)
