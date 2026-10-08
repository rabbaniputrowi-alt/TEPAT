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


@pytest.mark.parametrize("branch", ["B", "C", "A"])
@pytest.mark.parametrize("view", VIEWS)
def test_each_view_renders_on_every_synthetic_branch(branch, view):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.sidebar.selectbox(key="branch").set_value(branch).run()
    app.sidebar.radio(key="view").set_value(view).run()
    assert not app.exception
    assert app.title[0].value == view
    assert any("Data sintetis" in element.value for element in app.markdown)
