from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
VIEWS = ["Engagement Health", "Burnout Risk", "Role & Career Stage", "Manager Action Panel", "Data & Method"]


def run():
    return AppTest.from_file(APP, default_timeout=90).run()


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_renders(view):
    at = run()
    at.radio(key="view").set_value(view).run()
    assert not at.exception


def test_filters_reset_and_small_cohort():
    at = run()
    at.radio(key="f_ot").set_value("Yes").run()
    at.multiselect(key="f_dept").set_value(["Human Resources"]).run()
    at.multiselect(key="f_role").set_value(["Manager"]).run()
    assert not at.exception
    assert any("Fewer than 5 records" in m.value for m in at.markdown)  # suppressed safely
    [b for b in at.button if "Reset" in b.label][0].click().run()
    assert at.session_state["f_ot"] == "All" and at.session_state["f_dept"] == [] and not at.exception
    assert not any("Fewer than 5 records" in m.value for m in at.markdown)


@pytest.mark.parametrize("view", VIEWS[:4])
def test_empty_filter_result_is_safe(view):
    at = run()
    at.radio(key="view").set_value(view).run()
    at.slider(key="f_thr").set_value(25).run()
    at.checkbox(key="f_below").set_value(True).run()  # nobody below CEI 25 -> empty view
    assert not at.exception
