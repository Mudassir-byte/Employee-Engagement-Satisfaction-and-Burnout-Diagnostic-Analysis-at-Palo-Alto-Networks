"""Engagement, Satisfaction & Burnout Diagnostic - Streamlit workspace.

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from pan_hr import config as C
from pan_hr import data as D
from pan_hr import ui, views
from pan_hr.filters import TENURE_BASES, FilterState, apply_filters

st.set_page_config(page_title="Engagement & Burnout Diagnostic", page_icon="\U0001F9ED", layout="wide",
                   initial_sidebar_state="expanded")

DATA_PATH = Path(D.DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def load(path: str, mtime: float):
    """Cached on file path + modification time so a replaced CSV is never served stale."""
    return D.load_and_prepare(path)


raw, df, report, nulled = load(str(DATA_PATH), DATA_PATH.stat().st_mtime)

# --------------------------------------------------------------------------- #
# Session defaults & reset
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "f_dept": [], "f_role": [], "f_ot": "All", "f_thr": C.DEFAULT_LOW_ENGAGEMENT_THRESHOLD,
    "f_below": False, "f_basis": "Years at company", "f_tenure": (0, 40),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("rev", 0)
st.session_state.setdefault("xf_roles", [])
st.session_state.setdefault("xf_risk", [])
st.session_state.setdefault("f_motion", True)


def reset_filters() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["xf_roles"] = []
    st.session_state["xf_risk"] = []
    st.session_state["rev"] += 1  # new chart keys -> chart selections cleared


NAV = ["Engagement Health", "Burnout Risk", "Role & Career Stage", "Manager Action Panel", "Data & Method"]

with st.sidebar:
    st.markdown("### Filters")
    depts = sorted(df["Department"].unique())
    st.multiselect("Department (empty = all)", depts, key="f_dept")
    role_opts = sorted(df[df["Department"].isin(st.session_state["f_dept"] or depts)]["JobRole"].unique())
    st.session_state["f_role"] = [r for r in st.session_state["f_role"] if r in role_opts]
    st.multiselect("Job role (empty = all)", role_opts, key="f_role")
    st.radio("Overtime", ["All", "Yes", "No"], key="f_ot", horizontal=True,
             help="All records, only those working overtime, or only those who do not.")
    st.selectbox("Tenure measure", list(TENURE_BASES), key="f_basis")
    st.slider("Tenure range (years)", 0, 40, key="f_tenure")
    st.slider("Low-engagement threshold (CEI)", 25, 100, key="f_thr",
              help="CEI of 50 means an average rating of 2 out of 4. Used for the 'below threshold' shares and alerts.")
    st.checkbox("Restrict views to records below the threshold", key="f_below")
    st.button("Reset all filters and selections", on_click=reset_filters, width="stretch")
    st.divider()
    st.toggle("Motion", key="f_motion", help="Turn off chart transitions and staged appearance. "
                                             "Your system's reduced-motion setting is also respected.")
    st.caption("Aggregate analysis of the provided dataset. Groups under 5 records are combined or withheld.")

# --------------------------------------------------------------------------- #
# Filter state (+ chart cross-filter selections from the previous run)
# --------------------------------------------------------------------------- #
rev = st.session_state["rev"]


roles_sel = [r for r in st.session_state["xf_roles"] if r in set(df["JobRole"])]
risk_sel = [r for r in st.session_state["xf_risk"] if r in C.BURNOUT_LEVELS]

f = FilterState(
    departments=st.session_state["f_dept"], roles=st.session_state["f_role"], overtime=st.session_state["f_ot"],
    threshold=float(st.session_state["f_thr"]), below_only=st.session_state["f_below"],
    tenure_basis=st.session_state["f_basis"], tenure_range=tuple(st.session_state["f_tenure"]),
    risk_levels=risk_sel, cross_roles=roles_sel,
)
view = apply_filters(df, f)
ctx = views.Ctx(
    full=df, view=view,
    role_frame=apply_filters(df, f, use_roles=False),
    risk_frame=apply_filters(df, f, use_risk=False),
    f=f, motion=st.session_state["f_motion"], rev=rev, report=report, nulled=nulled,
)

st.markdown(ui.BASE_CSS, unsafe_allow_html=True)
if not st.session_state["f_motion"]:
    st.markdown(ui.NO_MOTION_CSS, unsafe_allow_html=True)

from pan_hr import privacy as P  # noqa: E402

st.markdown(ui.hero(P.fmt_n(len(view)), len(df)), unsafe_allow_html=True)
page = st.radio("View", NAV, horizontal=True, key="view", label_visibility="collapsed")

if roles_sel or risk_sel:
    chips = "".join(f'<span class="chip active">{ui.esc(x)}</span>' for x in roles_sel + risk_sel)
    st.markdown(f'<div class="section-cap">Chart selection cross-filtering all views: {chips} '
                '- use "Reset all filters and selections" to clear.</div>', unsafe_allow_html=True)

if page == "Data & Method":
    views.method(ctx)
elif len(view) < C.K_MIN:
    st.markdown(ui.note(
        f"Fewer than {C.K_MIN} records match the current filters, so results are withheld to protect privacy. "
        "Widen the filters or press Reset.", "caution"), unsafe_allow_html=True)
else:
    {"Engagement Health": views.overview, "Burnout Risk": views.burnout,
     "Role & Career Stage": views.career, "Manager Action Panel": views.actions}[page](ctx)

st.markdown('<div class="footer">Descriptive, cross-sectional analysis of the provided dataset. Scores and screening levels are '
            'aggregate prompts for investigating work conditions - not diagnoses, predictions, causal estimates or inputs to '
            'individual employment decisions.</div>', unsafe_allow_html=True)
