"""Streamlit views. All rendering consumes privacy-protected aggregates from pan_hr.analysis."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from . import analysis as A
from . import charts as H
from . import config as C
from . import privacy as P
from . import ui
from .filters import FilterState

PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


@dataclass
class Ctx:
    full: pd.DataFrame            # whole analysed dataset (for comparisons only)
    view: pd.DataFrame            # sidebar filters + all cross-filters
    role_frame: pd.DataFrame      # sidebar + risk cross-filter (role chart source)
    risk_frame: pd.DataFrame      # sidebar + role cross-filter (risk chart source)
    f: FilterState
    motion: bool
    rev: int
    report: object
    nulled: dict


def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def _points(key: str, axis: str) -> list[str]:
    try:
        return [p[axis] for p in st.session_state[key]["selection"]["points"] if p.get(axis) is not None]
    except (KeyError, TypeError):
        return []


def plot(fig, key: str, store: str | None = None, axis: str = "y", alt: str = "") -> None:
    """Render a Plotly chart. With `store`, point selections are persisted in
    st.session_state[store] (via a callback, so they survive switching views)."""
    if store is None:
        st.plotly_chart(fig, width="stretch", theme=None, key=key, config=PLOT_CONFIG, alt=alt or None)
        return

    def _save() -> None:
        st.session_state[store] = _points(key, axis)

    st.plotly_chart(fig, width="stretch", theme=None, key=key, on_select=_save,
                    selection_mode=("points",), config=PLOT_CONFIG, alt=alt or None)


def pct(x, digits: int = 0) -> str:
    return "withheld (<5)" if x is None or pd.isna(x) else f"{x:.{digits}%}"


def shares_for(d: pd.DataFrame, threshold: float) -> dict:
    n = len(d)
    n_cei = int(d["CEI"].notna().sum())
    return {
        "n": n,
        "below": P.safe_share(int((d["CEI"] < threshold).sum()), n_cei),
        "high": P.safe_share(int((d["BurnoutRisk"] == "High").sum()), n),
        "med": P.safe_share(int((d["BurnoutRisk"] == "Medium").sum()), n),
        "low": P.safe_share(int((d["BurnoutRisk"] == "Low").sum()), n),
        "ot": P.safe_share(int(d["OvertimeFlag"].sum()), n),
        "lowwlb": P.safe_share(int(d["LowWLBFlag"].sum()), n),
    }


def scope_line(c: Ctx) -> None:
    n = len(c.view)
    md(ui.note(f"Showing {n:,} of {len(c.full):,} records ({n / len(c.full):.0%}). Filters - {c.f.describe()}. "
               "Every chart and number on this page is computed on exactly these records.", ""))


# --------------------------------------------------------------------------- #
def overview(c: Ctx) -> None:
    d, thr = c.view, c.f.threshold
    scope_line(c)
    m, lo, hi, n_cei = A.mean_ci(d["CEI"])
    full_mean = c.full["CEI"].mean()
    s = shares_for(d, thr)
    wlb_mean = pd.to_numeric(d["WorkLifeBalance"], errors="coerce").mean()
    md(ui.kpi_row([
        ui.kpi("Mean CEI (0-100)", f"{m:.1f}", f"95% CI {lo:.1f}-{hi:.1f} | {m - full_mean:+.1f} vs full dataset"),
        ui.kpi("Records in view", f"{len(d):,}", f"{len(d) / len(c.full):.0%} of dataset | {A.precision_note(n_cei)}"),
        ui.kpi(f"Below CEI {thr:g}", pct(s["below"]), "share of records in view; threshold set in sidebar"),
        ui.kpi("Mean work-life balance", f"{wlb_mean:.2f} / 4", "Work-Life Balance Index (average rating)"),
        ui.kpi("Satisfaction stability", f"{d['SatisfactionStability'].mean():.0f} / 100",
               "proposed heuristic: agreement across the 4 CEI items"),
    ]))

    left, right = st.columns([1, 1])
    with left:
        md(ui.section("How engagement is distributed", "CEI = (involvement + job + environment + relationship satisfaction) / 16 x 100. Bins with fewer than 5 records are merged with a neighbour."))
        plot(H.cei_histogram(A.cei_histogram(d), m, c.motion), f"cei_hist_{c.rev}",
             alt="Histogram of Composite Engagement Index values in the current view")
    with right:
        md(ui.section("Satisfaction distributions", "Share of records giving each 1-4 rating. Grey hatched = cells under 5 records, combined."))
        dists = {col: A.score_distribution(d, col) for col in C.CEI_COLUMNS + ["WorkLifeBalance"]}
        plot(H.satisfaction_stack(dists, c.motion), f"sat_stack_{c.rev}",
             alt="Stacked bars of rating shares for each satisfaction dimension")

    md(ui.section("Department and role comparison",
                  "Dots are mean CEI with 95% confidence intervals; the axis is zoomed, so differences look larger than they are - read the scale. "
                  "Click a role to cross-filter every view; click again or use Reset to clear."))
    a, b = st.columns([1, 1.4])
    with a:
        dept = A.grouped_cei(d, "Department", thr, nominal=True)
        plot(H.group_bars(dept, comparator=m, motion=c.motion, title_axis="Mean CEI"), f"dept_chart_{c.rev}",
             alt="Mean CEI by department with confidence intervals")
    with b:
        roles = A.grouped_cei(c.role_frame, "JobRole", thr, nominal=True)
        plot(H.group_bars(roles, comparator=m, selected=c.f.cross_roles, motion=c.motion),
             f"role_chart_{c.rev}", store="xf_roles", axis="y", alt="Mean CEI by job role with confidence intervals")
    md(ui.note("Differences between departments and roles are small relative to their confidence intervals in this dataset; "
               "treat them as prompts for local review rather than rankings. Also note: the four CEI items are almost uncorrelated here "
               "(see Data & Method), so CEI is a descriptive average of four separate ratings rather than one tightly-measured construct.", ""))

    md(ui.section("Engagement vs attrition (contextual association only)",
                  "Attrition = 1 marks records of people who left. The dataset is a single snapshot, so it cannot show whether engagement was measured before or after an exit decision."))
    comp = A.compare_attrition(d)
    x, y = st.columns([1, 1.3])
    with x:
        plot(H.attrition_pair(comp, c.motion), f"attr_pair_{c.rev}", alt="Mean CEI for stayed versus left records")
    with y:
        if comp.reportable:
            md(ui.note(f"Records marked as left have a mean CEI {abs(comp.diff):.1f} points {'lower' if comp.diff > 0 else 'higher'} than records marked as stayed "
                       f"(difference 95% CI {comp.ci_lo:.1f} to {comp.ci_hi:.1f}; standardised effect d = {comp.cohens_d:.2f}; "
                       f"stayed n = {comp.n_stayed:,}, left n = {P.fmt_n(comp.n_left)}). "
                       "This is a descriptive association. It does not show that low engagement causes leaving, and it cannot be used to predict who will leave.", "caution"))
        by_risk = A.attrition_by_risk(d)
        if not by_risk.empty:
            t = by_risk.assign(**{"Screening level": by_risk["level"], "Records": by_risk["n"].map(P.fmt_n),
                                  "Share marked as left": by_risk["attrition_rate"].map(pct)})[
                ["Screening level", "Records", "Share marked as left"]]
            st.dataframe(t, hide_index=True, width="stretch")
            st.caption("Descriptive only. Attrition shares are withheld where a cell would be under 5 records.")

    with st.expander("Optional fairness audit (aggregated; audit only - never a risk or action rule)"):
        if st.toggle("Show aggregated comparison by gender and marital status", key=f"fair_{c.rev}"):
            md(ui.note("Shown only to check whether engagement differs across groups. These attributes are NOT used to assign risk, rank areas or recommend actions.", "caution"))
            for col, title in (("Gender", "Gender"), ("MaritalStatus", "Marital status")):
                ft = A.grouped_cei(d, col, thr, nominal=True)
                st.markdown(f"**{title}**")
                plot(H.group_bars(ft, comparator=m, motion=c.motion, height=200), f"fair_{col}_{c.rev}")


# --------------------------------------------------------------------------- #
def burnout(c: Ctx) -> None:
    d, thr = c.view, c.f.threshold
    scope_line(c)
    md(ui.note(C.SCREENING_DISCLAIMER + " High = OverTime Yes AND WorkLifeBalance <= 2; Medium = exactly one; Low = neither. "
               "It reflects one snapshot - persistence over several quarters cannot be assessed here.", "caution"))
    s = shares_for(d, thr)
    wsi = d["WSI"].mean()
    md(ui.kpi_row([
        ui.kpi("High screening level", pct(s["high"], 1), "overtime + low work-life balance", "risk-high"),
        ui.kpi("Medium", pct(s["med"], 1), "exactly one of the two conditions", "risk-med"),
        ui.kpi("Low", pct(s["low"], 1), "neither condition", "risk-low"),
        ui.kpi("Works overtime", pct(s["ot"]), "share of records in view"),
        ui.kpi("Work-life balance 1-2", pct(s["lowwlb"]), "share rating balance 'poor' or 'below average'"),
        ui.kpi("Workload stress (WSI)", f"{wsi:.0f} / 100", "proposed heuristic: travel + commute + overtime"),
    ]))

    a, b = st.columns([1, 1.15])
    with a:
        md(ui.section("Screening-level mix", "Click a bar to cross-filter all views to that level."))
        mix = A.risk_mix(c.risk_frame)
        plot(H.risk_bars(mix, c.f.risk_levels, c.motion), f"risk_chart_{c.rev}", store="xf_risk", axis="x",
             alt="Share of records at each burnout screening level")
    with b:
        md(ui.section("Overtime x work-life balance", "Cell colour and bold number = mean CEI, with n beneath. Blank = fewer than 5 records."))
        n, mean, _ = A.crosstab_cells(d, "OverTime", "WorkLifeBalance")
        plot(H.heatmap(n, mean, "Work-life balance rating (1 = poor, 4 = excellent)", "Works overtime", c.motion),
             f"ot_wlb_{c.rev}", alt="Heatmap of mean CEI by overtime and work-life balance")

    md(ui.section("Where the mix differs", "Stacked shares within each group. Hatched grey = cells under 5 records, combined."))
    x, y = st.columns(2)
    with x:
        st.markdown("**By department**")
        plot(H.risk_stack(A.risk_mix_by(d, "Department"), c.motion), f"risk_dept_{c.rev}")
    with y:
        st.markdown("**By job level**")
        order = [f"Level {i}" for i in range(1, 6)]
        plot(H.risk_stack(A.risk_mix_by(d, "JobLevelLabel", order, nominal=False), c.motion), f"risk_level_{c.rev}")

    md(ui.section("Engagement across workload conditions",
                  "Mean CEI (95% CI) by overtime, travel frequency and commute-distance band. Commute bands use the dataset's distance values; the unit is not documented."))
    p, q, r = st.columns(3)
    mean_all = d["CEI"].mean()
    with p:
        st.markdown("**Overtime**")
        t = A.grouped_cei(d.assign(_o=d["OverTime"].map({"Yes": "Overtime", "No": "No overtime"})), "_o", thr,
                          order=["No overtime", "Overtime"])
        plot(H.group_bars(t, comparator=mean_all, horizontal=False, motion=c.motion, height=290), f"wl_ot_{c.rev}")
    with q:
        st.markdown("**Business travel**")
        t = A.grouped_cei(d.assign(_t=d["BusinessTravel"].map(C.TRAVEL_LABELS)), "_t", thr,
                          order=[C.TRAVEL_LABELS[k] for k in C.TRAVEL_ORDER])
        plot(H.group_bars(t, comparator=mean_all, horizontal=False, motion=c.motion, height=290), f"wl_tr_{c.rev}")
    with r:
        st.markdown("**Commute distance**")
        t = A.grouped_cei(d, "CommuteBand", thr, order=[b[0] for b in C.COMMUTE_BANDS])
        plot(H.group_bars(t, comparator=mean_all, horizontal=False, motion=c.motion, height=290), f"wl_cm_{c.rev}")


# --------------------------------------------------------------------------- #
def career(c: Ctx) -> None:
    d, thr = c.view, c.f.threshold
    scope_line(c)
    m = d["CEI"].mean()
    n_csi = int(d["CSI"].notna().sum())
    md(ui.kpi_row([
        ui.kpi("Mean years at company", f"{d['YearsAtCompany'].mean():.1f}", "records in view"),
        ui.kpi("Mean years since promotion", f"{d['YearsSinceLastPromotion'].mean():.1f}", "records in view"),
        ui.kpi("Career Stagnation Index", f"{d['CSI'].mean():.0f} / 100" if n_csi >= C.K_MIN else "withheld",
               f"proposed heuristic | assessed for {P.fmt_n(n_csi)} records (tenure >= {C.CSI_MIN_TENURE_YEARS} yrs)"),
        ui.kpi("Mean CEI", f"{m:.1f}", "records in view"),
    ]))

    md(ui.section("Engagement by job level",
                  "Level 1 = entry; higher = more senior. Levels with fewer than 5 records are merged with the neighbouring level."))
    a, b = st.columns([1, 1.35])
    with a:
        lv = A.grouped_cei(d, "JobLevelLabel", thr, order=[f"Level {i}" for i in range(1, 6)])
        plot(H.group_bars(lv, comparator=m, horizontal=False, motion=c.motion), f"lvl_{c.rev}")
    with b:
        n, mean, _ = A.crosstab_cells(d, "JobRole", "JobLevelLabel")
        plot(H.heatmap(n, mean, "Job level", "Job role", c.motion), f"role_lvl_{c.rev}",
             alt="Heatmap of mean CEI by job role and job level")

    md(ui.section("Tenure and engagement", "Mean CEI by tenure band with 95% CI band. Bands under 5 records are merged with a neighbour."))
    x, y = st.columns(2)
    with x:
        st.markdown("**Years at company**")
        t = A.grouped_cei(d, "TenureBand", thr, order=[b[0] for b in C.TENURE_BANDS])
        plot(H.tenure_lines({"Years at company": t}, c.motion), f"ten_co_{c.rev}")
    with y:
        st.markdown("**Years in current role**")
        t = A.grouped_cei(d, "RoleTenureBand", thr, order=[b[0] for b in C.ROLE_TENURE_BANDS])
        plot(H.tenure_lines({"Years in current role": t}, c.motion), f"ten_role_{c.rev}")

    md(ui.section("Stagnation indicators (transparent, proposed)",
                  "The Career Stagnation Index is a proposed heuristic - the source documents do not define its formula - and is not a validated measure."))
    p, q = st.columns(2)
    with p:
        st.markdown("**Mean CEI by years since last promotion**")
        t = A.grouped_cei(d, "PromotionGapBand", thr, order=[b[0] for b in C.PROMOTION_GAP_BANDS])
        plot(H.group_bars(t, comparator=m, horizontal=False, motion=c.motion, height=300), f"promo_{c.rev}")
    with q:
        st.markdown("**Mean CEI by Career Stagnation Index band**")
        t = A.grouped_cei(d.dropna(subset=["CSIBand"]), "CSIBand", thr, order=[b[0] for b in C.CSI_BANDS])
        plot(H.group_bars(t, comparator=m, horizontal=False, motion=c.motion, height=300), f"csi_{c.rev}")

    with st.expander("How the Career Stagnation Index is built (inspect the components)"):
        st.markdown(
            "**CSI = 100 x [ 0.5 x (YearsSinceLastPromotion / YearsAtCompany) + 0.5 x (YearsInCurrentRole / YearsAtCompany) ]**, "
            "each ratio clipped to 0-1, defined only when YearsAtCompany >= 2 (shorter tenures give mechanically extreme ratios and are shown as *not assessed*). "
            "Equal weights are an assumption, not an estimate. Bands: <40 lower, 40-69 moderate, 70+ higher (also assumptions).")
        lab = P.protect_ordinal(d["CSIBand"], [b[0] for b in C.CSI_BANDS])
        comp = d.assign(_g=lab).dropna(subset=["_g"]).groupby("_g").agg(
            Records=("CEI", "size"), unpromoted=("csi_unpromoted_share", "mean"),
            role=("csi_role_share", "mean"), CEI=("CEI", "mean")).reset_index()
        if not comp.empty:
            comp = comp.rename(columns={"_g": "CSI band", "unpromoted": "Mean unpromoted share", "role": "Mean role share", "CEI": "Mean CEI"})
            st.dataframe(comp.round(2), hide_index=True, width="stretch")
    md(ui.note("Lower engagement among longer-tenured or less-recently-promoted groups would be an association in a snapshot; it does not establish that stagnation causes disengagement.", ""))


# --------------------------------------------------------------------------- #
def actions(c: Ctx) -> None:
    d, thr = c.view, c.f.threshold
    scope_line(c)
    md(ui.note("These alerts are prompts to investigate work conditions in aggregate areas - not assessments of any employee's performance. "
               "No individual records or rankings are shown, and areas under 5 records are combined or withheld.", "caution"))
    pr = A.priority_areas(d, thr)
    if pr.empty:
        md(ui.note("Not enough records per area in the current filters to assess areas (minimum 5). Widen the filters.", ""))
        return
    s = shares_for(d, thr)
    if c.f.overtime != "All":
        md(ui.note("The overtime filter is on, and overtime is one of the two screening conditions. With 'Yes' the high-screening share simply equals the share with "
                   "work-life balance 1-2 (so the workload prompt triggers easily); with 'No' it is zero by construction. Interpret workload prompts with overtime = All.", "caution"))
    flagged = pr[pr["n_flags"] > 0]
    md(ui.kpi_row([
        ui.kpi("Areas assessed", str(len(pr)), "Department x role groups with >= 5 records"),
        ui.kpi("Areas with >= 1 prompt", str(len(flagged)), "under the rules listed below"),
        ui.kpi(f"Records below CEI {thr:g}", pct(s["below"]), "share of records in view"),
        ui.kpi("High screening level", pct(s["high"], 1), f"plan prompt level: {C.HIGH_BURNOUT_SHARE_PROMPT:.0%}", "risk-high"),
    ]))

    md(ui.section("Gap between area engagement and its comparator",
                  "Comparator = the area's department mean when the department has other areas in view, otherwise the view mean. Colour = number of prompts triggered."))
    plot(H.gap_bars(pr, c.motion), f"gap_{c.rev}", alt="Engagement gap by area versus comparator")

    md(ui.section("Prioritised areas for human review", "Ordered by number of prompts, then by engagement gap. Aggregate areas only."))
    if flagged.empty:
        md(ui.note("No area triggers any prompt under the current rules and filters. That is a finding about this dataset: engagement and workload indicators are fairly uniform across areas.", "good"))
    for _, r in flagged.head(8).iterrows():
        tcls = {"Review first": "t3", "Review next": "t2", "Monitor": "t1"}.get(r["tier"], "")
        flags_html = "".join(f"<li><b>{ui.esc(A.FLAG_TEXT[f])}</b> - {ui.esc(A.ACTIONS[f])}</li>" for f in r["flags"])
        csi_txt = "withheld" if pd.isna(r["csi_mean"]) else f"{r['csi_mean']:.0f}"
        high_txt = pct(r["pct_high_burnout"], 1)
        meta = (f"n = {int(r['n']):,} | mean CEI {r['cei_mean']:.1f} ({r['cei_gap']:+.1f} vs {r['baseline_basis']}) | "
                f"high-screening share {high_txt} | mean WSI {r['wsi_mean']:.0f} | mean CSI {csi_txt}")
        md(f'<div class="area-card {tcls}"><h4>{ui.esc(r["group"])} <span class="chip active">{ui.esc(r["tier"])}</span></h4>'
           f'<div class="meta">{ui.esc(meta)}</div><ul>{flags_html}</ul>'
           '<div class="meta">Next step: validate with local managers and team conversations before any change.</div></div>')

    st.markdown("**All assessed areas**")
    t = pd.DataFrame({
        "Area": pr["group"], "Records": pr["n"], "Mean CEI": pr["cei_mean"].round(1),
        "Gap vs comparator": pr["cei_gap"].round(1), "Comparator": pr["baseline_basis"],
        f"% below CEI {thr:g}": pr["pct_below"].map(pct), "% high screening": pr["pct_high_burnout"].map(lambda x: pct(x, 1)),
        "Mean WSI": pr["wsi_mean"].round(0), "Mean CSI": pr["csi_mean"].round(0),
        "Prompts": pr["flags"].map(lambda fl: ", ".join(fl) if fl else "none"), "Tier": pr["tier"],
    })
    st.dataframe(t, hide_index=True, width="stretch")
    st.download_button("Download this aggregate table (CSV)", t.to_csv(index=False).encode(), "aggregate_priority_areas.csv", "text/csv")

    with st.expander("Rules behind the prompts (proposed heuristics - editable in pan_hr/config.py)"):
        st.markdown(
            f"- **Workload prompt:** high-screening share >= {C.HIGH_BURNOUT_SHARE_PROMPT:.0%} (threshold from the strategic plan). Evaluated only when the share is reportable (cells under 5 are withheld).\n"
            f"- **Engagement prompt:** mean CEI at least {C.ENGAGEMENT_GAP_POINTS:g} points below the comparator.\n"
            f"- **Stagnation prompt:** mean CSI at least {C.STAGNATION_GAP_POINTS:g} points above the view mean.\n"
            f"- **Workload-stress prompt:** mean WSI at least {C.WORKLOAD_GAP_POINTS:g} points above the view mean.\n"
            "- Tier: 3+ prompts = Review first, 2 = Review next, 1 = Monitor. Thresholds are judgement calls, not validated cut-offs, and have not been tested against outcomes.")


# --------------------------------------------------------------------------- #
def method(c: Ctx) -> None:
    rep = c.report
    md(ui.section("Dataset coverage and scope"))
    full = c.full
    unclass = int((full["BurnoutRisk"] == C.BURNOUT_UNCLASSIFIED).sum())
    md(ui.kpi_row([
        ui.kpi("Records", f"{rep.n_rows:,}", "rows in the supplied CSV"),
        ui.kpi("Fields", str(rep.n_cols), "31 expected"),
        ui.kpi("Validation", "Passed" if rep.passed else "FAILED", f"{len(rep.warnings)} warnings, {len(rep.failed)} failures"),
        ui.kpi("CEI computable", f"{int(full['CEI'].notna().sum()):,}", "records with all four inputs"),
        ui.kpi("Burnout level unclassified", str(unclass), "overtime or balance missing"),
        ui.kpi("CSI assessed", f"{int(full['CSI'].notna().sum()):,}", f"tenure >= {C.CSI_MIN_TENURE_YEARS} years"),
    ]))
    md(ui.note("The CSV does not prove these are actual Palo Alto Networks employee records, and nothing in it is externally benchmarked. "
               "Every result applies to the provided dataset only.", "caution"))
    if any(c.nulled.values()):
        md(ui.note(f"Out-of-range scale values set to missing in the analysis copy: {c.nulled}. Source data unchanged.", "caution"))
    else:
        md(ui.note("No invalid or missing scale values were found, so nothing was imputed, dropped or recoded. The source data is displayed and analysed as supplied.", "good"))
    st.dataframe(rep.to_frame(), hide_index=True, width="stretch")
    with st.expander("Field summary"):
        st.dataframe(rep.column_summary, hide_index=True, width="stretch")

    md(ui.section("Metric definitions"))
    defs = pd.DataFrame([
        ["Composite Engagement Index (CEI)", "Source-defined formula",
         "(JobInvolvement + JobSatisfaction + EnvironmentSatisfaction + RelationshipSatisfaction) / 16 x 100. Range 25-100 because each item is at least 1. Any missing input -> CEI not computed (no imputation)."],
        ["Burnout screening level", "Source-defined rule",
         "High: OverTime = Yes AND WorkLifeBalance <= 2. Medium: exactly one. Low: neither. Unknown input -> Unclassified. A screening proxy, not a diagnosis or validated prediction."],
        ["Work-Life Balance Index", "Source-defined",
         "Mean WorkLifeBalance rating (1-4)."],
        ["Satisfaction Stability Score", "Proposed heuristic",
         "100 x (1 - population SD of the four CEI items / 1.5). 100 = all four ratings agree. Not validated."],
        ["Workload Stress Indicator (WSI)", "Proposed heuristic",
         "100 x mean of three 0-1 components: travel (none 0, rarely 0.5, frequently 1), commute ((distance-1)/28, clipped; dataset-observed range, unit undocumented), overtime (yes 1). Equal weights are an assumption."],
        ["Career Stagnation Index (CSI)", "Proposed heuristic",
         "100 x [0.5 x YearsSinceLastPromotion/YearsAtCompany + 0.5 x YearsInCurrentRole/YearsAtCompany], ratios clipped 0-1, only if YearsAtCompany >= 2."],
    ], columns=["Metric", "Status", "Definition and assumptions"])
    st.dataframe(defs, hide_index=True, width="stretch")

    md(ui.section("Key limitations"))
    st.markdown(
        "- **Snapshot, not longitudinal.** Persistence of overtime or low work-life balance over several quarters cannot be confirmed; no multi-quarter monitoring exists here.\n"
        "- **Associations only.** Attrition and all score differences are descriptive. No causal, predictive or turnover-reduction claim is supported by this dataset.\n"
        "- **CEI internal consistency is very low.** In this dataset the four CEI items are essentially uncorrelated (mean inter-item r about 0, Cronbach's alpha about 0), so CEI is a descriptive average of four separate ratings, not evidence of a single engagement construct.\n"
        "- **Counter-intuitive patterns are reported as found.** Records with overtime have slightly higher, not lower, mean CEI, and CEI barely varies by role, level, tenure, travel or commute.\n"
        "- **Proxy measures.** Burnout level uses two fields; WSI and CSI are proposed heuristics with assumed weights and bands.\n"
        "- **Uncertainty.** Smaller groups have wider confidence intervals; many between-group CEI gaps are within them.\n"
        "- **Source-document criteria not testable here.** The strategic plan's multi-quarter persistence, turnover-reduction target and prediction-accuracy benchmarks require longitudinal data that this file does not contain.")
    md(ui.section("Privacy and fairness safeguards"))
    st.markdown(
        f"- Aggregates only; no employee-level rows, tooltips or rankings. Minimum cohort size **{C.K_MIN}** applied to groups, bands, rating cells and filter combinations; "
        "small groups are pooled or withheld, and a lone withheld cell is paired with its smallest neighbour so it cannot be recovered by subtraction.\n"
        "- Gender and marital status are never used to assign risk or actions; an optional aggregated audit is available on the overview.\n"
        "- Residual risk: repeatedly changing filters by one record's worth could still allow differencing; deploy behind access control with query logging.\n"
        "- Data stays local: the app makes no network calls and has no tracking or credentials.")
