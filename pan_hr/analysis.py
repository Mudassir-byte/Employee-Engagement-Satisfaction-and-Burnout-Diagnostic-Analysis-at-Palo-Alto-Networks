"""Privacy-protected aggregations used by the dashboard, tests and paper script."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import privacy as P


# --------------------------------------------------------------------------- #
# Basic statistics
# --------------------------------------------------------------------------- #
def mean_ci(values: pd.Series, z: float = 1.96) -> tuple[float, float, float, int]:
    """Mean and normal-approximation 95% CI half-range (mean, lo, hi, n) for non-missing values."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    m = float(v.mean())
    if n < 2:
        return (m, np.nan, np.nan, n)
    se = float(v.std(ddof=1)) / math.sqrt(n)
    return (m, m - z * se, m + z * se, n)


def precision_note(n: int) -> str:
    if n < 30:
        return "low precision (n<30)"
    if n < 100:
        return "moderate precision"
    return "good precision"


# --------------------------------------------------------------------------- #
# Group summaries
# --------------------------------------------------------------------------- #
def summarize_groups(df: pd.DataFrame, labels: pd.Series, threshold: float, k: int = C.K_MIN) -> pd.DataFrame:
    """One row per (already protected) group label with CEI, risk and workload metrics.

    `labels` must already have passed through a privacy.protect_* function; rows whose
    label is NaN are ignored. Sensitive shares are withheld via privacy.safe_share.
    """
    d = df.assign(_g=labels).dropna(subset=["_g"])
    rows = []
    for g, sub in d.groupby("_g", sort=False):
        n = len(sub)
        if n < k:  # defensive: never emit a small group
            continue
        m, lo, hi, n_cei = mean_ci(sub["CEI"])
        below = int((sub["CEI"] < threshold).sum())
        high = int((sub["BurnoutRisk"] == "High").sum())
        ot = int(sub["OvertimeFlag"].sum())
        lowwlb = int(sub["LowWLBFlag"].sum())
        left = int(sub["Attrition"].sum())
        csi = sub["CSI"].dropna()
        rows.append({
            "group": g, "n": n,
            "cei_mean": m, "cei_lo": lo, "cei_hi": hi, "cei_n": n_cei,
            "pct_below": P.safe_share(below, n_cei, k),
            "pct_overtime": P.safe_share(ot, n, k),
            "pct_low_wlb": P.safe_share(lowwlb, n, k),
            "pct_high_burnout": P.safe_share(high, n, k),
            "high_count_reportable": high >= k or high == 0,
            "wlb_mean": float(pd.to_numeric(sub["WorkLifeBalance"], errors="coerce").mean()),
            "wsi_mean": float(sub["WSI"].mean()),
            "csi_mean": float(csi.mean()) if len(csi) >= k else np.nan,
            "csi_n": len(csi),
            "attrition_rate": P.safe_share(left, n, k),
            "precision": precision_note(n_cei),
        })
    return pd.DataFrame(rows)


def grouped_cei(df: pd.DataFrame, col: str, threshold: float, order: list[str] | None = None,
                nominal: bool = False, k: int = C.K_MIN) -> pd.DataFrame:
    """CEI summary by a column with the appropriate small-cohort protection applied."""
    if df.empty:
        return pd.DataFrame()
    raw = df[col]
    if nominal or not order:
        lab = P.protect_nominal(raw, k)
        out = summarize_groups(df, lab, threshold, k)
        return out.sort_values("cei_mean", ascending=False).reset_index(drop=True) if not out.empty else out
    lab = P.protect_ordinal(raw, order, k)
    out = summarize_groups(df, lab, threshold, k)
    if out.empty:
        return out
    ordering = {g: i for i, g in enumerate(P.ordered_display_labels(lab, order))}
    return out.assign(_o=out["group"].map(ordering)).sort_values("_o").drop(columns="_o").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def score_distribution(df: pd.DataFrame, col: str, k: int = C.K_MIN) -> pd.DataFrame:
    """Share of respondents at each 1-4 score, with small cells withheld."""
    counts = pd.to_numeric(df[col], errors="coerce").value_counts().reindex([1, 2, 3, 4], fill_value=0)
    shown, pooled = P.withhold_cells(counts, k)
    total = counts.sum()
    out = pd.DataFrame({"score": counts.index, "count": shown.values})
    out["share"] = out["count"] / total if total else np.nan
    out.attrs.update(total=int(total), pooled=pooled, pooled_share=(pooled / total) if total else np.nan)
    return out


def risk_mix(df: pd.DataFrame, k: int = C.K_MIN) -> pd.DataFrame:
    counts = df["BurnoutRisk"].value_counts().reindex(C.BURNOUT_LEVELS + [C.BURNOUT_UNCLASSIFIED], fill_value=0)
    if counts[C.BURNOUT_UNCLASSIFIED] == 0:
        counts = counts.drop(C.BURNOUT_UNCLASSIFIED)
    shown, pooled = P.withhold_cells(counts, k)
    total = counts.sum()
    out = pd.DataFrame({"level": counts.index, "count": shown.values})
    out["share"] = out["count"] / total if total else np.nan
    out.attrs.update(total=int(total), pooled=pooled, pooled_share=(pooled / total) if total else np.nan)
    return out


def cei_histogram(df: pd.DataFrame, k: int = C.K_MIN, width: int = 10) -> pd.DataFrame:
    """CEI histogram in fixed-width bins (20-29 ... 90-100); bins with 0<n<k merge into a neighbour."""
    cei = df["CEI"].dropna()
    starts = list(range(20, 100, width))
    edges = starts + [101]
    labels = [f"{a}-{a + width - 1}" for a in starts[:-1]] + [f"{starts[-1]}-100"]
    cat = pd.cut(cei, bins=edges, labels=labels, right=False)
    lab = P.protect_ordinal(cat.astype(object), labels, k)
    out = lab.value_counts().rename_axis("bin").reset_index(name="count")
    order = {b: i for i, b in enumerate(P.ordered_display_labels(lab, labels))}
    out = out.assign(_o=out["bin"].map(order)).sort_values("_o").drop(columns="_o").reset_index(drop=True)
    out["share"] = out["count"] / out["count"].sum() if len(out) else np.nan
    return out


def crosstab_cells(df: pd.DataFrame, rows: str, cols: str, value: str = "CEI", k: int = C.K_MIN):
    """n and mean-value matrices for a two-way breakdown with small cells blanked."""
    n = pd.crosstab(df[rows], df[cols])
    mean = df.pivot_table(index=rows, columns=cols, values=value, aggfunc="mean")
    mean = mean.reindex(index=n.index, columns=n.columns)
    mask = P.mask_matrix(n, k) | (n == 0)
    return n.where(~mask), mean.where(~mask), mask


# --------------------------------------------------------------------------- #
# Burnout-risk breakdown (stacked mix by category)
# --------------------------------------------------------------------------- #
def risk_mix_by(df: pd.DataFrame, col: str, order: list[str] | None = None, nominal: bool = True,
                k: int = C.K_MIN) -> pd.DataFrame:
    """Long table (group, level, share, count) with per-group cell withholding."""
    if df.empty:
        return pd.DataFrame()
    lab = P.protect_nominal(df[col], k) if nominal else P.protect_ordinal(df[col], order or [], k)
    d = df.assign(_g=lab).dropna(subset=["_g"])
    rows = []
    for g, sub in d.groupby("_g", sort=False):
        counts = sub["BurnoutRisk"].value_counts().reindex(C.BURNOUT_LEVELS, fill_value=0)
        shown, pooled = P.withhold_cells(counts, k)
        total = counts.sum()
        for lev in C.BURNOUT_LEVELS:
            rows.append({"group": g, "n": total, "level": lev, "count": shown[lev], "share": shown[lev] / total})
        if pooled:
            rows.append({"group": g, "n": total, "level": C.WITHHELD_LABEL, "count": pooled, "share": pooled / total})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order_map = None
    if not nominal and order:
        order_map = {g: i for i, g in enumerate(P.ordered_display_labels(lab, order))}
        out["_o"] = out["group"].map(order_map)
        out = out.sort_values("_o").drop(columns="_o")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Engagement vs attrition (contextual only)
# --------------------------------------------------------------------------- #
@dataclass
class AttritionComparison:
    n_stayed: int
    n_left: int
    mean_stayed: float
    mean_left: float
    diff: float          # stayed - left
    ci_lo: float
    ci_hi: float
    cohens_d: float
    reportable: bool


def compare_attrition(df: pd.DataFrame, k: int = C.K_MIN) -> AttritionComparison:
    a = df.loc[df["Attrition"] == 0, "CEI"].dropna()
    b = df.loc[df["Attrition"] == 1, "CEI"].dropna()
    if len(a) < k or len(b) < k:
        return AttritionComparison(len(a), len(b), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False)
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    pooled_sd = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return AttritionComparison(len(a), len(b), float(a.mean()), float(b.mean()), float(diff),
                               float(diff - 1.96 * se), float(diff + 1.96 * se),
                               float(diff / pooled_sd) if pooled_sd else np.nan, True)


def attrition_by_risk(df: pd.DataFrame, k: int = C.K_MIN) -> pd.DataFrame:
    rows = []
    for lev in C.BURNOUT_LEVELS:
        sub = df[df["BurnoutRisk"] == lev]
        n = len(sub)
        if n < k:
            continue
        left = int(sub["Attrition"].sum())
        rows.append({"level": lev, "n": n, "attrition_rate": P.safe_share(left, n, k)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Manager Action Panel
# --------------------------------------------------------------------------- #
FLAG_TEXT = {
    "workload": "High-burnout screening share at or above 15%",
    "engagement": "Mean engagement notably below comparator",
    "stagnation": "Elevated career-stagnation heuristic",
    "stress": "Elevated workload-stress heuristic",
}
ACTIONS = {
    "workload": "Review overtime allocation, on-call/incident rotation and staffing coverage with the team; "
                "check whether protected recovery time (e.g. time off after peak periods) is realistic.",
    "engagement": "Hold a structured listening session (anonymous input) on workload, team environment and "
                  "manager support; compare with the department baseline before acting.",
    "stagnation": "Review promotion pipeline and role-rotation options; open career-development conversations "
                  "and check training access for long-tenured roles.",
    "stress": "Review travel scheduling and commute/remote-flexibility options; look for travel clustering "
              "on the same teams.",
}


def priority_areas(df: pd.DataFrame, threshold: float, k: int = C.K_MIN) -> pd.DataFrame:
    """Aggregate Department x Role areas with transparent rule-based flags (no individuals)."""
    if df.empty:
        return pd.DataFrame()
    lab = P.protect_nominal(df["Area"], k)
    summ = summarize_groups(df, lab, threshold, k)
    if summ.empty:
        return summ
    org_cei = df["CEI"].mean()
    org_csi = df["CSI"].mean()
    org_wsi = df["WSI"].mean()
    dept_of = df.assign(_g=lab).dropna(subset=["_g"]).groupby("_g")["Department"].agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else None)
    dept_mean = df.groupby("Department")["CEI"].mean()
    dept_n = df.groupby("Department").size()
    rows = []
    for _, r in summ.iterrows():
        dept = dept_of.get(r["group"])
        if dept is not None and dept_n.get(dept, 0) > r["n"]:
            baseline, basis = float(dept_mean[dept]), "department"
        else:
            baseline, basis = float(org_cei), "organisation (view)"
        gap = r["cei_mean"] - baseline
        flags = []
        share = r["pct_high_burnout"]
        if share is not None and not pd.isna(share) and share >= C.HIGH_BURNOUT_SHARE_PROMPT:
            flags.append("workload")
        if gap <= -C.ENGAGEMENT_GAP_POINTS:
            flags.append("engagement")
        if not pd.isna(r["csi_mean"]) and r["csi_mean"] - org_csi >= C.STAGNATION_GAP_POINTS:
            flags.append("stagnation")
        if not pd.isna(r["wsi_mean"]) and r["wsi_mean"] - org_wsi >= C.WORKLOAD_GAP_POINTS:
            flags.append("stress")
        tier = {0: "No flag", 1: "Monitor", 2: "Review next"}.get(len(flags), "Review first")
        rows.append({**r.to_dict(), "baseline": baseline, "baseline_basis": basis, "cei_gap": gap,
                     "flags": flags, "n_flags": len(flags), "tier": tier,
                     "share_withheld": share is None})
    out = pd.DataFrame(rows)
    return out.sort_values(["n_flags", "cei_gap"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Optional fairness check (audit only)
# --------------------------------------------------------------------------- #
def fairness_table(df: pd.DataFrame, col: str, threshold: float, k: int = C.K_MIN) -> pd.DataFrame:
    """Aggregated CEI / high-burnout share by a sensitive attribute. Audit only, never a decision rule."""
    lab = P.protect_nominal(df[col], k)
    out = summarize_groups(df, lab, threshold, k)
    return out[["group", "n", "cei_mean", "cei_lo", "cei_hi", "pct_below", "pct_high_burnout"]] if not out.empty else out
