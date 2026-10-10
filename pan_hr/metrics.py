"""Metric logic (pure pandas/numpy, no UI dependencies).

Measured / source-defined:  CEI, Burnout Risk (rule-based), Work-Life Balance Index.
Proposed heuristics (NOT validated, formulas not fully specified by source criteria):
    Workload Stress Indicator (WSI), Career Stagnation Index (CSI),
    Satisfaction Stability Score (SSS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Composite Engagement Index
# --------------------------------------------------------------------------- #
def compute_cei(df: pd.DataFrame) -> pd.Series:
    """CEI = (JI + JS + ES + RS) / 16 * 100.

    Range 25-100 (every item is scored 1-4, so the minimum possible total is 4).
    Complete-case rule: if ANY of the four inputs is missing the CEI is NaN; inputs
    are never imputed or averaged over fewer items.
    """
    items = df[C.CEI_COLUMNS].apply(pd.to_numeric, errors="coerce")
    total = items.sum(axis=1, min_count=len(C.CEI_COLUMNS))  # NaN unless all 4 present
    return total / C.CEI_MAX_TOTAL * 100.0


# --------------------------------------------------------------------------- #
# Burnout risk (rule-based screening proxy)
# --------------------------------------------------------------------------- #
def compute_burnout_risk(df: pd.DataFrame) -> pd.Series:
    """High: OverTime=Yes AND WLB<=2. Medium: exactly one. Low: neither.

    Records with a missing/unknown OverTime or WorkLifeBalance are 'Unclassified'
    (the rule cannot be evaluated) rather than being silently assigned a level.
    """
    ot = df["OverTime"]
    wlb = pd.to_numeric(df["WorkLifeBalance"], errors="coerce")
    known = ot.isin(["Yes", "No"]) & wlb.notna()
    cond_ot = ot.eq("Yes")
    cond_wlb = wlb.le(C.LOW_WLB_CUTOFF)
    n_true = cond_ot.astype(int) + cond_wlb.astype(int)
    level = pd.Series(C.BURNOUT_UNCLASSIFIED, index=df.index, dtype=object)
    level[known & (n_true == 0)] = "Low"
    level[known & (n_true == 1)] = "Medium"
    level[known & (n_true == 2)] = "High"
    return level


# --------------------------------------------------------------------------- #
# Satisfaction stability (proposed)
# --------------------------------------------------------------------------- #
def compute_satisfaction_stability(df: pd.DataFrame) -> pd.Series:
    """SSS = 100 * (1 - sd / 1.5) over the four CEI items (population sd).

    1.5 is the largest possible sd of four values in 1-4 (two 1s, two 4s), so SSS
    is 100 when all four ratings agree and 0 at maximal disagreement.
    """
    items = df[C.CEI_COLUMNS].apply(pd.to_numeric, errors="coerce")
    sd = items.std(axis=1, ddof=0).where(items.notna().all(axis=1))
    return (1 - sd / 1.5).clip(0, 1) * 100.0


# --------------------------------------------------------------------------- #
# Workload Stress Indicator (proposed heuristic)
# --------------------------------------------------------------------------- #
def wsi_components(df: pd.DataFrame) -> pd.DataFrame:
    travel = df["BusinessTravel"].map(C.TRAVEL_COEFFICIENTS)
    dist = pd.to_numeric(df["DistanceFromHome"], errors="coerce")
    commute = ((dist - C.COMMUTE_REF_MIN) / (C.COMMUTE_REF_MAX - C.COMMUTE_REF_MIN)).clip(0, 1)
    overtime = df["OverTime"].map({"Yes": 1.0, "No": 0.0})
    return pd.DataFrame({"wsi_travel": travel, "wsi_commute": commute, "wsi_overtime": overtime}, index=df.index)


def compute_wsi(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """WSI = 100 * (w_t*travel + w_c*commute + w_o*overtime); equal weights by default.

    travel: Non-Travel 0 / Rarely 0.5 / Frequently 1.  commute: (distance-1)/28 clipped
    to [0,1] (dataset-observed range; distance unit undocumented).  overtime: Yes=1.
    Any missing component -> NaN.
    """
    w = weights or C.WSI_WEIGHTS
    comp = wsi_components(df)
    total = comp["wsi_travel"] * w["travel"] + comp["wsi_commute"] * w["commute"] + comp["wsi_overtime"] * w["overtime"]
    return (total / sum(w.values()) * 100.0).where(comp.notna().all(axis=1))


# --------------------------------------------------------------------------- #
# Career Stagnation Index (proposed heuristic)
# --------------------------------------------------------------------------- #
def csi_components(df: pd.DataFrame) -> pd.DataFrame:
    yac = pd.to_numeric(df["YearsAtCompany"], errors="coerce")
    yslp = pd.to_numeric(df["YearsSinceLastPromotion"], errors="coerce")
    yir = pd.to_numeric(df["YearsInCurrentRole"], errors="coerce")
    eligible = yac >= C.CSI_MIN_TENURE_YEARS
    denom = yac.where(eligible)
    return pd.DataFrame({
        "csi_unpromoted_share": (yslp / denom).clip(0, 1),
        "csi_role_share": (yir / denom).clip(0, 1),
    }, index=df.index)


def compute_csi(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """CSI = 100 * (0.5 * YearsSinceLastPromotion/YearsAtCompany + 0.5 * YearsInCurrentRole/YearsAtCompany).

    Each ratio is clipped to [0,1]. Defined only when YearsAtCompany >= 2 (otherwise
    NaN, "not assessed") because ratios for brand-new hires are mechanically extreme.
    """
    w = weights or C.CSI_WEIGHTS
    comp = csi_components(df)
    total = comp["csi_unpromoted_share"] * w["unpromoted_share"] + comp["csi_role_share"] * w["role_share"]
    return (total / sum(w.values()) * 100.0).where(comp.notna().all(axis=1))


# --------------------------------------------------------------------------- #
# Banding helpers
# --------------------------------------------------------------------------- #
def band(series: pd.Series, bands: list[tuple[str, float, float]]) -> pd.Series:
    """Map numeric values to ordered band labels; out-of-band/NaN -> NaN."""
    out = pd.Series(np.nan, index=series.index, dtype=object)
    v = pd.to_numeric(series, errors="coerce")
    for label, lo, hi in bands:
        out[(v >= lo) & (v <= hi)] = label
    return out


def band_order(bands) -> list[str]:
    return [b[0] for b in bands]


def csi_band(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=object)
    for label, lo, hi in C.CSI_BANDS:
        out[(series >= lo) & (series < hi)] = label
    return out


# --------------------------------------------------------------------------- #
# Assemble analysis frame
# --------------------------------------------------------------------------- #
def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with all derived analysis columns appended (source columns untouched)."""
    out = df.copy()
    out["CEI"] = compute_cei(out)
    out["BurnoutRisk"] = compute_burnout_risk(out)
    out["SatisfactionStability"] = compute_satisfaction_stability(out)
    out = out.join(wsi_components(out)).join(csi_components(out))
    out["WSI"] = compute_wsi(out)
    out["CSI"] = compute_csi(out)
    out["CommuteBand"] = band(out["DistanceFromHome"], C.COMMUTE_BANDS)
    out["TenureBand"] = band(out["YearsAtCompany"], C.TENURE_BANDS)
    out["RoleTenureBand"] = band(out["YearsInCurrentRole"], C.ROLE_TENURE_BANDS)
    out["PromotionGapBand"] = band(out["YearsSinceLastPromotion"], C.PROMOTION_GAP_BANDS)
    out["CSIBand"] = csi_band(out["CSI"])
    out["OvertimeFlag"] = out["OverTime"].eq("Yes")
    out["LowWLBFlag"] = pd.to_numeric(out["WorkLifeBalance"], errors="coerce").le(C.LOW_WLB_CUTOFF)
    out["JobLevelLabel"] = "Level " + out["JobLevel"].astype("Int64").astype(str)
    out["Area"] = out["Department"].astype(str) + " \u00b7 " + out["JobRole"].astype(str)
    return out
