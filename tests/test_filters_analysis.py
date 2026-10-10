import pandas as pd
import pytest

from pan_hr import analysis as A
from pan_hr import config as C
from pan_hr.filters import FilterState, apply_filters


def test_filters_combine(df):
    f = FilterState(departments=["Sales"], overtime="Yes", tenure_range=(2, 10))
    out = apply_filters(df, f)
    assert (out["Department"] == "Sales").all() and (out["OverTime"] == "Yes").all()
    assert out["YearsAtCompany"].between(2, 10).all()
    assert len(out) == len(df[(df.Department == "Sales") & (df.OverTime == "Yes") & df.YearsAtCompany.between(2, 10)])


def test_default_filter_is_identity(df):
    assert len(apply_filters(df, FilterState())) == len(df)


def test_threshold_restriction_and_role_basis(df):
    f = FilterState(below_only=True, threshold=50, tenure_basis="Years in current role", tenure_range=(0, 3))
    out = apply_filters(df, f)
    assert (out["CEI"] < 50).all() and out["YearsInCurrentRole"].between(0, 3).all()


def test_cross_filters_optional(df):
    f = FilterState(risk_levels=["High"], cross_roles=["Manager"])
    assert len(apply_filters(df, f)) < len(apply_filters(df, f, use_risk=False)) <= len(apply_filters(df, f, use_risk=False, use_roles=False))


def test_filters_never_stale(df):
    a = len(apply_filters(df, FilterState(overtime="Yes")))
    b = len(apply_filters(df, FilterState(overtime="No")))
    assert a + b == len(df) and a != b


def test_attrition_comparison_descriptive(df):
    c = A.compare_attrition(df)
    assert c.reportable and c.n_left == 237 and c.diff > 0 and c.ci_lo < c.diff < c.ci_hi


def test_mean_ci_handles_small_samples():
    assert A.mean_ci(pd.Series([], dtype=float))[3] == 0
    m, lo, hi, n = A.mean_ci(pd.Series([5.0]))
    assert m == 5.0 and pd.isna(lo) and n == 1


def test_priority_flags_follow_rules(df):
    pr = A.priority_areas(df, 50)
    for _, r in pr.iterrows():
        assert ("engagement" in r["flags"]) == (r["cei_gap"] <= -C.ENGAGEMENT_GAP_POINTS)
        assert r["n_flags"] == len(r["flags"])
    assert pr["n_flags"].is_monotonic_decreasing


def test_hist_bins_cover_everyone(df):
    assert A.cei_histogram(df)["count"].sum() == df["CEI"].notna().sum()
