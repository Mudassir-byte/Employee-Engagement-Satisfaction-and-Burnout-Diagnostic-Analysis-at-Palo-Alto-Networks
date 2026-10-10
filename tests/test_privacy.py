import numpy as np
import pandas as pd

from pan_hr import analysis as A
from pan_hr import config as C
from pan_hr import privacy as P


def labels(counts: dict):
    return pd.Series([k for k, v in counts.items() for _ in range(v)])


def test_safe_share_rules():
    assert P.safe_share(3, 50) is None          # small numerator
    assert P.safe_share(47, 50) is None         # small complement
    assert P.safe_share(0, 50) == 0.0
    assert P.safe_share(10, 4) is None          # tiny cohort
    assert P.safe_share(10, 50) == 0.2


def test_fmt_n():
    assert P.fmt_n(3) == "<5" and P.fmt_n(5) == "5" and P.fmt_n(1234) == "1,234"


def test_protect_nominal_pools_small_and_complements():
    out = P.protect_nominal(labels({"A": 40, "B": 3, "C": 20, "D": 8}))
    counts = out.value_counts()
    assert (counts >= 5).all()
    assert "B" not in counts.index
    assert counts[C.OTHER_LABEL] >= 5  # lone small group is padded so it cannot be recovered by subtraction


def test_protect_nominal_all_small():
    assert P.protect_nominal(labels({"A": 2, "B": 1})).isna().all()


def test_protect_ordinal_merges_neighbours():
    s = labels({"a": 3, "b": 20, "c": 2, "d": 30})
    out = P.protect_ordinal(s, ["a", "b", "c", "d"])
    assert (out.value_counts() >= 5).all()
    assert out[s == "a"].iloc[0] == "a + b + c"


def test_withhold_cells_complementary():
    shown, pooled = P.withhold_cells(pd.Series([2, 40, 10, 8]))
    assert shown.isna().sum() == 2 and pooled == 10  # smallest other cell withheld too, pooled >= 5
    shown, pooled = P.withhold_cells(pd.Series([30, 40, 10, 8]))
    assert pooled == 0 and shown.notna().all()
    shown, _ = P.withhold_cells(pd.Series([1, 1, 1, 0]))
    assert shown.isna().all()


def test_mask_matrix():
    m = P.mask_matrix(pd.DataFrame({"x": [0, 3, 5]}))
    assert m["x"].tolist() == [False, True, False]


def test_no_group_below_k_in_any_summary(df):
    tiny = df[df["Department"].eq("Human Resources") & df["OverTime"].eq("Yes")]
    for col, nominal in (("JobRole", True), ("Area", True), ("Gender", True)):
        out = A.grouped_cei(tiny, col, 50, nominal=nominal)
        assert out.empty or (out["n"] >= C.K_MIN).all()
    mix = A.risk_mix_by(tiny, "JobRole")
    assert mix.empty or (mix.groupby("group")["n"].first() >= C.K_MIN).all()


def test_small_filter_view_is_suppressed(df):
    hr_mgr_ot = df[(df["JobRole"] == "Manager") & (df["Department"] == "Human Resources") & (df["OverTime"] == "Yes")]
    assert len(hr_mgr_ot) < C.K_MIN
    assert A.grouped_cei(hr_mgr_ot, "JobRole", 50, nominal=True).empty
    assert A.priority_areas(hr_mgr_ot, 50).empty
    assert not A.compare_attrition(hr_mgr_ot).reportable


def test_distribution_cells_withheld(df):
    sub = df.head(6)
    dist = A.score_distribution(sub, "JobSatisfaction")
    counts = dist["count"].dropna()
    assert ((counts == 0) | (counts >= C.K_MIN)).all()


def test_priority_output_has_no_individual_columns(df):
    pr = A.priority_areas(df, 50)
    assert not {"Age", "MonthlyIncome", "Gender", "EmployeeNumber"} & set(pr.columns)
    assert (pr["n"] >= C.K_MIN).all()


def test_heatmap_cells_blanked(df):
    n, mean, mask = A.crosstab_cells(df[df["JobRole"] == "Manager"], "JobLevel", "OverTime")
    assert (n.stack().dropna() >= C.K_MIN).all()
    assert mean.where(mask).stack().dropna().empty
