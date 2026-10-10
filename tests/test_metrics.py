import numpy as np
import pandas as pd
import pytest

from pan_hr import metrics as M
from conftest import make_row


def frame(*rows):
    return pd.DataFrame(list(rows))


def test_cei_bounds_and_formula():
    d = frame(make_row(JobInvolvement=4, JobSatisfaction=4, EnvironmentSatisfaction=4, RelationshipSatisfaction=4),
              make_row(JobInvolvement=1, JobSatisfaction=1, EnvironmentSatisfaction=1, RelationshipSatisfaction=1),
              make_row(JobInvolvement=2, JobSatisfaction=3, EnvironmentSatisfaction=4, RelationshipSatisfaction=3))
    assert M.compute_cei(d).tolist() == pytest.approx([100.0, 25.0, 75.0])


def test_cei_missing_input_is_nan_not_imputed():
    d = frame(make_row(JobSatisfaction=np.nan))
    assert M.compute_cei(d).isna().all()


def test_cei_on_real_data_range(df):
    assert df["CEI"].between(25, 100).all()
    assert df["CEI"].mean() == pytest.approx(68.0782, abs=1e-3)


@pytest.mark.parametrize("ot,wlb,expected", [
    ("Yes", 1, "High"), ("Yes", 2, "High"), ("Yes", 3, "Medium"), ("Yes", 4, "Medium"),
    ("No", 1, "Medium"), ("No", 2, "Medium"), ("No", 3, "Low"), ("No", 4, "Low"),
])
def test_burnout_rules(ot, wlb, expected):
    assert M.compute_burnout_risk(frame(make_row(OverTime=ot, WorkLifeBalance=wlb))).iloc[0] == expected


def test_burnout_unclassified_when_input_missing():
    d = frame(make_row(WorkLifeBalance=np.nan), make_row(OverTime=None))
    assert (M.compute_burnout_risk(d) == "Unclassified").all()


def test_burnout_counts_real_data(df):
    assert df["BurnoutRisk"].value_counts().to_dict() == {"Low": 756, "Medium": 588, "High": 126}


def test_wsi_components_and_bounds():
    hi = make_row(BusinessTravel="Travel_Frequently", DistanceFromHome=29, OverTime="Yes")
    lo = make_row(BusinessTravel="Non-Travel", DistanceFromHome=1, OverTime="No")
    mid = make_row(BusinessTravel="Travel_Rarely", DistanceFromHome=15, OverTime="No")
    w = M.compute_wsi(frame(hi, lo, mid))
    assert w.iloc[0] == pytest.approx(100) and w.iloc[1] == pytest.approx(0)
    assert w.iloc[2] == pytest.approx(100 * (0.5 + 0.5 + 0) / 3)


def test_wsi_custom_weights():
    d = frame(make_row(BusinessTravel="Travel_Frequently", DistanceFromHome=1, OverTime="No"))
    assert M.compute_wsi(d, {"travel": 1, "commute": 0, "overtime": 0}).iloc[0] == pytest.approx(100)


def test_csi_formula_and_min_tenure():
    d = frame(make_row(YearsAtCompany=10, YearsInCurrentRole=10, YearsSinceLastPromotion=10),
              make_row(YearsAtCompany=10, YearsInCurrentRole=5, YearsSinceLastPromotion=0),
              make_row(YearsAtCompany=1, YearsInCurrentRole=1, YearsSinceLastPromotion=1))
    c = M.compute_csi(d)
    assert c.iloc[0] == pytest.approx(100)
    assert c.iloc[1] == pytest.approx(25)
    assert np.isnan(c.iloc[2])  # tenure < 2 -> not assessed


def test_csi_clipped():
    d = frame(make_row(YearsAtCompany=5, YearsInCurrentRole=9, YearsSinceLastPromotion=9))
    assert M.compute_csi(d).iloc[0] == pytest.approx(100)


def test_stability_score():
    same = make_row()
    spread = make_row(JobInvolvement=1, JobSatisfaction=1, EnvironmentSatisfaction=4, RelationshipSatisfaction=4)
    s = M.compute_satisfaction_stability(frame(same, spread))
    assert s.iloc[0] == pytest.approx(100) and s.iloc[1] == pytest.approx(0)


def test_add_metrics_does_not_alter_source(raw):
    before = raw.copy()
    M.add_metrics(raw)
    pd.testing.assert_frame_equal(raw, before)
