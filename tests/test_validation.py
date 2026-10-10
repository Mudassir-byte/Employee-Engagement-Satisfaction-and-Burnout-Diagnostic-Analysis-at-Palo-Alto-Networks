import numpy as np
import pandas as pd

from pan_hr import validation as V


def test_real_dataset_passes(raw):
    rep = V.validate_dataframe(raw)
    assert rep.passed and rep.n_rows == 1470 and rep.n_cols == 31
    assert not rep.warnings


def test_missing_column_fails(raw):
    rep = V.validate_dataframe(raw.drop(columns=["Age"]))
    assert not rep.passed
    assert any("31 fields" in c.name for c in rep.failed)


def test_out_of_range_scale_detected_and_not_mutated(raw):
    bad = raw.copy()
    bad.loc[0, "JobSatisfaction"] = 7
    bad.loc[1, "WorkLifeBalance"] = 0
    snapshot = bad.copy()
    rep = V.validate_dataframe(bad)
    assert {c.name for c in rep.failed} >= {"JobSatisfaction within 1-4", "WorkLifeBalance within 1-4"}
    pd.testing.assert_frame_equal(bad, snapshot)  # validation never mutates
    clean, nulled = V.coerce_invalid_scales(bad)
    assert nulled["JobSatisfaction"] == 1 and nulled["WorkLifeBalance"] == 1
    assert np.isnan(clean.loc[0, "JobSatisfaction"])
    assert bad.loc[0, "JobSatisfaction"] == 7  # source copy untouched


def test_missing_and_duplicates_reported(raw):
    bad = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    bad.loc[3, "JobInvolvement"] = np.nan
    rep = V.validate_dataframe(bad)
    names = {c.name: c for c in rep.warnings}
    assert names["Missing values"].count == 1
    assert names["Duplicate rows"].count == 1


def test_bad_category_fails(raw):
    bad = raw.copy()
    bad.loc[0, "OverTime"] = "Maybe"
    assert any(c.name == "OverTime allowed values" for c in V.validate_dataframe(bad).failed)
