import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pan_hr import metrics as M  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return pd.read_csv(ROOT / "data" / "palo_alto_networks.csv")


@pytest.fixture(scope="session")
def df(raw):
    return M.add_metrics(raw)


def make_row(**kw):
    base = dict(JobInvolvement=3, JobSatisfaction=3, EnvironmentSatisfaction=3, RelationshipSatisfaction=3,
                WorkLifeBalance=3, OverTime="No", BusinessTravel="Non-Travel", DistanceFromHome=1,
                YearsAtCompany=10, YearsInCurrentRole=5, YearsSinceLastPromotion=2)
    base.update(kw)
    return base
