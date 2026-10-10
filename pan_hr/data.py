"""Data loading. Source file is read as-is; analysis columns are added to a copy."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import metrics as M
from . import validation as V

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "palo_alto_networks.csv"


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the CSV without altering values (no imputation, no dropping)."""
    return pd.read_csv(path)


def load_and_prepare(path: str | Path = DEFAULT_DATA_PATH):
    """Return (raw, analysis_frame, validation_report, nulled_scale_counts)."""
    raw = load_raw(path)
    report = V.validate_dataframe(raw)
    clean, nulled = V.coerce_invalid_scales(raw)
    return raw, M.add_metrics(clean), report, nulled
