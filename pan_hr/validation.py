"""Dataset validation. Never mutates the source frame; reports what it found."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str
    count: int = 0


@dataclass
class ValidationReport:
    n_rows: int
    n_cols: int
    checks: list[Check] = field(default_factory=list)
    column_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    open_categories: dict[str, list[str]] = field(default_factory=dict)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def passed(self) -> bool:
        return not self.failed

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"Check": c.name, "Status": c.status.upper(), "Affected": c.count, "Detail": c.detail}
             for c in self.checks]
        )


def _as_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or s.dtype == object


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Run schema, type, category, scale, duplicate and missing-value checks."""
    rep = ValidationReport(n_rows=len(df), n_cols=df.shape[1])
    add = rep.checks.append

    # 1. Columns
    missing_cols = sorted(set(C.EXPECTED_COLUMNS) - set(df.columns))
    extra_cols = sorted(set(df.columns) - set(C.EXPECTED_COLUMNS))
    if df.shape[1] == C.EXPECTED_N_COLUMNS and not missing_cols and not extra_cols:
        add(Check("Expected 31 fields present", "pass", "All 31 documented fields found; no extras."))
    else:
        add(Check("Expected 31 fields present", "fail",
                  f"Found {df.shape[1]} columns. Missing: {missing_cols or 'none'}; unexpected: {extra_cols or 'none'}.",
                  len(missing_cols) + len(extra_cols)))
    present = set(df.columns)

    # 2. Dtypes
    bad_num = [c for c in C.NUMERIC_COLUMNS if c in present and not pd.api.types.is_numeric_dtype(df[c])]
    bad_str = [c for c in C.STRING_COLUMNS if c in present and not _is_text(df[c])]
    if bad_num or bad_str:
        add(Check("Data types", "fail", f"Non-numeric: {bad_num or 'none'}; non-text: {bad_str or 'none'}.", len(bad_num) + len(bad_str)))
    else:
        add(Check("Data types", "pass", f"{len(C.NUMERIC_COLUMNS)} numeric and {len(C.STRING_COLUMNS)} text fields have the expected types."))

    # 3. Missing values
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        add(Check("Missing values", "pass", "No missing values in any field."))
    else:
        add(Check("Missing values", "warn", "Missing values by field: " + ", ".join(f"{k}={v}" for k, v in miss.items()), int(miss.sum())))

    # 4. Duplicates
    n_dup = int(df.duplicated().sum())
    add(Check("Duplicate rows", "pass" if n_dup == 0 else "warn",
              "No fully duplicated rows." if n_dup == 0 else f"{n_dup} fully duplicated rows (kept, not removed).", n_dup))

    # 5. 1-4 scales
    for col in C.SCALE_COLUMNS:
        if col not in present:
            continue
        v = _as_numeric(df[col])
        invalid = v.notna() & (~v.between(C.SCALE_MIN, C.SCALE_MAX) | (v % 1 != 0))
        n_bad = int(invalid.sum())
        add(Check(f"{col} within 1-4", "pass" if n_bad == 0 else "fail",
                  "All non-missing values are integers 1-4." if n_bad == 0 else f"{n_bad} values outside 1-4 or non-integer.", n_bad))

    # 6. Allowed categories
    for col, allowed in C.ALLOWED_VALUES.items():
        if col not in present:
            continue
        observed = set(df[col].dropna().unique())
        unexpected = observed - allowed
        n_bad = int(df[col].isin(list(unexpected)).sum()) if unexpected else 0
        add(Check(f"{col} allowed values", "pass" if not unexpected else "fail",
                  f"Observed {sorted(map(str, observed))}." if not unexpected else f"Unexpected values: {sorted(map(str, unexpected))}.", n_bad))

    # 7. Documented ranges and non-negativity
    for col, (lo, hi) in C.INTEGER_RANGES.items():
        if col in present:
            v = _as_numeric(df[col]).dropna()
            n_bad = int((~v.between(lo, hi)).sum())
            add(Check(f"{col} within {lo}-{hi}", "pass" if n_bad == 0 else "fail",
                      "Within documented range." if n_bad == 0 else f"{n_bad} values outside range.", n_bad))
    neg = {c: int((_as_numeric(df[c]) < 0).sum()) for c in C.NON_NEGATIVE_COLUMNS if c in present}
    neg = {k: v for k, v in neg.items() if v}
    add(Check("Non-negative numeric fields", "pass" if not neg else "fail",
              "No negative values." if not neg else f"Negative values: {neg}", sum(neg.values())))
    if "JobLevel" in present:
        v = _as_numeric(df["JobLevel"]).dropna()
        n_bad = int(((v < 1) | (v % 1 != 0)).sum())
        add(Check("JobLevel is a positive integer", "pass" if n_bad == 0 else "fail",
                  f"Observed levels {sorted(int(x) for x in v.unique())}." if n_bad == 0 else f"{n_bad} invalid values.", n_bad))

    # 8. Logical consistency (warnings only)
    pairs = [
        ("YearsInCurrentRole", "YearsAtCompany"),
        ("YearsSinceLastPromotion", "YearsAtCompany"),
        ("YearsWithCurrManager", "YearsAtCompany"),
        ("YearsAtCompany", "TotalWorkingYears"),
    ]
    for a, b in pairs:
        if a in present and b in present:
            n_bad = int((_as_numeric(df[a]) > _as_numeric(df[b])).sum())
            add(Check(f"{a} <= {b}", "pass" if n_bad == 0 else "warn",
                      "Consistent." if n_bad == 0 else f"{n_bad} records inconsistent (kept).", n_bad))

    # Open categories & column summary
    for col in C.OPEN_CATEGORY_COLUMNS:
        if col in present:
            rep.open_categories[col] = sorted(df[col].dropna().astype(str).unique())
    rows = []
    for col in df.columns:
        s = df[col]
        num = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "Field": col, "Type": "numeric" if num else "text",
            "Missing": int(s.isna().sum()), "Distinct": int(s.nunique(dropna=True)),
            "Min": f"{s.min():g}" if num else "-", "Max": f"{s.max():g}" if num else "-",
        })
    rep.column_summary = pd.DataFrame(rows)
    return rep


def coerce_invalid_scales(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a COPY where out-of-range/non-integer scale values become NaN.

    The source frame is untouched. The count of values nulled per field is returned
    so the UI and paper can disclose exactly what happened.
    """
    out = df.copy()
    changed: dict[str, int] = {}
    for col in C.SCALE_COLUMNS:
        if col not in out.columns:
            continue
        v = _as_numeric(out[col])
        bad = v.notna() & (~v.between(C.SCALE_MIN, C.SCALE_MAX) | (v % 1 != 0))
        changed[col] = int(bad.sum())
        out[col] = v.mask(bad, np.nan)
    return out, changed
