"""Filter state and application logic (UI-independent so it can be unit tested)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config as C

TENURE_BASES = {"Years at company": "YearsAtCompany", "Years in current role": "YearsInCurrentRole"}


@dataclass
class FilterState:
    departments: list[str] = field(default_factory=list)   # empty = all
    roles: list[str] = field(default_factory=list)         # empty = all
    overtime: str = "All"                                  # All | Yes | No
    threshold: float = C.DEFAULT_LOW_ENGAGEMENT_THRESHOLD
    below_only: bool = False
    tenure_basis: str = "Years at company"
    tenure_range: tuple[int, int] = (0, 40)
    risk_levels: list[str] = field(default_factory=list)   # cross-filter selection
    cross_roles: list[str] = field(default_factory=list)   # cross-filter selection

    def describe(self) -> str:
        parts = []
        parts.append("departments: " + (", ".join(self.departments) if self.departments else "all"))
        parts.append("roles: " + (", ".join(self.roles) if self.roles else "all"))
        parts.append({"All": "overtime: any", "Yes": "overtime: yes only", "No": "overtime: no only"}[self.overtime])
        parts.append(f"{self.tenure_basis.lower()}: {self.tenure_range[0]}-{self.tenure_range[1]}")
        if self.below_only:
            parts.append(f"CEI below {self.threshold:g} only")
        if self.risk_levels:
            parts.append("screening level selected: " + ", ".join(self.risk_levels))
        if self.cross_roles:
            parts.append("roles selected on chart: " + ", ".join(self.cross_roles))
        return "; ".join(parts)


def apply_filters(df: pd.DataFrame, f: FilterState, *, use_risk: bool = True, use_roles: bool = True) -> pd.DataFrame:
    """Apply sidebar filters (+ optional chart cross-filters). Returns a new frame."""
    m = pd.Series(True, index=df.index)
    if f.departments:
        m &= df["Department"].isin(f.departments)
    if f.roles:
        m &= df["JobRole"].isin(f.roles)
    if f.overtime in ("Yes", "No"):
        m &= df["OverTime"].eq(f.overtime)
    col = TENURE_BASES.get(f.tenure_basis, "YearsAtCompany")
    lo, hi = f.tenure_range
    m &= df[col].between(lo, hi)
    if f.below_only:
        m &= df["CEI"] < f.threshold
    if use_risk and f.risk_levels:
        m &= df["BurnoutRisk"].isin(f.risk_levels)
    if use_roles and f.cross_roles:
        m &= df["JobRole"].isin(f.cross_roles)
    return df[m]
