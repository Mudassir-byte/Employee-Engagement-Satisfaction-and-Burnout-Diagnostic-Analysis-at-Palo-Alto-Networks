"""Small-cohort protection (minimum cohort size k = 5, from the strategic plan).

Policy implemented here
  * Nominal groups (role, department, area): groups with n < k are pooled into an
    "Other" group; if that pool is still < k, the smallest remaining groups are added
    until it reaches k (complementary pooling) so a lone suppressed group cannot be
    recovered by subtraction.
  * Ordinal groups (bands, levels): a group with n < k is merged into its smaller
    neighbour until every group has n >= k.
  * Distribution cells (e.g. rating 1-4, risk levels): cells with 0 < n < k are
    withheld; if exactly one is withheld, the smallest other cell is withheld too.
  * Shares of a binary indicator: withheld when 0 < numerator < k or 0 < (n - numerator) < k.
  * Matrix cells (heatmaps): cells with 0 < n < k are blanked.
Residual risk (differencing across successive filter settings) is documented in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import K_MIN, OTHER_LABEL


def is_reportable(n: int | float, k: int = K_MIN) -> bool:
    return bool(n is not None and not pd.isna(n) and n >= k)


def fmt_n(n: int | float, k: int = K_MIN) -> str:
    """Display a count, hiding small ones."""
    if n is None or pd.isna(n):
        return "n/a"
    return f"{int(n):,}" if n >= k else f"<{k}"


def safe_share(num: int | float, den: int | float, k: int = K_MIN) -> float | None:
    """Share num/den, or None when the numerator or its complement is a small non-zero count."""
    if den is None or pd.isna(den) or den < k:
        return None
    num = float(num)
    if 0 < num < k or 0 < (den - num) < k:
        return None
    return num / den


def protect_nominal(labels: pd.Series, k: int = K_MIN, other: str = OTHER_LABEL) -> pd.Series:
    """Relabel records so every displayed nominal group has >= k records.

    Records whose labels are NaN stay NaN. If the whole population is < k every
    label becomes NaN (nothing reportable).
    """
    labels = labels.copy()
    counts = labels.value_counts(dropna=True)
    if counts.empty:
        return labels
    if counts.sum() < k:
        return pd.Series(np.nan, index=labels.index, dtype=object)
    small = list(counts[counts < k].index)
    if not small:
        return labels
    pool = int(counts[small].sum())
    members = set(small)
    if pool < k:
        for lab in counts[counts >= k].sort_values().index:  # smallest first
            members.add(lab)
            pool += int(counts[lab])
            if pool >= k:
                break
    return labels.where(~labels.isin(members), other)


def protect_ordinal(labels: pd.Series, order: list[str], k: int = K_MIN) -> pd.Series:
    """Merge adjacent ordered categories until each has >= k records."""
    counts = labels.value_counts()
    groups: list[list[str]] = [[c] for c in order if counts.get(c, 0) > 0]
    sizes = [int(counts[g[0]]) for g in groups]
    if sum(sizes) < k:
        return pd.Series(np.nan, index=labels.index, dtype=object)
    while len(groups) > 1 and min(sizes) < k:
        i = int(np.argmin(sizes))
        left = sizes[i - 1] if i > 0 else np.inf
        right = sizes[i + 1] if i < len(groups) - 1 else np.inf
        j = i - 1 if left <= right else i + 1
        a, b = sorted((i, j))
        groups[a] = groups[a] + groups[b]
        sizes[a] = sizes[a] + sizes[b]
        del groups[b], sizes[b]
    mapping = {c: (" + ".join(g) if len(g) > 1 else g[0]) for g in groups for c in g}
    return labels.map(mapping)


def ordered_display_labels(labels: pd.Series, order: list[str]) -> list[str]:
    """Display order for (possibly merged) ordinal labels."""
    present = list(pd.unique(labels.dropna()))
    pos = {c: i for i, c in enumerate(order)}
    return sorted(present, key=lambda lab: min(pos.get(p, 99) for p in str(lab).split(" + ")))


def withhold_cells(counts: pd.Series, k: int = K_MIN) -> tuple[pd.Series, int]:
    """Mask distribution cells with 0 < n < k (plus complementary cell). Returns (shown, pooled_n).

    Masked cells are NaN in `shown`; `pooled_n` is the combined count of all masked cells
    (always 0 or >= k unless the whole distribution is < k, in which case everything is masked).
    """
    c = counts.astype(float).copy()
    total = c.sum()
    if total < k:
        return pd.Series(np.nan, index=c.index, dtype=float), int(total)
    small = c[(c > 0) & (c < k)]
    if small.empty:
        return c, 0
    masked = set(small.index)
    if len(masked) == 1:
        others = c.drop(index=list(masked))
        others = others[others > 0]
        if not others.empty:
            masked.add(others.idxmin())
    pooled = int(c[list(masked)].sum())
    shown = c.copy()
    shown[list(masked)] = np.nan
    return shown, pooled


def mask_matrix(counts: pd.DataFrame, k: int = K_MIN) -> pd.DataFrame:
    """Boolean mask: True where a matrix cell must be blanked (0 < n < k)."""
    return (counts > 0) & (counts < k)
