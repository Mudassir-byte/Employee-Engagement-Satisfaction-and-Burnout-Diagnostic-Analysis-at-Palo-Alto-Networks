"""Reproduce every number and figure in the research paper from the supplied CSV.

Usage:  python scripts/reproduce_analysis.py [--out ../Research_Paper]
Requires: pandas, numpy, scipy, matplotlib (see requirements-dev.txt). Seeded; deterministic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy import stats  # noqa: E402

from pan_hr import analysis as A  # noqa: E402
from pan_hr import config as C  # noqa: E402
from pan_hr import data as D  # noqa: E402

SEED = 20261009
THR = C.DEFAULT_LOW_ENGAGEMENT_THRESHOLD


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def boot_diff(a, b, rng, n=5000):
    a, b = a.to_numpy(), b.to_numpy()
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def group_table(df, col, order=None, nominal=False):
    t = A.grouped_cei(df, col, THR, order=order, nominal=nominal)
    return t[["group", "n", "cei_mean", "cei_lo", "cei_hi", "pct_below", "pct_high_burnout", "wsi_mean", "csi_mean"]]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    raw, df, rep, nulled = D.load_and_prepare()
    R: dict = {}

    # --- validation & coverage
    R["validation"] = {"rows": rep.n_rows, "columns": rep.n_cols, "passed": rep.passed,
                       "failed_checks": len(rep.failed), "warnings": len(rep.warnings), "checks": len(rep.checks),
                       "missing_total": int(raw.isna().sum().sum()), "duplicates": int(raw.duplicated().sum()),
                       "scale_values_nulled": nulled, "open_categories": rep.open_categories}
    rep.to_frame().to_csv(out / "tables" / "validation_checks.csv", index=False)
    R["coverage"] = {"cei_computable": int(df.CEI.notna().sum()), "csi_assessed": int(df.CSI.notna().sum()),
                     "csi_not_assessed_tenure_lt2": int(df.CSI.isna().sum()),
                     "attrition_left": int(df.Attrition.sum()), "attrition_rate": float(df.Attrition.mean())}

    # --- CEI
    cei = df.CEI
    R["cei"] = {"mean": cei.mean(), "sd": cei.std(), "median": cei.median(), "min": cei.min(), "max": cei.max(),
                "p25": cei.quantile(.25), "p75": cei.quantile(.75),
                "ci95": [A.mean_ci(cei)[1], A.mean_ci(cei)[2]],
                "below_50": int((cei < 50).sum()), "below_50_share": float((cei < 50).mean()),
                "item_means": {c: float(df[c].mean()) for c in C.CEI_COLUMNS + ["WorkLifeBalance"]},
                "item_low_share_1_2": {c: float((df[c] <= 2).mean()) for c in C.CEI_COLUMNS + ["WorkLifeBalance"]},
                "cronbach_alpha": float(_alpha(df[C.CEI_COLUMNS]))}
    R["cei"]["inter_item_corr_mean"] = float(df[C.CEI_COLUMNS].corr().where(~np.eye(4, dtype=bool)).stack().mean())

    # --- burnout
    mix = df.BurnoutRisk.value_counts()
    R["burnout"] = {"counts": mix.to_dict(), "shares": (mix / len(df)).to_dict(),
                    "overtime_share": float(df.OvertimeFlag.mean()), "low_wlb_share": float(df.LowWLBFlag.mean()),
                    "wlb_mean": float(df.WorkLifeBalance.mean())}
    R["cei_by_burnout"] = {k: dict(zip(["mean", "lo", "hi", "n"], A.mean_ci(g.CEI))) for k, g in df.groupby("BurnoutRisk")}
    hi_lo = df[df.BurnoutRisk == "High"].CEI, df[df.BurnoutRisk == "Low"].CEI
    R["burnout_high_vs_low_cei"] = {"diff": float(hi_lo[0].mean() - hi_lo[1].mean()), "d": cohens_d(*hi_lo),
                                    "boot_ci": boot_diff(*hi_lo, rng)}
    for col in ["Department", "JobLevelLabel"]:
        ct = pd.crosstab(df[col], df.BurnoutRisk)
        R[f"burnout_by_{col}"] = {k: {lv: int(v) for lv, v in row.items()} for k, row in ct.iterrows()}
    R["high_share_by_department"] = {k: float(v) for k, v in df.groupby("Department").apply(lambda g: (g.BurnoutRisk == "High").mean(), include_groups=False).items()}

    # --- workload comparisons
    def two_group(col, a, b):
        x, y = df[df[col] == a].CEI, df[df[col] == b].CEI
        t = stats.ttest_ind(x, y, equal_var=False)
        u = stats.mannwhitneyu(x, y)
        return {"a": a, "b": b, "n_a": len(x), "n_b": len(y), "mean_a": x.mean(), "mean_b": y.mean(), "diff_a_minus_b": x.mean() - y.mean(),
                "d": cohens_d(x, y), "boot_ci": boot_diff(x, y, rng), "welch_p": float(t.pvalue), "mannwhitney_p": float(u.pvalue)}
    R["overtime_vs_not"] = two_group("OverTime", "Yes", "No")
    R["travel_anova"] = {}
    groups = [g.CEI.values for _, g in df.groupby("BusinessTravel")]
    f = stats.f_oneway(*groups); kw = stats.kruskal(*groups)
    R["travel_anova"] = {"F": float(f.statistic), "p": float(f.pvalue), "kruskal_p": float(kw.pvalue)}
    groups = [g.CEI.values for _, g in df.groupby("CommuteBand")]
    f = stats.f_oneway(*groups)
    R["commute_anova"] = {"F": float(f.statistic), "p": float(f.pvalue), "kruskal_p": float(stats.kruskal(*groups).pvalue)}
    R["cei_by_overtime"] = group_table(df, "OverTime", nominal=True).to_dict("records")
    R["cei_by_travel"] = group_table(df.assign(T=df.BusinessTravel.map(C.TRAVEL_LABELS)), "T", order=[C.TRAVEL_LABELS[k] for k in C.TRAVEL_ORDER]).to_dict("records")
    R["cei_by_commute"] = group_table(df, "CommuteBand", order=[b[0] for b in C.COMMUTE_BANDS]).to_dict("records")
    R["cei_by_wlb"] = {int(k): dict(zip(["mean", "lo", "hi", "n"], A.mean_ci(g.CEI))) for k, g in df.groupby("WorkLifeBalance")}
    R["spearman_cei_wlb"] = float(stats.spearmanr(df.CEI, df.WorkLifeBalance)[0])
    R["spearman_cei_wsi"] = float(stats.spearmanr(df.CEI, df.WSI)[0])

    # --- departments / roles / levels
    R["cei_by_department"] = group_table(df, "Department", nominal=True).to_dict("records")
    R["cei_by_role"] = group_table(df, "JobRole", nominal=True).to_dict("records")
    R["cei_by_level"] = group_table(df, "JobLevelLabel", order=[f"Level {i}" for i in range(1, 6)]).to_dict("records")
    groups = [g.CEI.values for _, g in df.groupby("JobRole")]
    R["role_anova"] = {"F": float(stats.f_oneway(*groups).statistic), "p": float(stats.f_oneway(*groups).pvalue), "kruskal_p": float(stats.kruskal(*groups).pvalue)}
    groups = [g.CEI.values for _, g in df.groupby("Department")]
    R["dept_anova"] = {"p": float(stats.f_oneway(*groups).pvalue)}
    groups = [g.CEI.values for _, g in df.groupby("JobLevel")]
    R["level_anova"] = {"p": float(stats.f_oneway(*groups).pvalue)}
    R["role_range_of_means"] = [float(min(r["cei_mean"] for r in R["cei_by_role"])), float(max(r["cei_mean"] for r in R["cei_by_role"]))]

    # --- tenure / career
    R["cei_by_tenure"] = group_table(df, "TenureBand", order=[b[0] for b in C.TENURE_BANDS]).to_dict("records")
    R["cei_by_role_tenure"] = group_table(df, "RoleTenureBand", order=[b[0] for b in C.ROLE_TENURE_BANDS]).to_dict("records")
    R["cei_by_promo_gap"] = group_table(df, "PromotionGapBand", order=[b[0] for b in C.PROMOTION_GAP_BANDS]).to_dict("records")
    R["cei_by_csi_band"] = group_table(df.dropna(subset=["CSIBand"]), "CSIBand", order=[b[0] for b in C.CSI_BANDS]).to_dict("records")
    for name, col in [("years_at_company", "YearsAtCompany"), ("years_in_role", "YearsInCurrentRole"),
                      ("years_since_promo", "YearsSinceLastPromotion"), ("csi", "CSI"), ("job_level", "JobLevel")]:
        sub = df.dropna(subset=[col, "CEI"])
        r = stats.spearmanr(sub[col], sub.CEI)
        R.setdefault("spearman_cei_vs", {})[name] = {"rho": float(r[0]), "p": float(r[1]), "n": len(sub)}

    # --- attrition (contextual)
    s, l = df[df.Attrition == 0].CEI, df[df.Attrition == 1].CEI
    t = stats.ttest_ind(s, l, equal_var=False); u = stats.mannwhitneyu(s, l)
    R["attrition_cei"] = {"n_stayed": len(s), "n_left": len(l), "mean_stayed": s.mean(), "mean_left": l.mean(),
                          "diff": s.mean() - l.mean(), "d": cohens_d(s, l), "boot_ci": boot_diff(s, l, rng),
                          "welch_t": float(t.statistic), "welch_p": float(t.pvalue), "mannwhitney_p": float(u.pvalue)}
    R["attrition_by_burnout"] = {k: {"n": len(g), "left": int(g.Attrition.sum()), "rate": float(g.Attrition.mean())} for k, g in df.groupby("BurnoutRisk")}
    R["attrition_by_overtime"] = {k: {"n": len(g), "left": int(g.Attrition.sum()), "rate": float(g.Attrition.mean())} for k, g in df.groupby("OverTime")}
    R["attrition_by_cei_band"] = {str(k): {"n": len(g), "rate": float(g.Attrition.mean())} for k, g in df.assign(b=pd.cut(df.CEI, [0, 50, 62.5, 75, 101], right=False)).groupby("b", observed=True)}
    chi = stats.chi2_contingency(pd.crosstab(df.BurnoutRisk, df.Attrition))
    R["attrition_burnout_chi2"] = {"chi2": float(chi[0]), "p": float(chi[1])}

    # --- heuristic sensitivity
    from pan_hr import metrics as M
    base = df.WSI
    alt = {"travel_heavy": {"travel": .5, "commute": .25, "overtime": .25}, "overtime_heavy": {"travel": .25, "commute": .25, "overtime": .5}}
    R["wsi_sensitivity_spearman_vs_equal"] = {k: float(stats.spearmanr(base, M.compute_wsi(df, w))[0]) for k, w in alt.items()}
    altc = {"promo_heavy": {"unpromoted_share": .75, "role_share": .25}, "role_heavy": {"unpromoted_share": .25, "role_share": .75}}
    sub = df.CSI.notna()
    R["csi_sensitivity_spearman_vs_equal"] = {k: float(stats.spearmanr(df.CSI[sub], M.compute_csi(df, w)[sub])[0]) for k, w in altc.items()}
    R["wsi_summary"] = {"mean": float(df.WSI.mean()), "sd": float(df.WSI.std())}
    R["csi_summary"] = {"mean": float(df.CSI.mean()), "sd": float(df.CSI.std()),
                        "band_counts": df.CSIBand.value_counts().to_dict()}
    R["wsi_by_department"] = {k: float(v) for k, v in df.groupby("Department").WSI.mean().items()}

    # --- action panel & fairness audit
    pr = A.priority_areas(df, THR)
    R["priority_areas"] = pr.drop(columns=["precision"]).assign(flags=pr["flags"].map(lambda x: ",".join(x))).to_dict("records")
    pr.drop(columns=["precision"]).assign(flags=pr["flags"].map(lambda x: ",".join(x))).to_csv(out / "tables" / "priority_areas.csv", index=False)
    R["fairness_audit"] = {c: A.fairness_table(df, c, THR).to_dict("records") for c in ["Gender", "MaritalStatus"]}
    R["priority_with_overtime_only"] = (lambda p: p[["group", "n", "cei_mean", "cei_gap", "pct_high_burnout", "flags", "tier"]].assign(flags=p["flags"].map(",".join)).to_dict("records"))(
        A.priority_areas(df[df.OverTime == "Yes"], THR))

    # --- save json/tables
    (out / "analysis_results.json").write_text(json.dumps(_clean(R), indent=2, default=str))
    for key in ["cei_by_department", "cei_by_role", "cei_by_level", "cei_by_tenure", "cei_by_role_tenure", "cei_by_promo_gap", "cei_by_csi_band", "cei_by_travel", "cei_by_commute", "cei_by_overtime"]:
        pd.DataFrame(R[key]).to_csv(out / "tables" / f"{key}.csv", index=False)
    make_figures(df, out / "figures", R)
    print("Wrote", out)


def _alpha(items: pd.DataFrame) -> float:
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if pd.isna(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def make_figures(df, fdir: Path, R: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": C.PAPER, "axes.facecolor": C.PAPER, "savefig.facecolor": C.PAPER})
    # 1 CEI histogram
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    h = A.cei_histogram(df)
    ax.bar(h["bin"], h["share"] * 100, color=C.INDIGO)
    ax.set_ylabel("% of records"); ax.set_xlabel("CEI (0-100 scale)")
    ax.axvline(x=4.5 - 0.0, color="none")
    ax.set_title("Composite Engagement Index distribution (n=1,470)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(fdir / "fig1_cei_distribution.png", dpi=200); plt.close(fig)
    # 2 satisfaction stacks
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    cols = C.CEI_COLUMNS + ["WorkLifeBalance"]
    left = np.zeros(len(cols))
    colors = {1: "#C8472B", 2: "#E3A06C", 3: "#7FB5AE", 4: C.TEAL}
    for s in (1, 2, 3, 4):
        v = np.array([(df[c] == s).mean() * 100 for c in cols])
        ax.barh([C.SCALE_LABELS[c] for c in cols], v, left=left, color=colors[s], label=str(s)); left += v
    ax.invert_yaxis(); ax.set_xlabel("% of records"); ax.legend(title="Rating", ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.75))
    ax.set_title("Rating distributions (1 = low, 4 = high)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(fdir / "fig2_satisfaction.png", dpi=200); plt.close(fig)
    # 3 burnout mix + CEI by level
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
    b = R["burnout"]["counts"]
    axs[0].bar(C.BURNOUT_LEVELS, [b[l] / len(df) * 100 for l in C.BURNOUT_LEVELS], color=[C.RISK_COLORS[l] for l in C.BURNOUT_LEVELS])
    for i, l in enumerate(C.BURNOUT_LEVELS):
        axs[0].text(i, b[l] / len(df) * 100 + 1, f"{b[l]/len(df):.1%}\n(n={b[l]})", ha="center", fontsize=8)
    axs[0].set_ylim(0, 65); axs[0].set_ylabel("% of records"); axs[0].set_title("Screening-level mix", loc="left", fontsize=10)
    for i, l in enumerate(C.BURNOUT_LEVELS):
        r = R["cei_by_burnout"][l]; axs[1].errorbar(i, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", color=C.RISK_COLORS[l], capsize=4, ms=7)
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(C.BURNOUT_LEVELS); axs[1].set_ylabel("Mean CEI (95% CI)"); axs[1].set_title("Mean CEI by level (zoomed axis)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(fdir / "fig3_burnout.png", dpi=200); plt.close(fig)
    # 4 role dots
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    t = pd.DataFrame(R["cei_by_role"]).sort_values("cei_mean")
    ax.errorbar(t.cei_mean, range(len(t)), xerr=[t.cei_mean - t.cei_lo, t.cei_hi - t.cei_mean], fmt="o", color=C.INDIGO, ecolor=C.SLATE, capsize=3)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.group); ax.axvline(df.CEI.mean(), ls=":", color=C.VERMILION)
    ax.set_xlabel("Mean CEI (95% CI; zoomed axis)"); ax.set_title("Engagement by job role", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(fdir / "fig4_roles.png", dpi=200); plt.close(fig)
    # 5 workload panels
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, key, title in zip(axs, ["cei_by_overtime", "cei_by_travel", "cei_by_commute"], ["Overtime", "Business travel", "Commute band"]):
        t = pd.DataFrame(R[key])
        ax.errorbar(range(len(t)), t.cei_mean, yerr=[t.cei_mean - t.cei_lo, t.cei_hi - t.cei_mean], fmt="o", color=C.INDIGO, capsize=4)
        ax.set_xticks(range(len(t))); ax.set_xticklabels([g.replace(" ", "\n", 1) for g in t.group], fontsize=7); ax.set_title(title, loc="left", fontsize=9)
    axs[0].set_ylabel("Mean CEI (95% CI)")
    fig.tight_layout(); fig.savefig(fdir / "fig5_workload.png", dpi=200); plt.close(fig)
    # 6 tenure & career
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, key, title in zip(axs, ["cei_by_tenure", "cei_by_role_tenure", "cei_by_csi_band"], ["Years at company", "Years in current role", "CSI band"]):
        t = pd.DataFrame(R[key])
        ax.errorbar(range(len(t)), t.cei_mean, yerr=[t.cei_mean - t.cei_lo, t.cei_hi - t.cei_mean], fmt="-o", color=C.TEAL, capsize=4)
        ax.set_xticks(range(len(t))); ax.set_xticklabels([g.replace(" (", "\n(") for g in t.group], fontsize=7); ax.set_title(title, loc="left", fontsize=9)
    axs[0].set_ylabel("Mean CEI (95% CI)")
    fig.tight_layout(); fig.savefig(fdir / "fig6_career.png", dpi=200); plt.close(fig)
    # 7 attrition
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    a = R["attrition_cei"]
    ax.bar(["Stayed", "Left"], [a["mean_stayed"], a["mean_left"]], color=[C.TEAL, C.SLATE])
    for i, v in enumerate([a["mean_stayed"], a["mean_left"]]): ax.text(i, v + 1, f"{v:.1f}", ha="center")
    ax.set_ylim(0, 100); ax.set_ylabel("Mean CEI"); ax.set_title("Mean CEI by attrition flag", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(fdir / "fig7_attrition.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "Research_Paper" / "analysis"))
    main(Path(ap.parse_args().out))
