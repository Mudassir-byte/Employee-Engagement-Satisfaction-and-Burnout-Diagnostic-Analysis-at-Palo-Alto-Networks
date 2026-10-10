# Engagement, Satisfaction & Burnout Diagnostic (Streamlit)

An aggregate-only HR analytics workspace built on the supplied 31-field dataset
(`data/palo_alto_networks.csv`, 1,470 records). It supports early identification of engagement gaps and
rule-based burnout-risk *screening* areas so leaders can investigate workload and work conditions.

> **Scope warning.** The CSV is a single snapshot and is not proven to be actual Palo Alto Networks employee data.
> All findings describe the provided dataset only. Scores are not diagnoses, predictions or causal estimates, and
> must not be used for individual employment decisions.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt
streamlit run app.py                                    # http://localhost:8501
```

Docker (config provided; see "Verification status" in the paper appendix for what was actually run):

```bash
docker build -t pan-engagement-dashboard .
docker run --rm -p 8501:8501 pan-engagement-dashboard
```

## Tests and reproducibility

```bash
pip install -r requirements-dev.txt
pytest                                   # 55 tests: validation, CEI, burnout rules, WSI/CSI, privacy, filters, app smoke tests
python scripts/reproduce_analysis.py     # regenerates ../Research_Paper/analysis (JSON, tables, figures)
```

## Layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit shell: sidebar filters, navigation, cross-filter state |
| `pan_hr/config.py` | Schema, metric definitions, thresholds, palette (single source of truth) |
| `pan_hr/validation.py` | 21 schema/type/range/category/duplicate/missing checks; never mutates source data |
| `pan_hr/metrics.py` | CEI, burnout screening, WLB, stability score, WSI, CSI (pure pandas) |
| `pan_hr/privacy.py` | Minimum cohort size (k=5) suppression, pooling, cell withholding |
| `pan_hr/analysis.py` | Protected aggregations, attrition comparison, action-panel rules, fairness audit |
| `pan_hr/filters.py` | Filter state and application (UI-independent) |
| `pan_hr/charts.py`, `views.py`, `ui.py` | Plotly figures, the five views, styling |
| `tests/` | pytest suite |

## Views

1. **Engagement Health** - CEI, rating distributions, department/role comparison (click a role to cross-filter), contextual attrition association, optional fairness audit.
2. **Burnout Risk** - screening-level mix (click to cross-filter), overtime x work-life-balance matrix, mix by department/level, engagement by overtime/travel/commute.
3. **Role & Career Stage** - level, role x level, tenure trends, promotion gap and Career Stagnation Index with inspectable components.
4. **Manager Action Panel** - prioritised *aggregate* areas, rule-based prompts, suggested investigation steps.
5. **Data & Method** - validation results, coverage, metric definitions, limitations, privacy rules.

Filters: department, role, overtime (All/Yes/No), tenure measure + range, low-engagement threshold, optional
"restrict to below threshold", reset. A Motion toggle (plus the OS reduced-motion setting) disables animation.

## Metrics

* **CEI** = (JobInvolvement + JobSatisfaction + EnvironmentSatisfaction + RelationshipSatisfaction) / 16 x 100 (range 25-100). Any missing input -> not computed.
* **Burnout screening level**: High = OverTime Yes AND WorkLifeBalance <= 2; Medium = exactly one; Low = neither.
* **Proposed heuristics (not validated; weights/bands are assumptions, editable in `config.py`)**: Workload Stress Indicator, Career Stagnation Index, Satisfaction Stability Score.

## Privacy design

Aggregates only; no row-level tables or tooltips; no employee rankings. Groups under 5 are pooled or withheld, rating/risk cells under 5
are withheld with complementary suppression, heatmap cells under 5 are blanked, and filter combinations that leave fewer than 5 records
show no results. Gender and marital status are never used for risk or action; an optional aggregated audit exists. The app makes no network
calls (fonts are system fonts) and has no tracking or credentials.

**Residual risk:** repeatedly nudging filters could allow differencing between near-identical views. For real deployment add access control and
query logging. Longitudinal persistence, prediction accuracy and turnover-reduction targets in the strategic plan cannot be assessed from this snapshot.
