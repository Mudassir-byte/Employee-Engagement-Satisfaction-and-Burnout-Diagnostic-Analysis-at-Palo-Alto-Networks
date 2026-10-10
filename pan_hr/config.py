"""Single source of truth for schema, metric definitions, thresholds and palette.

Everything the UI, tests and research paper quote about definitions lives here so
the documentation cannot drift from the code.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Privacy
# --------------------------------------------------------------------------- #
K_MIN = 5  # minimum cohort size from the strategic plan
OTHER_LABEL = "Other (small groups combined)"
WITHHELD_LABEL = "Withheld (cells < 5, combined)"

# --------------------------------------------------------------------------- #
# Schema (31 fields from the problem statement)
# --------------------------------------------------------------------------- #
NUMERIC_COLUMNS = [
    "Age", "Attrition", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel",
    "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]
STRING_COLUMNS = [
    "BusinessTravel", "Department", "EducationField", "Gender", "JobRole",
    "MaritalStatus", "OverTime",
]
EXPECTED_COLUMNS = sorted(NUMERIC_COLUMNS + STRING_COLUMNS)
EXPECTED_N_COLUMNS = 31

# 1-4 ordinal scales documented in the problem statement.
SCALE_COLUMNS = [
    "JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction",
    "RelationshipSatisfaction", "WorkLifeBalance",
]
SCALE_MIN, SCALE_MAX = 1, 4

# Closed categorical sets (documented). Department / JobRole / EducationField are
# documented only with "e.g.", so they are treated as open sets and reported.
ALLOWED_VALUES = {
    "Attrition": {0, 1},
    "BusinessTravel": {"Non-Travel", "Travel_Rarely", "Travel_Frequently"},
    "OverTime": {"Yes", "No"},
    "Gender": {"Male", "Female"},
    "MaritalStatus": {"Single", "Married", "Divorced"},
}
INTEGER_RANGES = {  # documented ranges (inclusive)
    "Education": (1, 5),
}
NON_NEGATIVE_COLUMNS = [
    "Age", "DistanceFromHome", "NumCompaniesWorked", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    "DailyRate", "HourlyRate", "MonthlyIncome", "MonthlyRate", "PercentSalaryHike",
]
OPEN_CATEGORY_COLUMNS = ["Department", "JobRole", "EducationField"]

# --------------------------------------------------------------------------- #
# Metric definitions
# --------------------------------------------------------------------------- #
CEI_COLUMNS = [
    "JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
]
CEI_MAX_TOTAL = 16  # 4 items x max 4

BURNOUT_LEVELS = ["Low", "Medium", "High"]
BURNOUT_UNCLASSIFIED = "Unclassified"
LOW_WLB_CUTOFF = 2  # WorkLifeBalance <= 2

# Workload Stress Indicator (PROPOSED HEURISTIC)
TRAVEL_COEFFICIENTS = {"Non-Travel": 0.0, "Travel_Rarely": 0.5, "Travel_Frequently": 1.0}
COMMUTE_REF_MIN, COMMUTE_REF_MAX = 1, 29  # dataset-observed range; unit unspecified
WSI_WEIGHTS = {"travel": 1 / 3, "commute": 1 / 3, "overtime": 1 / 3}

# Career Stagnation Index (PROPOSED HEURISTIC)
CSI_MIN_TENURE_YEARS = 2  # below this the ratios are mechanically unstable
CSI_WEIGHTS = {"unpromoted_share": 0.5, "role_share": 0.5}
CSI_BANDS = [("Lower (<40)", 0, 40), ("Moderate (40-69)", 40, 70), ("Higher (70+)", 70, 100.0001)]

# Band definitions (years / distance units as supplied; the commute unit is not documented)
COMMUTE_BANDS = [("Short (1-5)", 1, 5), ("Medium (6-15)", 6, 15), ("Long (16+)", 16, 10_000)]
TENURE_BANDS = [("0-1 yrs", 0, 1), ("2-4 yrs", 2, 4), ("5-9 yrs", 5, 9), ("10-19 yrs", 10, 19), ("20+ yrs", 20, 10_000)]
ROLE_TENURE_BANDS = [("0-1 yrs", 0, 1), ("2-3 yrs", 2, 3), ("4-6 yrs", 4, 6), ("7+ yrs", 7, 10_000)]
PROMOTION_GAP_BANDS = [("0-1 yrs", 0, 1), ("2-3 yrs", 2, 3), ("4-6 yrs", 4, 6), ("7+ yrs", 7, 10_000)]

TRAVEL_ORDER = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
TRAVEL_LABELS = {"Non-Travel": "No travel", "Travel_Rarely": "Travels rarely", "Travel_Frequently": "Travels frequently"}

# Manager Action Panel rule thresholds (PROPOSED HEURISTICS - editable, inspectable)
HIGH_BURNOUT_SHARE_PROMPT = 0.15      # from the strategic plan (workload review prompt)
ENGAGEMENT_GAP_POINTS = 5.0           # CEI points below comparator
STAGNATION_GAP_POINTS = 10.0          # mean CSI points above organisation view
WORKLOAD_GAP_POINTS = 10.0            # mean WSI points above organisation view
DEFAULT_LOW_ENGAGEMENT_THRESHOLD = 50  # CEI < 50  <=> average item score below 2 of 4

SCALE_LABELS = {
    "JobInvolvement": "Job involvement",
    "JobSatisfaction": "Job satisfaction",
    "EnvironmentSatisfaction": "Environment satisfaction",
    "RelationshipSatisfaction": "Relationship satisfaction",
    "WorkLifeBalance": "Work-life balance",
}

# --------------------------------------------------------------------------- #
# Palette (restrained: warm paper + ink; signal colours reserved for risk)
# --------------------------------------------------------------------------- #
INK = "#14202B"
PAPER = "#F5F1E8"
PAPER_DEEP = "#EAE3D2"
CARD = "#FBF9F4"
SLATE = "#5B6B78"
RULE = "#D6CFBD"
TEAL = "#1B7F79"
AMBER = "#D49A12"
VERMILION = "#D2441C"
INDIGO = "#2E4B73"
GREY = "#B4AEA0"
RISK_COLORS = {"Low": TEAL, "Medium": AMBER, "High": VERMILION, BURNOUT_UNCLASSIFIED: GREY}
SCREENING_DISCLAIMER = (
    "Rule-based screening proxy for aggregate work conditions in the provided dataset - "
    "not a clinical diagnosis, a validated prediction, or a judgement about any individual."
)
