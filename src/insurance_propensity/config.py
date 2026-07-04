"""Central project configuration."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
CHART_DIR = OUTPUT_DIR / "charts"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TARGET_COLUMN = "Response"
ID_COLUMN = "id"

TRAIN_COLUMNS = [
    "id",
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
    "Response",
]

TEST_COLUMNS = [column for column in TRAIN_COLUMNS if column != TARGET_COLUMN]

LIGHT_THEME = {
    "background": "#D6E4F0",
    "card": "#F7FAFC",
    "secondary_surface": "#E2E8F0",
    "control": "#EEF5FA",
    "plot": "#F8FBFE",
    "grid": "#C6D3DF",
    "border": "#B8C8D8",
    "text": "#172033",
    "muted_text": "#4B5D6B",
    "navy": "#0B1F33",
    "blue": "#2563EB",
    "teal": "#1F7A8C",
    "green": "#2E7D32",
    "amber": "#B7791F",
    "red": "#C62828",
}

DARK_THEME = {
    "background": "#0B1F33",
    "card": "#132F4C",
    "secondary_surface": "#1C3E63",
    "control": "#17385A",
    "plot": "#102A44",
    "grid": "#284766",
    "border": "#315B7F",
    "text": "#F8FAFC",
    "muted_text": "#CBD5E1",
    "blue": "#38BDF8",
    "teal": "#2DD4BF",
    "green": "#22C55E",
    "amber": "#F59E0B",
    "red": "#EF4444",
}


def ensure_project_directories() -> None:
    """Create project output directories if they do not exist."""
    for path in [
        DATA_PROCESSED_DIR,
        REPORT_DIR,
        CHART_DIR,
        PREDICTION_DIR,
        SUBMISSION_DIR,
        MODEL_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
