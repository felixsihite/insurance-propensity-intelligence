"""Production-style package for insurance cross-sell propensity intelligence.

The package exposes reusable components for data validation, feature
engineering, model training, customer scoring, reporting, and dashboard
delivery. Raw Kaggle files are treated as immutable source data; all derived
assets are generated into `data/processed`, `models`, and `outputs`.
"""

from __future__ import annotations

from dataclasses import dataclass

__version__ = "1.0.0"
PROJECT_NAME = "Insurance Customer Propensity Prediction & Customer Intelligence"
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction"


@dataclass(frozen=True)
class ProjectMetadata:
    """Human-readable package metadata used by reports and portfolio docs."""

    name: str
    version: str
    dataset_source_url: str
    business_objective: str


def get_project_metadata() -> ProjectMetadata:
    """Return canonical project metadata without importing heavy ML modules."""
    return ProjectMetadata(
        name=PROJECT_NAME,
        version=__version__,
        dataset_source_url=DATASET_SOURCE_URL,
        business_objective=(
            "Rank existing health insurance customers by their propensity to "
            "purchase vehicle insurance and support targeted cross-sell campaigns."
        ),
    )


__all__ = [
    "DATASET_SOURCE_URL",
    "PROJECT_NAME",
    "ProjectMetadata",
    "__version__",
    "get_project_metadata",
]
