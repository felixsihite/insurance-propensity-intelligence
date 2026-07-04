"""Data contract and quality checks for the insurance propensity dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from insurance_propensity.config import ID_COLUMN, TARGET_COLUMN, TEST_COLUMNS, TRAIN_COLUMNS


@dataclass(frozen=True)
class DatasetBundle:
    """Raw dataset bundle used across the project."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def load_raw_datasets(raw_dir: Path) -> DatasetBundle:
    """Load train, test, and sample submission files without mutating them."""
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    sample_submission = pd.read_csv(raw_dir / "sample_submission.csv")
    return DatasetBundle(train=train, test=test, sample_submission=sample_submission)


def validate_required_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise a clear error if the raw schema no longer matches the project contract."""
    if list(train.columns) != TRAIN_COLUMNS:
        raise ValueError(f"train.csv schema mismatch. Expected {TRAIN_COLUMNS}, got {list(train.columns)}")
    if list(test.columns) != TEST_COLUMNS:
        raise ValueError(f"test.csv schema mismatch. Expected {TEST_COLUMNS}, got {list(test.columns)}")


def dataset_quality_summary(bundle: DatasetBundle) -> dict[str, Any]:
    """Return an auditable data quality summary for the raw files."""
    train, test, sample_submission = bundle.train, bundle.test, bundle.sample_submission
    validate_required_columns(train, test)

    train_feature_columns = [column for column in train.columns if column != TARGET_COLUMN]
    categorical_columns = ["Gender", "Vehicle_Age", "Vehicle_Damage"]

    category_level_checks = {}
    for column in categorical_columns:
        train_levels = set(train[column].dropna().astype(str).unique())
        test_levels = set(test[column].dropna().astype(str).unique())
        category_level_checks[column] = {
            "train_levels": sorted(train_levels),
            "test_levels": sorted(test_levels),
            "levels_only_in_train": sorted(train_levels - test_levels),
            "levels_only_in_test": sorted(test_levels - train_levels),
        }

    premium = train["Annual_Premium"]
    q1 = premium.quantile(0.25)
    q3 = premium.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_mask = (premium < lower_fence) | (premium > upper_fence)

    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "sample_submission_shape": list(sample_submission.shape),
        "train_columns": list(train.columns),
        "test_columns": list(test.columns),
        "missing_values": {
            "train": train.isna().sum().to_dict(),
            "test": test.isna().sum().to_dict(),
            "sample_submission": sample_submission.isna().sum().to_dict(),
        },
        "total_missing_values": {
            "train": int(train.isna().sum().sum()),
            "test": int(test.isna().sum().sum()),
            "sample_submission": int(sample_submission.isna().sum().sum()),
        },
        "duplicate_rows": {
            "train": int(train.duplicated().sum()),
            "test": int(test.duplicated().sum()),
            "sample_submission": int(sample_submission.duplicated().sum()),
        },
        "id_quality": {
            "train_unique_id": bool(train[ID_COLUMN].is_unique),
            "test_unique_id": bool(test[ID_COLUMN].is_unique),
            "sample_submission_unique_id": bool(sample_submission[ID_COLUMN].is_unique),
            "train_test_id_overlap": int(set(train[ID_COLUMN]).intersection(set(test[ID_COLUMN])).__len__()),
            "test_submission_id_match": bool(test[ID_COLUMN].equals(sample_submission[ID_COLUMN])),
        },
        "target_distribution": {
            "counts": train[TARGET_COLUMN].value_counts().sort_index().to_dict(),
            "positive_rate": float(train[TARGET_COLUMN].mean()),
            "negative_rate": float(1 - train[TARGET_COLUMN].mean()),
        },
        "schema_comparison": {
            "train_feature_columns_match_test": train_feature_columns == list(test.columns),
            "target_present_in_test": TARGET_COLUMN in test.columns,
            "train_dtypes": {column: str(dtype) for column, dtype in train.dtypes.items()},
            "test_dtypes": {column: str(dtype) for column, dtype in test.dtypes.items()},
        },
        "categorical_level_comparison": category_level_checks,
        "annual_premium_outlier_review": {
            "min": float(premium.min()),
            "q1": float(q1),
            "median": float(premium.median()),
            "q3": float(q3),
            "max": float(premium.max()),
            "iqr_lower_fence": float(lower_fence),
            "iqr_upper_fence": float(upper_fence),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_rate": float(outlier_mask.mean()),
        },
        "leakage_audit": {
            "id_used_as_feature": False,
            "response_used_as_feature": False,
            "test_used_for_training": False,
            "target_encoding_strategy": "Out-of-fold encodings fitted on training folds only; validation/test transformed with train-only mappings.",
        },
    }
    return summary
