"""Ranking-oriented evaluation metrics for propensity models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_sweep(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """Evaluate business-friendly thresholds."""
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def decile_table(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """Create validation deciles where decile 1 is the highest score band."""
    frame = pd.DataFrame({"actual": y_true.to_numpy(), "score": y_prob})
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    frame["propensity_decile"] = pd.qcut(frame.index + 1, 10, labels=list(range(1, 11))).astype(int)
    overall_rate = frame["actual"].mean()
    total_responders = frame["actual"].sum()
    grouped = (
        frame.groupby("propensity_decile", observed=True)
        .agg(
            customers=("actual", "size"),
            responders=("actual", "sum"),
            avg_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
    )
    grouped["response_rate"] = grouped["responders"] / grouped["customers"]
    grouped["lift"] = grouped["response_rate"] / overall_rate
    grouped["capture_rate"] = grouped["responders"] / total_responders
    grouped["cumulative_customers"] = grouped["customers"].cumsum()
    grouped["cumulative_responders"] = grouped["responders"].cumsum()
    grouped["cumulative_capture_rate"] = grouped["cumulative_responders"] / total_responders
    grouped["population_share"] = grouped["cumulative_customers"] / len(frame)
    return grouped


def capture_at_k(deciles: pd.DataFrame, k: int) -> float:
    """Return responder capture rate for the top k percent of the ranked list."""
    top_deciles = max(1, int(k / 10))
    return float(deciles.loc[deciles["propensity_decile"].le(top_deciles), "capture_rate"].sum())


def binary_classification_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Calculate classification and ranking metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    deciles = decile_table(y_true, y_prob)
    threshold_df = threshold_sweep(y_true, y_prob)
    best_threshold_row = threshold_df.sort_values(["f1", "balanced_accuracy"], ascending=False).iloc[0].to_dict()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
        "top_decile_lift": float(deciles.loc[deciles["propensity_decile"].eq(1), "lift"].iloc[0]),
        "capture_rate_top_10_pct": capture_at_k(deciles, 10),
        "capture_rate_top_20_pct": capture_at_k(deciles, 20),
        "capture_rate_top_30_pct": capture_at_k(deciles, 30),
        "best_threshold": best_threshold_row,
    }