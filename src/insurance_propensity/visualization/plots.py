"""Project chart generation with a consistent insurance analytics theme."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from insurance_propensity.config import LIGHT_THEME
from insurance_propensity.models.modeling import FEATURE_COLUMNS


def set_theme() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.facecolor": LIGHT_THEME["background"],
            "axes.facecolor": LIGHT_THEME["card"],
            "axes.edgecolor": "#A7B7C8",
            "axes.labelcolor": LIGHT_THEME["text"],
            "xtick.color": LIGHT_THEME["text"],
            "ytick.color": LIGHT_THEME["text"],
            "text.color": LIGHT_THEME["text"],
            "axes.titleweight": "bold",
            "axes.titlecolor": LIGHT_THEME["text"],
        }
    )


def save_response_distribution(train: pd.DataFrame, output_path: Path) -> None:
    set_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    response = train["Response"].map({0: "No Response", 1: "Positive Response"})
    sns.countplot(x=response, hue=response, palette=[LIGHT_THEME["teal"], LIGHT_THEME["blue"]], ax=ax, legend=False)
    ax.set_title("Response Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_segment_response(train: pd.DataFrame, output_path: Path) -> None:
    set_theme()
    grouped = train.groupby("cross_sell_opportunity_segment", observed=True)["Response"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=grouped.values, y=grouped.index, color=LIGHT_THEME["blue"], ax=ax)
    ax.set_title("Observed Response Rate by Cross-Sell Segment")
    ax.set_xlabel("Response Rate")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_premium_distribution(train: pd.DataFrame, output_path: Path) -> None:
    set_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train["Annual_Premium"].astype(float), bins=60, color=LIGHT_THEME["teal"])
    ax.set_title("Annual Premium Distribution")
    ax.set_xlabel("Annual Premium")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_lift_chart(deciles: pd.DataFrame, output_path: Path) -> None:
    set_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=deciles, x="propensity_decile", y="lift", color=LIGHT_THEME["blue"], ax=ax)
    ax.axhline(1.0, color=LIGHT_THEME["red"], linestyle="--", linewidth=1.2)
    ax.set_title("Lift by Propensity Decile")
    ax.set_xlabel("Propensity Decile (1 = Highest)")
    ax.set_ylabel("Lift vs. Overall Response Rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_gain_chart(deciles: pd.DataFrame, output_path: Path) -> None:
    set_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(deciles["population_share"], deciles["cumulative_capture_rate"], marker="o", color=LIGHT_THEME["blue"], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=LIGHT_THEME["muted_text"], linewidth=1)
    ax.set_title("Cumulative Gain Chart")
    ax.set_xlabel("Cumulative Population Targeted")
    ax.set_ylabel("Cumulative Responders Captured")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_roc_pr_calibration(y_true, y_prob, output_dir: Path) -> None:
    set_theme()
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, color=LIGHT_THEME["blue"])
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, color=LIGHT_THEME["teal"])
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=180)
    plt.close(fig)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", color=LIGHT_THEME["blue"], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=LIGHT_THEME["muted_text"], linewidth=1)
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Response Rate")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix(y_true, y_prob, threshold: float, output_path: Path) -> None:
    set_theme()
    y_pred = (y_prob >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix at Threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def calculate_permutation_importance(model, x_valid: pd.DataFrame, y_valid: pd.Series, output_path: Path, max_rows: int = 20_000) -> pd.DataFrame:
    sample = x_valid[FEATURE_COLUMNS]
    target = y_valid
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=42)
        target = target.loc[sample.index]
    result = permutation_importance(model, sample, target, scoring="average_precision", n_repeats=5, random_state=42, n_jobs=-1)
    importances_mean = np.asarray(result["importances_mean"], dtype=float)
    importances_std = np.asarray(result["importances_std"], dtype=float)
    importance = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_mean": importances_mean,
                "importance_std": importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_path, index=False)
    return importance


def save_feature_importance(importance: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    set_theme()
    plot_data = importance.head(top_n).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=plot_data, x="importance_mean", y="feature", color=LIGHT_THEME["blue"], ax=ax)
    ax.set_title("Top Model Drivers")
    ax.set_xlabel("Permutation Importance (PR-AUC Decrease)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
