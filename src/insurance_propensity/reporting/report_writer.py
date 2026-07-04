"""Report writers for portfolio-ready model and business artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from insurance_propensity.config import REPORT_DIR


def write_json(path: Path, payload: dict) -> None:
    """Write compact, deterministic JSON artifacts for auditability."""
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_data_quality_report(summary: dict) -> None:
    """Write a concise markdown data quality report."""
    target = summary["target_distribution"]
    premium = summary["annual_premium_outlier_review"]
    content = f"""# Data Quality Report

## Raw Data Contract
- Train shape: `{summary['train_shape'][0]:,}` rows x `{summary['train_shape'][1]}` columns
- Test shape: `{summary['test_shape'][0]:,}` rows x `{summary['test_shape'][1]}` columns
- Sample submission shape: `{summary['sample_submission_shape'][0]:,}` rows x `{summary['sample_submission_shape'][1]}` columns
- Total missing values: train `{summary['total_missing_values']['train']}`, test `{summary['total_missing_values']['test']}`
- Duplicate rows: train `{summary['duplicate_rows']['train']}`, test `{summary['duplicate_rows']['test']}`
- Unique train ID: `{summary['id_quality']['train_unique_id']}`
- Unique test ID: `{summary['id_quality']['test_unique_id']}`

## Target Balance
- Positive responders: `{target['counts'].get(1, 0):,}`
- Non-responders: `{target['counts'].get(0, 0):,}`
- Positive response rate: `{target['positive_rate']:.2%}`

## Annual Premium Outlier Review
- IQR upper fence: `{premium['iqr_upper_fence']:,.2f}`
- Outlier count: `{premium['outlier_count']:,}`
- Outlier rate: `{premium['outlier_rate']:.2%}`

## Leakage Audit
The project excludes `id` and `Response` from predictive features, never trains on `test.csv`, and uses out-of-fold target encoding for regional and sales channel response-rate encodings.
"""
    (REPORT_DIR / "data_quality_report.md").write_text(content, encoding="utf-8")


def write_business_reports(best_name: str, best_metrics: dict, deciles: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Write model evaluation, campaign, and interpretation reports."""
    threshold = best_metrics["best_threshold"]["threshold"]
    confusion = best_metrics["confusion_matrix"]
    campaign = deciles.loc[
        deciles["propensity_decile"].isin([1, 2, 3]),
        [
            "propensity_decile",
            "customers",
            "responders",
            "response_rate",
            "lift",
            "cumulative_capture_rate",
        ],
    ]
    campaign.to_csv(REPORT_DIR / "campaign_targeting_report.csv", index=False)

    model_report = f"""# Model Evaluation Report

## Final Validation Model
Selected model: **{best_name}**

| Metric | Value |
|---|---:|
| ROC-AUC | {best_metrics['roc_auc']:.4f} |
| PR-AUC | {best_metrics['pr_auc']:.4f} |
| Precision @ 0.50 | {best_metrics['precision']:.4f} |
| Recall @ 0.50 | {best_metrics['recall']:.4f} |
| F1 @ 0.50 | {best_metrics['f1']:.4f} |
| Balanced Accuracy @ 0.50 | {best_metrics['balanced_accuracy']:.4f} |
| Top Decile Lift | {best_metrics['top_decile_lift']:.2f}x |
| Capture Rate Top 10% | {best_metrics['capture_rate_top_10_pct']:.2%} |
| Capture Rate Top 20% | {best_metrics['capture_rate_top_20_pct']:.2%} |
| Capture Rate Top 30% | {best_metrics['capture_rate_top_30_pct']:.2%} |

## Threshold Guidance
The F1-oriented validation threshold is **{threshold:.2f}**. For campaign operations, ranked targeting is more useful than a single classification threshold because the business action is prioritizing outreach capacity.

## Confusion Matrix @ 0.50
`{confusion}`

## Campaign Targeting Readout
Top deciles concentrate the highest observed response share. Recommended launch strategy:

1. Start with decile 1 for highest-intent outreach.
2. Expand to deciles 1-3 when the campaign team has broader capacity.
3. Keep lower deciles as holdout, nurture, or low-frequency education audiences.

This is a propensity-ranking simulation based on validation data, not a claim of guaranteed campaign ROI.
"""
    (REPORT_DIR / "model_evaluation_report.md").write_text(model_report, encoding="utf-8")

    top_drivers = "\n".join(
        f"- `{row.feature}`: {row.importance_mean:.5f}"
        for row in importance.head(10).itertuples(index=False)
    )
    interpretation = f"""# Model Interpretability Report

Primary interpretability artifact: **permutation importance**.

Permutation importance is used for the committed report because it is lightweight, deterministic, and works with the deployed scikit-learn pipeline. SHAP remains available as an optional local extension for deeper individual-level analysis.

## Global Drivers
{top_drivers}

## Business Interpretation
- Previous vehicle damage and not-yet-insured status are expected to be strong positive signals for cross-sell relevance.
- Vehicle age, customer age band, annual premium band, region, and sales channel help prioritize customer groups, but they should be treated as associative targeting signals rather than causal proof.
- The model is designed for customer ranking. Business users should review decile lift, cumulative gain, and capture rate before selecting campaign reach.

## Local Explanation Workflow
For an individual customer, the Streamlit dashboard surfaces the final propensity score, decile, segment, and recommendation. When SHAP is installed, the notebook can be extended with local force/waterfall explanations using the saved scorer bundle.
"""
    (REPORT_DIR / "interpretability_report.md").write_text(interpretation, encoding="utf-8")
