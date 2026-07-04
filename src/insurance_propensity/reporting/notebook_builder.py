"""Generate clean portfolio notebooks from the reproducible project pipeline."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def _new_code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def _new_markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def generate_notebooks(project_root: Path) -> None:
    """Create notebook files that open cleanly in VS Code and run top-to-bottom."""
    notebooks_dir = project_root / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    common_setup = """
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    """

    notebooks = {
        "01_data_quality_and_eda.ipynb": [
            _new_markdown_cell(
                """
                # 01 - Data Quality and EDA

                This notebook audits the raw Kaggle insurance cross-sell dataset, validates the competition contract, and identifies customer segments with materially different response behavior.
                """
            ),
            _new_code_cell(common_setup),
            _new_markdown_cell("## Dataset Contract"),
            _new_code_cell(
                """
                import pandas as pd

                from insurance_propensity.config import DATA_RAW_DIR, REPORT_DIR
                from insurance_propensity.data.validation import load_raw_datasets, dataset_quality_summary

                bundle = load_raw_datasets(DATA_RAW_DIR)
                summary = dataset_quality_summary(bundle)
                pd.Series(
                    {
                        "train_rows": summary["train_shape"][0],
                        "train_columns": summary["train_shape"][1],
                        "test_rows": summary["test_shape"][0],
                        "sample_submission_rows": summary["sample_submission_shape"][0],
                        "train_missing_values": summary["total_missing_values"]["train"],
                        "test_missing_values": summary["total_missing_values"]["test"],
                        "duplicate_train_rows": summary["duplicate_rows"]["train"],
                        "train_unique_id": summary["id_quality"]["train_unique_id"],
                        "test_submission_id_match": summary["id_quality"]["test_submission_id_match"],
                    }
                ).to_frame("value")
                """
            ),
            _new_markdown_cell("## Response Balance"),
            _new_code_cell(
                """
                response_distribution = (
                    bundle.train["Response"]
                    .value_counts(normalize=True)
                    .rename_axis("response")
                    .reset_index(name="share")
                    .assign(share=lambda frame: frame["share"].round(4))
                )
                response_distribution
                """
            ),
            _new_markdown_cell("## Segment-Level Signal"),
            _new_code_cell(
                """
                train = bundle.train
                segment_tables = []
                for column in ["Gender", "Vehicle_Age", "Vehicle_Damage", "Previously_Insured"]:
                    table = (
                        train.groupby(column, observed=True)["Response"]
                        .agg(customers="size", responders="sum", response_rate="mean")
                        .reset_index()
                        .rename(columns={column: "segment_value"})
                    )
                    table.insert(0, "dimension", column)
                    segment_tables.append(table)

                pd.concat(segment_tables, ignore_index=True).sort_values(
                    ["dimension", "response_rate"],
                    ascending=[True, False],
                )
                """
            ),
            _new_markdown_cell("## Reusable EDA Output"),
            _new_code_cell("pd.read_csv(REPORT_DIR / 'eda_highlights.csv').head(12)"),
        ],
        "02_feature_engineering.ipynb": [
            _new_markdown_cell(
                """
                # 02 - Feature Engineering

                This notebook documents the feature layer used by the model: deterministic customer intelligence fields, high-cardinality business-code encoding, and leakage-safe validation design.
                """
            ),
            _new_code_cell(common_setup),
            _new_markdown_cell("## Build Deterministic Customer Features"),
            _new_code_cell(
                """
                from insurance_propensity.config import DATA_RAW_DIR, TARGET_COLUMN
                from insurance_propensity.data.validation import load_raw_datasets
                from insurance_propensity.features.engineering import InsuranceFeatureBuilder, CrossValidatedTargetEncoder

                bundle = load_raw_datasets(DATA_RAW_DIR)
                train = bundle.train
                x = train.drop(columns=[TARGET_COLUMN])
                y = train[TARGET_COLUMN]

                builder = InsuranceFeatureBuilder().fit(x)
                feature_frame = builder.transform(x.head(10_000))
                feature_frame[
                    [
                        "Age",
                        "Annual_Premium",
                        "Vehicle_Age",
                        "Vehicle_Damage",
                        "age_group",
                        "premium_band",
                        "customer_risk_segment",
                        "cross_sell_opportunity_segment",
                    ]
                ].head(10)
                """
            ),
            _new_markdown_cell("## Validate Engineered Segment Coverage"),
            _new_code_cell(
                """
                (
                    feature_frame["cross_sell_opportunity_segment"]
                    .value_counts(normalize=True)
                    .rename_axis("segment")
                    .reset_index(name="share")
                    .assign(share=lambda frame: frame["share"].round(4))
                )
                """
            ),
            _new_markdown_cell("## Out-of-Fold Target Encoding"),
            _new_code_cell(
                """
                encoder = CrossValidatedTargetEncoder(columns=["Region_Code", "Policy_Sales_Channel"])
                encoded_sample = encoder.fit_transform(feature_frame, y.head(10_000))
                encoded_sample[["Region_Code_response_rate_te", "Policy_Sales_Channel_response_rate_te"]].describe()
                """
            ),
            _new_markdown_cell("## Modeling Feature Set"),
            _new_code_cell(
                """
                from insurance_propensity.models.modeling import FEATURE_COLUMNS

                feature_inventory = {
                    "total_features": len(FEATURE_COLUMNS),
                    "target_encoded_features": [column for column in FEATURE_COLUMNS if column.endswith("_response_rate_te")],
                    "business_segments": [column for column in FEATURE_COLUMNS if column.endswith("_segment")],
                }
                feature_inventory
                """
            ),
        ],
        "03_model_training_and_evaluation.ipynb": [
            _new_markdown_cell(
                """
                # 03 - Model Training and Evaluation

                This notebook reviews the reproducible training output and prioritizes ranking quality using PR-AUC, ROC-AUC, top-decile lift, and cumulative capture.
                """
            ),
            _new_code_cell(common_setup),
            _new_markdown_cell("## Candidate Model Comparison"),
            _new_code_cell(
                """
                import pandas as pd

                model_comparison = pd.read_csv(PROJECT_ROOT / "outputs" / "reports" / "model_comparison.csv")
                model_comparison.sort_values("pr_auc", ascending=False)
                """
            ),
            _new_markdown_cell("## Validation Metrics"),
            _new_code_cell(
                """
                import json

                metrics = json.loads((PROJECT_ROOT / "outputs" / "reports" / "metrics.json").read_text(encoding="utf-8"))
                pd.Series(
                    {
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                        "top_decile_lift": metrics["top_decile_lift"],
                        "capture_rate_top_30_pct": metrics["capture_rate_top_30_pct"],
                        "best_threshold": metrics["best_threshold"]["threshold"],
                    }
                ).to_frame("value")
                """
            ),
            _new_markdown_cell("## Decile Targeting Performance"),
            _new_code_cell(
                """
                validation_deciles = pd.read_csv(PROJECT_ROOT / "outputs" / "reports" / "validation_decile_report.csv")
                validation_deciles.head(10)
                """
            ),
            _new_markdown_cell("## Model Diagnostic Chart Inventory"),
            _new_code_cell(
                """
                chart_dir = PROJECT_ROOT / "outputs" / "charts"
                sorted(path.name for path in chart_dir.glob("*.png"))
                """
            ),
        ],
        "04_model_interpretability_and_business_insights.ipynb": [
            _new_markdown_cell(
                """
                # 04 - Interpretability and Business Insights

                This notebook translates model evidence into customer prioritization, campaign planning, and executive-facing recommendations.
                """
            ),
            _new_code_cell(common_setup),
            _new_markdown_cell("## Feature Drivers"),
            _new_code_cell(
                """
                import pandas as pd

                importance = pd.read_csv(PROJECT_ROOT / "outputs" / "reports" / "feature_importance.csv")
                importance.head(15)
                """
            ),
            _new_markdown_cell("## Ranked Customer Scores"),
            _new_code_cell(
                """
                scores = pd.read_csv(PROJECT_ROOT / "outputs" / "predictions" / "test_customer_scores.csv")
                scores[
                    [
                        "id",
                        "propensity_score",
                        "propensity_decile",
                        "age_group",
                        "premium_band",
                        "cross_sell_opportunity_segment",
                        "targeting_recommendation",
                    ]
                ].sort_values("propensity_score", ascending=False).head(10)
                """
            ),
            _new_markdown_cell("## Campaign Priority Mix"),
            _new_code_cell(
                """
                (
                    scores.groupby(["propensity_decile", "targeting_recommendation"], observed=True)
                    .agg(customers=("id", "size"), average_score=("propensity_score", "mean"))
                    .reset_index()
                    .sort_values(["propensity_decile", "customers"])
                )
                """
            ),
            _new_markdown_cell("## Executive Interpretation"),
            _new_code_cell(
                """
                interpretation_report = PROJECT_ROOT / "outputs" / "reports" / "interpretability_report.md"
                print(interpretation_report.read_text(encoding="utf-8")[:2_000])
                """
            ),
        ],
    }

    for filename, cells in notebooks.items():
        notebook = nbf.v4.new_notebook()
        notebook["cells"] = cells
        notebook["metadata"] = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
        nbf.write(notebook, notebooks_dir / filename)