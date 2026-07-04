"""Generate clean portfolio notebooks from the reproducible project pipeline."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def _new_code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


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
            nbf.v4.new_markdown_cell(
                "# 01 - Data Quality and EDA\n\nProfessional raw-data audit, response balance review, and customer intelligence exploration."
            ),
            _new_code_cell(common_setup),
            _new_code_cell(
                """
                import pandas as pd

                from insurance_propensity.config import DATA_RAW_DIR, REPORT_DIR
                from insurance_propensity.data.validation import load_raw_datasets, dataset_quality_summary

                bundle = load_raw_datasets(DATA_RAW_DIR)
                summary = dataset_quality_summary(bundle)
                summary["train_shape"], summary["target_distribution"]
                """
            ),
            _new_code_cell(
                """
                train = bundle.train
                response_by_gender = train.groupby("Gender")["Response"].agg(["count", "mean"]).sort_values("mean", ascending=False)
                response_by_vehicle_damage = train.groupby("Vehicle_Damage")["Response"].agg(["count", "mean"]).sort_values("mean", ascending=False)
                response_by_vehicle_age = train.groupby("Vehicle_Age")["Response"].agg(["count", "mean"]).sort_values("mean", ascending=False)

                display(response_by_gender)
                display(response_by_vehicle_damage)
                display(response_by_vehicle_age)
                """
            ),
            _new_code_cell("pd.read_csv(REPORT_DIR / 'eda_highlights.csv').head(12)"),
        ],
        "02_feature_engineering.ipynb": [
            nbf.v4.new_markdown_cell(
                "# 02 - Feature Engineering\n\nLeakage-safe customer features and out-of-fold target encoding."
            ),
            _new_code_cell(common_setup),
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
                feature_frame.head()
                """
            ),
            _new_code_cell(
                """
                encoder = CrossValidatedTargetEncoder(columns=["Region_Code", "Policy_Sales_Channel"])
                encoded_sample = encoder.fit_transform(feature_frame, y.head(10_000))
                encoded_sample[["Region_Code_response_rate_te", "Policy_Sales_Channel_response_rate_te"]].describe()
                """
            ),
        ],
        "03_model_training_and_evaluation.ipynb": [
            nbf.v4.new_markdown_cell(
                "# 03 - Model Training and Evaluation\n\nModel comparison with ROC-AUC, PR-AUC, lift, gain, and capture-rate metrics."
            ),
            _new_code_cell(common_setup),
            _new_code_cell(
                """
                import pandas as pd

                model_comparison = pd.read_csv(PROJECT_ROOT / "outputs" / "reports" / "model_comparison.csv")
                model_comparison.sort_values("pr_auc", ascending=False)
                """
            ),
            _new_code_cell(
                """
                validation_deciles = pd.read_csv(PROJECT_ROOT / "outputs" / "reports" / "validation_decile_report.csv")
                validation_deciles.head(10)
                """
            ),
        ],
        "04_model_interpretability_and_business_insights.ipynb": [
            nbf.v4.new_markdown_cell(
                "# 04 - Interpretability and Business Insights\n\nFeature drivers, campaign targeting simulation, and executive recommendations."
            ),
            _new_code_cell(common_setup),
            _new_code_cell(
                """
                import pandas as pd

                importance = pd.read_csv(PROJECT_ROOT / "outputs" / "reports" / "feature_importance.csv")
                importance.head(15)
                """
            ),
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
