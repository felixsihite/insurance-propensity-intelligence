"""Run the complete Insurance Customer Propensity portfolio pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from insurance_propensity.config import (  # noqa: E402
    CHART_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    ID_COLUMN,
    MODEL_DIR,
    PREDICTION_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    SUBMISSION_DIR,
    TARGET_COLUMN,
    ensure_project_directories,
)
from insurance_propensity.data.validation import dataset_quality_summary, load_raw_datasets  # noqa: E402
from insurance_propensity.evaluation.metrics import binary_classification_metrics, decile_table, threshold_sweep  # noqa: E402
from insurance_propensity.features.engineering import CrossValidatedTargetEncoder, InsuranceFeatureBuilder, add_propensity_deciles  # noqa: E402
from insurance_propensity.models.modeling import FEATURE_COLUMNS, available_model_specs, train_candidate_model  # noqa: E402
from insurance_propensity.reporting import generate_notebooks, write_business_reports, write_data_quality_report, write_json  # noqa: E402
from insurance_propensity.scoring.scorer import PropensityScorer, build_submission  # noqa: E402
from insurance_propensity.visualization.plots import (  # noqa: E402
    calculate_permutation_importance,
    save_confusion_matrix,
    save_feature_importance,
    save_gain_chart,
    save_lift_chart,
    save_premium_distribution,
    save_response_distribution,
    save_roc_pr_calibration,
    save_segment_response,
)

def prepare_model_frames(train: pd.DataFrame, test: pd.DataFrame):
    """Fit final feature builder and target encoder on train only, then transform train/test."""
    x_all = train.drop(columns=[TARGET_COLUMN])
    y_all = train[TARGET_COLUMN]
    feature_builder = InsuranceFeatureBuilder().fit(x_all)
    train_base = feature_builder.transform(x_all)
    test_base = feature_builder.transform(test)
    target_encoder = CrossValidatedTargetEncoder(columns=["Region_Code", "Policy_Sales_Channel"], random_state=RANDOM_STATE)
    train_model = target_encoder.fit_transform(train_base, y_all)
    test_model = target_encoder.transform(test_base)
    return feature_builder, target_encoder, train_model, test_model


def prepare_validation_split(train: pd.DataFrame):
    """Create leakage-safe train/validation matrices."""
    x_raw = train.drop(columns=[TARGET_COLUMN])
    y = train[TARGET_COLUMN]
    x_train_raw, x_valid_raw, y_train, y_valid = train_test_split(
        x_raw,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    feature_builder = InsuranceFeatureBuilder().fit(x_train_raw)
    x_train_base = feature_builder.transform(x_train_raw)
    x_valid_base = feature_builder.transform(x_valid_raw)
    target_encoder = CrossValidatedTargetEncoder(columns=["Region_Code", "Policy_Sales_Channel"], random_state=RANDOM_STATE)
    x_train_model = target_encoder.fit_transform(x_train_base, y_train)
    x_valid_model = target_encoder.transform(x_valid_base)
    return x_train_raw, x_valid_raw, y_train, y_valid, x_train_model, x_valid_model


def evaluate_models(x_train_model: pd.DataFrame, y_train: pd.Series, x_valid_model: pd.DataFrame, y_valid: pd.Series) -> tuple[list[dict], object, str, dict]:
    """Train candidates and choose the best ranking model by PR-AUC, then ROC-AUC."""
    model_rows: list[dict] = []
    candidates = []

    for model_name, estimator, use_sample_weight in available_model_specs(RANDOM_STATE):
        start = time.time()
        model = train_candidate_model(model_name, estimator, x_train_model, y_train, use_sample_weight)
        y_prob = model.predict_proba(x_valid_model[FEATURE_COLUMNS])[:, 1]
        metrics = binary_classification_metrics(y_valid, y_prob)
        metrics["training_seconds"] = round(time.time() - start, 2)
        metrics["model_name"] = model_name
        candidates.append((model_name, model, metrics, y_prob))
        model_rows.append(
            {
                "model_name": model_name,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "top_decile_lift": metrics["top_decile_lift"],
                "capture_rate_top_10_pct": metrics["capture_rate_top_10_pct"],
                "capture_rate_top_20_pct": metrics["capture_rate_top_20_pct"],
                "capture_rate_top_30_pct": metrics["capture_rate_top_30_pct"],
                "training_seconds": metrics["training_seconds"],
            }
        )
        print(f"Finished {model_name}: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

    best = sorted(candidates, key=lambda row: (row[2]["pr_auc"], row[2]["roc_auc"], row[2]["top_decile_lift"]), reverse=True)[0]
    best_name, best_model, best_metrics, best_y_prob = best
    return model_rows, best_model, best_name, best_metrics | {"validation_probabilities": best_y_prob}


def main() -> None:
    ensure_project_directories()
    print("Loading raw datasets...")
    bundle = load_raw_datasets(DATA_RAW_DIR)
    train, test, sample_submission = bundle.train, bundle.test, bundle.sample_submission

    print("Writing data quality reports...")
    quality = dataset_quality_summary(bundle)
    write_json(REPORT_DIR / "data_quality_report.json", quality)
    write_data_quality_report(quality)

    print("Preparing validation split...")
    x_train_raw, x_valid_raw, y_train, y_valid, x_train_model, x_valid_model = prepare_validation_split(train)

    print("Training and evaluating model candidates...")
    model_rows, best_validation_model, best_name, best_metrics = evaluate_models(x_train_model, y_train, x_valid_model, y_valid)
    model_comparison = pd.DataFrame(model_rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    model_comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    y_valid_prob = best_metrics.pop("validation_probabilities")
    validation_deciles = decile_table(y_valid, y_valid_prob)
    validation_deciles.to_csv(REPORT_DIR / "validation_decile_report.csv", index=False)
    threshold_sweep(y_valid, y_valid_prob).to_csv(REPORT_DIR / "threshold_optimization.csv", index=False)
    write_json(REPORT_DIR / "metrics.json", best_metrics)

    print("Creating charts and interpretation outputs...")
    train_plot_builder = InsuranceFeatureBuilder().fit(train.drop(columns=[TARGET_COLUMN]))
    train_plot_frame = train_plot_builder.transform(train.drop(columns=[TARGET_COLUMN]))
    train_plot_frame[TARGET_COLUMN] = train[TARGET_COLUMN].values
    save_response_distribution(train, CHART_DIR / "response_distribution.png")
    save_segment_response(train_plot_frame, CHART_DIR / "segment_response_rate.png")
    save_premium_distribution(train, CHART_DIR / "annual_premium_distribution.png")
    save_lift_chart(validation_deciles, CHART_DIR / "lift_chart.png")
    save_gain_chart(validation_deciles, CHART_DIR / "gain_chart.png")
    save_roc_pr_calibration(y_valid, y_valid_prob, CHART_DIR)
    best_threshold = float(best_metrics["best_threshold"]["threshold"])
    save_confusion_matrix(y_valid, y_valid_prob, best_threshold, CHART_DIR / "confusion_matrix.png")
    importance = calculate_permutation_importance(best_validation_model, x_valid_model, y_valid, REPORT_DIR / "feature_importance.csv")
    save_feature_importance(importance, CHART_DIR / "feature_importance.png")
    write_business_reports(best_name, best_metrics, validation_deciles, importance)

    print("Writing EDA highlight tables...")
    eda_tables = []
    for column in ["Gender", "Vehicle_Age", "Vehicle_Damage", "Previously_Insured", "age_group", "premium_band", "cross_sell_opportunity_segment"]:
        source = train_plot_frame if column in train_plot_frame.columns else train
        table = source.groupby(column, observed=True)[TARGET_COLUMN].agg(customers="size", responders="sum", response_rate="mean").reset_index()
        table.insert(0, "dimension", column)
        table = table.rename(columns={column: "segment_value"})
        eda_tables.append(table)
    eda_highlights = pd.concat(eda_tables, ignore_index=True).sort_values(["dimension", "response_rate"], ascending=[True, False])
    eda_highlights.to_csv(REPORT_DIR / "eda_highlights.csv", index=False)

    print("Training final scorer on full train.csv...")
    final_builder, final_encoder, train_processed, test_processed = prepare_model_frames(train, test)
    best_spec = next(spec for spec in available_model_specs(RANDOM_STATE) if spec[0] == best_name)
    final_model = train_candidate_model(best_spec[0], clone(best_spec[1]), train_processed, train[TARGET_COLUMN], best_spec[2])
    scorer = PropensityScorer(
        feature_builder=final_builder,
        target_encoder=final_encoder,
        model_pipeline=final_model,
        model_name=best_name,
        validation_metrics=best_metrics,
    )
    joblib.dump(scorer, MODEL_DIR / "final_propensity_model.joblib")

    train_processed_with_target = train_processed.copy()
    train_processed_with_target[TARGET_COLUMN] = train[TARGET_COLUMN].values
    if train_processed_with_target.isna().sum().sum() or test_processed.isna().sum().sum():
        raise ValueError("Processed datasets contain missing values; refusing to export.")
    train_processed_with_target.to_csv(DATA_PROCESSED_DIR / "train_processed.csv", index=False)
    test_processed.to_csv(DATA_PROCESSED_DIR / "test_processed.csv", index=False)

    print("Scoring validation and test customers...")
    validation_scored = x_valid_raw.copy()
    validation_scored[TARGET_COLUMN] = y_valid.values
    validation_scored["propensity_score"] = y_valid_prob
    validation_scored = add_propensity_deciles(validation_scored, "propensity_score")
    validation_scored.to_csv(PREDICTION_DIR / "validation_customer_scores.csv", index=False)

    scored_test = scorer.score(test)
    scored_test.to_csv(PREDICTION_DIR / "test_customer_scores.csv", index=False)
    submission = build_submission(scored_test)
    if not submission[ID_COLUMN].equals(sample_submission[ID_COLUMN]):
        raise ValueError("Submission IDs do not match sample_submission.csv.")
    submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

    print("Generating notebooks...")
    generate_notebooks(PROJECT_ROOT)

    manifest = {
        "final_model": best_name,
        "project_root": str(PROJECT_ROOT),
        "data_source": "https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction",
        "outputs": {
            "model": str(MODEL_DIR / "final_propensity_model.joblib"),
            "submission": str(SUBMISSION_DIR / "submission.csv"),
            "test_scores": str(PREDICTION_DIR / "test_customer_scores.csv"),
            "model_comparison": str(REPORT_DIR / "model_comparison.csv"),
        },
    }
    write_json(REPORT_DIR / "run_manifest.json", manifest)
    print(f"Pipeline complete. Final model: {best_name}")


if __name__ == "__main__":
    main()
