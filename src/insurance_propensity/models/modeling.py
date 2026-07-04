"""Model pipelines and training helpers."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from insurance_propensity.config import ID_COLUMN, TARGET_COLUMN


NUMERIC_FEATURES = [
    "Age",
    "Driving_License",
    "Previously_Insured",
    "Annual_Premium",
    "Vintage",
    "vehicle_age_encoded",
    "vehicle_damage_flag",
    "not_previously_insured_flag",
    "high_premium_flag",
    "Region_Code_response_rate_te",
    "Policy_Sales_Channel_response_rate_te",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Region_Code",
    "Policy_Sales_Channel",
    "Vehicle_Age",
    "Vehicle_Damage",
    "age_group",
    "premium_band",
    "vintage_band",
    "customer_risk_segment",
    "cross_sell_opportunity_segment",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class TrainedModelCandidate:
    """Container for a trained candidate model and its validation metrics."""

    name: str
    pipeline: Pipeline
    metrics: dict[str, Any]


def _optional_estimator(module_name: str, class_name: str) -> type[Any] | None:
    """Load optional model classes only when their package is installed."""
    if importlib.util.find_spec(module_name) is None:
        return None

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None

    estimator = getattr(module, class_name, None)
    return estimator if isinstance(estimator, type) else None


def build_preprocessor() -> ColumnTransformer:
    """Build a compact preprocessing stack that works across linear and tree models."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def available_model_specs(random_state: int = 42) -> list[tuple[str, Any, bool]]:
    """Return model candidates. The boolean indicates whether sample weights should be passed."""
    specs: list[tuple[str, Any, bool]] = [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=800, class_weight="balanced", random_state=random_state),
            False,
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=140,
                max_depth=12,
                min_samples_leaf=60,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            False,
        ),
        (
            "Hist Gradient Boosting",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=220,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                class_weight="balanced",
                random_state=random_state,
            ),
            False,
        ),
    ]

    xgb_classifier = _optional_estimator("xgboost", "XGBClassifier")
    if xgb_classifier is not None:
        specs.append(
            (
                "XGBoost",
                xgb_classifier(
                    n_estimators=450,
                    learning_rate=0.035,
                    max_depth=5,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                True,
            )
        )

    lgbm_classifier = _optional_estimator("lightgbm", "LGBMClassifier")
    if lgbm_classifier is not None:
        specs.append(
            (
                "LightGBM",
                lgbm_classifier(
                    n_estimators=500,
                    learning_rate=0.035,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                False,
            )
        )

    catboost_classifier = _optional_estimator("catboost", "CatBoostClassifier")
    if catboost_classifier is not None:
        specs.append(
            (
                "CatBoost",
                catboost_classifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.04,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    verbose=False,
                    random_seed=random_state,
                    auto_class_weights="Balanced",
                ),
                False,
            )
        )

    return specs


def train_candidate_model(name: str, estimator: Any, x_train: pd.DataFrame, y_train: pd.Series, use_sample_weight: bool) -> Pipeline:
    """Fit a model candidate inside a preprocessing pipeline."""
    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])
    fit_kwargs = {}
    if use_sample_weight:
        fit_kwargs["model__sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(x_train[FEATURE_COLUMNS], y_train, **fit_kwargs)
    return pipeline


def predict_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for a fitted model."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x[FEATURE_COLUMNS])
        return np.asarray(probabilities)[:, 1]

    decision = getattr(model, "decision_function")(x[FEATURE_COLUMNS])
    return 1 / (1 + np.exp(-decision))


def modeling_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the exact feature columns expected by the model."""
    blocked = {ID_COLUMN, TARGET_COLUMN}
    return [column for column in FEATURE_COLUMNS if column in df.columns and column not in blocked]