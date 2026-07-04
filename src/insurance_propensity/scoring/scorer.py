"""Reusable scoring bundle for batch and Streamlit workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from insurance_propensity.config import ID_COLUMN, TARGET_COLUMN
from insurance_propensity.features.engineering import CrossValidatedTargetEncoder, InsuranceFeatureBuilder, add_propensity_deciles
from insurance_propensity.models.modeling import FEATURE_COLUMNS


@dataclass
class PropensityScorer:
    """Production-style scorer wrapper around feature engineering, encoding, and model pipeline."""

    feature_builder: InsuranceFeatureBuilder
    target_encoder: CrossValidatedTargetEncoder
    model_pipeline: object
    model_name: str
    validation_metrics: dict

    def prepare_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        features = self.feature_builder.transform(raw_df)
        features = self.target_encoder.transform(features)
        return features

    def predict_proba(self, raw_df: pd.DataFrame) -> np.ndarray:
        features = self.prepare_features(raw_df)
        return self.model_pipeline.predict_proba(features[FEATURE_COLUMNS])[:, 1]

    def score(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        features = self.prepare_features(raw_df)
        output = raw_df.copy()
        output["propensity_score"] = self.model_pipeline.predict_proba(features[FEATURE_COLUMNS])[:, 1]
        intelligence_columns = [
            "age_group",
            "premium_band",
            "vintage_band",
            "customer_risk_segment",
            "cross_sell_opportunity_segment",
        ]
        for column in intelligence_columns:
            output[column] = features[column].values
        output = add_propensity_deciles(output, "propensity_score")
        return output


def build_submission(scored_test: pd.DataFrame) -> pd.DataFrame:
    """Return Kaggle-style sample submission format using probability scores."""
    return scored_test[[ID_COLUMN, "propensity_score"]].rename(columns={"propensity_score": TARGET_COLUMN})
