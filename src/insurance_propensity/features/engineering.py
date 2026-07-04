"""Leakage-aware feature engineering for customer propensity modeling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass
class InsuranceFeatureBuilder:
    """Build deterministic customer intelligence features from raw customer rows."""

    premium_high_quantile: float = 0.75
    premium_high_threshold_: float | None = None

    def fit(self, df: pd.DataFrame) -> "InsuranceFeatureBuilder":
        self.premium_high_threshold_ = float(df["Annual_Premium"].quantile(self.premium_high_quantile))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.premium_high_threshold_ is None:
            raise ValueError("InsuranceFeatureBuilder must be fitted before transform.")

        output = df.copy()
        output["Region_Code"] = output["Region_Code"].astype(float).round(0).astype(int).astype(str)
        output["Policy_Sales_Channel"] = output["Policy_Sales_Channel"].astype(float).round(0).astype(int).astype(str)

        output["age_group"] = pd.cut(
            output["Age"],
            bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
            right=True,
        ).astype(str)
        output["premium_band"] = pd.cut(
            output["Annual_Premium"],
            bins=[-np.inf, 20_000, 35_000, 50_000, 75_000, np.inf],
            labels=["Low", "Core", "Upper Core", "High", "Premium"],
            right=True,
        ).astype(str)
        output["vintage_band"] = pd.cut(
            output["Vintage"],
            bins=[-np.inf, 60, 120, 180, 240, np.inf],
            labels=["0-60d", "61-120d", "121-180d", "181-240d", "241d+"],
            right=True,
        ).astype(str)

        output["vehicle_age_encoded"] = output["Vehicle_Age"].map({"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}).fillna(-1).astype(int)
        output["vehicle_damage_flag"] = output["Vehicle_Damage"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        output["not_previously_insured_flag"] = (1 - output["Previously_Insured"].astype(int)).astype(int)
        output["high_premium_flag"] = (output["Annual_Premium"] >= self.premium_high_threshold_).astype(int)

        output["customer_risk_segment"] = np.select(
            [
                (output["not_previously_insured_flag"].eq(1)) & (output["vehicle_damage_flag"].eq(1)),
                (output["not_previously_insured_flag"].eq(1)) & (output["vehicle_age_encoded"].ge(1)),
                output["not_previously_insured_flag"].eq(1),
                (output["Previously_Insured"].eq(1)) & (output["vehicle_damage_flag"].eq(0)),
            ],
            [
                "Protection Gap with Damage History",
                "Uninsured Multi-Year Vehicle",
                "Uninsured Prospect",
                "Existing Vehicle Cover Signal",
            ],
            default="Standard Relationship",
        )

        output["cross_sell_opportunity_segment"] = np.select(
            [
                (output["not_previously_insured_flag"].eq(1))
                & (output["vehicle_damage_flag"].eq(1))
                & (output["Age"].between(30, 60)),
                (output["not_previously_insured_flag"].eq(1)) & (output["vehicle_damage_flag"].eq(1)),
                output["not_previously_insured_flag"].eq(1),
                output["Vehicle_Damage"].eq("Yes"),
            ],
            ["Priority", "High", "Nurture", "Monitor"],
            default="Low",
        )
        return output

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class CrossValidatedTargetEncoder:
    """Out-of-fold target encoder for high-cardinality business codes."""

    columns: list[str]
    n_splits: int = 5
    smoothing: float = 20.0
    random_state: int = 42
    global_mean_: float | None = None
    mappings_: dict[str, dict[str, float]] = field(default_factory=dict)

    def _fit_mapping(self, x: pd.Series, y: pd.Series) -> dict[str, float]:
        stats = pd.DataFrame({"category": x.astype(str), "target": y}).groupby("category")["target"].agg(["mean", "count"])
        assert self.global_mean_ is not None
        smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (stats["count"] + self.smoothing)
        return {str(category): float(value) for category, value in smoothed.items()}

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        output = df.copy()
        self.global_mean_ = float(pd.Series(y).mean())
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for column in self.columns:
            encoded = pd.Series(index=df.index, dtype=float)
            for train_idx, holdout_idx in cv.split(df, y):
                fold_mapping = self._fit_mapping(df.iloc[train_idx][column], pd.Series(y).iloc[train_idx])
                encoded.iloc[holdout_idx] = df.iloc[holdout_idx][column].astype(str).map(fold_mapping).fillna(self.global_mean_).to_numpy()
            output[f"{column}_response_rate_te"] = encoded.fillna(self.global_mean_).astype(float)
            self.mappings_[column] = self._fit_mapping(df[column], pd.Series(y))
        return output

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None or not self.mappings_:
            raise ValueError("CrossValidatedTargetEncoder must be fitted before transform.")

        output = df.copy()
        for column in self.columns:
            mapping = self.mappings_.get(column, {})
            output[f"{column}_response_rate_te"] = output[column].astype(str).map(mapping).fillna(self.global_mean_).astype(float)
        return output


def add_propensity_deciles(df: pd.DataFrame, score_col: str = "propensity_score") -> pd.DataFrame:
    """Add deciles where decile 1 is the highest-propensity customer group."""
    output = df.copy()
    ranks = output[score_col].rank(method="first", ascending=False)
    output["propensity_decile"] = pd.qcut(ranks, 10, labels=list(range(1, 11))).astype(int)
    output["targeting_recommendation"] = np.select(
        [
            output["propensity_decile"].le(1),
            output["propensity_decile"].le(3),
            output["propensity_decile"].le(5),
        ],
        ["Highest priority campaign", "Scaled priority outreach", "Nurture / test campaign"],
        default="Low priority holdout",
    )
    return output