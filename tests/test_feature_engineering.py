from insurance_propensity.config import DATA_RAW_DIR, TARGET_COLUMN
from insurance_propensity.data.validation import load_raw_datasets
from insurance_propensity.features.engineering import CrossValidatedTargetEncoder, InsuranceFeatureBuilder


def test_feature_engineering_no_missing_values_on_sample():
    bundle = load_raw_datasets(DATA_RAW_DIR)
    sample = bundle.train.head(5000)
    x = sample.drop(columns=[TARGET_COLUMN])
    y = sample[TARGET_COLUMN]

    builder = InsuranceFeatureBuilder().fit(x)
    features = builder.transform(x)
    encoder = CrossValidatedTargetEncoder(columns=["Region_Code", "Policy_Sales_Channel"], n_splits=3)
    encoded = encoder.fit_transform(features, y)

    assert encoded.isna().sum().sum() == 0
    assert "Region_Code_response_rate_te" in encoded.columns
    assert "Policy_Sales_Channel_response_rate_te" in encoded.columns
    assert TARGET_COLUMN not in encoded.columns
