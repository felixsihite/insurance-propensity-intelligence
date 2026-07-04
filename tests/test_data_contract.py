from pathlib import Path

from insurance_propensity.config import DATA_RAW_DIR, TEST_COLUMNS, TRAIN_COLUMNS
from insurance_propensity.data.validation import load_raw_datasets


def test_raw_dataset_contract():
    bundle = load_raw_datasets(DATA_RAW_DIR)

    assert list(bundle.train.columns) == TRAIN_COLUMNS
    assert list(bundle.test.columns) == TEST_COLUMNS
    assert bundle.train.isna().sum().sum() == 0
    assert bundle.test.isna().sum().sum() == 0
    assert bundle.train.duplicated().sum() == 0
    assert bundle.test.duplicated().sum() == 0
    assert bundle.train["id"].is_unique
    assert bundle.test["id"].is_unique
    assert bundle.sample_submission["id"].equals(bundle.test["id"])


def test_raw_files_are_present():
    for filename in ["train.csv", "test.csv", "sample_submission.csv"]:
        assert (Path(DATA_RAW_DIR) / filename).exists()