# Data Quality Report

## Raw Data Contract
- Train shape: `381,109` rows x `12` columns
- Test shape: `127,037` rows x `11` columns
- Sample submission shape: `127,037` rows x `2` columns
- Total missing values: train `0`, test `0`
- Duplicate rows: train `0`, test `0`
- Unique train ID: `True`
- Unique test ID: `True`

## Target Balance
- Positive responders: `46,710`
- Non-responders: `334,399`
- Positive response rate: `12.26%`

## Annual Premium Outlier Review
- IQR upper fence: `61,892.50`
- Outlier count: `10,320`
- Outlier rate: `2.71%`

## Leakage Audit
The project excludes `id` and `Response` from predictive features, never trains on `test.csv`, and uses out-of-fold target encoding for regional and sales channel response-rate encodings.
