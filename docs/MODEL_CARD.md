# Model Card

## Model Details

| Item | Description |
|---|---|
| Project | Insurance Customer Propensity Prediction & Customer Intelligence |
| Final model | Hist Gradient Boosting |
| Model family | Gradient boosting classifier |
| Target | `Response` |
| Positive class | Customer expressed interest in vehicle insurance |
| Primary use | Propensity ranking and campaign prioritization |

## Intended Use

The model ranks existing health insurance customers by predicted likelihood of responding to vehicle insurance cross-sell outreach. The recommended operational use is decile-based targeting, not a single hard classification threshold.

## Not Intended For

- Credit, lending, legal, or medical eligibility decisions
- Real-time production claims without further engineering
- Causal inference
- Customer lifetime value estimation
- Churn prediction
- Guaranteed ROI estimation

## Training and Validation

The raw training file is split with stratification. Test data is never used for model training. Target encodings are fitted with out-of-fold training folds and train-only mappings.

| Validation Metric | Result |
|---|---:|
| ROC-AUC | 0.8585 |
| PR-AUC | 0.3685 |
| Precision @ 0.50 | 0.2843 |
| Recall @ 0.50 | 0.9321 |
| Balanced accuracy @ 0.50 | 0.8022 |
| Top decile lift | 3.22x |
| Top 30% capture | 79.40% |

## Important Features

Top validation permutation-importance drivers:

- `Previously_Insured`
- `Policy_Sales_Channel_response_rate_te`
- `customer_risk_segment`
- `vehicle_damage_flag`
- `Age`
- `cross_sell_opportunity_segment`

## Monitoring Considerations

If this model were moved beyond portfolio/demo use, recommended monitoring would include:

- Score distribution drift
- Feature distribution drift
- Response rate by decile
- Calibration drift
- Segment-level performance
- Data schema validation

## Ethical and Business Notes

The model should be used to prioritize outreach, not to exclude customers from access to products. Segment insights should be reviewed by business stakeholders and treated as decision support.