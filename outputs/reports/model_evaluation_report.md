# Model Evaluation Report

## Final Validation Model
Selected model: **Hist Gradient Boosting**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8585 |
| PR-AUC | 0.3685 |
| Precision @ 0.50 | 0.2843 |
| Recall @ 0.50 | 0.9321 |
| F1 @ 0.50 | 0.4357 |
| Balanced Accuracy @ 0.50 | 0.8022 |
| Top Decile Lift | 3.22x |
| Capture Rate Top 10% | 32.25% |
| Capture Rate Top 20% | 58.26% |
| Capture Rate Top 30% | 79.40% |

## Threshold Guidance
The F1-oriented validation threshold is **0.65**. For campaign operations, ranked targeting is more useful than a single classification threshold because the business action is prioritizing outreach capacity.

## Confusion Matrix @ 0.50
`[[44955, 21925], [634, 8708]]`

## Campaign Targeting Readout
Top deciles concentrate the highest observed response share. Recommended launch strategy:

1. Start with decile 1 for highest-intent outreach.
2. Expand to deciles 1-3 when the campaign team has broader capacity.
3. Keep lower deciles as holdout, nurture, or low-frequency education audiences.

This is a propensity-ranking simulation based on validation data, not a claim of guaranteed campaign ROI.
