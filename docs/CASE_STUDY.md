# Case Study: Insurance Customer Propensity Prediction

## Context

The business wants to prioritize existing health insurance customers who are most likely to consider vehicle insurance. The practical objective is not just classification, but customer ranking for targeted cross-sell outreach.

## Business Question

Which customers should be prioritized first when campaign capacity is limited?

## Solution Overview

This project builds a full customer propensity intelligence workflow:

- Raw data contract validation
- Data quality and leakage audit
- Feature engineering with business-readable segments
- Out-of-fold response-rate encoding
- Multiple model comparison
- Validation decile lift analysis
- Batch scoring for `test.csv`
- Kaggle-style submission output
- Streamlit dashboard for scoring and business review
- SQL scripts for customer intelligence reporting

## Key Results

| Metric | Result |
|---|---:|
| Final model | Hist Gradient Boosting |
| ROC-AUC | 0.8585 |
| PR-AUC | 0.3685 |
| Top decile lift | 3.22x |
| Top 10% capture | 32.25% |
| Top 20% capture | 58.26% |
| Top 30% capture | 79.40% |

## Business Interpretation

The top three propensity deciles concentrate most validation responders. A campaign team can start with decile 1 for the strongest outreach priority, then expand to deciles 1-3 when capacity allows broader reach.

The strongest response signals are related to prior vehicle damage, previous insurance status, customer segment, sales channel response patterns, and age. These are interpreted as targeting associations, not causal proof.

## Portfolio Value

This project demonstrates the kind of end-to-end thinking expected from a Data Scientist:

- Business framing
- Data quality discipline
- Leakage prevention
- Imbalanced classification
- Ranking-based evaluation
- Interpretability
- Dashboard delivery
- Reproducible project structure

## Limitations

The dataset is cross-sectional. It does not support claims about customer lifetime value, churn, real campaign ROI, or time-series behavior. The dashboard is a portfolio demo and batch scoring interface, not a real-time production system.
