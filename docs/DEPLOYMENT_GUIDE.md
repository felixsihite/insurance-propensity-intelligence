# Deployment Guide

## Local Dashboard

Local Python used during verification:

```text
Python 3.13.1
```

Run from the project root:

```bash
python -m streamlit run streamlit_app/app.py
```

The app opens at:

```text
http://localhost:8501
```

## Direct Portfolio QA Links

```text
http://localhost:8501/?page=executive-summary&theme=light
http://localhost:8501/?page=model-performance&theme=dark
http://localhost:8501/?page=interpretability&theme=light
```

## Regenerate Project Artifacts

```bash
python scripts/run_project_pipeline.py
python scripts/generate_portfolio_assets.py
```

## Verify

```bash
pytest
python -m compileall -q src scripts streamlit_app
```

## Expected Outputs

- `models/final_propensity_model.joblib`
- `outputs/submission/submission.csv`
- `outputs/predictions/test_customer_scores.csv`
- `outputs/reports/model_comparison.csv`
- `outputs/dashboard_screenshots/`
- `outputs/portfolio_assets/`

## Notes

The dashboard is a portfolio-grade batch scoring and analytics interface. It is not presented as a fully managed production service.

## Streamlit Community Cloud

Use these settings when deploying from GitHub:

| Field | Value |
|---|---|
| Repository | `felixsihite/insurance-propensity-intelligence` |
| Branch | `main` |
| Main file path | `streamlit_app/app.py` |
| App URL | `insurance-propensity-intelligence` |
| Python version | `3.12` recommended |

Live app URL after deployment:

```text
https://insurance-propensity-intelligence.streamlit.app
```

If the subdomain is unavailable, choose a similar subdomain and update the README badge/link.
