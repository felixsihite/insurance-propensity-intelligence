# Model Interpretability Report

Interpretability method used in this environment: **permutation importance**.

The codebase is SHAP-ready and will use SHAP when the dependency is installed. The generated project artifact below uses permutation importance because SHAP is not available in the current Python environment.

## Global Drivers
- `Previously_Insured`: 0.05324
- `Policy_Sales_Channel_response_rate_te`: 0.04832
- `customer_risk_segment`: 0.04041
- `vehicle_damage_flag`: 0.03913
- `Age`: 0.03274
- `cross_sell_opportunity_segment`: 0.02886
- `vehicle_age_encoded`: 0.01737
- `Vehicle_Age`: 0.01298
- `Policy_Sales_Channel`: 0.00877
- `Region_Code_response_rate_te`: 0.00404

## Business Interpretation
- Previous vehicle damage and not-yet-insured status are expected to be strong positive signals for cross-sell relevance.
- Vehicle age, customer age band, annual premium band, region, and sales channel help prioritize customer groups, but they should be treated as associative targeting signals rather than causal proof.
- The model is designed for customer ranking. Business users should review decile lift, cumulative gain, and capture rate before selecting campaign reach.

## Local Explanation Workflow
For an individual customer, the Streamlit dashboard surfaces the final propensity score, decile, segment, and recommendation. When SHAP is installed, the notebook can be extended with local force/waterfall explanations using the saved scorer bundle.
