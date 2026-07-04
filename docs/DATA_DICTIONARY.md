# Data Dictionary

## Raw Fields

| Column | Type | Description |
|---|---|---|
| `id` | Identifier | Customer row identifier. Not used as a predictive feature. |
| `Gender` | Categorical | Customer gender value in the source data. |
| `Age` | Numeric | Customer age. |
| `Driving_License` | Binary | Whether the customer has a driving license. |
| `Region_Code` | Categorical code | Encoded customer region. |
| `Previously_Insured` | Binary | Whether the customer was previously insured for vehicle insurance. |
| `Vehicle_Age` | Categorical | Vehicle age group. |
| `Vehicle_Damage` | Categorical | Whether the customer had vehicle damage history. |
| `Annual_Premium` | Numeric | Annual premium amount. |
| `Policy_Sales_Channel` | Categorical code | Encoded policy sales channel. |
| `Vintage` | Numeric | Customer tenure in days. |
| `Response` | Binary target | Whether the customer responded positively. Present only in train data. |

## Engineered Fields

| Feature | Description |
|---|---|
| `age_group` | Customer age bucket for business segmentation. |
| `premium_band` | Annual premium bucket. |
| `vintage_band` | Customer tenure bucket. |
| `vehicle_age_encoded` | Ordered numeric representation of vehicle age. |
| `vehicle_damage_flag` | Binary damage-history flag. |
| `not_previously_insured_flag` | Binary inverse of previous insurance status. |
| `high_premium_flag` | Flag based on training-set premium quantile. |
| `customer_risk_segment` | Business-readable customer risk/opportunity segment. |
| `cross_sell_opportunity_segment` | Campaign prioritization segment. |
| `Region_Code_response_rate_te` | Out-of-fold region response-rate encoding. |
| `Policy_Sales_Channel_response_rate_te` | Out-of-fold sales-channel response-rate encoding. |
| `propensity_score` | Predicted response probability. |
| `propensity_decile` | Ranked decile where 1 is highest propensity. |
| `targeting_recommendation` | Campaign action label derived from propensity decile. |

## Data Quality Notes

- Train rows: 381,109
- Test rows: 127,037
- Missing values: 0
- Duplicate rows: 0
- Positive response rate: 12.26%