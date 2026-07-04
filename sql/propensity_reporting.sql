-- Propensity reporting after model scoring.
-- Assumed source table: scored_customers
-- Required fields: id, propensity_score, propensity_decile, targeting_recommendation,
-- cross_sell_opportunity_segment, customer_risk_segment, Gender, Age, Vehicle_Age,
-- Vehicle_Damage, Region_Code, Policy_Sales_Channel, Annual_Premium, Vintage

WITH ranked AS (
    SELECT
        *,
        CASE
            WHEN propensity_decile = 1 THEN 'Top 10%'
            WHEN propensity_decile <= 3 THEN 'Top 30%'
            WHEN propensity_decile <= 5 THEN 'Middle'
            ELSE 'Lower'
        END AS campaign_band
    FROM scored_customers
)

SELECT
    propensity_decile,
    campaign_band,
    targeting_recommendation,
    cross_sell_opportunity_segment,
    COUNT(*) AS customers,
    AVG(propensity_score) AS avg_propensity_score,
    MIN(propensity_score) AS min_propensity_score,
    MAX(propensity_score) AS max_propensity_score,
    AVG(Age) AS avg_age,
    AVG(Annual_Premium) AS avg_annual_premium,
    AVG(Vintage) AS avg_vintage
FROM ranked
GROUP BY
    propensity_decile,
    campaign_band,
    targeting_recommendation,
    cross_sell_opportunity_segment
ORDER BY
    propensity_decile,
    avg_propensity_score DESC;