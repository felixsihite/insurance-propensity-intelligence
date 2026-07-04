-- Segment performance analysis for cross-sell campaign planning.
-- Assumed source table: train_customers

WITH enriched AS (
    SELECT
        *,
        CASE
            WHEN Previously_Insured = 0 AND Vehicle_Damage = 'Yes' AND Age BETWEEN 30 AND 60 THEN 'Priority'
            WHEN Previously_Insured = 0 AND Vehicle_Damage = 'Yes' THEN 'High'
            WHEN Previously_Insured = 0 THEN 'Nurture'
            WHEN Vehicle_Damage = 'Yes' THEN 'Monitor'
            ELSE 'Low'
        END AS cross_sell_opportunity_segment,
        CASE
            WHEN Age <= 25 THEN '18-25'
            WHEN Age <= 35 THEN '26-35'
            WHEN Age <= 45 THEN '36-45'
            WHEN Age <= 55 THEN '46-55'
            WHEN Age <= 65 THEN '56-65'
            ELSE '66+'
        END AS age_group
    FROM train_customers
),
overall AS (
    SELECT AVG(CAST(Response AS FLOAT)) AS overall_response_rate
    FROM enriched
)

SELECT
    e.cross_sell_opportunity_segment,
    e.age_group,
    e.Vehicle_Age,
    e.Vehicle_Damage,
    e.Previously_Insured,
    COUNT(*) AS customers,
    SUM(e.Response) AS responders,
    AVG(CAST(e.Response AS FLOAT)) AS response_rate,
    AVG(CAST(e.Response AS FLOAT)) / NULLIF(o.overall_response_rate, 0) AS lift_vs_overall,
    AVG(e.Annual_Premium) AS avg_annual_premium,
    AVG(e.Vintage) AS avg_vintage
FROM enriched e
CROSS JOIN overall o
GROUP BY
    e.cross_sell_opportunity_segment,
    e.age_group,
    e.Vehicle_Age,
    e.Vehicle_Damage,
    e.Previously_Insured,
    o.overall_response_rate
HAVING COUNT(*) >= 100
ORDER BY lift_vs_overall DESC, customers DESC;