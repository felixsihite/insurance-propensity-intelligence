-- Insurance Customer Propensity Prediction & Customer Intelligence
-- Customer profile and response-rate analysis.
-- Assumed source table: train_customers

WITH base AS (
    SELECT
        id,
        Gender,
        Age,
        Driving_License,
        Region_Code,
        Previously_Insured,
        Vehicle_Age,
        Vehicle_Damage,
        Annual_Premium,
        Policy_Sales_Channel,
        Vintage,
        Response,
        CASE
            WHEN Age <= 25 THEN '18-25'
            WHEN Age <= 35 THEN '26-35'
            WHEN Age <= 45 THEN '36-45'
            WHEN Age <= 55 THEN '46-55'
            WHEN Age <= 65 THEN '56-65'
            ELSE '66+'
        END AS age_group,
        CASE
            WHEN Annual_Premium <= 20000 THEN 'Low'
            WHEN Annual_Premium <= 35000 THEN 'Core'
            WHEN Annual_Premium <= 50000 THEN 'Upper Core'
            WHEN Annual_Premium <= 75000 THEN 'High'
            ELSE 'Premium'
        END AS premium_band
    FROM train_customers
)

SELECT
    'overall' AS dimension,
    'all_customers' AS segment,
    COUNT(*) AS customers,
    SUM(Response) AS responders,
    AVG(CAST(Response AS FLOAT)) AS response_rate,
    AVG(Age) AS avg_age,
    AVG(Annual_Premium) AS avg_annual_premium,
    AVG(Vintage) AS avg_vintage
FROM base

UNION ALL

SELECT
    'gender' AS dimension,
    Gender AS segment,
    COUNT(*) AS customers,
    SUM(Response) AS responders,
    AVG(CAST(Response AS FLOAT)) AS response_rate,
    AVG(Age) AS avg_age,
    AVG(Annual_Premium) AS avg_annual_premium,
    AVG(Vintage) AS avg_vintage
FROM base
GROUP BY Gender

UNION ALL

SELECT
    'vehicle_age' AS dimension,
    Vehicle_Age AS segment,
    COUNT(*) AS customers,
    SUM(Response) AS responders,
    AVG(CAST(Response AS FLOAT)) AS response_rate,
    AVG(Age) AS avg_age,
    AVG(Annual_Premium) AS avg_annual_premium,
    AVG(Vintage) AS avg_vintage
FROM base
GROUP BY Vehicle_Age

UNION ALL

SELECT
    'vehicle_damage' AS dimension,
    Vehicle_Damage AS segment,
    COUNT(*) AS customers,
    SUM(Response) AS responders,
    AVG(CAST(Response AS FLOAT)) AS response_rate,
    AVG(Age) AS avg_age,
    AVG(Annual_Premium) AS avg_annual_premium,
    AVG(Vintage) AS avg_vintage
FROM base
GROUP BY Vehicle_Damage

UNION ALL

SELECT
    'previously_insured' AS dimension,
    CAST(Previously_Insured AS VARCHAR(20)) AS segment,
    COUNT(*) AS customers,
    SUM(Response) AS responders,
    AVG(CAST(Response AS FLOAT)) AS response_rate,
    AVG(Age) AS avg_age,
    AVG(Annual_Premium) AS avg_annual_premium,
    AVG(Vintage) AS avg_vintage
FROM base
GROUP BY Previously_Insured

UNION ALL

SELECT
    'premium_band' AS dimension,
    premium_band AS segment,
    COUNT(*) AS customers,
    SUM(Response) AS responders,
    AVG(CAST(Response AS FLOAT)) AS response_rate,
    AVG(Age) AS avg_age,
    AVG(Annual_Premium) AS avg_annual_premium,
    AVG(Vintage) AS avg_vintage
FROM base
GROUP BY premium_band;
