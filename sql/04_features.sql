-- 04_features.sql — which of the 28 anonymised (PCA) features separate fraud
-- from legitimate transactions? Standardised mean difference (Cohen's d):
-- (mean_fraud - mean_legit) / pooled SD. |d| > 0.8 is a large separation.
-- V1..V28 are PCA components, so a single feature is not interpretable on its
-- own — the point is which components carry the fraud signal, for the model.

-- Q1: standardised mean difference (Cohen's d) for V1..V28, fraud vs legit
WITH stats AS (
    SELECT
        is_fraud,
        AVG(v1) AS m_v1,  AVG(v2) AS m_v2,  AVG(v3) AS m_v3,  AVG(v4) AS m_v4,
        AVG(v5) AS m_v5,  AVG(v6) AS m_v6,  AVG(v7) AS m_v7,  AVG(v8) AS m_v8,
        AVG(v9) AS m_v9,  AVG(v10) AS m_v10, AVG(v11) AS m_v11, AVG(v12) AS m_v12,
        AVG(v13) AS m_v13, AVG(v14) AS m_v14, AVG(v15) AS m_v15, AVG(v16) AS m_v16,
        AVG(v17) AS m_v17, AVG(v18) AS m_v18, AVG(v19) AS m_v19, AVG(v20) AS m_v20,
        AVG(v21) AS m_v21, AVG(v22) AS m_v22, AVG(v23) AS m_v23, AVG(v24) AS m_v24,
        AVG(v25) AS m_v25, AVG(v26) AS m_v26, AVG(v27) AS m_v27, AVG(v28) AS m_v28,
        STDDEV(v1) AS s_v1,  STDDEV(v2) AS s_v2,  STDDEV(v3) AS s_v3,
        STDDEV(v4) AS s_v4,  STDDEV(v5) AS s_v5,  STDDEV(v6) AS s_v6,
        STDDEV(v7) AS s_v7,  STDDEV(v8) AS s_v8,  STDDEV(v9) AS s_v9,
        STDDEV(v10) AS s_v10, STDDEV(v11) AS s_v11, STDDEV(v12) AS s_v12,
        STDDEV(v13) AS s_v13, STDDEV(v14) AS s_v14, STDDEV(v15) AS s_v15,
        STDDEV(v16) AS s_v16, STDDEV(v17) AS s_v17, STDDEV(v18) AS s_v18,
        STDDEV(v19) AS s_v19, STDDEV(v20) AS s_v20, STDDEV(v21) AS s_v21,
        STDDEV(v22) AS s_v22, STDDEV(v23) AS s_v23, STDDEV(v24) AS s_v24,
        STDDEV(v25) AS s_v25, STDDEV(v26) AS s_v26, STDDEV(v27) AS s_v27,
        STDDEV(v28) AS s_v28
    FROM transactions
    GROUP BY is_fraud
),
pivoted AS (
    SELECT 'V1' AS feature,
        (SELECT m_v1 FROM stats WHERE is_fraud = 1) - (SELECT m_v1 FROM stats WHERE is_fraud = 0) AS diff,
        SQRT(((SELECT s_v1 FROM stats WHERE is_fraud = 1) * (SELECT s_v1 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v1 FROM stats WHERE is_fraud = 0) * (SELECT s_v1 FROM stats WHERE is_fraud = 0)) / 2) AS pooled_sd
    UNION ALL SELECT 'V2',
        (SELECT m_v2 FROM stats WHERE is_fraud = 1) - (SELECT m_v2 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v2 FROM stats WHERE is_fraud = 1) * (SELECT s_v2 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v2 FROM stats WHERE is_fraud = 0) * (SELECT s_v2 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V3',
        (SELECT m_v3 FROM stats WHERE is_fraud = 1) - (SELECT m_v3 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v3 FROM stats WHERE is_fraud = 1) * (SELECT s_v3 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v3 FROM stats WHERE is_fraud = 0) * (SELECT s_v3 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V4',
        (SELECT m_v4 FROM stats WHERE is_fraud = 1) - (SELECT m_v4 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v4 FROM stats WHERE is_fraud = 1) * (SELECT s_v4 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v4 FROM stats WHERE is_fraud = 0) * (SELECT s_v4 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V5',
        (SELECT m_v5 FROM stats WHERE is_fraud = 1) - (SELECT m_v5 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v5 FROM stats WHERE is_fraud = 1) * (SELECT s_v5 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v5 FROM stats WHERE is_fraud = 0) * (SELECT s_v5 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V6',
        (SELECT m_v6 FROM stats WHERE is_fraud = 1) - (SELECT m_v6 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v6 FROM stats WHERE is_fraud = 1) * (SELECT s_v6 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v6 FROM stats WHERE is_fraud = 0) * (SELECT s_v6 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V7',
        (SELECT m_v7 FROM stats WHERE is_fraud = 1) - (SELECT m_v7 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v7 FROM stats WHERE is_fraud = 1) * (SELECT s_v7 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v7 FROM stats WHERE is_fraud = 0) * (SELECT s_v7 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V8',
        (SELECT m_v8 FROM stats WHERE is_fraud = 1) - (SELECT m_v8 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v8 FROM stats WHERE is_fraud = 1) * (SELECT s_v8 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v8 FROM stats WHERE is_fraud = 0) * (SELECT s_v8 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V9',
        (SELECT m_v9 FROM stats WHERE is_fraud = 1) - (SELECT m_v9 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v9 FROM stats WHERE is_fraud = 1) * (SELECT s_v9 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v9 FROM stats WHERE is_fraud = 0) * (SELECT s_v9 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V10',
        (SELECT m_v10 FROM stats WHERE is_fraud = 1) - (SELECT m_v10 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v10 FROM stats WHERE is_fraud = 1) * (SELECT s_v10 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v10 FROM stats WHERE is_fraud = 0) * (SELECT s_v10 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V11',
        (SELECT m_v11 FROM stats WHERE is_fraud = 1) - (SELECT m_v11 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v11 FROM stats WHERE is_fraud = 1) * (SELECT s_v11 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v11 FROM stats WHERE is_fraud = 0) * (SELECT s_v11 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V12',
        (SELECT m_v12 FROM stats WHERE is_fraud = 1) - (SELECT m_v12 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v12 FROM stats WHERE is_fraud = 1) * (SELECT s_v12 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v12 FROM stats WHERE is_fraud = 0) * (SELECT s_v12 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V13',
        (SELECT m_v13 FROM stats WHERE is_fraud = 1) - (SELECT m_v13 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v13 FROM stats WHERE is_fraud = 1) * (SELECT s_v13 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v13 FROM stats WHERE is_fraud = 0) * (SELECT s_v13 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V14',
        (SELECT m_v14 FROM stats WHERE is_fraud = 1) - (SELECT m_v14 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v14 FROM stats WHERE is_fraud = 1) * (SELECT s_v14 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v14 FROM stats WHERE is_fraud = 0) * (SELECT s_v14 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V15',
        (SELECT m_v15 FROM stats WHERE is_fraud = 1) - (SELECT m_v15 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v15 FROM stats WHERE is_fraud = 1) * (SELECT s_v15 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v15 FROM stats WHERE is_fraud = 0) * (SELECT s_v15 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V16',
        (SELECT m_v16 FROM stats WHERE is_fraud = 1) - (SELECT m_v16 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v16 FROM stats WHERE is_fraud = 1) * (SELECT s_v16 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v16 FROM stats WHERE is_fraud = 0) * (SELECT s_v16 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V17',
        (SELECT m_v17 FROM stats WHERE is_fraud = 1) - (SELECT m_v17 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v17 FROM stats WHERE is_fraud = 1) * (SELECT s_v17 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v17 FROM stats WHERE is_fraud = 0) * (SELECT s_v17 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V18',
        (SELECT m_v18 FROM stats WHERE is_fraud = 1) - (SELECT m_v18 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v18 FROM stats WHERE is_fraud = 1) * (SELECT s_v18 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v18 FROM stats WHERE is_fraud = 0) * (SELECT s_v18 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V19',
        (SELECT m_v19 FROM stats WHERE is_fraud = 1) - (SELECT m_v19 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v19 FROM stats WHERE is_fraud = 1) * (SELECT s_v19 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v19 FROM stats WHERE is_fraud = 0) * (SELECT s_v19 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V20',
        (SELECT m_v20 FROM stats WHERE is_fraud = 1) - (SELECT m_v20 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v20 FROM stats WHERE is_fraud = 1) * (SELECT s_v20 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v20 FROM stats WHERE is_fraud = 0) * (SELECT s_v20 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V21',
        (SELECT m_v21 FROM stats WHERE is_fraud = 1) - (SELECT m_v21 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v21 FROM stats WHERE is_fraud = 1) * (SELECT s_v21 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v21 FROM stats WHERE is_fraud = 0) * (SELECT s_v21 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V22',
        (SELECT m_v22 FROM stats WHERE is_fraud = 1) - (SELECT m_v22 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v22 FROM stats WHERE is_fraud = 1) * (SELECT s_v22 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v22 FROM stats WHERE is_fraud = 0) * (SELECT s_v22 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V23',
        (SELECT m_v23 FROM stats WHERE is_fraud = 1) - (SELECT m_v23 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v23 FROM stats WHERE is_fraud = 1) * (SELECT s_v23 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v23 FROM stats WHERE is_fraud = 0) * (SELECT s_v23 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V24',
        (SELECT m_v24 FROM stats WHERE is_fraud = 1) - (SELECT m_v24 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v24 FROM stats WHERE is_fraud = 1) * (SELECT s_v24 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v24 FROM stats WHERE is_fraud = 0) * (SELECT s_v24 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V25',
        (SELECT m_v25 FROM stats WHERE is_fraud = 1) - (SELECT m_v25 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v25 FROM stats WHERE is_fraud = 1) * (SELECT s_v25 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v25 FROM stats WHERE is_fraud = 0) * (SELECT s_v25 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V26',
        (SELECT m_v26 FROM stats WHERE is_fraud = 1) - (SELECT m_v26 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v26 FROM stats WHERE is_fraud = 1) * (SELECT s_v26 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v26 FROM stats WHERE is_fraud = 0) * (SELECT s_v26 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V27',
        (SELECT m_v27 FROM stats WHERE is_fraud = 1) - (SELECT m_v27 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v27 FROM stats WHERE is_fraud = 1) * (SELECT s_v27 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v27 FROM stats WHERE is_fraud = 0) * (SELECT s_v27 FROM stats WHERE is_fraud = 0)) / 2)
    UNION ALL SELECT 'V28',
        (SELECT m_v28 FROM stats WHERE is_fraud = 1) - (SELECT m_v28 FROM stats WHERE is_fraud = 0),
        SQRT(((SELECT s_v28 FROM stats WHERE is_fraud = 1) * (SELECT s_v28 FROM stats WHERE is_fraud = 1)
            + (SELECT s_v28 FROM stats WHERE is_fraud = 0) * (SELECT s_v28 FROM stats WHERE is_fraud = 0)) / 2)
)
SELECT
    feature,
    ROUND(diff, 3) AS mean_diff_fraud_minus_legit,
    ROUND(diff / NULLIF(pooled_sd, 0), 3) AS cohens_d
FROM pivoted
ORDER BY ABS(diff / NULLIF(pooled_sd, 0)) DESC;
