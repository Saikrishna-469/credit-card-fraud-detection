-- 03_amount_time.sql — does fraud concentrate in certain amounts or hours?

-- Q1: fraud rate by transaction-size band
SELECT
    amount_band,
    COUNT(*) AS n_transactions,
    SUM(is_fraud) AS n_fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount,
    ROUND(100.0 * SUM(is_fraud) / (SELECT SUM(is_fraud) FROM transactions), 1)
        AS share_of_all_fraud_pct
FROM transactions
GROUP BY amount_band
ORDER BY MIN(amount);

-- Q2: fraud by hour of the (relative) day — count, rate, and fraud $
SELECT
    hour,
    COUNT(*) AS n_transactions,
    SUM(is_fraud) AS n_fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY hour
ORDER BY hour;

-- Q3: fraud rate by hour band (night / morning / afternoon / evening)
SELECT
    CASE
        WHEN hour BETWEEN 0 AND 5  THEN '00–05 (night)'
        WHEN hour BETWEEN 6 AND 11 THEN '06–11 (morning)'
        WHEN hour BETWEEN 12 AND 17 THEN '12–17 (afternoon)'
        ELSE '18–23 (evening)'
    END AS hour_band,
    COUNT(*) AS n_transactions,
    SUM(is_fraud) AS n_fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY hour_band
ORDER BY MIN(hour);
