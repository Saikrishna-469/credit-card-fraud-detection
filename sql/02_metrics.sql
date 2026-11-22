-- 02_metrics.sql — headline KPIs: class balance, amounts, fraud by day.

-- Q1: headline counts and class balance
SELECT
    COUNT(*) AS n_transactions,
    SUM(is_fraud) AS n_fraud,
    COUNT(*) - SUM(is_fraud) AS n_legit,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(1.0 * COUNT(*) / NULLIF(SUM(is_fraud), 0), 1) AS one_fraud_in_every_n
FROM transactions;

-- Q2: money at stake — total, fraud and legitimate amounts
SELECT
    is_fraud,
    COUNT(*) AS n,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS mean_amount,
    ROUND(MEDIAN(amount), 2) AS median_amount,
    ROUND(MAX(amount), 2) AS max_amount,
    ROUND(100.0 * SUM(amount) / (SELECT SUM(amount) FROM transactions), 2) AS share_of_total_amount_pct
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;

-- Q3: fraud by day (train = day 1, test = day 2 in the time-based split)
SELECT
    day,
    COUNT(*) AS n_transactions,
    SUM(is_fraud) AS n_fraud,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY day
ORDER BY day;

-- Q4: amount distribution percentiles by class
SELECT
    is_fraud,
    ROUND(QUANTILE_CONT(amount, 0.50), 2) AS p50,
    ROUND(QUANTILE_CONT(amount, 0.90), 2) AS p90,
    ROUND(QUANTILE_CONT(amount, 0.99), 2) AS p99,
    ROUND(QUANTILE_CONT(amount, 0.999), 2) AS p999
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;
