-- 01_load.sql — load the Kaggle ULB credit-card transactions into DuckDB.
--
-- Source: creditcard.csv (Kaggle "Credit Card Fraud Detection", mlg-ulb/creditcardfraud).
-- 284,807 transactions over 2 days in September 2013, European cardholders.
-- Columns: Time (seconds since the first transaction in the file), V1..V28
-- (PCA-transformed features — the originals are confidential), Amount, Class
-- (1 = fraud, 0 = legitimate).
--
-- Derived columns:
--   day        1 = first 24h (Time < 86400), 2 = second 24h. Used for the
--              time-based train/test split: a fraud model must be tested on
--              transactions that happen AFTER the ones it trained on.
--   hour       hour of the (relative) day, 0-23. Time is relative to the first
--              transaction, not clock time — see notes/cleaning_notes.md.
--   amount_band  transaction-size bands used in every amount analysis.

CREATE OR REPLACE TABLE transactions AS
SELECT
    "Time"  AS txn_time,
    "V1"    AS v1,  "V2"  AS v2,  "V3"  AS v3,  "V4"  AS v4,  "V5"  AS v5,
    "V6"    AS v6,  "V7"  AS v7,  "V8"  AS v8,  "V9"  AS v9,  "V10" AS v10,
    "V11"   AS v11, "V12" AS v12, "V13" AS v13, "V14" AS v14, "V15" AS v15,
    "V16"   AS v16, "V17" AS v17, "V18" AS v18, "V19" AS v19, "V20" AS v20,
    "V21"   AS v21, "V22" AS v22, "V23" AS v23, "V24" AS v24, "V25" AS v25,
    "V26"   AS v26, "V27" AS v27, "V28" AS v28,
    "Amount" AS amount,
    CAST("Class" AS INTEGER) AS is_fraud,
    CASE WHEN "Time" < 86400 THEN 1 ELSE 2 END AS day,
    CAST(FLOOR(("Time" % 86400) / 3600) AS INTEGER) AS hour,
    CASE
        WHEN "Amount" < 10   THEN '< €10'
        WHEN "Amount" < 25   THEN '€10–25'
        WHEN "Amount" < 50   THEN '€25–50'
        WHEN "Amount" < 100  THEN '€50–100'
        WHEN "Amount" < 250  THEN '€100–250'
        WHEN "Amount" < 500  THEN '€250–500'
        WHEN "Amount" < 1000 THEN '€500–1,000'
        ELSE '€1,000+'
    END AS amount_band
FROM read_csv('data/raw/creditcard.csv', header = true);
