#!/usr/bin/env python3
"""Load the Kaggle ULB credit-card transactions CSV into DuckDB and run QA.

Usage (from the project root):
    python3 load.py

Reads sql/01_load.sql, builds data/transactions.duckdb, then prints a QA
summary. The raw CSV lives in data/raw/ (git-ignored); the .duckdb file is
also git-ignored — re-run this script to rebuild it.
"""
import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "transactions.duckdb"
SQL = ROOT / "sql" / "01_load.sql"

con = duckdb.connect(str(DB))
con.execute(SQL.read_text())
print("Loaded tables:", [r[0] for r in con.execute("SHOW TABLES").fetchall()])

qa = {
    "total transactions": "SELECT COUNT(*) FROM transactions",
    "fraud transactions": "SELECT SUM(is_fraud) FROM transactions",
    "fraud rate (%)": "SELECT ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) FROM transactions",
    "time range (seconds)": "SELECT MIN(txn_time), MAX(txn_time) FROM transactions",
    "day 1 / day 2 split": "SELECT day, COUNT(*) FROM transactions GROUP BY day ORDER BY day",
    "duplicate rows (all columns equal)":
        "SELECT COUNT(*) - COUNT(DISTINCT (txn_time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount, is_fraud)) FROM transactions",
    "nulls in any column": "SELECT COUNT(*) FROM transactions WHERE txn_time IS NULL OR amount IS NULL OR is_fraud IS NULL",
    "total amount": "SELECT ROUND(SUM(amount), 2) FROM transactions",
    "fraud amount": "SELECT ROUND(SUM(amount), 2) FROM transactions WHERE is_fraud = 1",
}
for label, q in qa.items():
    print(f"- {label}: {con.execute(q).fetchone()}")
con.close()
print("OK")
