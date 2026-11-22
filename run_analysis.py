#!/usr/bin/env python3
"""Run the SQL analysis (sql/02–04) and save dashboard CSVs.

Usage (from the project root):
    python3 run_analysis.py

Reads data/transactions.duckdb (built by load.py), prints every query in
sql/02_metrics.sql, sql/03_amount_time.sql and sql/04_features.sql, and
writes the result of each query to dashboard/<nn>_<name>.csv.
"""
import re
import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "transactions.duckdb"
DASH = ROOT / "dashboard"
DASH.mkdir(exist_ok=True)

con = duckdb.connect(str(DB), read_only=True)


def split_queries(sql_text):
    """Split a .sql file into (name, sql) pairs on '-- Qn:' comment headers."""
    parts = re.split(r"-- (Q\d+):[^\n]*\n", sql_text)
    # parts: [preamble, 'Q1', sql1, 'Q2', sql2, ...]
    out = []
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i], parts[i + 1]))
    return out


for fname in ["02_metrics.sql", "03_amount_time.sql", "04_features.sql"]:
    sql_text = (ROOT / "sql" / fname).read_text()
    prefix = fname[:2]
    for qname, qsql in split_queries(sql_text):
        df = con.execute(qsql).fetchdf()
        print(f"\n=== {fname} {qname} ===")
        print(df.to_string(index=False))
        out = DASH / f"{prefix}_{qname}.csv"
        df.to_csv(out, index=False)
        print(f"saved {out.name} ({len(df)} rows)")

con.close()
print("\nOK")
