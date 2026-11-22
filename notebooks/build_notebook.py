#!/usr/bin/env python3
"""Build the EDA notebook (notebooks/01_eda.ipynb) and the charts in images/.

Usage (from the project root):
    python3 notebooks/build_notebook.py

Reads data/transactions.duckdb (built by load.py), reproduces the headline
charts with matplotlib, fits two fraud models — logistic regression and
random forest, both class-weighted — on a TIME-BASED split (train = day 1,
test = day 2, so the model is tested on transactions that happen after the
ones it trained on), tunes the alert threshold on out-of-fold predictions
from the training day, and prices the result with a simple cost model:
a missed fraud costs its full amount; a false alert costs EUR 15 of analyst
time (assumption — stated in the README and the notebook).

Accuracy is reported only to show why it is meaningless here: predicting
"legitimate" for every transaction scores 99.85% and catches zero fraud.
The metrics that matter are precision, recall, average precision and the
euros caught vs spent.
"""
import base64
import io

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "transactions.duckdb"
IMG = ROOT / "images"
IMG.mkdir(exist_ok=True)
DASH = ROOT / "dashboard"
DASH.mkdir(exist_ok=True)

plt.rcParams.update({"figure.figsize": (10, 5.5), "font.size": 11})
NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
RUST = "#c0392b"
GREY = "#6b7280"

FP_COST_EUR = 15.0  # analyst time per false alert — assumption, see README

con = duckdb.connect(str(DB), read_only=True)
df = con.execute("SELECT * FROM transactions ORDER BY txn_time").fetchdf()
print(f"transactions: {len(df):,}  fraud: {int(df['is_fraud'].sum())}")

charts = {}  # name -> (png bytes, caption)


def save_chart(name, fig, caption):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    charts[name] = (buf.getvalue(), caption)
    (IMG / f"{name}.png").write_bytes(buf.getvalue())
    print(f"chart: {name}.png")


# ---------------------------------------------------------------- charts
# 1. Fraud rate by amount band
band = con.execute("""
    SELECT amount_band, COUNT(*) AS n, SUM(is_fraud) AS fraud,
           ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
    FROM transactions GROUP BY amount_band ORDER BY MIN(amount)
""").fetchdf()
order = ["< €10", "€10–25", "€25–50", "€50–100", "€100–250", "€250–500",
         "€500–1,000", "€1,000+"]
band["amount_band"] = pd.Categorical(band["amount_band"], categories=order, ordered=True)
band = band.sort_values("amount_band")
fig, ax = plt.subplots()
bars = ax.bar(band["amount_band"], band["fraud_rate_pct"], color=RUST, width=0.6)
ax.set_ylabel("Fraud rate (%)")
ax.set_xlabel("Transaction amount band")
ax.set_title("Fraud concentrates at the extremes: tiny probes and big tickets")
for b, v, n in zip(bars, band["fraud_rate_pct"], band["fraud"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}%  ({int(n)})",
            ha="center", fontsize=10)
plt.xticks(rotation=25, ha="right")
save_chart("fraud_rate_by_amount_band", fig,
           "Fraud rate by amount band; count of frauds in brackets. "
           "Half of all frauds are under €10 — the fraudster's probe — while "
           "€500–1,000 has the highest rate (0.405%).")

# 2. Fraud rate by hour
hour = con.execute("""
    SELECT hour, COUNT(*) AS n, SUM(is_fraud) AS fraud,
           ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
    FROM transactions GROUP BY hour ORDER BY hour
""").fetchdf()
fig, ax = plt.subplots()
ax.bar(hour["hour"], hour["fraud_rate_pct"], color=NAVY, width=0.7)
ax.axhline(0.1727, color=RUST, ls="--", lw=1.5, label="Overall rate 0.173%")
ax.set_ylabel("Fraud rate (%)")
ax.set_xlabel("Hour of the (relative) day")
ax.set_title("Fraud rate is 3–4x higher overnight (00–05)")
ax.set_xticks(range(0, 24))
ax.legend()
save_chart("fraud_rate_by_hour", fig,
           "Fraud rate by hour. Hours are relative to the first transaction "
           "in the file, not clock time. Hour 02 peaks at 1.71%.")

# 3. Amount distribution, fraud vs legit (log scale)
fig, ax = plt.subplots()
bins = np.logspace(np.log10(0.5), np.log10(26000), 60)
ax.hist(df.loc[df["is_fraud"] == 0, "amount"], bins=bins, alpha=0.6,
        label="Legitimate", color=TEAL)
ax.hist(df.loc[df["is_fraud"] == 1, "amount"], bins=bins, alpha=0.85,
        label="Fraud", color=RUST)
ax.set_xscale("log")
ax.set_xlabel("Transaction amount (€, log scale)")
ax.set_ylabel("Transactions")
ax.set_title("Fraud is small: median €9.25 vs €22.00 for legitimate")
ax.legend()
ax.axvline(9.25, color=RUST, ls="--", lw=1)
ax.axvline(22.00, color=TEAL, ls="--", lw=1)
save_chart("amount_distribution", fig,
           "Amount distribution on a log scale. Fraud's median is €9.25 "
           "against €22.00 for legitimate — small test purchases dominate.")

# ---------------------------------------------------------------- models
FEATURES = [f"v{i}" for i in range(1, 29)] + ["log_amount", "hour"]
df["log_amount"] = np.log1p(df["amount"])
X = df[FEATURES].to_numpy()
y = df["is_fraud"].to_numpy()
amount = df["amount"].to_numpy()

train_mask = df["day"].to_numpy() == 1
test_mask = ~train_mask
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
amt_test = amount[test_mask]
print(f"train (day 1): {len(y_train):,} rows, {int(y_train.sum())} fraud | "
      f"test (day 2): {len(y_test):,} rows, {int(y_test.sum())} fraud")

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


def tune_threshold(model, X_tr, y_tr):
    """Threshold maximising F1 on 3-fold out-of-fold predictions (train only)."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best = int(np.argmax(f1[:-1]))
    return float(thr[best]), float(f1[best])


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    thr, oof_f1 = tune_threshold(model, X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    out = {"model": name, "threshold": thr, "oof_f1_at_threshold": round(oof_f1, 3)}
    for label, t in [("0.50", 0.5), ("tuned", thr)]:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        out[f"roc_auc@{label}"] = round(roc_auc_score(y_te, proba), 4)
        out[f"avg_precision@{label}"] = round(average_precision_score(y_te, proba), 4)
        out[f"precision@{label}"] = round(precision_score(y_te, pred, zero_division=0), 3)
        out[f"recall@{label}"] = round(recall_score(y_te, pred, zero_division=0), 3)
        out[f"f1@{label}"] = round(f1_score(y_te, pred, zero_division=0), 3)
        out[f"tp@{label}"] = int(tp)
        out[f"fp@{label}"] = int(fp)
        out[f"fn@{label}"] = int(fn)
        out[f"alerts@{label}"] = int(tp + fp)
    out["proba"] = proba
    return out


lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                            max_features="sqrt", n_jobs=-1, random_state=42)

res_lr = evaluate("Logistic regression", lr, X_train_s, y_train, X_test_s, y_test)
res_rf = evaluate("Random forest", rf, X_train, y_train, X_test, y_test)
for r in (res_lr, res_rf):
    print(r["model"], "| tuned thr", r["threshold"],
          "| ROC-AUC", r["roc_auc@tuned"], "| AP", r["avg_precision@tuned"],
          "| P/R/F1 @tuned:", r["precision@tuned"], r["recall@tuned"], r["f1@tuned"],
          "| alerts", r["alerts@tuned"], "TP", r["tp@tuned"], "FP", r["fp@tuned"],
          "FN", r["fn@tuned"])

# Baseline: predict everything legitimate
print("baseline (predict all legit): accuracy",
      round(1 - y_test.mean(), 4), "| recall 0.000 | fraud € caught 0.00")

# 4. Precision–recall curves on day 2
fig, ax = plt.subplots()
for r, color in [(res_lr, NAVY), (res_rf, TEAL)]:
    prec, rec, _ = precision_recall_curve(y_test, r["proba"])
    ax.plot(rec, prec, color=color, lw=2,
            label=f"{r['model']} (AP {r['avg_precision@tuned']:.3f})")
ax.axhline(y_test.mean(), color=RUST, ls="--", lw=1.5,
           label=f"Chance (fraud rate {y_test.mean():.4f})")
ax.set_xlabel("Recall (share of fraud caught)")
ax.set_ylabel("Precision (share of alerts that are fraud)")
ax.set_title("Precision–recall on day 2: both models beat chance by ~500x")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower left")
save_chart("precision_recall_curve", fig,
           "Precision–recall curves on the day-2 test set. Average precision "
           "is the honest metric at 0.17% fraud — ROC-AUC flatters.")

# 5. Confusion matrix, random forest at tuned threshold
pred_rf = (res_rf["proba"] >= res_rf["threshold"]).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_rf).ravel()
fig, ax = plt.subplots(figsize=(6, 5))
mat = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(mat, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted legit", "Predicted fraud"])
ax.set_yticklabels(["Actually legit", "Actually fraud"])
ax.set_title(f"Random forest @ threshold {res_rf['threshold']:.3f} — day 2 "
             f"({len(y_test):,} transactions)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if mat[i, j] > mat.max() / 2 else "black")
save_chart("confusion_matrix_rf", fig,
           "Confusion matrix for the random forest at the F1-tuned threshold, "
           "day-2 test set.")

# 6. Random-forest feature importance (top 12)
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
top = imp.head(12).sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top.index, top.values, color=TEAL)
ax.set_xlabel("Random-forest feature importance")
ax.set_title("V14 dominates the fraud signal, with V10 and V17 next")
save_chart("feature_importance", fig,
           "Top 12 features by random-forest importance. V14 dominates "
           "(importance ~0.20), followed by V10 and V17 — the same components "
           "with the largest standardised mean differences in "
           "sql/04_features.sql (V14 d = -2.26, V10 -1.61, V17 -1.35).")

# ---------------------------------------------------------------- cost model
rows = []
fraud_total_day2 = float(amt_test[y_test == 1].sum())
for r in (res_lr, res_rf):
    pred = (r["proba"] >= r["threshold"]).astype(int)
    caught = float(amt_test[(y_test == 1) & (pred == 1)].sum())
    missed = fraud_total_day2 - caught
    n_fp = int(((y_test == 0) & (pred == 1)).sum())
    fp_cost = n_fp * FP_COST_EUR
    net = caught - fp_cost
    rows.append({
        "model": r["model"], "threshold": round(r["threshold"], 3),
        "fraud_caught_eur": round(caught, 2),
        "fraud_missed_eur": round(missed, 2),
        "false_alerts": n_fp,
        "false_alert_cost_eur": round(fp_cost, 2),
        "net_benefit_eur": round(net, 2),
        "share_of_fraud_value_caught_pct": round(100 * caught / fraud_total_day2, 1),
    })
rows.append({"model": "No model (catch nothing)", "threshold": None,
             "fraud_caught_eur": 0.0, "fraud_missed_eur": round(fraud_total_day2, 2),
             "false_alerts": 0, "false_alert_cost_eur": 0.0,
             "net_benefit_eur": 0.0, "share_of_fraud_value_caught_pct": 0.0})
cost_df = pd.DataFrame(rows)
print(cost_df.to_string(index=False))
print(f"fraud value on day 2: €{fraud_total_day2:,.2f}")
cost_df.to_csv(DASH / "05_cost_model.csv", index=False)

metrics = pd.DataFrame([{k: v for k, v in r.items() if k != "proba"}
                        for r in (res_lr, res_rf)])
metrics.to_csv(DASH / "05_model_metrics.csv", index=False)

with open(ROOT / "notebooks" / "model_results.txt", "w") as f:
    f.write("Model results — time-based split, train day 1 / test day 2\n")
    f.write(f"Train: {len(y_train):,} rows, {int(y_train.sum())} fraud; "
            f"Test: {len(y_test):,} rows, {int(y_test.sum())} fraud\n\n")
    f.write(metrics.to_string(index=False))
    f.write("\n\nCost model (day 2, false alert = EUR 15 analyst time):\n")
    f.write(cost_df.to_string(index=False))
    f.write(f"\n\nTotal fraud value on day 2: EUR {fraud_total_day2:,.2f}\n")
    f.write("Baseline 'predict all legit': accuracy "
            f"{1 - y_test.mean():.4f}, recall 0, fraud caught EUR 0.00\n")
print("model_results.txt written")

# ---------------------------------------------------------------- notebook
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Credit Card Fraud Detection — EDA and modelling\n\n"
        "**Business question:** a card issuer sees 284,807 transactions in two "
        "days and 492 of them are fraud (0.17%). Every fraud that slips through "
        "costs the full transaction amount; every false alert costs analyst time "
        "and annoys a good customer. How do you catch the fraud without "
        "drowning the review team?\n\n"
        "This notebook reproduces the headline charts and the two models "
        "(logistic regression and random forest, both class-weighted) from "
        "`notebooks/build_notebook.py`. The models are trained on day 1 and "
        "tested on day 2 — a time-based split, so the test set is genuinely "
        "in the future relative to training."),
    nbf.v4.new_code_cell(
        "import duckdb\n"
        "con = duckdb.connect('data/transactions.duckdb', read_only=True)\n"
        "df = con.execute('SELECT * FROM transactions ORDER BY txn_time').fetchdf()\n"
        "print(len(df), 'transactions,', int(df['is_fraud'].sum()), 'fraud')"),
    nbf.v4.new_markdown_cell(
        "## 1. Fraud concentrates at the extremes\n\n"
        "Half of all frauds (249 of 492) are under €10 — small test purchases "
        "before the card is used for real. The highest fraud *rate* is in the "
        "€500–1,000 band (0.405%)."),
    nbf.v4.new_code_cell(
        "# see notebooks/build_notebook.py — chart: fraud_rate_by_amount_band\n"
        "from IPython.display import Image\n"
        "Image('images/fraud_rate_by_amount_band.png')"),
    nbf.v4.new_markdown_cell(
        "## 2. Fraud is a night-time problem\n\n"
        "Fraud rate 00–05 is 0.518% vs 0.12–0.17% in the daytime; hour 02 alone "
        "hits 1.71%. Hours are relative to the first transaction in the file, "
        "not clock time — see notes/cleaning_notes.md."),
    nbf.v4.new_code_cell(
        "Image('images/fraud_rate_by_hour.png')"),
    nbf.v4.new_markdown_cell(
        "## 3. Fraud is small — median €9.25 vs €22.00\n\n"
        "The amount distributions overlap heavily, which is why amount alone "
        "cannot be the detector; the V-features have to do the work."),
    nbf.v4.new_code_cell(
        "Image('images/amount_distribution.png')"),
    nbf.v4.new_markdown_cell(
        "## 4. Models — time-based split, threshold tuned on training data only\n\n"
        "Logistic regression and random forest, both class-weighted, features "
        "V1–V28 + log(Amount) + hour. The alert threshold maximises F1 on "
        "3-fold out-of-fold predictions from day 1, then is applied untouched "
        "to day 2."),
    nbf.v4.new_code_cell(
        "# see notebooks/build_notebook.py for the full training code\n"
        "# and notebooks/model_results.txt for the numbers"),
    nbf.v4.new_markdown_cell(
        "## 5. Precision–recall: the honest picture at 0.17% fraud\n\n"
        "ROC-AUC flatters at this imbalance; average precision is the metric "
        "that matters. Both models beat the 0.0015 chance baseline by ~500x."),
    nbf.v4.new_code_cell(
        "Image('images/precision_recall_curve.png')"),
    nbf.v4.new_markdown_cell(
        "## 6. What the trade-off costs in euros\n\n"
        "A missed fraud costs its full amount; a false alert costs EUR 15 of "
        "analyst time (assumption). See `dashboard/05_cost_model.csv`."),
    nbf.v4.new_code_cell(
        "Image('images/confusion_matrix_rf.png')"),
    nbf.v4.new_code_cell(
        "Image('images/feature_importance.png')"),
]
# attach chart PNGs as outputs of the Image() cells
img_cells = {"fraud_rate_by_amount_band": 3, "fraud_rate_by_hour": 5,
             "amount_distribution": 7, "precision_recall_curve": 11,
             "confusion_matrix_rf": 13, "feature_importance": 14}
for name, idx in img_cells.items():
    png, _ = charts[name]
    nb.cells[idx].outputs = [nbf.v4.new_output(
        "display_data",
        data={"image/png": base64.b64encode(png).decode()},
        metadata={})]
nbf.write(nb, ROOT / "notebooks" / "01_eda.ipynb")
print("notebook written: notebooks/01_eda.ipynb")
print("OK")
