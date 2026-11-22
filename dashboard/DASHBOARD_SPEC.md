# Dashboard spec — Credit Card Fraud Detection (Power BI / Tableau)

Static dashboard: `dashboard/index.html`. To rebuild it in Power BI or
Tableau, connect to the CSVs in `dashboard/` (all outputs of
`run_analysis.py` and `notebooks/build_notebook.py`).

## Pages

### Page 1 — Overview
- **KPI cards:** 284,807 transactions · 492 frauds (0.1727%, 1 in 579) ·
  €25.16M total value · €60,128 fraud value (0.24% of value).
- **Chart:** fraud rate by amount band (`03_Q1.csv`, bar) — annotate the
  <€10 band: 249 of 492 frauds (50.6%) but only €459 of fraud value.
- **Chart:** fraud rate by hour (`03_Q2.csv`, bar with 0.173% reference line)
  — annotate hour 02 at 1.71% and the 00–05 band at 0.518%.
- **Table:** amount percentiles by class (`02_Q4.csv`) — fraud median €9.25
  vs legitimate €22.00.

### Page 2 — What separates fraud (features)
- **Table:** standardised mean difference (Cohen's d) for V1–V28
  (`04_Q1.csv`, sorted by |d|). Highlight V14 (d = −2.26), V4 (+2.01),
  V11 (+1.88), V12 (−1.87) — separations this large are rare in real data.
- **Chart:** random-forest feature importance (`images/feature_importance.png`
  or rebuild from the model) — top 12. V14 dominates (importance ~0.20),
  followed by V10 and V17.
- **Note card:** V-features are PCA components; they carry signal but no
  business meaning. Do not build a rule that says "block when V14 is low"
  without the PCA pipeline behind it.

### Page 3 — Model performance and cost
- **Table:** model metrics (`05_model_metrics.csv`) — ROC-AUC, average
  precision, precision / recall / F1 at the 0.50 and tuned thresholds, for
  logistic regression and random forest, all measured on day 2 only.
- **Chart:** precision–recall curves (`images/precision_recall_curve.png`)
  with the chance baseline (fraud rate 0.0015).
- **Table:** cost model (`05_cost_model.csv`) — fraud € caught, fraud €
  missed, false alerts × €15, net benefit, share of fraud value caught.
  Add a **what-if parameter** for the false-alert cost (€5 / €15 / €50):
  at €50 the tuned threshold is too aggressive and the net benefit falls.
- **Callout:** baseline "predict everything legitimate" scores 99.85%
  accuracy and catches €0 of fraud. Never report accuracy alone.

## Filters (apply to all pages)
- Day (1 / 2) — note day 1 is the training day, day 2 the test day.
- Amount band, hour band.
- Class (fraud / legitimate) for the EDA pages only — never filter the
  model pages to one class.

## Refresh
Re-run `python3 load.py`, `python3 run_analysis.py` and
`python3 notebooks/build_notebook.py` from the project root; the CSVs and
images refresh in place. The raw CSV is git-ignored — download it first
(see README "How to reproduce").
