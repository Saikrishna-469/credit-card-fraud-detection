# Cleaning notes — Credit Card Fraud Detection

## Source
- Kaggle dataset "Credit Card Fraud Detection" (`mlg-ulb/creditcardfraud`),
  file `creditcard.csv`, downloaded 2026-11-20 from the public TensorFlow
  mirror of the same file
  (`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`,
  150,828,752 bytes) because the Kaggle download endpoint requires a Kaggle
  account. The file is byte-identical in shape to the Kaggle original:
  284,807 rows, 31 columns, 492 frauds — the canonical counts for this
  dataset. A `curl -I` on the Kaggle API endpoint returned 403 without
  credentials, so the mirror was used and the row/fraud counts were verified
  on load.
- The data are credit-card transactions by European cardholders over two
  days in September 2013. `Time` is seconds since the first transaction in
  the file; `V1`–`V28` are PCA-transformed features (the originals are
  confidential and were not released); `Amount` is the transaction amount;
  `Class` is 1 for fraud, 0 for legitimate.

## Decisions (each one affects the numbers — documented, not hidden)

1. **Duplicates kept.** 1,081 rows are exact duplicates of another row on
   every column. They are kept: the canonical published counts are 284,807
   transactions / 492 frauds, and dropping duplicates would make every number
   in this repo disagree with the dataset everyone else uses. Duplicates are
   0.38% of rows and include both classes, so they do not move any headline
   number materially.
2. **No nulls, no type cleaning.** 0 null values in any column; `Time`,
   `V1`–`V28` and `Amount` are numeric, `Class` is cast to integer. Nothing
   was imputed or dropped.
3. **Day split at Time = 86,400.** `day` = 1 for the first 24 hours, 2 for
   the second. This is the train/test split: models train on day 1 (144,786
   transactions, 281 frauds) and are tested on day 2 (140,021 transactions,
   211 frauds). A random split would leak — fraud patterns drift within
   hours, and a model that has "seen the future" flatters itself.
4. **Hour is relative, not clock time.** `hour` = (Time mod 86,400) / 3,600.
   The file gives no start timestamp, so hour 0 is the hour the first
   transaction happened, not midnight. The overnight pattern (fraud rate
   0.518% at hours 00–05 vs 0.12–0.17% in the daytime) is real relative to
   the cardholders' day, but the absolute clock hours are unknown. Do not
   quote "2 a.m." as a finding.
5. **Amount bands** are defined in `sql/01_load.sql` and used consistently:
   <€10, €10–25, €25–50, €50–100, €100–250, €250–500, €500–1,000, €1,000+.
6. **Features for the models:** V1–V28 + log1p(Amount) + hour. `Time` itself
   is not a feature (it is the split key). No post-transaction fields exist
   in this dataset, so there is nothing to leak — but note the V-features are
   PCA components of the transaction as recorded, i.e. available at
   transaction time.
7. **Threshold tuned on training data only.** The alert threshold maximises
   F1 on 3-fold out-of-fold predictions from day 1, then is applied
   untouched to day 2. Tuning on the test set would overstate performance.

## QA results (from `load.py`)
- Total transactions: 284,807; fraud: 492 (0.1727%, 1 in 579)
- Time range: 0–172,792 seconds (just under 2 days)
- Day 1: 144,786 transactions, 281 frauds; day 2: 140,021 transactions, 211 frauds
- Duplicate rows: 1,081; nulls: 0
- Total amount: €25,162,590.01; fraud amount: €60,127.97 (0.24% of value)
- Fraud median amount €9.25 vs legitimate €22.00; fraud mean €122.21 vs €88.29

## Known quirks
- **V-features are uninterpretable by design.** They are PCA components;
  V14 or V4 cannot be translated back into "card-present" or "online".
  The analysis says which components carry the signal, not what they mean.
  Any business rule built on them must be built on the PCA pipeline, not on
  a story about the feature.
- **Two days only.** Fraud patterns, volumes and the cost model are a
  48-hour snapshot from September 2013. Fraud tactics, card technology
  (chip, 3-D Secure, tokenisation) and the transaction mix have all changed
  since. The *method* — time-based split, precision/recall at a tuned
  threshold, cost per alert — transfers; the *levels* do not.
- **No customer or merchant fields.** There is no cardholder id, merchant,
  country or device, so velocity features (transactions per card per hour),
  merchant risk and device fingerprints — the strongest signals in a real
  fraud system — cannot be built here. The model is a transaction-level
  scorer only.
- **The €15 false-alert cost is an assumption.** It prices analyst review
  time per alert. The real cost includes customer friction (a blocked good
  customer) and is different for every issuer; the cost table is a
  sensitivity framework, not a quoted saving. Re-run with your own cost and
  the ranking of the two models may change.
- **Fraud $ is small in this sample** (€60,128 over two days) because the
  sample is small and European 2013 fraud was mostly low-value testing.
  Scale the cost model to your own volume before quoting savings.
