#!/usr/bin/env python3
"""Regenerate ONLY images/feature_importance.png with the corrected title.

The first build of build_notebook.py hardcoded the wrong top-3 features in
the chart title (V14/V4/V12); the trained forest actually ranks V14, V10,
V17 first. This script retrains the same random forest (same seed, same
features, same day-1 training set), redraws the chart, and patches the
embedded image in notebooks/01_eda.ipynb. It does not touch any numbers.
"""
import base64
import io
import json

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "transactions.duckdb"

plt.rcParams.update({"figure.figsize": (10, 5.5), "font.size": 11})
TEAL = "#2a9d8f"

con = duckdb.connect(str(DB), read_only=True)
df = con.execute("SELECT * FROM transactions ORDER BY txn_time").fetchdf()
df["log_amount"] = np.log1p(df["amount"])
FEATURES = [f"v{i}" for i in range(1, 29)] + ["log_amount", "hour"]
train = df[df["day"] == 1]
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                            max_features="sqrt", n_jobs=-1, random_state=42)
rf.fit(train[FEATURES].to_numpy(), train["is_fraud"].to_numpy())

imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp.head(5).to_string())
top = imp.head(12).sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top.index, top.values, color=TEAL)
ax.set_xlabel("Random-forest feature importance")
ax.set_title("V14 dominates the fraud signal, with V10 and V17 next")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
plt.close(fig)
png = buf.getvalue()
(ROOT / "images" / "feature_importance.png").write_bytes(png)
print("chart rewritten: images/feature_importance.png")

# patch the embedded image in the notebook (cell 14, the feature_importance cell)
nb_path = ROOT / "notebooks" / "01_eda.ipynb"
nb = json.loads(nb_path.read_text())
cell = nb["cells"][14]
assert "feature_importance" in cell["source"][-1] or "feature_importance" in "".join(cell["source"]), cell["source"]
cell["outputs"] = [{
    "output_type": "display_data",
    "data": {"image/png": base64.b64encode(png).decode()},
    "metadata": {},
}]
nb_path.write_text(json.dumps(nb, indent=1))
print("notebook patched: cell 14 output replaced")
print("OK")
