"""Refit the shipped ML1 model (LightGBM) in the devdesk repo and dump it with joblib.

Usage (from the devdesk folder, in its ml environment):
    uv run --group ml python ../focus-streamlit/dump_model.py
Writes model/focus_model.joblib, model/features_meta.joblib and dataset/focus_features.csv next to this file.
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path.cwd()))  # the devdesk repo: ml.train_focus holds the data loading and model settings
from ml import behacom
from ml.focus_features import CATEGORIES
from ml.train_focus import ROOT, make_dataset, models

HERE = Path(__file__).resolve().parent

X, y, _ = make_dataset(behacom.load(ROOT / "data/raw/behacom/behacom_slim.csv"), behacom.TZ)
model = models()["lightgbm"]().fit(X, y)
meta = {"features": list(X.columns), "classes": list(model.classes_), "categories": list(CATEGORIES)}

(HERE / "model").mkdir(exist_ok=True)
(HERE / "dataset").mkdir(exist_ok=True)
joblib.dump(model, HERE / "model/focus_model.joblib")
joblib.dump(meta, HERE / "model/features_meta.joblib")
# Training table for the dashboard tab (features + rule label), rounded to keep the file small.
X.round(3).assign(label=y.to_numpy()).to_csv(HERE / "dataset/focus_features.csv", index=False)
print(f"rows {len(y)}, classes {meta['classes']}")
