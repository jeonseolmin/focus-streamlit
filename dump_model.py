"""Refit the shipped ML1 model (LightGBM) in the devdesk repo and dump it with joblib.

Usage (from the devdesk folder, in its ml environment):
    uv run --group ml python ../focus-streamlit/dump_model.py
Writes models/focus_lgbm.pkl next to this file.
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path.cwd()))  # the devdesk repo: ml.train_focus holds the data loading and model settings
from ml import behacom
from ml.train_focus import ROOT, make_dataset, models

X, y, _ = make_dataset(behacom.load(ROOT / "data/raw/behacom/behacom_slim.csv"), behacom.TZ)
model = models()["lightgbm"]().fit(X, y)
out = Path(__file__).resolve().parent / "models" / "focus_lgbm.pkl"
out.parent.mkdir(exist_ok=True)
joblib.dump(model, out)
print(f"saved {out} ({out.stat().st_size // 1024} KB), rows {len(y)}, classes {list(model.classes_)}")
