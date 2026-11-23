import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES, build_training_frame, load

TRAIN_END = "2026-06-30"
QUANTILES = (0.1, 0.9)
CATEGORICAL = ["store_code", "sku_code", "cat_code", "dow"]


def _regressor(**kw):
    cat_mask = [f in CATEGORICAL for f in FEATURES]
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=cat_mask, random_state=7, **kw
    )


def training_rows(frame, end=TRAIN_END):
    rows = frame[(frame["date"] <= end) & (frame["train_ok"] == 1) & (frame["history_days"] >= 35)]
    return rows.dropna(subset=["lag_28", "mean_28"])


def train(data_dir, model_dir):
    data = load(data_dir)
    frame = build_training_frame(data)
    rows = training_rows(frame)
    X = rows[FEATURES].to_numpy(dtype=float)
    y = rows["qty_clean"].to_numpy(dtype=float)
    models = {"mean": _regressor(loss="poisson").fit(X, y)}
    for q in QUANTILES:
        models[f"q{int(q * 100)}"] = _regressor(loss="quantile", quantile=q).fit(X, y)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "demand_model.pkl", "wb") as f:
        pickle.dump(models, f)
    meta = {
        "train_end": TRAIN_END,
        "rows": int(len(rows)),
        "excluded_stockout": int(((frame["date"] <= TRAIN_END) & (frame["stockout"] == 1)).sum()),
        "excluded_outliers": int(((frame["date"] <= TRAIN_END) & (frame["outlier"] == 1)).sum()),
        "features": FEATURES,
        "models": list(models),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_models(model_dir):
    with open(Path(model_dir) / "demand_model.pkl", "rb") as f:
        return pickle.load(f)


def predict(models, X):
    mean = np.clip(models["mean"].predict(X), 0, None)
    lo = np.clip(models["q10"].predict(X), 0, None)
    hi = np.clip(models["q90"].predict(X), 0, None)
    lo = np.minimum(lo, mean)
    hi = np.maximum(hi, mean)
    return mean, lo, hi
