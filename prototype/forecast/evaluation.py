import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decision import decide
from .features import build_training_frame
from .model import TRAIN_END

FIRST_ORIGIN = pd.Timestamp(TRAIN_END)
LAST_DAY = pd.Timestamp("2026-10-31")
BUCKETS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 1.01]
BIG_ERROR = 0.25


def origins():
    out = []
    o = FIRST_ORIGIN
    while o + pd.Timedelta(days=7) <= LAST_DAY:
        out.append(o)
        o += pd.Timedelta(days=7)
    return out


def weekly_backtest(svc, truth, limit=None):
    sales = svc.data["sales"]
    pairs = sales[["store", "sku"]].drop_duplicates().itertuples(index=False)
    rows = []
    for store, sku in pairs:
        p = svc.products.loc[sku]
        for o in origins():
            week = sales[(sales["store"] == store) & (sales["sku"] == sku) & (sales["date"] > o) & (sales["date"] <= o + pd.Timedelta(days=7))]
            if len(week) < 7:
                continue
            tw = truth[(truth["store"] == store) & (truth["sku"] == sku) & (truth["date"] > o) & (truth["date"] <= o + pd.Timedelta(days=7))]
            f = svc.forecast(store, sku, o)
            d = decide(f, 0, 0, int(p["shelf_life_days"]), int(p["pack_size"]))
            rows.append({
                "store": store, "sku": sku, "origin": o.strftime("%Y-%m-%d"),
                "actual": float(week["qty"].sum()), "stockout_days": int(week["stockout"].sum()),
                "entry_errors": int(tw["entry_error"].sum()),
                "truth": float(tw["true_mean"].sum()), "point": f.point_7d, "low": f.low_7d, "high": f.high_7d,
                "confidence": f.confidence, "flags": ",".join(f.flags), "route": d.route, "target": d.target_stock,
            })
            if limit and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def summarize(bt):
    ok = bt[(bt["stockout_days"] == 0) & (bt["entry_errors"] == 0)].copy()
    ok["abs_err"] = (ok["point"] - ok["actual"]).abs()
    ok["rel_err"] = ok["abs_err"] / ok["actual"].clip(lower=1)
    bt = bt.copy()
    bt["abs_err_truth"] = (bt["point"] - bt["truth"]).abs()

    def wape(df, col="actual"):
        return float((df["point"] - df[col]).abs().sum() / df[col].sum() * 100)

    m = {
        "weeks": int(len(bt)),
        "weeks_no_stockout": int(len(ok)),
        "weeks_excluded_stockout": int((bt["stockout_days"] > 0).sum()),
        "weeks_excluded_entry_error": int(((bt["entry_errors"] > 0) & (bt["stockout_days"] == 0)).sum()),
        "wape_week_actual": round(wape(ok), 2),
        "wape_week_truth": round(wape(bt, "truth"), 2),
        "mae_week": round(float(ok["abs_err"].mean()), 2),
        "bias_week": round(float((ok["point"].sum() - ok["actual"].sum()) / ok["actual"].sum() * 100), 2),
        "coverage_80_week": round(float(((ok["actual"] >= ok["low"]) & (ok["actual"] <= ok["high"])).mean() * 100), 1),
        "share_big_error": round(float((ok["rel_err"] > BIG_ERROR).mean() * 100), 1),
        "service_level": round(float((ok["target"] >= ok["actual"]).mean() * 100), 1),
    }
    ok["bucket"] = pd.cut(ok["confidence"], BUCKETS, right=False)
    by = ok.groupby("bucket", observed=True).apply(lambda g: pd.Series({
        "weeks": len(g),
        "wape": (g["point"] - g["actual"]).abs().sum() / g["actual"].sum() * 100,
        "big_error_pct": (g["rel_err"] > BIG_ERROR).mean() * 100,
        "service_level": (g["target"] >= g["actual"]).mean() * 100,
    }), include_groups=False).reset_index()
    by["bucket"] = by["bucket"].astype(str)
    routes = ok.groupby("route").apply(lambda g: pd.Series({
        "weeks": len(g),
        "share": len(g) / len(ok) * 100,
        "wape": (g["point"] - g["actual"]).abs().sum() / g["actual"].sum() * 100,
        "big_error_pct": (g["rel_err"] > BIG_ERROR).mean() * 100,
    }), include_groups=False).reset_index()
    worst = ok.sort_values("rel_err", ascending=False).head(8)
    return m, by, routes, worst


def outlier_detection(data, truth):
    frame = build_training_frame(data)
    f = frame.merge(truth[["date", "store", "sku", "entry_error"]], on=["date", "store", "sku"])
    tp = int(((f["outlier"] == 1) & (f["entry_error"] == 1)).sum())
    fp = int(((f["outlier"] == 1) & (f["entry_error"] == 0)).sum())
    fn = int(((f["outlier"] == 0) & (f["entry_error"] == 1)).sum())
    fp_rows = f[(f["outlier"] == 1) & (f["entry_error"] == 0)]
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": round(tp / max(tp + fp, 1) * 100, 1), "recall": round(tp / max(tp + fn, 1) * 100, 1),
        "fp_top_sku": fp_rows["sku"].value_counts().head(4).to_dict(),
    }


def save(results_dir, m, by, routes, worst, od, bt):
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    bt.to_csv(results_dir / "weekly_backtest.csv", index=False)
    by.to_csv(results_dir / "confidence_buckets.csv", index=False)
    routes.to_csv(results_dir / "routes.csv", index=False)
    worst.to_csv(results_dir / "worst_weeks.csv", index=False)
    (results_dir / "weekly_metrics.json").write_text(json.dumps({"metrics": m, "outliers": od}, ensure_ascii=False, indent=2), encoding="utf-8")


def threshold_table(bt, pairs):
    ok = bt[(bt["stockout_days"] == 0) & (bt["entry_errors"] == 0)].copy()
    ok["rel_err"] = (ok["point"] - ok["actual"]).abs() / ok["actual"].clip(lower=1)
    rows = []
    for t_auto, t_review in pairs:
        cold = ok["flags"].fillna("").str.contains("cold_start")
        auto = (ok["confidence"] >= t_auto) & ~cold
        review = (ok["confidence"] >= t_review) & ~auto
        a = ok[auto]
        rows.append({
            "t_auto": t_auto, "t_review": t_review,
            "auto_share": round(auto.mean() * 100, 1),
            "review_share": round(review.mean() * 100, 1),
            "reject_share": round((~auto & ~review).mean() * 100, 1),
            "auto_wape": round(float((a["point"] - a["actual"]).abs().sum() / max(a["actual"].sum(), 1) * 100), 1),
            "auto_big_error": round(float((a["rel_err"] > BIG_ERROR).mean() * 100), 1) if len(a) else 0.0,
            "auto_service_level": round(float((a["target"] >= a["actual"]).mean() * 100), 1) if len(a) else 0.0,
        })
    return pd.DataFrame(rows)
