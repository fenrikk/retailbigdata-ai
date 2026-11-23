import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import STUDENT
from .data_gen import generate
from .decision import ROUTE_UA, T_AUTO, T_REVIEW, decide
from .features import FEATURES, build_training_frame, load
from .model import TRAIN_END, load_models, predict, train
from .service import ForecastService

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def cmd_generate(args):
    info = generate(DATA)
    print("Згенеровано тестовий набір даних RetailBigData:")
    labels = {
        "sales": "записів продажів (день × магазин × товар)", "weather": "записів погоди",
        "products": "товарів", "stores": "магазинів", "entry_errors": "внесених помилок касира (×100)",
        "stockout_days": "днів з відсутністю товару", "missing_temp": "пропусків температури",
    }
    for k, v in info.items():
        print(f"  {labels[k]}: {v}")


def cmd_train(args):
    meta = train(DATA, MODELS)
    print(f"Навчання на даних до {meta['train_end']} включно")
    print(f"  рядків для навчання: {meta['rows']}")
    print(f"  виключено днів з відсутністю товару: {meta['excluded_stockout']}")
    print(f"  виключено підозрілих значень (помилки касира): {meta['excluded_outliers']}")
    print(f"  ознак: {len(meta['features'])}; моделі: {', '.join(meta['models'])}")
    print(f"Модель збережено: {MODELS / 'demand_model.pkl'}")


def _stock(svc, store, sku):
    st = svc.data["stock"]
    row = st[(st["store"] == store) & (st["sku"] == sku)]
    if len(row):
        return int(row["stock_on_hand"].iloc[0]), int(row["in_transit"].iloc[0])
    return 0, 0


def cmd_forecast(args):
    svc = ForecastService(DATA, MODELS)
    ctx = json.loads(args.context) if args.context else {}
    f = svc.forecast(args.store, args.sku, args.date, ctx)
    p = svc.products.loc[args.sku]
    stock, transit = (args.stock, args.transit) if args.stock is not None else _stock(svc, args.store, args.sku)
    d = decide(f, stock, transit, int(p["shelf_life_days"]), int(p["pack_size"]))
    print(f"Прогноз попиту: {p['name']} ({args.sku}), магазин {args.store}, від {f.origin} на 7 днів")
    print(f"  джерело: {f.source}")
    print(f"  по днях: {', '.join(f'{v:.1f}' for v in f.daily)}")
    print(f"  сума за 7 днів: {f.point_7d:.1f}  (80 % інтервал {f.low_7d:.1f} – {f.high_7d:.1f})")
    print(f"  впевненість: {f.confidence:.3f}")
    print(f"  прапорці: {', '.join(f.flags) if f.flags else 'немає'}")
    print(f"Рішення (Decision Engine): залишок {stock}, в дорозі {transit}")
    print(f"  маршрут: {d.route} — {ROUTE_UA[d.route]}")
    print(f"  причина: {d.reason}")
    print(f"  цільовий запас: {d.target_stock}; замовити: {d.order_qty} шт.")
    if args.json:
        print(json.dumps({"forecast": f.as_dict(), "decision": d.as_dict()}, ensure_ascii=False))


def holdout_frame():
    data = load(DATA)
    frame = build_training_frame(data)
    truth = pd.read_csv(DATA / "truth.csv", parse_dates=["date"])
    frame = frame.merge(truth[["date", "store", "sku", "true_mean"]], on=["date", "store", "sku"])
    test = frame[(frame["date"] > TRAIN_END)].dropna(subset=["lag_28", "mean_28"]).copy()
    return test


def cmd_evaluate(args):
    models = load_models(MODELS)
    test = holdout_frame()
    X = test[FEATURES].to_numpy(dtype=float)
    mean, lo, hi = predict(models, X)
    test["pred"] = mean
    test["lo"], test["hi"] = lo, hi
    test["naive"] = test["dow_mean_4w"]
    ok = test[test["train_ok"] == 1]
    y = ok["qty"].to_numpy(float)

    def wape(p, t):
        return float(np.abs(p - t).sum() / t.sum() * 100)

    res = {
        "test_rows": int(len(test)),
        "eval_rows": int(len(ok)),
        "wape_model": round(wape(ok["pred"].to_numpy(), y), 2),
        "wape_naive": round(wape(ok["naive"].to_numpy(), y), 2),
        "mae_model": round(float(np.abs(ok["pred"] - y).mean()), 3),
        "mae_naive": round(float(np.abs(ok["naive"] - y).mean()), 3),
        "bias_model": round(float((ok["pred"].sum() - y.sum()) / y.sum() * 100), 2),
        "coverage_80": round(float(((y >= ok["lo"]) & (y <= ok["hi"])).mean() * 100), 1),
        "wape_vs_true_mean": round(wape(ok["pred"].to_numpy(), ok["true_mean"].to_numpy()), 2),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "holdout_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Перевірка на відкладених даних (після {TRAIN_END} до 2026-10-31), денний рівень")
    print(f"  рядків у тесті: {res['test_rows']}, з них без відсутності товару: {res['eval_rows']}")
    print(f"  WAPE модель:            {res['wape_model']:.2f} %")
    print(f"  WAPE базове правило:    {res['wape_naive']:.2f} %   (середнє того ж дня тижня за 4 тижні)")
    print(f"  MAE модель / правило:   {res['mae_model']:.3f} / {res['mae_naive']:.3f} шт.")
    print(f"  зміщення (bias) моделі: {res['bias_model']:+.2f} %")
    print(f"  покриття 80 % інтервалу: {res['coverage_80']:.1f} %")
    print(f"  WAPE відносно істинного середнього попиту: {res['wape_vs_true_mean']:.2f} %")


def cmd_scenarios(args):
    from .scenarios import run
    svc = ForecastService(DATA, MODELS)
    truth = pd.read_csv(DATA / "truth.csv", parse_dates=["date"])
    res = run(svc, truth)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "scenarios.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'ID':<4}{'магазин / товар':<30}{'дата':<12}{'прогноз 7 дн':>13}{'істинний':>10}{'похибка':>9}{'впевн.':>8}  маршрут")
    for r in res:
        err = "—" if r["error_pct"] is None else f"{r['error_pct']:+.1f}%"
        tr = "—" if r["truth_7d"] is None else f"{r['truth_7d']:.1f}"
        print(f"{r['id']:<4}{r['store'] + ' / ' + r['sku']:<30}{r['origin']:<12}{r['point_7d']:>13.1f}{tr:>10}{err:>9}"
              f"{r['confidence']:>8.3f}  {r['route']}")
        extra = f"база без акції {r['base_7d']:.1f}; " if r["base_7d"] else ""
        verdict = "так" if r["ok"] else "ні"
        print(f"    {r['title']}; {extra}прапорці: {', '.join(r['flags']) or '—'}; коректно: {verdict}")
    good = sum(r["ok"] for r in res)
    print(f"Коректних сценаріїв: {good} з {len(res)} ({good / len(res) * 100:.0f} %)")


def cmd_backtest(args):
    from .evaluation import origins, outlier_detection, save, summarize, threshold_table, weekly_backtest
    svc = ForecastService(DATA, MODELS)
    truth = pd.read_csv(DATA / "truth.csv", parse_dates=["date"])
    bt = weekly_backtest(svc, truth)
    m, by, routes, worst = summarize(bt)
    od = outlier_detection(svc.data, truth)
    save(RESULTS, m, by, routes, worst, od, bt)
    tt = threshold_table(bt, [(0.85, 0.60), (0.80, 0.50), (0.80, 0.60), (0.75, 0.60), (0.70, 0.60), (0.65, 0.50)])
    tt.to_csv(RESULTS / "thresholds.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"Тижневий бектест: {len(origins())} тижнів × {bt[['store', 'sku']].drop_duplicates().shape[0]} пар «магазин × товар»")
    print(f"  прогнозів: {m['weeks']}; для оцінки взято {m['weeks_no_stockout']} (виключено: нестача товару "
          f"{m['weeks_excluded_stockout']}, помилки касира у факті {m['weeks_excluded_entry_error']})")
    print(f"  WAPE (тиждень, факт продажів): {m['wape_week_actual']:.2f} %")
    print(f"  WAPE (тиждень, істинний попит): {m['wape_week_truth']:.2f} %")
    print(f"  MAE: {m['mae_week']:.2f} шт.; зміщення: {m['bias_week']:+.2f} %")
    print(f"  покриття 80 % інтервалу: {m['coverage_80_week']:.1f} %")
    print(f"  частка тижнів з похибкою > 25 %: {m['share_big_error']:.1f} %")
    print(f"  рівень обслуговування (цільовий запас ≥ попиту): {m['service_level']:.1f} %")
    print("Якість за рівнем впевненості:")
    print(by.round(1).to_string(index=False))
    print(f"Маршрути при порогах {T_AUTO:.2f} / {T_REVIEW:.2f}:")
    print(routes.round(1).to_string(index=False))
    print("Варіанти порогів:")
    print(tt.to_string(index=False))
    print(f"Пошук помилок касира: знайдено {od['true_positive']} з {od['true_positive'] + od['false_negative']}, "
          f"хибних спрацювань {od['false_positive']} (точність {od['precision']} %, повнота {od['recall']} %)")


def main(argv=None):
    print(f"RetailBigData · Demand Forecast Service · {STUDENT}")
    ap = argparse.ArgumentParser(prog="python -m forecast.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    sub.add_parser("train")
    sub.add_parser("evaluate")
    sub.add_parser("scenarios")
    sub.add_parser("backtest")
    f = sub.add_parser("forecast")
    f.add_argument("--store", required=True)
    f.add_argument("--sku", required=True)
    f.add_argument("--date", default="2026-10-31")
    f.add_argument("--context")
    f.add_argument("--stock", type=int)
    f.add_argument("--transit", type=int, default=0)
    f.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        {"generate": cmd_generate, "train": cmd_train, "forecast": cmd_forecast, "evaluate": cmd_evaluate,
         "scenarios": cmd_scenarios, "backtest": cmd_backtest}[args.cmd](args)
    except ValueError as e:
        print(f"Помилка: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
