from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "store_code", "sku_code", "cat_code", "dow", "month", "doy_sin", "doy_cos",
    "holiday", "pre_holiday", "temp_c", "discount", "price_ratio",
    "lag_7", "lag_14", "lag_21", "lag_28", "mean_7", "mean_28", "dow_mean_4w",
    "stockout_28", "shelf_life_days",
]
MIN_HISTORY_DAYS = 56
OUTLIER_FACTOR = 5.0


def promo_plan(data_dir):
    path = Path(data_dir) / "promo_plan.csv"
    if not path.exists():
        return {}
    plan = pd.read_csv(path, parse_dates=["date"])
    return {(r.store, r.sku, r.date): r.discount for r in plan.itertuples()}


def load(data_dir):
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["date"])
    weather = pd.read_csv(data_dir / "weather.csv", parse_dates=["date"])
    products = pd.read_csv(data_dir / "products.csv")
    stores = pd.read_csv(data_dir / "stores.csv")
    hol = pd.read_csv(data_dir / "holidays.csv", parse_dates=["date"])
    stock = pd.read_csv(data_dir / "stock.csv")
    return {"sales": sales, "weather": weather, "products": products, "stores": stores, "holidays": hol, "stock": stock,
            "promo_plan": promo_plan(data_dir)}


def climatology(weather):
    w = weather.dropna(subset=["temp_c"]).copy()
    w["doy"] = w["date"].dt.day_of_year
    return w.groupby(["city", "doy"])["temp_c"].mean()


def clean_sales(sales):
    s = sales.sort_values(["store", "sku", "date"]).copy()
    med = s.groupby(["store", "sku"])["qty"].transform(lambda x: x.rolling(28, min_periods=7, center=True).median())
    protected = (s["discount"] > 0)
    if "holiday" in s:
        protected |= (s["holiday"] > 0) | (s["pre_holiday"] > 0)
    s["outlier"] = ((s["qty"] > OUTLIER_FACTOR * med.clip(lower=1) + 10) & ~protected).astype(int)
    s["qty_clean"] = np.where(s["outlier"] == 1, med.round(), s["qty"])
    s["train_ok"] = ((s["outlier"] == 0) & (s["stockout"] == 0)).astype(int)
    return s


def calendar(frame, hol):
    weights = dict(zip(hol["date"], hol["weight"]))
    frame["dow"] = frame["date"].dt.day_of_week
    frame["month"] = frame["date"].dt.month
    doy = frame["date"].dt.day_of_year
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    frame["holiday"] = frame["date"].map(weights).fillna(0.0)
    frame["pre_holiday"] = (frame["date"] + pd.Timedelta(days=1)).map(weights).fillna(0.0)
    return frame


def add_weather(frame, weather, clim):
    f = frame.merge(weather, on=["date", "city"], how="left")
    miss = f["temp_c"].isna()
    if miss.any():
        keys = list(zip(f.loc[miss, "city"], f.loc[miss, "date"].dt.day_of_year))
        f.loc[miss, "temp_c"] = [clim.get(k, 10.0) for k in keys]
    f["temp_imputed"] = miss.astype(int)
    return f


def codes(frame, products, stores):
    store_codes = {s: i for i, s in enumerate(stores["store"])}
    sku_codes = {s: i for i, s in enumerate(products["sku"])}
    cats = {c: i for i, c in enumerate(sorted(products["category"].unique()))}
    p = products.set_index("sku")
    frame["store_code"] = frame["store"].map(store_codes)
    frame["sku_code"] = frame["sku"].map(sku_codes)
    frame["cat_code"] = frame["sku"].map(p["category"]).map(cats)
    frame["shelf_life_days"] = frame["sku"].map(p["shelf_life_days"])
    frame["price_ratio"] = frame["price"] / frame["sku"].map(p["base_price"])
    return frame


def lag_features(s):
    g = s.groupby(["store", "sku"])["qty_clean"]
    for k in (7, 14, 21, 28):
        s[f"lag_{k}"] = g.shift(k)
    shifted = g.shift(7)
    s["mean_7"] = shifted.groupby([s["store"], s["sku"]]).transform(lambda x: x.rolling(7, min_periods=3).mean())
    s["mean_28"] = shifted.groupby([s["store"], s["sku"]]).transform(lambda x: x.rolling(28, min_periods=7).mean())
    s["dow_mean_4w"] = s[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
    so = s.groupby(["store", "sku"])["stockout"].shift(7)
    s["stockout_28"] = so.groupby([s["store"], s["sku"]]).transform(lambda x: x.rolling(28, min_periods=1).sum())
    s["history_days"] = s.groupby(["store", "sku"]).cumcount()
    return s


def build_training_frame(data):
    sales = calendar(data["sales"].copy(), data["holidays"])
    sales = clean_sales(sales)
    sales = sales.drop(columns=["city"]).merge(data["stores"][["store", "city"]], on="store")
    sales = add_weather(sales, data["weather"], climatology(data["weather"]))
    sales = codes(sales, data["products"], data["stores"])
    sales = sales.sort_values(["store", "sku", "date"]).reset_index(drop=True)
    sales = lag_features(sales)
    return sales
