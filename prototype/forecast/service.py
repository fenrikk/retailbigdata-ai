from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (FEATURES, MIN_HISTORY_DAYS, add_weather, calendar, clean_sales, climatology, codes,
                       lag_features, load)
from .model import TRAIN_END, load_models, predict

HORIZON = 7
CAP_COLD_START = 0.40
CAP_CENSORED = 0.70
CAP_OUT_OF_RANGE = 0.60
RANGE_MARGIN = 1.0
CAP_WEATHER_SHIFT = 0.70
WEATHER_SHIFT_C = 8.0
PENALTY_TEMP_IMPUTED = 0.05
PENALTY_OUTLIER = 0.10


@dataclass
class Forecast:
    store: str
    sku: str
    origin: str
    daily: list
    point_7d: float
    low_7d: float
    high_7d: float
    confidence: float
    source: str
    flags: list = field(default_factory=list)

    def as_dict(self):
        return {
            "store": self.store, "sku": self.sku, "origin": self.origin,
            "daily": self.daily, "point_7d": self.point_7d, "low_7d": self.low_7d,
            "high_7d": self.high_7d, "confidence": self.confidence,
            "source": self.source, "flags": self.flags,
        }


class ForecastService:
    def __init__(self, data_dir, model_dir, models=None):
        self.data = load(data_dir)
        self.clim = climatology(self.data["weather"])
        self.model_dir = Path(model_dir)
        self.models = models
        if self.models is None:
            try:
                self.models = load_models(self.model_dir)
            except (FileNotFoundError, OSError):
                self.models = None
        self.products = self.data["products"].set_index("sku")
        self.city = dict(zip(self.data["stores"]["store"], self.data["stores"]["city"]))
        w = self.data["weather"]
        seen = w[w["date"] <= TRAIN_END].groupby("city")["temp_c"]
        self.temp_range = {c: (lo, hi) for c, lo, hi in zip(seen.min().index, seen.min().values, seen.max().values)}

    def _series(self, store, sku, origin):
        s = self.data["sales"]
        return s[(s["store"] == store) & (s["sku"] == sku) & (s["date"] <= origin)].copy()

    def _future(self, store, sku, origin, context):
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        base_price = float(self.products.loc[sku, "base_price"])
        plan = self.data.get("promo_plan", {})
        disc = context.get("discount", [float(plan.get((store, sku, d), 0.0)) for d in dates])
        if np.isscalar(disc):
            disc = [float(disc)] * HORIZON
        return pd.DataFrame({
            "date": dates, "store": store, "city": self.city[store], "sku": sku,
            "price": [round(base_price * (1 - d), 2) for d in disc], "discount": disc,
            "qty": np.nan, "stockout": 0,
        })

    def _features(self, hist, fut, context):
        frame = pd.concat([hist, fut], ignore_index=True)
        frame = calendar(frame, self.data["holidays"])
        frame = clean_sales(frame)
        weather = self.data["weather"]
        if "temp_c" in context:
            temps = context["temp_c"]
            if np.isscalar(temps):
                temps = [temps] * HORIZON
            override = pd.DataFrame({"date": fut["date"], "city": fut["city"].iloc[0], "temp_c": temps})
            weather = pd.concat([weather[~weather.set_index(["date", "city"]).index.isin(
                override.set_index(["date", "city"]).index)], override], ignore_index=True)
        if context.get("weather_unavailable"):
            weather = weather[~weather["date"].isin(fut["date"])]
        frame = frame.drop(columns=["city"]).assign(city=self.city[frame["store"].iloc[0]])
        frame = add_weather(frame, weather, self.clim)
        frame = codes(frame, self.data["products"], self.data["stores"])
        frame = frame.sort_values("date").reset_index(drop=True)
        return lag_features(frame)

    def _fallback(self, hist, origin, store, sku, flags):
        recent = hist[hist["date"] > origin - pd.Timedelta(days=28)]
        qty = recent["qty"].clip(upper=recent["qty"].median() * 5 + 10) if len(recent) else pd.Series([0.0])
        daily_mean = float(qty.mean()) if len(qty) else 0.0
        point = round(daily_mean * HORIZON, 1)
        return Forecast(store, sku, origin.strftime("%Y-%m-%d"), [round(daily_mean, 2)] * HORIZON, point,
                        round(point * 0.6, 1), round(point * 1.4, 1), 0.0, "fallback_rule", flags + ["model_unavailable"])

    def forecast(self, store, sku, origin, context=None):
        context = context or {}
        if store not in self.city:
            raise ValueError(f"невідомий магазин: {store}")
        if sku not in self.products.index:
            raise ValueError(f"невідомий товар: {sku}")
        origin = pd.Timestamp(origin)
        hist = self._series(store, sku, origin)
        flags = []
        if self.models is None or context.get("model_down"):
            return self._fallback(hist, origin, store, sku, flags)
        fut = self._future(store, sku, origin, context)
        frame = self._features(hist, fut, context)
        rows = frame[frame["date"] > origin]
        X = rows[FEATURES].to_numpy(dtype=float)
        mean, lo, hi = predict(self.models, X)
        point = float(mean.sum())
        low = point - float(np.sqrt(((mean - lo) ** 2).sum()))
        high = point + float(np.sqrt(((hi - mean) ** 2).sum()))
        rel = (high - low) / (2 * max(point, 1.0))
        conf = float(np.clip(1.0 - rel, 0.0, 1.0))
        history_days = len(hist)
        if history_days < MIN_HISTORY_DAYS:
            flags.append("cold_start")
            conf = min(conf, CAP_COLD_START)
        last28 = hist[hist["date"] > origin - pd.Timedelta(days=28)]
        if last28["stockout"].sum() >= 3:
            flags.append("censored_demand")
            conf = min(conf, CAP_CENSORED)
        hist_clean = frame[frame["date"] <= origin]
        if hist_clean[hist_clean["date"] > origin - pd.Timedelta(days=28)]["outlier"].sum() > 0:
            flags.append("outlier_in_history")
            conf -= PENALTY_OUTLIER
        lo_t, hi_t = self.temp_range.get(self.city[store], (-99, 99))
        if ((rows["temp_c"] > hi_t + RANGE_MARGIN) | (rows["temp_c"] < lo_t - RANGE_MARGIN)).any():
            flags.append("out_of_range")
            conf = min(conf, CAP_OUT_OF_RANGE)
        past = frame[(frame["date"] <= origin) & (frame["date"] > origin - pd.Timedelta(days=14))]["temp_c"]
        if len(past) and abs(float(rows["temp_c"].mean()) - float(past.mean())) > WEATHER_SHIFT_C:
            flags.append("weather_shift")
            conf = min(conf, CAP_WEATHER_SHIFT)
        if rows["temp_imputed"].any():
            flags.append("weather_imputed")
            conf -= PENALTY_TEMP_IMPUTED
        conf = round(max(conf, 0.0), 3)
        return Forecast(store, sku, origin.strftime("%Y-%m-%d"), [round(float(v), 2) for v in mean],
                        round(point, 1), round(max(low, 0.0), 1), round(high, 1), conf, "ml_model", flags)
