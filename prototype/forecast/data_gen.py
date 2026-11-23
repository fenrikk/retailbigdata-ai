from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

START = date(2024, 11, 1)
END = date(2026, 10, 31)
SEED = 2026

STORES = [
    ("Київ-017", "Київ", 1.35),
    ("Київ-022", "Київ", 1.00),
    ("Львів-003", "Львів", 1.10),
    ("Одеса-009", "Одеса", 0.95),
    ("Харків-005", "Харків", 0.85),
    ("Вінниця-002", "Вінниця", 0.70),
]

CITY_CLIMATE = {
    "Київ": (8.5, 13.0),
    "Львів": (8.0, 11.5),
    "Одеса": (11.0, 11.5),
    "Харків": (8.0, 14.0),
    "Вінниця": (8.0, 12.5),
}

PRODUCTS = [
    ("MILK-1L", "Молоко 2,5 %, 1 л", "Молочні продукти", 42.9, 60, 0.00, 1.25, 1.6, 7, 12, "daily"),
    ("KEFIR-1L", "Кефір 1 %, 1 л", "Молочні продукти", 39.5, 28, 0.04, 1.15, 1.5, 7, 12, "daily"),
    ("CHEESE-HARD", "Сир твердий, 200 г", "Молочні продукти", 98.0, 22, -0.02, 1.45, 1.8, 30, 10, "weekend"),
    ("YOGURT-CUP", "Йогурт полуничний, 300 г", "Молочні продукти", 31.0, 30, 0.05, 1.10, 1.7, 14, 12, "daily"),
    ("BREAD-RYE", "Хліб житній", "Хліб", 34.5, 55, -0.01, 1.20, 1.3, 3, 1, "daily"),
    ("BAGUETTE", "Багет французький", "Хліб", 29.0, 26, 0.00, 1.35, 1.4, 2, 1, "weekend"),
    ("EGGS-10", "Яйця курячі, 10 шт.", "Бакалія", 64.0, 34, -0.02, 1.60, 1.5, 21, 10, "weekend"),
    ("BUCKWHEAT-1KG", "Крупа гречана, 1 кг", "Бакалія", 58.0, 14, -0.05, 1.10, 2.2, 365, 10, "flat"),
    ("PASTA-500", "Макарони, 500 г", "Бакалія", 36.0, 18, -0.03, 1.05, 2.0, 365, 20, "flat"),
    ("SUNOIL-1L", "Олія соняшникова, 1 л", "Бакалія", 79.0, 16, -0.02, 1.40, 2.4, 365, 12, "flat"),
    ("SUGAR-1KG", "Цукор, 1 кг", "Бакалія", 33.0, 12, 0.02, 1.30, 1.8, 365, 10, "flat"),
    ("COFFEE-250", "Кава мелена, 250 г", "Напої", 189.0, 9, -0.03, 1.25, 2.5, 365, 6, "weekend"),
    ("TEA-100", "Чай чорний, 100 пак.", "Напої", 119.0, 7, -0.06, 1.20, 2.0, 365, 6, "flat"),
    ("WATER-1.5L", "Вода мінеральна, 1,5 л", "Напої", 21.0, 40, 0.45, 1.10, 1.4, 365, 6, "weekend"),
    ("JUICE-1L", "Сік апельсиновий, 1 л", "Напої", 54.0, 15, 0.12, 1.50, 1.9, 180, 12, "weekend"),
    ("SPARKLING-075", "Вино ігристе, 0,75 л", "Напої", 249.0, 4, 0.00, 6.00, 2.0, 365, 6, "weekend"),
    ("ICECREAM-80", "Морозиво пломбір, 80 г", "Заморожені продукти", 25.0, 18, 0.95, 1.30, 1.6, 180, 24, "weekend"),
    ("DUMPLINGS-800", "Пельмені, 800 г", "Заморожені продукти", 129.0, 10, -0.18, 1.30, 1.9, 120, 8, "flat"),
    ("CHICKEN-FILLET", "Філе куряче, 1 кг", "М'ясо", 189.0, 13, -0.04, 1.55, 1.7, 5, 1, "weekend"),
    ("SAUSAGE-BOIL", "Ковбаса варена, 500 г", "М'ясо", 139.0, 11, -0.03, 1.70, 1.6, 10, 5, "weekend"),
    ("APPLES-1KG", "Яблука, 1 кг", "Овочі та фрукти", 39.0, 32, 0.00, 1.20, 1.4, 14, 1, "daily"),
    ("BANANAS-1KG", "Банани, 1 кг", "Овочі та фрукти", 69.0, 27, 0.03, 1.15, 1.5, 6, 1, "daily"),
    ("TANGERINES-1KG", "Мандарини, 1 кг", "Овочі та фрукти", 89.0, 14, -0.10, 2.80, 1.8, 14, 1, "weekend"),
    ("CHIPS-130", "Чипси картопляні, 130 г", "Снеки", 59.0, 12, 0.10, 1.80, 2.0, 120, 14, "weekend"),
]

NEW_PRODUCTS = [
    ("KOMBUCHA-033", "Комбуча імбирна, 0,33 л", "Напої", 65.0, 6, 0.30, 1.20, 2.2, 90, 12, "weekend"),
    ("PROTEIN-BAR", "Протеїновий батончик, 60 г", "Снеки", 49.0, 8, 0.00, 1.10, 1.8, 180, 20, "flat"),
]
NEW_PRODUCT_LAUNCH = date(2026, 10, 1)

WEEKDAY = {
    "daily": [1.00, 0.96, 0.97, 1.00, 1.08, 1.18, 0.90],
    "weekend": [0.85, 0.82, 0.86, 0.92, 1.10, 1.40, 1.15],
    "flat": [1.02, 0.98, 0.98, 1.00, 1.04, 1.06, 0.92],
}


def easter(year):
    a = year % 19
    b = year % 4
    c = year % 7
    d = (19 * a + 15) % 30
    e = (2 * b + 4 * c + 6 * d + 6) % 7
    return date(year, 3, 22) + timedelta(days=d + e + 13)


def holidays():
    days = {}
    for y in (2024, 2025, 2026, 2027):
        for m, d, k in ((1, 1, 1.0), (3, 8, 0.8), (8, 24, 0.6), (12, 25, 0.9), (12, 31, 1.0), (10, 1, 0.4)):
            days[date(y, m, d)] = k
        days[easter(y)] = 0.9
    return days


def _season_factor(category, doy):
    s = np.cos(2 * np.pi * (doy - 15) / 365.25)
    if category == "Овочі та фрукти":
        return 1.0 + 0.10 * s
    if category == "Заморожені продукти":
        return 1.0 - 0.05 * s
    return 1.0 + 0.04 * s


def generate(out_dir, seed=SEED):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(START, END, freq="D")
    hol = holidays()

    weather_rows = []
    for city, (mean, amp) in CITY_CLIMATE.items():
        noise = np.zeros(len(days))
        for i in range(1, len(days)):
            noise[i] = 0.75 * noise[i - 1] + rng.normal(0, 2.0)
        doy = days.dayofyear.values
        temp = mean - amp * np.cos(2 * np.pi * (doy - 15) / 365.25) + noise
        heat = (days >= "2026-07-14") & (days <= "2026-07-24")
        temp = np.where(heat, temp + 7.0, temp)
        for d, t in zip(days, temp):
            weather_rows.append((d.date(), city, round(float(t), 1)))
    weather = pd.DataFrame(weather_rows, columns=["date", "city", "temp_c"])
    wmap = {(r.date, r.city): r.temp_c for r in weather.itertuples()}

    products = PRODUCTS + NEW_PRODUCTS
    rows = []
    for sku, name, cat, price, base, tcoef, hcoef, plift, shelf, pack, wtype in products:
        launch = NEW_PRODUCT_LAUNCH if (sku, name) in [(p[0], p[1]) for p in NEW_PRODUCTS] else START
        for store, city, size in STORES:
            store_mult = size * rng.uniform(0.9, 1.1)
            promo_days = set()
            d0 = START
            while d0 <= END:
                if rng.random() < 0.09:
                    length = int(rng.integers(5, 9))
                    disc = float(rng.choice([0.15, 0.20, 0.25, 0.30]))
                    for k in range(length):
                        promo_days.add((d0 + timedelta(days=k), disc))
                    d0 += timedelta(days=length + 14)
                else:
                    d0 += timedelta(days=7)
            promo_map = dict(promo_days)
            stockout_left = 0
            for d in days:
                dd = d.date()
                if dd < launch:
                    continue
                temp = wmap[(dd, city)]
                doy = d.dayofyear
                dow = d.dayofweek
                h = hol.get(dd, 0.0)
                pre = hol.get(dd + timedelta(days=1), 0.0)
                hol_mult = 1.0
                if cat in ("Напої", "М'ясо", "Молочні продукти", "Овочі та фрукти", "Снеки"):
                    hol_mult += (hcoef - 1.0) * max(h * 0.6, pre)
                if sku == "TANGERINES-1KG" and dd.month == 12 and dd.day >= 15:
                    hol_mult *= 1.6
                if sku == "SPARKLING-075" and dd.month == 12 and dd.day >= 27:
                    hol_mult *= 2.5
                disc = promo_map.get(dd, 0.0)
                promo_mult = 1.0 + (plift - 1.0) * (disc / 0.25) if disc else 1.0
                temp_mult = float(np.exp(tcoef * (temp - 15.0) / 10.0))
                ramp = 1.0
                if dd >= NEW_PRODUCT_LAUNCH and launch == NEW_PRODUCT_LAUNCH:
                    ramp = min(1.0, 0.4 + 0.03 * (dd - launch).days)
                lam = base * store_mult * WEEKDAY[wtype][dow] * _season_factor(cat, doy) * temp_mult * hol_mult * promo_mult * ramp
                demand = int(rng.negative_binomial(8, 8 / (8 + lam)))
                if stockout_left == 0 and rng.random() < 0.006:
                    stockout_left = int(rng.integers(1, 4))
                stockout = 0
                sold = demand
                if stockout_left > 0:
                    sold = int(demand * rng.uniform(0.15, 0.6))
                    stockout = 1
                    stockout_left -= 1
                rows.append((dd, store, city, sku, round(price * (1 - disc), 2), disc, sold, stockout, round(lam, 3)))

    sales = pd.DataFrame(rows, columns=["date", "store", "city", "sku", "price", "discount", "qty", "stockout", "true_mean"])
    err_idx = rng.choice(len(sales), size=max(1, len(sales) // 2500), replace=False)
    sales.loc[err_idx, "qty"] = sales.loc[err_idx, "qty"] * 100
    sales["entry_error"] = 0
    sales.loc[err_idx, "entry_error"] = 1

    miss = rng.choice(len(weather), size=len(weather) // 200, replace=False)
    weather.loc[miss, "temp_c"] = np.nan

    stock_rows = []
    last = sales[sales["date"] == END]
    for r in last.itertuples():
        mean7 = r.true_mean * 7
        stock_rows.append((r.store, r.sku, int(rng.uniform(0.1, 0.6) * mean7), int(rng.uniform(0.0, 0.3) * mean7)))
    stock = pd.DataFrame(stock_rows, columns=["store", "sku", "stock_on_hand", "in_transit"])

    prod = pd.DataFrame(
        [(p[0], p[1], p[2], p[3], p[8], p[9]) for p in products],
        columns=["sku", "name", "category", "base_price", "shelf_life_days", "pack_size"],
    )
    stores = pd.DataFrame(STORES, columns=["store", "city", "size_factor"])
    hol_df = pd.DataFrame(sorted(hol.items()), columns=["date", "weight"])

    truth = sales[["date", "store", "sku", "true_mean", "entry_error"]]
    sales[sales["discount"] > 0][["date", "store", "sku", "discount"]].to_csv(out_dir / "promo_plan.csv", index=False)
    sales.drop(columns=["true_mean", "entry_error"]).to_csv(out_dir / "sales.csv", index=False)
    truth.to_csv(out_dir / "truth.csv", index=False)
    weather.to_csv(out_dir / "weather.csv", index=False)
    stock.to_csv(out_dir / "stock.csv", index=False)
    prod.to_csv(out_dir / "products.csv", index=False)
    stores.to_csv(out_dir / "stores.csv", index=False)
    hol_df.to_csv(out_dir / "holidays.csv", index=False)
    return {
        "sales": len(sales),
        "weather": len(weather),
        "products": len(prod),
        "stores": len(stores),
        "entry_errors": int(len(err_idx)),
        "stockout_days": int(sales["stockout"].sum()),
        "missing_temp": int(weather["temp_c"].isna().sum()),
    }
