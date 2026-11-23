from dataclasses import dataclass, field

import pandas as pd

from .decision import decide


@dataclass
class Scenario:
    sid: str
    title: str
    store: str
    sku: str
    origin: str
    context: dict = field(default_factory=dict)
    tol: float = None
    route: str = None
    not_route: str = None
    flag: str = None
    uplift: float = None
    max_conf: float = None
    stock: tuple = None
    avg_weekly_order: float = None
    source: str = None
    expected: str = ""


SCENARIOS = [
    Scenario("T1", "Звичайний тиждень, щоденний товар", "Київ-017", "MILK-1L", "2026-10-24",
             tol=0.15, route="auto_order",
             expected="сума ±15 % від істинного попиту; автоматичне замовлення"),
    Scenario("T2", "Спека 14–24.07 (+7 °C), морозиво", "Одеса-009", "ICECREAM-80", "2026-07-13",
             tol=0.20, expected="ріст прогнозу разом зі спекою; сума ±20 % від істинного попиту"),
    Scenario("T3", "Запланована акція −25 % на 7 днів", "Київ-022", "MILK-1L", "2026-10-24",
             context={"discount": 0.25}, uplift=1.30,
             expected="прогноз з акцією щонайменше на 30 % вищий, ніж без акції"),
    Scenario("T4", "Новий товар (продається 24 дні)", "Київ-017", "KOMBUCHA-033", "2026-10-24",
             flag="cold_start", not_route="auto_order",
             expected="прапорець cold_start; без автоматичного замовлення"),
    Scenario("T5", "Компонент ШІ недоступний", "Київ-017", "MILK-1L", "2026-10-24",
             context={"model_down": True}, source="fallback_rule", route="reject_ai",
             expected="резервне правило (середнє за 28 днів); рішення за менеджером"),
    Scenario("T6", "Тиждень зі святом 24.08, хліб", "Львів-003", "BREAD-RYE", "2026-08-17",
             tol=0.15, expected="сума ±15 % від істинного попиту"),
    Scenario("T7", "Сервіс погоди недоступний, вода", "Харків-005", "WATER-1.5L", "2026-08-02",
             context={"weather_unavailable": True}, flag="weather_imputed", tol=0.20,
             expected="кліматична норма замість прогнозу погоди, прапорець weather_imputed; ±20 %"),
    Scenario("T8", "Товар з малими продажами (ігристе)", "Вінниця-002", "SPARKLING-075", "2026-09-12",
             tol=0.25, expected="сума ±25 % від істинного попиту"),
    Scenario("T9", "Нестача товару 5 днів за 28 днів", "Київ-022", "TEA-100", "2026-08-23",
             flag="censored_demand", max_conf=0.70, not_route="auto_order",
             expected="прапорець censored_demand, впевненість ≤ 0,70, без автозамовлення"),
    Scenario("T10", "Помилка касира 06.09 (×100) в історії", "Київ-017", "PASTA-500", "2026-09-10",
             flag="outlier_in_history", tol=0.15,
             expected="помилку замінено медіаною, прапорець outlier_in_history; ±15 %"),
    Scenario("T11", "Замовлення понад ліміт (середнє 150)", "Київ-017", "MILK-1L", "2026-10-24",
             avg_weekly_order=150, route="manager_review",
             expected="замовлення > 3 × середнього → перевірка менеджером"),
    Scenario("T12", "Сезонний товар (мандарини, жовтень)", "Харків-005", "TANGERINES-1KG", "2026-10-10",
             tol=0.20, expected="сума ±20 % від істинного попиту"),
    Scenario("T13", "Тиждень зі святом 01.10, сік", "Київ-017", "JUICE-1L", "2026-09-27",
             tol=0.15, expected="сума ±15 % від істинного попиту"),
    Scenario("T14", "Новий товар (продається 24 дні), батончик", "Одеса-009", "PROTEIN-BAR", "2026-10-24",
             tol=0.20, not_route="auto_order", expected="без автозамовлення; сума ±20 % від істинного попиту"),
    Scenario("T15", "Спека 14–24.07, вода", "Київ-017", "WATER-1.5L", "2026-07-13",
             tol=0.20, expected="сума ±20 % від істинного попиту (31 °C уже траплялось у Києві в навчальних даних)"),
    Scenario("T16", "Різке похолодання (з ≈16 до 1–7 °C), морозиво", "Харків-005", "ICECREAM-80", "2026-09-22",
             flag="weather_shift", not_route="auto_order",
             expected="прапорець weather_shift, без автоматичного замовлення"),
]


def truth_7d(truth, store, sku, origin):
    o = pd.Timestamp(origin)
    t = truth[(truth["store"] == store) & (truth["sku"] == sku) & (truth["date"] > o) & (truth["date"] <= o + pd.Timedelta(days=7))]
    return float(t["true_mean"].sum()) if len(t) == 7 else None


def run(svc, truth, scenarios=SCENARIOS):
    out = []
    for sc in scenarios:
        f = svc.forecast(sc.store, sc.sku, sc.origin, sc.context)
        p = svc.products.loc[sc.sku]
        if sc.stock:
            stock, transit = sc.stock
        else:
            st = svc.data["stock"]
            row = st[(st["store"] == sc.store) & (st["sku"] == sc.sku)]
            stock, transit = (int(row["stock_on_hand"].iloc[0]), int(row["in_transit"].iloc[0])) if len(row) else (0, 0)
        d = decide(f, stock, transit, int(p["shelf_life_days"]), int(p["pack_size"]), sc.avg_weekly_order)
        checks = []
        quality = []
        tr = None if {"discount", "temp_c"} & set(sc.context) else truth_7d(truth, sc.store, sc.sku, sc.origin)
        err = None
        if tr is not None:
            err = (f.point_7d - tr) / tr
        if sc.tol is not None:
            quality.append(err is not None and abs(err) <= sc.tol)
        if sc.route:
            checks.append(d.route == sc.route)
        if sc.not_route:
            checks.append(d.route != sc.not_route)
        if sc.flag:
            checks.append(sc.flag in f.flags)
        if sc.source:
            checks.append(f.source == sc.source)
        if sc.max_conf is not None:
            checks.append(f.confidence <= sc.max_conf)
        base = None
        if sc.uplift:
            ctx = {k: v for k, v in sc.context.items() if k != "discount"}
            base = svc.forecast(sc.store, sc.sku, sc.origin, ctx).point_7d
            checks.append(f.point_7d >= sc.uplift * base)
        out.append({
            "id": sc.sid, "title": sc.title, "store": sc.store, "sku": sc.sku, "origin": sc.origin,
            "context": sc.context, "expected": sc.expected, "truth_7d": None if tr is None else round(tr, 1),
            "point_7d": f.point_7d, "low_7d": f.low_7d, "high_7d": f.high_7d, "error_pct": None if err is None else round(err * 100, 1),
            "confidence": f.confidence, "flags": f.flags, "source": f.source, "route": d.route,
            "order_qty": d.order_qty, "reason": d.reason, "base_7d": base, "safe": all(checks), "accurate": all(quality) if quality else None,
            "ok": all(checks) and all(quality),
        })
    return out
