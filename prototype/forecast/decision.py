import math
from dataclasses import dataclass

T_AUTO = 0.75
T_REVIEW = 0.60
MAX_ORDER_FACTOR = 3.0
PERISHABLE_DAYS = 7


@dataclass
class Decision:
    route: str
    order_qty: int
    target_stock: float
    reason: str

    def as_dict(self):
        return {"route": self.route, "order_qty": self.order_qty, "target_stock": self.target_stock, "reason": self.reason}


ROUTE_UA = {
    "auto_order": "автоматичне замовлення",
    "manager_review": "перевірка менеджером",
    "reject_ai": "прогноз ШІ відхилено, резервне правило + менеджер",
}


def decide(forecast, stock_on_hand, in_transit, shelf_life_days, pack_size, avg_weekly_order=None,
           t_auto=T_AUTO, t_review=T_REVIEW):
    perishable = shelf_life_days <= PERISHABLE_DAYS
    safety_k = 0.5 if perishable else 1.0
    target = forecast.point_7d + safety_k * (forecast.high_7d - forecast.point_7d)
    need = max(0.0, target - stock_on_hand - in_transit)
    qty = int(math.ceil(need / pack_size) * pack_size) if need > 0 else 0
    conf = forecast.confidence
    if forecast.source != "ml_model" or conf < t_review:
        route = "reject_ai"
        reason = "модель недоступна" if forecast.source != "ml_model" else f"впевненість {conf:.2f} < {t_review}"
    elif conf < t_auto:
        route = "manager_review"
        reason = f"впевненість {conf:.2f} у діапазоні [{t_review}; {t_auto})"
    else:
        route = "auto_order"
        reason = f"впевненість {conf:.2f} ≥ {t_auto}"
    if route == "auto_order" and "cold_start" in forecast.flags:
        route, reason = "manager_review", "новий товар"
    if route == "auto_order" and avg_weekly_order and qty > MAX_ORDER_FACTOR * avg_weekly_order:
        route, reason = "manager_review", f"замовлення {qty} > {MAX_ORDER_FACTOR:g}× середнього"
    return Decision(route, qty, round(target, 1), reason)
