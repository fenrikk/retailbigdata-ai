import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forecast.scenarios import SCENARIOS, run
from forecast.service import ForecastService


class PrototypeScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = ForecastService(ROOT / "data", ROOT / "models")
        truth = pd.read_csv(ROOT / "data" / "truth.csv", parse_dates=["date"])
        cls.results = {r["id"]: r for r in run(cls.svc, truth)}
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "scenarios.json").write_text(
            json.dumps(list(cls.results.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    def _check(self, sid):
        r = self.results[sid]
        self.assertIsNotNone(r["point_7d"])
        self.assertLessEqual(r["low_7d"], r["point_7d"])
        self.assertGreaterEqual(r["high_7d"], r["point_7d"])
        self.assertTrue(r["safe"], r)

    def test_t1_regular_week(self):
        self._check("T1")

    def test_t2_heat_wave(self):
        self._check("T2")

    def test_t3_promo(self):
        self._check("T3")

    def test_t4_new_product(self):
        self._check("T4")

    def test_t5_model_down(self):
        self._check("T5")

    def test_t6_holiday_week(self):
        self._check("T6")

    def test_t7_weather_unavailable(self):
        self._check("T7")

    def test_t8_low_volume(self):
        self._check("T8")

    def test_t9_censored(self):
        self._check("T9")

    def test_t10_entry_error(self):
        self._check("T10")

    def test_t11_order_limit(self):
        self._check("T11")

    def test_t12_seasonal(self):
        self._check("T12")

    def test_t13_holiday_juice(self):
        self._check("T13")

    def test_t14_new_product(self):
        self._check("T14")

    def test_t15_heat_water(self):
        self._check("T15")

    def test_t16_weather_shift(self):
        self._check("T16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
