"""Deterministic synthetic test history. These prices are NOT market records."""
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast_engine import DataError, build_features, clean_prices, forecast, signal_label
from market_data import load_csv


def synthetic_prices(n=1600, seed=34):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-02", periods=n)
    daily_returns = rng.normal(0.0002, 0.017, n)
    return pd.DataFrame({"Date": days, "Close": 20 * np.exp(np.cumsum(daily_returns))})


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = synthetic_prices()

    def test_futures_horizons(self):
        for horizon in (5, 10, 20, 60):
            with self.subTest(horizon=horizon):
                f = forecast(self.prices, horizon)
                self.assertTrue(0 <= f.probability_up <= 1)
                self.assertGreater(f.last_price, 0)
                self.assertLessEqual(f.lower_price, f.upper_price)
                self.assertGreater(f.backtest_count, 50)
                self.assertGreater(f.test_start_date, f.train_start_date)
                self.assertEqual(f.last_date, self.prices.iloc[-1]["Date"])
                self.assertIn(signal_label(f)[0], ["NO VERIFIED EDGE", "INSUFFICIENT EVIDENCE", "UPWARD LEAN · EXPERIMENTAL", "DOWNWARD LEAN · EXPERIMENTAL", "MIXED · EXPERIMENTAL"])

    def test_future_values_do_not_change_past_features(self):
        original = build_features(self.prices)
        changed = self.prices.copy()
        changed.loc[changed.index[-1], "Close"] *= 1.5
        mutated = build_features(changed)
        pd.testing.assert_frame_equal(original.iloc[:-1], mutated.iloc[:-1])

    def test_csv_upload_and_duplicates(self):
        text = b"Date,Close\n" + self.prices.to_csv(index=False, header=False).encode("utf8")
        df = load_csv(text)
        self.assertEqual(len(df), len(self.prices))
        dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        self.assertEqual(len(clean_prices(dup)), len(self.prices))

    def test_reject_non_daily(self):
        months = pd.date_range("1980-01-01", periods=520, freq="MS")
        monthly = pd.DataFrame({"Date": months, "Close": np.arange(520) + 10})
        with self.assertRaisesRegex(DataError, "daily prices"):
            clean_prices(monthly)

    def test_reject_short_bad_data(self):
        with self.assertRaises(DataError):
            clean_prices(pd.DataFrame({"Date":["2020-01-01"], "Close":["bad"]}))
        with self.assertRaises(DataError):
            clean_prices(pd.DataFrame({"whatever":[1]*500}))

    def test_no_future_data_in_model_training_labels(self):
        # Changing future observations cannot retroactively affect historical features.
        f = build_features(self.prices)
        t = 600
        other = self.prices.copy()
        other.loc[t+1:,"Close"] *= 1.11
        f2 = build_features(other)
        pd.testing.assert_frame_equal(f.loc[:self.prices.loc[t,"Date"]], f2.loc[:self.prices.loc[t,"Date"]])


if __name__ == "__main__":
    unittest.main()
