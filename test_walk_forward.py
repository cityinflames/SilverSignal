from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from forecast_engine import forecast
from test_engine import synthetic_prices


class WalkForwardTests(unittest.TestCase):
    def test_unseen_prices_cannot_change_issued_predictions(self):
        prices = synthetic_prices(1200)
        original = forecast(prices, 20).backtest
        cutoff = original.iloc[48]["Date"]
        changed = prices.copy()
        changed.loc[changed.Date > cutoff, "Close"] *= 1.3
        revised = forecast(changed, 20).backtest
        fields = ["Model estimated up probability", "Model predicted return", "Baseline up probability"]
        mask = original.Date <= cutoff
        pd.testing.assert_frame_equal(original.loc[mask, fields], revised.loc[mask, fields])
        self.assertFalse(original["Observed forward return"].equals(revised["Observed forward return"]))
        for _, block in original.groupby("Fold"):
            self.assertLess(block["Latest training outcome"].max(), block.Date.min())
        spaced = original.loc[original["Non-overlapping origin"]]
        self.assertTrue((spaced["Outcome date"].iloc[:-1].values <= spaced.Date.iloc[1:].values).all())

    def test_dashboard_renders_both_band_endpoints(self):
        prices = synthetic_prices(850)
        result = forecast(prices, 20)
        with patch("market_data.fetch_silver_futures", return_value=prices):
            app = AppTest.from_file(str(Path(__file__).resolve().parent / "app.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(f"USD {result.lower_price:,.2f} to USD {result.upper_price:,.2f}" in item.value
                            for item in app.markdown))

