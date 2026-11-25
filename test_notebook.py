from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pandas as pd

from forecast_notebook import read_notebook, update_notebook


class NotebookTests(unittest.TestCase):
    def test_restart_and_delayed_scoring(self):
        days = pd.bdate_range("2025-03-03", periods=8)
        prices = pd.DataFrame({"Date": days, "Close": [20., 21., 22., 23., 24., 19., 33., 15.]})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data" / "forecasts.sqlite3"
            self.assertTrue(read_notebook(path).empty)
            def visit(n, day, model=.8):
                result = SimpleNamespace(historic=prices.iloc[:n], last_date=days[n-1],
                                         horizon=5, last_price=float(prices.Close.iloc[n-1]),
                                         predicted_price=24., probability_up=model, baseline_probability_up=.4)
                return update_notebook(path, result, "10y", pd.Timestamp(day, tz="America/New_York"))
            visit(1, "2025-03-04")
            visit(1, "2025-03-04", .1)
            self.assertEqual(len(read_notebook(path)), 1)
            visit(5, "2025-03-08")
            self.assertTrue(read_notebook(path).target.isna().all())
            visit(8, "2025-03-13")
            row = read_notebook(path).iloc[-1]
            self.assertEqual((row.target, row.actual, row.model_up), ("2025-03-10", 19., .8))
            self.assertAlmostEqual(row.model_up_brier, .64)
            self.assertAlmostEqual(row.baseline_up_brier, .16)
            self.assertEqual((row.model_up_correct, row.baseline_up_correct), (0., 1.))
            prices.loc[5, "Close"] = 50.
            self.assertFalse(visit(8, "2025-05-01"))
            self.assertEqual(read_notebook(path).iloc[-1].actual, 19.)

    def test_dashboard_records_and_notebook_reopens_offline(self):
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        from test_engine import synthetic_prices

        prices = synthetic_prices(850)
        today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
        prices["Date"] = pd.bdate_range(end=today - pd.Timedelta(days=1), periods=len(prices))
        with TemporaryDirectory() as directory, patch("forecast_notebook.notebook_path", return_value=Path(directory)/"calls.db"):
            entry = str(Path(__file__).resolve().parent / "app.py")
            with patch("market_data.fetch_silver_futures", return_value=prices):
                app = AppTest.from_file(entry, default_timeout=120).run()
            self.assertFalse(app.exception)
            with patch("market_data.fetch_silver_futures", side_effect=AssertionError("unexpected network call")):
                app.sidebar.radio[0].set_value("Forecast notebook").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.dataframe[0].value), 1)
            self.assertTrue(app.dataframe[0].value["Outcome date"].isna().all())

