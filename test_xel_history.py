import unittest

import pandas as pd

from xel_history import balances


class XelReturnsTest(unittest.TestCase):
    def test_cash_and_reinvestment_after_weekend_purchase(self):
        days = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
        history = pd.DataFrame({"Close": [8, 10, 8, 12], "Adj Close": [6, 8, 8, 12],
                                "Dividends": [0, 1, 2, 0]}, index=days)
        values = balances(history, "2024-01-06", 100)
        self.assertEqual(values.index[0], days[1])
        self.assertEqual(values.iloc[0].tolist(), [100, 100, 100])
        self.assertEqual(values.iloc[-1].tolist(), [120, 140, 150])

    def test_missing_dividend_history(self):
        with self.assertRaises(ValueError):
            balances(pd.DataFrame(), "2024-01-01", 100)


    def test_month_end_schedule_and_dividend_eligibility(self):
        days = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-28",
                               "2024-04-01", "2024-04-29", "2024-04-30"])
        history = pd.DataFrame({"Close": 10, "Adj Close": [8, 10, 10, 10, 10, 10],
                                "Dividends": [0, 2, 0, 0, 0, 0]}, index=days)
        values = balances(history, days[0], 50, monthly=True)
        self.assertEqual(values["Total contributed"].tolist(), [50, 100, 100, 150, 150, 200])
        self.assertEqual(values.iloc[-1].tolist(), [200, 210, 212.5, 200])
