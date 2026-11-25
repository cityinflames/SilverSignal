"""First-issued Yahoo SI=F forecasts. All persistence is local to this user."""
from contextlib import closing
from pathlib import Path
import os
import sqlite3

import pandas as pd

MODEL = "v0.2 logistic/ridge"


def notebook_path():
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SilverSignal" / "forecasts.sqlite3"


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.execute("""CREATE TABLE IF NOT EXISTS forecasts (
        origin TEXT, horizon INTEGER, history TEXT, model TEXT, saved_utc TEXT NOT NULL,
        origin_price REAL, projected_price REAL, model_up REAL, baseline_up REAL,
        target TEXT, actual REAL, PRIMARY KEY(origin, horizon, history, model))""")
    return db


def update_notebook(path, result, history, now):
    """Called only after a successful Yahoo forecast. Keep first issue and first settlement."""
    market_day = now.tz_convert("America/New_York").date()
    prices = result.historic.set_index("Date")["Close"]
    age = (market_day - result.last_date.date()).days
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        for row, origin, horizon, saved in db.execute(
                "SELECT rowid, origin, horizon, saved_utc FROM forecasts WHERE target IS NULL").fetchall():
            if pd.Timestamp(origin) not in prices.index:
                continue
            later = prices.loc[prices.index > pd.Timestamp(origin)]
            if len(later) < horizon:
                continue
            target = later.index[horizon - 1].date()
            if target >= market_day or target <= pd.Timestamp(saved).tz_convert("America/New_York").date():
                continue
            db.execute("UPDATE forecasts SET target=?, actual=? WHERE rowid=?",
                       (target.isoformat(), float(later.iloc[horizon - 1]), row))
        if 1 <= age <= 4:
            db.execute("""INSERT INTO forecasts
                (origin,horizon,history,model,saved_utc,origin_price,projected_price,model_up,baseline_up)
                VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(origin,horizon,history,model) DO NOTHING""",
                       (result.last_date.date().isoformat(), result.horizon, history, MODEL,
                        now.tz_convert("UTC").isoformat(), result.last_price, result.predicted_price,
                        result.probability_up, result.baseline_probability_up))
    return 1 <= age <= 4


def read_notebook(path):
    with closing(connect(path)) as db:
        rows = pd.read_sql_query("SELECT * FROM forecasts ORDER BY origin DESC, horizon, history", db)
    settled = rows.actual.notna()
    observed = (rows.loc[settled, "actual"] > rows.loc[settled, "origin_price"]).astype(int)
    for col in ["model_up", "baseline_up"]:
        p = rows.loc[settled, col]
        rows.loc[settled, col + "_brier"] = (p - observed) ** 2
        rows.loc[settled, col + "_correct"] = ((p >= .5) == observed).astype(float)
    return rows
