"""Transparent, price-only research model for silver. No live-data dependencies."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_OBSERVATIONS = 650
FEATURES = (
    "return_1d", "return_5d", "return_20d", "return_60d",
    "above_ma20", "above_ma60", "ma20_vs_ma60",
    "volatility_20d", "volatility_60d", "rsi14", "dist_252d_high",
)


class DataError(ValueError):
    """A market-data or CSV problem that can be shown to the user."""


def clean_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Date/Close frame; never interpolate missing trading days."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataError("No silver prices were found.")
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    date_col = lookup.get("date") or lookup.get("datetime")
    close_col = lookup.get("close") or lookup.get("adj close") or lookup.get("price")
    if date_col is None or close_col is None:
        raise DataError("CSV needs Date and Close columns (or Date and Price).")
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    cleaned = frame[close_col].astype(str).str.replace(r"[$,\s]", "", regex=True)
    out["Close"] = pd.to_numeric(cleaned, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out["Close"] > 0]
    out = out.sort_values("Date", kind="stable").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    if len(out) < MIN_OBSERVATIONS:
        raise DataError(f"At least {MIN_OBSERVATIONS} valid daily prices are needed; found {len(out)}.")
    if out["Date"].iloc[-1] > pd.Timestamp.now(tz="UTC").tz_localize(None).normalize():
        raise DataError("The dataset includes dates in the future. Check your Date column.")
    # Protect against weekly/monthly CSVs being misread as daily history.
    median_gap = out["Date"].diff().dt.days.dropna().median()
    if median_gap > 4:
        raise DataError("The model requires daily prices, not weekly or monthly observations.")
    return out


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Features use current and *past* prices only (no forward-looking inputs)."""
    close = prices.set_index("Date")["Close"].astype(float)
    change = close.pct_change(fill_method=None)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    ups = change.clip(lower=0)
    downs = (-change).clip(lower=0)
    avg_up = ups.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_down = downs.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    # 50% means flat momentum; 100% means only rises during the lookback.
    rsi = 100 * avg_up / (avg_up + avg_down).replace(0, np.nan)
    rsi = rsi.fillna(50) / 100
    features = pd.DataFrame({
        "return_1d": close.pct_change(1, fill_method=None),
        "return_5d": close.pct_change(5, fill_method=None),
        "return_20d": close.pct_change(20, fill_method=None),
        "return_60d": close.pct_change(60, fill_method=None),
        "above_ma20": close / ma20 - 1,
        "above_ma60": close / ma60 - 1,
        "ma20_vs_ma60": ma20 / ma60 - 1,
        "volatility_20d": change.rolling(20, min_periods=20).std(),
        "volatility_60d": change.rolling(60, min_periods=60).std(),
        "rsi14": rsi,
        "dist_252d_high": close / close.rolling(252, min_periods=252).max() - 1,
    }, index=close.index)
    return features.replace([np.inf, -np.inf], np.nan).dropna()


@dataclass
class ForecastResult:
    last_date: pd.Timestamp
    last_price: float
    horizon: int
    baseline_probability_up: float
    probability_up: float
    predicted_return: float
    predicted_price: float
    lower_price: float
    upper_price: float
    backtest_accuracy: float
    baseline_accuracy: float
    model_brier: float
    baseline_brier: float
    model_mae: float
    baseline_mae: float
    backtest_count: int
    approximate_independent_windows: int
    training_count: int
    train_start_date: pd.Timestamp
    test_start_date: pd.Timestamp
    last_rsi: float
    ma20: float
    ma60: float
    backtest: pd.DataFrame
    historic: pd.DataFrame


def forecast(prices: pd.DataFrame, horizon: int = 20) -> ForecastResult:
    if horizon not in (5, 10, 20, 60):
        raise DataError("Forecast window must be 5, 10, 20 or 60 trading sessions.")
    prices = clean_prices(prices)
    features = build_features(prices)
    close = prices.set_index("Date")["Close"]
    future_return = close.shift(-horizon) / close - 1
    labeled = features.copy()
    labeled["future_return"] = future_return.reindex(features.index)
    labeled = labeled.dropna(subset=["future_return"])
    if len(labeled) < 200:
        raise DataError("There is not enough history to fit and test this horizon.")

    # Chronological holdout. Embargo horizon prevents training labels from
    # incorporating prices at/after the first holdout prediction date.
    split = int(len(labeled) * 0.75)
    train = labeled.iloc[:split - horizon]
    test = labeled.iloc[split:]
    if len(train) < 150 or len(test) < 30:
        raise DataError("Insufficient training or holdout records.")

    direction = lambda data: (data["future_return"] > 0).astype(int)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.35, max_iter=1000))
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=30))
    if direction(train).nunique() < 2:
        raise DataError("Training data contains only one market direction.")
    # Freeze both models and the benchmark at each 20-session refit.
    # For an origin k, origin k-h has an outcome ending at k. Exclude it
    # as well, so even the latest training outcome predates the fold.
    blocks = []
    for fold, origin in enumerate(range(split, len(labeled), 20), start=1):
        past = labeled.iloc[:origin - horizon]
        block = labeled.iloc[origin:origin + 20]
        classifier.fit(past[list(FEATURES)], direction(past))
        regressor.fit(past[list(FEATURES)], past["future_return"])
        blocks.append(pd.DataFrame({
            "Date": block.index,
            "Outcome date": close.index[close.index.get_indexer(block.index) + horizon],
            "Fold": fold,
            "Latest training outcome": close.index[close.index.get_loc(past.index[-1]) + horizon],
            "Observed up": direction(block).to_numpy(dtype=bool),
            "Model estimated up probability": classifier.predict_proba(block[list(FEATURES)])[:, 1],
            "Baseline up probability": direction(past).mean(),
            "Observed forward return": block["future_return"].to_numpy(),
            "Model predicted return": regressor.predict(block[list(FEATURES)]),
        }))
    predicted = pd.concat(blocks, ignore_index=True)
    predicted["Non-overlapping origin"] = np.arange(len(predicted)) % horizon == 0
    true_up = predicted["Observed up"]
    p_test = predicted["Model estimated up probability"]
    prior = predicted["Baseline up probability"]
    reg_test = predicted["Model predicted return"].to_numpy()
    accuracy = float(accuracy_score(true_up, p_test >= 0.5))
    baseline_accuracy = float(accuracy_score(true_up, prior >= 0.5))
    brier = float(brier_score_loss(true_up, p_test))
    baseline_brier = float(brier_score_loss(true_up, prior))
    mae = float(mean_absolute_error(test["future_return"], reg_test))
    baseline_mae = float(test["future_return"].abs().mean())
    residual = test["future_return"].to_numpy() - reg_test
    lower_residual, upper_residual = np.percentile(residual, [10, 90])

    # Refit using every fully observed label, never the final unlabeled rows.
    classifier.fit(labeled[list(FEATURES)], direction(labeled))
    regressor.fit(labeled[list(FEATURES)], labeled["future_return"])
    latest = features.iloc[[-1]][list(FEATURES)]
    prob = float(classifier.predict_proba(latest)[0, 1])
    predicted_return = float(regressor.predict(latest)[0])
    last_price = float(close.iloc[-1])
    predicted_price = max(0.01, last_price * (1 + predicted_return))
    lower_price = max(0.01, last_price * (1 + predicted_return + float(lower_residual)))
    upper_price = max(0.01, last_price * (1 + predicted_return + float(upper_residual)))

    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    return ForecastResult(
        last_date=close.index[-1], last_price=last_price, horizon=horizon,
        baseline_probability_up=float(direction(labeled).mean()), probability_up=prob, predicted_return=predicted_return,
        predicted_price=predicted_price, lower_price=lower_price, upper_price=upper_price,
        backtest_accuracy=accuracy, baseline_accuracy=baseline_accuracy,
        model_brier=brier, baseline_brier=baseline_brier,
        model_mae=mae, baseline_mae=baseline_mae,
        backtest_count=len(test),
        approximate_independent_windows=math.ceil(len(test) / horizon),
        training_count=len(train), train_start_date=train.index[0],
        test_start_date=test.index[0], last_rsi=float(latest["rsi14"].iloc[0] * 100),
        ma20=ma20, ma60=ma60, backtest=predicted,
        historic=prices,
    )


def signal_label(result: ForecastResult) -> tuple[str, str]:
    """Do not imply a proven trading signal when the backtest lacks an edge."""
    if result.approximate_independent_windows < 20:
        return ("INSUFFICIENT EVIDENCE", "Fewer than 20 non-overlapping outcomes were observed.")
    return ("NO VERIFIED EDGE", "These historical scores alone do not establish a repeatable forecasting advantage.")
