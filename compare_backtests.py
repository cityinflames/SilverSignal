"""Reproduce v0.1 fixed-fit and v0.2 scores on the same 20-session origins."""
from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast_engine import FEATURES, build_features, clean_prices, forecast


def compare(prices):
    prices = clean_prices(prices)
    result = forecast(prices, 20)
    features = build_features(prices)
    close = prices.set_index("Date")["Close"]
    returns = (close.shift(-20) / close - 1).reindex(features.index).dropna()
    train = returns.iloc[:int(len(returns) * .75) - 20]
    x = features.loc[train.index, list(FEATURES)]
    direction = train > 0
    fixed = make_pipeline(StandardScaler(), LogisticRegression(C=.35, max_iter=1000))
    fixed.fit(x, direction)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(x, train)
    bt = result.backtest.set_index("Date")
    test = features.loc[bt.index, list(FEATURES)]
    actual = bt["Observed up"].astype(int)
    model = bt["Model estimated up probability"]
    prior = bt["Baseline up probability"]

    print(f"{len(prices)} prices; {len(bt)} origins {bt.index.min():%Y-%m-%d} through {bt.index.max():%Y-%m-%d}")
    for name, prob in [("v0.1 model", fixed.predict_proba(test)[:, 1]),
                       ("v0.1 baseline", direction.mean()), ("v0.2 model", model),
                       ("v0.2 baseline", prior), ("Research only: equal model/prior blend", (model + prior) / 2)]:
        print(name, f"accuracy {((prob >= .5) == actual).mean():.1%}, Brier {((prob - actual)**2).mean():.4f}")
    print(f"Return MAE: fixed {(ridge.predict(test) - bt['Observed forward return']).abs().mean():.2%}; "
          f"walk-forward {result.model_mae:.2%}; zero {result.baseline_mae:.2%}")
    spaced = bt.loc[bt["Non-overlapping origin"]]
    print(f"{len(spaced)} non-overlapping outcomes, {bt['Fold'].nunique()} refits")
    for col in ["Model estimated up probability", "Baseline up probability"]:
        p = spaced[col]
        y = spaced["Observed up"].astype(int)
        print(col, f"{((p >= .5) == y).mean():.1%}, {((p - y)**2).mean():.4f}")


if __name__ == "__main__":
    default = Path(__file__).parent / "silver_usd_2017_2025.csv"
    compare(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else default))
