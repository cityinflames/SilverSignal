# SilverSignal

A small, mobile-responsive research dashboard that studies **historical daily silver prices** and estimates whether the price may be higher in 5, 10, 20 or 60 trading sessions. It **does not know when silver will go up**, place trades, or promise accuracy. It shows backtest comparisons so a poor model cannot quietly masquerade as a sure thing.

## Start on Windows

1. Install Python 3.10 or newer from https://www.python.org/downloads/ and check **Add python.exe to PATH** if prompted.
2. Extract this ZIP, open the `SilverSignal` folder, and double-click `start_windows.bat`.
3. On the first run, Python downloads the required free packages. The app then opens in your browser at http://localhost:8501 (open it manually if necessary).
4. The app fetches silver futures daily historical prices from Yahoo Finance (`SI=F`). This needs internet access. Alternatively, upload a CSV containing daily **Date** and **Close** columns, with at least 650 observations. It does not ship with made-up market data.

**Android / phone:** Streamlit is a browser app, not an APK. You can open it on a phone on the same Wi-Fi network as the running Windows laptop by visiting `http://LAPTOP-LAN-IP:8501` (if the Windows firewall allows access). You can also deploy it to a Streamlit-compatible hosting service if you want it online. Do **not** forward port 8501 publicly on your router.

## What you see

- Last delayed data price, historical moving averages, RSI
- An *experimental* probability estimate for the price ending higher at each selected horizon
- A ridge-regression price projection and illustrative residual-based historical band
- A chart showing history, averages and a forecast endpoint
- A chronologically held-out test against naive benchmarks: direction accuracy, Brier score, and forecast-return MAE
- CSV export of held-out predictions and cleaned prices

## Important methodological notes

- `SI=F` denotes Yahoo Finance continuous silver futures. Futures contract rolls may produce synthetic-looking jumps; it is **not** a coin shop's immediate spot or scrap offer.
- A `Date,Close` upload is treated as real historical daily closing prices. CSV observations must be sorted or sortable and must not include future dates or obviously invalid prices. This is a research tool, **not** a data auditor.
- After a 252-session indicator warm-up, the latest quarter of labeled forecast dates forms an expanding-window backtest. Every 20 sessions, fit both regressions and the frequency baseline on outcomes ending strictly before the first forecast in that block. Standardization is fitted inside each training pipeline. For today's after-close estimate, both models are refitted on all completed outcomes.
- The dashboard scores all forecast dates and a second subset spaced one full horizon apart. Shared market regimes can affect even non-overlapping outcomes. The CSV includes fold, last training outcome date, target date, model and baseline probabilities, realized return and the subset flag. The baseline uses the same information cutoff as the model. A flat outcome counts as "did not rise".
- The range is illustrative. It uses historical walk-forward return errors for the latest forecast. Its future coverage has not been verified. A favorable sample alone does not establish a repeatable edge.
- The model does not account for Fed decisions, inflation surprises, gold prices, supply/demand reports, recessions, wars, ETF flows, or premiums charged by silver dealers. Model probabilities have **not** been independently calibrated.
- The 10th-to-90th-percentile band is based on out-of-sample errors for the chosen horizon; it is not a guaranteed 80% interval in future markets.
- Use for learning and comparison, not trading or financial decisions by itself.

## Tech

Python / Streamlit, pandas, scikit-learn, Plotly, and yfinance. No key required, though Yahoo may rate-limit or block some requests. CSV upload is the alternative.

## Test

After installing packages: `python -m unittest discover -s tests -v`. These tests use **synthetic** local prices, never present them as market history. Live Yahoo fetching cannot be verified without a network connection.

## Files

- `app.py` – interface and visualization
- `forecast_engine.py` – indicator math and leakage-conscious chronological test
- `market_data.py` – remote source and CSV parsing
- `tests/test_engine.py` – deterministic math and input tests
- `start_windows.bat` – one-click Windows setup and launch

The bundled historical comparison and its limitations are documented in [VERIFICATION.md](VERIFICATION.md). Run `python compare_backtests.py` to repeat it, or pass a path to your own Date/Close CSV.

Source reference: https://finance.yahoo.com/quote/SI%3DF/history/ and https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html

## Forecast notebook (v0.3)

With Yahoo silver futures selected, opening a forecast records the first call for that price date, horizon, training-history setting and model version. Switch the sidebar View to Forecast notebook to see pending calls or export all rows. You can open the notebook without an internet connection once dependencies are installed.

The dashboard uses Yahoo bars dated before today in New York to avoid partial daily bars. A new record requires a price date within the prior four calendar days. The recorded UTC timestamp shows when the app actually made the call, which can be later than its starting price. CSV uploads remain available for historical research and never enter the live notebook.

Opening a later Yahoo forecast checks pending calls against the same SI=F series. The outcome is the close on the Nth observed session after the original price date, and must be dated after the day the call was recorded. A missing origin or too little subsequent history leaves the call pending. The first saved prices, model probability, training up-rate and settled outcome are preserved on refresh. Scored calls report directional accuracy and Brier against that saved training up-rate. No forecasts are issued or scored while the app is closed. Notebook statistics describe only the calls opened and can include overlapping outcomes.

On Windows the notebook lives in `%LOCALAPPDATA%\SilverSignal\forecasts.sqlite3`. It persists across restarts and extractions for the same Windows user. Close the app before copying this file for backup. The ZIP contains no personal notebook data.
