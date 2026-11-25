# SilverSignal v0.2 verification

## One-month comparison

The v0.1 numbers supplied with the request were 36.5% model accuracy, 60.2% baseline accuracy, 0.382 model Brier and 0.249 baseline Brier. That exact price snapshot was unavailable. Yahoo downloading in the test environment failed DNS resolution. The table below uses one other fixed history for both versions.

The fixture contains 2,346 daily USD-per-ounce silver observations from January 2, 2017 through December 31, 2025. Values come from the [Exchange-Rates.org U.S. silver history](https://www.exchange-rates.org/precious-metals/silver-price/united-states/2025) annual ounce tables for 2017 through 2025 (replace the year in that URL). This series has its own publication calendar and is distinct from Yahoo's continuous silver futures contract. The fixture can be used in the CSV upload screen. The app still opens with the original Yahoo data source.

Both runs scored the same 519 forecast dates from December 8, 2023 through December 3, 2025, after the close, with outcomes 20 observed sessions later. The final result matured on December 31. Flat returns count as down for the binary event “price rose.” Each Brier score is the average squared probability error; lower is better.

| Method | Direction accuracy | Brier | Forward return MAE |
|---|---:|---:|---:|
| v0.1 fixed model | 35.6% | 0.3225 | 7.34% |
| v0.1 fixed training up-rate / zero return | 31.2% | 0.2510 | 6.99% |
| v0.2 walk-forward model | 43.2% | 0.2852 | 6.97% |
| v0.2 expanding training up-rate / zero return | 64.2% | 0.2446 | 6.99% |
| Research-only 50/50 blend of model and expanding up-rate | 44.7% | 0.2597 | — |

The baseline changes because it can learn from newly matured outcomes at every refit. The directional call changes when its historical up-rate crosses 50%. It would be misleading to compare v0.2 solely with the old frozen benchmark. The equal-weight probability blend improved Brier versus the refitted logistic model but lost to its updated baseline. It was an exploratory comparison on these dates, so it was excluded from the live model.

Using only the first forecast and every twentieth forecast thereafter leaves 26 outcomes. Model and baseline each had 46.2% accuracy. Model Brier was 0.2790 against baseline 0.2496. The small return MAE difference and the overlapping daily scores do not establish a repeatable advantage. **NO VERIFIED EDGE**.

## Audit and changes

- v0.1 features used trailing/current closes, scalers fitted on training rows, and a horizon gap before a chronological holdout. No look-ahead was identified in those calculations. Its one fixed fit could become stale, its baseline stayed frozen, and daily outcomes overlapped. A favorable point estimate previously sufficed to show an experimental lean. These scores alone no longer qualify a signal.
- v0.2 makes 20-session refits on expanding data. Each fold excludes every label ending on or after the first prediction date. Model and frequency benchmark use identical training dates. Each exported row identifies its fold, forecast date, realization date, latest training outcome and non-overlap flag.
- The live projection still uses the original logistic and ridge model settings with all matured labels available at the latest close. The error band uses the empirical 10th and 90th percentiles of historical out-of-sample return errors. Its prospective coverage has not been tested. The revised layout gives the range a full-width line that wraps on narrow screens.
- Duplicate dates now keep the last valid input consistently, tomorrow's dates are rejected, and chart averages retain their lookback history when the chart window changes. Daily quotes are assumed complete at the forecast origin. Market-data revisions, futures rolls and publication delays are outside this test.

## Repeat the checks

From the extracted project folder, after running the Windows launcher:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe compare_backtests.py
```

The original engine was also run separately on the bundled fixture. Its results agree with the fixed-fit reproduction in the comparison script. Use an optional CSV path with the script to repeat on your own frozen price history. The exported cleaned-price CSV is the input to retain for future comparisons.

The test suite exercises all four forecast horizons, CSV validation, indicator causality, the embargo at each fold, future-price perturbations across refits, and Streamlit rendering of both uncertainty endpoints with an injected price feed. Synthetic prices are used only for software checks. The live Yahoo service, a clean online package install, phone rendering and prospective forecast performance were not verified.


All eight tests passed from an extracted release copy. The Windows test environment blocked local socket connections. For the Streamlit element-tree check, the test runner used a socket-free event loop. The actual web-server health check timed out here, so browser startup and the launcher were not verified end to end. The launcher bytes match v0.1.

## v0.3 notebook verification

All ten tests passed on Windows. Streamlit's element-tree test opened the dashboard with an injected feed, saved a forecast, and opened its notebook without another data request. A separate test reopened the database on successive calls, repeated a forecast with altered probabilities, advanced a controlled history through the fifth session, verified both Brier scores and direction calls, then changed that session's price and confirmed the original settlement survived. The UI harness used a socket-free event loop. Live Yahoo access and prospective accuracy were not verified.
