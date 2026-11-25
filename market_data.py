"""Import genuine silver futures history or user-supplied daily silver prices."""
from __future__ import annotations

import io

import pandas as pd
from forecast_engine import clean_prices, DataError

SILVER_FUTURES_SYMBOL = "SI=F"


def fetch_silver_futures(period: str = "10y") -> pd.DataFrame:
    """Yahoo Finance delayed continuous silver futures; not the local scrap spot quote."""
    if period not in ("5y", "10y", "max"):
        raise DataError("Unsupported history period.")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError("Install requirements.txt to enable Yahoo Finance downloading.") from exc
    try:
        df = yf.Ticker(SILVER_FUTURES_SYMBOL).history(
            period=period, interval="1d", auto_adjust=False, actions=False,
            timeout=20, raise_errors=True,
        )
    except Exception as exc:
        raise DataError("Yahoo Finance could not return silver data. Check the connection or upload a CSV. ") from exc
    if df is None or df.empty or "Close" not in df.columns:
        raise DataError("Yahoo Finance returned no valid silver close prices. Try again later or use CSV.")
    dates = pd.to_datetime(df.index).tz_localize(None).normalize()
    return clean_prices(pd.DataFrame({"Date": dates, "Close": df["Close"].to_numpy()}))


def load_csv(uploaded_bytes: bytes) -> pd.DataFrame:
    if len(uploaded_bytes) > 8_000_000:
        raise DataError("CSV exceeds 8MB. Please upload a smaller daily-price file.")
    try:
        raw = pd.read_csv(io.BytesIO(uploaded_bytes), encoding="utf-8-sig")
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DataError("Cannot read that CSV. Use a UTF-8 file with Date and Close columns.") from exc
    return clean_prices(raw)
