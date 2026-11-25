"""Historical investment calculator for Xcel Energy and Hawaiian Electric Industries."""
from datetime import date
from math import isfinite

import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(symbol="XEL"):
    import yfinance as yf

    history = yf.Ticker(symbol).history(
        start="2000-01-01", end=date.today().isoformat(), interval="1d",
        auto_adjust=False, actions=True, timeout=20, raise_errors=True,
    )
    history.index = history.index.tz_localize(None).normalize()
    return history.sort_index()


def balances(history, start, amount, monthly=False):
    """Use split-adjusted closes and dividends on the same share basis."""
    required = ["Close", "Adj Close", "Dividends"]
    if not isfinite(amount) or amount <= 0:
        raise ValueError("Enter a positive investment amount.")
    if not set(required).issubset(history.columns):
        raise ValueError("The provider did not return complete price and dividend data.")
    rows = history.loc[pd.Timestamp(start):, required]
    if rows.empty or not rows.index.is_unique or not rows.index.is_monotonic_increasing:
        raise ValueError("No usable trading history for this start date.")
    if not rows.apply(lambda col: col.map(isfinite)).all().all() or (rows[required[:2]] <= 0).any().any():
        raise ValueError("The provider returned incomplete or invalid prices.")
    start = pd.Timestamp(start)
    dates = [start]
    if monthly:
        months = pd.period_range(start, rows.index[-1], freq="M")
        dates = [start + pd.DateOffset(months=i) for i in range(len(months))]
    deposits = pd.Series(0.0, index=rows.index)
    for day in dates:
        index = rows.index.searchsorted(day)
        if index < len(rows):
            deposits.iloc[index] += amount
    shares = (deposits / rows["Close"]).cumsum()
    cash = (shares.shift(fill_value=0) * rows["Dividends"]).cumsum()
    price = shares * rows["Close"]
    result = pd.DataFrame({
        "Original shares": price,
        "Shares plus cash dividends": price + cash,
        "Dividends reinvested": (deposits / rows["Adj Close"]).cumsum() * rows["Adj Close"],
    })
    if monthly:
        result["Total contributed"] = deposits.cumsum()
    return result


def show_calculator(symbol="XEL"):
    name, exchange = {"XEL": ("Xcel Energy", "NASDAQ"),
                      "HE": ("Hawaiian Electric Industries", "NYSE")}[symbol]
    st.title(f"{name} investment history")
    st.caption(f"{exchange}: {symbol} · Historical returns in US dollars")
    try:
        history = fetch_history(symbol)
    except Exception:
        st.error(f"Yahoo Finance could not load {symbol} history. Try again later.")
        return
    if history.empty:
        st.error(f"Yahoo Finance returned no {symbol} history.")
        return
    first, last = history.index[0], history.index[-1]
    default = max(first, last - pd.DateOffset(years=10))
    monthly = st.radio("Contribution schedule", ["One time", "Monthly"], horizontal=True) == "Monthly"
    start = st.date_input("Start date", default.date(), min_value=first.date(), max_value=last.date())
    if monthly:
        st.caption("Buy on the same date each month, starting on your chosen date. Short months use month-end. Weekends and holidays roll forward to the next trading close.")
        amount = st.number_input("Monthly contribution ($)", min_value=1.0, value=50.0, step=5.0)
    else:
        amount = st.number_input("Initial investment ($)", min_value=1.0, value=1000.0, step=100.0)
    reinvest = st.toggle("Reinvest dividends", value=True)
    try:
        values = balances(history, start, amount, monthly)
    except ValueError as exc:
        st.error(str(exc))
        return
    selected = "Dividends reinvested" if reinvest else "Shares plus cash dividends"
    end_value = values[selected].iloc[-1]
    contributed = values["Total contributed"].iloc[-1] if monthly else amount
    if monthly:
        st.write(f"Total contributed: ${contributed:,.2f} ({round(contributed / amount)} purchases)")
    st.metric("Ending value", f"${end_value:,.2f}", f"{(end_value / contributed - 1):+.1%}")
    st.write(f"Gain or loss: ${end_value - contributed:,.2f}")
    st.caption(f"First purchase at the close on {values.index[0]:%b %d, %Y}. Valued at the close on {last:%b %d, %Y}.")
    st.line_chart(values[[selected, "Total contributed" if monthly else "Original shares"]], y_label="Value ($)")
    st.dataframe(values.tail(1).T.rename(columns={last: "Ending value ($)"}).style.format("{:,.2f}"))
    st.caption("Yahoo Finance daily history. Reinvestment uses dividend-adjusted closes. With reinvestment off, dividends accumulate as cash from their ex-dividend dates and earn no interest. Assumes fractional shares and excludes taxes and fees.")

