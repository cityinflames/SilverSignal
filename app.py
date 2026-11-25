"""SilverSignal: a responsive, evidence-first silver forecasting dashboard."""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path
import math
import sqlite3
from forecast_notebook import notebook_path, read_notebook, update_notebook

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast_engine import DataError, forecast, signal_label
from market_data import fetch_silver_futures, load_csv

st.set_page_config(page_title="SilverSignal · Historical Forecast Lab", page_icon=str(Path(__file__).with_name("silversignal-icon-1.png")), layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {color-scheme: dark;}
[data-testid="stAppViewContainer"] {background: #0b121a; color: #e7ecf1;}
[data-testid="stSidebar"] {background: #101c29;}
.block-container {padding-top: 2rem; max-width: 1260px;}
h1, h2, h3 {font-family: 'Segoe UI', sans-serif; letter-spacing: -.025em;}
h1 {font-size: 2.9rem !important;}
[data-testid="stMetric"] {background: #142332; border:1px solid #263b4d; border-radius: 14px; padding: 15px;}
.hero-eyebrow {font-size: .76rem; font-weight: 700; letter-spacing: .2em; color: #dbb98c;}
.hero-sub {font-size: 1.05rem; color: #a6b8c8; max-width: 750px;}
.tag {display: inline-block; padding: 7px 13px; border-radius: 40px; background: #2b3440;
      color: #f6d6aa; border: 1px solid #53616d; font-size: .80rem; letter-spacing: .06em;}
.note {color:#a6b8c8; font-size:.9rem; line-height:1.6;}
.good {color:#83d3a4;} .warn {color:#ffce8b;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_yahoo(period: str) -> pd.DataFrame:
    return fetch_silver_futures(period)


@st.cache_data(show_spinner=False)
def cached_model(prices: pd.DataFrame, horizon: int):
    return forecast(prices, horizon)


st.logo(str(Path(__file__).with_name("silversignal-icon-1.png")))
st.sidebar.markdown("### SilverSignal")
st.sidebar.caption("Silver market research · experimental model")
view = st.sidebar.radio("View", ["Dashboard", "Forecast notebook", "XEL investment history", "Hawaiian Electric investment history"])
if view in ("XEL investment history", "Hawaiian Electric investment history"):
    from xel_history import show_calculator
    show_calculator("HE" if view.startswith("Hawaiian Electric") else "XEL")
    st.stop()
if view == "Forecast notebook":
    st.title("Forecast notebook")
    st.caption("First-issued Yahoo SI=F calls saved on this computer. Each row is a forecast you actually opened. Times are UTC.")
    try:
        rows = read_notebook(notebook_path())
    except (OSError, sqlite3.Error) as exc:
        st.error(f"Could not open the forecast notebook: {exc}")
        st.stop()
    if rows.empty:
        st.info("No forecasts recorded yet. Open the dashboard with Yahoo silver futures and select a horizon.")
        st.stop()
    history_key = st.selectbox("Training history", rows.history.unique())
    horizon_key = st.selectbox("Trading sessions", sorted(rows.loc[rows.history == history_key, "horizon"].unique()))
    selected = rows[(rows.history == history_key) & (rows.horizon == horizon_key)]
    settled = selected[selected.target.notna()]
    st.write(f"{len(settled)} scored · {len(selected) - len(settled)} pending")
    if len(settled):
        for name, col in [("Model", "model_up"), ("Baseline", "baseline_up")]:
            st.write(f"{name}: {settled[col + '_correct'].mean():.1%} direction accuracy · "
                     f"{settled[col + '_brier'].mean():.3f} Brier score")
        st.caption("Lower Brier scores are better. Nearby forecasts can share most of the same daily price moves.")
    st.dataframe(selected.rename(columns={"origin":"Price date", "saved_utc":"Recorded (UTC)",
                 "origin_price":"Starting price", "projected_price":"Expected price",
                 "target":"Outcome date", "actual":"Actual close", "model_up":"Model P(up)",
                 "baseline_up":"Baseline P(up)"}), hide_index=True)
    st.download_button("Export notebook CSV", rows.to_csv(index=False), "silversignal_notebook.csv", "text/csv")
    st.stop()
source = st.sidebar.radio("Price history source", ["Yahoo silver futures", "Upload daily CSV"], index=0)
period = st.sidebar.selectbox("Download history", ["10y", "5y", "max"], index=0,
                              disabled=source != "Yahoo silver futures")
uploaded = None
if source == "Upload daily CSV":
    uploaded = st.sidebar.file_uploader("Choose CSV", type=["csv"], help="Columns: Date, Close. At least 650 daily records.")
    st.sidebar.caption("Data must be actual daily silver prices, preferably USD per troy ounce.")
if st.sidebar.button("Refresh downloaded prices", disabled=source != "Yahoo silver futures", use_container_width=True):
    cached_yahoo.clear()
    cached_model.clear()
    st.rerun()
st.sidebar.divider()
st.sidebar.markdown("**What this model uses**")
st.sidebar.markdown("Past price changes, 20/60-day averages, volatility, 14-day RSI, and distance from a 252-day high. It does **not** know tomorrow's news, Fed decisions, mine supply, or industrial demand.")
st.sidebar.markdown("**Data warning**")
st.sidebar.caption("SI=F is a delayed Yahoo Finance continuous front-month *futures* series. Contract rolling can create artificial jumps. A dealer's spot or sterling-scrap quote will differ.")

st.markdown('<div class="hero-eyebrow">HISTORICAL MARKET RESEARCH / SILVER</div>', unsafe_allow_html=True)
st.title("SilverSignal")
st.markdown('<div class="hero-sub">When might silver rise? Study historical price behavior, see a testable prediction, and check whether the model performed better than a simple guess.</div>', unsafe_allow_html=True)

with st.container():
    horizon = st.segmented_control("Forecast horizon (trading sessions)", options=[5, 10, 20, 60],
                                   default=20, format_func=lambda v: {5:"1 week",10:"2 weeks",20:"1 month",60:"3 months"}[v])
if horizon is None:
    horizon = 20

if source == "Upload daily CSV" and uploaded is None:
    st.info("Upload a Date/Close CSV in the sidebar to start. Switch to Yahoo silver futures for automatic history.")
    with st.expander("Example CSV format"):
        st.code("Date,Close\n2024-01-02,23.97\n2024-01-03,23.82\n2024-01-04,23.74", language="csv")
    st.stop()

try:
    with st.spinner("Loading market history and running chronological backtest..."):
        prices = cached_yahoo(period) if source == "Yahoo silver futures" else load_csv(uploaded.getvalue())
        if source == "Yahoo silver futures":
            # A provider can publish a partial daily bar. Only use prior calendar dates.
            market_day = pd.Timestamp.now(tz="America/New_York").date()
            prices = prices.loc[prices["Date"].dt.date < market_day]
        result = cached_model(prices, horizon)
except (DataError, ValueError) as exc:
    st.error(str(exc))
    st.caption("Try another source or check CSV dates and prices.")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected data or forecasting problem: {type(exc).__name__}: {exc}")
    st.stop()

if source == "Yahoo silver futures":
    try:
        recorded = update_notebook(notebook_path(), result, period, pd.Timestamp.now(tz="UTC"))
        if recorded:
            st.caption("Saved in Forecast notebook. The first call for this price date, horizon and history is kept.")
        else:
            st.caption("Price history is too old for a new notebook entry.")
    except (OSError, sqlite3.Error) as exc:
        st.warning(f"Forecast available, but the local notebook could not be updated: {exc}")

freshness = (dt.date.today() - result.last_date.date()).days
if freshness > 7:
    st.warning(f"Latest observation is {freshness} calendar days old. Do not treat this as a current silver forecast.")
label, explanation = signal_label(result)
st.markdown(f'<span class="tag">{label}</span>', unsafe_allow_html=True)
st.caption(explanation)

m1, m2, m3 = st.columns(3)
m1.metric("LAST DATA PRICE", f"${result.last_price:,.2f}")
m2.metric("MODEL: UPWARD ESTIMATE", f"{result.probability_up:.0%}")
m3.metric(f"PROJECTED / {horizon} SESSIONS", f"${result.predicted_price:,.2f}",
          delta=f"{result.predicted_return:+.1%} modeled return")
st.markdown(f"**Indicative 10th–90th percentile range**  "
            f"USD {result.lower_price:,.2f} to USD {result.upper_price:,.2f}")
st.caption(f"Data through {result.last_date:%b %d, %Y} · "+
           ("Yahoo SI=F delayed front-month silver futures" if source == "Yahoo silver futures" else "Your uploaded CSV")+
           " · All dollar prices are per troy ounce if your source is quoted that way. Forecasts are experimental.")

tab1, tab2, tab3 = st.tabs(["Price & outlook", "Did the model work?", "Method & limitations"])

with tab1:
    c_main, c_small = st.columns([2.4,1], gap="large")
    with c_main:
        st.subheader("Silver price history")
        chart_history = result.historic.copy()
        chart_history["MA20"] = chart_history["Close"].rolling(20,min_periods=20).mean()
        chart_history["MA60"] = chart_history["Close"].rolling(60,min_periods=60).mean()
        choice = st.selectbox("Chart window", ["1 year", "3 years", "5 years", "All"], index=0)
        if choice != "All":
            window_days = {"1 year":365, "3 years":1096, "5 years":1826}[choice]
            chart_history = chart_history[chart_history["Date"] >= result.last_date-pd.Timedelta(days=window_days)].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_history["Date"],y=chart_history["Close"],name="Silver close",mode="lines",line=dict(color="#e7c28f",width=2.8)))
        fig.add_trace(go.Scatter(x=chart_history["Date"],y=chart_history["MA20"],name="20-session avg",mode="lines",line=dict(color="#8ac8dc",width=1.5)))
        fig.add_trace(go.Scatter(x=chart_history["Date"],y=chart_history["MA60"],name="60-session avg",mode="lines",line=dict(color="#8b9bc7",width=1.5)))
        future_date = result.last_date + pd.offsets.BDay(horizon)
        fig.add_trace(go.Scatter(x=[result.last_date,future_date],y=[result.last_price,result.predicted_price],name="Model projection",mode="lines+markers",line=dict(color="#f7a96f",dash="dash",width=2),marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=[future_date,future_date],y=[result.lower_price,result.upper_price],name="Error-based band",mode="lines",line=dict(color="#f7a96f",width=9),opacity=.35))
        fig.update_layout(height=420,margin=dict(l=0,r=15,b=20,t=15),paper_bgcolor="#0b121a",plot_bgcolor="#0b121a",
                          font=dict(color="#a6b8c8"),legend=dict(orientation="h",y=-.18,x=0),
                          yaxis_title="USD / troy oz (if market source)",xaxis=dict(showgrid=False),
                          yaxis=dict(gridcolor="#1b2d3b"),hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The forecast endpoint is approximately business-day dated; exchange holidays and futures roll dates are not modeled. The vertical band reflects past holdout residuals, not a guaranteed range.")
    with c_small:
        st.subheader("Technical snapshot")
        direction_text = "above" if result.ma20 > result.ma60 else "below"
        st.metric("20-session average",f"${result.ma20:,.2f}")
        st.metric("60-session average",f"${result.ma60:,.2f}")
        st.metric("RSI (14)",f"{result.last_rsi:.1f}")
        st.markdown(f"The short average is **{direction_text}** the longer average. RSI reflects past momentum—not a turning-point clock.")
        if result.last_rsi >= 70:
            st.caption("RSI is elevated. Markets can stay elevated for a long time.")
        elif result.last_rsi <= 30:
            st.caption("RSI is depressed. Low readings do not guarantee a rebound.")
        else:
            st.caption("RSI is between common 30/70 reference levels.")

with tab2:
    st.subheader("Out-of-sample results")
    st.write("The final quarter of eligible dates is tested in order. Every 20 sessions the model and baseline are refitted using only outcomes that ended before that block began.")
    cols = st.columns(4)
    cols[0].metric("Model direction accuracy", f"{result.backtest_accuracy:.1%}")
    cols[1].metric("Naive direction baseline", f"{result.baseline_accuracy:.1%}")
    cols[2].metric("Model Brier score ↓",f"{result.model_brier:.3f}")
    cols[3].metric("Naive Brier score ↓",f"{result.baseline_brier:.3f}")
    st.caption("Brier score assesses probabilistic forecasts; lower is better. The baseline always guesses the most common training direction, using the training up-rate as its probability.")
    st.write(f"**Test started:** {result.test_start_date:%b %d, %Y} · **Daily test predictions:** {result.backtest_count:,} · **Non-overlapping {horizon}-session windows:** {result.approximate_independent_windows:,} · **Initial training rows:** {result.training_count:,}")
    if result.backtest_accuracy <= result.baseline_accuracy or result.model_brier >= result.baseline_brier:
        st.warning("This model did not outperform the naive direction baseline on both validation metrics. Don't treat its probability as an actionable edge.")
    else:
        st.info("The model beat the naive baseline on these two metrics in this particular historical holdout. That does not establish a reliable trading edge.")
    st.caption(f"Future-return mean absolute error: model {result.model_mae:.2%}, zero-return baseline {result.baseline_mae:.2%}. Horizons overlap: {result.backtest_count} predictions are not {result.backtest_count} independent trials.")
    bt = result.backtest.copy()
    spaced = bt.loc[bt["Non-overlapping origin"]]
    observed = spaced["Observed up"]
    summary = []
    for name, column in [("Model", "Model estimated up probability"), ("Baseline", "Baseline up probability")]:
        probability = spaced[column]
        summary.append({"Forecast": name, "Direction accuracy": f"{((probability >= .5) == observed).mean():.1%}",
                        "Brier score": f"{((probability - observed.astype(int)) ** 2).mean():.3f}"})
    st.markdown("**Non-overlapping outcomes**")
    st.caption(f"Starting at the first test date, score every {horizon}th forecast. These {len(spaced)} outcomes share no daily returns, but can still reflect the same market regime.")
    st.table(pd.DataFrame(summary).set_index("Forecast"))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bt["Date"],y=bt["Model estimated up probability"],name="Estimated up probability",line=dict(color="#e7c28f",width=1.7)))
    fig2.add_hline(y=.5,line_color="#7d97aa",line_dash="dash")
    fig2.update_layout(height=310, margin=dict(l=0,r=10,t=10,b=10),paper_bgcolor="#0b121a",plot_bgcolor="#0b121a",font=dict(color="#a6b8c8"),
                       xaxis=dict(showgrid=False),yaxis=dict(title="Model estimate",range=[0,1],tickformat=".0%",gridcolor="#1b2d3b"))
    st.plotly_chart(fig2,use_container_width=True)
    st.download_button("Export backtest predictions (CSV)",bt.to_csv(index=False).encode("utf-8"),file_name="silver_backtest.csv",mime="text/csv")

with tab3:
    st.subheader("How the model works")
    st.markdown("""
1. Gather hundreds to thousands of **historical daily closes**; remove duplicate, invalid, and missing values.
2. Compute features using only information available **on or before each day's close**: 1/5/20/60-day returns, moving averages, volatility, RSI, and distance to a 252-session high.
3. Label each day by whether silver ended higher **5, 10, 20 or 60 trading sessions later**, depending on your selected horizon.
4. Fit a regularized **logistic regression** for direction and a **ridge regression** for the size of the return.
5. Walk forward through the final quarter in 20-session blocks. Refit on expanding history with a full horizon gap. Score against the updated training up-rate and a zero-return projection.
6. Refit on all fully known historical outcomes, then estimate direction and a potential future price. The display band uses the 10th and 90th percentiles of residuals on the held-out period.
""")
    st.warning("Research tool, not investment advice. No historical-price model can reliably tell you when silver *will* rise. Historical tests can be unstable, affected by changing market regimes, overlapping outcomes, unmodeled news and futures-contract changes. Model probabilities are **not externally calibrated**. Avoid using the estimates alone to buy, sell or time silver.")
    st.markdown("**Raw data & transparency**")
    st.download_button("Download cleaned price history (CSV)",result.historic.to_csv(index=False).encode("utf-8"),file_name="silver_history.csv",mime="text/csv")
    st.markdown("Yahoo Finance silver futures page: [SI=F historical prices](https://finance.yahoo.com/quote/SI%3DF/history/). Historical CSV can instead contain real spot prices, if your own data source provides them.")
    st.caption("SilverSignal v0.3 · No accounts, no payments, no trade execution, no automatic monitoring. Models run locally wherever you host the Streamlit app.")


