"""
MarketPulse - Stock Market Analytics Dashboard
================================================
A Streamlit + DuckDB + yfinance dashboard for exploring market data,
technical indicators, volatility, comparisons, and sector performance.

Run locally:
    streamlit run app.py

Deploys directly on Streamlit Community Cloud with no API keys, Docker,
or environment variables required. If live data cannot be fetched (no
internet, rate limiting, etc.) the app automatically falls back to a
bundled demo dataset so it always works.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import data_loader as dl
from src import database as db
from src import transformations as tr

# ---------------------------------------------------------------------------
# Page configuration & style
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MarketPulse | Stock Market Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main > div { padding-top: 1.2rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .mp-header {
        padding: 1.1rem 1.4rem;
        border-radius: 12px;
        background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .mp-header h1 { margin: 0; font-size: 1.9rem; }
    .mp-header p { margin: 0.2rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
    .mp-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .mp-badge-live { background: #16a34a33; color: #16a34a; border: 1px solid #16a34a; }
    .mp-badge-demo { background: #f59e0b33; color: #b45309; border: 1px solid #f59e0b; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("📈 MarketPulse")
st.sidebar.caption("Stock Market Analytics Dashboard")

all_tickers = list(dl.DEFAULT_TICKERS.keys())

data_mode = st.sidebar.radio(
    "Data Source",
    options=["Live (Yahoo Finance)", "Demo Data"],
    index=0,
    help="Live data is fetched from Yahoo Finance via yfinance. "
         "If it's unavailable, MarketPulse automatically falls back to bundled demo data.",
)
use_live = data_mode.startswith("Live")

period_label_to_code = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
}
period_label = st.sidebar.selectbox("Live Data Lookback", list(period_label_to_code.keys()), index=3)
period_code = period_label_to_code[period_label]

selected_tickers = st.sidebar.multiselect(
    "Tickers to Load",
    options=all_tickers,
    default=all_tickers[:10],
    help="Choose which tickers to pull into the dashboard.",
)

if not selected_tickers:
    st.sidebar.warning("Select at least one ticker to continue.")
    st.stop()

refresh = st.sidebar.button("🔄 Refresh Data", width="stretch")
if refresh:
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, DuckDB, Plotly & yfinance.")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=True, ttl=3600)
def load_and_prepare(tickers: tuple, period: str, live: bool):
    raw_df, source = dl.get_market_data(tickers=list(tickers), period=period, use_live=live)
    clean_df = tr.clean_ohlcv(raw_df)
    return clean_df, source


try:
    with st.spinner("Loading market data..."):
        market_df, data_source = load_and_prepare(tuple(sorted(selected_tickers)), period_code, use_live)
except Exception as exc:  # noqa: BLE001
    st.error(f"Unexpected error while loading data: {exc}")
    market_df, data_source = dl.load_sample_data(), "demo"

if market_df is None or market_df.empty:
    st.error(
        "No market data could be loaded from either the live feed or the bundled "
        "demo dataset. Please check that `data/sample_data.csv` exists in the repository."
    )
    st.stop()

badge_class = "mp-badge-live" if data_source == "live" else "mp-badge-demo"
badge_text = "LIVE DATA" if data_source == "live" else "DEMO DATA"

st.markdown(
    f"""
    <div class="mp-header">
        <h1>📈 MarketPulse <span class="mp-badge {badge_class}">{badge_text}</span></h1>
        <p>Real-time-friendly stock market analytics — Market Overview, Technical Signals,
        Volatility, Comparisons & Sector Insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if data_source == "demo":
    st.info(
        "Showing bundled **demo data** (live Yahoo Finance data was unavailable or "
        "insufficient for the selected tickers). All analytics below still work fully "
        "against this dataset.",
        icon="ℹ️",
    )

failed = st.session_state.get("_failed_tickers")
if failed and data_source == "live":
    st.warning(f"Could not fetch live data for: {', '.join(failed)}. They were skipped.", icon="⚠️")


# ---------------------------------------------------------------------------
# DuckDB setup
# ---------------------------------------------------------------------------

con = db.get_connection()
db.load_dataframe(con, market_df, table_name="market_data")

available_tickers = dl.get_available_tickers(market_df)
available_sectors = dl.get_available_sectors(market_df)


# ---------------------------------------------------------------------------
# Tabs / Pages
# ---------------------------------------------------------------------------

tab_overview, tab_explorer, tab_technical, tab_volatility, tab_compare, tab_sector = st.tabs(
    [
        "🏠 Market Overview",
        "🔍 Stock Explorer",
        "📉 Technical Analytics",
        "🌪️ Volatility Analysis",
        "⚖️ Stock Comparison",
        "🏢 Sector Analysis",
    ]
)

# --- Market Overview -------------------------------------------------------
with tab_overview:
    st.subheader("Market Overview")

    metrics = an.market_overview_metrics(con, "market_data")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tracked Tickers", metrics["num_tickers"])
    col2.metric("Avg. % Change", f"{metrics['avg_pct_change']}%")
    col3.metric("Advancing", metrics["advancing"])
    col4.metric("Declining", metrics["declining"])

    st.markdown("##### Latest Snapshot")
    snapshot = metrics["snapshot"]
    if not snapshot.empty:
        display_snap = snapshot.copy()
        display_snap["Latest_Date"] = pd.to_datetime(display_snap["Latest_Date"]).dt.date
        st.dataframe(
            display_snap.rename(columns={
                "Latest_Close": "Close",
                "Prior_Close": "Prior Close",
                "Latest_Volume": "Volume",
                "Pct_Change": "% Change",
            }),
            width="stretch",
            hide_index=True,
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 🚀 Top Gainers")
        gainers, losers = db.query_top_movers(con, "market_data", limit=5)
        if not gainers.empty:
            fig = px.bar(
                gainers, x="Ticker", y="Pct_Change", color="Pct_Change",
                color_continuous_scale="Greens", text="Pct_Change",
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="% Change", coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Not enough data to compute movers yet.")

    with col_b:
        st.markdown("##### 📉 Top Losers")
        if not losers.empty:
            fig = px.bar(
                losers, x="Ticker", y="Pct_Change", color="Pct_Change",
                color_continuous_scale="Reds_r", text="Pct_Change",
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="% Change", coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Not enough data to compute movers yet.")

    st.markdown("##### Market Breadth")
    breadth = metrics["breadth"]
    if not breadth.empty and breadth["Count"].sum() > 0:
        fig = px.pie(breadth, names="Status", values="Count", hole=0.5,
                      color="Status",
                      color_discrete_map={"Advancing": "#16a34a", "Declining": "#dc2626", "Unchanged": "#9ca3af"})
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### 📊 Volume Leaders")
    vol_leaders = db.query_volume_leaders(con, "market_data", limit=10)
    if not vol_leaders.empty:
        fig = px.bar(vol_leaders, x="Ticker", y="Avg_Volume", color="Sector")
        fig.update_layout(yaxis_title="Average Volume")
        st.plotly_chart(fig, width="stretch")


# --- Stock Explorer ----------------------------------------------------------
with tab_explorer:
    st.subheader("Stock Explorer")

    explorer_ticker = st.selectbox("Select a ticker", available_tickers, key="explorer_ticker")
    ticker_df = market_df[market_df["Ticker"] == explorer_ticker].sort_values("Date")

    if ticker_df.empty:
        st.warning("No data available for this ticker.")
    else:
        summary = db.query_summary_stats(con, explorer_ticker, "market_data")
        if not summary.empty:
            s = summary.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Records", int(s["Num_Records"]))
            c2.metric("Min Close", f"${s['Min_Close']:.2f}")
            c3.metric("Max Close", f"${s['Max_Close']:.2f}")
            c4.metric("Avg Close", f"${s['Avg_Close']:.2f}")
            c5.metric("Avg Volume", f"{int(s['Avg_Volume']):,}")

        chart_type = st.radio("Chart type", ["Candlestick", "Line"], horizontal=True)

        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=ticker_df["Date"], open=ticker_df["Open"], high=ticker_df["High"],
                low=ticker_df["Low"], close=ticker_df["Close"], name=explorer_ticker,
            )])
        else:
            fig = px.line(ticker_df, x="Date", y="Close", title=None)

        fig.update_layout(
            title=f"{explorer_ticker} Price History",
            xaxis_title="Date", yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=(chart_type == "Candlestick"),
        )
        st.plotly_chart(fig, width="stretch")

        vol_fig = px.bar(ticker_df, x="Date", y="Volume", title=f"{explorer_ticker} Trading Volume")
        st.plotly_chart(vol_fig, width="stretch")

        with st.expander("View raw data"):
            st.dataframe(ticker_df, width="stretch", hide_index=True)


# --- Technical Analytics ----------------------------------------------------
with tab_technical:
    st.subheader("Technical Analytics")

    tech_ticker = st.selectbox("Select a ticker", available_tickers, key="tech_ticker")
    tech_df = an.build_technical_dataset(market_df, tech_ticker)

    if tech_df.empty:
        st.warning("Not enough data to compute technical indicators for this ticker.")
    else:
        signals = an.latest_signal_summary(tech_df)
        if signals:
            cols = st.columns(len(signals))
            for col, (name, value) in zip(cols, signals.items()):
                col.metric(name, value)

        st.markdown("##### Price with Moving Averages & Bollinger Bands")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["Close"], name="Close", line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["MA_20"], name="MA 20", line=dict(color="#f59e0b")))
        fig.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["MA_50"], name="MA 50", line=dict(color="#16a34a")))
        fig.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["BB_Upper"], name="BB Upper",
                                  line=dict(color="rgba(150,150,150,0.5)", dash="dot")))
        fig.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["BB_Lower"], name="BB Lower",
                                  line=dict(color="rgba(150,150,150,0.5)", dash="dot"),
                                  fill="tonexty", fillcolor="rgba(150,150,150,0.1)"))
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, width="stretch")

        col_r, col_m = st.columns(2)
        with col_r:
            st.markdown("##### RSI (14)")
            fig_rsi = px.line(tech_df, x="Date", y="RSI_14")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(yaxis_range=[0, 100], yaxis_title="RSI")
            st.plotly_chart(fig_rsi, width="stretch")

        with col_m:
            st.markdown("##### MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=tech_df["Date"], y=tech_df["MACD_Hist"], name="Histogram",
                                       marker_color="rgba(100,100,100,0.4)"))
            fig_macd.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["MACD"], name="MACD", line=dict(color="#2563eb")))
            fig_macd.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["MACD_Signal"], name="Signal", line=dict(color="#f59e0b")))
            fig_macd.update_layout(yaxis_title="MACD")
            st.plotly_chart(fig_macd, width="stretch")

        with st.expander("View enriched data table"):
            st.dataframe(tech_df, width="stretch", hide_index=True)


# --- Volatility Analysis -----------------------------------------------------
with tab_volatility:
    st.subheader("Volatility Analysis")

    vol_ranking = an.volatility_ranking(market_df)
    if vol_ranking.empty:
        st.warning("Not enough data to compute volatility.")
    else:
        st.markdown("##### Annualized Volatility Ranking")
        fig = px.bar(
            vol_ranking, x="Ticker", y="Annualized_Volatility_Pct",
            color="Sector" if "Sector" in vol_ranking.columns else None,
        )
        fig.update_layout(yaxis_title="Annualized Volatility (%)")
        st.plotly_chart(fig, width="stretch")

        vol_ticker = st.selectbox("Inspect rolling volatility for", available_tickers, key="vol_ticker")
        window = st.slider("Rolling window (trading days)", min_value=5, max_value=60, value=21, step=1)
        roll_df = an.rolling_volatility_series(market_df, vol_ticker, window=window)

        vol_col = f"Volatility_{window}d"
        if not roll_df.empty and vol_col in roll_df.columns:
            fig2 = px.line(roll_df, x="Date", y=vol_col, title=f"{vol_ticker} — {window}-Day Rolling Volatility")
            fig2.update_layout(yaxis_title="Annualized Volatility (%)")
            st.plotly_chart(fig2, width="stretch")

        var_95 = an.value_at_risk(market_df, vol_ticker, confidence=0.95)
        if var_95 is not None:
            st.metric(f"{vol_ticker} — 1-Day Historical VaR (95% confidence)", f"{var_95}%")
            st.caption(
                "Interpretation: on 95% of trading days, the single-day loss for this "
                "stock is not expected to exceed this percentage, based on historical returns."
            )

        with st.expander("View volatility ranking table"):
            st.dataframe(vol_ranking, width="stretch", hide_index=True)


# --- Stock Comparison ---------------------------------------------------------
with tab_compare:
    st.subheader("Stock Comparison")

    compare_tickers = st.multiselect(
        "Select 2+ tickers to compare",
        options=available_tickers,
        default=available_tickers[: min(4, len(available_tickers))],
        key="compare_tickers",
    )

    if len(compare_tickers) < 2:
        st.info("Select at least two tickers to compare.")
    else:
        norm_df = an.normalized_price_series(market_df, compare_tickers)
        if not norm_df.empty:
            fig = px.line(
                norm_df, x="Date", y="Indexed_Price", color="Ticker",
                title="Indexed Price Comparison (Start = 100)",
            )
            fig.update_layout(yaxis_title="Indexed Price")
            st.plotly_chart(fig, width="stretch")

        table = an.comparison_table(market_df, compare_tickers)
        if not table.empty:
            st.markdown("##### Comparison Metrics")
            st.dataframe(table, width="stretch", hide_index=True)

        st.markdown("##### Correlation Matrix (Daily Closing Prices)")
        corr_input = db.query_correlation_matrix_input(con, compare_tickers, "market_data")
        if not corr_input.empty and corr_input.shape[1] > 1:
            corr_matrix = corr_input.corr()
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto",
            )
            st.plotly_chart(fig_corr, width="stretch")
        else:
            st.caption("Not enough overlapping data to compute correlations.")


# --- Sector Analysis -----------------------------------------------------------
with tab_sector:
    st.subheader("Sector Analysis")

    sector_perf = an.sector_summary(con, "market_data")
    composition = an.sector_composition(market_df)

    col1, col2 = st.columns([3, 2])
    with col1:
        if not sector_perf.empty:
            st.markdown("##### Sector Performance (Avg. Daily Return vs Volatility)")
            fig = px.scatter(
                sector_perf, x="Annualized_Volatility_Pct", y="Avg_Daily_Return_Pct",
                size="Num_Stocks", color="Sector", text="Sector",
                labels={
                    "Annualized_Volatility_Pct": "Annualized Volatility (%)",
                    "Avg_Daily_Return_Pct": "Avg. Daily Return (%)",
                },
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, width="stretch")

    with col2:
        if not composition.empty:
            st.markdown("##### Sector Composition")
            fig = px.pie(composition, names="Sector", values="Num_Stocks", hole=0.4)
            st.plotly_chart(fig, width="stretch")

    if not sector_perf.empty:
        st.markdown("##### Cumulative Sector Returns")
        fig = px.bar(
            sector_perf.sort_values("Cumulative_Sum_Return_Pct", ascending=False),
            x="Sector", y="Cumulative_Sum_Return_Pct", color="Sector",
        )
        fig.update_layout(yaxis_title="Cumulative Return (%)", showlegend=False)
        st.plotly_chart(fig, width="stretch")

        with st.expander("View sector performance table"):
            st.dataframe(sector_perf, width="stretch", hide_index=True)

    sector_filter = st.selectbox("Drill into sector", ["All"] + available_sectors, key="sector_drill")
    if sector_filter != "All":
        sector_stocks = market_df[market_df["Sector"] == sector_filter]
        sector_tickers = sorted(sector_stocks["Ticker"].unique().tolist())
        st.markdown(f"##### {sector_filter} — Stocks: {', '.join(sector_tickers)}")
        norm_df = an.normalized_price_series(market_df, sector_tickers)
        if not norm_df.empty:
            fig = px.line(norm_df, x="Date", y="Indexed_Price", color="Ticker",
                          title=f"{sector_filter} — Indexed Price Comparison")
            st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "MarketPulse Dashboard · Data via Yahoo Finance (yfinance) with automatic demo-data "
    "fallback · Built with Streamlit, DuckDB, Pandas, NumPy & Plotly. "
    "For educational/informational purposes only — not investment advice."
)
