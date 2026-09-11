"""
analytics.py
-------------
Higher-level analytics functions that combine the transformations and
database (DuckDB) layers to power each dashboard page: Market Overview,
Technical Analytics, Volatility Analysis, Stock Comparison, and Sector
Analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import database as db
from src import transformations as tr


# ---------------------------------------------------------------------------
# Market Overview
# ---------------------------------------------------------------------------

def market_overview_metrics(con, table_name: str = "market_data") -> dict:
    """High-level KPIs for the Market Overview page."""
    snapshot = db.query_latest_snapshot(con, table_name)
    breadth = db.query_market_breadth(con, table_name)

    if snapshot.empty:
        return {
            "num_tickers": 0,
            "avg_pct_change": 0.0,
            "advancing": 0,
            "declining": 0,
            "latest_date": None,
            "snapshot": snapshot,
            "breadth": breadth,
        }

    return {
        "num_tickers": snapshot["Ticker"].nunique(),
        "avg_pct_change": round(float(snapshot["Pct_Change"].mean()), 2),
        "advancing": int(breadth.loc[breadth["Status"] == "Advancing", "Count"].sum()),
        "declining": int(breadth.loc[breadth["Status"] == "Declining", "Count"].sum()),
        "latest_date": snapshot["Latest_Date"].max(),
        "snapshot": snapshot,
        "breadth": breadth,
    }


# ---------------------------------------------------------------------------
# Technical Analytics
# ---------------------------------------------------------------------------

def build_technical_dataset(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return the enriched OHLCV + indicator dataframe for a single ticker."""
    subset = df[df["Ticker"] == ticker].copy()
    if subset.empty:
        return subset
    subset = tr.apply_full_pipeline(subset)
    return subset


def latest_signal_summary(tech_df: pd.DataFrame) -> dict:
    """Simple human-readable interpretation of the latest technical indicators."""
    if tech_df is None or tech_df.empty:
        return {}

    last = tech_df.iloc[-1]
    signals = {}

    rsi = last.get("RSI_14")
    if pd.notna(rsi):
        if rsi >= 70:
            signals["RSI"] = f"{rsi:.1f} (Overbought)"
        elif rsi <= 30:
            signals["RSI"] = f"{rsi:.1f} (Oversold)"
        else:
            signals["RSI"] = f"{rsi:.1f} (Neutral)"

    macd = last.get("MACD")
    macd_signal = last.get("MACD_Signal")
    if pd.notna(macd) and pd.notna(macd_signal):
        signals["MACD"] = "Bullish crossover" if macd > macd_signal else "Bearish crossover"

    ma20 = last.get("MA_20")
    ma50 = last.get("MA_50")
    if pd.notna(ma20) and pd.notna(ma50):
        signals["Trend (MA20 vs MA50)"] = "Uptrend" if ma20 > ma50 else "Downtrend"

    close = last.get("Close")
    bb_upper = last.get("BB_Upper")
    bb_lower = last.get("BB_Lower")
    if pd.notna(close) and pd.notna(bb_upper) and pd.notna(bb_lower):
        if close >= bb_upper:
            signals["Bollinger Bands"] = "Price at/above upper band"
        elif close <= bb_lower:
            signals["Bollinger Bands"] = "Price at/below lower band"
        else:
            signals["Bollinger Bands"] = "Within bands"

    return signals


# ---------------------------------------------------------------------------
# Volatility Analysis
# ---------------------------------------------------------------------------

def volatility_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Annualized volatility ranked across all tickers in the given dataframe."""
    enriched = tr.add_daily_returns(df)
    vol = tr.compute_annualized_volatility(enriched)
    if vol.empty:
        return vol

    if "Sector" in df.columns:
        sector_map = df.drop_duplicates("Ticker").set_index("Ticker")["Sector"]
        vol["Sector"] = vol["Ticker"].map(sector_map)

    return vol.sort_values("Annualized_Volatility_Pct", ascending=False).reset_index(drop=True)


def rolling_volatility_series(df: pd.DataFrame, ticker: str, window: int = 21) -> pd.DataFrame:
    subset = df[df["Ticker"] == ticker].copy()
    if subset.empty:
        return subset
    subset = tr.add_daily_returns(subset)
    subset = tr.add_rolling_volatility(subset, window=window)
    return subset


def value_at_risk(df: pd.DataFrame, ticker: str, confidence: float = 0.95) -> float | None:
    """Historical (non-parametric) Value at Risk, expressed as a positive percentage."""
    subset = df[df["Ticker"] == ticker].copy()
    subset = tr.add_daily_returns(subset)
    returns = subset["Daily_Return"].dropna()
    if returns.empty:
        return None
    var_pct = -np.percentile(returns, (1 - confidence) * 100) * 100
    return round(float(var_pct), 2)


# ---------------------------------------------------------------------------
# Stock Comparison
# ---------------------------------------------------------------------------

def comparison_table(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Side-by-side comparison table: return, volatility, avg volume, price range."""
    rows = []
    for ticker in tickers:
        subset = df[df["Ticker"] == ticker].copy()
        if subset.empty:
            continue
        subset = tr.add_daily_returns(subset)
        subset = tr.add_cumulative_returns(subset)

        total_return = subset["Cumulative_Return_Pct"].iloc[-1] if not subset.empty else np.nan
        volatility = subset["Log_Return"].std() * np.sqrt(252) * 100 if subset["Log_Return"].notna().any() else np.nan
        sharpe = (
            (subset["Daily_Return"].mean() / subset["Daily_Return"].std()) * np.sqrt(252)
            if subset["Daily_Return"].std() not in (0, np.nan) and subset["Daily_Return"].notna().any()
            else np.nan
        )

        rows.append({
            "Ticker": ticker,
            "Sector": subset["Sector"].iloc[0] if "Sector" in subset.columns else "N/A",
            "Total Return (%)": round(total_return, 2) if pd.notna(total_return) else None,
            "Annualized Volatility (%)": round(volatility, 2) if pd.notna(volatility) else None,
            "Sharpe Ratio (approx.)": round(sharpe, 2) if pd.notna(sharpe) else None,
            "Avg Volume": int(subset["Volume"].mean()) if "Volume" in subset.columns else None,
            "Latest Close": round(subset["Close"].iloc[-1], 2),
        })
    return pd.DataFrame(rows)


def normalized_price_series(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Prices indexed to 100 at the start of the period, for fair visual comparison."""
    frames = []
    for ticker in tickers:
        subset = df[df["Ticker"] == ticker].sort_values("Date").copy()
        if subset.empty:
            continue
        base = subset["Close"].iloc[0]
        if base == 0 or pd.isna(base):
            continue
        subset["Indexed_Price"] = subset["Close"] / base * 100
        frames.append(subset[["Date", "Ticker", "Indexed_Price"]])
    if not frames:
        return pd.DataFrame(columns=["Date", "Ticker", "Indexed_Price"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Sector Analysis
# ---------------------------------------------------------------------------

def sector_summary(con, table_name: str = "market_data") -> pd.DataFrame:
    return db.query_sector_performance(con, table_name)


def sector_composition(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Sector" not in df.columns:
        return pd.DataFrame(columns=["Sector", "Num_Stocks"])
    return (
        df.drop_duplicates("Ticker")
        .groupby("Sector")["Ticker"]
        .nunique()
        .reset_index(name="Num_Stocks")
        .sort_values("Num_Stocks", ascending=False)
    )
