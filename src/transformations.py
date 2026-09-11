"""
transformations.py
--------------------
Pandas / NumPy based data transformations and technical indicators used
throughout MarketPulse: returns, moving averages, RSI, MACD, Bollinger Bands,
volatility metrics, and general ETL-style cleaning helpers.

All functions are defensive: they handle empty/short input gracefully and
never raise on missing data, returning NaN-filled columns instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic ETL / cleaning helpers
# ---------------------------------------------------------------------------

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dtypes, drop invalid rows, sort by Ticker/Date."""
    if df is None or df.empty:
        return df

    df = df.copy()
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def filter_by_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and log daily returns per ticker."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()
    df["Log_Return"] = df.groupby("Ticker")["Close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    return df


def add_cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative return (indexed to first available price = 0%)."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])

    def _cum(close: pd.Series) -> pd.Series:
        base = close.iloc[0]
        if base == 0 or pd.isna(base):
            return pd.Series(np.nan, index=close.index)
        return (close / base - 1.0) * 100.0

    df["Cumulative_Return_Pct"] = df.groupby("Ticker")["Close"].transform(_cum)
    return df


# ---------------------------------------------------------------------------
# Moving averages & trend indicators
# ---------------------------------------------------------------------------

def add_moving_averages(df: pd.DataFrame, windows=(20, 50, 200)) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    for w in windows:
        col = f"MA_{w}"
        df[col] = df.groupby("Ticker")["Close"].transform(
            lambda s, w=w: s.rolling(window=w, min_periods=max(1, w // 4)).mean()
        )
    return df


def add_ema(df: pd.DataFrame, spans=(12, 26)) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])
    for span in spans:
        col = f"EMA_{span}"
        df[col] = df.groupby("Ticker")["Close"].transform(
            lambda s, span=span: s.ewm(span=span, adjust=False).mean()
        )
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])

    def _macd(group: pd.DataFrame) -> pd.DataFrame:
        ema_fast = group["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = group["Close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            "MACD": macd_line,
            "MACD_Signal": signal_line,
            "MACD_Hist": histogram,
        }, index=group.index)

    macd_results = df.groupby("Ticker", group_keys=False).apply(_macd)
    df[["MACD", "MACD_Signal", "MACD_Hist"]] = macd_results
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])

    def _rsi(close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # neutral when insufficient data
        return rsi

    df["RSI_14"] = df.groupby("Ticker")["Close"].transform(_rsi)
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"])

    def _bb(group: pd.DataFrame) -> pd.DataFrame:
        mid = group["Close"].rolling(window=window, min_periods=max(1, window // 4)).mean()
        std = group["Close"].rolling(window=window, min_periods=max(1, window // 4)).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        return pd.DataFrame({
            "BB_Mid": mid,
            "BB_Upper": upper,
            "BB_Lower": lower,
        }, index=group.index)

    bb_results = df.groupby("Ticker", group_keys=False).apply(_bb)
    df[["BB_Mid", "BB_Upper", "BB_Lower"]] = bb_results
    return df


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def add_rolling_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualized rolling volatility based on daily log returns."""
    if df is None or df.empty:
        return df
    df = df.copy()
    if "Log_Return" not in df.columns:
        df = add_daily_returns(df)

    df = df.sort_values(["Ticker", "Date"])
    df[f"Volatility_{window}d"] = df.groupby("Ticker")["Log_Return"].transform(
        lambda s: s.rolling(window=window, min_periods=max(2, window // 4)).std() * np.sqrt(252) * 100
    )
    return df


def compute_annualized_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Single annualized volatility figure per ticker over the whole period."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Annualized_Volatility_Pct"])

    if "Log_Return" not in df.columns:
        df = add_daily_returns(df)

    result = (
        df.groupby("Ticker")["Log_Return"]
        .std()
        .mul(np.sqrt(252) * 100)
        .reset_index(name="Annualized_Volatility_Pct")
    )
    return result


def apply_full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full ETL/enrichment pipeline used across the app in one call."""
    df = clean_ohlcv(df)
    df = add_daily_returns(df)
    df = add_cumulative_returns(df)
    df = add_moving_averages(df)
    df = add_ema(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_rolling_volatility(df)
    return df
