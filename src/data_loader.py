"""
data_loader.py
----------------
Handles all data ingestion for MarketPulse:
  * Live data fetching from Yahoo Finance via yfinance
  * Fallback to bundled demo/sample CSV data when live data is unavailable
  * Streamlit caching to avoid redundant network calls
  * Defensive error handling so the app never crashes on bad/missing data
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_data.csv")

# Default universe of tickers tracked by the dashboard, mapped to their sector.
# This mirrors the tickers available in the bundled sample_data.csv so that the
# live-data path and the fallback path stay consistent.
DEFAULT_TICKERS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "NVDA": "Technology",
    "META": "Technology",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "JPM": "Financials",
    "BAC": "Financials",
    "V": "Financials",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "UNH": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "DIS": "Communication Services",
    "NFLX": "Communication Services",
}

REQUIRED_COLUMNS = ["Date", "Ticker", "Sector", "Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Sample / fallback data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    """
    Load the bundled demo dataset from data/sample_data.csv.
    This guarantees the app always has something to display, even with
    no internet connection or when yfinance is rate-limited / unreachable.
    """
    try:
        df = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["Date"])
        df = _clean_ohlcv(df)
        return df
    except FileNotFoundError:
        st.error(
            f"Sample data file not found at `{SAMPLE_DATA_PATH}`. "
            "Please make sure data/sample_data.csv exists in the repository."
        )
        return _empty_frame()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load sample data: {exc}")
        return _empty_frame()


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes, drop bad rows, sort chronologically."""
    if df.empty:
        return df

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close"])
    df = df[df["Close"] > 0]
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Live data (yfinance)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_live_data(
    tickers: Iterable[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch live OHLCV data from Yahoo Finance for a list of tickers.

    Returns a tidy long-format DataFrame with columns:
        Date, Ticker, Sector, Open, High, Low, Close, Volume

    On any failure (network error, empty response, bad ticker, rate limiting)
    the offending ticker is simply skipped -- this function never raises.
    An empty DataFrame is returned if nothing could be fetched, and the
    caller is expected to fall back to sample data in that case.
    """
    try:
        import yfinance as yf
    except ImportError:
        st.warning("yfinance is not installed. Falling back to demo data.")
        return _empty_frame()

    tickers = list(tickers)
    if not tickers:
        return _empty_frame()

    frames = []
    failed_tickers = []

    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period=period, interval=interval)
            if hist is None or hist.empty:
                failed_tickers.append(ticker)
                continue

            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else hist.columns[0]

            tidy = pd.DataFrame({
                "Date": pd.to_datetime(hist[date_col]).dt.tz_localize(None),
                "Ticker": ticker,
                "Sector": DEFAULT_TICKERS.get(ticker, "Other"),
                "Open": hist.get("Open"),
                "High": hist.get("High"),
                "Low": hist.get("Low"),
                "Close": hist.get("Close"),
                "Volume": hist.get("Volume"),
            })
            frames.append(tidy)
        except Exception:  # noqa: BLE001
            failed_tickers.append(ticker)
            continue

    if not frames:
        return _empty_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined = _clean_ohlcv(combined)

    if failed_tickers:
        st.session_state["_failed_tickers"] = failed_tickers

    return combined


# ---------------------------------------------------------------------------
# Unified loader with automatic fallback
# ---------------------------------------------------------------------------

def get_market_data(
    tickers: Iterable[str] | None = None,
    period: str = "1y",
    use_live: bool = True,
) -> tuple[pd.DataFrame, str]:
    """
    Primary entry point used by the app to obtain market data.

    Tries live yfinance data first (if use_live=True); if that fails or
    returns no usable rows, transparently falls back to the bundled sample
    dataset. Returns a tuple of (dataframe, source_label) where source_label
    is either "live" or "demo" so the UI can inform the user which data
    source is currently active.
    """
    tickers = list(tickers) if tickers else list(DEFAULT_TICKERS.keys())

    if use_live:
        try:
            live_df = fetch_live_data(tickers=tickers, period=period)
        except Exception:  # noqa: BLE001
            live_df = _empty_frame()

        if live_df is not None and not live_df.empty:
            # Require a reasonable minimum amount of data to trust the live feed
            if live_df["Ticker"].nunique() >= max(1, len(tickers) // 3):
                return live_df, "live"

    sample_df = load_sample_data()
    if tickers:
        filtered = sample_df[sample_df["Ticker"].isin(tickers)]
        if not filtered.empty:
            return filtered.reset_index(drop=True), "demo"
    return sample_df, "demo"


@st.cache_data(show_spinner=False)
def get_available_tickers(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return list(DEFAULT_TICKERS.keys())
    return sorted(df["Ticker"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def get_available_sectors(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "Sector" not in df.columns:
        return sorted(set(DEFAULT_TICKERS.values()))
    return sorted(df["Sector"].dropna().unique().tolist())
