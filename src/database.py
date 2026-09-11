"""
database.py
------------
DuckDB integration for MarketPulse. Provides an in-memory DuckDB connection
that market data is registered into, plus a library of real SQL queries used
for analytics (market overview stats, sector aggregation, top movers,
correlation inputs, comparisons, etc).

DuckDB is used here specifically for its ability to run fast, expressive SQL
directly against Pandas DataFrames without needing an external database
server -- ideal for a serverless Streamlit Cloud deployment.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st


@st.cache_resource(show_spinner=False)
def get_connection() -> duckdb.DuckDBPyConnection:
    """Create (once, cached) an in-memory DuckDB connection for this session."""
    return duckdb.connect(database=":memory:")


def load_dataframe(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, table_name: str = "market_data") -> None:
    """Register/replace a DataFrame as a queryable DuckDB table."""
    if df is None:
        df = pd.DataFrame(columns=["Date", "Ticker", "Sector", "Open", "High", "Low", "Close", "Volume"])
    con.register("_tmp_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df")
    con.unregister("_tmp_df")


def run_query(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    """Run an arbitrary SQL query and return the result as a DataFrame."""
    try:
        if params:
            return con.execute(sql, params).fetchdf()
        return con.execute(sql).fetchdf()
    except Exception as exc:  # noqa: BLE001
        st.error(f"SQL query failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Prebuilt analytical queries
# ---------------------------------------------------------------------------

def query_latest_snapshot(con: duckdb.DuckDBPyConnection, table_name: str = "market_data") -> pd.DataFrame:
    """Latest close, prior close, and % change per ticker (SQL window functions)."""
    sql = f"""
        WITH ranked AS (
            SELECT
                Ticker,
                Sector,
                Date,
                Close,
                Volume,
                ROW_NUMBER() OVER (PARTITION BY Ticker ORDER BY Date DESC) AS rn
            FROM {table_name}
        ),
        latest AS (
            SELECT * FROM ranked WHERE rn = 1
        ),
        prior AS (
            SELECT * FROM ranked WHERE rn = 2
        )
        SELECT
            l.Ticker,
            l.Sector,
            l.Date AS Latest_Date,
            l.Close AS Latest_Close,
            p.Close AS Prior_Close,
            l.Volume AS Latest_Volume,
            ROUND(((l.Close - p.Close) / NULLIF(p.Close, 0)) * 100, 2) AS Pct_Change
        FROM latest l
        LEFT JOIN prior p USING (Ticker)
        ORDER BY Pct_Change DESC
    """
    return run_query(con, sql)


def query_top_movers(con: duckdb.DuckDBPyConnection, table_name: str = "market_data", limit: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top gainers and top losers based on latest % change."""
    snapshot = query_latest_snapshot(con, table_name)
    if snapshot.empty:
        return snapshot, snapshot
    gainers = snapshot.sort_values("Pct_Change", ascending=False).head(limit)
    losers = snapshot.sort_values("Pct_Change", ascending=True).head(limit)
    return gainers, losers


def query_sector_performance(con: duckdb.DuckDBPyConnection, table_name: str = "market_data") -> pd.DataFrame:
    """Average return and volatility aggregated by sector using SQL."""
    sql = f"""
        WITH daily AS (
            SELECT
                Ticker,
                Sector,
                Date,
                Close,
                Close / LAG(Close) OVER (PARTITION BY Ticker ORDER BY Date) - 1 AS Daily_Return
            FROM {table_name}
        )
        SELECT
            Sector,
            COUNT(DISTINCT Ticker) AS Num_Stocks,
            ROUND(AVG(Daily_Return) * 100, 4) AS Avg_Daily_Return_Pct,
            ROUND(STDDEV(Daily_Return) * SQRT(252) * 100, 2) AS Annualized_Volatility_Pct,
            ROUND(SUM(Daily_Return) * 100, 2) AS Cumulative_Sum_Return_Pct
        FROM daily
        WHERE Daily_Return IS NOT NULL
        GROUP BY Sector
        ORDER BY Avg_Daily_Return_Pct DESC
    """
    return run_query(con, sql)


def query_ticker_history(con: duckdb.DuckDBPyConnection, ticker: str, table_name: str = "market_data") -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM {table_name}
        WHERE Ticker = ?
        ORDER BY Date
    """
    return run_query(con, sql, params=[ticker])


def query_volume_leaders(con: duckdb.DuckDBPyConnection, table_name: str = "market_data", limit: int = 10) -> pd.DataFrame:
    sql = f"""
        SELECT
            Ticker,
            Sector,
            ROUND(AVG(Volume), 0) AS Avg_Volume,
            MAX(Volume) AS Max_Volume
        FROM {table_name}
        GROUP BY Ticker, Sector
        ORDER BY Avg_Volume DESC
        LIMIT {int(limit)}
    """
    return run_query(con, sql)


def query_correlation_matrix_input(con: duckdb.DuckDBPyConnection, tickers: list[str], table_name: str = "market_data") -> pd.DataFrame:
    """Pivot table of daily close prices by ticker, suitable for a correlation matrix."""
    if not tickers:
        return pd.DataFrame()
    placeholders = ", ".join([f"'{t}'" for t in tickers])
    sql = f"""
        SELECT Date, Ticker, Close
        FROM {table_name}
        WHERE Ticker IN ({placeholders})
        ORDER BY Date
    """
    long_df = run_query(con, sql)
    if long_df.empty:
        return long_df
    pivot = long_df.pivot_table(index="Date", columns="Ticker", values="Close")
    return pivot


def query_summary_stats(con: duckdb.DuckDBPyConnection, ticker: str, table_name: str = "market_data") -> pd.DataFrame:
    sql = f"""
        SELECT
            Ticker,
            COUNT(*) AS Num_Records,
            MIN(Date) AS Start_Date,
            MAX(Date) AS End_Date,
            ROUND(MIN(Close), 2) AS Min_Close,
            ROUND(MAX(Close), 2) AS Max_Close,
            ROUND(AVG(Close), 2) AS Avg_Close,
            ROUND(AVG(Volume), 0) AS Avg_Volume
        FROM {table_name}
        WHERE Ticker = ?
        GROUP BY Ticker
    """
    return run_query(con, sql, params=[ticker])


def query_market_breadth(con: duckdb.DuckDBPyConnection, table_name: str = "market_data") -> pd.DataFrame:
    """Count of advancing vs declining stocks on the latest trading date."""
    snapshot = query_latest_snapshot(con, table_name)
    if snapshot.empty:
        return pd.DataFrame(columns=["Status", "Count"])
    advancing = int((snapshot["Pct_Change"] > 0).sum())
    declining = int((snapshot["Pct_Change"] < 0).sum())
    unchanged = int((snapshot["Pct_Change"] == 0).sum())
    return pd.DataFrame({
        "Status": ["Advancing", "Declining", "Unchanged"],
        "Count": [advancing, declining, unchanged],
    })
