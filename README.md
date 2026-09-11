# 📈 MarketPulse — Stock Market Analytics Dashboard

MarketPulse is a professional, interactive stock market analytics dashboard built with
**Streamlit**, **Pandas**, **NumPy**, **Plotly**, **DuckDB**, and **yfinance**. It requires
**no API keys, Docker, or environment variables** and deploys directly to Streamlit
Community Cloud.

## Features

- **Market Overview** — KPIs, top gainers/losers, market breadth, and volume leaders.
- **Stock Explorer** — Candlestick/line charts, volume, and summary stats per ticker.
- **Technical Analytics** — Moving averages, EMA, MACD, RSI (14), Bollinger Bands, with
  plain-language signal interpretation.
- **Volatility Analysis** — Annualized volatility ranking, rolling volatility, and
  historical Value-at-Risk (VaR).
- **Stock Comparison** — Indexed price comparison, return/volatility/Sharpe table, and a
  correlation matrix.
- **Sector Analysis** — Sector performance (return vs. volatility), composition, and
  cumulative sector returns computed with real SQL.
- **Real DuckDB SQL analytics** — Window functions, aggregations, and pivots run directly
  against in-memory DuckDB tables (see `src/database.py`).
- **ETL pipeline** — Cleaning, returns, indicators, and volatility enrichment
  (see `src/transformations.py`).
- **Live data + fallback** — Live prices via `yfinance`; if unavailable, the app
  automatically falls back to a bundled demo dataset (`data/sample_data.csv`) so it
  always works, even offline.
- **Streamlit caching** — `@st.cache_data` / `@st.cache_resource` used throughout to
  minimize redundant network and computation cost.

## Project Structure

```
marketpulse/
├── app.py                    # Main Streamlit application (entry point)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # yfinance fetching, CSV fallback, caching
│   ├── transformations.py    # Pandas/NumPy ETL & technical indicators
│   ├── database.py           # DuckDB connection & SQL analytics queries
│   └── analytics.py          # Page-level analytics combining the above
├── data/
│   └── sample_data.csv       # Bundled demo/fallback dataset (21 tickers, 2 years)
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Data Sources

- **Live**: Pulled on demand from Yahoo Finance via the `yfinance` package. No API key
  is required.
- **Demo/Fallback**: `data/sample_data.csv` contains ~2 years of synthetic-but-realistic
  daily OHLCV data for 21 tickers across 7 sectors. Used automatically if live data
  cannot be fetched (no internet access, rate limiting, or an unusable response).

You can switch between **Live** and **Demo Data** at any time from the sidebar.

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, branch, and set the main file path to `app.py`.
4. Click **Deploy** — no secrets or environment variables are needed.

## Tech Stack

| Layer          | Technology            |
|----------------|------------------------|
| UI / App       | Streamlit              |
| Data wrangling | Pandas, NumPy          |
| Charts         | Plotly                 |
| SQL analytics  | DuckDB (in-memory)     |
| Market data    | yfinance (Yahoo Finance) |

## Disclaimer

MarketPulse is for educational and informational purposes only. Nothing in this
dashboard constitutes financial or investment advice.
