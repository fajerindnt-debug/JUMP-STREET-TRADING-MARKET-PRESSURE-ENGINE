# JUMP STREET TRADING — Market Pressure Engine

Streamlit dashboard for XAU/USD using Twelve Data OHLC candles. Signal-only; no automated execution.

## Features
- H1 / M15 / M5 / M1 multi-timeframe pressure proxy
- M5-centered pressure weighting
- Momentum, acceleration, EMA20/EMA50 trend context
- Market state: bullish dominance, bearish dominance, balanced/transition
- Candlestick chart and quant diagnostics
- CSV export
- Secrets-based API key configuration

## Deploy
1. Create a GitHub repository and upload `app.py`, `market_pressure.py`, `requirements.txt`, `README.md`, `.gitignore`.
2. Deploy that repository on Streamlit Community Cloud; main file is `app.py`.
3. In app Settings -> Secrets, add:
   `TWELVE_DATA_API_KEY = "YOUR_NEW_API_KEY"`
4. Reboot the app.

Do not commit `.streamlit/secrets.toml`.

## Local run
`pip install -r requirements.txt`
Set environment variable `TWELVE_DATA_API_KEY`, then run `streamlit run app.py`.

## Limitations
Pressure is inferred from OHLC candle shape, ATR-normalized price momentum, acceleration, and EMA trend context. It is not true order flow, order-book depth, aggressor-side volume, or institutional positioning. Initial weights (H1 15%, M15 25%, M5 45%, M1 15%) are research defaults, not optimized. Include broker spread, slippage, commissions, and chronological out-of-sample validation before considering live use.
