"""JST Market Pressure Engine.

OHLC-derived pressure proxy for research/display only.
It is not exchange order flow, aggressor-side volume, or order-book data.
This module is separate from the user's V8.5 signal engine.
"""
from __future__ import annotations

import requests
import numpy as np
import pandas as pd


def fetch_ohlc(symbol: str, interval: str, api_key: str, outputsize: int = 250) -> pd.DataFrame:
    if not api_key:
        raise ValueError("Twelve Data API key kosong.")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": int(outputsize),
        "apikey": api_key,
        "format": "JSON",
    }
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        message = payload.get("message") or payload.get("status") or str(payload)
        raise RuntimeError(f"Twelve Data tidak mengembalikan OHLC: {message}")

    df = pd.DataFrame(payload["values"])
    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom OHLC tidak lengkap: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if len(df) < 55:
        raise ValueError(f"Data {interval} terlalu sedikit: {len(df)} bar.")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14, min_periods=14).mean()
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["clv"] = ((2 * out["close"] - out["high"] - out["low"]) / candle_range).clip(-1, 1).fillna(0)
    out["body_ratio"] = ((out["close"] - out["open"]).abs() / candle_range).replace([np.inf, -np.inf], np.nan).fillna(0)
    out["momentum"] = out["close"].pct_change(5)
    return out


def _timeframe_pressure(df: pd.DataFrame, lookback: int) -> dict:
    x = add_indicators(df).tail(max(lookback, 5)).copy()
    if x.empty:
        raise ValueError("Data timeframe kosong.")
    candle_range = (x["high"] - x["low"]).replace(0, np.nan)
    clv = ((2 * x["close"] - x["high"] - x["low"]) / candle_range).clip(-1, 1).fillna(0)
    body_direction = np.sign(x["close"] - x["open"])
    body_strength = x["body_ratio"].clip(0, 1)
    # Equal-bar OHLC proxy. Volume is intentionally not used as true aggressor flow.
    score = (0.65 * clv + 0.35 * body_direction * body_strength).clip(-1, 1)
    net = float(score.mean())
    buy = float((net + 1) * 50)
    sell = 100.0 - buy

    last = x.iloc[-1]
    if last["close"] > last["ema20"] > last["ema50"]:
        trend = "BULLISH"
    elif last["close"] < last["ema20"] < last["ema50"]:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    mom = float(last["momentum"]) if pd.notna(last["momentum"]) else 0.0
    momentum_state = "BULLISH" if mom > 0 else "BEARISH" if mom < 0 else "NEUTRAL"
    atr = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
    atr_median = float(x["atr"].dropna().median()) if x["atr"].notna().any() else 0.0
    if atr_median <= 0:
        volatility_state = "UNKNOWN"
    elif atr > atr_median * 1.25:
        volatility_state = "HIGH"
    elif atr < atr_median * 0.75:
        volatility_state = "LOW"
    else:
        volatility_state = "NORMAL"

    return {
        "buy_pressure": buy,
        "sell_pressure": sell,
        "net_pressure": net * 100,
        "trend": trend,
        "momentum_state": momentum_state,
        "volatility_state": volatility_state,
        "bars": len(x),
    }


def build_pressure_report(frames: dict, lookback: int = 14) -> dict:
    required = ["H1", "M15", "M5", "M1"]
    missing = [tf for tf in required if tf not in frames or frames[tf] is None or frames[tf].empty]
    if missing:
        raise ValueError(f"Timeframe tidak tersedia: {', '.join(missing)}")
    tf_results = {tf: _timeframe_pressure(frames[tf], lookback) for tf in required}
    weights = {"H1": 0.15, "M15": 0.25, "M5": 0.45, "M1": 0.15}
    net = sum(tf_results[tf]["net_pressure"] * weights[tf] for tf in required)
    buy = float(np.clip(50 + net / 2, 0, 100))
    sell = 100.0 - buy
    gap = abs(buy - sell)
    return {
        "buy_pressure": buy,
        "sell_pressure": sell,
        "gap": gap,
        "timeframes": tf_results,
        "diagnostics": {
            "weighted_net_pressure": round(net, 2),
            "weights": weights,
            "primary_timeframe": "M5",
            "method": "OHLC candle-location/body proxy",
            "data_limit": "Not actual order flow or order-book imbalance",
        },
    }


def classify_market(report: dict, threshold: float = 60, gap_min: float = 15) -> dict:
    buy = float(report["buy_pressure"])
    sell = float(report["sell_pressure"])
    gap = float(report["gap"])
    dominant = "BUYERS" if buy > sell else "SELLERS" if sell > buy else "BALANCED"
    if max(buy, sell) >= float(threshold) and gap >= float(gap_min):
        state = "BUY PRESSURE DOMINANT" if buy > sell else "SELL PRESSURE DOMINANT"
        explanation = f"{dominant.title()} lebih dominan berdasarkan proxy OHLC dengan gap {gap:.1f} poin."
        notes = ["Tekanan dihitung dari posisi close dan body candle.", "Konfirmasi struktur dan risiko tetap diperlukan."]
    else:
        state = "BALANCED / MIXED"
        explanation = f"Belum ada dominasi yang melewati threshold; gap saat ini {gap:.1f} poin."
        notes = ["Tekanan relatif berimbang atau belum cukup kuat.", "Hindari menganggap proxy ini sebagai sinyal transaksi mandiri."]
    return {"state": state, "dominant": dominant, "explanation": explanation, "notes": notes}
