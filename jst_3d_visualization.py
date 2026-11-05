"""
JST APEX QUANT — 3D VISUALIZATION MODULE
File 4: jst_3d_visualization.py

Standalone visualization helpers for the Streamlit dashboard.
Research visualization only: this module does not generate or modify trade signals.

Required:
    numpy
    pandas
    plotly

Public functions:
    probability_surface(...)
    monte_carlo_3d(...)
    volatility_surface(...)
    return_distribution_surface(...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


NAVY = "#07111f"
PANEL = "#101d2e"
BLUE = "#2589ff"
WHITE = "#e8edf3"
MUTED = "#91a4ba"


def _clean_returns(returns, min_size=10):
    values = pd.to_numeric(pd.Series(returns), errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(values) < min_size:
        raise ValueError(f"Minimal {min_size} observasi return diperlukan; tersedia {len(values)}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Return mengandung nilai non-finite.")
    return values


def _layout(fig, title, height=620):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        paper_bgcolor=NAVY,
        plot_bgcolor=PANEL,
        font=dict(color=WHITE),
        margin=dict(l=10, r=10, t=55, b=10),
        scene=dict(
            bgcolor=NAVY,
            xaxis=dict(backgroundcolor=PANEL, gridcolor="#26384d"),
            yaxis=dict(backgroundcolor=PANEL, gridcolor="#26384d"),
            zaxis=dict(backgroundcolor=PANEL, gridcolor="#26384d"),
        ),
    )
    return fig


def probability_surface(
    returns,
    x_points=70,
    y_points=55,
    x_scale=None,
    volatility_scales=None,
):
    """
    Gaussian-kernel probability-density surface of empirical returns.

    X axis: return value.
    Y axis: bandwidth multiplier.
    Z axis: estimated density.

    This is a descriptive KDE surface, not a calibrated future probability.
    """
    r = _clean_returns(returns)
    std = float(np.std(r, ddof=1))
    if std <= 0:
        raise ValueError("Variasi return nol; probability surface tidak dapat dihitung.")

    q25, q75 = np.quantile(r, [0.25, 0.75])
    robust_scale = (q75 - q25) / 1.349
    scale = min(std, robust_scale) if robust_scale > 0 else std
    bandwidth = max(0.9 * scale * (len(r) ** (-1 / 5)), std * 1e-4)

    if x_scale is None:
        lo, hi = np.quantile(r, [0.005, 0.995])
        pad = max((hi - lo) * 0.15, std * 0.5)
        x_scale = np.linspace(lo - pad, hi + pad, int(x_points))
    else:
        x_scale = np.asarray(x_scale, dtype=float)

    if volatility_scales is None:
        volatility_scales = np.linspace(0.5, 2.0, int(y_points))
    else:
        volatility_scales = np.asarray(volatility_scales, dtype=float)

    z = np.empty((len(volatility_scales), len(x_scale)), dtype=float)
    norm = np.sqrt(2.0 * np.pi)
    for i, multiplier in enumerate(volatility_scales):
        h = bandwidth * float(multiplier)
        u = (x_scale[None, :] - r[:, None]) / h
        z[i, :] = np.exp(-0.5 * u * u).mean(axis=0) / (h * norm)

    fig = go.Figure(data=[go.Surface(
        x=x_scale,
        y=volatility_scales,
        z=z,
        colorscale="Viridis",
        colorbar=dict(title="Density"),
        hovertemplate="Return: %{x:.6f}<br>Bandwidth scale: %{y:.2f}<br>Density: %{z:.5f}<extra></extra>",
    )])
    fig.update_layout(scene=dict(
        xaxis_title="Observed return",
        yaxis_title="Bandwidth multiplier",
        zaxis_title="Estimated density",
    ))
    return _layout(fig, "3D Probability Density Surface — Empirical KDE")


def monte_carlo_3d(
    returns,
    initial_price,
    horizon=40,
    paths=250,
    seed=42,
    displayed_paths=80,
):
    """
    3D path visualization using an IID Gaussian log-return baseline.

    X axis: future bar.
    Y axis: simulation path identifier.
    Z axis: simulated price.

    This is a scenario visualization, not a price target or forecast guarantee.
    """
    r = _clean_returns(returns)
    if initial_price <= 0:
        raise ValueError("initial_price harus lebih besar dari nol.")
    if int(horizon) < 1 or int(paths) < 2:
        raise ValueError("horizon minimal 1 dan paths minimal 2.")

    log_r = np.log1p(r[r > -1])
    if len(log_r) < 10:
        raise ValueError("Minimal 10 log-return valid diperlukan.")
    mu = float(np.mean(log_r))
    sigma = float(np.std(log_r, ddof=1))

    rng = np.random.default_rng(int(seed))
    shocks = rng.normal(mu, sigma, size=(int(paths), int(horizon)))
    simulated = float(initial_price) * np.exp(np.cumsum(shocks, axis=1))
    simulated = np.column_stack([np.full(int(paths), initial_price), simulated])

    n_show = min(int(displayed_paths), int(paths))
    x = np.arange(int(horizon) + 1)
    y = np.arange(n_show)
    xx, yy = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(
        x=xx,
        y=yy,
        z=simulated[:n_show],
        colorscale="Blues",
        colorbar=dict(title="Price"),
        hovertemplate="Future bar: %{x}<br>Path: %{y}<br>Price: %{z:,.2f}<extra></extra>",
    )])
    fig.update_layout(scene=dict(
        xaxis_title="Future M5 bar",
        yaxis_title="Simulation path",
        zaxis_title="Simulated price",
    ))

    terminal = simulated[:, -1]
    summary = {
        "paths_simulated": int(paths),
        "paths_displayed": int(n_show),
        "horizon_bars": int(horizon),
        "mean_terminal_price": float(np.mean(terminal)),
        "median_terminal_price": float(np.median(terminal)),
        "q05_terminal_price": float(np.quantile(terminal, 0.05)),
        "q95_terminal_price": float(np.quantile(terminal, 0.95)),
        "probability_terminal_above_initial": float(np.mean(terminal > initial_price)),
        "assumption": "IID Gaussian log returns; ignores jumps, changing regimes, spread, fees and slippage.",
    }
    return _layout(fig, "3D Monte Carlo Scenario Surface"), summary


def volatility_surface(
    returns,
    short_windows=(5, 10, 20, 30, 40),
    long_windows=(20, 30, 50, 75, 100),
):
    """
    Historical rolling-volatility surface across short and long windows.

    X axis: observation index.
    Y axis: rolling window length.
    Z axis: annualization-free rolling standard deviation per bar.
    """
    r = _clean_returns(returns, min_size=max(long_windows) + 5)
    max_window = max(long_windows)
    start = max_window - 1
    x = np.arange(start, len(r))

    z_rows = []
    y_values = []
    for window in sorted(set(int(w) for w in tuple(short_windows) + tuple(long_windows))):
        if window < 2 or window > len(r):
            continue
        s = pd.Series(r).rolling(window).std(ddof=1).to_numpy()
        z_rows.append(s[start:])
        y_values.append(window)

    if not z_rows:
        raise ValueError("Tidak ada window volatilitas yang valid.")

    z = np.asarray(z_rows, dtype=float)
    xx, yy = np.meshgrid(x, np.asarray(y_values, dtype=float))
    fig = go.Figure(data=[go.Surface(
        x=xx,
        y=yy,
        z=z,
        colorscale="Turbo",
        colorbar=dict(title="Std / bar"),
        hovertemplate="Observation: %{x}<br>Window: %{y}<br>Volatility: %{z:.6f}<extra></extra>",
    )])
    fig.update_layout(scene=dict(
        xaxis_title="Observation index",
        yaxis_title="Rolling window (bars)",
        zaxis_title="Volatility (std / bar)",
    ))
    return _layout(fig, "3D Rolling Volatility Surface")


def return_distribution_surface(returns, bins=45, windows=8):
    """
    Rolling empirical histogram surface.

    X axis: return-bin midpoint.
    Y axis: rolling sample/window number.
    Z axis: relative frequency in each window.
    """
    r = _clean_returns(returns, min_size=max(30, int(bins)))
    bins = int(bins)
    windows = int(windows)
    if bins < 5 or windows < 2:
        raise ValueError("bins minimal 5 dan windows minimal 2.")

    edges = np.linspace(
        float(np.quantile(r, 0.005)),
        float(np.quantile(r, 0.995)),
        bins + 1,
    )
    centers = (edges[:-1] + edges[1:]) / 2
    chunks = np.array_split(r, windows)
    rows = []
    labels = []
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        counts, _ = np.histogram(chunk, bins=edges)
        total = counts.sum()
        rows.append(counts / total if total else np.zeros_like(counts, dtype=float))
        labels.append(i + 1)

    z = np.asarray(rows, dtype=float)
    xx, yy = np.meshgrid(centers, np.asarray(labels, dtype=float))
    fig = go.Figure(data=[go.Surface(
        x=xx,
        y=yy,
        z=z,
        colorscale="Plasma",
        colorbar=dict(title="Relative freq."),
        hovertemplate="Return bin: %{x:.6f}<br>Window: %{y}<br>Relative frequency: %{z:.4f}<extra></extra>",
    )])
    fig.update_layout(scene=dict(
        xaxis_title="Return bin midpoint",
        yaxis_title="Chronological sample window",
        zaxis_title="Relative frequency",
    ))
    return _layout(fig, "3D Rolling Return Distribution Surface")
