import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from market_pressure import (
    fetch_ohlc, build_pressure_report, classify_market, add_indicators
)

# Additive research module. Does not modify the existing pressure/signal logic.
try:
    import jst_advanced_quant as aq
    ADVANCED_QUANT_AVAILABLE = True
    ADVANCED_QUANT_ERROR = ""
except Exception as exc:
    ADVANCED_QUANT_AVAILABLE = False
    ADVANCED_QUANT_ERROR = str(exc)


st.set_page_config(
    page_title="JUMP STREET TRADING",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp{background:#07111f;color:#edf4fc}
[data-testid="stSidebar"]{background:#0a1626}
.block-container{max-width:1600px;padding-top:1.2rem}
div[data-testid="stMetric"]{background:#101d2e;border:1px solid #26384d;padding:14px;border-radius:12px}
.panel{background:#101d2e;border:1px solid #26384d;border-radius:14px;padding:16px}
.muted{color:#91a4ba}
</style>
""", unsafe_allow_html=True)


def api_key():
    try:
        k = st.secrets.get("TWELVE_DATA_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return os.getenv("TWELVE_DATA_API_KEY", "")


def safe_float(value, default=np.nan):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def fmt_pct(value, digits=3):
    value = safe_float(value)
    return "N/A" if not np.isfinite(value) else f"{value * 100:.{digits}f}%"


# ---------------------------------------------------------------------
# Sidebar — existing Market Pressure controls retained
# ---------------------------------------------------------------------
st.sidebar.title("JUMP STREET TRADING")
st.sidebar.caption("MARKET PRESSURE ENGINE + QUANT RESEARCH")

symbol = st.sidebar.text_input("Symbol", "XAU/USD").upper().strip()
lookback = st.sidebar.slider("Pressure lookback", 5, 40, 14)
threshold = st.sidebar.slider("Dominance threshold", 50, 80, 60)
gap_min = st.sidebar.slider("Minimum pressure gap", 5, 40, 15)

if st.sidebar.button("Refresh data", use_container_width=True):
    st.cache_data.clear()

key = api_key()
if not key:
    st.error("API key belum diatur. Tambahkan TWELVE_DATA_API_KEY ke Streamlit Secrets.")
    st.code('TWELVE_DATA_API_KEY = "API_KEY_BARU_ANDA"', language="toml")
    st.stop()


@st.cache_data(ttl=45, show_spinner=False)
def load(symbol, key):
    out = {}
    for tf, interval, n in [
        ("H1", "1h", 250),
        ("M15", "15min", 250),
        ("M5", "5min", 300),
        ("M1", "1min", 300),
    ]:
        out[tf] = fetch_ohlc(symbol, interval, key, n)
    return out


st.markdown(
    '<p class="muted">QUANT RESEARCH TERMINAL / XAUUSD</p>',
    unsafe_allow_html=True
)
st.title("JUMP STREET TRADING")
st.subheader("Market Pressure Engine")
st.caption("H1 context · M15 structure · M5 primary pressure · M1 micro timing")

try:
    with st.spinner("Loading multi-timeframe market data..."):
        frames = load(symbol, key)
    report = build_pressure_report(frames, lookback)
    decision = classify_market(report, threshold, gap_min)
except Exception as e:
    st.error(f"Data/engine error: {e}")
    st.stop()

if "M1" not in frames or frames["M1"].empty:
    st.error("Data M1 kosong. Periksa symbol atau koneksi data.")
    st.stop()

price = float(frames["M1"].iloc[-1]["close"])
a, b, c, d = st.columns(4)
a.metric("LAST PRICE", f"{price:,.2f}")
b.metric("MARKET STATE", decision["state"])
c.metric("DOMINANT SIDE", decision["dominant"])
d.metric("PRESSURE GAP", f"{report['gap']:.1f} pts")

left, right = st.columns([1.35, 1], gap="large")

with left:
    st.markdown("### Buyer / Seller pressure")
    st.write(f"BUYING PRESSURE — **{report['buy_pressure']:.1f}%**")
    st.progress(max(0, min(100, int(report["buy_pressure"]))))
    st.write(f"SELLING PRESSURE — **{report['sell_pressure']:.1f}%**")
    st.progress(max(0, min(100, int(report["sell_pressure"]))))

    st.markdown("### Multi-timeframe matrix")
    matrix = []
    for tf in ["H1", "M15", "M5", "M1"]:
        x = report["timeframes"][tf]
        matrix.append({
            "TF": tf,
            "Buyer %": round(x["buy_pressure"], 1),
            "Seller %": round(x["sell_pressure"], 1),
            "Net pressure": round(x["net_pressure"], 1),
            "Trend": x["trend"],
            "Momentum": x["momentum_state"],
            "Volatility": x["volatility_state"],
        })
    matrix_df = pd.DataFrame(matrix)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

with right:
    st.markdown("### Engine decision")
    st.markdown(
        f'<div class="panel"><h2>{decision["state"]}</h2>'
        f'<p class="muted">{decision["explanation"]}</p></div>',
        unsafe_allow_html=True
    )
    st.markdown("### Pressure diagnostics")
    for k, v in report["diagnostics"].items():
        st.write(f"**{k.replace('_', ' ').title()}:** {v}")
    st.info(
        "OHLC-derived pressure proxy; this is not actual exchange order flow "
        "or order-book data."
    )


# ---------------------------------------------------------------------
# Return series for research diagnostics. M5 is the primary research TF.
# These diagnostics do not feed back into Market Pressure decisions.
# ---------------------------------------------------------------------
m5 = frames["M5"].copy()
m5["close"] = pd.to_numeric(m5["close"], errors="coerce")
m5_returns = m5["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
m5_returns = m5_returns.astype(float)

tabs = st.tabs([
    "Overview",
    "Charting",
    "Market Pressure",
    "Quant Math Lab",
    "Probability & Risk",
    "Quant Research",
    "Methodology & Risk",
])

with tabs[0]:
    cols = st.columns(4)
    for col, tf in zip(cols, ["H1", "M15", "M5", "M1"]):
        x = report["timeframes"][tf]
        col.markdown(
            f'<div class="panel"><span class="muted">{tf}</span>'
            f'<h3>{x["trend"]}</h3><b>Net: {x["net_pressure"]:+.1f}</b>'
            f'<p class="muted">Momentum: {x["momentum_state"]}</p></div>',
            unsafe_allow_html=True
        )
    st.markdown("### Interpretation")
    for note in decision["notes"]:
        st.write("• " + note)

    st.markdown("### Research module status")
    if ADVANCED_QUANT_AVAILABLE:
        st.success("Advanced Quant module loaded. Research-only mode.")
    else:
        st.warning(
            "Advanced Quant module belum dapat dimuat. "
            "Pastikan jst_advanced_quant.py ada di root repository dan dependensinya terpasang."
        )
        st.code(ADVANCED_QUANT_ERROR or "Unknown import error")


with tabs[1]:
    tf = st.selectbox("Timeframe", ["M5", "M15", "H1", "M1"])
    df = add_indicators(frames[tf]).tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.datetime,
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        name="Price",
        increasing_line_color="#2589ff",
        decreasing_line_color="#e8edf3",
    ))
    fig.add_trace(go.Scatter(
        x=df.datetime, y=df.ema20, name="EMA 20",
        line=dict(color="#f5b942", width=1.2)
    ))
    fig.add_trace(go.Scatter(
        x=df.datetime, y=df.ema50, name="EMA 50",
        line=dict(color="#b18cff", width=1.2)
    ))
    fig.update_layout(
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#101d2e",
        plot_bgcolor="#0b1728",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


with tabs[2]:
    st.markdown("### Pressure component scores")
    comp = pd.DataFrame([
        {"Component": k.replace("_", " ").title(), "Score": v}
        for k, v in report["components"].items()
    ])
    st.bar_chart(comp.set_index("Component"))
    st.markdown("### Timeframe details")
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download pressure CSV",
        matrix_df.to_csv(index=False),
        "jst_pressure_report.csv",
        "text/csv",
    )


with tabs[3]:
    st.markdown("### Advanced statistical diagnostics — M5 returns")
    st.caption(
        "Semua perhitungan di tab ini bersifat observasional. "
        "Output tidak mengubah keputusan Market Pressure."
    )

    if not ADVANCED_QUANT_AVAILABLE:
        st.error("Modul jst_advanced_quant.py gagal diimpor.")
        st.code(ADVANCED_QUANT_ERROR)
    elif len(m5_returns) < 30:
        st.warning(f"Data return M5 belum cukup. Tersedia {len(m5_returns)} observasi; minimal 30.")
    else:
        try:
            dist = aq.distribution_report(m5_returns)
            vol = aq.volatility_report(m5_returns)
            robust = aq.robust_statistics(m5_returns)
            ci = aq.mean_confidence_interval(m5_returns)
            htest = aq.hypothesis_test_mean(m5_returns)
            ts = aq.time_series_diagnostics(m5_returns, max_lags=20)
            brk = aq.structural_break_cusum(m5_returns)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("OBSERVATIONS", f"{dist['n']}")
            m2.metric("MEAN RETURN / BAR", fmt_pct(dist["mean"], 5))
            m3.metric("SKEWNESS", f"{dist['skewness']:.3f}")
            m4.metric("EXCESS KURTOSIS", f"{dist['excess_kurtosis']:.3f}")

            st.markdown("#### Distribution")
            q = dist["quantiles"]
            qdf = pd.DataFrame({
                "Quantile": list(q.keys()),
                "Return": list(q.values()),
            })
            st.dataframe(qdf, use_container_width=True, hide_index=True)

            hist = px.histogram(
                x=m5_returns,
                nbins=50,
                title="M5 empirical return distribution",
                labels={"x": "M5 simple return", "y": "Frequency"},
            )
            hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="#101d2e",
                plot_bgcolor="#0b1728",
            )
            st.plotly_chart(hist, use_container_width=True)

            st.markdown("#### Volatility")
            v1, v2, v3 = st.columns(3)
            v1.metric("REALIZED VOL / BAR", fmt_pct(vol["realized_volatility_per_bar"], 5))
            v2.metric("EWMA VOL / BAR", fmt_pct(vol["ewma_volatility_per_bar"], 5))
            v3.metric("ROLLING VOL / BAR", fmt_pct(vol["rolling_volatility_latest"], 5))

            st.markdown("#### Robust statistics and inference")
            st.json({
                "median": robust["median"],
                "MAD": robust["mad"],
                "robust_sigma": robust["robust_sigma"],
                "trimmed_mean_10pct": robust["trimmed_mean_10pct"],
                "winsorized_std_5pct": robust["winsorized_std_5pct"],
                "mean_confidence_interval_95pct": ci,
                "one_sample_t_test_vs_zero": htest,
            })

            st.markdown("#### Stationarity and structural-break screening")
            st.json({
                "ADF_statistic": ts["adf_statistic"],
                "ADF_pvalue": ts["adf_pvalue"],
                "KPSS": ts["kpss"],
                "CUSUM_screen": brk,
                "interpretation": ts["interpretation"],
            })
        except Exception as exc:
            st.error(f"Quant Math Lab error: {exc}")


with tabs[4]:
    st.markdown("### Probability & Risk Research")
    st.caption(
        "Win probability Bayesian membutuhkan hasil trade yang sudah ditutup. "
        "Jangan masukkan perkiraan atau sinyal yang belum memiliki outcome."
    )

    if not ADVANCED_QUANT_AVAILABLE:
        st.error("Modul Advanced Quant belum tersedia.")
        st.code(ADVANCED_QUANT_ERROR)
    else:
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            wins = st.number_input("Closed wins", min_value=0, value=0, step=1)
        with pcol2:
            losses = st.number_input("Closed losses", min_value=0, value=0, step=1)

        if wins + losses > 0:
            bayes = aq.bayesian_win_probability(int(wins), int(losses))
            x1, x2, x3 = st.columns(3)
            x1.metric("BAYESIAN POSTERIOR MEAN", f"{bayes['posterior_mean'] * 100:.2f}%")
            x2.metric("95% CREDIBLE LOWER", f"{bayes['credible_lower'] * 100:.2f}%")
            x3.metric("95% CREDIBLE UPPER", f"{bayes['credible_upper'] * 100:.2f}%")
            st.caption(
                f"Observations: {bayes['observations']}. "
                "Beta(1,1) prior; this is a posterior estimate, not a forecast guarantee."
            )
        else:
            st.info("Masukkan jumlah win dan loss yang sudah terealisasi untuk menghitung posterior.")

        if len(m5_returns) >= 20:
            try:
                risk = aq.risk_report(m5_returns, confidence=0.95)
                st.markdown("#### Historical risk metrics — M5")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("HISTORICAL VaR 95%", fmt_pct(risk["historical_var_loss_per_bar"], 5))
                r2.metric("EXPECTED SHORTFALL 95%", fmt_pct(risk["historical_expected_shortfall_per_bar"], 5))
                r3.metric("MAX DRAWDOWN", fmt_pct(risk["max_drawdown"], 3))
                r4.metric("SHARPE / BAR", f"{risk['sharpe_per_bar_unannualized']:.4f}")
                st.caption(
                    "VaR/ES dihitung per bar dari distribusi historis. Sharpe belum diannualisasi; "
                    "angka ini bukan estimasi kerugian maksimum."
                )
            except Exception as exc:
                st.warning(f"Risk calculation: {exc}")
        else:
            st.warning("Minimal 20 return M5 diperlukan untuk risk report.")


with tabs[5]:
    st.markdown("### Quant Research")
    st.caption("Eksperimen skenario menggunakan return historis M5. Bukan prediksi harga pasti.")

    if not ADVANCED_QUANT_AVAILABLE:
        st.error("Modul Advanced Quant belum tersedia.")
        st.code(ADVANCED_QUANT_ERROR)
    elif len(m5_returns) < 20:
        st.warning("Minimal 20 return M5 diperlukan untuk simulasi.")
    else:
        mc_col1, mc_col2, mc_col3 = st.columns(3)
        with mc_col1:
            horizon = st.number_input("Forecast horizon (M5 bars)", 1, 240, 20)
        with mc_col2:
            paths = st.number_input("Monte Carlo paths", 100, 10000, 1000, step=100)
        with mc_col3:
            seed = st.number_input("Random seed", 0, 100000, 42)

        if st.button("Run Monte Carlo", key="run_monte_carlo"):
            try:
                mc = aq.monte_carlo_gbm(
                    m5_returns,
                    horizon=int(horizon),
                    paths=int(paths),
                    initial_price=price,
                    seed=int(seed),
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MEAN TERMINAL", f"{mc['mean_terminal']:,.2f}")
                c2.metric("MEDIAN TERMINAL", f"{mc['median_terminal']:,.2f}")
                c3.metric("5TH PERCENTILE", f"{mc['q05_terminal']:,.2f}")
                c4.metric("95TH PERCENTILE", f"{mc['q95_terminal']:,.2f}")

                st.metric(
                    "SIMULATED P(TERMINAL > INITIAL)",
                    f"{mc['probability_above_initial'] * 100:.2f}%"
                )
                paths_df = pd.DataFrame(mc["sample_paths"]).T
                paths_df.index.name = "M5 bar"
                path_fig = go.Figure()
                for col in paths_df.columns[:30]:
                    path_fig.add_trace(go.Scatter(
                        x=paths_df.index,
                        y=paths_df[col],
                        mode="lines",
                        line=dict(width=1),
                        opacity=0.45,
                        showlegend=False,
                    ))
                path_fig.update_layout(
                    title="Sample Monte Carlo paths (up to 30 paths shown)",
                    template="plotly_dark",
                    height=450,
                    paper_bgcolor="#101d2e",
                    plot_bgcolor="#0b1728",
                    xaxis_title="Future M5 bars",
                    yaxis_title="Simulated price",
                )
                st.plotly_chart(path_fig, use_container_width=True)
                st.info(mc["assumptions"])
            except Exception as exc:
                st.error(f"Monte Carlo error: {exc}")

        st.markdown("#### Optional GARCH(1,1)-Student-t")
        if st.button("Estimate GARCH", key="run_garch"):
            try:
                garch = aq.garch_volatility_report(m5_returns)
                st.json(garch)
            except Exception as exc:
                st.error(f"GARCH error: {exc}")


with tabs[6]:
    st.markdown("### Core calculations")
    st.latex(r"CLV_t=2\\frac{C_t-L_t}{H_t-L_t}-1")
    st.latex(r"M_t=\\frac{C_t-C_{t-n}}{ATR_t}")
    st.latex(r"A_t=M_t-M_{t-1}")
    st.write(
        "Initial weights: H1 15%, M15 25%, M5 45%, M1 15%. "
        "These are research defaults, not optimized parameters."
    )
    st.warning(
        "Not an order-flow feed. OHLC cannot identify aggressor-side volume, hidden liquidity, "
        "institutional orders, spread or slippage. Validate on completed candles and "
        "out-of-sample data before any live use."
    )
    st.markdown("### Advanced Quant module boundary")
    st.write(
        "The Advanced Quant module reads M5 returns for diagnostics and scenario research. "
        "Its outputs are not connected to the existing Market Pressure classification, "
        "thresholds, weights, or decision rules."
    )
    st.write(
        "The dashboard remains signal/research only. No broker order execution is implemented."
    )

st.divider()
st.caption(
    "JUMP STREET TRADING · MARKET PRESSURE ENGINE + ADVANCED QUANT RESEARCH "
    "· SIGNAL / RESEARCH ONLY · NO AUTO-ORDER"
)
