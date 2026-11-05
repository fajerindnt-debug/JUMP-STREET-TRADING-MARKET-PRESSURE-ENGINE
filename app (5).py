import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from market_pressure import fetch_ohlc, build_pressure_report, classify_market, add_indicators

st.set_page_config(page_title="JUMP STREET TRADING", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp{background:#07111f;color:#edf4fc}
[data-testid="stSidebar"]{background:#0a1626}
.block-container{max-width:1600px;padding-top:1.2rem}
div[data-testid="stMetric"]{background:#101d2e;border:1px solid #26384d;padding:14px;border-radius:12px}
.panel{background:#101d2e;border:1px solid #26384d;border-radius:14px;padding:16px}
.muted{color:#91a4ba}
</style>""", unsafe_allow_html=True)

def api_key():
    try:
        k=st.secrets.get("TWELVE_DATA_API_KEY","")
        if k: return k
    except Exception: pass
    return os.getenv("TWELVE_DATA_API_KEY","")

st.sidebar.title("JUMP STREET TRADING")
st.sidebar.caption("MARKET PRESSURE ENGINE")
symbol=st.sidebar.text_input("Symbol","XAU/USD").upper().strip()
lookback=st.sidebar.slider("Pressure lookback",5,40,14)
threshold=st.sidebar.slider("Dominance threshold",50,80,60)
gap_min=st.sidebar.slider("Minimum pressure gap",5,40,15)
if st.sidebar.button("Refresh data",use_container_width=True):
    st.cache_data.clear()

key=api_key()
if not key:
    st.error("API key belum diatur. Tambahkan TWELVE_DATA_API_KEY ke Streamlit Secrets.")
    st.code('TWELVE_DATA_API_KEY = "API_KEY_BARU_ANDA"',language="toml")
    st.stop()

@st.cache_data(ttl=45,show_spinner=False)
def load(symbol,key):
    out={}
    for tf,interval,n in [("H1","1h",250),("M15","15min",250),("M5","5min",300),("M1","1min",300)]:
        out[tf]=fetch_ohlc(symbol,interval,key,n)
    return out

st.markdown('<p class="muted">QUANT RESEARCH TERMINAL / XAUUSD</p>',unsafe_allow_html=True)
st.title("JUMP STREET TRADING")
st.subheader("Market Pressure Engine")
st.caption("H1 context · M15 structure · M5 primary pressure · M1 micro timing")
try:
    with st.spinner("Loading multi-timeframe market data..."):
        frames=load(symbol,key)
    report=build_pressure_report(frames,lookback)
    decision=classify_market(report,threshold,gap_min)
except Exception as e:
    st.error(f"Data/engine error: {e}")
    st.stop()

price=float(frames["M1"].iloc[-1]["close"])
a,b,c,d=st.columns(4)
a.metric("LAST PRICE",f"{price:,.2f}")
b.metric("MARKET STATE",decision["state"])
c.metric("DOMINANT SIDE",decision["dominant"])
d.metric("PRESSURE GAP",f"{report['gap']:.1f} pts")

left,right=st.columns([1.35,1],gap="large")
with left:
    st.markdown("### Buyer / Seller pressure")
    st.write(f"BUYING PRESSURE — **{report['buy_pressure']:.1f}%**")
    st.progress(int(report["buy_pressure"]))
    st.write(f"SELLING PRESSURE — **{report['sell_pressure']:.1f}%**")
    st.progress(int(report["sell_pressure"]))
    st.markdown("### Multi-timeframe matrix")
    matrix=[]
    for tf in ["H1","M15","M5","M1"]:
        x=report["timeframes"][tf]
        matrix.append({"TF":tf,"Buyer %":round(x["buy_pressure"],1),"Seller %":round(x["sell_pressure"],1),
                       "Net pressure":round(x["net_pressure"],1),"Trend":x["trend"],
                       "Momentum":x["momentum_state"],"Volatility":x["volatility_state"]})
    st.dataframe(pd.DataFrame(matrix),use_container_width=True,hide_index=True)
with right:
    st.markdown("### Engine decision")
    st.markdown(f'<div class="panel"><h2>{decision["state"]}</h2><p class="muted">{decision["explanation"]}</p></div>',unsafe_allow_html=True)
    st.markdown("### Pressure diagnostics")
    for k,v in report["diagnostics"].items():
        st.write(f"**{k.replace('_',' ').title()}:** {v}")
    st.info("OHLC-derived pressure proxy; this is not actual exchange order flow or order-book data.")

overview,chart,research,method=st.tabs(["Overview","Charting","Quant Research","Methodology & Risk"])
with overview:
    cols=st.columns(4)
    for col,tf in zip(cols,["H1","M15","M5","M1"]):
        x=report["timeframes"][tf]
        col.markdown(f'<div class="panel"><span class="muted">{tf}</span><h3>{x["trend"]}</h3><b>Net: {x["net_pressure"]:+.1f}</b><p class="muted">Momentum: {x["momentum_state"]}</p></div>',unsafe_allow_html=True)
    st.markdown("### Interpretation")
    for note in decision["notes"]: st.write("• "+note)
with chart:
    tf=st.selectbox("Timeframe",["M5","M15","H1","M1"])
    df=add_indicators(frames[tf]).tail(180)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=df.datetime,open=df.open,high=df.high,low=df.low,close=df.close,
        name="Price",increasing_line_color="#2589ff",decreasing_line_color="#e8edf3"))
    fig.add_trace(go.Scatter(x=df.datetime,y=df.ema20,name="EMA 20",line=dict(color="#f5b942",width=1.2)))
    fig.add_trace(go.Scatter(x=df.datetime,y=df.ema50,name="EMA 50",line=dict(color="#b18cff",width=1.2)))
    fig.update_layout(template="plotly_dark",height=650,xaxis_rangeslider_visible=False,
        paper_bgcolor="#101d2e",plot_bgcolor="#0b1728",margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig,use_container_width=True)
with research:
    st.markdown("### Pressure component scores")
    comp=pd.DataFrame([{"Component":k.replace("_"," ").title(),"Score":v} for k,v in report["components"].items()])
    st.bar_chart(comp.set_index("Component"))
    st.markdown("### Timeframe details")
    st.dataframe(pd.DataFrame(matrix),use_container_width=True,hide_index=True)
    st.download_button("Download CSV",pd.DataFrame(matrix).to_csv(index=False),"jst_pressure_report.csv","text/csv")
with method:
    st.markdown("### Core calculations")
    st.latex(r"CLV_t=2\\frac{C_t-L_t}{H_t-L_t}-1")
    st.latex(r"M_t=\\frac{C_t-C_{t-n}}{ATR_t}")
    st.latex(r"A_t=M_t-M_{t-1}")
    st.write("Initial weights: H1 15%, M15 25%, M5 45%, M1 15%. These are research defaults, not optimized parameters.")
    st.warning("Not an order-flow feed. OHLC cannot identify aggressor-side volume, hidden liquidity, institutional orders, spread or slippage. Validate on completed candles and out-of-sample data before any live use.")
st.divider()
st.caption("JUMP STREET TRADING · MARKET PRESSURE ENGINE · SIGNAL ONLY · NO AUTO-ORDER")
