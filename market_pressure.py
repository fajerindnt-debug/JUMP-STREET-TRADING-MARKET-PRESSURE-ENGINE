import requests
import pandas as pd
import numpy as np

URL="https://api.twelvedata.com/time_series"
WEIGHTS={"H1":0.15,"M15":0.25,"M5":0.45,"M1":0.15}
MINUTES={"1h":60,"15min":15,"5min":5,"1min":1}

def fetch_ohlc(symbol,interval,api_key,outputsize=250):
    params={"symbol":symbol,"interval":interval,"outputsize":outputsize,
            "apikey":api_key,"timezone":"UTC","format":"JSON"}
    r=requests.get(URL,params=params,timeout=25)
    r.raise_for_status()
    payload=r.json()
    if "values" not in payload:
        raise RuntimeError(payload.get("message",str(payload)))
    df=pd.DataFrame(payload["values"])
    need=["datetime","open","high","low","close"]
    if any(c not in df for c in need): raise RuntimeError("Missing OHLC fields")
    df["datetime"]=pd.to_datetime(df["datetime"],errors="coerce",utc=True)
    for c in ["open","high","low","close"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=need).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    # Exclude last candle if its interval has not elapsed.
    if len(df)>2:
        end=df.datetime.iloc[-1]+pd.Timedelta(minutes=MINUTES[interval])
        if end>pd.Timestamp.now(tz="UTC"): df=df.iloc[:-1].copy()
    if len(df)<60: raise RuntimeError(f"Only {len(df)} completed candles available")
    return df

def add_indicators(df):
    x=df.copy()
    x["ema20"]=x.close.ewm(span=20,adjust=False).mean()
    x["ema50"]=x.close.ewm(span=50,adjust=False).mean()
    pc=x.close.shift(1)
    tr=pd.concat([x.high-x.low,(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1)
    x["atr"]=tr.rolling(14,min_periods=5).mean()
    x["range"]=(x.high-x.low).replace(0,np.nan)
    x["body"]=x.close-x.open
    x["body_fraction"]=(x.body.abs()/x["range"]).clip(0,1).fillna(0)
    x["clv"]=(2*(x.close-x.low)/x["range"]-1).clip(-1,1).fillna(0)
    return x

def _last(s,default=0.0):
    v=s.iloc[-1] if len(s) else default
    return float(v) if pd.notna(v) and np.isfinite(v) else float(default)

def timeframe_summary(df,lookback=14):
    x=add_indicators(df).dropna(subset=["atr"]).copy()
    if len(x)<max(30,lookback+5): raise RuntimeError("Not enough indicator history")
    w=x.tail(lookback)
    candle=float((0.60*np.sign(w.body)*w.body_fraction+0.40*w.clv).clip(-1,1).mean())
    atr=_last(x.atr); close=_last(x.close)
    old=float(x.close.iloc[-min(lookback+1,len(x))])
    mom=np.tanh(((close-old)/atr)/2) if atr>0 else 0.0
    ms=(x.close.diff(5)/x.atr.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna().tail(lookback)
    ms=np.tanh(ms/2)
    accel=float(ms.diff().dropna().mean()*2) if len(ms)>2 else 0.0
    accel=float(np.clip(accel,-1,1))
    last=x.iloc[-1]
    if last.close>last.ema20>last.ema50: trend,trend_score="BULLISH",1.0
    elif last.close<last.ema20<last.ema50: trend,trend_score="BEARISH",-1.0
    else: trend,trend_score="MIXED",0.0
    net=float(np.clip(.35*candle+.30*mom+.15*accel+.20*trend_score,-1,1))
    buy=(net+1)*50
    med=float(x.atr.dropna().tail(50).median())
    vol="UNKNOWN" if med<=0 else "HIGH" if atr>med*1.25 else "LOW" if atr<med*.75 else "NORMAL"
    mom_state="BULLISH" if mom>.15 else "BEARISH" if mom<-.15 else "NEUTRAL"
    return {"buy_pressure":buy,"sell_pressure":100-buy,"net_pressure":net*100,
            "momentum":float(mom),"acceleration":accel,"trend":trend,
            "momentum_state":mom_state,"volatility":atr/close if close else 0,
            "volatility_state":vol,"candle_pressure":candle,"trend_score":trend_score}

def build_pressure_report(frames,lookback=14):
    tfs={tf:timeframe_summary(frames[tf],lookback) for tf in WEIGHTS}
    net=sum(tfs[tf]["net_pressure"]*WEIGHTS[tf] for tf in WEIGHTS)
    buy=float(np.clip(50+net/2,0,100))
    m5=tfs["M5"]
    alignment=float(np.mean([tfs[t]["trend_score"] for t in ["H1","M15","M5"]]))
    return {"buy_pressure":buy,"sell_pressure":100-buy,"gap":abs(2*buy-100),
      "timeframes":tfs,
      "components":{"candle_pressure":m5["candle_pressure"],"momentum":m5["momentum"],
                    "acceleration":m5["acceleration"],"trend_context":m5["trend_score"],
                    "multi_timeframe_alignment":alignment},
      "diagnostics":{"momentum":m5["momentum_state"],
        "acceleration":"POSITIVE" if m5["acceleration"]>.05 else "NEGATIVE" if m5["acceleration"]<-.05 else "FLAT",
        "trend_alignment":"BULLISH" if alignment>.33 else "BEARISH" if alignment<-.33 else "MIXED",
        "candle_pressure":"BUYERS" if m5["candle_pressure"]>.1 else "SELLERS" if m5["candle_pressure"]<-.1 else "BALANCED",
        "volatility_state":m5["volatility_state"]}}

def classify_market(report,threshold=60,min_gap=15):
    b,s,g=report["buy_pressure"],report["sell_pressure"],report["gap"]
    m5,m15=report["timeframes"]["M5"],report["timeframes"]["M15"]
    if b>=threshold and g>=min_gap and m5["net_pressure"]>0:
        state,side,explanation="BULLISH DOMINANCE","BUYERS","Buyer-side OHLC pressure proxy is dominant."
    elif s>=threshold and g>=min_gap and m5["net_pressure"]<0:
        state,side,explanation="BEARISH DOMINANCE","SELLERS","Seller-side OHLC pressure proxy is dominant."
    else:
        state,side,explanation="BALANCED / TRANSITION","NO CLEAR SIDE","No sufficiently strong pressure advantage; avoid forcing a directional conclusion."
    notes=[f"M5 primary pressure: {m5['net_pressure']:+.1f}.",
           f"M15 context: {m15['trend']} ({m15['net_pressure']:+.1f}).",
           f"H1 context: {report['timeframes']['H1']['trend']}.",
           f"M1 momentum: {report['timeframes']['M1']['momentum_state']}."]
    return {"state":state,"dominant":side,"explanation":explanation,"notes":notes}
