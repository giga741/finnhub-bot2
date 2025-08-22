# main.py — Pre-rally bot (Finnhub + TwelveData)
# Env: FINNHUB_API_KEY, TWELVEDATA_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
# Optional: TZ=Europe/Rome, ACTIVE_HOURS_START=08:45, ACTIVE_HOURS_END=23:15,
# SCORE_MIN=3, ENABLE_SHORT_ONLY=true, SAFE_MODE=true, DRY_RUN=false, POLL_SECONDS=60

import os, asyncio
from datetime import datetime, time, timedelta, timezone
from typing import List, Tuple, Dict, Optional
import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ---------- Timezone ----------
TZ = os.getenv("TZ", "Europe/Rome")
try:
    import zoneinfo
    TZINFO = zoneinfo.ZoneInfo(TZ)
except Exception:
    TZINFO = timezone(timedelta(hours=2))  # fallback

ACTIVE_HOURS_START = os.getenv("ACTIVE_HOURS_START", "08:45")
ACTIVE_HOURS_END   = os.getenv("ACTIVE_HOURS_END", "23:15")
POLL_SECONDS       = int(os.getenv("POLL_SECONDS", "60"))

SAFE_MODE         = os.getenv("SAFE_MODE", "true").lower() == "true"
DRY_RUN           = os.getenv("DRY_RUN", "false").lower() == "true"
SCORE_MIN         = int(os.getenv("SCORE_MIN", "3"))
ENABLE_SHORT_ONLY = os.getenv("ENABLE_SHORT_ONLY", "true").lower() == "true"

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
TWELVE_KEY  = os.getenv("TWELVEDATA_API_KEY", "")
TG_TOKEN    = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT     = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------- Assets ----------
ASSETS = [
    "PLTR","GOOGL","TSLA","AAPL","MU","AMD",
    "FCT.MI","XOM","VLO","GM",
    "MC.PA","KO","DIS",
    "IFX.DE","REY.MI",
    "EURUSD","USDJPY","GBPUSD",
    "ETHUSD","BTCUSD",
]
SYMBOL_MAP = {
    "PLTR":{"finnhub":"PLTR","twelvedata":"PLTR"},
    "GOOGL":{"finnhub":"GOOGL","twelvedata":"GOOGL"},
    "TSLA":{"finnhub":"TSLA","twelvedata":"TSLA"},
    "AAPL":{"finnhub":"AAPL","twelvedata":"AAPL"},
    "MU":{"finnhub":"MU","twelvedata":"MU"},
    "AMD":{"finnhub":"AMD","twelvedata":"AMD"},
    "XOM":{"finnhub":"XOM","twelvedata":"XOM"},
    "VLO":{"finnhub":"VLO","twelvedata":"VLO"},
    "GM":{"finnhub":"GM","twelvedata":"GM"},
    "MC.PA":{"finnhub":"MC.PA","twelvedata":"MC:PA"},
    "KO":{"finnhub":"KO","twelvedata":"KO"},
    "DIS":{"finnhub":"DIS","twelvedata":"DIS"},
    "FCT.MI":{"finnhub":"FCT.MI","twelvedata":"FCT:MI"},
    "IFX.DE":{"finnhub":"IFX.DE","twelvedata":"IFX:XETRA"},
    "REY.MI":{"finnhub":"REY.MI","twelvedata":"REY:MI"},
    "EURUSD":{"finnhub":"OANDA:EUR_USD","twelvedata":"EUR/USD"},
    "USDJPY":{"finnhub":"OANDA:USD_JPY","twelvedata":"USD/JPY"},
    "GBPUSD":{"finnhub":"OANDA:GBP_USD","twelvedata":"GBP/USD"},
    "ETHUSD":{"finnhub":"BINANCE:ETHUSDT","twelvedata":"ETH/USD"},
    "BTCUSD":{"finnhub":"BINANCE:BTCUSDT","twelvedata":"BTC/USD"},
}
INTRADAY_M1 = {"EURUSD","USDJPY","GBPUSD","ETHUSD","BTCUSD"}
TF_LIST = ["15min","1h","1day"]  # M15, H1, D1

# ---------- State ----------
LAST_SIGNALS: List[dict] = []
LAST_ERROR: Optional[str] = None
STARTED_AT = datetime.now(TZINFO).isoformat()

# ---------- Indicators ----------
def ema(series: List[float], period: int) -> List[float]:
    if not series: return []
    k = 2.0 / (period + 1.0)
    out = []; e = series[0]
    for p in series:
        e = p*k + e*(1.0-k); out.append(e)
    return out

def sma(series: List[float], period: int) -> List[float]:
    out=[]; s=0.0
    for i,v in enumerate(series):
        s += v
        if i >= period: s -= series[i-period]
        out.append(s/period if i+1>=period else s/max(1,i+1))
    return out

def rsi(prices: List[float], period: int = 14) -> List[float]:
    if len(prices) < period+1: return [50.0]*len(prices)
    gains=[0.0]; losses=[0.0]
    for i in range(1,len(prices)):
        ch=prices[i]-prices[i-1]
        gains.append(max(ch,0.0)); losses.append(abs(min(ch,0.0)))
    ag=sum(gains[1:period+1])/period; al=sum(losses[1:period+1])/period
    rsis=[50.0]*len(prices)
    for i in range(period+1,len(prices)):
        ag=(ag*(period-1)+gains[i])/period
        al=(al*(period-1)+losses[i])/period
        rsis[i] = 100.0 if al==0 else 100.0 - (100.0/(1.0+ag/al))
    return rsis

def macd(prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
    e12=ema(prices,12); e26=ema(prices,26)
    line=[a-b for a,b in zip(e12,e26)]
    sig=ema(line,9); hist=[a-b for a,b in zip(line,sig)]
    return line,sig,hist

def true_range(h,l,c):
    tr=[h[0]-l[0]]
    for i in range(1,len(c)):
        tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    return tr

def atr(h,l,c,period=14):
    tr=true_range(h,l,c)
    if not tr: return []
    if len(tr)<period:
        avg=sum(tr)/max(1,len(tr)); return [avg]*len(c)
    sm=sum(tr[:period])/period; out=[sm]
    for i in range(period,len(tr)):
        sm=(sm*(period-1)+tr[i])/period; out.append(sm)
    while len(out)<len(c): out.insert(0,out[0])
    return out

def adx(h,l,c,period=14):
    if len(c)<period+2: return [0.0]*len(c)
    plus_dm=[0.0]; minus_dm=[0.0]; tr=[0.0]
    for i in range(1,len(c)):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        plus_dm.append(up if up>dn and up>0 else 0.0)
        minus_dm.append(dn if dn>up and dn>0 else 0.0)
        tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    def smooth(arr):
        sm=[0.0]*len(arr); sm[period]=sum(arr[1:period+1])
        for i in range(period+1,len(arr)): sm[i]=sm[i-1]-(sm[i-1]/period)+arr[i]
        return sm
    tr_s=smooth(tr); p_s=smooth(plus_dm); m_s=smooth(minus_dm)
    dx=[0.0]*len(c)
    for i in range(period,len(c)):
        if tr_s[i]==0: continue
        pdi=100.0*(p_s[i]/tr_s[i]); mdi=100.0*(m_s[i]/tr_s[i]); den=pdi+mdi
        if den==0: continue
        dx[i]=100.0*abs(pdi-mdi)/den
    ad=[0.0]*len(c); val=sum(dx[period:period*2])/period if len(c)>=period*2 else dx[-1]
    for i in range(period*2,len(c)): val=(val*(period-1)+dx[i])/period; ad[i]=val
    return ad

def compression_index(c,look=20):
    if len(c)<look: return 1.0
    w=c[-look:]; rng=max(w)-min(w)
    if rng<=0: return 0.0
    wig=sum(abs(w[i]-w[i-1]) for i in range(1,len(w)))/(look-1)
    return wig/rng  # lower = tighter

def support_resistance(c,look=30):
    if len(c)<look: return None,None
    w=c[-look:]; return min(w),max(w)

def detect_pattern_basic(c):
    if not c or len(c)<20: return None
    last=c[-20:]
    hs=(max(last[:10]) - max(last[10:]))
    ls=(min(last[10:]) - min(last[:10]))
    if hs>0 and ls>0: return "Triangolo ascendente"
    if last[-1]<last[0] and (max(last)-min(last))/max(1e-9,last[-1])<0.03: return "Flag ribassista"
    return None

# ---------- Data fetch ----------
async def td_series(symbol, interval, client):
    url="https://api.twelvedata.com/time_series"
    p={"symbol":symbol,"interval":interval,"apikey":TWELVE_KEY,"outputsize":500,"order":"ASC"}
    r=await client.get(url,params=p,timeout=30); r.raise_for_status(); return r.json()

async def fh_candle(symbol, interval, client):
    url="https://finnhub.io/api/v1/stock/candle"
    res={"15min":"15","1h":"60","1day":"D","1min":"1"}[interval]
    now=int(datetime.now(tz=timezone.utc).timestamp()); fro=now-60*60*24*60
    p={"symbol":symbol,"resolution":res,"from":fro,"to":now,"token":FINNHUB_KEY}
    r=await client.get(url,params=p,timeout=30); r.raise_for_status(); return r.json()

def parse_td(js):
    vals=js.get("values") or js.get("data") or []
    c,h,l,o,v=[],[],[],[],[]
    for row in vals:
        try:
            c.append(float(row["close"])); h.append(float(row["high"]))
            l.append(float(row["low"]));  o.append(float(row["open"]))
            v.append(float(row.get("volume",0.0)))
        except: pass
    return c,h,l,o,v

def parse_fh(js):
    if js.get("s")!="ok": return [],[],[],[],[]
    c=list(map(float,js.get("c",[]))); h=list(map(float,js.get("h",[])))
    l=list(map(float,js.get("l",[]))); o=list(map(float,js.get("o",[])))
    v=list(map(float,js.get("v",[]))) if "v" in js else [0.0]*len(c)
    return c,h,l,o,v

# ---------- Scoring ----------
def score_tf(c,h,l,v):
    if len(c)<50: return 0, {}
    ema9=ema(c,9); ema21=ema(c,21)
    r=rsi(c,14); m_line,m_sig,_=macd(c); a=atr(h,l,c,14); ad=adx(h,l,c,14)
    vol_sma = sma(v,20) if v else [0.0]*len(c)

    cond={}
    cond["EMA"]  = ema9[-1] < ema21[-1]
    cond["RSI"]  = r[-1] < 50.0
    cond["MACD"] = m_line[-1] < m_sig[-1]
    cond["COMP"] = compression_index(c,20) < 0.55
    cond["VOL"]  = (v[-1] > vol_sma[-1]) if v and vol_sma else False
    cond["ADX"]  = ad[-1] > 20.0
    cond["TREND_DOWN"] = (ema21[-1] < ema21[-2]) and (c[-1] < ema21[-1])

    score=0
    for k in ["EMA","RSI","MACD","COMP","ADX","TREND_DOWN"]:
        score += 2 if cond.get(k,False) else 0
    score += 1 if cond.get("VOL",False) else 0
    return min(10,score), cond

def cycle_phase(c):
    if len(c)<60: return "Neutrale"
    e21=ema(c,21); e50=ema(c,50)
    comp=compression_index(c,20)
    slope=(e21[-1]-e50[-1])/max(1e-9,c[-1])
    if comp<0.45 and abs(slope)<0.004: return "Accumulo"
    if e21[-1]<e50[-1] and abs(slope)>0.004: return "Correzione"
    if e21[-1]>e50[-1] and abs(slope)>0.004: return "Esplosione"
    return "Recupero" if e21[-1]>e50[-1] else "Neutrale"

def exceptional(c_d1):
    if len(c_d1)<120: return False
    last=c_d1[-120:]; rng=max(last)-min(last)
    if rng<=0: return False
    wig=sum(abs(last[i]-last[i-1]) for i in range(1,len(last)))/(rng*(len(last)-1))
    return wig<0.2

# ---------- Telegram ----------
async def tg_send(text, client):
    if SAFE_MODE or DRY_RUN:
        print("[SAFE/DRY] Telegram suppressed:\n"+text); return
    if not TG_TOKEN or not TG_CHAT:
        print("[WARN] TELEGRAM env missing; skipping send."); return
    url=f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload={"chat_id":TG_CHAT,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}
    r=await client.post(url,json=payload,timeout=20)
    try: r.raise_for_status()
    except Exception as e: print("[TG ERROR]", e, r.text)

def stars(n): return "★"*max(0,min(10,n)) + "☆"*(10-max(0,min(10,n)))
def ck(b): return "✅" if b else "❌"

def fmt_line_tf(tf, flags):
    parts=[
        f"{ck(flags.get('EMA',False))}EMA ↓",
        f"{ck(flags.get('RSI',False))}RSI",
        f"{ck(flags.get('MACD',False))}MACD ↓",
        f"{ck(flags.get('COMP',False))}Compressione",
    ]
    if flags.get("VOL",False): parts.append("✅Volumi")
    if flags.get("ADX",False): parts.append("✅ADX")
    return f"• {tf}: " + "  ".join(parts)

def build_message(symbol, price, change_pct, score, m15, h1, d1, phase, pattern, hot):
    red="🔴"; dir_txt="SHORT (ribasso)" if ENABLE_SHORT_ONLY else "Misto"
    header = f"{red} {symbol} {price:.2f} ({change_pct:+.2f}%) | {datetime.now(TZINFO).strftime('%H:%M')}\n🕰️ Direzione: {dir_txt}"
    lines=[
        header,"",
        "📊 Segnali attivi:",
        fmt_line_tf("M15", m15),
        fmt_line_tf("H1",  h1),
        fmt_line_tf("D1",  d1),"",
        f"🏷️ Fase attuale: {phase}",
        f"📐 Pattern attivo: {pattern if pattern else 'n.d.'}","",
        f"⭐ Score segnale: {stars(score)}",
    ]
    if hot: lines.append("🔥 Titolo caldo del ciclo")
    return "\n".join(lines)

# ---------- Helpers ----------
def within_hours(now_dt):
    try:
        sh,sm = map(int, ACTIVE_HOURS_START.split(":"))
        eh,em = map(int, ACTIVE_HOURS_END.split(":"))
        t=now_dt.timetz()
        return time(sh,sm) <= t <= time(eh,em)
    except: return True

# ---------- Core ----------
async def process_symbol(sym_key, client):
    out=[]; maps=SYMBOL_MAP.get(sym_key)
    if maps is None: return out

    data={}
    for tf in TF_LIST:
        c=h=l=o=v=[],[],[],[],[]
        try:
            td=await td_series(maps["twelvedata"], tf, client); c,h,l,o,v = parse_td(td)
        except: pass
        if len(c)<50:
            try:
                fh=await fh_candle(maps["finnhub"], tf, client); c,h,l,o,v = parse_fh(fh)
            except: pass
        data[tf]=(c,h,l,o,v)

    if len(data.get("15min",([],[],[],[],[]))[0])<50: return out

    total=0; flags={}
    for tf in ["15min","1h","1day"]:
        c,h,l,o,v = data[tf]
        s,f = score_tf(c,h,l,v)
        flags[tf]=f; total+=s
    score = min(10, max(0, int(round(total/3))))
    if score < SCORE_MIN: return out

    c15=data["15min"][0]; price=c15[-1]
    change_pct=((c15[-1]-c15[-2])/max(1e-9,c15[-2]))*100.0 if len(c15)>=2 else 0.0
    cD=data["1day"][0]; phase=cycle_phase(cD) if cD else "Neutrale"
    pattern=detect_pattern_basic(cD) if cD else None
    hot = score>=9 and exceptional(cD)

    msg=build_message(sym_key, price, change_pct, score, flags["15min"], flags["1h"], flags["1day"], phase, pattern, hot)
    await tg_send(msg, client)

    out.append({"symbol":sym_key,"score":score,"phase":phase,"pattern":pattern,"hot":hot,"time":datetime.now(TZINFO).isoformat()})
    return out

async def main_loop():
    global LAST_SIGNALS, LAST_ERROR
    async with httpx.AsyncClient() as client:
        while True:
            try:
                now=datetime.now(TZINFO)
                if not within_hours(now):
                    await asyncio.sleep(POLL_SECONDS); continue
                batch=[]
                for s in ASSETS:
                    try:
                        sigs=await process_symbol(s, client); batch.extend(sigs)
                    except Exception as e:
                        print("[ERR]", s, e)
                batch.sort(key=lambda x: x["score"], reverse=True)
                LAST_SIGNALS=batch[-50:]; LAST_ERROR=None
            except Exception as e:
                LAST_ERROR=str(e); print("[LOOP ERROR]", e)
            await asyncio.sleep(POLL_SECONDS)

# ---------- API ----------
app = FastAPI()

@app.api_route("/", methods=["GET","HEAD"])
def root():
    return {"service":"pre-rally-bot","started_at":STARTED_AT,"safe_mode":SAFE_MODE,"score_min":SCORE_MIN}

@app.api_route("/health", methods=["GET","HEAD"])
def health():
    return JSONResponse({"ok": LAST_ERROR is None, "error": LAST_ERROR, "signals_cached": len(LAST_SIGNALS)})

@app.get("/last_signals")
def last_signals():
    return JSONResponse(LAST_SIGNALS)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(main_loop())

if __name__=="__main__":
    import uvicorn
    port=int(os.getenv("PORT","8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
