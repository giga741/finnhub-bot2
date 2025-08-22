import os, threading, time, math, json, traceback
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import pytz
from flask import Flask, jsonify, request

# =========================
# CONFIGURAZIONE BASE
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()

TZ = pytz.timezone("Europe/Rome")
ACTIVE_START = (8, 45)   # 08:45
ACTIVE_END   = (23, 0)   # 23:00
MIN_SCORE = 3            # filtro minimo attivo (utente approvato)
SEND_EXCEPTIONAL = True  # notifica speciale 10/10 + compressione D1

APP_NAME = "Pre-Rally Scanner"
VERSION = "1.2.0"

# =========================
# LISTA ASSET (20) + MAPPATURE
# td = TwelveData; fh = Finnhub (fallback)
# type: stock | forex | crypto  (M1 attivo solo su forex/crypto)
# =========================
ASSETS = {
    # Tech
    "PLTR":  {"td":"PLTR",         "fh":"PLTR",     "type":"stock", "name":"Palantir"},
    "GOOGL": {"td":"GOOGL",        "fh":"GOOGL",    "type":"stock", "name":"Alphabet A"},
    "TSLA":  {"td":"TSLA",         "fh":"TSLA",     "type":"stock", "name":"Tesla"},
    "AAPL":  {"td":"AAPL",         "fh":"AAPL",     "type":"stock", "name":"Apple"},
    "IFX":   {"td":"IFX:XETRA",    "fh":"IFX.DE",   "type":"stock", "name":"Infineon"},
    "REY":   {"td":"REY:XMIL",     "fh":"REY.MI",   "type":"stock", "name":"Reply"},
    "MU":    {"td":"MU",           "fh":"MU",       "type":"stock", "name":"Micron"},
    "AMD":   {"td":"AMD",          "fh":"AMD",      "type":"stock", "name":"AMD"},
    "NVDA":  {"td":"NVDA",         "fh":"NVDA",     "type":"stock", "name":"NVIDIA"},
    "ARM":   {"td":"ARM",          "fh":"ARM",      "type":"stock", "name":"ARM Holdings"},
    # Industrials / Energy
    "VLO":   {"td":"VLO",          "fh":"VLO",      "type":"stock", "name":"Valero Energy"},
    "GM":    {"td":"GM",           "fh":"GM",       "type":"stock", "name":"General Motors"},
    # Consumer / Lusso
    "MC":    {"td":"MC:EPA",       "fh":"MC.PA",    "type":"stock", "name":"LVMH"},
    "DIS":   {"td":"DIS",          "fh":"DIS",      "type":"stock", "name":"Disney"},
    "KO":    {"td":"KO",           "fh":"KO",       "type":"stock", "name":"Coca-Cola"},
    # Forex
    "EUR/USD":{"td":"EUR/USD",     "fh":"OANDA:EUR_USD", "type":"forex", "name":"EUR/USD"},
    "USD/JPY":{"td":"USD/JPY",     "fh":"OANDA:USD_JPY", "type":"forex", "name":"USD/JPY"},
    "GBP/USD":{"td":"GBP/USD",     "fh":"OANDA:GBP_USD", "type":"forex", "name":"GBP/USD"},
    # Crypto
    "ETH/USD":{"td":"ETH/USD",     "fh":"",         "type":"crypto","name":"Ethereum/USD"},
    "BTC/USD":{"td":"BTC/USD",     "fh":"",         "type":"crypto","name":"Bitcoin/USD"},
}

# =========================
# TIMEFRAME
# =========================
TFS = {
    "M1":  {"td":"1min",  "fh":"1"},
    "M15": {"td":"15min", "fh":"15"},
    "H1":  {"td":"1h",    "fh":"60"},
    "D1":  {"td":"1day",  "fh":"D"},
}

# =========================
# STATO: per evitare duplicati
# =========================
last_signal_key = {}  # {(symbol, tf): candle_time_iso}
app = Flask(__name__)

# =========================
# UTILS
# =========================
def now_rome():
    return datetime.now(TZ)

def in_active_hours(dt=None):
    dt = dt or now_rome()
    start = dt.replace(hour=ACTIVE_START[0], minute=ACTIVE_START[1], second=0, microsecond=0)
    end   = dt.replace(hour=ACTIVE_END[0],   minute=ACTIVE_END[1],   second=0, microsecond=0)
    return start <= dt <= end

def star_bar(score):
    # score 0-10 -> 5 stelle
    full = int(round(score/2.0))
    return "★"*full + "☆"*(5-full)

def pct(x):
    return f"{int(round(10* x))/10:.1f}%"

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

# =========================
# DATA PROVIDERS
# =========================
def fetch_td(symbol_td, tf_td, limit=300):
    """
    TwelveData time_series
    """
    if not TWELVEDATA_API_KEY or not symbol_td:
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol_td,
        "interval": tf_td,
        "outputsize": str(limit),
        "order": "asc",
        "apikey": TWELVEDATA_API_KEY
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    # cols: datetime, open, high, low, close, volume
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def fetch_fh_stock(symbol_fh, tf_fh, limit=300):
    if not FINNHUB_API_KEY or not symbol_fh:
        return None
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {"symbol":symbol_fh, "resolution":tf_fh, "count":limit, "token":FINNHUB_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code!=200:
        return None
    js = r.json()
    if js.get("s")!="ok":
        return None
    df = pd.DataFrame({"datetime": pd.to_datetime(js["t"], unit="s"),
                       "open": js["o"], "high": js["h"], "low": js["l"], "close": js["c"], "volume": js["v"]})
    df = df.astype({"open":"float","high":"float","low":"float","close":"float","volume":"float"})
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def fetch_fh_fx(symbol_fh, tf_fh, limit=300):
    if not FINNHUB_API_KEY or not symbol_fh:
        return None
    url = "https://finnhub.io/api/v1/forex/candle"
    params = {"symbol":symbol_fh, "resolution":tf_fh, "count":limit, "token":FINNHUB_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code!=200:
        return None
    js = r.json()
    if js.get("s")!="ok":
        return None
    df = pd.DataFrame({"datetime": pd.to_datetime(js["t"], unit="s"),
                       "open": js["o"], "high": js["h"], "low": js["l"], "close": js["c"], "volume": js["v"]})
    df = df.astype({"open":"float","high":"float","low":"float","close":"float","volume":"float"})
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def get_ohlc(symbol_key, tf_key, limit=300):
    """
    Prova TwelveData -> Finnhub (fallback)
    tf_key: 'M1'|'M15'|'H1'|'D1'
    """
    m = ASSETS[symbol_key]
    tf = TFS[tf_key]
    # Primary: TwelveData
    df = fetch_td(m["td"], tf["td"], limit=limit)
    if df is not None and len(df)>=50:
        return df
    # Fallback: Finnhub
    if m["type"]=="stock":
        df = fetch_fh_stock(m["fh"], tf["fh"], limit=limit)
    elif m["type"]=="forex":
        df = fetch_fh_fx(m["fh"], tf["fh"], limit=limit)
    else:
        df = None  # crypto: preferiamo TD
    return df

# =========================
# INDICATORI
# =========================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    up = np.where(delta>0, delta, 0.0)
    down = np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    return 100 - (100/(1+rs))

def bbands(close, length=20, stdev=2):
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = ma + stdev*sd
    lower = ma - stdev*sd
    width = (upper - lower) / (ma.replace(0,np.nan)).abs()
    return ma, upper, lower, width

def atr(df, period=14):
    h,l,c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h-l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def slope(series, length=10):
    # pendenza semplice: regressione lineare su indice
    if len(series) < length:
        return pd.Series([np.nan]*len(series), index=series.index)
    out = []
    for i in range(len(series)):
        if i < length-1:
            out.append(np.nan)
            continue
        y = series.iloc[i-length+1:i+1]
        x = np.arange(length)
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y.values, rcond=None)[0]
        out.append(m)
    return pd.Series(out, index=series.index)

def detect_triangle(df, lookback=20):
    # Heuristica leggera: HH in calo e LL in aumento
    highs = df["high"].rolling(lookback).max()
    lows  = df["low"].rolling(lookback).min()
    # approssimiamo pendenza di linee HH e LL
    hh_slope = slope(highs.fillna(method="bfill"), length=min(lookback, 10))
    ll_slope = slope(lows.fillna(method="bfill"),  length=min(lookback, 10))
    if pd.isna(hh_slope.iloc[-1]) or pd.isna(ll_slope.iloc[-1]):
        return False
    return (hh_slope.iloc[-1] < 0) and (ll_slope.iloc[-1] > 0)

def support_resistance(df, left=3, right=3):
    # pivot semplice ultimo
    highs = df["high"].values
    lows  = df["low"].values
    n = len(df)
    sr = {"support": None, "resistance": None}
    for i in range(n-right-1, left+right, -1):
        # pivot high
        if all(highs[i] > highs[i-j] for j in range(1, left+1)) and all(highs[i] > highs[i+j] for j in range(1, right+1)):
            sr["resistance"] = highs[i]; break
    for i in range(n-right-1, left+right, -1):
        # pivot low
        if all(lows[i] < lows[i-j] for j in range(1, left+1)) and all(lows[i] < lows[i+j] for j in range(1, right+1)):
            sr["support"] = lows[i]; break
    return sr

def weekly_cycle_heatmap(d1_df):
    # Fasi: accumulo / esplosione / correzione / recupero
    if d1_df is None or len(d1_df)<60:
        return "—"
    # settimanalizza
    w = d1_df.set_index("datetime")[["close","high","low"]].resample("W").last().dropna()
    e20 = ema(w["close"], 10)
    e50 = ema(w["close"], 20)
    bw_ma, up, lo, bw = bbands(w["close"], 20, 2)
    atr_w = (w["high"]-w["low"]).rolling(6).mean()
    cond_trend_up = (e20 > e50) & (e20.diff()>0)
    cond_trend_dn = (e20 < e50) & (e20.diff()<0)
    cond_compress = bw < bw.rolling(60).quantile(0.2)
    phase = "accumulo"
    if cond_trend_up.iloc[-1] and not cond_compress.iloc[-1]:
        phase = "esplosione"
    elif cond_trend_dn.iloc[-1] and not cond_compress.iloc[-1]:
        phase = "correzione"
    elif cond_compress.iloc[-1]:
        phase = "accumulo"
    else:
        phase = "recupero"
    return phase

def indicators_and_score(df):
    """ Calcola indicatori + score long/short """
    out = {}
    if df is None or len(df) < 60:
        out["ok"] = False
        return out
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    ma, up, lo, bw = bbands(df["close"], 20, 2)
    df["bb_mid"], df["bb_up"], df["bb_lo"], df["bb_bw"] = ma, up, lo, bw
    df["atr14"] = atr(df, 14)
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["ema20_slope"] = slope(df["ema20"].fillna(method="bfill"), 8)

    # compressione: larghezza BB sotto 20° percentile su rolling 120
    roll_quant = df["bb_bw"].rolling(120, min_periods=60).quantile(0.2)
    df["compress"] = df["bb_bw"] < roll_quant

    # condizioni LONG
    c = df.iloc[-1]
    prev = df.iloc[-2]
    long_pts = 0
    short_pts = 0

    # trend
    if c["ema20"] > c["ema50"]: long_pts += 1
    if c["ema20"] < c["ema50"]: short_pts += 1

    # momentum
    if c["rsi14"] > 52: long_pts += 1
    if c["rsi14"] < 48: short_pts += 1

    # posizione prezzo
    if c["close"] > c["ema20"]: long_pts += 1
    if c["close"] < c["ema20"]: short_pts += 1

    # compressione (pre-rally)
    if bool(c["compress"]): 
        long_pts += 2; short_pts += 2  # neutra ma propedeutica

    # slope ema20
    if c["ema20_slope"] > 0: long_pts += 1
    if c["ema20_slope"] < 0: short_pts += 1

    # breakout micro: chiusura sopra banda alta / sotto banda bassa
    if c["close"] > c["bb_up"]: long_pts += 2
    if c["close"] < c["bb_lo"]: short_pts += 2

    # volume relativo
    if c["volume"] > 1.2*(c["vol_sma20"] if not math.isnan(c["vol_sma20"]) else c["volume"]+1):
        # volume alto avvalora la direzione della candela
        if c["close"] >= prev["close"]: 
            long_pts += 1
        else:
            short_pts += 1

    # pattern base
    pat_triangle = detect_triangle(df, 20)
    if pat_triangle:
        long_pts += 1; short_pts += 1

    direction = "LONG" if long_pts >= short_pts else "SHORT"
    score = long_pts if direction=="LONG" else short_pts

    # supporti/resistenze
    sr = support_resistance(df, 3, 3)
    sr_flag = None
    if sr["resistance"] and c["close"]>sr["resistance"]:
        sr_flag = "Breakout Resistenza"
        score += 1 if direction=="LONG" else 0
    if sr["support"] and c["close"]<sr["support"]:
        sr_flag = "Breakdown Supporto"
        score += 1 if direction=="SHORT" else 0

    out.update({
        "ok": True,
        "score": int(score),
        "direction": direction,
        "rsi": float(c["rsi14"]),
        "ema20": float(c["ema20"]),
        "ema50": float(c["ema50"]),
        "bb_bw": float(c["bb_bw"]) if not math.isnan(c["bb_bw"]) else None,
        "compress": bool(c["compress"]),
        "pattern_triangle": bool(pat_triangle),
        "sr_flag": sr_flag,
        "last_dt": pd.to_datetime(df.iloc[-1]["datetime"]).to_pydatetime(),
        "last_close": float(c["close"]),
        "last_high": float(c["high"]),
        "last_low": float(c["low"]),
    })
    return out

# =========================
# TELEGRAM
# =========================
def tg_send(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] Token/Chat ID non impostati. Messaggio:\n", text[:2000])
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code!=200:
            print("Telegram error:", r.text[:500])
    except Exception as e:
        print("Telegram exception:", e)

def format_signal(symbol_key, tf_key, ind, weekly_phase, asset_name):
    dir_label = "🧭 Direzione: LONG (rialzo)" if ind["direction"]=="LONG" else "🧭 Direzione: SHORT (ribasso)"
    stars = star_bar(ind["score"])
    compress_txt = "Compressione attiva ✅" if ind["compress"] else "Compressione assente"
    sr_txt = f" | <b>{ind['sr_flag']}</b>" if ind.get("sr_flag") else ""
    patt_txt = "Triangolo ↗↘ rilevato" if ind["pattern_triangle"] else "Pattern: —"
    hot = "🔥 Titolo caldo del ciclo" if (weekly_phase=="esplosione" and ind["score"]>=7) else ""
    when = ind["last_dt"].astimezone(TZ).strftime("%d/%m %H:%M")
    header = f"<b>{asset_name} ({symbol_key}) · {tf_key}</b>\n{stars}"
    body = (
        f"{dir_label}\n"
        f"Prezzo: <b>{ind['last_close']:.4f}</b>  | RSI14: <b>{ind['rsi']:.1f}</b>\n"
        f"EMA20 {ind['ema20']:.4f} · EMA50 {ind['ema50']:.4f}\n"
        f"{compress_txt}{sr_txt}\n"
        f"{patt_txt} · Fase ciclo settimanale: <b>{weekly_phase}</b>\n"
        f"<i>{APP_NAME} v{VERSION} · {when}</i>"
    )
    return f"{header}\n\n{body}\n{hot}".strip()

def format_exceptional(symbol_key, asset_name, tf_key, ind):
    when = ind["last_dt"].astimezone(TZ).strftime("%d/%m %H:%M")
    title = f"〽️ <b>SEGNALE ECCEZIONALE</b> · {asset_name} ({symbol_key}) · {tf_key}"
    stars = "★★★★★"
    line1 = f"{stars}  Score: <b>{ind['score']}/10</b>  |  {('LONG' if ind['direction']=='LONG' else 'SHORT')}"
    line2 = f"Compressione multi‑day su D1 + trigger {tf_key} confermato"
    line3 = f"Prezzo {ind['last_close']:.4f} · RSI {ind['rsi']:.1f} · EMA20 {ind['ema20']:.4f} / EMA50 {ind['ema50']:.4f}"
    line4 = f"<i>{APP_NAME} v{VERSION} · {when}</i>"
    return "\n".join([title, line1, line2, line3, line4])

# =========================
# SCAN LOGIC
# =========================
def candle_key(dt, tf_key):
    # normalizza timestamp della candela per anti-duplicati
    if tf_key=="M1":
        k = dt.replace(second=0, microsecond=0)
    elif tf_key=="M15":
        minute = (dt.minute//15)*15
        k = dt.replace(minute=minute, second=0, microsecond=0)
    elif tf_key=="H1":
        k = dt.replace(minute=0, second=0, microsecond=0)
    else:
        k = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return k.isoformat()

def exceptional_check(symbol_key, ind):
    if not SEND_EXCEPTIONAL or ind["score"]<10:
        return False
    # compressione forte su D1
    d1 = get_ohlc(symbol_key, "D1", limit=200)
    if d1 is None or len(d1)<120:
        return False
    ma, up, lo, bw = bbands(d1["close"], 20, 2)
    last_bw = float(bw.iloc[-1]) if not math.isnan(bw.iloc[-1]) else None
    thr = float(bw.rolling(120).quantile(0.15).iloc[-1]) if len(bw.dropna())>50 else None
    if last_bw and thr and last_bw < thr:
        return True
    return False

def run_scan_for_timeframe(tf_key):
    for sym, meta in ASSETS.items():
        # M1 solo forex/crypto
        if tf_key=="M1" and meta["type"] not in ("forex","crypto"):
            continue
        try:
            df = get_ohlc(sym, tf_key, limit=320)
            if df is None or len(df)<60:
                print(f"[{tf_key}] {sym}: dati insufficienti")
                continue
            ind = indicators_and_score(df)
            if not ind.get("ok"):
                continue

            # anti-duplicati per candela
            k = (sym, tf_key)
            ck = candle_key(ind["last_dt"], tf_key)
            if last_signal_key.get(k)==ck:
                continue

            # filtro score
            if ind["score"] < MIN_SCORE:
                continue

            # ciclo settimanale (usa D1)
            d1 = get_ohlc(sym, "D1", limit=220)
            weekly_phase = weekly_cycle_heatmap(d1)

            text = format_signal(sym, tf_key, ind, weekly_phase, meta["name"])
            tg_send(text)

            # eccezionale
            if exceptional_check(sym, ind):
                tg_send(format_exceptional(sym, meta["name"], tf_key, ind))

            last_signal_key[k] = ck

        except Exception as e:
            print(f"Errore su {sym} {tf_key}:", e)
            traceback.print_exc()

def boundary_minutes(dt):
    return {
        "M1": True,
        "M15": dt.minute % 15 == 0,
        "H1": dt.minute == 0
    }

def scheduler_loop():
    print(f"== {APP_NAME} v{VERSION} avviato ==")
    while True:
        try:
            dt = now_rome()
            if in_active_hours(dt):
                b = boundary_minutes(dt)
                if b["H1"]:
                    run_scan_for_timeframe("H1")
                if b["M15"]:
                    run_scan_for_timeframe("M15")
                # M1: solo fx/crypto, ogni minuto
                run_scan_for_timeframe("M1")
            else:
                # reset anti-duplicati a fine giornata
                if dt.hour == 23 and dt.minute >= 1:
                    last_signal_key.clear()
            # sincronizza al prossimo minuto
            sleep_s = 60 - dt.second
            time.sleep(max(1, sleep_s))
        except Exception as e:
            print("Scheduler exception:", e)
            time.sleep(10)

# =========================
# FLASK (health + trigger)
# =========================
@app.route("/")
def home():
    return jsonify({"app": APP_NAME, "version": VERSION, "status":"ok", "time": now_rome().isoformat()})

@app.route("/force", methods=["POST","GET"])
def force():
    tf = request.args.get("tf","M15").upper()
    if tf not in ("M1","M15","H1","D1"):
        tf = "M15"
    run_scan_for_timeframe(tf)
    return jsonify({"forced": tf, "time": now_rome().isoformat()})

def main():
    # avvio thread scheduler
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    # avvio web
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
