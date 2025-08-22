import os, threading, time, math, traceback
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import pytz
from flask import Flask, jsonify, request

# =========================
# CONFIG BASE
# =========================
APP_NAME = "Pre‑Rally Scanner (FH primary, TD fallback)"
VERSION = "1.1.0"

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY","").strip()
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY","").strip()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN","").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","").strip()

TZ = pytz.timezone("Europe/Rome")
ACTIVE_START = (8,45)   # 08:45
ACTIVE_END   = (23,0)   # 23:00

# =========================
# LISTA 13 TITOLI USA (finale)
# =========================
ASSETS = [
    "PLTR","GOOGL","TSLA","NVDA","AMD","MU",   # Tech 6
    "VLO","GM","XOM",                           # Industrials/Energy 3
    "DIS","KO",                                 # Consumer/Media 2
    "JPM","BAC"                                 # Financial 2
]

# =========================
# TIMEFRAMES
# =========================
TFS = {
    "M15": {"fh":"15", "td":"15min"},
    "H1":  {"fh":"60", "td":"1h"},
    "D1":  {"fh":"D",  "td":"1day"},
}

# Anti-duplicati (1 segnale per candela/timeframe)
last_signal_key = {}  # {(symbol, tf): iso_of_candle}

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
    width = (upper - lower) / (ma.replace(0, np.nan)).abs()
    return ma, upper, lower, width

def candle_key(dt, tf_key):
    if tf_key=="M15":
        minute = (dt.minute//15)*15
        k = dt.replace(minute=minute, second=0, microsecond=0)
    elif tf_key=="H1":
        k = dt.replace(minute=0, second=0, microsecond=0)
    else: # D1
        k = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return k.isoformat()

def tg_send(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM-MOCK]\n", text[:1000])
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"HTML", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print("Telegram error:", r.text[:400])
    except Exception as e:
        print("Telegram exception:", e)

# =========================
# DATA PROVIDERS
# =========================
def fh_fetch(symbol, resolution, limit=320):
    if not FINNHUB_API_KEY: return None
    try:
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {"symbol": symbol, "resolution": resolution, "count": limit, "token": FINNHUB_API_KEY}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return None
        js = r.json()
        if js.get("s") != "ok": return None
        df = pd.DataFrame({
            "datetime": pd.to_datetime(js["t"], unit="s"),
            "open": js["o"], "high": js["h"], "low": js["l"], "close": js["c"], "volume": js["v"]
        })
        df = df.astype({"open":"float","high":"float","low":"float","close":"float","volume":"float"})
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception:
        return None

def td_fetch(symbol, interval, limit=320):
    if not TWELVEDATA_API_KEY: return None
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": symbol, "interval": interval, "outputsize": str(limit), "order": "asc", "apikey": TWELVEDATA_API_KEY}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return None
        js = r.json()
        if "values" not in js: return None
        df = pd.DataFrame(js["values"])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception:
        return None

def get_ohlc(symbol, tf_key, limit=320):
    tf = TFS[tf_key]
    # Primario: Finnhub
    df = fh_fetch(symbol, tf["fh"], limit=limit)
    if df is not None and len(df) >= 80:
        return df
    # Fallback: TwelveData
    df = td_fetch(symbol, tf["td"], limit=limit)
    return df

# =========================
# STRATEGIA LIGHT: SOLO PRE‑RALLY (compressione)
# Criteri (M15/H1/D1):
#  1) Compressione: BB width < quantile 20% su rolling 120
#  2) RSI neutro: 40–60
#  3) EMA20 ~ EMA50: distanza relativa < 0.4%
#  4) Volume non morto: vol >= 0.7× vol_sma20
# INVIO se punti >= MIN_POINTS_TO_SEND
# =========================
MIN_POINTS_TO_SEND = 2  # alza a 3 se vuoi meno notifiche

def pre_rally_signal(df):
    if df is None or len(df) < 140:
        return None
    d = df.copy()
    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["rsi14"] = rsi(d["close"], 14)
    mid, up, lo, bw = bbands(d["close"], 20, 2)
    d["bb_bw"] = bw
    d["vol_sma20"] = d["volume"].rolling(20).mean()

    bw_q20 = d["bb_bw"].rolling(120, min_periods=100).quantile(0.20)
    c = d.iloc[-1]

    cond_compress = (not math.isnan(c["bb_bw"])) and (not math.isnan(bw_q20.iloc[-1])) and (c["bb_bw"] < bw_q20.iloc[-1])
    cond_rsi_neutral = (40 <= c["rsi14"] <= 60)
    ema_rel_dist = abs(c["ema20"] - c["ema50"]) / max(abs(c["ema50"]), 1e-9)
    cond_ema_near = ema_rel_dist < 0.004
    cond_vol_ok = (not math.isnan(c["vol_sma20"])) and (c["vol_sma20"] > 0) and (c["volume"] >= 0.7 * c["vol_sma20"])

    points = int(cond_compress) + int(cond_rsi_neutral) + int(cond_ema_near) + int(cond_vol_ok)

    return {
        "ok": points >= MIN_POINTS_TO_SEND,
        "points": points,
        "compress": bool(cond_compress),
        "rsi": float(c["rsi14"]) if not math.isnan(c["rsi14"]) else None,
        "ema20": float(c["ema20"]) if not math.isnan(c["ema20"]) else None,
        "ema50": float(c["ema50"]) if not math.isnan(c["ema50"]) else None,
        "vol_ok": bool(cond_vol_ok),
        "last_dt": pd.to_datetime(d.iloc[-1]["datetime"]).to_pydatetime(),
        "last_close": float(c["close"]),
    }

def format_msg(symbol, tf_key, ind):
    when = ind["last_dt"].astimezone(TZ).strftime("%d/%m %H:%M")
    bullets = []
    bullets.append("Compressione ✅" if ind["compress"] else "Compressione —")
    bullets.append(f"RSI14 {ind['rsi']:.1f}" if ind["rsi"] is not None else "RSI —")
    if ind["ema20"] is not None and ind["ema50"] is not None:
        pct = abs(ind["ema20"]-ind["ema50"]) / max(abs(ind["ema50"]),1e-9) * 100
        bullets.append(f"EMA20≈EMA50 ({pct:.2f}%)")
    else:
        bullets.append("EMA20≈EMA50 —")
    bullets.append("Vol OK" if ind["vol_ok"] else "Vol —")

    return (
        f"<b>{symbol} · {tf_key}</b>\n"
        f"Setup: <b>Pre‑Rally</b> · Punti: <b>{ind['points']}</b>/4\n"
        f"Prezzo: <b>{ind['last_close']:.4f}</b>\n"
        f"{' · '.join(bullets)}\n"
        f"<i>{APP_NAME} v{VERSION} · {when}</i>"
    )

# =========================
# SCAN & SCHEDULER
# =========================
def boundary_minutes(dt):
    return {
        "M15": dt.minute % 15 == 0,
        "H1":  dt.minute == 0,
        "D1":  dt.hour == 0 and dt.minute == 0
    }

def run_scan_for_timeframe(tf_key):
    for sym in ASSETS:
        try:
            df = get_ohlc(sym, tf_key, limit=240 if tf_key!="D1" else 400)
            if df is None or len(df) < 140:
                print(f"[{tf_key}] {sym}: dati insufficienti")
                continue
            sig = pre_rally_signal(df)
            if not sig or not sig["ok"]:
                continue

            ck = candle_key(sig["last_dt"], tf_key)
            k = (sym, tf_key)
            if last_signal_key.get(k) == ck:
                continue

            tg_send(format_msg(sym, tf_key, sig))
            last_signal_key[k] = ck

        except Exception as e:
            print(f"Errore {sym} {tf_key}:", e)
            traceback.print_exc()

def scheduler_loop():
    print(f"== {APP_NAME} v{VERSION} avviato ==")
    while True:
        try:
            dt = now_rome()
            if in_active_hours(dt):
                b = boundary_minutes(dt)
                if b["D1"]: run_scan_for_timeframe("D1")
                if b["H1"]: run_scan_for_timeframe("H1")
                if b["M15"]: run_scan_for_timeframe("M15")
            else:
                if dt.hour == 23 and dt.minute >= 1:
                    last_signal_key.clear()
            time.sleep(max(1, 60 - dt.second))
        except Exception as e:
            print("Scheduler exception:", e)
            time.sleep(10)

# =========================
# FLASK ROUTES
# =========================
@app.route("/")
def home():
    return jsonify({"status":"running","app":APP_NAME,"version":VERSION,"time":now_rome().isoformat()})

@app.route("/health")
def health():
    return jsonify({"status":"ok","app":APP_NAME,"version":VERSION,"time":now_rome().isoformat()}), 200

@app.route("/force", methods=["GET"])
def force():
    tf = (request.args.get("tf","M15") or "M15").upper()
    if tf not in TFS:
        tf = "M15"
    run_scan_for_timeframe(tf)
    return jsonify({"forced_tf": tf, "t": now_rome().isoformat()})

def main():
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    port = int(os.getenv("PORT","10000"))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()                   
