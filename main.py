# main.py — Render Web + Strategia completa + /health
import os, threading, time, requests
from datetime import datetime, timedelta
import pandas as pd

import numpy as np
# compat per vecchi import interni: from numpy import NaN
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas_ta as ta
import pytz
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

# === ENV ===
TOKEN   = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TD_KEY  = os.getenv("TWELVEDATA_API_KEY", "").strip()
FH_KEY  = os.getenv("FINNHUB_API_KEY", "").strip()
PROVIDER = os.getenv("PROVIDER", "twelvedata").strip().lower()  # twelvedata | finnhub | both
TZ       = os.getenv("TZ", "Europe/Rome")
PORT     = int(os.getenv("PORT", "10000"))

tz = pytz.timezone(TZ)
TG_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

# === LISTA SIMBOLI (puoi modificare) ===
SYMBOLS = [
    "AAPL", "TSLA", "GOOGL", "PLTR",
    "REY.MI", "FCT.MI", "IFX.DE",
    "EUR/USD", "BTC/USD"
]

# ---------- Telegram ----------
def send(msg: str):
    if not TOKEN or not CHAT_ID: 
        print("⚠️ Manca TELEGRAM_TOKEN o TELEGRAM_CHAT_ID"); return
    try:
        requests.post(TG_URL, json={"chat_id": CHAT_ID, "text": msg, "parse_mode":"HTML"}, timeout=10)
    except Exception as e:
        print("Errore Telegram:", e)

def send_block(lines):
    send("\n".join(lines))

# ---------- Provider Dati ----------
def td_ohlc(symbol: str, interval: str, limit=400):
    url = "https://api.twelvedata.com/time_series"
    p = {"symbol": symbol, "interval": interval, "outputsize": limit, "apikey": TD_KEY, "format": "JSON"}
    r = requests.get(url, params=p, timeout=20)
    j = r.json()
    if "values" not in j: return None
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.rename(columns={"datetime":"time"})
    return df[["time","open","high","low","close","volume"]]

def fh_ohlc(symbol: str, interval: str, limit=400):
    # mappa intervalli: 15min->15, 1hour->60, 1day->D
    mapping = {"15min":15, "1hour":60, "1day":"D"}
    res = mapping.get(interval); 
    if res is None: return None
    if res == "D": resolution = "D"
    else: resolution = str(res)
    import time as _t
    to = int(_t.time())
    span = 400 * (60*15 if res!="D" else 86400)
    frm = to - span
    url = "https://finnhub.io/api/v1/stock/candle"
    p = {"symbol": symbol, "resolution": resolution, "from": frm, "to": to, "token": FH_KEY}
    r = requests.get(url, params=p, timeout=20)
    j = r.json()
    if j.get("s") != "ok": return None
    df = pd.DataFrame({
        "time": pd.to_datetime(j["t"], unit="s"),
        "open": j["o"], "high": j["h"], "low": j["l"], "close": j["c"], "volume": j["v"]
    }).sort_values("time").reset_index(drop=True)
    return df

def get_ohlc(symbol: str, interval: str):
    """Usa TwelveData, fallback Finnhub. Con PROVIDER=both prova in ordine TD->FH."""
    if PROVIDER in ("twelvedata","both"):
        df = td_ohlc(symbol, interval)
        if df is not None: return df
    if PROVIDER in ("finnhub","both"):
        # Attenzione: su Finnhub forex/crypto hanno simboli diversi; per semplicità preferiamo TD.
        return fh_ohlc(symbol, interval)
    return None

# ---------- Indicatori & Score ----------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema21"] = ta.ema(out["close"], length=21)
    out["ema50"] = ta.ema(out["close"], length=50)
    out["rsi"]   = ta.rsi(out["close"], length=14)
    macd = ta.macd(out["close"])
    out["macd"] = macd["MACD_12_26_9"]
    out["macd_signal"] = macd["MACDs_12_26_9"]
    adx = ta.adx(out["high"], out["low"], out["close"])
    out["adx"] = adx["ADX_14"]
    bb = ta.bbands(out["close"], length=20, std=2.0)
    out["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / out["close"]
    return out

def score_row(row):
    score = 0
    notes = []
    direction = "NEUTRAL"

    if row["ema21"] > row["ema50"]:
        direction = "LONG";  score += 2; notes.append("EMA ↑")
    elif row["ema21"] < row["ema50"]:
        direction = "SHORT"; score += 2; notes.append("EMA ↓")

    if row["rsi"] > 55: score += 1; notes.append("RSI forte")
    if row["rsi"] < 45: score += 1; notes.append("RSI debole")

    if row["macd"] > row["macd_signal"]: score += 1; notes.append("MACD ↑")
    else:                                   score += 1; notes.append("MACD ↓")

    if row["adx"] > 18: score += 2; notes.append("Trend")

    # compressione (bande strette)
    # soglia dinamica: quantile 20% sugli ultimi 200
    try:
        bw_q = row["bb_width_q"]
        if row["bb_width"] <= bw_q:
            score += 3; notes.append("Compressione")
    except:
        pass

    exceptional = ("Compressione" in notes) and (score >= 10)
    return score, direction, notes, exceptional

def format_stars(n):
    n = int(max(0, min(10, n)))
    return "★"*n + "☆"*(10-n)

def analyze_symbol(symbol: str):
    tf_list = [("15min","M15"), ("1hour","H1"), ("1day","D1")]
    per_tf = {}
    total = 0
    exceptional = False

    for tf, label in tf_list:
        df = get_ohlc(symbol, tf)
        if df is None or len(df) < 80: 
            continue
        ind = compute_indicators(df)
        # quantile su bb_width per compressione
        try:
            q = ind["bb_width"].tail(200).quantile(0.2)
            ind["bb_width_q"] = q
        except:
            ind["bb_width_q"] = np.nan
        last = ind.iloc[-1]
        sc, dirn, notes, exc = score_row(last)
        per_tf[tf] = {"score": sc, "dir": dirn, "notes": notes, "px": float(last["close"])}
        total += sc
        exceptional = exceptional or (tf == "1day" and exc)

    if not per_tf:
        return None

    # direzione “principale” = TF con score più alto
    main_dir = max(per_tf.values(), key=lambda x: x["score"])["dir"]
    return {
        "symbol": symbol,
        "per_tf": per_tf,
        "total": total,
        "direction": main_dir,
        "exceptional": exceptional
    }

def format_signal(s):
    d = s["direction"]
    arrow = "🟢" if d == "LONG" else ("🔴" if d == "SHORT" else "🟡")
    lines = [f"{arrow} <b>{s['symbol']}</b> | {d} | Score: <b>{format_stars(min(10,s['total']))}</b>"]
    lines.append("📊 <b>Segnali attivi:</b>")
    for tf in ["15min","1hour","1day"]:
        if tf in s["per_tf"]:
            tfname = {"15min":"M15","1hour":"H1","1day":"D1"}[tf]
            notes = " ".join("✅"+n for n in s["per_tf"][tf]["notes"][:4])
            lines.append(f"• {tfname}: {notes}")
    if s["exceptional"]:
        lines.append("🚨 <b>SEGNALE ECCEZIONALE</b> (Compressione D1 + score alto)")
    return lines

# ---------- LOOP ----------
def scan_once():
    results = []
    for sym in SYMBOLS:
        try:
            r = analyze_symbol(sym)
            if r and r["total"] >= 3:  # filtro minimo 3★
                results.append(r)
        except Exception as e:
            print(f"Errore su {sym}: {e}")
    results.sort(key=lambda x: x["total"], reverse=True)
    for r in results[:3]:  # invia i migliori
        send_block(format_signal(r))

def bot_loop():
    send("🚀 Bot online (Render Web) ✅")
    next_scan = datetime.now(tz)  # prima scansione subito
    while True:
        now = datetime.now(tz)
        if now >= next_scan:
            try:
                scan_once()
            except Exception as e:
                send(f"⚠️ Errore scansione: {e}")
            next_scan = now + timedelta(minutes=15)
        time.sleep(5)

# ---------- Web (Render) ----------
app = Flask(__name__)

@app.route("/")
def home(): return "OK"

@app.route("/health")
def health(): return "UP"

def start_bot_thread():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    start_bot_thread()
    app.run(host="0.0.0.0", port=PORT)

