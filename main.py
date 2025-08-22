import os, time, math, json, threading, traceback, statistics
from datetime import datetime, timedelta
import pytz
import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify

# =========================
# CONFIG
# =========================
TZ = pytz.timezone("Europe/Rome")
ACTIVE_START = (8, 45)   # 08:45
ACTIVE_END   = (23, 0)   # 23:00
CHECK_EVERY_MIN = 15     # run cadence in minutes
MIN_SCORE_TO_ALERT = 3

# Secrets (Render -> Environment)
FINNHUB_API_KEY     = os.getenv("FINNHUB_API_KEY", "")
TWELVEDATA_API_KEY  = os.getenv("TWELVEDATA_API_KEY", "")
TELEGRAM_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
# Facoltativo: limiti
TD_RATE_SLEEP = float(os.getenv("TD_RATE_SLEEP", "1.2"))  # pausa tra chiamate TwelveData

# =========================
# ASSET LIST & SYMBOL MAPS
# =========================
# type: equity | forex | crypto
ASSETS = [
    # --- Tech US
    {"name":"Palantir", "type":"equity", "fin":"PLTR",   "td":"PLTR"},
    {"name":"Google (Alphabet A)", "type":"equity", "fin":"GOOGL", "td":"GOOGL"},
    {"name":"Tesla",   "type":"equity", "fin":"TSLA",   "td":"TSLA"},
    {"name":"Apple",   "type":"equity", "fin":"AAPL",   "td":"AAPL"},
    {"name":"Micron",  "type":"equity", "fin":"MU",     "td":"MU"},
    {"name":"AMD",     "type":"equity", "fin":"AMD",    "td":"AMD"},
    # --- EU Tech / Industrial
    {"name":"Infineon",     "type":"equity", "fin":"IFX.DE", "td":"IFX:XETRA"},
    {"name":"Reply",        "type":"equity", "fin":"REY.MI", "td":"REY:MI"},
    {"name":"Fincantieri",  "type":"equity", "fin":"FCT.MI", "td":"FCT:MI"},
    # --- Energy / Auto / Consumer
    {"name":"ExxonMobil", "type":"equity", "fin":"XOM",   "td":"XOM"},
    {"name":"Valero",     "type":"equity", "fin":"VLO",   "td":"VLO"},
    {"name":"General Motors", "type":"equity", "fin":"GM", "td":"GM"},
    {"name":"LVMH",       "type":"equity", "fin":"MC.PA", "td":"MC:PAR"},
    {"name":"Coca-Cola",  "type":"equity", "fin":"KO",    "td":"KO"},
    {"name":"Disney",     "type":"equity", "fin":"DIS",   "td":"DIS"},
    # --- Forex
    {"name":"EUR/USD", "type":"forex", "fin":"OANDA:EUR_USD", "td":"EUR/USD"},
    {"name":"USD/JPY", "type":"forex", "fin":"OANDA:USD_JPY", "td":"USD/JPY"},
    {"name":"GBP/USD", "type":"forex", "fin":"OANDA:GBP_USD", "td":"GBP/USD"},
    # --- Crypto (USD)
    {"name":"ETH/USD", "type":"crypto", "fin":"COINBASE:ETH-USD", "td":"ETH/USD"},
    {"name":"BTC/USD", "type":"crypto", "fin":"COINBASE:BTC-USD", "td":"BTC/USD"},
]

# quali asset usano M1 oltre a M15/H1
M1_WHITELIST = {a["name"] for a in ASSETS if a["type"] in ("forex","crypto")}

# =========================
# UTILS
# =========================
def now_rome():
    return datetime.now(TZ)

def within_active_hours(ts: datetime):
    start = ts.replace(hour=ACTIVE_START[0], minute=ACTIVE_START[1], second=0, microsecond=0)
    end   = ts.replace(hour=ACTIVE_END[0],   minute=ACTIVE_END[1],   second=0, microsecond=0)
    return start <= ts <= end

def unix_time(dt):
    return int(dt.timestamp())

def safe_float(x):
    try: return float(x)
    except: return np.nan

def pct(a, b):
    if b == 0 or math.isclose(b,0.0): return 0.0
    return 100.0 * (a - b) / b

def stars_from_score(score):
    # 0..10 -> 0..5 stelle (½ arrotondato)
    s = max(0, min(10, int(round(score))))
    half = s / 2.0
    full = int(half)
    half_star = (half - full) >= 0.5
    return "★" * full + ("½" if half_star else "") + "☆" * (5 - full - (1 if half_star else 0))

# =========================
# DATA PROVIDERS
# =========================
def fetch_finnhub_candles(symbol, market_type, resolution, bars=300):
    """
    resolution: '1','5','15','60','D'
    market_type: 'equity' | 'forex' | 'crypto'
    """
    if not FINNHUB_API_KEY:
        return None
    end = int(time.time())
    # 300 bars* interval
    if resolution == 'D':
        start = end - 86400*500
    elif resolution == '60':
        start = end - 3600*500
    elif resolution == '15':
        start = end - 900*500
    elif resolution == '5':
        start = end - 300*500
    else:
        start = end - 60*500
    base = "https://finnhub.io/api/v1"
    if market_type == 'equity':
        url = f"{base}/stock/candle"
        params = {"symbol": symbol, "resolution": resolution, "from": start, "to": end, "token": FINNHUB_API_KEY}
    elif market_type == 'forex':
        url = f"{base}/forex/candle"
        params = {"symbol": symbol, "resolution": resolution, "from": start, "to": end, "token": FINNHUB_API_KEY}
    else: # crypto
        url = f"{base}/crypto/candle"
        params = {"symbol": symbol, "resolution": resolution, "from": start, "to": end, "token": FINNHUB_API_KEY}

    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return None
    j = r.json()
    if j.get("s") != "ok":
        return None
    df = pd.DataFrame({
        "datetime": pd.to_datetime(j["t"], unit="s"),
        "open": j["o"], "high": j["h"], "low": j["l"], "close": j["c"], "volume": j.get("v", [np.nan]*len(j["c"]))
    })
    df = df.dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)
    return df

def fetch_twelvedata_series(symbol, interval, outputsize=300):
    if not TWELVEDATA_API_KEY:
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        return None
    j = r.json()
    if "values" not in j:
        return None
    vals = j["values"]
    df = pd.DataFrame(vals)
    # TD returns strings
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime","open","high","low","close","volume"]]

def get_ohlc(asset_name, tf):
    """
    tf: '1min' | '15min' | '1h' | '1day'
    maps to Finnhub resolutions: '1','15','60','D'
    """
    asset = next(a for a in ASSETS if a["name"] == asset_name)
    res_map = {"1min":"1", "15min":"15", "1h":"60", "1day":"D"}
    # 1) prova Finnhub
    df = fetch_finnhub_candles(asset["fin"], asset["type"], res_map[tf])
    if df is not None and len(df) > 50:
        return df
    # 2) fallback TwelveData
    td = {"1min":"1min","15min":"15min","1h":"1h","1day":"1day"}[tf]
    df2 = fetch_twelvedata_series(asset["td"], td, outputsize=400 if tf!="1day" else 600)
    return df2

# =========================
# INDICATORI
# =========================
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1*delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50.0)

def bollinger_width(close, n=20):
    ma = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = ma + 2*std
    lower = ma - 2*std
    width = (upper - lower) / ma
    return width, ma, upper, lower

def ema_distance(close, n1=20, n2=50):
    e1 = ema(close, n1)
    e2 = ema(close, n2)
    dist = (e1 - e2).abs() / close
    trend_up = (close > e1) & (e1 > e2)
    trend_dn = (close < e1) & (e1 < e2)
    return dist, trend_up, trend_dn, e1, e2

def volume_surge(vol, n=20):
    base = vol.rolling(n).mean()
    return vol / (base.replace(0,np.nan))

def bb_compression_percentile(width, lookback=120):
    # percentile del valore attuale rispetto alla storia recente
    hist = width.tail(lookback+1).iloc[:-1].dropna()
    if len(hist) < 10 or math.isnan(width.iloc[-1]):
        return 100.0
    rank = sum(hist <= width.iloc[-1]) / len(hist) * 100.0
    return rank

def simple_support_resistance(df, lookback=40, swing=3):
    # trova swing high/low recenti
    highs = df["high"].values
    lows  = df["low"].values
    res_levels = []
    sup_levels = []
    for i in range(swing, len(df)-swing):
        if highs[i] == max(highs[i-swing:i+swing+1]):
            res_levels.append(highs[i])
        if lows[i] == min(lows[i-swing:i+swing+1]):
            sup_levels.append(lows[i])
    res = np.median(res_levels[-5:]) if res_levels else np.nan
    sup = np.median(sup_levels[-5:]) if sup_levels else np.nan
    return sup, res

def triangle_flag_heuristic(df, window=30):
    seg = df.tail(window)
    highs = seg["high"].reset_index(drop=True)
    lows  = seg["low"].reset_index(drop=True)
    x = np.arange(len(seg))
    # regressione lineare
    def slope(y):
        xmean = x.mean(); ymean = y.mean()
        num = ((x - xmean)*(y - ymean)).sum()
        den = ((x - xmean)**2).sum()
        return num/den if den!=0 else 0.0
    s_high = slope(highs.values)
    s_low  = slope(lows.values)
    # convergenza se opposte e in modulo non troppo grandi
    if (s_high < 0 and s_low > 0) and (abs(s_high)+abs(s_low) < (highs.mean()*0.02)):
        return "triangolo"
    # flag: trend deciso + canalino leggero contrario
    closes = seg["close"].values
    trend = slope(pd.Series(closes))
    if abs(trend) > (seg["close"].mean()*0.02):
        return "flag"
    return None

def weekly_phase_from_daily(df_d1):
    # Fasi: accumulo (piatto con range stretto), esplosione (trend + range ampio), correzione (contro-trend)
    closes = df_d1["close"]
    width, ma, up, lo = bollinger_width(closes, n=20)
    e20 = ema(closes, 20)
    slope = (e20.iloc[-1] - e20.iloc[-5]) / (5 if len(e20)>5 else 1)
    rng = (up - lo).iloc[-1]
    narrow = bb_compression_percentile(width, lookback=100) < 30 and rng < (closes.iloc[-1]*0.06)
    strong = abs(slope) > closes.iloc[-1]*0.002
    if narrow and not strong:
        return "accumulo"
    if strong:
        return "esplosione" if slope>0 else "correzione"
    return "accumulo"

# =========================
# STRATEGIA & SCORING
# =========================
def analyze_asset(asset):
    name = asset["name"]
    # --- timeframes
    tfs = ["15min","1h"]
    if name in M1_WHITELIST: tfs = ["1min"] + tfs
    tfs += ["1day"]

    frames = {}
    for tf in tfs:
        df = get_ohlc(name, tf)
        if df is None or len(df) < 60:
            return None  # dati insufficienti -> skip
        frames[tf] = df

    # indicatori chiave su M15, H1
    res = {"name": name, "calc":{}, "score":0, "flags":[]}

    def calc_on(df):
        out = {}
        close = df["close"]
        vol   = df["volume"].fillna(0)
        width, ma, up, lo = bollinger_width(close, n=20)
        dist, trend_up, trend_dn, e20, e50 = ema_distance(close, 20, 50)
        r = {
            "rsi": rsi(close, 14).iloc[-1],
            "bb_width": width.iloc[-1],
            "bb_pct": bb_compression_percentile(width, lookback=120),
            "ema20": e20.iloc[-1],
            "ema50": e50.iloc[-1],
            "ema_dist": dist.iloc[-1],
            "ema_dist_mean": dist.tail(30).mean(),
            "volume_surge": volume_surge(vol, 20).iloc[-1],
            "close": close.iloc[-1]
        }
        # breakout S/R
        sup, resi = simple_support_resistance(df)
        r["support"] = sup
        r["resistance"] = resi
        r["breakout_up"] = (not math.isnan(resi)) and (close.iloc[-1] > resi * 1.001)
        r["breakout_dn"] = (not math.isnan(sup))  and (close.iloc[-1] < sup  * 0.999)
        r["pattern"] = triangle_flag_heuristic(df)
        r["trend_up"] = trend_up.iloc[-1]
        r["trend_dn"] = trend_dn.iloc[-1]
        return r

    m15 = calc_on(frames["15min"])
    h1  = calc_on(frames["1h"])
    res["calc"]["M15"] = m15
    res["calc"]["H1"]  = h1

    # D1 per fase/alert speciale
    d1  = calc_on(frames["1day"])
    res["calc"]["D1"]  = d1
    # compressione daily multi-day
    d1_width_series, _, _, _ = bollinger_width(frames["1day"]["close"], 20)
    d1_pct = bb_compression_percentile(d1_width_series, lookback=200)
    res["calc"]["D1_pct"] = d1_pct

    # Direzione prevalente
    long_bias  = (m15["trend_up"] or h1["trend_up"]) and m15["rsi"]>50 and h1["rsi"]>50
    short_bias = (m15["trend_dn"] or h1["trend_dn"]) and m15["rsi"]<50 and h1["rsi"]<50
    direction = "LONG" if long_bias and not short_bias else "SHORT" if short_bias and not long_bias else "NEUTRO"
    res["direction"] = direction

    # Scoring (0-10)
    score = 0
    # 1) Compressione BB (più su M15)
    if m15["bb_pct"] < 30: score += 2; res["flags"].append("BB squeeze M15")
    if h1["bb_pct"]  < 30: score += 1
    # 2) EMA squeeze 20/50
    if m15["ema_dist"] < m15["ema_dist_mean"]*0.8: score += 1
    if h1["ema_dist"]  < h1["ema_dist_mean"]*0.8: score += 1
    # 3) RSI 50-cross / momentum
    if 48 <= m15["rsi"] <= 55 or 48 <= h1["rsi"] <= 55: score += 2; res["flags"].append("RSI 50-cross")
    # 4) Volume surge
    if m15["volume_surge"] >= 1.5 or h1["volume_surge"] >= 1.5: score += 2; res["flags"].append("Volumi ↑")
    # 5) Breakout S/R
    if m15["breakout_up"] or h1["breakout_up"] or m15["breakout_dn"] or h1["breakout_dn"]:
        score += 2; res["flags"].append("Breakout S/R")
    # 6) Pattern
    if m15["pattern"] or h1["pattern"]:
        score += 1; res["flags"].append(f"Pattern {m15['pattern'] or h1['pattern']}")

    score = min(10, score)
    res["score"] = score

    # Heatmap settimanale (fase ciclo)
    phase = weekly_phase_from_daily(frames["1day"])
    res["weekly_phase"] = phase

    # Exceptional alert
    exceptional = (score >= 10) and (d1_pct < 10)
    res["exceptional"] = exceptional

    return res

# =========================
# TELEGRAM
# =========================
def tg_send(html_text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram secrets mancanti, messaggio non inviato.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": html_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, data=data, timeout=20)
        if r.status_code != 200:
            print("[TG] Errore invio:", r.text)
    except Exception as e:
        print("[TG] Eccezione:", e)

def format_msg(result):
    name = result["name"]
    s = result["score"]
    dirn = result["direction"]
    stars = stars_from_score(s)
    m15 = result["calc"]["M15"]
    h1  = result["calc"]["H1"]
    d1  = result["calc"]["D1"]
    flags = " | ".join(result["flags"]) if result["flags"] else "—"

    label_dir = ""
    if dirn == "SHORT":
        label_dir = "\n<b>🧭 Direzione: SHORT (ribasso)</b>"

    hot = ""
    if result["weekly_phase"] == "esplosione" and s >= 7:
        hot = "\n🔥 <b>Titolo caldo del ciclo</b>"

    # Previsioni operative: percentuali (no parola "Probabilità")
    # semplice ripartizione coerente con direzione e segnali
    long_pct  = 50
    short_pct = 50
    if dirn == "LONG":
        long_pct = min(85, 55 + int(s*3))
        short_pct = 100 - long_pct
    elif dirn == "SHORT":
        short_pct = min(85, 55 + int(s*3))
        long_pct = 100 - short_pct

    # supporti/resistenze
    sup_m15 = m15["support"]; res_m15 = m15["resistance"]
    sup_h1  = h1["support"];  res_h1  = h1["resistance"]

    def fmt_level(x):
        return f"{x:.4f}" if (x is not None and not math.isnan(x)) else "—"

    header = (
        f"<b>{name} — Pre‑rally Scanner</b>\n"
        f"{stars}{hot}{label_dir}\n"
        f"Fase ciclo (sett): <b>{result['weekly_phase']}</b>\n"
        f"Score segnale: <b>{s}/10</b> | Trigger: <i>{flags}</i>\n"
    )

    # mini tabella timeframe
    tf_block = (
        "<b>Timeframe</b>\n"
        f"M15 — RSI {m15['rsi']:.1f} | BB%tile {m15['bb_pct']:.0f} | Vol x{m15['volume_surge']:.2f} | "
        f"S {fmt_level(sup_m15)} / R {fmt_level(res_m15)} | "
        f"BRK ⬆ { '✓' if m15['breakout_up'] else '–' } ⬇ { '✓' if m15['breakout_dn'] else '–' }\n"
        f"H1  — RSI {h1['rsi']:.1f}  | BB%tile {h1['bb_pct']:.0f}  | Vol x{h1['volume_surge']:.2f}  | "
        f"S {fmt_level(sup_h1)} / R {fmt_level(res_h1)}  | "
        f"BRK ⬆ { '✓' if h1['breakout_up'] else '–' } ⬇ { '✓' if h1['breakout_dn'] else '–' }\n"
        f"D1  — BB%tile {result['calc']['D1_pct']:.0f} | EMA20 vs 50: "
        f"{'↑' if d1['trend_up'] else ('↓' if d1['trend_dn'] else '→')}\n"
    )

    # Previsione operativa
    prev = (
        "<b>Previsione operativa</b>\n"
        f"{long_pct}% — Scenario LONG: breakout/ritest sopra R H1, RSI>52 su M15, "
        f"volumi ≥1.5x; gestione su pullback EMA20 M15.\n"
        f"{short_pct}% — Scenario SHORT: perdita S H1 e chiusure sotto EMA20/50, "
        f"RSI<48 su M15, volumi ≥1.5x; pullback falliti verso R per continuazione.\n"
    )

    consiglio = (
        "<i>Consiglio personale:</i> attendi conferma su M15 (candela chiara + volume) "
        "e verifica allineamento H1. Evita entrate in mezzo al range; meglio "
        "break+retest o pullback su EMA20 M15. Stop sempre oltre S/R.\n"
    )

    exceptional = ""
    if result["exceptional"]:
        exceptional = "\n<b>⚠️ Segnale eccezionale</b>: Score 10/10 + compressione D1 multi‑day. " \
                      "Contesto favorevole a espansione di volatilità."

    return header + "\n" + tf_block + "\n" + prev + "\n" + consiglio + exceptional

# =========================
# LOOP PROGRAMMATO
# =========================
_last_run_key = None

def should_run_now(ts):
    # Allinea ai multipli di 15 min
    return ts.minute % CHECK_EVERY_MIN == 0

def runner_loop():
    global _last_run_key
    print("[INIT] Runner avviato.")
    while True:
        try:
            ts = now_rome()
            key = ts.strftime("%Y-%m-%d %H:%M")
            if within_active_hours(ts) and should_run_now(ts) and key != _last_run_key:
                _last_run_key = key
                print(f"[RUN] {key}")
                run_scan_cycle()
            time.sleep(5)  # controllo ogni 5s per beccare il minuto giusto
        except Exception:
            print("[ERR] Loop:", traceback.format_exc())
            time.sleep(5)

def run_scan_cycle():
    for asset in ASSETS:
        try:
            res = analyze_asset(asset)
            if not res: 
                print(f"[SKIP] Dati insufficienti per {asset['name']}")
                continue
            if res["score"] >= MIN_SCORE_TO_ALERT:
                msg = format_msg(res)
                tg_send(msg)
                print(f"[ALERT] {asset['name']} score={res['score']} dir={res['direction']}")
            else:
                print(f"[NOALERT] {asset['name']} score={res['score']}")
            # piccola pausa per non stressare provider fallback
            time.sleep(0.3)
        except Exception:
            print(f"[ERR] analyze {asset['name']}:", traceback.format_exc())
            time.sleep(0.5)

# =========================
# WEB SERVER (Render)
# =========================
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok": True, "time": now_rome().isoformat()})

@app.get("/")
def root():
    return "Bot pre‑rally attivo. Usa /health per check."

def main():
    # Avvia runner in background
    t = threading.Thread(target=runner_loop, daemon=True)
    t.start()
    # Flask web (Render usa PORT)
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
