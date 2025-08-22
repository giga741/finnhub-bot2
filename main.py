""" Bot segnali pre‑rally — Finnhub + TwelveData Compatibile con Railway / Render / Koyeb.

✅ Sicurezza integrata per deploy senza rischi:

SAFE_MODE: nessun messaggio Telegram inviato, solo log.

DRY_RUN: calcola i segnali ma non li notifica (override di SAFE_MODE).

ROLLBACK: basta rimettere il vecchio main.py. Questo file è self‑contained.


Requisiti (pip): httpx, fastapi, uvicorn (opzionali ma consigliati): python‑dateutil

Environment variables richieste:

FINNHUB_API_KEY

TWELVEDATA_API_KEY

TELEGRAM_TOKEN

TELEGRAM_CHAT_ID


Opzionali (con default):

TZ=Europe/Rome

ACTIVE_HOURS_START=08:45

ACTIVE_HOURS_END=23:15

SCORE_MIN=3

ENABLE_SHORT_ONLY=true

SAFE_MODE=true

DRY_RUN=false

POLL_SECONDS=60


Note timeframe:

H1 e M15 su TUTTI gli asset.

M1 solo Forex+Crypto (EURUSD, USDJPY, GBPUSD, ETHUSD, BTCUSD).


Formato visivo (mix #1 + #4) e migliorie approvate già incluse:

score a stelle

Direzione: SHORT

Supporto/Resistenza auto (semplice)

Riconoscimento pattern base (triangolo/flag) — euristica leggera

Notifica speciale per segnali eccezionali (score 10/10 + compressione multi‑day D1) """ import os import asyncio import math import json from datetime import datetime, time, timedelta, timezone from typing import Dict, List, Tuple, Optional


import httpx from fastapi import FastAPI from fastapi.responses import JSONResponse, PlainTextResponse

-------------------------

Config e utility orario

-------------------------

TZ = os.getenv("TZ", "Europe/Rome") try: import zoneinfo  # py>=3.9 TZINFO = zoneinfo.ZoneInfo(TZ) except Exception: TZINFO = timezone(timedelta(hours=2))  # fallback CEST approx

ACTIVE_HOURS_START = os.getenv("ACTIVE_HOURS_START", "08:45") ACTIVE_HOURS_END = os.getenv("ACTIVE_HOURS_END", "23:15") POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))

SAFE_MODE = os.getenv("SAFE_MODE", "true").lower() == "true" DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true" SCORE_MIN = int(os.getenv("SCORE_MIN", "3")) ENABLE_SHORT_ONLY = os.getenv("ENABLE_SHORT_ONLY", "true").lower() == "true"

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "") TWELVE_KEY = os.getenv("TWELVEDATA_API_KEY", "") TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "") TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

-------------------------

Asset list & simboli

-------------------------

ASSETS = [ # Tech "PLTR", "GOOGL", "TSLA", "AAPL", "MU", "AMD", # Industrials / Energy "FCT.MI", "XOM", "VLO", "GM", # Consumer / Lusso "MC.PA", "KO", "DIS", # UE focus aggiuntivi "IFX.DE", "REY.MI", # Forex "EURUSD", "USDJPY", "GBPUSD", # Crypto "ETHUSD", "BTCUSD", ]

Mappatura simboli per provider

SYMBOL_MAP = { # USA & generici: uguali "PLTR": {"finnhub": "PLTR", "twelvedata": "PLTR"}, "GOOGL": {"finnhub": "GOOGL", "twelvedata": "GOOGL"}, "TSLA": {"finnhub": "TSLA", "twelvedata": "TSLA"}, "AAPL": {"finnhub": "AAPL", "twelvedata": "AAPL"}, "MU": {"finnhub": "MU", "twelvedata": "MU"}, "AMD": {"finnhub": "AMD", "twelvedata": "AMD"}, "XOM": {"finnhub": "XOM", "twelvedata": "XOM"}, "VLO": {"finnhub": "VLO", "twelvedata": "VLO"}, "GM": {"finnhub": "GM", "twelvedata": "GM"}, "MC.PA": {"finnhub": "MC.PA", "twelvedata": "MC:PA"}, "KO": {"finnhub": "KO", "twelvedata": "KO"}, "DIS": {"finnhub": "DIS", "twelvedata": "DIS"}, # UE specifici "FCT.MI": {"finnhub": "FCT.MI", "twelvedata": "FCT:MI"}, "IFX.DE": {"finnhub": "IFX.DE", "twelvedata": "IFX:XETRA"}, "REY.MI": {"finnhub": "REY.MI", "twelvedata": "REY:MI"}, # Forex "EURUSD": {"finnhub": "OANDA:EUR_USD", "twelvedata": "EUR/USD"}, "USDJPY": {"finnhub": "OANDA:USD_JPY", "twelvedata": "USD/JPY"}, "GBPUSD": {"finnhub": "OANDA:GBP_USD", "twelvedata": "GBP/USD"}, # Crypto (spot) "ETHUSD": {"finnhub": "BINANCE:ETHUSDT", "twelvedata": "ETH/USD"}, "BTCUSD": {"finnhub": "BINANCE:BTCUSDT", "twelvedata": "BTC/USD"}, }

Quali asset hanno M1

INTRADAY_M1 = {"EURUSD", "USDJPY", "GBPUSD", "ETHUSD", "BTCUSD"}

TIMEFRAMES = ["1h", "15min"]  # per tutti

M1 dinamico in runtime

Stato runtime per health

LAST_SIGNALS: List[dict] = [] LAST_ERROR: Optional[str] = None STARTED_AT = datetime.now(TZINFO).isoformat()

-------------------------

Indicatori leggeri (no pandas)

-------------------------

def ema(series: List[float], period: int) -> List[float]: if not series or period <= 1: return series[:] k = 2 / (period + 1) out = [] ema_prev = series[0] for price in series: ema_prev = price * k + ema_prev * (1 - k) out.append(ema_prev) return out

def rsi(prices: List[float], period: int = 14) -> List[float]: if len(prices) < period + 1: return [50.0] * len(prices) gains = [0.0] losses = [0.0] for i in range(1, len(prices)): change = prices[i] - prices[i - 1] gains.append(max(change, 0)) losses.append(abs(min(change, 0))) avg_gain = sum(gains[1:period + 1]) / period avg_loss = sum(losses[1:period + 1]) / period rsis = [50.0] * len(prices) for i in range(period + 1, len(prices)): avg_gain = (avg_gain * (period - 1) + gains[i]) / period avg_loss = (avg_loss * (period - 1) + losses[i]) / period if avg_loss == 0: rs = float('inf') rsis[i] = 100.0 else: rs = avg_gain / avg_loss rsis[i] = 100 - (100 / (1 + rs)) return rsis

def true_range(h: List[float], l: List[float], c: List[float]) -> List[float]: tr = [h[0] - l[0]] for i in range(1, len(c)): tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))) return tr

def atr(h: List[float], l: List[float], c: List[float], period: int = 14) -> List[float]: tr = true_range(h, l, c) out = [] if not tr: return out smoothed = sum(tr[:period]) / max(period, 1) out = [smoothed] for i in range(period, len(tr)): smoothed = (smoothed * (period - 1) + tr[i]) / period out.append(smoothed) # align length while len(out) < len(c): out.insert(0, out[0]) return out

def detect_compression(c: List[float], period: int = 20) -> float: if len(c) < period: return 0.0 window = c[-period:] rng = max(window) - min(window) if rng == 0: return 1.0 # normalizza per prezzo return (sum(abs(window[i] - window[i - 1]) for i in range(1, len(window))) / (rng * (period - 1)))

def support_resistance(c: List[float], lookback: int = 30) -> Tuple[Optional[float], Optional[float]]: if len(c) < lookback: return None, None w = c[-lookback:] return min(w), max(w)

def detect_pattern_basic(c: List[float]) -> Optional[str]: if len(c) < 20: return None last = c[-20:] # Triangolo/flag euristico: massimi decrescenti + minimi crescenti (range che si stringe) highslope = (max(last[:10]) - max(last[10:])) lowslope = (min(last[10:]) - min(last[:10])) if highslope > 0 and lowslope > 0: return "Compressione a triangolo" # Flag grezza: trend ribasso recente + canale stretto laterale if last[-1] < last[0] and (max(last) - min(last)) / last[-1] < 0.03: return "Flag ribassista" return None

-------------------------

Data fetchers

-------------------------

async def fetch_finnhub_ohlc(symbol: str, interval: str, client: httpx.AsyncClient) -> Dict: base = "https://finnhub.io/api/v1/stock/candle" # Map intervalli res_map = {"1h": "60", "15min": "15", "1min": "1", "1day": "D"} resolution = res_map.get(interval, "60") now = int(datetime.now(tz=timezone.utc).timestamp()) fro = now - 60 * 60 * 24 * 60  # 60 giorni params = { "symbol": symbol, "resolution": resolution, "from": fro, "to": now, "token": FINNHUB_KEY, } r = await client.get(base, params=params, timeout=30) r.raise_for_status() data = r.json() # Expected keys: c,h,l,o,t,s return data

async def fetch_twelvedata_ohlc(symbol: str, interval: str, client: httpx.AsyncClient) -> Dict: base = "https://api.twelvedata.com/time_series" # TD intervals: 1h, 15min, 1min, 1day params = { "symbol": symbol, "interval": interval, "apikey": TWELVE_KEY, "outputsize": 500, "format": "JSON", "order": "ASC", } r = await client.get(base, params=params, timeout=30) r.raise_for_status() return r.json()

def parse_twelvedata_series(js: Dict) -> Tuple[List[float], List[float], List[float], List[float]]: values = js.get("values") or js.get("data") or [] closes, highs, lows, opens = [], [], [], [] for row in values: try: closes.append(float(row["close"])) highs.append(float(row["high"])) lows.append(float(row["low"])) opens.append(float(row["open"])) except Exception: continue return closes, highs, lows, opens

def parse_finnhub_series(js: Dict) -> Tuple[List[float], List[float], List[float], List[float]]: if js.get("s") != "ok": return [], [], [], [] c = js.get("c", []) h = js.get("h", []) l = js.get("l", []) o = js.get("o", []) return list(map(float, c)), list(map(float, h)), list(map(float, l)), list(map(float, o))

-------------------------

Segnali pre‑rally (versione light)

-------------------------

def compute_signal(c: List[float], h: List[float], l: List[float], tf: str) -> Tuple[int, Dict[str, float]]: """Restituisce (score, metrics). Score 0-10""" if len(c) < 50: return 0, {}

ema_fast = ema(c, 9)
ema_slow = ema(c, 21)
r = rsi(c, 14)
a = atr(h, l, c, 14)

# Compressione (più basso = più compresso). Convertiamo a punteggio 0-3
comp = detect_compression(c, 20)
comp_score = 3 - min(3.0, max(0.0, comp * 3))

# Momentum (EMA9 sotto EMA21 per SHORT)
mom_score = 0
if ENABLE_SHORT_ONLY:
    mom_score = 3 if ema_fast[-1] < ema_slow[-1] else 0
else:
    mom_score = 3 if abs(ema_fast[-1] - ema_slow[-1]) / (c[-1] + 1e-9) > 0.002 else 1

# RSI zona (debolezza per SHORT)
rsi_val = r[-1]
rsi_score = 2 if rsi_val < 45 else (1 if rsi_val < 50 else 0)

# Volatilità moderata (evitare ATR troppo alto)
atr_rel = a[-1] / (c[-1] + 1e-9)
vol_score = 2 if atr_rel < 0.02 else (1 if atr_rel < 0.03 else 0)

score = int(comp_score + mom_score + rsi_score + vol_score)
metrics = {
    "comp": float(comp),
    "ema_spread": float(ema_fast[-1] - ema_slow[-1]),
    "rsi": float(rsi_val),
    "atr_rel": float(atr_rel),
}
return min(score, 10), metrics

def exceptional_signal(c: List[float]) -> bool: # compress multi‑day euristica: ult. 5 giorni range molto stretto if len(c) < 200: return False last = c[-120:] rng = max(last) - min(last) if rng <= 0: return False wiggle = sum(abs(last[i] - last[i - 1]) for i in range(1, len(last))) / (rng * (len(last) - 1)) return wiggle < 0.2

-------------------------

Telegram

-------------------------

async def tg_send(text: str, client: httpx.AsyncClient) -> None: if SAFE_MODE or DRY_RUN: print("[SAFE/DRY] Telegram message suppressed:\n" + text) return if not TG_TOKEN or not TG_CHAT: print("[WARN] TELEGRAM env missing; skipping send.") return url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage" payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True} r = await client.post(url, json=payload, timeout=20) try: r.raise_for_status() except Exception as e: print("[TG ERROR]", e, r.text)

def stars(score: int) -> str: full = "★" * max(0, min(10, score)) empty = "☆" * (10 - len(full)) return full + empty

def fmt_signal_msg(symbol: str, tf: str, score: int, metrics: Dict[str, float], pattern: Optional[str], supp: Optional[float], resi: Optional[float], direction: str = "SHORT", hot: bool = False) -> str: header = f"<b>{symbol}</b>\n🟡🟡🟡 Semaforo segnale\n{stars(score)}" if hot: header = "🔥 <b>Titolo caldo del ciclo</b>\n" + header lines = [ header, f"\n<b>🧭 Direzione:</b> {direction}", f"<b>Timeframe:</b> {tf}", f"<b>Score:</b> {score}/10", f"<b>RSI:</b> {metrics.get('rsi', 0):.1f}", f"<b>ATR%:</b> {metrics.get('atr_rel', 0)*100:.2f}%", f"<b>Compressione:</b> {metrics.get('comp', 0):.3f}", ] if supp is not None and resi is not None: lines.append(f"<b>Supporto:</b> {supp:.4f}  |  <b>Resistenza:</b> {resi:.4f}") if pattern: lines.append(f"<b>Pattern:</b> {pattern}") lines.append("\n<i>Consiglio personale:</i> attendi conferma con close sotto supporto e volumi in aumento su M15. Evita ingressi centrali, preferisci pullback al livello rotto.") return "\n".join(lines)

-------------------------

Core loop

-------------------------

def within_active_hours(now_dt: datetime) -> bool: try: sh, sm = map(int, ACTIVE_HOURS_START.split(":")) eh, em = map(int, ACTIVE_HOURS_END.split(":")) start_t = time(sh, sm) end_t = time(eh, em) t = now_dt.timetz() return (t >= start_t and t <= end_t) except Exception: return True

async def process_symbol(symbol_key: str, client: httpx.AsyncClient) -> List[dict]: out_signals = [] maps = SYMBOL_MAP.get(symbol_key) if not maps: return out_signals

# Timeframes per asset
tfs = TIMEFRAMES[:]
if symbol_key in INTRADAY_M1:
    tfs = tfs + ["1min"]

for tf in tfs:
    # 1) TwelveData (primario per coerenza)
    td_symbol = maps["twelvedata"]
    try:
        td = await fetch_twelvedata_ohlc(td_symbol, tf, client)
        c, h, l, o = parse_twelvedata_series(td)
    except Exception:
        c, h, l, o = [], [], [], []
    # 2) Finnhub fallback
    if len(c) < 50:
        fh_symbol = maps["finnhub"]
        try:
            fh = await fetch_finnhub_ohlc(fh_symbol, tf, client)
            c, h, l, o = parse_finnhub_series(fh)
        except Exception:
            c, h, l, o = [], [], [], []

    if len(c) < 50:
        continue

    score, metrics = compute_signal(c, h, l, tf)
    if score < SCORE_MIN:
        continue

    s, r = support_resistance(c)
    pattern = detect_pattern_basic(c)
    hot = score >= 10 and exceptional_signal(c)

    sig = {
        "symbol": symbol_key,
        "tf": tf,
        "score": score,
        "metrics": metrics,
        "support": s,
        "resistance": r,
        "pattern": pattern,
        "hot": hot,
    }
    out_signals.append(sig)

    # Notifica per singolo timeframe
    msg = fmt_signal_msg(symbol_key, tf, score, metrics, pattern, s, r, direction="SHORT" if ENABLE_SHORT_ONLY else "MISTA", hot=hot)
    await tg_send(msg, client)

return out_signals

async def main_loop(): global LAST_SIGNALS, LAST_ERROR async with httpx.AsyncClient() as client: while True: try: now = datetime.now(TZINFO) if not within_active_hours(now): await asyncio.sleep(POLL_SECONDS) continue

batch_signals: List[dict] = []
            for sym in ASSETS:
                try:
                    sigs = await process_symbol(sym, client)
                    batch_signals.extend(sigs)
                except Exception as e:
                    print(f"[ERR] {sym}: {e}")

            # Ordinamento per score desc
            batch_signals.sort(key=lambda x: x["score"], reverse=True)
            LAST_SIGNALS = batch_signals[-50:]
            LAST_ERROR = None
        except Exception as e:
            LAST_ERROR = str(e)
            print("[LOOP ERROR]", e)
        await asyncio.sleep(POLL_SECONDS)

-------------------------

FastAPI app + health

-------------------------

app = FastAPI()

@app.get("/") def root(): return {"service": "pre‑rally‑bot", "started_at": STARTED_AT, "safe_mode": SAFE_MODE, "score_min": SCORE_MIN}

@app.get("/health") def health(): ok = LAST_ERROR is None return JSONResponse({"ok": ok, "error": LAST_ERROR, "signals_cached": len(LAST_SIGNALS)})

@app.get("/last_signals") def last_signals(): return JSONResponse(LAST_SIGNALS)

Background task on startup

@app.on_event("startup") async def startup_event(): asyncio.create_task(main_loop())

Entrypoint per Railway/Render

if name == "main": import uvicorn port = int(os.getenv("PORT", "8080")) uvicorn.run(app, host="0.0.0.0", port=port)

 
