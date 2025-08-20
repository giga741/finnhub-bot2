# ✅ Codice main.py completo e ottimizzato per Finnhub + TwelveData
# Include: 20 asset, strategia pre-rally, multi-timeframe, Telegram, fascia oraria attiva

import os, requests, logging
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

FINNHUB_API_KEY      = os.getenv("FINNHUB_API_KEY", "").strip()
TWELVEDATA_API_KEY   = os.getenv("TWELVEDATA_API_KEY", "").strip()
TELEGRAM_TOKEN       = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SYMBOLS = [
    "PLTR", "GOOGL", "TSLA", "AAPL", "IFX.DE", "REY.MI", "MU", "AMD",
    "FCT.MI", "XOM", "VLO", "GM",
    "MC.PA", "KO", "DIS",
    "EUR/USD", "USD/JPY", "GBP/USD",
    "ETH/USD", "BTC/USD"
]

TIMEFRAMES = ["1min", "15min", "1h", "1day"]  # usati per analisi con TwelveData
FASCIA_ORARIA = (8, 45, 23, 0)  # dalle 08:45 alle 23:00 (ora italiana)

app = FastAPI()
scheduler = BackgroundScheduler(timezone="UTC")

# Telegram

def tg(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=15)
        if r.status_code != 200:
            log.warning("Telegram %s %s", r.status_code, r.text)
    except Exception as e:
        log.warning("Telegram ex: %s", e)

# TwelveData

def get_ohlcv(symbol: str, interval: str):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 30,
        "apikey": TWELVEDATA_API_KEY
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return data.get("values", []) if "values" in data else []

def calc_indicators(candles):
    if len(candles) < 10:
        return {}
    closes = [float(c['close']) for c in reversed(candles)]
    rsi = 100 - (100 / (1 + (sum(closes[-5:])/5) / (sum(closes[-10:-5])/5)))  # Semplificato
    ema8 = sum(closes[-8:]) / 8
    ema21 = sum(closes[-21:]) / 21 if len(closes) >= 21 else ema8
    compression = max(closes[-5:]) - min(closes[-5:]) < 0.5 * (max(closes) - min(closes))
    return {"rsi": rsi, "ema8": ema8, "ema21": ema21, "compression": compression}

# Finnhub live quote

def get_price(symbol: str):
    if "/" in symbol:
        fx = symbol.replace("/", "")
        return get_finnhub(fx)
    url = "https://finnhub.io/api/v1/quote"
    r = requests.get(url, params={"symbol": symbol, "token": FINNHUB_API_KEY}, timeout=10)
    q = r.json()
    return q.get("c"), q.get("pc")

def get_finnhub(fx):
    url = f"https://finnhub.io/api/v1/quote"
    r = requests.get(url, params={"symbol": fx, "token": FINNHUB_API_KEY}, timeout=10)
    q = r.json()
    return q.get("c"), q.get("pc")

def scan():
    now = datetime.utcnow() + timedelta(hours=2)
    if not (FASCIA_ORARIA[0] <= now.hour <= FASCIA_ORARIA[2] and (now.hour != FASCIA_ORARIA[2] or now.minute <= FASCIA_ORARIA[3])):
        log.info("Fuori orario attivo")
        return

    for sym in SYMBOLS:
        try:
            price, prev = get_price(sym)
            if not price or not prev:
                continue
            change = (price - prev) / prev * 100
            if abs(change) < 1.5:
                continue

            info = []
            score = 0
            for tf in TIMEFRAMES:
                candles = get_ohlcv(sym, tf)
                ind = calc_indicators(candles)
                if not ind:
                    continue
                if ind["rsi"] < 35 or ind["rsi"] > 65:
                    score += 1; info.append(f"RSI={ind['rsi']:.1f}")
                if ind["ema8"] > ind["ema21"]:
                    score += 1; info.append("EMA incrocio")
                if ind["compression"]:
                    score += 1; info.append("Compressione")

            if score >= 2:
                ts = now.strftime("%H:%M")
                emoji = "🟢" if score >= 3 else "🟡"
                msg = f"{emoji} <b>{sym}</b> {change:+.2f}% <b>{price:.2f}</b> alle {ts}\n" + " • ".join(info)
                tg(msg)
        except Exception as e:
            log.warning("Errore %s: %s", sym, e)

@app.on_event("startup")
def startup():
    log.info("Avvio finnhub-bot finale…")
    scheduler.add_job(scan, CronTrigger(minute="*/5"))
    scheduler.start()
    scan()

@app.get("/")
def root():
    return {"status": "ok", "time": datetime.utcnow().isoformat()} 

