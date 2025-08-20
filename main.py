# Script principale aggiornato – GigaBot v1.0
# Pre-rally detection multi-timeframe + visual Telegram alerts

import os, time, logging, requests
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pre-rally-bot")

FINNHUB_API_KEY  = os.getenv("FINNHUB_API_KEY", "")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TWELVEDATA_API   = os.getenv("TWELVEDATA_API", "")

SYMBOLS = [
    "PLTR", "GOOGL", "TSLA", "AAPL", "IFX.DE", "REY.MI", "MU", "AMD",
    "FCT.MI", "XOM", "VLO", "GM", "MC.PA", "KO", "DIS",
    "EUR/USD", "USD/JPY", "GBP/USD", "ETH/USD", "BTC/USD"
]

ITALY_TZ_OFFSET = 2  # UTC+2

app = FastAPI()
_sched = None

def tg(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=15)
        if r.status_code != 200:
            log.warning("Telegram %s %s", r.status_code, r.text)
    except Exception as e:
        log.warning("Telegram ex: %s", e)

def get_quote(symbol: str):
    try:
        r = requests.get("https://finnhub.io/api/v1/quote", params={"symbol": symbol, "token": FINNHUB_API_KEY}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return {}

def is_active_hour():
    now = datetime.utcnow().hour + ITALY_TZ_OFFSET
    return 8 <= now < 23

def format_signal(sym, data, is_hot=False, score=3, indicators=None):
    now = datetime.utcnow().strftime("%H:%M")
    stars = "⭐️" * score + "☆" * (5 - score)
    tf_signals = "\n".join([
        f"• M15: {'✅ ' + ', '.join(indicators.get('M15', [])) if indicators.get('M15') else '—'}",
        f"• H1 : {'✅ ' + ', '.join(indicators.get('H1', [])) if indicators.get('H1') else '—'}",
        f"• D1 : {'✅ ' + ', '.join(indicators.get('D1', [])) if indicators.get('D1') else '—'}"
    ])

    comment = (
        "💬 <b>Commento:</b>\n"
        "Possibile inizio pre-rally. Configurazione interessante su più timeframe.\n"
        "Conferma sopra i livelli chiave potrebbe generare un'accelerazione."
    )

    msg = f"{'🔥 Titolo caldo del ciclo\n' if is_hot else ''}" \
          f"<b>🟢 {sym}</b> {data.get('c', '—')} ({data.get('dp', '')}%) | {now}\n\n" \
          f"<b>📊 Segnali attivi:</b>\n{tf_signals}\n\n" \
          f"{comment}\n\n" \
          f"<b>📈 Score segnale:</b> {stars}"

    return msg

def fake_indicators():  # simulazione per test
    return {
        "M15": ["EMA", "RSI"],
        "H1": ["Volumi", "Compressione"],
        "D1": []
    }

def scan():
    if not is_active_hour():
        log.info("🕒 Fuori orario. Nessuna scansione.")
        return

    best = None
    for sym in SYMBOLS:
        q = get_quote(sym)
        if not q or not q.get("c") or not q.get("pc"):
            continue

        change = (q["c"] - q["pc"]) / q["pc"] * 100
        q["dp"] = f"{change:+.2f}"

        indicators = fake_indicators()  # da sostituire con veri calcoli
        score = len(indicators.get("M15", [])) + len(indicators.get("H1", []))
        is_hot = score >= 4 and not best
        if is_hot: best = sym

        msg = format_signal(sym, q, is_hot=is_hot, score=min(score, 5), indicators=indicators)
        tg(msg)
        time.sleep(1.5)

@app.on_event("startup")
def startup():
    global _sched
    log.info("⏳ Avvio pre-rally-bot…")
    _sched = BackgroundScheduler(timezone="UTC")
    _sched.add_job(scan, CronTrigger(minute="*/15"))
    _sched.start()
    scan()

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

