import os, time, logging, requests
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("finnhub-bot")

FINNHUB_API_KEY  = os.getenv("FINNHUB_API_KEY", "").strip()
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SYMBOLS = ["AAPL", "TSLA", "GOOGL", "PLTR"]  # Modificabile

app = FastAPI()
_sched = None

def tg(msg: str):
    try:
        url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r=requests.post(url,json={"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"HTML"},timeout=15)
        if r.status_code!=200: log.warning("Telegram %s %s", r.status_code, r.text)
    except Exception as e:
        log.warning("Telegram ex: %s", e)

def finnhub_quote(symbol: str):
    url = "https://finnhub.io/api/v1/quote"
    r = requests.get(url, params={"symbol":symbol,"token":FINNHUB_API_KEY}, timeout=15)
    r.raise_for_status()
    return r.json()

def scan_once():
    for sym in SYMBOLS:
        try:
            q = finnhub_quote(sym)
            c, pc = q.get("c"), q.get("pc")
            if not c or not pc: 
                log.info("%s: dati incompleti %s", sym, q); 
                continue

            change = (c - pc) / pc * 100.0
            log.info("%s price=%.2f (%.2f%%)", sym, c, change)

            if abs(change) >= 2.0:
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                msg = f"<b>[FINNHUB]</b> {sym} {change:+.2f}%  prezzo: <b>{c:.2f}</b>\n{ts}"
                tg(msg)
        except Exception as e:
            log.warning("Errore %s: %s", sym, e)

@app.on_event("startup")
def startup():
    global _sched
    log.info("Avvio finnhub-bot…")
    _sched = BackgroundScheduler(timezone="UTC")
    _sched.add_job(scan_once, CronTrigger(minute="*/5"))
    _sched.start()
    scan_once()

@app.get("/")
@app.get("/health")
def health():
    return {"status":"ok","time":datetime.utcnow().isoformat()}

