import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from dateutil import tz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

# =========================
# Config & Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("capital-bot")

# ⚠️ Render non ha Europe/Rome: usiamo UTC
APP_TZ = tz.gettz("UTC")

CAPITAL_BASE_URL = os.getenv("CAPITAL_BASE_URL", "https://api-capital.backend-capital.com")
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY")
CAPITAL_IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")  # email di login (LIVE) o clientId: usiamo email
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD")      # password della API key

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_PASSWORD, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID]):
    logger.warning("Mancano una o più variabili d'ambiente richieste. Controlla ENV su Render.")

# =========================
# Asset e termini di ricerca (multi-term)
# =========================
ASSET_SEARCH_TERMS: Dict[str, List[str]] = {
    "PLTR":       ["PLTR", "PALANTIR", "PALANTIR TECHNOLOGIES"],
    "GOOGL":      ["GOOGL", "ALPHABET", "ALPHABET INC", "GOOGLE"],
    "ETH/USD":    ["ETH/USD", "ETHUSD", "ETHEREUM"],
    "USD/JPY":    ["USD/JPY", "USDJPY"],
    "LDO.MI":     ["LDO.MI", "LEONARDO", "LEONARDO S.P.A"],
    "Fincantieri":["FINCANTIERI"],
    "Infineon":   ["INFINEON", "INFINEON TECHNOLOGIES"],
    "TSLA":       ["TSLA", "TESLA"],
    "EUR/USD":    ["EUR/USD", "EURUSD"],
    "GBP/USD":    ["GBP/USD", "GBPUSD"],
}

# Timeframe
TF_RES_MAP = {
    "M1": "MINUTE",
    "M15": "MINUTE_15",
    "H1": "HOUR",
    "D1": "DAY",
}
M1_ASSETS = {"USD/JPY", "EUR/USD", "GBP/USD", "ETH/USD"}

# =========================
# Capital.com REST client
# =========================
class CapitalAPI:
    def __init__(self, base_url: str, api_key: str, identifier: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        self.session = requests.Session()
        self.cst: Optional[str] = None
        self.sec_token: Optional[str] = None
        self.last_touch: float = 0.0

    def _auth_headers(self) -> Dict[str, str]:
        headers = {"X-CAP-API-KEY": self.api_key}
        if self.cst and self.sec_token:
            headers.update({"CST": self.cst, "X-SECURITY-TOKEN": self.sec_token})
        return headers

    def login(self) -> None:
        url = f"{self.base_url}/api/v1/session"
        payload = {"identifier": self.identifier, "password": self.password, "encryptedPassword": False}
        headers = {"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"}
        r = self.session.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Login failed: {r.status_code} {r.text}")
        self.cst = r.headers.get("CST")
        self.sec_token = r.headers.get("X-SECURITY-TOKEN")
        if not self.cst or not self.sec_token:
            raise RuntimeError("Login ok ma CST/X-SECURITY-TOKEN mancanti")
        self.last_touch = time.time()
        logger.info("[CapitalAPI] Sessione avviata (auto‑refresh abilitato)")

    def _ensure_session(self) -> None:
        if not self.cst or not self.sec_token or (time.time() - self.last_touch) > 540:
            self.login()

    def _get(self, path: str, params: Dict = None) -> requests.Response:
        self._ensure_session()
        url = f"{self.base_url}{path}"
        r = self.session.get(url, headers=self._auth_headers(), params=params or {}, timeout=30)
        if r.status_code == 401:
            logger.warning("[CapitalAPI] 401 ricevuto, re‑login…")
            self.login()
            r = self.session.get(url, headers=self._auth_headers(), params=params or {}, timeout=30)
        self.last_touch = time.time()
        return r

    def search_markets(self, term: str) -> List[Dict]:
        r = self._get("/api/v1/markets", params={"searchTerm": term})
        if r.status_code != 200:
            logger.warning(f"Markets search failed for {term}: {r.status_code} {r.text}")
            return []
        return r.json().get("markets", [])

    def get_prices(self, epic: str, resolution: str, max_points: int = 300) -> Dict:
        r = self._get(f"/api/v1/prices/{epic}", params={"resolution": resolution, "max": max_points})
        if r.status_code != 200:
            raise RuntimeError(f"Prices failed for {epic}: {r.status_code} {r.text}")
        return r.json()

# =========================
# Utility
# =========================
def _sanitize(s: str) -> str:
    # Rimuove caratteri non alfanumerici (USD/JPY -> USDJPY)
    return "".join(ch for ch in s if ch.isalnum())

def select_best_market(markets: List[Dict], wanted_label: str) -> Optional[Dict]:
    if not markets:
        return None
    ranked = []
    w = wanted_label.upper()
    for m in markets:
        instr = m.get("instrument", {})
        name = str(instr.get("name", "")).upper()
        symbol = str(instr.get("symbol", "")).upper()
        epic = instr.get("epic")
        product = m.get("product", "")
        status = m.get("snapshot", {}).get("marketStatus")
        score = 0
        if any(k in name for k in [w, _sanitize(w)]) or any(k in symbol for k in [w, _sanitize(w)]):
            score += 3
        if product == "CFD":
            score += 2
        if status == "TRADEABLE":
            score += 1
        if epic:
            ranked.append((score, m))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1] if ranked else None

def build_epic_map(api: CapitalAPI, asset_terms: Dict[str, List[str]]) -> Dict[str, str]:
    mapping = {}
    for label, terms in asset_terms.items():
        epic = None
        for term in terms + [_sanitize(t) for t in terms]:
            try:
                mkts = api.search_markets(term)
                if mkts:
                    best = select_best_market(mkts, label)
                    if not best:
                        # fallback: prendi semplicemente il primo tradeable
                        tradeables = [m for m in mkts if m.get("snapshot", {}).get("marketStatus") == "TRADEABLE"]
                        best = tradeables[0] if tradeables else mkts[0]
                    epic = best["instrument"]["epic"]
                    logger.info(f"Mappato {label} ({term}) → {epic}")
                    break
            except Exception as e:
                logger.warning(f"Search errore per {label}/{term}: {e}")
        if not epic:
            logger.warning(f"Nessun epic trovato per {label}")
        else:
            mapping[label] = epic
    return mapping

# =========================
# Indicatori tecnici
# =========================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(span=period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(span=period, adjust=False).mean()

def bollinger(series: pd.Series, period: int = 20, ndev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(window=period, min_periods=period).mean()
    sd = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = ma + ndev * sd
    lower = ma - ndev * sd
    return lower, ma, upper

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# =========================
# Strategia pre‑rally
# =========================
def detect_pre_rally(df: pd.DataFrame) -> Optional[Dict]:
    if df is None or len(df) < 100:
        return None
    close = df["close"]; high = df["high"]; low = df["low"]

    ema20 = ema(close, 20); ema50 = ema(close, 50)
    rsi14 = rsi(close, 14)
    atr14 = atr(high, low, close, 14); atr_ma = atr14.rolling(14).mean()

    lower, mid, upper = bollinger(close, 20, 2.0)
    bbw = (upper - lower) / (mid + 1e-9)
    bbw_p20 = bbw.rolling(200).quantile(0.20)

    _, _, hist = macd(close)

    comp_ok = (bbw.iloc[-1] <= bbw_p20.iloc[-1]) and (atr14.iloc[-1] <= 0.8 * (atr_ma.iloc[-1] if not np.isnan(atr_ma.iloc[-1]) else atr14.iloc[-1]))
    trend_long = ema20.iloc[-1] > ema50.iloc[-1]
    trend_short = ema20.iloc[-1] < ema50.iloc[-1]

    hist_up = hist.iloc[-1] > hist.iloc[-2]
    hist_down = hist.iloc[-1] < hist.iloc[-2]

    rsi_cross_up = rsi14.iloc[-1] > 50 >= rsi14.iloc[-2]
    rsi_cross_down = rsi14.iloc[-1] < 50 <= rsi14.iloc[-2]

    price = close.iloc[-1]
    if comp_ok and trend_long and hist_up and rsi_cross_up:
        return {"side": "LONG", "price": float(price), "reason": "Compressione + EMA20>EMA50 + MACD↑ + RSI cross 50"}
    if comp_ok and trend_short and hist_down and rsi_cross_down:
        return {"side": "SHORT", "price": float(price), "reason": "Compressione + EMA20<EMA50 + MACD↓ + RSI cross 50"}
    return None

# =========================
# Telegram
# =========================
def send_telegram(msg: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code != 200:
            logger.warning(f"Telegram send errore: {r.status_code} {r.text}")
    except Exception as e:
        logger.exception(f"Telegram exception: {e}")

# =========================
# Data prep
# =========================
def prices_to_df(raw: Dict) -> Optional[pd.DataFrame]:
    prices = raw.get("prices", [])
    if not prices:
        return None
    rows = []
    for p in prices:
        t = p.get("snapshotTimeUTC") or p.get("snapshotTime")
        o = (p["openPrice"]["bid"] + p["openPrice"]["ask"]) / 2.0
        c = (p["closePrice"]["bid"] + p["closePrice"]["ask"]) / 2.0
        h = (p["highPrice"]["bid"] + p["highPrice"]["ask"]) / 2.0
        l = (p["lowPrice"]["bid"] + p["lowPrice"]["ask"]) / 2.0
        v = p.get("lastTradedVolume", 0)
        rows.append((t, o, h, l, c, v))
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"]).dropna()
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(APP_TZ)
    df.set_index("time", inplace=True)
    return df

# =========================
# Scanner
# =========================
class Scanner:
    def __init__(self, api: CapitalAPI, epic_map: Dict[str, str]):
        self.api = api
        self.epic_map = epic_map

    def scan_once(self, tf_label: str, labels: List[str]):
        res = TF_RES_MAP[tf_label]
        for label in labels:
            epic = self.epic_map.get(label)
            if not epic:
                logger.warning(f"Skip {label}: epic non trovato")
                continue
            try:
                raw = self.api.get_prices(epic, res, max_points=400)
                df = prices_to_df(raw)
                if df is None or len(df) < 60:
                    logger.info(f"{label} {tf_label}: dati insufficienti")
                    continue
                sig = detect_pre_rally(df)
                if sig:
                    now = datetime.now(APP_TZ).strftime("%Y-%m-%d %H:%M")
                    msg = (
                        f"<b>[CAPITAL][{tf_label}] {label} → {sig['side']}</b>\n"
                        f"Prezzo: <b>{sig['price']:.5f}</b> | {now}\n"
                        f"Motivo: {sig['reason']}\n"
                        f"Epic: <code>{epic}</code>"
                    )
                    logger.info(msg)
                    send_telegram(msg)
                else:
                    logger.info(f"{label} {tf_label}: nessun setup")
            except Exception as e:
                logger.exception(f"Errore scan {label} {tf_label}: {e}")

# =========================
# FastAPI + Scheduler
# =========================
app = FastAPI()
_scheduler: Optional[BackgroundScheduler] = None
_scanner: Optional[Scanner] = None

@app.on_event("startup")
def _startup():
    global _scheduler, _scanner
    logger.info("Avvio bot…")
    api = CapitalAPI(CAPITAL_BASE_URL, CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_PASSWORD)
    api.login()
    epic_map = build_epic_map(api, ASSET_SEARCH_TERMS)
    _scanner = Scanner(api, epic_map)

    _scheduler = BackgroundScheduler(timezone="UTC")  # timezone stabile su Render

    # M1 (solo forex+crypto) ogni 1 minuto
    _scheduler.add_job(lambda: _scanner.scan_once("M1", [k for k in ASSET_SEARCH_TERMS if k in M1_ASSETS]),
                       CronTrigger(minute="*"))

    # M15 ogni 15 minuti
    _scheduler.add_job(lambda: _scanner.scan_once("M15", list(ASSET_SEARCH_TERMS.keys())),
                       CronTrigger(minute="0,15,30,45"))

    # H1 ogni ora al minuto 0
    _scheduler.add_job(lambda: _scanner.scan_once("H1", list(ASSET_SEARCH_TERMS.keys())),
                       CronTrigger(minute="0"))

    # D1 ogni giorno alle 23:59 UTC
    _scheduler.add_job(lambda: _scanner.scan_once("D1", list(ASSET_SEARCH_TERMS.keys())),
                       CronTrigger(hour="23", minute="59"))

    _scheduler.start()

    # Primo giro subito
    try:
        _scanner.scan_once("M15", list(ASSET_SEARCH_TERMS.keys()))
    except Exception as e:
        logger.exception(f"Scan di prova fallito: {e}")

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(APP_TZ).isoformat()}
