# === main.py (login smart + bot) =============================================
import os, time, json, logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from dateutil import tz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

# ---------------------------------------------------------------------
# LOG
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("capital-bot")

# ---------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------
def _clean(x: Optional[str]) -> str: return (x or "").strip()

API_KEY   = _clean(os.getenv("CAPITAL_API_KEY"))
API_PWD   = _clean(os.getenv("CAPITAL_PASSWORD"))           # password della CHIAVE API
IDENT     = _clean(os.getenv("CAPITAL_IDENTIFIER"))          # email o clientId (metti una qualsiasi)
CHAT_ID   = _clean(os.getenv("TELEGRAM_CHAT_ID"))
TG_TOKEN  = _clean(os.getenv("TELEGRAM_TOKEN"))

# URL fissi
LIVE_BASE = "https://api-capital.backend-capital.com"
DEMO_BASE = "https://demo-api-capital.backend-capital.com"

APP_TZ = tz.gettz("UTC")  # Render: usa UTC

# ---------------------------------------------------------------------
# ASSET & TIMEFRAME
# ---------------------------------------------------------------------
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
TF_RES_MAP = {"M1":"MINUTE","M15":"MINUTE_15","H1":"HOUR","D1":"DAY"}
M1_ASSETS = {"USD/JPY","EUR/USD","GBP/USD","ETH/USD"}

# ---------------------------------------------------------------------
# CLIENT CAPITAL
# ---------------------------------------------------------------------
class CapitalAPI:
    def __init__(self, base_url: str, api_key: str, identifier: str, password: str):
        self.base = base_url.rstrip("/")
        self.key = api_key; self.identifier = identifier; self.password = password
        self.s = requests.Session(); self.cst=None; self.sec=None; self.last=0.0

    def _headers(self):
        h = {"X-CAP-API-KEY": self.key}
        if self.cst and self.sec: h.update({"CST": self.cst, "X-SECURITY-TOKEN": self.sec})
        return h

    def login(self):
        url = f"{self.base}/api/v1/session"
        payload = {"identifier": self.identifier, "password": self.password, "encryptedPassword": False}
        log.info("[CapitalAPI] login → base=%s id=%s key=%s pwd_len=%d",
                 self.base, ("email" if "@" in self.identifier else "clientId"),
                 (self.key[:3]+"…"+self.key[-3:]) if self.key else "None", len(self.password))
        r = self.s.post(url, headers={"X-CAP-API-KEY": self.key,"Content-Type":"application/json"},
                        data=json.dumps(payload), timeout=25)
        if r.status_code != 200:
            raise RuntimeError(f"Login failed: {r.status_code} {r.text}")
        self.cst = r.headers.get("CST"); self.sec = r.headers.get("X-SECURITY-TOKEN")
        if not self.cst or not self.sec: raise RuntimeError("Login ok ma CST/X-SECURITY-TOKEN mancanti")
        self.last = time.time(); log.info("[CapitalAPI] Sessione avviata ✅")

    def _ensure(self):
        if not self.cst or not self.sec or (time.time()-self.last)>540: self.login()

    def _get(self, path:str, params:Dict=None):
        self._ensure(); url=f"{self.base}{path}"
        r=self.s.get(url, headers=self._headers(), params=params or {}, timeout=25)
        if r.status_code==401:
            log.warning("[CapitalAPI] 401 → re-login…"); self.login()
            r=self.s.get(url, headers=self._headers(), params=params or {}, timeout=25)
        self.last=time.time(); return r

    def search(self, term:str)->List[Dict]:
        r=self._get("/api/v1/markets", {"searchTerm":term})
        return r.json().get("markets", []) if r.status_code==200 else []

    def prices(self, epic:str, res:str, maxp:int=300)->Dict:
        r=self._get(f"/api/v1/prices/{epic}", {"resolution":res, "max":maxp})
        if r.status_code!=200: raise RuntimeError(f"Prices failed for {epic}: {r.status_code} {r.text}")
        return r.json()

# ---------------------------------------------------------------------
# LOGIN SMART (prova LIVE/DEMO × email/clientId)
# ---------------------------------------------------------------------
def smart_login() -> CapitalAPI:
    email = IDENT if "@" in IDENT else "omarvitellaro@gmail.com"
    cid   = IDENT if IDENT.isdigit() else "33320253"
    attempts = [
        ("LIVE","email",    LIVE_BASE,email),
        ("LIVE","clientId", LIVE_BASE,cid),
        ("DEMO","email",    DEMO_BASE,email),
        ("DEMO","clientId", DEMO_BASE,cid),
    ]
    last_err = None
    for env, idtype, base, ident in attempts:
        try:
            api = CapitalAPI(base, API_KEY, ident, API_PWD)
            api.login()
            log.info("✅ LOGIN OK con %s / %s", env, idtype)
            return api
        except Exception as e:
            last_err = str(e)
            log.warning("❌ LOGIN KO con %s / %s → %s", env, idtype, last_err)
    # se siamo qui, tutti KO
    raise RuntimeError(f"Tutti i tentativi di login hanno fallito: {last_err}")

# ---------------------------------------------------------------------
# UTIL & INDICATORI
# ---------------------------------------------------------------------
def _sanitize(s:str)->str: return "".join(ch for ch in s if ch.isalnum())
def ema(s:pd.Series, p:int)->pd.Series: return s.ewm(span=p, adjust=False).mean()
def rsi(s:pd.Series, p:int=14)->pd.Series:
    d=s.diff(); up=np.where(d>0,d,0.0); dn=np.where(d<0,-d,0.0)
    ru=pd.Series(up,index=s.index).ewm(span=p,adjust=False).mean()
    rd=pd.Series(dn,index=s.index).ewm(span=p,adjust=False).mean()
    rs=ru/(rd+1e-9); return 100-(100/(1+rs))
def true_range(h,l,c): pc=c.shift(1); return pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
def atr(h,l,c,p=14): return true_range(h,l,c).ewm(span=p,adjust=False).mean()
def boll(s,p=20,k=2.0): ma=s.rolling(p,min_periods=p).mean(); sd=s.rolling(p,min_periods=p).std(ddof=0); return ma-k*sd,ma,ma+k*sd
def macd(s,fast=12,slow=26,signal=9): ef,es=ema(s,fast),ema(s,slow); line=ef-es; sig=ema(line,signal); return line,sig,line-sig

def detect_pre_rally(df:pd.DataFrame)->Optional[Dict]:
    if df is None or len(df)<100: return None
    c=df["close"]; h=df["high"]; l=df["low"]
    ema20,ema50=ema(c,20),ema(c,50); rsi14=rsi(c,14)
    atr14=atr(h,l,c,14); atr_ma=atr14.rolling(14).mean()
    low,mid,up=boll(c,20,2.0); bbw=(up-low)/(mid+1e-9); bbw_p20=bbw.rolling(200).quantile(0.20)
    _,_,hist=macd(c)
    comp=(bbw.iloc[-1]<=bbw_p20.iloc[-1]) and (atr14.iloc[-1] <= 0.8*(atr_ma.iloc[-1] if not np.isnan(atr_ma.iloc[-1]) else atr14.iloc[-1]))
    long = ema20.iloc[-1]>ema50.iloc[-1] and hist.iloc[-1]>hist.iloc[-2] and (rsi14.iloc[-1]>50>=rsi14.iloc[-2])
    short= ema20.iloc[-1]<ema50.iloc[-1] and hist.iloc[-1]<hist.iloc[-2] and (rsi14.iloc[-1]<50<=rsi14.iloc[-2])
    price=c.iloc[-1]
    if comp and long:  return {"side":"LONG","price":float(price),"reason":"Compressione + EMA20>EMA50 + MACD↑ + RSI cross 50"}
    if comp and short: return {"side":"SHORT","price":float(price),"reason":"Compressione + EMA20<EMA50 + MACD↓ + RSI cross 50"}
    return None

def to_df(raw:Dict)->Optional[pd.DataFrame]:
    prices=raw.get("prices",[]); 
    if not prices: return None
    rows=[]
    for p in prices:
        t=p.get("snapshotTimeUTC") or p.get("snapshotTime")
        o=(p["openPrice"]["bid"]+p["openPrice"]["ask"])/2.0
        c=(p["closePrice"]["bid"]+p["closePrice"]["ask"])/2.0
        h=(p["highPrice"]["bid"]+p["highPrice"]["ask"])/2.0
        l=(p["lowPrice"]["bid"]+p["lowPrice"]["ask"])/2.0
        v=p.get("lastTradedVolume",0)
        rows.append((t,o,h,l,c,v))
    df=pd.DataFrame(rows,columns=["time","open","high","low","close","volume"]).dropna()
    df["time"]=pd.to_datetime(df["time"],utc=True).dt.tz_convert(APP_TZ); df.set_index("time",inplace=True); return df

def send_telegram(msg:str):
    try:
        url=f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        r=requests.post(url,json={"chat_id":CHAT_ID,"text":msg,"parse_mode":"HTML","disable_web_page_preview":True},timeout=20)
        if r.status_code!=200: log.warning("Telegram err %s %s", r.status_code, r.text)
    except Exception as e: log.warning("Telegram exception: %s", e)

def _sanitize(s:str)->str: return "".join(ch for ch in s if ch.isalnum())

def select_best(mkts:List[Dict], label:str)->Optional[Dict]:
    if not mkts: return None
    w=label.upper(); ranked=[]
    for m in mkts:
        inst=m.get("instrument",{}); name=str(inst.get("name","")).upper(); sym=str(inst.get("symbol","")).upper()
        epic=inst.get("epic"); product=m.get("product",""); status=m.get("snapshot",{}).get("marketStatus")
        score=0
        if any(k in name for k in [w,_sanitize(w)]) or any(k in sym for k in [w,_sanitize(w)]): score+=3
        if product=="CFD": score+=2
        if status=="TRADEABLE": score+=1
        if epic: ranked.append((score,m))
    ranked.sort(key=lambda x:x[0],reverse=True); return ranked[0][1] if ranked else None

def build_epics(api:CapitalAPI, terms:Dict[str,List[str]])->Dict[str,str]:
    out={}
    for label, arr in terms.items():
        epic=None
        for term in arr + [_sanitize(t) for t in arr]:
            try:
                mkts=api.search(term)
                if mkts:
                    best=select_best(mkts,label) or mkts[0]
                    epic=best["instrument"]["epic"]; log.info("Mappato %s (%s) → %s", label, term, epic); break
            except Exception as e: log.warning("search err %s/%s: %s", label, term, e)
        if epic: out[label]=epic
        else: log.warning("Nessun epic trovato per %s", label)
    return out

# ---------------------------------------------------------------------
# SCANNER
# ---------------------------------------------------------------------
class Scanner:
    def __init__(self, api:CapitalAPI, epic_map:Dict[str,str]): self.api=api; self.epic=epic_map
    def scan_once(self, tf:str, labels:List[str]):
        res=TF_RES_MAP[tf]
        for label in labels:
            epic=self.epic.get(label)
            if not epic: log.warning("Skip %s: epic non trovato", label); continue
            try:
                raw=self.api.prices(epic,res,maxp=400); df=to_df(raw)
                if df is None or len(df)<60: log.info("%s %s: dati insufficienti", label, tf); continue
                sig=detect_pre_rally(df)
                if sig:
                    now=datetime.now(APP_TZ).strftime("%Y-%m-%d %H:%M")
                    msg=(f"<b>[CAPITAL][{tf}] {label} → {sig['side']}</b>\n"
                         f"Prezzo: <b>{sig['price']:.5f}</b> | {now}\n"
                         f"Motivo: {sig['reason']}\nEpic: <code>{epic}</code>")
                    log.info(msg); send_telegram(msg)
                else: log.info("%s %s: nessun setup", label, tf)
            except Exception as e:
                log.exception("Errore scan %s %s: %s", label, tf, e)

# ---------------------------------------------------------------------
# FASTAPI + SCHEDULER
# ---------------------------------------------------------------------
app = FastAPI()
_sched: Optional[BackgroundScheduler] = None
_scan: Optional[Scanner] = None
_api: Optional[CapitalAPI] = None

@app.on_event("startup")
def startup():
    global _sched,_scan,_api
    log.info("Avvio bot…")
    try:
        _api = smart_login()  # tenta tutte le combinazioni
    except Exception as e:
        log.error("❌ Login non riuscito. Motivo: %s", e)
        return  # NON crashare: la health resta up per vedere l'errore

    epic_map = build_epics(_api, ASSET_SEARCH_TERMS)
    _scan = Scanner(_api, epic_map)

    _sched = BackgroundScheduler(timezone="UTC")
    _sched.add_job(lambda:_scan.scan_once("M1",[k for k in ASSET_SEARCH_TERMS if k in M1_ASSETS]), CronTrigger(minute="*"))
    _sched.add_job(lambda:_scan.scan_once("M15",list(ASSET_SEARCH_TERMS.keys())), CronTrigger(minute="0,15,30,45"))
    _sched.add_job(lambda:_scan.scan_once("H1", list(ASSET_SEARCH_TERMS.keys())), CronTrigger(minute="0"))
    _sched.add_job(lambda:_scan.scan_once("D1", list(ASSET_SEARCH_TERMS.keys())), CronTrigger(hour="23", minute="59"))
    _sched.start()

    # primo giro
    try: _scan.scan_once("M15", list(ASSET_SEARCH_TERMS.keys()))
    except Exception as e: log.warning("Scan iniziale fallito: %s", e)

@app.get("/")
@app.get("/health")
def health():
    return {
        "status":"ok",
        "time": datetime.now(APP_TZ).isoformat(),
        "has_api": bool(_api),
    }
# =====================================================================

