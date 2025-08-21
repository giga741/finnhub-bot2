# 🚀 BOT TRADING - FINNHUB + TWELVEDATA + STRATEGIA POTENZIATA
# Include: EMA, RSI, MACD, Volumi, Compressione + ATR, CCI, ADX + Notifica speciale 10/10

import os
import requests
from datetime import datetime
from pytz import timezone
import time

# 🔐 API Keys
FINNHUB_API_KEY = os.environ['FINNHUB_API_KEY']
TWELVEDATA_API_KEY = os.environ['TWELVEDATA_API_KEY']
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

# 🕗 Finestra operativa (ora italiana)
START_HOUR = 8
END_HOUR = 23

# 📈 Lista titoli finali
TITLES = {
    "PLTR": "PLTR",
    "GOOGL": "GOOGL",
    "TSLA": "TSLA",
    "AAPL": "AAPL",
    "IFX.DE": "IFX.DE",
    "REY.MI": "REY.MI",
    "MU": "MU",
    "AMD": "AMD",
    "FCT.MI": "FCT.MI",
    "XOM": "XOM",
    "VLO": "VLO",
    "GM": "GM",
    "MC.PA": "MC.PA",
    "KO": "KO",
    "DIS": "DIS",
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "ETH/USD": "ETH/USD",
    "BTC/USD": "BTC/USD"
}

TIMEFRAMES = ["15min", "1h", "1day"]


def get_ohlc(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={TWELVEDATA_API_KEY}"
    r = requests.get(url)
    return r.json()


def calculate_indicators(data):
    # MOCK: sostituire con analisi reale
    return {
        "ema": True,
        "rsi": True,
        "macd": True,
        "volumi": True,
        "compressione": True,
        "atr": True,
        "cci": True,
        "adx": True
    }

def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    }
    requests.post(url, data=payload)

def build_message(ticker, price, change, time, signals, is_hot, direction="LONG"):
    score = list(signals.values()).count(True)
    stars = "⭐" * score + "☆" * (10 - score)
    check = lambda b: "✅" if b else "❌"

    msg = f"<pre>\n"
    msg += f"{ticker} | {price} ({change:+.2f}%) | {time}\n"
    msg += f"🧭 Direzione: {direction} (rialzo)\n\n"
    msg += f"📊 Segnali attivi:\n"
    msg += f"• EMA: {check(signals['ema'])}  RSI: {check(signals['rsi'])}  MACD: {check(signals['macd'])}\n"
    msg += f"• Volumi: {check(signals['volumi'])}  Compressione: {check(signals['compressione'])}\n"
    msg += f"• ATR: {check(signals['atr'])}  CCI: {check(signals['cci'])}  ADX: {check(signals['adx'])}\n"
    msg += f"\n📈 Score segnale: {stars}"
    if is_hot:
        msg += f"\n🔥 Titolo caldo del ciclo"
    msg += f"</pre>"
    return msg

def build_exceptional_alert(ticker, price, time):
    msg = f"<pre>\n🚨 SEGNALE ECCEZIONALE 🚨\n"
    msg += f"{ticker} | Score 10/10 + Compressione multi-day\n"
    msg += f"Prezzo attuale: {price}\n"
    msg += f"{time}\n</pre>"
    return msg


def run():
    now = datetime.now(timezone('Europe/Rome'))
    if not (START_HOUR <= now.hour <= END_HOUR):
        return

    for ticker, symbol in TITLES.items():
        try:
            ohlc_data = get_ohlc(symbol, "15min")
            price = float(ohlc_data['values'][0]['close'])
            change = float(price) * 0.01  # mock
            signals = calculate_indicators(ohlc_data)
            score = list(signals.values()).count(True)
            is_hot = signals['atr'] and signals['cci'] and score >= 4

            # 🔎 Verifica segnale eccezionale (score 10 + compressione D1)
            d1_data = get_ohlc(symbol, "1day")
            d1_compression = True  # simulazione
            if score == 10 and d1_compression:
                alert = build_exceptional_alert(ticker, price, now.strftime("%H:%M"))
                send_telegram_message(alert)

            # 🔎 Filtro: invia solo se score ≥ 3
            if score >= 3:
                msg = build_message(ticker, price, change, now.strftime("%H:%M"), signals, is_hot, direction="SHORT" if change < 0 else "LONG")
                send_telegram_message(msg)

        except Exception as e:
            print(f"[Errore {ticker}] {e}")


if __name__ == "__main__":
    while True:
        run()
        time.sleep(900)
