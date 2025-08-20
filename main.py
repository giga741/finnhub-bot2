# 🚀 BOT TRADING - FINNHUB + TWELVEDATA + STRATEGIA POTENZIATA
# Include: EMA, RSI, MACD, Volumi, Compressione + ATR, CCI, ADX (se disponibile)

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

# 📈 Lista titoli finali (Ticker TwelveData / Finnhub)
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

# 📊 Indicatori da usare per ogni timeframe
TIMEFRAMES = ["15min", "1h"]  # eventualmente aggiungi "1min" per crypto/forex

# 🔍 Funzione per ottenere dati OHLC da TwelveData

def get_ohlc(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={TWELVEDATA_API_KEY}"
    r = requests.get(url)
    return r.json()

# 🔍 Indicatori tecnici

def calculate_indicators(data):
    # Mock check: semplificato, sostituire con vera logica
    indicators = {
        "ema": True,
        "rsi": True,
        "macd": False,
        "volumi": True,
        "compressione": True,
        "atr": True,
        "cci": True,
        "adx": True
    }
    return indicators

# 📬 Invio messaggio Telegram

def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    }
    requests.post(url, data=payload)

# 📌 Costruzione del messaggio formato MIX #1 + #4 migliorato

def build_message(ticker, price, change, time, signals, is_hot):
    score = signals.count(True)
    stars = "⭐" * score + "☆" * (5 - score)
    check = lambda b: "✅" if b else "❌"

    msg = f"\n<pre>"
    msg += f"🟢 {ticker} {price} ({change:+.2f}%) | {time}\n"
    msg += f"📊 Segnali attivi:\n"
    msg += f"• M15 : {check(signals['ema'])} EMA, {check(signals['rsi'])} RSI\n"
    msg += f"• H1  : {check(signals['compressione'])} Compressione, {check(signals['volumi'])} Volumi\n"
    msg += f"• ATR : {check(signals['atr'])}  CCI: {check(signals['cci'])}  ADX: {check(signals['adx'])}\n"
    msg += f"\n📈 Score segnale: {stars}"
    if is_hot:
        msg += f"\n🔥 Titolo caldo del ciclo"
    msg += f"</pre>"
    return msg

# 🔁 Ciclo principale (mock semplificato)

def run():
    now = datetime.now(timezone('Europe/Rome'))
    hour = now.hour
    if not (START_HOUR <= hour <= END_HOUR):
        return

    for ticker, symbol in TITLES.items():
        try:
            data = get_ohlc(symbol, "15min")
            price = float(data['values'][0]['close'])
            change = float(price) * 0.01 * (-1 if "PLTR" in symbol else 1)  # simulazione
            signals = calculate_indicators(data)
            is_hot = signals['atr'] and signals['cci'] and score >= 4
            msg = build_message(ticker, price, change, now.strftime("%H:%M"), signals, is_hot)
            send_telegram_message(msg)
        except Exception as e:
            print(f"Errore su {ticker}: {e}")

if __name__ == "__main__":
    while True:
        run()
        time.sleep(900)  # ogni 15 minuti
