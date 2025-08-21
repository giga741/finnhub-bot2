import os
import requests
import time
from datetime import datetime
from pytz import timezone

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

assets = {
    "PLTR": "PLTR",
    "GOOGL": "GOOGL",
    "TSLA": "TSLA",
    "AAPL": "AAPL",
    "REY.MI": "REY.MI",
    "FCT.MI": "FCT.MI",
    "IFX.DE": "IFX.DE",
    "ETH/USD": "ETH/USD",
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "BTC/USD": "BTC/USD",
    "LVMH.PA": "MC.PA",
    "KO": "KO",
    "DIS": "DIS",
    "MU": "MU",
    "AMD": "AMD",
    "XOM": "XOM",
    "VLO": "VLO",
    "GM": "GM"
}

timeframes = {
    "15min": "15min",
    "60min": "60min",
    "1day": "1day"
}

def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={TWELVE_API_KEY}&outputsize=50"
    response = requests.get(url)
    data = response.json()
    return data.get("values", [])

def analyze(asset):
    signals = []
    for tf_label, tf_interval in timeframes.items():
        data = fetch_data(asset, tf_interval)
        if not data or len(data) < 20:
            continue

        close_prices = [float(x['close']) for x in data[:20]]
        ema9 = sum(close_prices[:9]) / 9
        ema21 = sum(close_prices[:21]) / 21
        rsi = compute_rsi(close_prices)

        score = 0
        if ema9 > ema21:
            score += 1
        if rsi < 30:
            score += 1

        if score >= 2:
            signals.append((tf_label, score, ema9, ema21, rsi))

    return signals

def compute_rsi(prices, period=14):
    deltas = [prices[i] - prices[i+1] for i in range(len(prices)-1)]
    gains = sum([d for d in deltas[:period] if d > 0])
    losses = -sum([d for d in deltas[:period] if d < 0])
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)

def scan():
    now = datetime.now(timezone("Europe/Rome")).strftime("%H:%M")
    print(f"[{now}] Avvio scansione...")
    for name, symbol in assets.items():
        signals = analyze(symbol)
        if signals:
            message = f"📡 *Segnale attivo su {name}*\n"
            for tf, score, ema9, ema21, rsi in signals:
                message += f"⏱️ *{tf}* | Score: {score}/2 | EMA9: {ema9:.2f} | EMA21: {ema21:.2f} | RSI: {rsi:.1f}\n"
            send_telegram(message)

if __name__ == "__main__":
    scan()

