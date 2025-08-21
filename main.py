# 🚀 BOT FINNHUB + TWELVEDATA + STRATEGIA POTENZIATA
# include: EMA, RSI, MACD, Volumi, Compressione + ATR, CCI, ADX

import os, requests, time
from datetime import datetime
from pytz import timezone

# 🔐 API KEYS
FINNHUB_API_KEY = os.environ['FINNHUB_API_KEY']
TWELVEDATA_API_KEY = os.environ['TWELVEDATA_API_KEY']
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

# ⏱️ FASCIA ORARIA
START_HOUR = 8
END_HOUR = 23

# ✅ SCORE MINIMO PER INVIO NOTIFICA
MIN_SCORE = 3

# 🧾 LISTA TITOLI
TITLES = {
    "PLTR": "PLTR", "GOOGL": "GOOGL", "TSLA": "TSLA", "AAPL": "AAPL",
    "IFX.DE": "IFX.DE", "REY.MI": "REY.MI", "MU": "MU", "AMD": "AMD",
    "FCT.MI": "FCT.MI", "XOM": "XOM", "VLO": "VLO", "GM": "GM",
    "MC.PA": "MC.PA", "KO": "KO", "DIS": "DIS",
    "EUR/USD": "EUR/USD", "USD/JPY": "USD/JPY", "GBP/USD": "GBP/USD",
    "ETH/USD": "ETH/USD", "BTC/USD": "BTC/USD"
}

def get_ohlc(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={TWELVEDATA_API_KEY}"
    r = requests.get(url)
    return r.json()

def calculate_indicators(data):
    # Mock di esempio – da sostituire con calcoli veri
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

def build_message(ticker, price, change, time, signals, is_short, is_hot):
    score = sum(signals.values())
    stars = "⭐" * score + "☆" * (8 - score)
    check = lambda b: "✅" if b else "❌"

    # Header colore e direzione
    direction_icon = "🔴" if is_short else "🟢"
    direction_label = "SHORT (ribasso)" if is_short else "LONG (rialzo)"

    msg = f"<pre>{direction_icon} {ticker} {price:.2f} ({change:+.2f}%) | {time}\n"
    msg += f"🧭 Direzione: {direction_label}\n"
    msg += f"\n📊 Segnali attivi:"
    msg += f"\n• EMA: {check(signals['ema'])}  RSI: {check(signals['rsi'])}  MACD: {check(signals['macd'])}"
    msg += f"\n• Volumi: {check(signals['volumi'])}  Compressione: {check(signals['compressione'])}"
    msg += f"\n• ATR: {check(signals['atr'])}  CCI: {check(signals['cci'])}  ADX: {check(signals['adx'])}"
    msg += f"\n\n📈 Score segnale: {stars}"
    if is_hot:
        msg += f"\n🔥 Titolo caldo del ciclo"
    msg += f"</pre>"
    return msg

def run():
    now = datetime.now(timezone("Europe/Rome"))
    hour = now.hour
    if not (START_HOUR <= hour <= END_HOUR):
        return

    for ticker, symbol in TITLES.items():
        try:
            data = get_ohlc(symbol, "15min")
            price = float(data['values'][0]['close'])
            prev_close = float(data['values'][1]['close'])
            change = (price - prev_close) / prev_close * 100.0
            signals = calculate_indicators(data)

            score = sum(signals.values())
            if score < MIN_SCORE:
                continue

            is_short = change < 0
            is_hot = signals['atr'] and signals['cci'] and score >= 5

            msg = build_message(ticker, price, change, now.strftime("%H:%M"), signals, is_short, is_hot)
            send_telegram_message(msg)
        except Exception as e:
            print(f"[{ticker}] Errore: {e}")

if __name__ == "__main__":
    while True:
        run()
        time.sleep(900)  # ogni 15 minuti
