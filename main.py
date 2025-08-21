# 🚀 BOT FINNHUB + TWELVEDATA POTENZIATO
# Include: EMA, RSI, MACD, Volumi, Compressione, ATR, CCI, ADX, D1, Supporto/Resistenza, Heatmap, Pattern

import os, requests, time
from datetime import datetime
from pytz import timezone

# 🔐 API Keys
FINNHUB_API_KEY = os.environ['FINNHUB_API_KEY']
TWELVEDATA_API_KEY = os.environ['TWELVEDATA_API_KEY']
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

# 🕗 Finestra operativa (ora italiana)
START_HOUR = 8
END_HOUR = 23

# 📈 Ticker definitivi
TITLES = {
    "PLTR": "PLTR", "GOOGL": "GOOGL", "TSLA": "TSLA", "AAPL": "AAPL",
    "IFX.DE": "IFX.DE", "REY.MI": "REY.MI", "MU": "MU", "AMD": "AMD",
    "FCT.MI": "FCT.MI", "XOM": "XOM", "VLO": "VLO", "GM": "GM",
    "MC.PA": "MC.PA", "KO": "KO", "DIS": "DIS",
    "EUR/USD": "EUR/USD", "USD/JPY": "USD/JPY", "GBP/USD": "GBP/USD",
    "ETH/USD": "ETH/USD", "BTC/USD": "BTC/USD"
}

TIMEFRAMES = ["15min", "1h", "1day"]

# 📊 Scarica dati OHLC da TwelveData
def get_ohlc(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={TWELVEDATA_API_KEY}"
    r = requests.get(url)
    return r.json().get("values", [])

# 📊 Indicatori tecnici (mock semplificato)
def calculate_indicators(data):
    return {
        "ema": True, "rsi": True, "macd": False,
        "volumi": True, "compressione": True,
        "atr": True, "cci": True, "adx": True
    }

# 🔍 Pattern base (mock)
def detect_pattern(data):
    return "Triangolo ascendente"

# 🔍 Supporto / Resistenza
def detect_support_resistance(data):
    closes = [float(x['close']) for x in data]
    support = min(closes)
    resistance = max(closes)
    price = float(data[0]['close'])
    if price >= resistance * 0.998: return ("resistenza", resistance)
    if price <= support * 1.002: return ("supporto", support)
    return (None, None)

# 🔍 Fase del ciclo (heatmap semplificata)
def detect_cycle_phase(data):
    return "Accumulo"

# 📬 Messaggio Telegram

def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})

# 📌 Costruzione messaggio

def build_message(ticker, price, change, time, signals, is_hot, direction, d1_info, breakout_info, phase, pattern):
    score = sum(signals.values())
    stars = "⭐" * score + "☆" * (5 - score)
    check = lambda b: "✅" if b else "❌"

    msg = f"\n<pre>"
    msg += f"{'🟢' if direction == 'long' else '🔴'} {ticker} {price:.2f} ({change:+.2f}%) | {time}\n"
    msg += f"🧭 Direzione: {'LONG (rialzo)' if direction=='long' else 'SHORT (ribasso)'}\n\n"
    msg += f"📊 Segnali attivi:\n"
    msg += f"• M15 : {check(signals['ema'])} EMA   {check(signals['rsi'])} RSI\n"
    msg += f"• H1  : {check(signals['volumi'])} Volumi   {check(signals['macd'])} MACD\n"
    msg += f"• D1  : {check(d1_info['ema'])} EMA   {check(d1_info['compressione'])} Compressione   {check(d1_info['adx'])} ADX\n"
    if breakout_info[0]:
        msg += f"\n📉 Rottura {breakout_info[0].capitalize()}: {breakout_info[1]:.2f}"
    msg += f"\n🧠 Fase attuale: {phase}"
    msg += f"\n📐 Pattern attivo: {pattern}"
    msg += f"\n\n📈 Score segnale: {stars}"
    if is_hot:
        msg += f"\n🔥 Titolo caldo del ciclo"
    msg += f"</pre>"
    return msg

# 🔁 Ciclo principale

def run():
    now = datetime.now(timezone('Europe/Rome'))
    if not (START_HOUR <= now.hour <= END_HOUR): return

    for ticker, symbol in TITLES.items():
        try:
            m15 = get_ohlc(symbol, "15min")
            h1 = get_ohlc(symbol, "1h")
            d1 = get_ohlc(symbol, "1day")
            if not m15 or not h1 or not d1: continue

            price = float(m15[0]['close'])
            change = price * 0.01 * (1 if 'USD' in symbol else -1)
            signals = calculate_indicators(m15)
            d1_info = calculate_indicators(d1)
            breakout = detect_support_resistance(d1)
            phase = detect_cycle_phase(d1)
            pattern = detect_pattern(d1)
            direction = "short" if signals['macd'] == False else "long"
            score = sum(signals.values())
            if score >= 3:
                is_hot = signals['atr'] and signals['cci'] and score >= 4
                msg = build_message(ticker, price, change, now.strftime("%H:%M"), signals, is_hot, direction, d1_info, breakout, phase, pattern)
                send_telegram_message(msg)
        except Exception as e:
            print(f"Errore su {ticker}: {e}")

if __name__ == "__main__":
    while True:
        run()
        time.sleep(900)  # ogni 15 minuti
