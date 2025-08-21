import os
import requests
import datetime
import time
from pytz import timezone

FINNHUB_API_KEY = os.environ['FINNHUB_API_KEY']
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

ASSETS = [
    "PLTR", "GOOGL", "TSLA", "AAPL", "IFX.DE", "REY.MI", "MU", "AMD",
    "FCT.MI", "XOM", "VLO", "GM", "MC.PA", "KO", "DIS",
    "EUR/USD", "USD/JPY", "GBP/USD", "ETH/USD", "BTC/USD"
]

TIMEFRAMES = {
    "M15": "15",
    "H1": "60",
    "D1": "D"
}

TZ = timezone('Europe/Rome')

def get_candles(symbol, resolution, limit=200):
    url = f'https://finnhub.io/api/v1/forex/candle' if "/" in symbol else f'https://finnhub.io/api/v1/stock/candle'
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'count': limit,
        'token': FINNHUB_API_KEY
    }
    r = requests.get(url, params=params)
    data = r.json()
    if data.get('s') != 'ok':
        return []
    return list(zip(data['t'], data['o'], data['h'], data['l'], data['c']))

def calculate_ema(values, period):
    weights = [2 / (period + 1) * (1 - 2 / (period + 1)) ** i for i in range(period)]
    ema = []
    for i in range(period, len(values)):
        ema.append(sum([values[i-j-1] * weights[j] for j in range(period)]))
    return ema

def calculate_rsi(values, period=14):
    gains = []
    losses = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-diff)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = []
    for i in range(period, len(values)):
        if avg_loss == 0:
            rs = 0
        else:
            rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsis.append(rsi)
        if i + 1 < len(gains):
            avg_gain = (avg_gain * (period - 1) + gains[i + 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i + 1]) / period
    return rsis

def detect_signal(symbol):
    final_score = 0
    signal_data = []

    for tf_name, tf_code in TIMEFRAMES.items():
        candles = get_candles(symbol, tf_code, 100)
        if not candles or len(candles) < 20:
            continue

        closes = [c[4] for c in candles]
        ema = calculate_ema(closes, 14)
        rsi = calculate_rsi(closes)

        latest_close = closes[-1]
        ema_last = ema[-1] if ema else 0
        rsi_last = rsi[-1] if rsi else 0

        tf_score = 0
        indicators = []

        if latest_close > ema_last:
            tf_score += 1
            indicators.append("EMA")

        if rsi_last > 60:
            tf_score += 1
            indicators.append("RSI")

        if tf_name != "D1":
            support = min(closes[-10:-1])
            resistance = max(closes[-10:-1])
            if latest_close > resistance:
                tf_score += 1
                indicators.append("Resistenza")
            elif latest_close < support:
                tf_score += 1
                indicators.append("Supporto")

        if tf_score > 0:
            signal_data.append((tf_name, tf_score, indicators))

        if tf_name != "D1":
            final_score += tf_score

    if final_score >= 3:
        return format_message(symbol, final_score, signal_data)
    elif final_score == 6:
        return format_message(symbol, final_score, signal_data, is_special=True)

    return None

def format_message(symbol, score, data, is_special=False):
    now = datetime.datetime.now(TZ).strftime("%H:%M")

    direction = "LONG (rialzo)" if any("EMA" in d[2] and "RSI" in d[2] for d in data) else "SHORT (ribasso)"
    score_stars = "⭐" * min(score, 5)
    hot_label = "\n🔥 *Titolo caldo del ciclo!*\n" if is_special else ""

    lines = [f"*📍 {symbol}*", f"🧭 *Direzione:* {direction}", f"{hot_label}", f"📊 *Score:* {score_stars}\n"]
    for tf, pts, ind in data:
        ind_str = ", ".join(ind)
        lines.append(f"• {tf}: {ind_str}")
    lines.append(f"\n🕒 {now}")
    return "\n".join(lines)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)

def scan():
    for asset in ASSETS:
        signal = detect_signal(asset)
        if signal:
            send_telegram(signal)

if __name__ == "__main__":
    scan()
