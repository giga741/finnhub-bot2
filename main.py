# main.py — Render Web + bot thread + /health
import os, threading, time, requests
from flask import Flask
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
PROVIDER = os.getenv("PROVIDER", "twelvedata").strip().lower()  # in futuro
PORT = int(os.getenv("PORT", "10000"))

API_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

def send(msg: str):
    if not TOKEN or not CHAT_ID: 
        print("⚠️ Manca TELEGRAM_TOKEN o TELEGRAM_CHAT_ID")
        return
    try:
        requests.post(API_URL, json={"chat_id": CHAT_ID, "text": msg, "parse_mode":"HTML"}, timeout=10)
    except Exception as e:
        print("Errore Telegram:", e)

def bot_loop():
    # Messaggio di avvio (una volta)
    send("🚀 Bot online (Render Web) ✅")
    while True:
        # QUI metteremo le tue regole/strategie reali.
        # Per ora teniamo vivo il ciclo senza spam:
        time.sleep(60)

app = Flask(__name__)

@app.route("/")
def home():
    return "OK"

@app.route("/health")
def health():
    return "UP"

def start_bot_thread():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    start_bot_thread()
    app.run(host="0.0.0.0", port=PORT)
